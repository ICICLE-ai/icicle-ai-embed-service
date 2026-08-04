from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Real type for the checker; the runtime import below may be absent.
    from httpx import AsyncClient

# httpx is only needed when metrics are enabled. Import defensively so the
# service still runs (metrics-free) if it is somehow absent.
try:
    import httpx as _httpx
except Exception:  # pragma: no cover - exercised only when httpx is absent
    _httpx = None


class MlflowMetrics:
    """Fail-open MLflow metric logger for embed requests.

    Talks to the MLflow REST API directly (no `mlflow` client dependency) and
    authenticates every call with the *incoming request's* already-validated
    ``X-Tapis-Token``. Reusing the caller's token means no service account and no
    token-refresh logic: each request carries a fresh, valid token.

    Every operation is best-effort. Any failure — MLflow down, pod asleep, token
    not authorised on the pod, network blip — is logged once and swallowed, so
    metric logging can never add latency to or break ``/v1/embed`` (it always
    runs as a background task, after the response is sent).

    Only anonymous request shape is logged (batch size, char count, cache
    hits/misses, latency, model, input type). No username, tenant, token claims,
    or input text ever leaves the service.
    """

    def __init__(self) -> None:
        self._enabled = (
            settings.mlflow_enabled
            and _httpx is not None
            and bool(settings.mlflow_tracking_uri)
        )
        self._base = settings.mlflow_tracking_uri.rstrip("/")
        self._experiment = settings.mlflow_experiment
        self._timeout = settings.mlflow_timeout_seconds
        # Resolved lazily on first successful log (needs a caller token), then
        # cached per worker process.
        self._experiment_id: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def announce(self) -> None:
        """Log the metrics configuration once at startup."""
        if self._enabled:
            logger.info(
                "MLflow metrics enabled -> %s (experiment=%s, timeout=%ss)",
                self._base,
                self._experiment,
                self._timeout,
            )
        elif settings.mlflow_enabled:
            logger.warning(
                "MLflow metrics requested (MLFLOW_ENABLED=true) but disabled: "
                "missing httpx or MLFLOW_TRACKING_URI."
            )
        else:
            logger.info("MLflow metrics disabled (MLFLOW_ENABLED=false).")

    async def _api(
        self,
        client: AsyncClient,
        method: str,
        path: str,
        token: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        resp = await client.request(
            method,
            f"{self._base}/api/2.0/mlflow{path}",
            headers={"X-Tapis-Token": token, "Content-Type": "application/json"},
            json=json,
            params=params,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def _ensure_experiment(
        self, client: AsyncClient, token: str
    ) -> str | None:
        if self._experiment_id is not None:
            return self._experiment_id

        # Prefer an existing experiment; create it only if it doesn't exist.
        try:
            data = await self._api(
                client,
                "GET",
                "/experiments/get-by-name",
                token,
                params={"experiment_name": self._experiment},
            )
            self._experiment_id = data["experiment"]["experiment_id"]
            return self._experiment_id
        except Exception:
            pass

        try:
            data = await self._api(
                client, "POST", "/experiments/create", token, json={"name": self._experiment}
            )
            self._experiment_id = data["experiment_id"]
            return self._experiment_id
        except Exception:
            # Another worker may have created it in the meantime — re-fetch once.
            try:
                data = await self._api(
                    client,
                    "GET",
                    "/experiments/get-by-name",
                    token,
                    params={"experiment_name": self._experiment},
                )
                self._experiment_id = data["experiment"]["experiment_id"]
                return self._experiment_id
            except Exception as exc:
                logger.warning(
                    "MLflow: could not resolve experiment '%s': %s",
                    self._experiment,
                    exc,
                )
                return None

    async def log_embed(
        self, token: str, metrics: dict[str, float], tags: dict[str, str]
    ) -> None:
        """Log one MLflow run for an embed request. Never raises."""
        if not self._enabled:
            return
        assert _httpx is not None  # narrowed by self._enabled
        ts = int(time.time() * 1000)
        try:
            async with _httpx.AsyncClient(timeout=self._timeout) as client:
                experiment_id = await self._ensure_experiment(client, token)
                if experiment_id is None:
                    return

                run = await self._api(
                    client,
                    "POST",
                    "/runs/create",
                    token,
                    json={
                        "experiment_id": experiment_id,
                        "start_time": ts,
                        "tags": [{"key": k, "value": v} for k, v in tags.items()],
                    },
                )
                run_id = run["run"]["info"]["run_id"]

                await self._api(
                    client,
                    "POST",
                    "/runs/log-batch",
                    token,
                    json={
                        "run_id": run_id,
                        "metrics": [
                            {"key": k, "value": float(v), "timestamp": ts, "step": 0}
                            for k, v in metrics.items()
                        ],
                    },
                )

                await self._api(
                    client,
                    "POST",
                    "/runs/update",
                    token,
                    json={
                        "run_id": run_id,
                        "status": "FINISHED",
                        "end_time": int(time.time() * 1000),
                    },
                )
        except Exception as exc:
            logger.warning("MLflow metric logging failed (ignored): %s", exc)


_metrics = MlflowMetrics()


def get_metrics() -> MlflowMetrics:
    return _metrics
