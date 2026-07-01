from __future__ import annotations

import hashlib
import logging
import time
from array import array
from typing import TYPE_CHECKING, cast

from .settings import settings

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Real type for the checker; the runtime import below may be absent.
    from redis.asyncio import Redis

# redis is an optional dependency: the service must run (cache-free) even if it
# is not installed or not reachable. Import defensively.
try:
    from redis.asyncio import Redis as _Redis
except Exception:  # pragma: no cover - exercised only when redis is absent
    _Redis = None


def _encode_vector(vec: list[float]) -> bytes:
    """Pack a vector as raw float32 bytes (~5x smaller than JSON, zero-copy decode)."""
    return array("f", vec).tobytes()


def _decode_vector(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    return arr.tolist()


class EmbeddingCache:
    """Content-addressed, fail-open Redis cache for embedding vectors.

    - Keys are a sha256 over everything that affects the vector (model, input
      type, instruction, normalize, text), so a different model/instruction/text
      automatically maps to a different key. No user id: embeddings are
      deterministic, so the cache is safely shared across users.
    - Values are float32 bytes.
    - Every operation degrades to a no-op (treated as a miss) on any Redis
      error, so a cache outage can never break /v1/embed.
    - Eviction is delegated to Redis (maxmemory + allkeys-lru), not done here.

    Swapping the backend location is a one-line config change: set REDIS_URL.
    """

    def __init__(self) -> None:
        self._redis: "Redis | None" = None
        self._enabled = settings.cache_enabled and _Redis is not None
        self._prefix = settings.cache_key_prefix
        self._ttl = settings.cache_ttl_seconds

    @property
    def available(self) -> bool:
        return self._enabled and self._redis is not None

    async def connect(self) -> None:
        """Best-effort connect. On any failure, the cache stays disabled."""
        if not self._enabled:
            logger.info(
                "Embedding cache disabled (CACHE_ENABLED=false or redis not installed)."
            )
            return
        assert _Redis is not None  # guaranteed by self._enabled
        start = time.monotonic()
        try:
            self._redis = _Redis.from_url(
                settings.redis_url,
                socket_connect_timeout=settings.redis_timeout_seconds,
                socket_timeout=settings.redis_timeout_seconds,
            )
            await self._redis.ping()
            await self._apply_eviction_policy()
            logger.info(
                "Embedding cache connected: %s in %.3fs (ttl=%ss, prefix=%s)",
                settings.redis_url,
                time.monotonic() - start,
                self._ttl,
                self._prefix,
            )
        except Exception as exc:
            logger.warning(
                "Cache unavailable — running without it after %.3fs "
                "(url=%s, timeout=%ss): %s",
                time.monotonic() - start,
                settings.redis_url,
                settings.redis_timeout_seconds,
                exc,
            )
            self._redis = None

    async def _apply_eviction_policy(self) -> None:
        """Push the LRU policy to Redis on startup (best effort).

        The maxmemory *budget* (e.g. 80% of pod RAM) is intentionally left to the
        Redis/pod config — that is where pod sizing lives. Managed Redis often
        forbids CONFIG SET, so this is best-effort and never fatal.
        """
        if not (settings.redis_maxmemory or settings.redis_maxmemory_policy):
            return
        assert self._redis is not None  # called only after a successful connect
        try:
            if settings.redis_maxmemory:
                await self._redis.config_set("maxmemory", settings.redis_maxmemory)
            if settings.redis_maxmemory_policy:
                await self._redis.config_set(
                    "maxmemory-policy", settings.redis_maxmemory_policy
                )
            logger.info(
                "Cache eviction configured (maxmemory=%s, policy=%s)",
                settings.redis_maxmemory or "(unchanged)",
                settings.redis_maxmemory_policy or "(unchanged)",
            )
        except Exception as exc:
            logger.warning(
                "Could not set Redis eviction policy (managed Redis?): %s", exc
            )

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:  # pragma: no cover
                pass
            self._redis = None

    def make_key(
        self,
        model: str,
        input_type: str,
        instruction: str | None,
        normalize: bool,
        text: str,
    ) -> str:
        # \x1f (unit separator) can't appear in normal text, so fields can't collide.
        raw = "\x1f".join(
            (model, input_type, instruction or "", str(int(normalize)), text)
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{self._prefix}{digest}"

    async def get_many(self, keys: list[str]) -> list[list[float] | None]:
        """Return cached vectors aligned to `keys`; None for misses or on error."""
        if not self.available or not keys:
            return [None] * len(keys)
        assert self._redis is not None  # narrowed by self.available
        try:
            blobs = await self._redis.mget(keys)
        except Exception as exc:
            logger.warning("Cache read failed — treating as all-miss: %s", exc)
            return [None] * len(keys)
        # Values are always raw bytes — we never enable decode_responses.
        return [(_decode_vector(cast(bytes, b)) if b else None) for b in blobs]

    async def set_many(self, items: dict[str, list[float]]) -> None:
        """Write vectors with TTL. Overwrites existing keys (refreshes value+TTL)."""
        if not self.available or not items:
            return
        assert self._redis is not None  # narrowed by self.available
        try:
            pipe = self._redis.pipeline(transaction=False)
            for key, vec in items.items():
                pipe.set(key, _encode_vector(vec), ex=self._ttl)
            await pipe.execute()
        except Exception as exc:
            logger.warning("Cache write failed — entries dropped: %s", exc)


_cache = EmbeddingCache()


def get_cache() -> EmbeddingCache:
    return _cache
