"""Shared test bootstrap.

Fail fast with a clear message when required auth env vars are missing.
`src.app.settings` loads at import time during test collection.
"""

from __future__ import annotations

import os


def _validate_required_tapis_env() -> None:
    required = ("TAPIS_ISSUER", "TAPIS_JWKS_URL", "TAPIS_TENANT_ID")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required env var(s) for tests: "
            + ", ".join(missing)
            + ". Set them in your shell/.env or GitHub Actions repo variables."
        )


_validate_required_tapis_env()
