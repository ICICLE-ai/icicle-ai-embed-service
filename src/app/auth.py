from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Header, HTTPException, status

from .settings import settings

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    username: str
    tenant_id: str
    claims: dict[str, Any]


@lru_cache(maxsize=1)
def _jwk_client() -> jwt.PyJWKClient:
    logger.info("Initialising JWKS client from %s", settings.tapis_jwks_url)
    return jwt.PyJWKClient(settings.tapis_jwks_url)


def _decode_token(token: str) -> dict[str, Any]:
    client = _jwk_client()
    try:
        signing_key = client.get_signing_key_from_jwt(token).key
        claims: dict[str, Any] = jwt.decode(
            token,
            signing_key,
            algorithms=["RS256"],
            issuer=settings.tapis_issuer,
            options={"verify_aud": False},
        )
    except jwt.ExpiredSignatureError as exc:
        logger.warning("Rejected expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please obtain a fresh access token.",
        ) from exc
    except jwt.InvalidIssuerError as exc:
        logger.warning("Rejected token with wrong issuer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token issuer. Expected issuer: {settings.tapis_issuer}",
        ) from exc
    except jwt.PyJWTError as exc:
        logger.warning("Rejected invalid token: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed. Ensure you are sending a valid Tapis access token.",
        ) from exc

    if claims.get("tapis/token_type") != "access":
        logger.warning("Rejected non-access token type: %s", claims.get("tapis/token_type"))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Only Tapis access tokens are accepted. Received token type: "
                f"'{claims.get('tapis/token_type')}'."
            ),
        )

    tenant_id = claims.get("tapis/tenant_id")
    if tenant_id != settings.tapis_tenant_id:
        logger.warning(
            "Rejected token from tenant '%s' (allowed: '%s')",
            tenant_id,
            settings.tapis_tenant_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Access denied. This service only accepts tokens from the "
                f"'{settings.tapis_tenant_id}' tenant. Your token belongs to '{tenant_id}'."
            ),
        )

    return claims


async def get_current_user(
    x_tapis_token: str = Header(..., alias="X-Tapis-Token"),
) -> UserContext:
    claims = _decode_token(x_tapis_token)
    username = claims.get("tapis/username") or claims.get("sub")
    if not username:
        logger.warning("Token accepted but has no username claim")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is valid but does not contain a username. Contact your Tapis administrator.",
        )
    tenant_id = claims["tapis/tenant_id"]
    logger.info("Authenticated user '%s' (tenant: %s)", username, tenant_id)
    return UserContext(username=username, tenant_id=tenant_id, claims=claims)
