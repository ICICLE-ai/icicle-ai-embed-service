from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import anyio
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import UserContext, get_current_user
from .embedder import Embedder, get_embedder, init_embedder
from .schemas import EmbedItem, EmbedRequest, EmbedResponse, ModelInfoResponse
from .settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Initialising embedder ...")
    try:
        # Model load is CPU/GPU-heavy and blocking; run off the event loop.
        await anyio.to_thread.run_sync(init_embedder)
    except Exception as exc:
        logger.error("Failed to initialise embedder: %s", exc)
        raise SystemExit(
            "Embedder failed to initialise. Check MODEL_PATH / MODEL_REPO / MODEL_FILE "
            "and that you have network access to Hugging Face on first boot."
        ) from exc
    logger.info("Embedder ready.")
    yield


app = FastAPI(
    title="ICICLE AI Embed Service",
    version="0.1.0",
    description=(
        "Local embedding generator built on llama-cpp-python + Qwen3-Embedding GGUF. "
        "Takes text in, returns vectors that can be stored via the ICICLE AI Vector Service. "
        "All endpoints (except /healthz) require a valid X-Tapis-Token from the icicleai tenant."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-Tapis-Token", "Content-Type", "Authorization"],
)


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/model", response_model=ModelInfoResponse)
async def model_info(
    embedder: Embedder = Depends(get_embedder),
    _user: UserContext = Depends(get_current_user),
) -> ModelInfoResponse:
    return ModelInfoResponse(
        model=embedder.model_name,
        dim=embedder.dim,
        n_ctx=settings.n_ctx,
    )


@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(
    payload: EmbedRequest,
    embedder: Embedder = Depends(get_embedder),
    current_user: UserContext = Depends(get_current_user),
) -> EmbedResponse:
    texts = [payload.input] if isinstance(payload.input, str) else payload.input
    is_query = payload.input_type == "query"

    # Log shape, never the text itself.
    logger.info(
        "Embedding %d text(s) for user '%s' (type=%s, normalize=%s, total_chars=%d)",
        len(texts),
        current_user.username,
        payload.input_type,
        payload.normalize,
        sum(len(t) for t in texts),
    )

    vectors = await anyio.to_thread.run_sync(
        embedder.embed, texts, is_query, payload.instruction, payload.normalize
    )

    data = [EmbedItem(index=i, embedding=v) for i, v in enumerate(vectors)]
    return EmbedResponse(
        model=embedder.model_name,
        dim=embedder.dim,
        input_type=payload.input_type,
        normalized=payload.normalize,
        data=data,
    )
