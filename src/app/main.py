from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from anyio.to_thread import run_sync
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .auth import UserContext, get_current_user
from .cache import get_cache
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
        await run_sync(init_embedder)
    except Exception as exc:
        logger.error("Failed to initialise embedder: %s", exc)
        raise SystemExit(
            "Embedder failed to initialise. Check MODEL_PATH / MODEL_REPO / MODEL_FILE "
            "and that you have network access to Hugging Face on first boot."
        ) from exc
    # Cache is optional and fail-open; connect after the model is ready.
    await get_cache().connect()
    logger.info("Embedder ready.")
    yield
    await get_cache().close()


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


async def _embed_texts(
    embedder: Embedder,
    texts: list[str],
    is_query: bool,
    instruction: str | None,
    normalize: bool,
    background_tasks: BackgroundTasks,
    username: str,
) -> list[list[float]]:
    """Embed `texts`, serving query vectors from Redis when possible.

    Only queries are cached: documents are embedded once and live in the vector
    service, so caching them mostly wastes memory. Cache misses are logged, the
    embedder runs only on the misses, and the cache is populated asynchronously
    (write-behind) after the response is sent.
    """
    cache = get_cache()
    if not is_query or not cache.available:
        return await run_sync(
            embedder.embed, texts, is_query, instruction, normalize
        )

    keys = [
        cache.make_key(embedder.model_name, "query", instruction, normalize, t)
        for t in texts
    ]
    results = await cache.get_many(keys)
    miss_idx = [i for i, v in enumerate(results) if v is None]
    hits = len(texts) - len(miss_idx)

    if miss_idx:
        logger.info(
            "Cache: %d/%d query hit(s); embedding %d miss(es) for user '%s'",
            hits,
            len(texts),
            len(miss_idx),
            username,
        )
        miss_texts = [texts[i] for i in miss_idx]
        computed = await run_sync(
            embedder.embed, miss_texts, True, instruction, normalize
        )
        to_store: dict[str, list[float]] = {}
        for j, i in enumerate(miss_idx):
            results[i] = computed[j]
            to_store[keys[i]] = computed[j]
        # Write-behind: never blocks the response, never breaks it if Redis is down.
        background_tasks.add_task(cache.set_many, to_store)
    else:
        logger.info(
            "Cache: %d/%d query hit(s) (all hit) for user '%s'",
            hits,
            len(texts),
            username,
        )

    return results  # type: ignore[return-value]  # all None entries filled above


@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(
    payload: EmbedRequest,
    background_tasks: BackgroundTasks,
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

    vectors = await _embed_texts(
        embedder,
        texts,
        is_query,
        payload.instruction,
        payload.normalize,
        background_tasks,
        current_user.username,
    )

    data = [EmbedItem(index=i, embedding=v) for i, v in enumerate(vectors)]
    return EmbedResponse(
        model=embedder.model_name,
        dim=embedder.dim,
        input_type=payload.input_type,
        normalized=payload.normalize,
        data=data,
    )
