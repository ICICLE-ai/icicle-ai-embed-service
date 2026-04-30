from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from threading import Lock

from llama_cpp import Llama

from .settings import settings

logger = logging.getLogger(__name__)


# Qwen3-Embedding is instruction-aware: queries get wrapped with a task prompt,
# documents are embedded as-is. See the model card:
# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _format_query(text: str, instruction: str | None) -> str:
    task = instruction or DEFAULT_QUERY_INSTRUCTION
    return f"Instruct: {task}\nQuery: {text}"


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _resolve_model_path() -> str:
    if settings.model_path:
        path = Path(settings.model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"MODEL_PATH points to '{path}' but that file does not exist."
            )
        return str(path)

    from huggingface_hub import hf_hub_download

    logger.info(
        "Downloading %s / %s from Hugging Face (cached)",
        settings.model_repo,
        settings.model_file,
    )
    return hf_hub_download(
        repo_id=settings.model_repo,
        filename=settings.model_file,
    )


class Embedder:
    def __init__(self) -> None:
        model_path = _resolve_model_path()
        logger.info("Loading GGUF model from %s", model_path)

        # Don't override pooling_type — llama.cpp reads it from the GGUF metadata
        # (Qwen3-Embedding is baked with last-token pooling). Overriding corrupts vectors.
        kwargs: dict = dict(
            model_path=model_path,
            embedding=True,
            n_ctx=settings.n_ctx,
            n_batch=settings.n_batch,
            n_gpu_layers=settings.n_gpu_layers,
            verbose=False,
        )
        if settings.n_threads > 0:
            kwargs["n_threads"] = settings.n_threads

        self._llm = Llama(**kwargs)
        # llama-cpp-python's embed() mutates the shared context; serialize access.
        self._lock = Lock()
        self._model_file = os.path.basename(model_path)
        self._dim = self._probe_dim()
        logger.info(
            "Model ready (dim=%d, ctx=%d, model=%s)",
            self._dim,
            settings.n_ctx,
            self._model_file,
        )

    def _probe_dim(self) -> int:
        with self._lock:
            vec = self._llm.embed("dimension probe")
        if vec and isinstance(vec[0], list):
            vec = vec[0]
        return len(vec)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        if settings.model_path:
            return Path(settings.model_path).stem
        return f"{settings.model_repo}/{settings.model_file}"

    def embed(
        self,
        texts: list[str],
        is_query: bool,
        instruction: str | None,
        normalize: bool,
    ) -> list[list[float]]:
        prepared = [_format_query(t, instruction) if is_query else t for t in texts]
        out: list[list[float]] = []
        with self._lock:
            for text in prepared:
                vec = self._llm.embed(text)
                if vec and isinstance(vec[0], list):
                    vec = vec[0]
                vec = [float(x) for x in vec]
                if normalize:
                    vec = _l2_normalize(vec)
                out.append(vec)
        return out


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    if _embedder is None:
        raise RuntimeError("Embedder not initialised. Lifespan did not run.")
    return _embedder


def init_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
