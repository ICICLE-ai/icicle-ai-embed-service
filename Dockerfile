# syntax=docker/dockerfile:1.7

# ---- Build stage --------------------------------------------------------------
# Compiles llama-cpp-python's native extension. ccache + BuildKit cache mounts
# turn the second build into a near-no-op even when source changes.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Allow BuildKit to cache apt's downloaded .debs across builds.
RUN rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ccache

# Route compiler invocations through ccache so re-builds of llama.cpp are fast.
ENV PATH="/usr/lib/ccache:${PATH}" \
    CCACHE_DIR=/root/.cache/ccache

COPY pyproject.toml ./
COPY src/ ./src/

# Cache pip's wheel cache and ccache between builds — the heavy line is the
# llama-cpp-python C++ compile, and these mounts skip it after the first build.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install --prefix=/install .

# Drop dev-only packages that ride along with pip but aren't needed at runtime.
RUN find /install/lib/python3.11/site-packages -maxdepth 1 \
        \( -name 'pip*' -o -name 'setuptools*' -o -name 'wheel*' \
           -o -name '_distutils_hack*' -o -name 'pkg_resources*' \) \
        -exec rm -rf {} + && \
    find /install -depth -name '__pycache__' -exec rm -rf {} + && \
    find /install -depth -name '*.pyc' -delete


# ---- Runtime stage ------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    XDG_CACHE_HOME=/home/app/.cache

RUN rm -f /etc/apt/apt.conf.d/docker-clean

# libgomp1 is required by llama.cpp's OpenMP-parallelized matmul kernels.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/false --create-home app

# Pull only the trimmed runtime artifacts from the builder stage.
COPY --from=builder /install/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /install/bin /usr/local/bin

WORKDIR /home/app
COPY src/ ./src/

RUN mkdir -p "$HF_HOME" && chown -R app:app /home/app

# Declare the cache directory as a mount point so deployers know where to plug
# in a persistent volume for model weights. The image does NOT name a specific
# volume — that is a deployment concern (Docker -v / k8s PVC / Tapis Pods volume).
VOLUME ["/home/app/.cache/huggingface"]

USER app

EXPOSE 8000

# Loading the GGUF takes time; give the first probe a long start-period.
# Uses python (already in the image) instead of curl to keep the layer slim.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status==200 else sys.exit(1)" || exit 1

ENTRYPOINT ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
