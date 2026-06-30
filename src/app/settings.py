from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- Model ---
    model_path: str | None = None
    model_repo: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    model_file: str = "Qwen3-Embedding-0.6B-Q8_0.gguf"

    n_ctx: int = 8192
    n_threads: int = 0
    n_gpu_layers: int = -1
    n_batch: int = 512

    # --- Server ---
    app_env: str = "dev"
    allowed_origins: list[str] = ["*"]

    # --- Request limits (DOS guards) ---
    max_inputs_per_request: int = 256
    max_chars_per_input: int = 200_000

    # --- Auth (Tapis) ---
    tapis_issuer: str
    tapis_jwks_url: str
    tapis_tenant_id: str

    # --- Cache (Redis) ---
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    redis_timeout_seconds: float = 0.5
    cache_ttl_seconds: int = 2_592_000
    cache_key_prefix: str = "emb:v1:"
    redis_maxmemory: str | None = None
    redis_maxmemory_policy: str | None = "allkeys-lru"


settings = Settings()  # type: ignore[call-arg]
