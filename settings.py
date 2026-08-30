from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List, Dict


# ═══════════════════════════════════════════════════════════════════
# CATÁLOGO DE MODELOS
# ═══════════════════════════════════════════════════════════════════
#
# provider="azure" → usa Azure OpenAI Embeddings API (text-embedding-3-small)
# provider="local" → usa SentenceTransformer (descarga local desde HuggingFace)

AVAILABLE_EMBEDDING_MODELS: Dict[str, Dict] = {

    # ── 1. Azure OpenAI text-embedding-3-small ──────────────────────
    # 1536 dims · contexto 8191 tokens · vía API Azure (no descarga local)
    "text-embedding-3-small": {
        "dimensions":   1536,
        "params_M":     None,
        "size_gb":      0,
        "max_tokens":   8191,
        "description": (
            "Azure OpenAI text-embedding-3-small (1536 dims). "
            "Vía API — no requiere GPU local."
        ),
        "use_prefix":   False,
        "use_task":     False,
        "provider":     "azure",
    },

    # ── 2. E5-Large — local, alta calidad en español ─────────────────
    "intfloat/multilingual-e5-large": {
        "dimensions":   1024,
        "params_M":     560,
        "size_gb":      2.24,
        "max_tokens":   512,
        "description": (
            "Alta calidad semántica local (560M, ~2.2 GB). "
            "No requiere API externa."
        ),
        "use_prefix":   True,
        "use_task":     False,
        "provider":     "local",
    },

    # ── 3. Jina Embeddings v3 — local, máxima calidad ────────────────
    "jinaai/jina-embeddings-v3": {
        "dimensions":   1024,
        "params_M":     570,
        "size_gb":      2.24,
        "max_tokens":   8192,
        "description": (
            "Máxima calidad local (570M, ~2.2 GB). Contexto 8192 tokens. "
            "Requiere trust_remote_code=True."
        ),
        "use_prefix":   False,
        "use_task":     True,
        "task_passage": "retrieval.passage",
        "task_query":   "retrieval.query",
        "provider":     "local",
    },

    # ── 4. all-MiniLM-L6-v2 — local, ligero y rápido ──────────────────
    # 384 dims · contexto 256 tokens · modelo entrenado mayormente en inglés,
    # con soporte multilingüe limitado. Útil para pruebas rápidas o bajo
    # consumo de recursos; para español se recomiendan los modelos 2 o 3.
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions":   384,
        "params_M":     22.7,
        "size_gb":      0.09,
        "max_tokens":   256,
        "description": (
            "Modelo local ligero y rápido (22.7M, ~90 MB). 384 dims. "
            "Soporte multilingüe limitado (entrenado mayormente en inglés)."
        ),
        "use_prefix":   False,
        "use_task":     False,
        "provider":     "local",
    },
}


# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

class Settings(BaseSettings):
    # ================= API =================
    API_TITLE: str
    API_VERSION: str
    API_HOST: str
    API_PORT: int
    DEBUG: bool

    # ================= Storage =================
    TEMP_DIR: Path
    MAX_FILE_SIZE: int
    ALLOWED_EXTENSIONS: List[str]

    # ================= Qdrant =================
    QDRANT_HOST: str
    QDRANT_PORT: int

    # ================= API Externa =================
    COLLECTIONS_API_URL: str

    # ================= Embeddings =================
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int

    # ================= Azure Embeddings =================
    # Solo requerido cuando EMBEDDING_MODEL = "text-embedding-3-small"
    AZURE_EMBEDDING_KEY: str = ""
    AZURE_EMBEDDING_ENDPOINT: str = ""
    AZURE_EMBEDDING_DEPLOYMENT: str = "text-embedding-3-small"
    AZURE_EMBEDDING_API_VERSION: str = "2024-05-01-preview"

    # ================= Text Processing =================
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # ================= ASR (audio/video) =================
    ASR_MODEL_SIZE: str = "base"
    ASR_DEVICE: str = "cpu"
    ASR_COMPUTE_TYPE: str = "int8"

    # ================= OCR (imagenes) =================
    OCR_LANGUAGES: str = "es,en"

    # ================= Moodle =================
    # Solo requiere el token (con permiso de descarga de archivos vía
    # webservice habilitado) — ya no se hace login con usuario/contraseña.
    MOODLE_URL: str
    MOODLE_TOKEN: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()


def get_ocr_languages() -> List[str]:
    return [lang.strip() for lang in settings.OCR_LANGUAGES.split(",") if lang.strip()]


# ═══════════════════════════════════════════════════════════════════
# HELPERS DE CATÁLOGO
# ═══════════════════════════════════════════════════════════════════

def get_model_info(model_name: str) -> Dict:
    if model_name not in AVAILABLE_EMBEDDING_MODELS:
        valid = list(AVAILABLE_EMBEDDING_MODELS.keys())
        raise ValueError(
            f"Modelo '{model_name}' no está en el catálogo. "
            f"Modelos disponibles: {valid}"
        )
    return AVAILABLE_EMBEDDING_MODELS[model_name]


def get_model_dimensions(model_name: str) -> int:
    return get_model_info(model_name)["dimensions"]


def uses_prefix(model_name: str) -> bool:
    return get_model_info(model_name).get("use_prefix", False)


# ═══════════════════════════════════════════════════════════════════
# VALIDACIÓN AL INICIO
# ═══════════════════════════════════════════════════════════════════

def validate_settings():
    settings.TEMP_DIR.mkdir(exist_ok=True, parents=True)

    if settings.EMBEDDING_MODEL not in AVAILABLE_EMBEDDING_MODELS:
        raise ValueError(
            f"❌ EMBEDDING_MODEL='{settings.EMBEDDING_MODEL}' no está en el catálogo. "
            f"Modelos válidos: {list(AVAILABLE_EMBEDDING_MODELS.keys())}"
        )

    info = AVAILABLE_EMBEDDING_MODELS[settings.EMBEDDING_MODEL]
    expected_dim = info["dimensions"]
    provider = info.get("provider", "local")

    if settings.EMBEDDING_DIMENSION != expected_dim:
        print(
            f"⚠️  EMBEDDING_DIMENSION={settings.EMBEDDING_DIMENSION} en .env "
            f"no coincide con el catálogo ({expected_dim}). "
            f"Se usará la dimensión del catálogo: {expected_dim}"
        )

    if provider == "azure":
        if not settings.AZURE_EMBEDDING_KEY:
            raise ValueError(
                "❌ AZURE_EMBEDDING_KEY no está configurado en .env. "
                "Requerido para usar text-embedding-3-small."
            )
        if not settings.AZURE_EMBEDDING_ENDPOINT:
            raise ValueError(
                "❌ AZURE_EMBEDDING_ENDPOINT no está configurado en .env. "
                "Requerido para usar text-embedding-3-small."
            )
        print(f"✅ Modelo por defecto : {settings.EMBEDDING_MODEL}")
        print(f"   Proveedor         : Azure OpenAI")
        print(f"   Endpoint          : {settings.AZURE_EMBEDDING_ENDPOINT}")
        print(f"   Deployment        : {settings.AZURE_EMBEDDING_DEPLOYMENT}")
        print(f"   Dimensión         : {expected_dim}")
        print(f"   Contexto máx      : {info['max_tokens']} tokens")
    else:
        chars_per_token = 4
        approx_tokens = settings.CHUNK_SIZE // chars_per_token
        if approx_tokens > info["max_tokens"]:
            print(
                f"⚠️  CHUNK_SIZE={settings.CHUNK_SIZE} chars (~{approx_tokens} tokens) "
                f"supera el contexto del modelo ({info['max_tokens']} tokens). "
                f"El texto será truncado silenciosamente."
            )
        print(f"✅ Modelo por defecto : {settings.EMBEDDING_MODEL}")
        print(f"   Proveedor         : Local (HuggingFace)")
        print(f"   Parámetros        : {info['params_M']}M")
        print(f"   Peso aproximado   : ~{info['size_gb']} GB")
        print(f"   Dimensión         : {expected_dim}")
        print(f"   Contexto máx      : {info['max_tokens']} tokens")
        print(f"   CHUNK_SIZE actual : {settings.CHUNK_SIZE} chars (~{approx_tokens} tokens)")