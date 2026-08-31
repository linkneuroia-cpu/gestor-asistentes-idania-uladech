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
        "requires_key": "AZURE_EMBEDDING_KEY",
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

    # ── 5. OpenAI text-embedding-3-small (no Azure) ──────────────────
    # 1536 dims · contexto 8191 tokens · vía API OpenAI directa
    "openai/text-embedding-3-small": {
        "dimensions":   1536,
        "params_M":     None,
        "size_gb":      0,
        "max_tokens":   8191,
        "description": (
            "OpenAI text-embedding-3-small (1536 dims) vía API directa de "
            "OpenAI (no Azure). No requiere GPU local."
        ),
        "use_prefix":   False,
        "use_task":     False,
        "provider":     "openai",
        "requires_key": "OPENAI_API_KEY",
    },
}


# ═══════════════════════════════════════════════════════════════════
# CATÁLOGOS DE ESTRATEGIAS DEL PIPELINE RAG (Strategy Pattern)
# ═══════════════════════════════════════════════════════════════════
#
# Cada catálogo mapea nombre_de_estrategia -> {provider, description,
# requires_key}. "requires_key" es el nombre del campo en Settings que debe
# tener un valor no vacío para que la estrategia sea utilizable; None si no
# requiere ninguna API key (estrategias locales).

AVAILABLE_ETL_DOCUMENT_STRATEGIES: Dict[str, Dict] = {
    "local": {
        "provider": "local",
        "description": (
            "Extracción local (PyMuPDF/pdfplumber/python-docx/pptx/EasyOCR). "
            "Gratis, rápida, sin llamadas a API."
        ),
        "requires_key": None,
    },
    "gemini_vision": {
        "provider": "google",
        "description": (
            "Gemini 1.5 Flash — lectura nativa de PDF, produce Markdown "
            "limpio con descripción detallada de imágenes/tablas."
        ),
        "requires_key": "GOOGLE_API_KEY",
    },
    "gpt4o_mini_vision": {
        "provider": "openai",
        "description": (
            "GPT-4o-mini Vision — PyMuPDF convierte cada página a imagen, "
            "GPT-4o-mini genera Markdown a partir de las imágenes."
        ),
        "requires_key": "OPENAI_API_KEY",
    },
    "deepseek_vision": {
        "provider": "deepseek",
        "description": (
            "DeepSeek Vision — PyMuPDF convierte cada página a imagen, "
            "DeepSeek genera Markdown a partir de las imágenes."
        ),
        "requires_key": "DEEPSEEK_API_KEY",
    },
}

AVAILABLE_ETL_AUDIO_STRATEGIES: Dict[str, Dict] = {
    "faster_whisper_local": {
        "provider": "local",
        "description": "faster-whisper local — gratis, ya en uso en el sistema.",
        "requires_key": None,
    },
    "whisper_api": {
        "provider": "openai",
        "description": "OpenAI Whisper API — transcripción en la nube.",
        "requires_key": "OPENAI_API_KEY",
    },
    "deepgram": {
        "provider": "deepgram",
        "description": "Deepgram — transcripción de alta velocidad con diarización de hablantes.",
        "requires_key": "DEEPGRAM_API_KEY",
    },
}

AVAILABLE_CONTEXTUAL_STRATEGIES: Dict[str, Dict] = {
    "none": {
        "provider": None,
        "description": "Sin enriquecimiento contextual (comportamiento actual, sin costo adicional).",
        "requires_key": None,
    },
    "deepseek": {
        "provider": "deepseek",
        "description": "DeepSeek genera una cabecera contextual de 50-100 palabras por chunk antes de vectorizar.",
        "requires_key": "DEEPSEEK_API_KEY",
    },
    "gpt4o_mini": {
        "provider": "openai",
        "description": "GPT-4o-mini genera una cabecera contextual de 50-100 palabras por chunk antes de vectorizar.",
        "requires_key": "OPENAI_API_KEY",
    },
}

AVAILABLE_SPARSE_EMBEDDING_STRATEGIES: Dict[str, Dict] = {
    "bm25": {
        "provider": "fastembed",
        "description": "BM25 (Qdrant/bm25 vía FastEmbed) — estándar de búsqueda léxica dispersa.",
        "requires_key": None,
    },
    "bge_m3": {
        "provider": "fastembed",
        "description": "BGE-M3 disperso (Qdrant/bge-m3 vía FastEmbed) — mayor calidad semántica dispersa.",
        "requires_key": None,
    },
}

AVAILABLE_RERANK_STRATEGIES: Dict[str, Dict] = {
    "bge_local": {
        "provider": "local",
        "description": "BGE-Reranker (cross-encoder local, BAAI/bge-reranker-v2-m3) — gratis, sin API.",
        "requires_key": None,
    },
    "cohere": {
        "provider": "cohere",
        "description": "Cohere Rerank v3 — reranking en la nube de alta calidad.",
        "requires_key": "COHERE_API_KEY",
    },
}

AVAILABLE_GENERATION_MODELS: Dict[str, Dict] = {
    "gpt4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "description": "GPT-4o — máxima calidad de generación.",
        "requires_key": "OPENAI_API_KEY",
    },
    "gpt4o_mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "description": "GPT-4o-mini — rápido y económico.",
        "requires_key": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider": "google",
        "model": "gemini-1.5-flash",
        "description": "Gemini 1.5 Flash.",
        "requires_key": "GOOGLE_API_KEY",
    },
    "deepseek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "description": "DeepSeek Chat.",
        "requires_key": "DEEPSEEK_API_KEY",
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
    QDRANT_API_KEY: str = ""
    QDRANT_HTTPS: bool = False
    DEFAULT_VECTOR_SIZE: int = 1536
    DEFAULT_DISTANCE: str = "Cosine"

    @property
    def QDRANT_PROTOCOL(self) -> str:
        return "https" if self.QDRANT_HTTPS else "http"

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

    # ================= LLM Generation & Vision (Pipeline RAG) =================
    # Todas opcionales — se validan de forma perezosa por estrategia (no
    # bloquean el arranque de la app si faltan; solo la estrategia que las
    # requiere queda inutilizable hasta que se configuren).
    OPENAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    COHERE_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""

    # ================= RAG Pipeline Defaults (Strategy Pattern) =================
    # Selección por defecto de estrategia por etapa, usada cuando no hay
    # override en runtime_config. Ver AVAILABLE_*_STRATEGIES arriba.
    ETL_DOCUMENT_STRATEGY: str = "local"
    ETL_AUDIO_STRATEGY: str = "faster_whisper_local"
    CONTEXTUAL_STRATEGY: str = "none"
    SPARSE_EMBEDDING_STRATEGY: str = "bm25"
    RERANK_STRATEGY: str = "bge_local"
    RERANK_TOP_K: int = 50
    RERANK_TOP_N: int = 5
    RERANK_BOOST_MULTIPLIER: float = 1.15
    GENERATION_STRATEGY: str = "gpt4o_mini"

    # ================= Postgres (configuración del gestor, asistentes) =================
    PG_HOST: str = "10.0.0.92"
    PG_PORT: int = 5432
    PG_USER: str = "postgres"
    PG_PASSWORD: str = ""
    PG_DATABASE: str = "U_F1_Profundizacion"

    # ================= Autenticación del gestor =================
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "changeme123"
    SESSION_SECRET_KEY: str = "change-this-session-secret-in-env"

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


# Catálogos indexados por nombre de etapa, para resolución genérica
# (usado por strategies/registry.py).
STAGE_CATALOGS: Dict[str, Dict[str, Dict]] = {
    "etl_document": AVAILABLE_ETL_DOCUMENT_STRATEGIES,
    "etl_audio": AVAILABLE_ETL_AUDIO_STRATEGIES,
    "contextual": AVAILABLE_CONTEXTUAL_STRATEGIES,
    "dense": AVAILABLE_EMBEDDING_MODELS,
    "sparse": AVAILABLE_SPARSE_EMBEDDING_STRATEGIES,
    "rerank": AVAILABLE_RERANK_STRATEGIES,
    "generation": AVAILABLE_GENERATION_MODELS,
}

STAGE_DEFAULTS: Dict[str, str] = {
    "etl_document": "ETL_DOCUMENT_STRATEGY",
    "etl_audio": "ETL_AUDIO_STRATEGY",
    "contextual": "CONTEXTUAL_STRATEGY",
    "dense": "EMBEDDING_MODEL",
    "sparse": "SPARSE_EMBEDDING_STRATEGY",
    "rerank": "RERANK_STRATEGY",
    "generation": "GENERATION_STRATEGY",
}


def get_strategy_info(stage: str, strategy_name: str) -> Dict:
    """Busca `strategy_name` en el catálogo de `stage` (una de las claves de
    STAGE_CATALOGS). Lanza ValueError con las opciones válidas si no existe,
    igual que get_model_info()."""
    if stage not in STAGE_CATALOGS:
        raise ValueError(
            f"Etapa '{stage}' desconocida. Etapas válidas: {list(STAGE_CATALOGS.keys())}"
        )
    catalog = STAGE_CATALOGS[stage]
    if strategy_name not in catalog:
        raise ValueError(
            f"Estrategia '{strategy_name}' no está en el catálogo de '{stage}'. "
            f"Disponibles: {list(catalog.keys())}"
        )
    return catalog[strategy_name]


def strategy_is_usable(stage: str, strategy_name: str) -> bool:
    """True si la estrategia no requiere API key, o si la key requerida
    tiene un valor no vacío en settings."""
    info = get_strategy_info(stage, strategy_name)
    requires_key = info.get("requires_key")
    if not requires_key:
        return True
    return bool(getattr(settings, requires_key, "") or "")


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