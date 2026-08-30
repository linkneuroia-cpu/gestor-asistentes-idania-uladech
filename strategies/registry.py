"""Factories por etapa del pipeline RAG.

Resuelve, para cada etapa, qué estrategia concreta usar:
override en memoria (runtime_config) -> default de .env -> catálogo de
settings.py (error claro si el nombre no existe). Cachea una instancia por
(etapa, nombre) — mismo patrón de singleton perezoso que
`definitions.get_service()` usa hoy para VectorizationService.
"""
from typing import Any, Dict, Optional

from settings import settings, get_strategy_info
from strategies.runtime_config import get_runtime_config

_DEFAULT_SETTING_FIELD: Dict[str, str] = {
    "etl_document": "ETL_DOCUMENT_STRATEGY",
    "etl_audio": "ETL_AUDIO_STRATEGY",
    "contextual": "CONTEXTUAL_STRATEGY",
    "dense": "EMBEDDING_MODEL",
    "sparse": "SPARSE_EMBEDDING_STRATEGY",
    "rerank": "RERANK_STRATEGY",
    "generation": "GENERATION_STRATEGY",
}

_instances: Dict[str, Dict[str, Any]] = {stage: {} for stage in _DEFAULT_SETTING_FIELD}


def resolve_strategy_name(stage: str, name: Optional[str] = None) -> str:
    """override explícito (validado contra el catálogo) -> runtime_config
    -> default de .env."""
    if name:
        get_strategy_info(stage, name)  # valida que exista, lanza ValueError si no
        return name
    override = get_runtime_config().get(stage)
    if override:
        return override
    return getattr(settings, _DEFAULT_SETTING_FIELD[stage])


def get_all_active() -> Dict[str, str]:
    """Nombre de estrategia vigente por etapa — usado por HybridVectorizationService
    para construir el PipelineConfig automáticamente, y por la UI de
    configuración para mostrar la selección actual."""
    return {stage: resolve_strategy_name(stage) for stage in _DEFAULT_SETTING_FIELD}


def _build(stage: str, name: str) -> Any:
    if stage == "etl_document":
        from strategies.etl_document import (
            DeepSeekVisionStrategy,
            GeminiVisionStrategy,
            GPT4oMiniVisionStrategy,
            LocalExtractionStrategy,
        )

        mapping = {
            "local": LocalExtractionStrategy,
            "gemini_vision": GeminiVisionStrategy,
            "gpt4o_mini_vision": GPT4oMiniVisionStrategy,
            "deepseek_vision": DeepSeekVisionStrategy,
        }
        return mapping[name]()

    if stage == "etl_audio":
        from strategies.etl_audio import (
            DeepgramStrategy,
            FasterWhisperLocalStrategy,
            OpenAIWhisperAPIStrategy,
        )

        mapping = {
            "faster_whisper_local": FasterWhisperLocalStrategy,
            "whisper_api": OpenAIWhisperAPIStrategy,
            "deepgram": DeepgramStrategy,
        }
        return mapping[name]()

    if stage == "contextual":
        from strategies.contextual import (
            DeepSeekContextualStrategy,
            GPT4oMiniContextualStrategy,
            NoOpContextualStrategy,
        )

        mapping = {
            "none": NoOpContextualStrategy,
            "deepseek": DeepSeekContextualStrategy,
            "gpt4o_mini": GPT4oMiniContextualStrategy,
        }
        return mapping[name]()

    if stage == "dense":
        from strategies.dense_embedding import build_dense_strategy

        return build_dense_strategy(name)

    if stage == "sparse":
        from strategies.sparse_embedding import BGEM3SparseStrategy, BM25SparseStrategy

        mapping = {"bm25": BM25SparseStrategy, "bge_m3": BGEM3SparseStrategy}
        return mapping[name]()

    if stage == "rerank":
        from strategies.rerank import BGERerankStrategy, CohereRerankStrategy

        mapping = {"bge_local": BGERerankStrategy, "cohere": CohereRerankStrategy}
        return mapping[name]()

    if stage == "generation":
        from strategies.generation import (
            DeepSeekGenerationStrategy,
            GeminiGenerationStrategy,
            GPT4oGenerationStrategy,
            GPT4oMiniGenerationStrategy,
        )

        mapping = {
            "gpt4o": GPT4oGenerationStrategy,
            "gpt4o_mini": GPT4oMiniGenerationStrategy,
            "gemini": GeminiGenerationStrategy,
            "deepseek": DeepSeekGenerationStrategy,
        }
        return mapping[name]()

    raise ValueError(f"Etapa '{stage}' desconocida. Etapas válidas: {list(_DEFAULT_SETTING_FIELD)}")


def _get_or_build(stage: str, name: Optional[str] = None) -> Any:
    resolved_name = resolve_strategy_name(stage, name)
    cache = _instances[stage]
    if resolved_name not in cache:
        cache[resolved_name] = _build(stage, resolved_name)
    return cache[resolved_name]


def get_etl_document_strategy(name: Optional[str] = None):
    return _get_or_build("etl_document", name)


def get_etl_audio_strategy(name: Optional[str] = None):
    return _get_or_build("etl_audio", name)


def get_contextual_strategy(name: Optional[str] = None):
    return _get_or_build("contextual", name)


def get_dense_strategy(name: Optional[str] = None):
    return _get_or_build("dense", name)


def get_sparse_strategy(name: Optional[str] = None):
    return _get_or_build("sparse", name)


def get_rerank_strategy(name: Optional[str] = None):
    return _get_or_build("rerank", name)


def get_generation_strategy(name: Optional[str] = None):
    return _get_or_build("generation", name)
