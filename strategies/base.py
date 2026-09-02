"""Interfaces (ABC) para cada etapa configurable del pipeline RAG."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class DocumentExtractionStrategy(ABC):
    """Extrae el contenido de un documento (PDF/PPT/Word) como Markdown
    limpio, con tablas completas y descripción de imágenes/gráficos."""

    @abstractmethod
    async def extract(self, file_path: str) -> str:
        ...


class AudioTranscriptionStrategy(ABC):
    """Transcribe audio o video a segmentos con timestamps."""

    @abstractmethod
    async def transcribe(self, file_path: str) -> List[Dict[str, Any]]:
        """Retorna una lista de {"start": float, "end": float, "text": str}."""
        ...


class ContextualEnrichmentStrategy(ABC):
    """Genera una cabecera contextual de 50-100 palabras para un chunk,
    dado el documento completo del que proviene (Contextual Retrieval)."""

    @abstractmethod
    async def enrich(self, full_document_text: str, chunk_text: str) -> str:
        """Retorna la cabecera contextual, o "" si no aplica."""
        ...


class DenseEmbeddingStrategy(ABC):
    """Genera vectores densos para texto."""

    vector_size: int

    @abstractmethod
    async def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        ...


class SparseEmbeddingStrategy(ABC):
    """Genera vectores dispersos (BM25/BGE-M3) nativos de Qdrant."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Retorna una lista de {"indices": List[int], "values": List[float]}."""
        ...


class RerankStrategy(ABC):
    """Reordena candidatos recuperados por relevancia real a la consulta."""

    @abstractmethod
    async def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        """`candidates`: lista de dicts con al menos {"text": str, ...}.
        Retorna los `top_n` candidatos reordenados, cada uno con un campo
        "rerank_score" agregado."""
        ...


class GenerationStrategy(ABC):
    """Genera la respuesta final del tutor a partir del prompt del sistema
    y el prompt de usuario (contexto recuperado + pregunta)."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Optional[int]]:
        """`history`: turnos previos de la conversación (hasta 30 pares
        pregunta/respuesta), cada uno {"role": "user"|"assistant",
        "content": str}, en orden cronológico — se insertan entre el
        system prompt y la pregunta actual como turnos reales, no como
        texto aplanado. Retorna (respuesta, tokens_totales_consumidos —
        None si el proveedor no lo informa)."""
        ...
