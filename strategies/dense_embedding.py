"""Estrategias de embedding denso.

Reproduce las ramas de proveedor que hoy viven dentro de
`VectorizationService.__init__`/`.embed()` (definitions.py) como clases de
Strategy Pattern independientes, sin modificar `VectorizationService`
(que sigue sirviendo a las colecciones legacy tal cual). Cada instancia
queda ligada a un `model_name` concreto del catálogo
`settings.AVAILABLE_EMBEDDING_MODELS`.
"""
from typing import List

from strategies.base import DenseEmbeddingStrategy


class AzureOpenAIDenseStrategy(DenseEmbeddingStrategy):
    """Azure OpenAI Embeddings, en batches de 100 (mismo límite que
    VectorizationService.embed() rama 'azure')."""

    def __init__(self, model_name: str):
        from settings import settings, get_model_info

        info = get_model_info(model_name)
        self.vector_size = info["dimensions"]
        self._model_name = model_name

        from openai import AzureOpenAI

        self._client = AzureOpenAI(
            api_key=settings.AZURE_EMBEDDING_KEY,
            azure_endpoint=settings.AZURE_EMBEDDING_ENDPOINT,
            api_version=settings.AZURE_EMBEDDING_API_VERSION,
        )
        self._deployment = settings.AZURE_EMBEDDING_DEPLOYMENT

    async def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        import asyncio

        loop = asyncio.get_event_loop()
        results: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await loop.run_in_executor(
                None,
                lambda b=batch: self._client.embeddings.create(
                    model=self._deployment, input=b
                ),
            )
            results.extend(item.embedding for item in response.data)
        return results


class OpenAIDenseStrategy(DenseEmbeddingStrategy):
    """OpenAI Embeddings API directa (no Azure), en batches de 100."""

    def __init__(self, model_name: str):
        from settings import settings, get_model_info

        info = get_model_info(model_name)
        self.vector_size = info["dimensions"]
        # El catálogo usa el prefijo "openai/" para distinguirlo del
        # equivalente Azure; el nombre real para la API va sin el prefijo.
        self._api_model = model_name.split("/", 1)[-1]

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY no está configurado. Requerido para el "
                f"modelo de embedding denso '{model_name}'."
            )

        from openai import OpenAI

        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        import asyncio

        loop = asyncio.get_event_loop()
        results: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await loop.run_in_executor(
                None,
                lambda b=batch: self._client.embeddings.create(
                    model=self._api_model, input=b
                ),
            )
            results.extend(item.embedding for item in response.data)
        return results


class LocalSentenceTransformerDenseStrategy(DenseEmbeddingStrategy):
    """SentenceTransformer local — misma lógica que la rama 'local' de
    VectorizationService.embed() (prefijos E5, task LoRA Jina, o plano)."""

    def __init__(self, model_name: str):
        from settings import get_model_info

        info = get_model_info(model_name)
        self.vector_size = info["dimensions"]
        self._use_prefix = info.get("use_prefix", False)
        self._use_task = info.get("use_task", False)
        self._task_passage = info.get("task_passage", "retrieval.passage")
        self._task_query = info.get("task_query", "retrieval.query")

        import torch
        from sentence_transformers import SentenceTransformer

        # device=None no siempre auto-detecta CUDA en esta versión de
        # sentence-transformers (se vio cargar en CPU con una GPU disponible
        # y torch.cuda.is_available()==True) — se fuerza explícito.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, trust_remote_code=self._use_task, device=device)

    async def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        import asyncio

        loop = asyncio.get_event_loop()

        if self._use_task:
            task = self._task_query if is_query else self._task_passage
            return await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts, task=task, normalize_embeddings=True
                ).tolist(),
            )

        if self._use_prefix:
            prefix = "query: " if is_query else "passage: "
            inputs = [prefix + t for t in texts]
        else:
            inputs = texts

        return await loop.run_in_executor(
            None,
            lambda: self._model.encode(inputs, normalize_embeddings=True).tolist(),
        )


_PROVIDER_CLASS_MAP = {
    "azure": AzureOpenAIDenseStrategy,
    "openai": OpenAIDenseStrategy,
    "local": LocalSentenceTransformerDenseStrategy,
}


def build_dense_strategy(model_name: str) -> DenseEmbeddingStrategy:
    """Factory usada por strategies.registry: elige la clase de proveedor
    según settings.AVAILABLE_EMBEDDING_MODELS[model_name]["provider"]."""
    from settings import get_model_info

    info = get_model_info(model_name)
    provider = info.get("provider", "local")
    strategy_class = _PROVIDER_CLASS_MAP.get(provider, LocalSentenceTransformerDenseStrategy)
    return strategy_class(model_name)
