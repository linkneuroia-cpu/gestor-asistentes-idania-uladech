"""Estrategias de embedding disperso (vectores sparse nativos de Qdrant).

BM25 vía FastEmbed es la estrategia por defecto: no requiere API ni
dependencias pesadas (usa ONNX runtime, ya instalado por FastEmbed). BGE-M3
disperso requiere la librería opcional `FlagEmbedding` (no incluida en
requirements.txt): en entornos con Application Control / restricciones de
DLL de Windows, su dependencia transitiva `pyarrow` puede quedar bloqueada
(bloqueó incluso la carga de sentence-transformers al probarla en este
proyecto) — por eso queda fuera del set de dependencias por defecto y se
importa de forma perezosa solo si el usuario selecciona explícitamente
esta estrategia.
"""
from typing import Any, Dict, List

from strategies.base import SparseEmbeddingStrategy

_sparse_model_cache: Dict[str, Any] = {}


def _get_fastembed_sparse_model(model_name: str):
    if model_name not in _sparse_model_cache:
        from fastembed import SparseTextEmbedding

        _sparse_model_cache[model_name] = SparseTextEmbedding(model_name=model_name)
    return _sparse_model_cache[model_name]


class BM25SparseStrategy(SparseEmbeddingStrategy):
    """BM25 (Qdrant/bm25 vía FastEmbed) — estándar de búsqueda léxica
    dispersa, sin API ni GPU."""

    _fastembed_model_name = "Qdrant/bm25"

    async def embed(self, texts: List[str]) -> List[Dict[str, Any]]:
        import asyncio

        loop = asyncio.get_event_loop()

        def _call():
            model = _get_fastembed_sparse_model(self._fastembed_model_name)
            return [
                {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
                for emb in model.embed(texts)
            ]

        return await loop.run_in_executor(None, _call)


class BGEM3SparseStrategy(SparseEmbeddingStrategy):
    """BGE-M3 disperso ("lexical weights"), vía la librería opcional
    FlagEmbedding (BAAI/bge-m3). No forma parte de requirements.txt por su
    dependencia transitiva pesada (pyarrow/datasets) — instalar manualmente
    con `pip install FlagEmbedding` si el entorno lo permite."""

    def __init__(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:
            raise ValueError(
                "La estrategia dispersa 'bge_m3' requiere el paquete opcional "
                "'FlagEmbedding' (pip install FlagEmbedding), no instalado por "
                "defecto porque su dependencia transitiva 'pyarrow' puede "
                "quedar bloqueada por políticas de Application Control en "
                "Windows. Use 'bm25' si no puede instalarlo en este entorno."
            ) from e

        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    async def embed(self, texts: List[str]) -> List[Dict[str, Any]]:
        import asyncio

        loop = asyncio.get_event_loop()

        def _call():
            output = self._model.encode(
                texts, return_dense=False, return_sparse=True, return_colbert_vecs=False
            )
            result = []
            for lexical_weights in output["lexical_weights"]:
                indices = [int(token_id) for token_id in lexical_weights.keys()]
                values = [float(w) for w in lexical_weights.values()]
                result.append({"indices": indices, "values": values})
            return result

        return await loop.run_in_executor(None, _call)
