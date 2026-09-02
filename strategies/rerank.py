"""Estrategias de reranking: reordenan el Top-K recuperado por Qdrant a un
Top-N hiperrelevante. `apply_source_boost` aplica el multiplicador de
prioridad a `source_type == "curso_propio"` después del reranking, para
que el material propio del curso siempre gane frente a bibliografía
igualmente relevante según el cross-encoder/API.
"""
from typing import Any, Dict, List

from strategies.base import RerankStrategy

_cross_encoder_cache: Dict[str, Any] = {}


def _get_cross_encoder(model_name: str):
    if model_name not in _cross_encoder_cache:
        import torch
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        # El parámetro device=... del constructor NO mueve el modelo en esta
        # combinación de versiones (sentence-transformers 3.1.1 +
        # transformers 4.57.6): queda en CPU aunque se pida "cuda" y haya
        # GPU disponible. Se fuerza moviendo el modelo interno con .to() —
        # medido en vivo: 27.8s -> 0.7s para 30 candidatos (~40x). Este es
        # el modelo más pesado del pipeline (568M params, hasta 50
        # candidatos por consulta) — el que más se beneficia de GPU.
        if torch.cuda.is_available():
            model.model.to("cuda")
        _cross_encoder_cache[model_name] = model
    return _cross_encoder_cache[model_name]


class BGERerankStrategy(RerankStrategy):
    """BGE-Reranker (cross-encoder local, BAAI/bge-reranker-v2-m3). Gratis,
    sin API, pero requiere descargar el modelo (~1-2 GB) la primera vez."""

    _model_name = "BAAI/bge-reranker-v2-m3"

    async def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        import asyncio

        if not candidates:
            return []

        loop = asyncio.get_event_loop()

        def _call():
            model = _get_cross_encoder(self._model_name)
            pairs = [(query, c["text"]) for c in candidates]
            scores = model.predict(pairs)
            for c, score in zip(candidates, scores):
                c["rerank_score"] = float(score)
            ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
            return ranked[:top_n]

        return await loop.run_in_executor(None, _call)


class CohereRerankStrategy(RerankStrategy):
    """Cohere Rerank v3 — reranking en la nube de alta calidad."""

    _model_name = "rerank-v3.5"

    def __init__(self):
        from settings import settings

        if not settings.COHERE_API_KEY:
            raise ValueError(
                "COHERE_API_KEY no está configurado. Requerido para la "
                "estrategia de reranking 'cohere'."
            )
        import cohere

        self._client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)

    async def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int
    ) -> List[Dict[str, Any]]:
        import asyncio

        if not candidates:
            return []

        loop = asyncio.get_event_loop()

        def _call():
            documents = [c["text"] for c in candidates]
            response = self._client.rerank(
                model=self._model_name,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            ranked = []
            for result in response.results:
                candidate = candidates[result.index]
                candidate["rerank_score"] = float(result.relevance_score)
                ranked.append(candidate)
            return ranked

        return await loop.run_in_executor(None, _call)


def apply_source_boost(
    candidates: List[Dict[str, Any]],
    boost_multiplier: float,
    top_n: int,
) -> List[Dict[str, Any]]:
    """Multiplica `rerank_score` por `boost_multiplier` para los candidatos
    con `payload.source_type == "curso_propio"`, reordena de mayor a menor,
    y trunca a `top_n`."""
    for c in candidates:
        payload = c.get("payload", {})
        base_score = c.get("rerank_score", 0.0)
        if payload.get("source_type") == "curso_propio":
            c["boosted_score"] = base_score * boost_multiplier
        else:
            c["boosted_score"] = base_score

    ranked = sorted(candidates, key=lambda c: c["boosted_score"], reverse=True)
    return ranked[:top_n]
