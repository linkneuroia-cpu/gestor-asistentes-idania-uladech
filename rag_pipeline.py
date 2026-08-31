"""
rag_pipeline.py
================
Orquestación de lectura del pipeline RAG: recuperación híbrida (dense +
sparse fusionados por Qdrant) → reranking → boosting por `source_type` →
construcción del prompt → generación con el tutor. Usado por los
endpoints `/api/rag/*` en app.py y por la sección "Pruebas del LLM".
"""
from typing import Any, Dict, List, Optional

from qdrant_admin import get_qdrant_admin
from settings import settings
from strategies import registry as strategy_registry
from strategies.generation import SYSTEM_PROMPT_TEMPLATE
from strategies.rerank import apply_source_boost


async def hybrid_retrieve(
    *,
    collection_name: str,
    query: str,
    dense_strategy_name: Optional[str] = None,
    sparse_strategy_name: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Embebe la consulta con las estrategias dense+sparse activas y
    recupera candidatos fusionados (RRF) en una sola llamada a Qdrant."""
    dense = strategy_registry.get_dense_strategy(dense_strategy_name)
    sparse = strategy_registry.get_sparse_strategy(sparse_strategy_name)

    dense_vector = (await dense.embed([query], is_query=True))[0]
    sparse_vector = (await sparse.embed([query]))[0]

    admin = get_qdrant_admin()
    hits = admin.hybrid_search(
        collection_name=collection_name,
        dense_query=dense_vector,
        sparse_indices=sparse_vector["indices"],
        sparse_values=sparse_vector["values"],
        limit=top_k or settings.RERANK_TOP_K,
    )

    return [
        {
            "id": h.id,
            "score": h.score,
            "text": (h.payload or {}).get("text", ""),
            "payload": h.payload or {},
        }
        for h in hits
    ]


async def retrieve_rerank_boost(
    *,
    collection_name: str,
    query: str,
    dense_strategy_name: Optional[str] = None,
    sparse_strategy_name: Optional[str] = None,
    rerank_strategy_name: Optional[str] = None,
    top_k: Optional[int] = None,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """hybrid_retrieve() -> reranker.rerank() sobre todo el pool recuperado
    -> apply_source_boost() (prioriza curso_propio) -> top_n final."""
    candidates = await hybrid_retrieve(
        collection_name=collection_name,
        query=query,
        dense_strategy_name=dense_strategy_name,
        sparse_strategy_name=sparse_strategy_name,
        top_k=top_k,
    )
    if not candidates:
        return []

    n = top_n or settings.RERANK_TOP_N
    reranker = strategy_registry.get_rerank_strategy(rerank_strategy_name)
    reranked = await reranker.rerank(query, candidates, top_n=len(candidates))
    return apply_source_boost(reranked, settings.RERANK_BOOST_MULTIPLIER, top_n=n)


def build_generation_prompt(query: str, candidates: List[Dict[str, Any]]) -> str:
    """Agrupa los candidatos recuperados en bloques [MATERIAL DEL CURSO] /
    [BIBLIOGRAFÍA COMPLEMENTARIA] con citas (filename, diapositiva/minuto/
    fragmento), seguidos de la pregunta del estudiante."""
    curso_blocks: List[str] = []
    biblio_blocks: List[str] = []

    for c in candidates:
        payload = c.get("payload", {})

        citation_parts = [payload.get("filename") or "documento"]
        if payload.get("slide_number") is not None:
            citation_parts.append(f"diapositiva {payload['slide_number']}")
        start_time = payload.get("start_time")
        if start_time is not None:
            minutes, seconds = divmod(int(start_time), 60)
            citation_parts.append(f"minuto {minutes}:{seconds:02d}")
        elif payload.get("chunk") is not None:
            citation_parts.append(f"fragmento {payload['chunk']}")
        citation = ", ".join(str(p) for p in citation_parts)

        header = payload.get("contextual_header") or ""
        text = payload.get("text") or c.get("text", "")
        block = f"[Fuente: {citation}]\n{(header + chr(10)) if header else ''}{text}"

        if payload.get("source_type") == "bibliografia":
            biblio_blocks.append(block)
        else:
            curso_blocks.append(block)

    parts = []
    if curso_blocks:
        parts.append("[MATERIAL DEL CURSO]\n" + "\n\n".join(curso_blocks))
    if biblio_blocks:
        parts.append("[BIBLIOGRAFÍA COMPLEMENTARIA]\n" + "\n\n".join(biblio_blocks))
    context_block = "\n\n".join(parts) if parts else "(sin fragmentos recuperados)"

    return f"{context_block}\n\nPregunta del estudiante: {query}"


async def answer_query(
    *,
    collection_name: str,
    query: str,
    dense_strategy_name: Optional[str] = None,
    sparse_strategy_name: Optional[str] = None,
    rerank_strategy_name: Optional[str] = None,
    generation_strategy_name: Optional[str] = None,
    top_n: Optional[int] = None,
    extra_system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Pipeline completo: retrieve_rerank_boost() -> build_generation_prompt()
    -> generation_strategy.generate(SYSTEM_PROMPT_TEMPLATE [+ extra_system_prompt
    del asistente, si lo hay], user_prompt, history). `extra_system_prompt` se
    AGREGA al prompt fijo, nunca lo reemplaza — así la regla de prioridad
    curso_propio/bibliografía siempre se respeta. `history`: turnos previos
    de la conversación (ver assistants.get_recent_history), para que el
    tutor recuerde lo ya preguntado en la misma sesión."""
    candidates = await retrieve_rerank_boost(
        collection_name=collection_name,
        query=query,
        dense_strategy_name=dense_strategy_name,
        sparse_strategy_name=sparse_strategy_name,
        rerank_strategy_name=rerank_strategy_name,
        top_n=top_n,
    )

    user_prompt = build_generation_prompt(query, candidates)
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    if extra_system_prompt:
        system_prompt = f"{SYSTEM_PROMPT_TEMPLATE}\n\nContexto adicional de este asistente:\n{extra_system_prompt}"
    generator = strategy_registry.get_generation_strategy(generation_strategy_name)
    answer = await generator.generate(system_prompt, user_prompt, history=history)

    sources = [
        {
            "filename": c["payload"].get("filename"),
            "source_type": c["payload"].get("source_type"),
            "chunk": c["payload"].get("chunk"),
            "slide_number": c["payload"].get("slide_number"),
            "start_time": c["payload"].get("start_time"),
            "score": c.get("boosted_score", c.get("rerank_score", c.get("score"))),
            "text": c["payload"].get("text"),
        }
        for c in candidates
    ]

    return {
        "answer": answer,
        "sources": sources,
        "config_used": {
            "dense": strategy_registry.resolve_strategy_name("dense", dense_strategy_name),
            "sparse": strategy_registry.resolve_strategy_name("sparse", sparse_strategy_name),
            "rerank": strategy_registry.resolve_strategy_name("rerank", rerank_strategy_name),
            "generation": strategy_registry.resolve_strategy_name(
                "generation", generation_strategy_name
            ),
        },
    }
