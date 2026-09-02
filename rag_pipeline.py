"""
rag_pipeline.py
================
Orquestación de lectura del pipeline RAG: recuperación híbrida (dense +
sparse fusionados por Qdrant) → reranking → boosting por `source_type` →
construcción del prompt → generación con el tutor. Usado por los
endpoints `/api/rag/*` en app.py y por la sección "Pruebas del LLM".
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

from qdrant_admin import get_qdrant_admin
from settings import settings
from strategies import registry as strategy_registry
from strategies.generation import SYSTEM_PROMPT_TEMPLATE
from strategies.rerank import apply_source_boost

# `rerank_score` es la probabilidad de relevancia del cross-encoder (0-1,
# sigmoid ya aplicado por BGE/Cohere) ANTES del boost de curso_propio.
# retrieve_rerank_boost() siempre devuelve top_n candidatos aunque ninguno
# sea realmente relevante (p.ej. preguntas conversacionales/meta que no
# citan ningún documento, o "no entiendo la parte X" sin contexto previo)
# — sin este filtro esos candidatos igual se mandaban al LLM como contexto
# (y se mostraban como "fuentes"), y el modelo terminaba construyendo una
# respuesta con contenido con ~0.001% de relevancia real en vez de admitir
# que no tiene con qué responder. Se filtra sobre el score SIN boost: el
# boost es para priorizar orden entre candidatos ya relevantes, no una
# señal de relevancia en sí misma.
MIN_SOURCE_RELEVANCE = 0.01


async def hybrid_retrieve(
    *,
    collection_name: str,
    query: str,
    dense_strategy_name: Optional[str] = None,
    sparse_strategy_name: Optional[str] = None,
    top_k: Optional[int] = None,
    timings: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Embebe la consulta con las estrategias dense+sparse activas y
    recupera candidatos fusionados (RRF) en una sola llamada a Qdrant."""
    dense = strategy_registry.get_dense_strategy(dense_strategy_name)
    sparse = strategy_registry.get_sparse_strategy(sparse_strategy_name)

    # dense.embed() y sparse.embed() son independientes entre sí (ambas solo
    # necesitan el texto de la consulta) — antes se esperaban una después de
    # la otra, sumando sus tiempos en vez de solaparlos.
    t0 = time.perf_counter()
    (dense_vector,), (sparse_vector,) = await asyncio.gather(
        dense.embed([query], is_query=True),
        sparse.embed([query]),
    )
    if timings is not None:
        timings["embed_query_ms"] = (time.perf_counter() - t0) * 1000

    admin = get_qdrant_admin()
    # admin.hybrid_search() es una llamada síncrona (qdrant_client con HTTP
    # sync) — sin run_in_executor bloquearía el event loop entero mientras
    # dura la consulta a Qdrant, serializando a TODOS los usuarios
    # concurrentes (de este asistente y de cualquier otro) hasta que
    # termine. Este era el único punto del pipeline de lectura sin
    # offload; embed()/rerank() ya lo hacían.
    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    hits = await loop.run_in_executor(
        None,
        lambda: admin.hybrid_search(
            collection_name=collection_name,
            dense_query=dense_vector,
            sparse_indices=sparse_vector["indices"],
            sparse_values=sparse_vector["values"],
            limit=top_k or settings.RERANK_TOP_K,
        ),
    )
    if timings is not None:
        timings["qdrant_search_ms"] = (time.perf_counter() - t0) * 1000

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
    timings: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """hybrid_retrieve() -> reranker.rerank() sobre todo el pool recuperado
    -> apply_source_boost() (prioriza curso_propio) -> top_n final."""
    candidates = await hybrid_retrieve(
        collection_name=collection_name,
        query=query,
        dense_strategy_name=dense_strategy_name,
        sparse_strategy_name=sparse_strategy_name,
        top_k=top_k,
        timings=timings,
    )
    if not candidates:
        return []

    n = top_n or settings.RERANK_TOP_N
    reranker = strategy_registry.get_rerank_strategy(rerank_strategy_name)
    t0 = time.perf_counter()
    reranked = await reranker.rerank(query, candidates, top_n=len(candidates))
    if timings is not None:
        timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
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
        if payload.get("section_name"):
            citation_parts.append(f"semana/sección: {payload['section_name']}")
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
    t_total = time.perf_counter()
    timings: Dict[str, float] = {}
    candidates = await retrieve_rerank_boost(
        collection_name=collection_name,
        query=query,
        dense_strategy_name=dense_strategy_name,
        sparse_strategy_name=sparse_strategy_name,
        rerank_strategy_name=rerank_strategy_name,
        top_n=top_n,
        timings=timings,
    )
    # Se filtra ACÁ (antes de armar el prompt), no solo al construir
    # `sources` más abajo — si no, el LLM seguía viendo en su contexto
    # candidatos irrelevantes aunque la UI ya no los mostrara como fuentes,
    # y a veces respondía con eso en vez de admitir que no tiene material
    # para esa pregunta. Con la lista vacía, build_generation_prompt() cae
    # en "(sin fragmentos recuperados)" y el system prompt le indica qué
    # hacer en ese caso.
    candidates = [c for c in candidates if c.get("rerank_score", 1.0) >= MIN_SOURCE_RELEVANCE]

    user_prompt = build_generation_prompt(query, candidates)
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    if extra_system_prompt:
        system_prompt = f"{SYSTEM_PROMPT_TEMPLATE}\n\nContexto adicional de este asistente:\n{extra_system_prompt}"
    generator = strategy_registry.get_generation_strategy(generation_strategy_name)
    t0 = time.perf_counter()
    answer, tokens_consumidos = await generator.generate(system_prompt, user_prompt, history=history)
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = (time.perf_counter() - t_total) * 1000
    print(
        "⏱️  answer_query: "
        + ", ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
    )

    # `candidates` ya viene filtrado por relevancia (arriba) — refleja
    # exactamente lo que el LLM tuvo disponible para responder.
    sources = [
        {
            "filename": c["payload"].get("filename"),
            "source_type": c["payload"].get("source_type"),
            "section_name": c["payload"].get("section_name"),
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
        "tokens_consumidos": tokens_consumidos,
        "config_used": {
            "dense": strategy_registry.resolve_strategy_name("dense", dense_strategy_name),
            "sparse": strategy_registry.resolve_strategy_name("sparse", sparse_strategy_name),
            "rerank": strategy_registry.resolve_strategy_name("rerank", rerank_strategy_name),
            "generation": strategy_registry.resolve_strategy_name(
                "generation", generation_strategy_name
            ),
            "timings_ms": {k: round(v) for k, v in timings.items()},
        },
    }
