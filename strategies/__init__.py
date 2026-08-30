"""
Capa de Strategy Pattern del pipeline RAG configurable.

Cada etapa del pipeline (extracción de documentos, transcripción de
audio/video, enriquecimiento contextual, embedding denso, embedding
disperso, reranking, generación) tiene una interfaz abstracta en `base.py`
y una o más implementaciones concretas por proveedor en su propio módulo.
`registry.py` resuelve qué implementación usar en cada llamada:
override en memoria (`runtime_config`) → default de `.env` → error si el
nombre no existe en el catálogo de `settings.py`.
"""
