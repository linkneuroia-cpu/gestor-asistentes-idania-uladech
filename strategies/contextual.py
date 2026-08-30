"""Estrategias de Contextual Retrieval: generan una cabecera de 50-100
palabras que sitúa un chunk dentro de su documento completo, antes de
vectorizarlo (mejora la recuperación de chunks ambiguos fuera de contexto).
"""
from strategies.base import ContextualEnrichmentStrategy

# Ventana de caracteres del documento completo enviada como contexto al LLM
# (evita exceder el límite de tokens en documentos muy largos).
_MAX_DOCUMENT_CONTEXT_CHARS = 8000

_PROMPT_TEMPLATE = (
    "Aquí está el documento completo (puede estar truncado):\n"
    "<documento>\n{document}\n</documento>\n\n"
    "Aquí está el fragmento (chunk) que queremos situar dentro del "
    "documento completo:\n"
    "<fragmento>\n{chunk}\n</fragmento>\n\n"
    "Da una cabecera contextual breve y específica (50 a 100 palabras, en "
    "español) que sitúe este fragmento dentro del documento completo, para "
    "mejorar su recuperación en una búsqueda. Responde únicamente con la "
    "cabecera, sin introducciones ni comillas."
)


class NoOpContextualStrategy(ContextualEnrichmentStrategy):
    """Sin enriquecimiento contextual — comportamiento por defecto, sin
    costo adicional de LLM."""

    async def enrich(self, full_document_text: str, chunk_text: str) -> str:
        return ""


class _OpenAICompatibleContextualStrategy(ContextualEnrichmentStrategy):
    _model = "gpt-4o-mini"
    _base_url = None
    _api_key_field = "OPENAI_API_KEY"
    _provider_label = "contextual"

    def __init__(self):
        from settings import settings

        api_key = getattr(settings, self._api_key_field, "")
        if not api_key:
            raise ValueError(
                f"{self._api_key_field} no está configurado. Requerido "
                f"para la estrategia contextual '{self._provider_label}'."
            )
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    async def enrich(self, full_document_text: str, chunk_text: str) -> str:
        import asyncio

        document_excerpt = full_document_text[:_MAX_DOCUMENT_CONTEXT_CHARS]
        prompt = _PROMPT_TEMPLATE.format(document=document_excerpt, chunk=chunk_text)

        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return (response.choices[0].message.content or "").strip()

        return await asyncio.get_event_loop().run_in_executor(None, _call)


class GPT4oMiniContextualStrategy(_OpenAICompatibleContextualStrategy):
    _model = "gpt-4o-mini"
    _base_url = None
    _api_key_field = "OPENAI_API_KEY"
    _provider_label = "gpt4o_mini"


class DeepSeekContextualStrategy(_OpenAICompatibleContextualStrategy):
    _model = "deepseek-chat"
    _api_key_field = "DEEPSEEK_API_KEY"
    _provider_label = "deepseek"

    def __init__(self):
        from settings import settings

        self._base_url = settings.DEEPSEEK_BASE_URL
        super().__init__()
