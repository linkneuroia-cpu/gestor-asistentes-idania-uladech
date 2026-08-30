"""Estrategias de generación final (LLM tutor) y el system prompt fijo
pedido: prioriza [MATERIAL DEL CURSO] sobre [BIBLIOGRAFÍA COMPLEMENTARIA]."""
from strategies.base import GenerationStrategy

SYSTEM_PROMPT_TEMPLATE = (
    "Eres un tutor experto. Responde a la consulta utilizando únicamente "
    "los fragmentos recuperados. REGLA DE PRIORIDAD: Tienes dos fuentes "
    "etiquetadas: [MATERIAL DEL CURSO] y [BIBLIOGRAFÍA COMPLEMENTARIA]. "
    "Basa tu respuesta principalmente en el [MATERIAL DEL CURSO]. Usa la "
    "bibliografía solo si el material del curso no cubre la respuesta. Si "
    "hay contradicciones, la verdad absoluta es el [MATERIAL DEL CURSO]. "
    "Si la respuesta no está, indícalo."
)


class _OpenAICompatibleGenerationStrategy(GenerationStrategy):
    _model = "gpt-4o-mini"
    _base_url = None
    _api_key_field = "OPENAI_API_KEY"
    _provider_label = "generation"

    def __init__(self):
        from settings import settings

        api_key = getattr(settings, self._api_key_field, "")
        if not api_key:
            raise ValueError(
                f"{self._api_key_field} no está configurado. Requerido "
                f"para la estrategia de generación '{self._provider_label}'."
            )
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import asyncio

        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        return await asyncio.get_event_loop().run_in_executor(None, _call)


class GPT4oGenerationStrategy(_OpenAICompatibleGenerationStrategy):
    _model = "gpt-4o"
    _base_url = None
    _api_key_field = "OPENAI_API_KEY"
    _provider_label = "gpt4o"


class GPT4oMiniGenerationStrategy(_OpenAICompatibleGenerationStrategy):
    _model = "gpt-4o-mini"
    _base_url = None
    _api_key_field = "OPENAI_API_KEY"
    _provider_label = "gpt4o_mini"


class DeepSeekGenerationStrategy(_OpenAICompatibleGenerationStrategy):
    _model = "deepseek-chat"
    _api_key_field = "DEEPSEEK_API_KEY"
    _provider_label = "deepseek"

    def __init__(self):
        from settings import settings

        self._base_url = settings.DEEPSEEK_BASE_URL
        super().__init__()


class GeminiGenerationStrategy(GenerationStrategy):
    _model = "gemini-1.5-flash"

    def __init__(self):
        from settings import settings

        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY no está configurado. Requerido para la "
                "estrategia de generación 'gemini'."
            )
        from google import genai

        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import asyncio
        from google.genai import types

        def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=[user_prompt],
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            return response.text or ""

        return await asyncio.get_event_loop().run_in_executor(None, _call)
