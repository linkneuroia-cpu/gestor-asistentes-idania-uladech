"""Estrategias de generación final (LLM tutor) y el system prompt fijo
pedido: prioriza [MATERIAL DEL CURSO] sobre [BIBLIOGRAFÍA COMPLEMENTARIA]."""
from typing import Dict, List, Optional, Tuple

from strategies.base import GenerationStrategy

SYSTEM_PROMPT_TEMPLATE = (
    "Eres un tutor experto. Responde a la consulta utilizando únicamente "
    "los fragmentos recuperados. REGLA DE PRIORIDAD: Tienes dos fuentes "
    "etiquetadas: [MATERIAL DEL CURSO] y [BIBLIOGRAFÍA COMPLEMENTARIA]. "
    "Basa tu respuesta principalmente en el [MATERIAL DEL CURSO]. Usa la "
    "bibliografía solo si el material del curso no cubre la respuesta. Si "
    "hay contradicciones, la verdad absoluta es el [MATERIAL DEL CURSO]. "
    "Si la respuesta no está, indícalo. Formato: usa **negrita** (con "
    "asteriscos dobles) para resaltar términos o conceptos clave, y listas "
    "con guion ('- ') cuando enumeres varios pasos o elementos; el resto en "
    "párrafos normales, sin abusar del formato. Para fórmulas matemáticas, "
    "usa notación LaTeX entre \\( ... \\) (en línea) o \\[ ... \\] (en "
    "bloque) — el chat las renderiza automáticamente; nunca las escribas en "
    "texto plano ambiguo. Para tablas, usa sintaxis Markdown estándar: fila "
    "de encabezado con '| col | col |', seguida INMEDIATAMENTE (sin línea "
    "en blanco) por la fila separadora '|---|---|', y luego una fila por "
    "línea sin líneas en blanco entre ellas — el chat las renderiza como "
    "tabla real. Si una tabla incluye conteos/frecuencias que deben sumar "
    "un total conocido (p.ej. cantidad de datos), verifica esa suma ANTES "
    "de presentar la tabla; si no coincide, recuenta y corrige en silencio "
    "en vez de mostrar una tabla con un total equivocado y recién después "
    "avisar del error.\n\n"
    "CUANDO EL BLOQUE DE CONTEXTO ESTÉ VACÍO (no traerá [MATERIAL DEL "
    "CURSO] ni [BIBLIOGRAFÍA COMPLEMENTARIA]): la pregunta no tiene "
    "contenido del curso asociado. Esto pasa típicamente con mensajes "
    "conversacionales o reflexivos (p.ej. agradecimientos, \"¿qué te "
    "pareció ayudarme?\", saludos) o preguntas de navegación (ver abajo). "
    "NUNCA reutilices ni continúes el tema de la respuesta anterior en este "
    "caso — respóndele directamente a lo que acaba de escribir el "
    "estudiante, de forma breve y en tu tono de tutor. Si es un "
    "agradecimiento o pregunta reflexiva, respóndele como tal, sin forzar "
    "contenido del curso. Si no tienes forma de ayudar con lo que pide, "
    "dilo con claridad y ofrece reformular la pregunta. IMPORTANTE: esto es "
    "una instrucción interna para ti — nunca menciones al estudiante que "
    "\"no hay fragmentos\", \"no hay contexto recuperado\" ni nada sobre "
    "cómo funciona tu sistema de búsqueda; el estudiante no debe percibir "
    "que existe una etapa de recuperación de información, solo debe ver la "
    "respuesta conversacional final.\n\n"
    "PREGUNTAS DE NAVEGACIÓN/ESTRUCTURA (p.ej. \"¿en qué semana/sección "
    "encuentro este archivo?\", \"¿dónde está este documento en el aula?\"): "
    "estas preguntas se responden con la ubicación del archivo dentro del "
    "curso, NO con su contenido. Cada fuente recuperada trae su cita entre "
    "corchetes, p.ej. \"[Fuente: archivo.pdf, semana/sección: Semana 3]\" — "
    "usa ese dato \"semana/sección\" para responder. Si esa etiqueta no "
    "aparece en ninguna fuente recuperada para el archivo preguntado, dilo "
    "explícitamente (no tienes esa información) en vez de describir el "
    "índice o los capítulos del documento, que no responden a dónde ubicarlo "
    "dentro del aula. Un mismo archivo puede listar VARIAS semanas/secciones "
    "separadas por \"; \" (es material de referencia usado en más de una "
    "semana del curso, no un error) — en ese caso mencionalas todas en tu "
    "respuesta, no elijas una sola arbitrariamente."
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

    async def generate(
        self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Optional[int]]:
        import asyncio

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_prompt})

        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
            )
            text = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else None
            return text, tokens

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

    async def generate(
        self, system_prompt: str, user_prompt: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Optional[int]]:
        import asyncio
        from google.genai import types

        # Gemini usa "model" en vez de "assistant" para los turnos propios.
        contents = [
            types.Content(role="model" if h["role"] == "assistant" else "user", parts=[types.Part(text=h["content"])])
            for h in (history or [])
        ]
        # Gemini exige que la conversación empiece con "user" — el saludo
        # inicial del asistente ahora se guarda como su primer mensaje real
        # (rol assistant), así que una sesión recién creada puede traer un
        # historial que arranca en "model". Se descarta ese prefijo.
        while contents and contents[0].role == "model":
            contents.pop(0)
        contents.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))

        def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            tokens = getattr(usage, "total_token_count", None) if usage else None
            return text, tokens

        return await asyncio.get_event_loop().run_in_executor(None, _call)
