"""Estrategias de extracción de documentos (PDF/PPT/Word) -> Markdown limpio.

`local` reutiliza el pipeline de extracción ya existente en definitions.py
(PyMuPDF/pdfplumber/python-docx/pptx/EasyOCR) sin modificarlo. Las demás
usan un LLM con visión para producir Markdown con tablas completas y
descripciones detalladas de imágenes/gráficos, según lo configurado.
"""
import base64
from pathlib import Path
from typing import List

from strategies.base import DocumentExtractionStrategy

VISION_PROMPT = (
    "Convierte el contenido de este documento a Markdown limpio y fiel al "
    "original. Reglas: (1) las tablas deben transcribirse completas, sin "
    "cortar filas ni columnas, usando sintaxis de tabla Markdown; (2) cada "
    "imagen, gráfico o diagrama debe describirse en el texto con el mayor "
    "detalle posible (qué muestra, ejes, valores relevantes, relaciones); "
    "(3) conserva la estructura de encabezados y el orden del documento; "
    "(4) no agregues comentarios ni texto que no esté en el original."
)


class LocalExtractionStrategy(DocumentExtractionStrategy):
    """Extracción local existente (PyMuPDF/pdfplumber/python-docx/pptx/
    EasyOCR vía definitions._extract_text). Gratis, rápida, sin API."""

    async def extract(self, file_path: str) -> str:
        import definitions

        return await definitions._extract_text(file_path)


class GeminiVisionStrategy(DocumentExtractionStrategy):
    """Gemini 1.5 Flash — lee el PDF nativamente (sin convertir a imágenes)."""

    def __init__(self):
        from settings import settings

        if not settings.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY no está configurado. Requerido para la "
                "estrategia ETL 'gemini_vision'."
            )
        from google import genai

        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self._model = "gemini-1.5-flash"

    async def extract(self, file_path: str) -> str:
        import asyncio
        from google.genai import types

        pdf_bytes = Path(file_path).read_bytes()

        def _call():
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    VISION_PROMPT,
                ],
            )
            return response.text or ""

        return await asyncio.get_event_loop().run_in_executor(None, _call)


class _PageImageVisionStrategy(DocumentExtractionStrategy):
    """Base común para GPT-4o-mini/DeepSeek: PyMuPDF renderiza cada página
    a imagen PNG y se envía a un LLM con visión, vía cliente OpenAI-
    compatible (openai o DeepSeek según base_url)."""

    _provider_label = "vision"
    _model = "gpt-4o-mini"
    _base_url = None  # None -> OpenAI por defecto
    _api_key_field = "OPENAI_API_KEY"

    def __init__(self):
        from settings import settings

        api_key = getattr(settings, self._api_key_field, "")
        if not api_key:
            raise ValueError(
                f"{self._api_key_field} no está configurado. Requerido para "
                f"la estrategia ETL de visión por página ({self._provider_label})."
            )
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)

    def _render_pages_to_png(self, file_path: str) -> List[bytes]:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        try:
            images = []
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                images.append(pix.tobytes("png"))
            return images
        finally:
            doc.close()

    async def extract(self, file_path: str) -> str:
        import asyncio

        page_images = self._render_pages_to_png(file_path)

        def _call_page(png_bytes: bytes) -> str:
            b64 = base64.b64encode(png_bytes).decode("ascii")
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content or ""

        loop = asyncio.get_event_loop()
        pages_markdown = []
        for png_bytes in page_images:
            page_md = await loop.run_in_executor(None, _call_page, png_bytes)
            pages_markdown.append(page_md)

        return "\n\n".join(pages_markdown)


class GPT4oMiniVisionStrategy(_PageImageVisionStrategy):
    _provider_label = "gpt4o_mini_vision"
    _model = "gpt-4o-mini"
    _base_url = None
    _api_key_field = "OPENAI_API_KEY"


class DeepSeekVisionStrategy(_PageImageVisionStrategy):
    _provider_label = "deepseek_vision"
    # "deepseek-chat" es solo texto y rechaza contenido de imagen. El
    # soporte de visión de DeepSeek vive en un modelo aparte (lanzado
    # 2026-08-21), todavía marcado "-exp" (experimental) — puede cambiar
    # de nombre o quedar deprecado sin mucho aviso.
    _model = "deepseek-v4-flash-vision-exp"
    _api_key_field = "DEEPSEEK_API_KEY"

    def __init__(self):
        from settings import settings

        self._base_url = settings.DEEPSEEK_BASE_URL
        super().__init__()
