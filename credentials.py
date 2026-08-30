"""
credentials.py
===============
Gestión de credenciales (API keys) del pipeline RAG, configurables desde
la sección "Claves de API" de la interfaz: registro de campos, prueba de
conexión real contra cada proveedor, y persistencia tanto en `settings`
(memoria, efecto inmediato) como en `.env` (disco, sobrevive un reinicio).
"""
import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from settings import settings

ENV_PATH = Path(__file__).parent / ".env"

CREDENTIAL_FIELDS: Dict[str, Dict[str, Any]] = {
    "OPENAI_API_KEY": {"label": "OpenAI API Key", "provider": "openai", "secret": True},
    "GOOGLE_API_KEY": {"label": "Google (Gemini) API Key", "provider": "google", "secret": True},
    "DEEPSEEK_API_KEY": {"label": "DeepSeek API Key", "provider": "deepseek", "secret": True},
    "DEEPSEEK_BASE_URL": {"label": "DeepSeek — Base URL", "provider": "deepseek", "secret": False},
    "COHERE_API_KEY": {"label": "Cohere API Key", "provider": "cohere", "secret": True},
    "DEEPGRAM_API_KEY": {"label": "Deepgram API Key", "provider": "deepgram", "secret": True},
    "AZURE_EMBEDDING_KEY": {"label": "Azure OpenAI Embeddings — Key", "provider": "azure", "secret": True},
    "AZURE_EMBEDDING_ENDPOINT": {"label": "Azure OpenAI Embeddings — Endpoint", "provider": "azure", "secret": False},
}

# Proveedor -> campo de API key que su prueba de conexión requiere (para que
# la UI sepa qué credencial probar dado un provider de un catálogo de
# estrategias, p.ej. "openai" -> "OPENAI_API_KEY").
PROVIDER_KEY_FIELD: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "azure": "AZURE_EMBEDDING_KEY",
}


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "••••"
    return "••••" + value[-4:]


def list_credentials() -> List[Dict[str, Any]]:
    result = []
    for field, meta in CREDENTIAL_FIELDS.items():
        value = getattr(settings, field, "") or ""
        result.append({
            "field": field,
            "label": meta["label"],
            "provider": meta["provider"],
            "secret": meta["secret"],
            "configured": bool(value),
            "masked_value": _mask(value) if meta["secret"] else value,
        })
    return result


def _update_env_file(key: str, value: str) -> None:
    """Actualiza o agrega la línea KEY=value en .env, preservando el resto
    del archivo (comentarios, otras variables, orden)."""
    if not ENV_PATH.exists():
        ENV_PATH.write_text(f"{key}={value}\n", encoding="utf-8")
        return

    text = ENV_PATH.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(f"{key}={value}", text, count=1)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += f"{key}={value}\n"
    ENV_PATH.write_text(text, encoding="utf-8")


def set_credential(field: str, value: str) -> Dict[str, Any]:
    if field not in CREDENTIAL_FIELDS:
        raise ValueError(
            f"Campo de credencial '{field}' desconocido. Válidos: {list(CREDENTIAL_FIELDS)}"
        )

    setattr(settings, field, value)
    _update_env_file(field, value)

    meta = CREDENTIAL_FIELDS[field]
    return {
        "field": field,
        "configured": bool(value),
        "masked_value": _mask(value) if meta["secret"] else value,
    }


# ============================================================================
# PRUEBA DE CONEXIÓN REAL POR PROVEEDOR
# ============================================================================

def _test_openai(api_key: str) -> None:
    from openai import OpenAI

    OpenAI(api_key=api_key).models.list()


def _test_google(api_key: str) -> None:
    from google import genai

    for _ in genai.Client(api_key=api_key).models.list():
        break


def _test_deepseek(api_key: str) -> None:
    from openai import OpenAI

    OpenAI(api_key=api_key, base_url=settings.DEEPSEEK_BASE_URL).models.list()


def _test_cohere(api_key: str) -> None:
    import cohere

    cohere.ClientV2(api_key=api_key).models.list(page_size=1)


def _test_deepgram(api_key: str) -> None:
    from deepgram import DeepgramClient

    DeepgramClient(api_key=api_key).manage.v1.projects.list()


def _test_azure(key: str, endpoint: str) -> None:
    from openai import AzureOpenAI

    if not endpoint:
        raise ValueError(
            "Falta AZURE_EMBEDDING_ENDPOINT — configúralo también antes de probar la conexión."
        )
    client = AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=settings.AZURE_EMBEDDING_API_VERSION,
    )
    client.embeddings.create(model=settings.AZURE_EMBEDDING_DEPLOYMENT, input=["conexión de prueba"])


_PROVIDER_TESTS = {
    "openai": lambda: _test_openai(settings.OPENAI_API_KEY),
    "google": lambda: _test_google(settings.GOOGLE_API_KEY),
    "deepseek": lambda: _test_deepseek(settings.DEEPSEEK_API_KEY),
    "cohere": lambda: _test_cohere(settings.COHERE_API_KEY),
    "deepgram": lambda: _test_deepgram(settings.DEEPGRAM_API_KEY),
    "azure": lambda: _test_azure(settings.AZURE_EMBEDDING_KEY, settings.AZURE_EMBEDDING_ENDPOINT),
}


async def test_credential(field: str, value: Optional[str] = None) -> Dict[str, Any]:
    """Prueba la conexión real del proveedor asociado a `field`. Si se pasa
    `value`, se prueba ese valor temporalmente (sin persistirlo); si no, se
    usa el valor ya guardado en settings."""
    if field not in CREDENTIAL_FIELDS:
        raise ValueError(f"Campo de credencial '{field}' desconocido.")

    provider = CREDENTIAL_FIELDS[field]["provider"]
    test_fn = _PROVIDER_TESTS.get(provider)
    if test_fn is None:
        raise ValueError(f"No hay prueba de conexión implementada para el proveedor '{provider}'.")

    original = getattr(settings, field, "")
    if value is not None:
        setattr(settings, field, value)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, test_fn)
        return {"success": True, "message": "Conexión exitosa."}
    except Exception as e:
        return {"success": False, "message": str(e)[:400]}
    finally:
        if value is not None:
            setattr(settings, field, original)
