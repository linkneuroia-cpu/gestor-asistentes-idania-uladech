"""
credentials.py
===============
Gestión de credenciales (API keys) del pipeline RAG, configurables desde
la sección "Claves de API" de la interfaz: registro de campos, prueba de
conexión real contra cada proveedor, y persistencia tanto en `settings`
(memoria, efecto inmediato) como en Postgres (tabla `configuracion` —
"todas las configuraciones del gestor" centralizadas ahí, no en .env).
"""
import asyncio
from typing import Any, Dict, List, Optional

from settings import settings

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


def _save_to_db(field: str, value: str, es_secreto: bool) -> None:
    import db

    db.execute(
        "INSERT INTO configuracion (clave, valor, es_secreto) VALUES (%s, %s, %s) "
        "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, es_secreto = EXCLUDED.es_secreto, updated_at = now()",
        (field, value, es_secreto),
    )


def set_credential(field: str, value: str) -> Dict[str, Any]:
    if field not in CREDENTIAL_FIELDS:
        raise ValueError(
            f"Campo de credencial '{field}' desconocido. Válidos: {list(CREDENTIAL_FIELDS)}"
        )

    meta = CREDENTIAL_FIELDS[field]
    setattr(settings, field, value)
    _save_to_db(field, value, meta["secret"])

    return {
        "field": field,
        "configured": bool(value),
        "masked_value": _mask(value) if meta["secret"] else value,
    }


def load_all_into_settings() -> None:
    """Hydrata `settings` en memoria con las credenciales ya guardadas en
    Postgres — se llama una vez al arrancar (lifespan de app.py), porque
    `settings` es una instancia nueva en cada arranque (solo lee .env por
    defecto) y las credenciales ahora viven en la tabla `configuracion`,
    no en .env."""
    import db

    try:
        rows = db.fetch_all(
            "SELECT clave, valor FROM configuracion WHERE clave = ANY(%s)",
            (list(CREDENTIAL_FIELDS.keys()),),
        )
    except Exception as e:
        print(f"⚠️ No se pudieron cargar credenciales desde Postgres: {e}")
        return

    for row in rows:
        if row["valor"] is not None:
            setattr(settings, row["clave"], row["valor"])
    if rows:
        print(f"🔑 {len(rows)} credencial(es) cargada(s) desde Postgres")


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
