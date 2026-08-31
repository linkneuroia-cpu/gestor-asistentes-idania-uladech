"""Override en memoria de la estrategia activa por etapa.

Sembrado desde `.env` (a través de settings.py) cuando no hay override;
sobrescrito vía POST /api/config/{stage}. Se persiste en la tabla
`configuracion` de Postgres (clave con prefijo "stage:") — mismo lugar
donde vive el resto de la configuración del gestor (ver credentials.py).
Carga perezosa (no al importar el módulo, sino en el primer get/set) para
no romper el arranque si Postgres aún no está disponible en ese instante;
ante cualquier error de conexión se degrada a "sin overrides" (usa los
defaults de .env) en vez de tumbar el proceso.
"""
from typing import Dict, Optional

_KEY_PREFIX = "stage:"


class RuntimeConfig:
    def __init__(self):
        self._overrides: Optional[Dict[str, str]] = None

    def _ensure_loaded(self) -> Dict[str, str]:
        if self._overrides is not None:
            return self._overrides
        try:
            import db

            rows = db.fetch_all(
                "SELECT clave, valor FROM configuracion WHERE clave LIKE %s",
                (f"{_KEY_PREFIX}%",),
            )
            self._overrides = {r["clave"][len(_KEY_PREFIX):]: r["valor"] for r in rows}
        except Exception as e:
            print(f"⚠️ No se pudo cargar la configuración de estrategias desde Postgres: {e}")
            self._overrides = {}
        return self._overrides

    def get(self, stage: str) -> Optional[str]:
        return self._ensure_loaded().get(stage)

    def set(self, stage: str, strategy_name: str) -> None:
        self._ensure_loaded()[stage] = strategy_name
        try:
            import db

            db.execute(
                "INSERT INTO configuracion (clave, valor, es_secreto) VALUES (%s, %s, FALSE) "
                "ON CONFLICT (clave) DO UPDATE SET valor = EXCLUDED.valor, updated_at = now()",
                (f"{_KEY_PREFIX}{stage}", strategy_name),
            )
        except Exception as e:
            print(f"⚠️ No se pudo guardar la estrategia de '{stage}' en Postgres: {e}")

    def clear(self, stage: str) -> None:
        self._ensure_loaded().pop(stage, None)
        try:
            import db

            db.execute("DELETE FROM configuracion WHERE clave = %s", (f"{_KEY_PREFIX}{stage}",))
        except Exception as e:
            print(f"⚠️ No se pudo limpiar la estrategia de '{stage}' en Postgres: {e}")

    def get_all(self) -> Dict[str, str]:
        return dict(self._ensure_loaded())


_runtime_config = RuntimeConfig()


def get_runtime_config() -> RuntimeConfig:
    return _runtime_config
