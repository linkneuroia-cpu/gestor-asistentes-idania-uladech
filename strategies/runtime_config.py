"""Override en memoria de la estrategia activa por etapa.

Sembrado desde `.env` (a través de settings.py) cuando no hay override;
sobrescrito vía POST /api/config/{stage}. Se persiste en un archivo JSON
junto al proyecto para que la selección sobreviva un reinicio del
servidor (mismo espíritu que la persistencia de credenciales en .env) —
si el archivo no existe o está corrupto, simplemente se arranca sin
overrides (usa los defaults de .env), sin romper el arranque.
"""
import json
from pathlib import Path
from typing import Dict, Optional

_STATE_PATH = Path(__file__).parent.parent / "runtime_config.json"


class RuntimeConfig:
    def __init__(self):
        self._overrides: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        try:
            _STATE_PATH.write_text(json.dumps(self._overrides, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            print(f"⚠️ No se pudo persistir runtime_config.json: {e}")

    def get(self, stage: str) -> Optional[str]:
        return self._overrides.get(stage)

    def set(self, stage: str, strategy_name: str) -> None:
        self._overrides[stage] = strategy_name
        self._save()

    def clear(self, stage: str) -> None:
        self._overrides.pop(stage, None)
        self._save()

    def get_all(self) -> Dict[str, str]:
        return dict(self._overrides)


_runtime_config = RuntimeConfig()


def get_runtime_config() -> RuntimeConfig:
    return _runtime_config
