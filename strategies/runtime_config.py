"""Override en memoria de la estrategia activa por etapa.

Sembrado implícitamente desde `.env` (a través de settings.py) al no haber
override; sobrescrito vía POST /api/config/{stage}. Se resetea al reiniciar
el servidor — no hay persistencia en disco (limitación aceptada, ver plan).
Mismo estilo que `core.job_store`: un singleton mutable en memoria.
"""
from typing import Dict, Optional


class RuntimeConfig:
    def __init__(self):
        self._overrides: Dict[str, str] = {}

    def get(self, stage: str) -> Optional[str]:
        return self._overrides.get(stage)

    def set(self, stage: str, strategy_name: str) -> None:
        self._overrides[stage] = strategy_name

    def clear(self, stage: str) -> None:
        self._overrides.pop(stage, None)

    def get_all(self) -> Dict[str, str]:
        return dict(self._overrides)


_runtime_config = RuntimeConfig()


def get_runtime_config() -> RuntimeConfig:
    return _runtime_config
