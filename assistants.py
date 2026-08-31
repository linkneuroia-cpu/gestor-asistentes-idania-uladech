"""
assistants.py
==============
CRUD de asistentes (chatbots públicos por RD) en Postgres, generación del
token/URL pública, y helpers de sesión/mensajes usados por la ruta pública
del asistente (ver app.py `/asistente/{token}`).
"""
import secrets
from typing import Any, Dict, List, Optional

import db
from settings import settings

_UNSET = object()  # distingue "no viene en el request" de "viene como null" (limpiar a config global)


def _public_url(token: str) -> str:
    return f"{settings.PUBLIC_BASE_URL}/asistente/{token}"


def _with_url(row: Dict[str, Any]) -> Dict[str, Any]:
    row["public_url"] = _public_url(row["token"])
    return row


def list_asistentes() -> List[Dict[str, Any]]:
    rows = db.fetch_all(
        """
        SELECT a.*, r.nombre AS rd_nombre
        FROM asistentes a
        JOIN rds r ON r.id = a.rd_id
        ORDER BY a.id
        """
    )
    return [_with_url(r) for r in rows]


def get_asistente(asistente_id: int) -> Optional[Dict[str, Any]]:
    row = db.fetch_one(
        "SELECT a.*, r.nombre AS rd_nombre FROM asistentes a JOIN rds r ON r.id = a.rd_id WHERE a.id = %s",
        (asistente_id,),
    )
    return _with_url(row) if row else None


def get_asistente_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Solo asistentes activos — usado por la ruta pública. `rd_id` ya
    viene incluido (columna propia de asistentes, sin necesidad de join)."""
    return db.fetch_one(
        "SELECT * FROM asistentes WHERE token = %s AND activo = TRUE",
        (token,),
    )


def create_asistente(
    nombre: str,
    rd_id: int,
    prompt_maestro: Optional[str] = None,
    created_by: Optional[int] = None,
    dense_strategy: Optional[str] = None,
    sparse_strategy: Optional[str] = None,
    rerank_strategy: Optional[str] = None,
    generation_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    if not db.fetch_one("SELECT id FROM rds WHERE id = %s", (rd_id,)):
        raise ValueError(f"La RD {rd_id} no existe")
    token = secrets.token_urlsafe(24)
    row = db.execute_returning(
        "INSERT INTO asistentes "
        "(nombre, rd_id, prompt_maestro, token, created_by, dense_strategy, sparse_strategy, rerank_strategy, generation_strategy) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
        (nombre, rd_id, prompt_maestro, token, created_by, dense_strategy, sparse_strategy, rerank_strategy, generation_strategy),
    )
    return _with_url(row)


def update_asistente(
    asistente_id: int,
    nombre: Optional[str] = None,
    rd_id: Optional[int] = None,
    prompt_maestro: Optional[str] = None,
    activo: Optional[bool] = None,
    dense_strategy: Any = _UNSET,
    sparse_strategy: Any = _UNSET,
    rerank_strategy: Any = _UNSET,
    generation_strategy: Any = _UNSET,
) -> Dict[str, Any]:
    """Los 4 campos de estrategia distinguen "no venía en el request"
    (`_UNSET`, default — deja el valor actual intacto) de "venía como
    `null`" (limpia esa etapa a "usar configuración global") — necesario
    para que el formulario de edición pueda volver una etapa a global sin
    tocar las demás. nombre/rd_id/prompt_maestro/activo mantienen el
    comportamiento previo (solo cambian si vienen informados)."""
    existing = db.fetch_one("SELECT * FROM asistentes WHERE id = %s", (asistente_id,))
    if not existing:
        raise ValueError(f"El asistente {asistente_id} no existe")
    if rd_id is not None and not db.fetch_one("SELECT id FROM rds WHERE id = %s", (rd_id,)):
        raise ValueError(f"La RD {rd_id} no existe")

    row = db.execute_returning(
        "UPDATE asistentes SET nombre=%s, rd_id=%s, prompt_maestro=%s, activo=%s, "
        "dense_strategy=%s, sparse_strategy=%s, rerank_strategy=%s, generation_strategy=%s, "
        "updated_at=now() WHERE id=%s RETURNING *",
        (
            nombre if nombre is not None else existing["nombre"],
            rd_id if rd_id is not None else existing["rd_id"],
            prompt_maestro if prompt_maestro is not None else existing["prompt_maestro"],
            activo if activo is not None else existing["activo"],
            existing["dense_strategy"] if dense_strategy is _UNSET else dense_strategy,
            existing["sparse_strategy"] if sparse_strategy is _UNSET else sparse_strategy,
            existing["rerank_strategy"] if rerank_strategy is _UNSET else rerank_strategy,
            existing["generation_strategy"] if generation_strategy is _UNSET else generation_strategy,
            asistente_id,
        ),
    )
    return _with_url(row)


def delete_asistente(asistente_id: int) -> None:
    if not db.fetch_one("SELECT id FROM asistentes WHERE id = %s", (asistente_id,)):
        raise ValueError(f"El asistente {asistente_id} no existe")
    db.execute("DELETE FROM asistentes WHERE id = %s", (asistente_id,))  # CASCADE: sesiones + mensajes


# ── Sesiones y mensajes (usado por la ruta pública) ─────────────────────

def get_or_create_sesion(
    asistente_id: int,
    moodle_courseid: int,
    moodle_userid: int,
    moodle_username: Optional[str],
    moodle_fullname: Optional[str],
    qdrant_collection_name: str,
) -> Dict[str, Any]:
    """Reusa la sesión existente para (asistente, curso, usuario) si hay una,
    si no crea una nueva. Así el historial se acumula por usuario/curso."""
    existing = db.fetch_one(
        "SELECT * FROM asistente_sesiones WHERE asistente_id=%s AND moodle_courseid=%s AND moodle_userid=%s "
        "ORDER BY started_at DESC LIMIT 1",
        (asistente_id, moodle_courseid, moodle_userid),
    )
    if existing:
        db.execute("UPDATE asistente_sesiones SET last_activity_at = now() WHERE id = %s", (existing["id"],))
        return existing

    return db.execute_returning(
        "INSERT INTO asistente_sesiones "
        "(asistente_id, moodle_courseid, moodle_userid, moodle_username, moodle_fullname, qdrant_collection_name) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
        (asistente_id, moodle_courseid, moodle_userid, moodle_username, moodle_fullname, qdrant_collection_name),
    )


def get_sesion(sesion_id: int) -> Optional[Dict[str, Any]]:
    return db.fetch_one("SELECT * FROM asistente_sesiones WHERE id = %s", (sesion_id,))


def get_mensajes(sesion_id: int) -> List[Dict[str, Any]]:
    return db.fetch_all(
        "SELECT * FROM asistente_mensajes WHERE sesion_id = %s ORDER BY created_at", (sesion_id,)
    )


def save_mensaje(
    sesion_id: int,
    rol: str,
    contenido: str,
    fuentes: Optional[list] = None,
    config_usado: Optional[dict] = None,
) -> Dict[str, Any]:
    from psycopg2.extras import Json

    return db.execute_returning(
        "INSERT INTO asistente_mensajes (sesion_id, rol, contenido, fuentes_json, config_usado_json) "
        "VALUES (%s, %s, %s, %s, %s) RETURNING *",
        (
            sesion_id,
            rol,
            contenido,
            Json(fuentes) if fuentes is not None else None,
            Json(config_usado) if config_usado is not None else None,
        ),
    )
