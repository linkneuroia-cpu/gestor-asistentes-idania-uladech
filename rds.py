"""
rds.py
======
CRUD de RD (aulas) en Postgres, y su relación con colecciones de Qdrant
(colecciones_rd) y asistentes. Cada RD es un registro propio (14 semillas
+ las que se agreguen), no una categoría que agrupa cursos.
"""
from typing import Any, Dict, List, Optional

import db


def list_rds() -> List[Dict[str, Any]]:
    """Todas las RD, con conteo de colecciones y asistentes asignados."""
    return db.fetch_all(
        """
        SELECT r.*,
               COUNT(DISTINCT c.id) AS total_colecciones,
               COUNT(DISTINCT a.id) AS total_asistentes
        FROM rds r
        LEFT JOIN colecciones_rd c ON c.rd_id = r.id
        LEFT JOIN asistentes a ON a.rd_id = r.id
        GROUP BY r.id
        ORDER BY r.id
        """
    )


def get_rd(rd_id: int) -> Optional[Dict[str, Any]]:
    return db.fetch_one("SELECT * FROM rds WHERE id = %s", (rd_id,))


def get_rd_detail(rd_id: int) -> Optional[Dict[str, Any]]:
    rd = get_rd(rd_id)
    if not rd:
        return None
    rd["colecciones"] = db.fetch_all(
        "SELECT * FROM colecciones_rd WHERE rd_id = %s ORDER BY id", (rd_id,)
    )
    rd["asistentes"] = db.fetch_all(
        "SELECT id, nombre, token, activo, created_at FROM asistentes WHERE rd_id = %s ORDER BY id",
        (rd_id,),
    )
    return rd


def create_rd(nombre: str, moodle_courseid: int, moodle_course_url: Optional[str] = None) -> Dict[str, Any]:
    return db.execute_returning(
        "INSERT INTO rds (nombre, moodle_courseid, moodle_course_url) VALUES (%s, %s, %s) RETURNING *",
        (nombre, moodle_courseid, moodle_course_url),
    )


def update_rd(
    rd_id: int,
    nombre: Optional[str] = None,
    moodle_courseid: Optional[int] = None,
    moodle_course_url: Optional[str] = None,
) -> Dict[str, Any]:
    existing = get_rd(rd_id)
    if not existing:
        raise ValueError(f"La RD {rd_id} no existe")
    return db.execute_returning(
        "UPDATE rds SET nombre=%s, moodle_courseid=%s, moodle_course_url=%s, updated_at=now() "
        "WHERE id=%s RETURNING *",
        (
            nombre if nombre is not None else existing["nombre"],
            moodle_courseid if moodle_courseid is not None else existing["moodle_courseid"],
            moodle_course_url if moodle_course_url is not None else existing["moodle_course_url"],
            rd_id,
        ),
    )


def delete_rd(rd_id: int) -> None:
    """Lanza ValueError si la RD no existe, o si tiene colecciones/
    asistentes asignados (FK ON DELETE RESTRICT) — hay que reasignarlos o
    eliminarlos primero."""
    if not get_rd(rd_id):
        raise ValueError(f"La RD {rd_id} no existe")
    import psycopg2

    try:
        db.execute("DELETE FROM rds WHERE id = %s", (rd_id,))
    except psycopg2.errors.ForeignKeyViolation:
        raise ValueError(
            "No se puede eliminar: esta RD tiene colecciones o asistentes asignados. "
            "Reasígnalos o elimínalos primero."
        )


def link_collection(qdrant_collection_name: str, rd_id: int, moodle_courseid: int) -> Dict[str, Any]:
    """Registra en Postgres que una colección de Qdrant pertenece a una RD
    + curso específico. Llamado desde POST /api/collections."""
    if not get_rd(rd_id):
        raise ValueError(f"La RD {rd_id} no existe")
    return db.execute_returning(
        "INSERT INTO colecciones_rd (qdrant_collection_name, rd_id, moodle_courseid) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (qdrant_collection_name) DO UPDATE SET rd_id = EXCLUDED.rd_id, moodle_courseid = EXCLUDED.moodle_courseid "
        "RETURNING *",
        (qdrant_collection_name, rd_id, moodle_courseid),
    )


def unlink_collection(qdrant_collection_name: str) -> None:
    db.execute("DELETE FROM colecciones_rd WHERE qdrant_collection_name = %s", (qdrant_collection_name,))


def resolve_collection(rd_id: int, moodle_courseid: int) -> Optional[str]:
    """(rd_id, courseid) -> nombre de colección de Qdrant, o None si no hay
    ninguna colección de esa RD asignada a ese curso. Usado por la ruta
    pública del asistente."""
    row = db.fetch_one(
        "SELECT qdrant_collection_name FROM colecciones_rd WHERE rd_id = %s AND moodle_courseid = %s",
        (rd_id, moodle_courseid),
    )
    return row["qdrant_collection_name"] if row else None
