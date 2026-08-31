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
    """Todas las RD, con conteo de colecciones reales, cursos registrados
    (con o sin colección) y asistentes asignados."""
    return db.fetch_all(
        """
        SELECT r.*,
               COUNT(DISTINCT c.id) FILTER (WHERE c.qdrant_collection_name IS NOT NULL) AS total_colecciones,
               COUNT(DISTINCT c.id) AS total_courseids,
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
    """Crea la RD y registra su courseid inicial en la lista de cursos
    (colecciones_rd, todavía sin colección) — igual que el seed de las 14
    RD originales."""
    rd = db.execute_returning(
        "INSERT INTO rds (nombre, moodle_courseid, moodle_course_url) VALUES (%s, %s, %s) RETURNING *",
        (nombre, moodle_courseid, moodle_course_url),
    )
    db.execute(
        "INSERT INTO colecciones_rd (rd_id, moodle_courseid) VALUES (%s, %s)",
        (rd["id"], moodle_courseid),
    )
    return rd


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
    """Lanza ValueError si la RD no existe, o si tiene colecciones reales o
    asistentes asignados (FK ON DELETE RESTRICT) — hay que reasignarlos o
    eliminarlos primero. Los courseid sin colección (solo registrados en la
    lista) se limpian automáticamente, ya que no representan nada que perder."""
    if not get_rd(rd_id):
        raise ValueError(f"La RD {rd_id} no existe")
    import psycopg2

    db.execute(
        "DELETE FROM colecciones_rd WHERE rd_id = %s AND qdrant_collection_name IS NULL", (rd_id,)
    )
    try:
        db.execute("DELETE FROM rds WHERE id = %s", (rd_id,))
    except psycopg2.errors.ForeignKeyViolation:
        raise ValueError(
            "No se puede eliminar: esta RD tiene colecciones o asistentes asignados. "
            "Reasígnalos o elimínalos primero."
        )


def link_collection(qdrant_collection_name: str, rd_id: int, moodle_courseid: int) -> Dict[str, Any]:
    """Registra en Postgres que una colección de Qdrant pertenece a una RD
    + curso específico. Llamado desde POST /api/collections (creación).
    Si ese (rd_id, courseid) ya estaba registrado sin colección (vía
    add_courseid), "llena" esa fila en vez de crear una duplicada."""
    if not get_rd(rd_id):
        raise ValueError(f"La RD {rd_id} no existe")
    import psycopg2

    try:
        return db.execute_returning(
            "INSERT INTO colecciones_rd (qdrant_collection_name, rd_id, moodle_courseid) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (rd_id, moodle_courseid) DO UPDATE SET qdrant_collection_name = EXCLUDED.qdrant_collection_name "
            "RETURNING *",
            (qdrant_collection_name, rd_id, moodle_courseid),
        )
    except psycopg2.errors.UniqueViolation:
        raise ValueError(
            f"El curso {moodle_courseid} de la RD {rd_id} ya tiene asignada otra colección."
        )


def reassign_collection(qdrant_collection_name: str, rd_id: int, moodle_courseid: int) -> Dict[str, Any]:
    """Cambia el RD/courseid de una colección ya vinculada. Usado por el
    listado de Colecciones para reasignar."""
    if not get_rd(rd_id):
        raise ValueError(f"La RD {rd_id} no existe")
    if not db.fetch_one(
        "SELECT 1 FROM colecciones_rd WHERE qdrant_collection_name = %s", (qdrant_collection_name,)
    ):
        raise ValueError(f"La colección '{qdrant_collection_name}' no tiene un vínculo RD que reasignar")
    import psycopg2

    try:
        return db.execute_returning(
            "UPDATE colecciones_rd SET rd_id = %s, moodle_courseid = %s "
            "WHERE qdrant_collection_name = %s RETURNING *",
            (rd_id, moodle_courseid, qdrant_collection_name),
        )
    except psycopg2.errors.UniqueViolation:
        raise ValueError(f"El curso {moodle_courseid} de la RD {rd_id} ya tiene asignada otra colección.")


def add_courseid(rd_id: int, moodle_courseid: int) -> Dict[str, Any]:
    """Registra un courseid más bajo una RD, todavía sin colección asignada
    — para que "Gestión RD" liste todos los cursos que replican la plantilla
    aunque aún no tengan contenido vectorizado."""
    if not get_rd(rd_id):
        raise ValueError(f"La RD {rd_id} no existe")
    import psycopg2

    try:
        return db.execute_returning(
            "INSERT INTO colecciones_rd (rd_id, moodle_courseid) VALUES (%s, %s) RETURNING *",
            (rd_id, moodle_courseid),
        )
    except psycopg2.errors.UniqueViolation:
        raise ValueError(f"El curso {moodle_courseid} ya está registrado en esta RD.")


def remove_courseid_entry(entry_id: int) -> None:
    """Quita un courseid de la lista de una RD. Solo tiene sentido si esa
    fila todavía no tiene colección asignada (si la tiene, hay que
    reasignar o eliminar la colección primero)."""
    row = db.fetch_one("SELECT * FROM colecciones_rd WHERE id = %s", (entry_id,))
    if not row:
        raise ValueError(f"El registro {entry_id} no existe")
    if row["qdrant_collection_name"]:
        raise ValueError(
            "Este curso ya tiene una colección asignada — reasígnala o elimínala antes de quitarlo."
        )
    db.execute("DELETE FROM colecciones_rd WHERE id = %s", (entry_id,))


def unlink_collection(qdrant_collection_name: str) -> None:
    """Al eliminar una colección de Qdrant, el courseid sigue siendo válido
    para su RD (solo se quedó sin colección) — se limpia el vínculo en vez
    de borrar la fila, para no perder el registro del curso."""
    db.execute(
        "UPDATE colecciones_rd SET qdrant_collection_name = NULL WHERE qdrant_collection_name = %s",
        (qdrant_collection_name,),
    )


def resolve_collection(rd_id: int, moodle_courseid: int) -> Optional[str]:
    """(rd_id, courseid) -> nombre de colección de Qdrant, o None si no hay
    ninguna colección de esa RD asignada a ese curso. Usado por la ruta
    pública del asistente."""
    row = db.fetch_one(
        "SELECT qdrant_collection_name FROM colecciones_rd WHERE rd_id = %s AND moodle_courseid = %s",
        (rd_id, moodle_courseid),
    )
    return row["qdrant_collection_name"] if row else None
