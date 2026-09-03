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


def _decorate(row: Dict[str, Any]) -> Dict[str, Any]:
    """URL pública + `tipo` derivado. `tipo` NO es una columna: se deriva
    siempre de rd_id para que no pueda desincronizarse del vínculo real."""
    row["public_url"] = _public_url(row["token"])
    row["tipo"] = "rd" if row.get("rd_id") else "normal"
    return row


# Promedio de calificación (1 útil / -1 no útil) por asistente, agregado vía
# asistente_sesiones — se calcula en el LEFT JOIN del listado para evitar
# N+1 queries. NULL si el asistente todavía no tiene ninguna calificación.
_CALIF_SUBQUERY = """
    (SELECT s.asistente_id,
            COUNT(*) FILTER (WHERE m.calificacion IS NOT NULL) AS calificaciones_total,
            ROUND(AVG(m.calificacion) FILTER (WHERE m.calificacion IS NOT NULL)::numeric, 2) AS calificacion_promedio
     FROM asistente_mensajes m
     JOIN asistente_sesiones s ON s.id = m.sesion_id
     WHERE m.rol = 'assistant'
     GROUP BY s.asistente_id) c
"""


def get_calificaciones_por_curso(asistente_id: int) -> Dict[str, Any]:
    """Desglose de la calificación de un asistente por curso de Moodle:
    misma agregación que _CALIF_SUBQUERY (mensajes 'assistant' calificados,
    vía asistente_sesiones) pero agrupando por s.moodle_courseid en vez de
    por asistente, más el total general. Cada curso puede tener una calidad
    de retrieval distinta aunque el asistente sea el mismo."""
    por_curso = db.fetch_all(
        """
        SELECT s.moodle_courseid,
               COUNT(*)                                     AS calificaciones_total,
               COUNT(*) FILTER (WHERE m.calificacion = 1)    AS positivas,
               COUNT(*) FILTER (WHERE m.calificacion = -1)   AS negativas,
               ROUND(AVG(m.calificacion)::numeric, 2)        AS calificacion_promedio
        FROM asistente_mensajes m
        JOIN asistente_sesiones s ON s.id = m.sesion_id
        WHERE m.rol = 'assistant' AND m.calificacion IS NOT NULL AND s.asistente_id = %s
        GROUP BY s.moodle_courseid
        ORDER BY s.moodle_courseid
        """,
        (asistente_id,),
    )
    total = {
        "calificaciones_total": sum(r["calificaciones_total"] for r in por_curso),
        "positivas": sum(r["positivas"] for r in por_curso),
        "negativas": sum(r["negativas"] for r in por_curso),
    }
    # Promedio global recalculado desde los conteos (ponderado por curso) —
    # no es el promedio de los promedios, que daría distinto si un curso
    # tiene 2 votos y otro 200.
    total["calificacion_promedio"] = (
        round((total["positivas"] - total["negativas"]) / total["calificaciones_total"], 2)
        if total["calificaciones_total"] else None
    )
    return {"total": total, "por_curso": por_curso}


def list_asistentes() -> List[Dict[str, Any]]:
    # LEFT JOIN: un asistente "normal" (rd_id NULL) no tiene fila en rds; uno
    # sin mensajes calificados todavía no tiene fila en la subquery `c`.
    rows = db.fetch_all(
        f"""
        SELECT a.*, r.nombre AS rd_nombre,
               c.calificacion_promedio, COALESCE(c.calificaciones_total, 0) AS calificaciones_total
        FROM asistentes a
        LEFT JOIN rds r ON r.id = a.rd_id
        LEFT JOIN {_CALIF_SUBQUERY} ON c.asistente_id = a.id
        ORDER BY a.id
        """
    )
    return [_decorate(r) for r in rows]


def get_asistente(asistente_id: int) -> Optional[Dict[str, Any]]:
    row = db.fetch_one(
        f"""
        SELECT a.*, r.nombre AS rd_nombre,
               c.calificacion_promedio, COALESCE(c.calificaciones_total, 0) AS calificaciones_total
        FROM asistentes a
        LEFT JOIN rds r ON r.id = a.rd_id
        LEFT JOIN {_CALIF_SUBQUERY} ON c.asistente_id = a.id
        WHERE a.id = %s
        """,
        (asistente_id,),
    )
    return _decorate(row) if row else None


def get_asistente_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Solo asistentes activos — usado por la ruta pública. `rd_id` ya
    viene incluido (columna propia de asistentes, sin necesidad de join)."""
    return db.fetch_one(
        "SELECT * FROM asistentes WHERE token = %s AND activo = TRUE",
        (token,),
    )


_STRATEGY_FIELDS = (
    "etl_document_strategy",
    "etl_audio_strategy",
    "contextual_strategy",
    "rerank_strategy",
    "generation_strategy",
)
# dense_strategy/sparse_strategy YA NO se escriben desde aquí: son el
# embedding maestro de la RD (rds.dense_strategy/sparse_strategy), no del
# asistente — todas las colecciones de una RD deben compartirlo. Las
# columnas siguen existiendo en `asistentes` (no se eliminaron, evita una
# migración destructiva) pero quedan sin uso de aquí en adelante.


def _check_token_disponible(token: str, excluir_asistente_id: Optional[int] = None) -> None:
    """Valida que ningún OTRO asistente ya use este token — el token es la
    llave de acceso pública (/asistente/{token}), tiene que ser única."""
    existing = db.fetch_one("SELECT id FROM asistentes WHERE token = %s", (token,))
    if existing and existing["id"] != excluir_asistente_id:
        raise ValueError(f"El token '{token}' ya está en uso por otro asistente")


def create_asistente(
    nombre: str,
    rd_id: Optional[int],
    moodle_courseid: Optional[int] = None,
    prompt_maestro: Optional[str] = None,
    created_by: Optional[int] = None,
    etl_document_strategy: Optional[str] = None,
    etl_audio_strategy: Optional[str] = None,
    contextual_strategy: Optional[str] = None,
    rerank_strategy: Optional[str] = None,
    generation_strategy: Optional[str] = None,
    mensaje_bienvenida: Optional[str] = None,
    mensaje_reencuentro: Optional[str] = None,
    max_actividades_sesion: int = 10,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """`rd_id=None`: asistente "normal", sin RD — `moodle_courseid` es
    entonces obligatorio: resuelve su única colección directamente por ese
    courseid (ver rds.resolve_collection_normal), en vez de RD+courseid.
    `token`: si se pasa, se usa tal cual (validando que no choque con otro
    asistente); si no, se genera uno aleatorio como antes."""
    if rd_id is not None and not db.fetch_one("SELECT id FROM rds WHERE id = %s", (rd_id,)):
        raise ValueError(f"La RD {rd_id} no existe")
    if rd_id is None and not moodle_courseid:
        raise ValueError("Un asistente normal (sin RD) requiere un Course ID de Moodle")
    if token:
        token = token.strip()
        _check_token_disponible(token)
    else:
        token = secrets.token_urlsafe(24)
    row = db.execute_returning(
        "INSERT INTO asistentes "
        "(nombre, rd_id, moodle_courseid, prompt_maestro, token, created_by, "
        "etl_document_strategy, etl_audio_strategy, contextual_strategy, "
        "rerank_strategy, generation_strategy, mensaje_bienvenida, "
        "mensaje_reencuentro, max_actividades_sesion) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
        (
            nombre, rd_id, moodle_courseid, prompt_maestro, token, created_by,
            etl_document_strategy, etl_audio_strategy, contextual_strategy,
            rerank_strategy, generation_strategy, mensaje_bienvenida,
            mensaje_reencuentro, max_actividades_sesion,
        ),
    )
    return _decorate(row)


def update_asistente(
    asistente_id: int,
    nombre: Optional[str] = None,
    rd_id: Any = _UNSET,
    moodle_courseid: Any = _UNSET,
    prompt_maestro: Optional[str] = None,
    activo: Optional[bool] = None,
    etl_document_strategy: Any = _UNSET,
    etl_audio_strategy: Any = _UNSET,
    contextual_strategy: Any = _UNSET,
    rerank_strategy: Any = _UNSET,
    generation_strategy: Any = _UNSET,
    mensaje_bienvenida: Any = _UNSET,
    mensaje_reencuentro: Any = _UNSET,
    max_actividades_sesion: Optional[int] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Los campos de estrategia (y mensaje_bienvenida/mensaje_reencuentro)
    distinguen "no venía en el request" (`_UNSET`, default — deja el valor
    actual intacto) de "venía como `null`" (limpia ese campo a su
    comportamiento por defecto) — necesario para que el modal de edición
    pueda volver un campo a default sin tocar los demás. `rd_id`/
    `moodle_courseid` usan el MISMO patrón `_UNSET` (antes usaban
    `Optional[int] = None`, lo que hacía imposible limpiar `rd_id` a NULL
    para convertir un asistente RD-based en "normal" — pasar `None`
    explícito se confundía con "no vino en el request" y se ignoraba
    silenciosamente). nombre/prompt_maestro/activo/max_actividades_sesion/
    token mantienen el comportamiento previo (solo cambian si vienen
    informados). Cambiar el token invalida cualquier enlace ya compartido
    con el token viejo."""
    existing = db.fetch_one("SELECT * FROM asistentes WHERE id = %s", (asistente_id,))
    if not existing:
        raise ValueError(f"El asistente {asistente_id} no existe")

    rd_id_value = existing["rd_id"] if rd_id is _UNSET else rd_id
    courseid_value = existing["moodle_courseid"] if moodle_courseid is _UNSET else moodle_courseid
    if rd_id_value is not None and not db.fetch_one("SELECT id FROM rds WHERE id = %s", (rd_id_value,)):
        raise ValueError(f"La RD {rd_id_value} no existe")
    # Mismo invariante que create_asistente: sin RD, el courseid es lo único
    # que resuelve la colección (rds.resolve_collection_normal).
    if rd_id_value is None and not courseid_value:
        raise ValueError("Un asistente normal (sin RD) requiere un Course ID de Moodle")

    if token:
        token = token.strip()
        _check_token_disponible(token, excluir_asistente_id=asistente_id)

    incoming = {
        "etl_document_strategy": etl_document_strategy,
        "etl_audio_strategy": etl_audio_strategy,
        "contextual_strategy": contextual_strategy,
        "rerank_strategy": rerank_strategy,
        "generation_strategy": generation_strategy,
    }
    strategy_values = [
        existing[field] if incoming[field] is _UNSET else incoming[field] for field in _STRATEGY_FIELDS
    ]
    mensaje_bienvenida_value = (
        existing["mensaje_bienvenida"] if mensaje_bienvenida is _UNSET else mensaje_bienvenida
    )
    mensaje_reencuentro_value = (
        existing["mensaje_reencuentro"] if mensaje_reencuentro is _UNSET else mensaje_reencuentro
    )

    row = db.execute_returning(
        "UPDATE asistentes SET nombre=%s, rd_id=%s, moodle_courseid=%s, prompt_maestro=%s, activo=%s, "
        "etl_document_strategy=%s, etl_audio_strategy=%s, contextual_strategy=%s, "
        "rerank_strategy=%s, generation_strategy=%s, mensaje_bienvenida=%s, "
        "mensaje_reencuentro=%s, max_actividades_sesion=%s, token=%s, "
        "updated_at=now() WHERE id=%s RETURNING *",
        (
            nombre if nombre is not None else existing["nombre"],
            rd_id_value,
            courseid_value,
            prompt_maestro if prompt_maestro is not None else existing["prompt_maestro"],
            activo if activo is not None else existing["activo"],
            *strategy_values,
            mensaje_bienvenida_value,
            mensaje_reencuentro_value,
            max_actividades_sesion if max_actividades_sesion is not None else existing["max_actividades_sesion"],
            token if token else existing["token"],
            asistente_id,
        ),
    )
    return _decorate(row)


def get_asistente_config_for_collection(qdrant_collection_name: str) -> Optional[Dict[str, Any]]:
    """Resuelve colección -> RD -> asistente activo más reciente de esa RD,
    y retorna sus 7 campos de estrategia. Para una colección "normal" (sin
    RD), resuelve en cambio por moodle_courseid: el asistente "normal"
    dueño de ese mismo curso (`a.rd_id = c.rd_id` nunca matchea cuando
    ambos son NULL — NULL <> NULL en SQL). `None` si la colección no está
    vinculada a ningún asistente activo. Usado por core._build_pipeline_config()
    para que Semiautomático/Automático/Actualización consulten
    automáticamente la config del asistente dueño de la colección destino,
    sin UI nueva en esos wizards."""
    return db.fetch_one(
        "SELECT a.* FROM colecciones c "
        "JOIN asistentes a ON a.activo = TRUE AND ("
        "  (c.rd_id IS NOT NULL AND a.rd_id = c.rd_id) "
        "  OR (c.rd_id IS NULL AND a.rd_id IS NULL AND a.moodle_courseid = c.moodle_courseid)"
        ") "
        "WHERE c.qdrant_collection_name = %s "
        "ORDER BY a.updated_at DESC LIMIT 1",
        (qdrant_collection_name,),
    )


def delete_asistente(asistente_id: int) -> None:
    if not db.fetch_one("SELECT id FROM asistentes WHERE id = %s", (asistente_id,)):
        raise ValueError(f"El asistente {asistente_id} no existe")
    db.execute("DELETE FROM asistentes WHERE id = %s", (asistente_id,))  # CASCADE: sesiones + mensajes


# ── Sesiones y mensajes (usado por la ruta pública) ─────────────────────

def compose_saludo(nombre: str, mensaje_bienvenida: Optional[str], student_fullname: Optional[str]) -> str:
    """Arma el saludo inicial (mismo criterio en los 3 lugares que antes lo
    componían por separado: la página del chat, "Nueva conversación", y
    ahora también al guardarlo como el primer mensaje real de la sesión).
    Sin esto el saludo solo se pintaba en el navegador y nunca quedaba en
    `asistente_mensajes` — al recargar la página o revisar una conversación
    vieja desde la sidebar, desaparecía."""
    saludo = f"Hola {student_fullname}, " if student_fullname else "Hola, "
    resto = mensaje_bienvenida or f"soy **{nombre}**. Pregúntame lo que necesites sobre el curso."
    return saludo + resto


def get_or_create_sesion(
    asistente_id: int,
    moodle_courseid: int,
    moodle_userid: int,
    moodle_username: Optional[str],
    moodle_fullname: Optional[str],
    qdrant_collection_name: str,
    saludo_inicial: Optional[str] = None,
    force_new: bool = False,
) -> Dict[str, Any]:
    """Reusa la sesión existente para (asistente, curso, usuario) si hay una,
    si no crea una nueva. Así el historial se acumula por usuario/curso.
    `force_new=True` (botón "Nueva conversación") salta la reutilización y
    siempre inserta una sesión nueva, sin arrastrar la memoria anterior.
    `saludo_inicial`: si se crea una sesión nueva, se guarda como su primer
    mensaje (rol assistant) — así el saludo queda en el historial real."""
    if not force_new:
        existing = db.fetch_one(
            "SELECT * FROM asistente_sesiones WHERE asistente_id=%s AND moodle_courseid=%s AND moodle_userid=%s "
            "ORDER BY started_at DESC LIMIT 1",
            (asistente_id, moodle_courseid, moodle_userid),
        )
        if existing:
            db.execute("UPDATE asistente_sesiones SET last_activity_at = now() WHERE id = %s", (existing["id"],))
            return existing

    sesion = db.execute_returning(
        "INSERT INTO asistente_sesiones "
        "(asistente_id, moodle_courseid, moodle_userid, moodle_username, moodle_fullname, qdrant_collection_name) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
        (asistente_id, moodle_courseid, moodle_userid, moodle_username, moodle_fullname, qdrant_collection_name),
    )
    if saludo_inicial:
        save_mensaje(sesion["id"], "assistant", saludo_inicial)
    return sesion


def get_sesion(sesion_id: int) -> Optional[Dict[str, Any]]:
    return db.fetch_one("SELECT * FROM asistente_sesiones WHERE id = %s", (sesion_id,))


def list_sesiones(asistente_id: int, moodle_courseid: int, moodle_userid: int) -> List[Dict[str, Any]]:
    """Todas las conversaciones de un alumno con este asistente en este
    curso, más recientes primero, con una vista previa (su primer mensaje).
    Usado por la sidebar de historial de la página pública. Incluye
    `nombre` (renombre manual, NULL si no se renombró) y `carpeta_id`
    (NULL si no está en ninguna carpeta)."""
    return db.fetch_all(
        "SELECT s.id, s.started_at, s.last_activity_at, s.nombre, s.carpeta_id, "
        "  (SELECT contenido FROM asistente_mensajes m WHERE m.sesion_id = s.id AND m.rol = 'user' "
        "   ORDER BY m.created_at ASC LIMIT 1) AS preview "
        "FROM asistente_sesiones s "
        "WHERE s.asistente_id = %s AND s.moodle_courseid = %s AND s.moodle_userid = %s "
        "ORDER BY s.last_activity_at DESC",
        (asistente_id, moodle_courseid, moodle_userid),
    )


def rename_sesion(sesion_id: int, nombre: Optional[str]) -> Dict[str, Any]:
    """`nombre=None` (o string vacío) vuelve la sesión a su nombre por
    defecto (el preview del primer mensaje, calculado en list_sesiones)."""
    nombre = nombre.strip() if nombre and nombre.strip() else None
    row = db.execute_returning(
        "UPDATE asistente_sesiones SET nombre = %s WHERE id = %s RETURNING *",
        (nombre, sesion_id),
    )
    if not row:
        raise ValueError(f"La sesión {sesion_id} no existe")
    return row


def mover_sesion_a_carpeta(sesion_id: int, carpeta_id: Optional[int]) -> Dict[str, Any]:
    """`carpeta_id=None` saca la sesión de cualquier carpeta."""
    row = db.execute_returning(
        "UPDATE asistente_sesiones SET carpeta_id = %s WHERE id = %s RETURNING *",
        (carpeta_id, sesion_id),
    )
    if not row:
        raise ValueError(f"La sesión {sesion_id} no existe")
    return row


def create_carpeta(asistente_id: int, moodle_userid: int, nombre: str) -> Dict[str, Any]:
    nombre = nombre.strip()
    if not nombre:
        raise ValueError("El nombre de la carpeta no puede estar vacío")
    return db.execute_returning(
        "INSERT INTO asistente_carpetas (asistente_id, moodle_userid, nombre) VALUES (%s, %s, %s) RETURNING *",
        (asistente_id, moodle_userid, nombre),
    )


def list_carpetas(asistente_id: int, moodle_userid: int) -> List[Dict[str, Any]]:
    """Carpetas del alumno para este asistente, más nuevas primero."""
    return db.fetch_all(
        "SELECT * FROM asistente_carpetas WHERE asistente_id = %s AND moodle_userid = %s ORDER BY created_at DESC",
        (asistente_id, moodle_userid),
    )


def get_carpeta(carpeta_id: int) -> Optional[Dict[str, Any]]:
    return db.fetch_one("SELECT * FROM asistente_carpetas WHERE id = %s", (carpeta_id,))


def delete_carpeta(carpeta_id: int) -> None:
    """Las sesiones dentro quedan sin carpeta (carpeta_id -> NULL vía
    ON DELETE SET NULL, ya definido en el esquema) — no se borran."""
    if not db.fetch_one("SELECT id FROM asistente_carpetas WHERE id = %s", (carpeta_id,)):
        raise ValueError(f"La carpeta {carpeta_id} no existe")
    db.execute("DELETE FROM asistente_carpetas WHERE id = %s", (carpeta_id,))


def count_preguntas(sesion_id: int) -> int:
    """Cantidad de preguntas (mensajes rol='user') ya hechas en la sesión —
    usado para aplicar el tope `max_actividades_sesion` del asistente."""
    row = db.fetch_one(
        "SELECT COUNT(*) AS n FROM asistente_mensajes WHERE sesion_id = %s AND rol = 'user'", (sesion_id,)
    )
    return row["n"] if row else 0


def get_mensajes(sesion_id: int) -> List[Dict[str, Any]]:
    return db.fetch_all(
        "SELECT * FROM asistente_mensajes WHERE sesion_id = %s ORDER BY created_at", (sesion_id,)
    )


def get_recent_history(sesion_id: int, max_pairs: int = 30) -> List[Dict[str, str]]:
    """Últimos `max_pairs` turnos (pregunta+respuesta, hasta `2*max_pairs`
    filas crudas) de la sesión, en orden cronológico, listos para pasar
    como historial real de conversación al LLM (no como texto aplanado).
    Usado por /asistente/{token}/mensaje ANTES de guardar la pregunta
    actual, para no duplicarla en el historial."""
    rows = db.fetch_all(
        "SELECT rol, contenido FROM ("
        "  SELECT rol, contenido, created_at FROM asistente_mensajes "
        "  WHERE sesion_id = %s ORDER BY created_at DESC LIMIT %s"
        ") sub ORDER BY created_at",
        (sesion_id, max_pairs * 2),
    )
    return [{"role": r["rol"], "content": r["contenido"]} for r in rows]


def save_mensaje(
    sesion_id: int,
    rol: str,
    contenido: str,
    fuentes: Optional[list] = None,
    config_usado: Optional[dict] = None,
    tokens_consumidos: Optional[int] = None,
    tiempo_respuesta_ms: Optional[int] = None,
) -> Dict[str, Any]:
    from psycopg2.extras import Json

    return db.execute_returning(
        "INSERT INTO asistente_mensajes (sesion_id, rol, contenido, fuentes_json, config_usado_json, tokens_consumidos, tiempo_respuesta_ms) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
        (
            sesion_id,
            rol,
            contenido,
            Json(fuentes) if fuentes is not None else None,
            Json(config_usado) if config_usado is not None else None,
            tokens_consumidos,
            tiempo_respuesta_ms,
        ),
    )


def get_mensaje_con_sesion(mensaje_id: int) -> Optional[Dict[str, Any]]:
    """Mensaje + datos de su sesión (asistente_id, courseid, userid), para
    validar pertenencia antes de calificar — mismo criterio de acceso que
    _validar_sesion_del_alumno pero partiendo del mensaje, no de la sesión."""
    return db.fetch_one(
        "SELECT m.*, s.asistente_id, s.moodle_courseid, s.moodle_userid "
        "FROM asistente_mensajes m JOIN asistente_sesiones s ON s.id = m.sesion_id "
        "WHERE m.id = %s",
        (mensaje_id,),
    )


def calificar_mensaje(mensaje_id: int, calificacion: int, comentario: Optional[str] = None) -> Dict[str, Any]:
    """Guarda o reemplaza la calificación de utilidad de un mensaje 'assistant'
    (1 = me ayudó, -1 = no me ayudó). El alumno no califica al mentor/RD,
    solo si la interacción le fue útil. `comentario` solo se persiste cuando
    calificacion == -1 (se descarta si viene con 1, para no dejar basura de
    un cambio de opinión anterior)."""
    if calificacion not in (1, -1):
        raise ValueError("calificacion debe ser 1 o -1")
    comentario_final = ((comentario or "").strip() or None) if calificacion == -1 else None
    row = db.execute_returning(
        "UPDATE asistente_mensajes SET calificacion = %s, calificacion_comentario = %s "
        "WHERE id = %s AND rol = 'assistant' RETURNING *",
        (calificacion, comentario_final, mensaje_id),
    )
    if not row:
        raise ValueError(f"El mensaje {mensaje_id} no existe o no es una respuesta del asistente")
    return row
