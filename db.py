"""
db.py
=====
Capa de acceso a Postgres (toda la configuración del gestor, RD,
asistentes, sesiones/mensajes de chat). Cliente síncrono (psycopg2), igual
que el resto de clientes externos de este proyecto (OpenAI, Qdrant) —
las rutas async lo llaman vía `run_in_executor` (ver `db.run`).

Esquema: ver db_schema.sql (crear) / db_drop.sql (eliminar todo).
"""
import asyncio
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool

from settings import settings

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        # minconn=5: precalienta 5 conexiones al arrancar. El host de Postgres
        # está en otra red (~15-25ms de ping) — abrir una conexión nueva ahí
        # cuesta ~70ms (handshake TCP + auth) vs ~40ms de una query sobre una
        # conexión ya abierta. Con minconn=1 (el default anterior), cualquier
        # solicitud concurrente pagaba ese costo extra de conexión; con 5
        # precalentadas, las primeras 5 solicitudes simultáneas no lo pagan.
        # maxconn=20: cada mensaje de un asistente encadena varias llamadas
        # cortas a Postgres (historial, guardar pregunta, resolver RD,
        # guardar respuesta) — con varios alumnos preguntando a la vez hace
        # falta margen por encima de esas 5 precalentadas.
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=5,
            maxconn=20,
            host=settings.PG_HOST,
            port=settings.PG_PORT,
            user=settings.PG_USER,
            password=settings.PG_PASSWORD,
            dbname=settings.PG_DATABASE,
            connect_timeout=10,
            # Sin keepalives, un firewall/NAT entre esta máquina y el host
            # remoto de Postgres corta en silencio las conexiones ociosas
            # entre turnos de uso — el socket queda "muerto" pero psycopg2
            # no se entera hasta el siguiente query (InterfaceError
            # "connection already closed"). Con keepalives, el SO manda un
            # paquete cada 30s de inactividad y detecta/renueva la conexión
            # antes de que el intermediario la cierre.
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
        )
    return _pool


@contextmanager
def _cursor():
    pool = _get_pool()
    conn = pool.getconn()
    broken = False
    try:
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                yield cur
    except (psycopg2.InterfaceError, psycopg2.OperationalError):
        # Conexión caída (red, timeout del lado del servidor, idle
        # demasiado tiempo, etc.) — se descarta en vez de devolverla al
        # pool, si no el pool sigue entregando la misma conexión rota a
        # cada request siguiente hasta reiniciar el proceso.
        broken = True
        raise
    finally:
        pool.putconn(conn, close=broken)


def fetch_all(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def fetch_one(sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    with _cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def execute(sql: str, params: tuple = ()) -> None:
    with _cursor() as cur:
        cur.execute(sql, params)


def execute_returning(sql: str, params: tuple = ()) -> Dict[str, Any]:
    """Para INSERT/UPDATE ... RETURNING *."""
    with _cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else {}


async def run(fn, *args, **kwargs):
    """Ejecuta una función síncrona de este módulo (fetch_all/fetch_one/
    execute/execute_returning) sin bloquear el event loop, desde rutas
    async — mismo patrón que ya usa este proyecto para clientes externos
    síncronos (OpenAI, Qdrant, etc.)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def check_connection() -> bool:
    try:
        fetch_one("SELECT 1 AS ok")
        return True
    except Exception as e:
        print(f"⚠️ No se pudo conectar a Postgres: {e}")
        return False
