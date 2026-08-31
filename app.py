"""
app.py
======
FastAPI — un solo puerto (8100), tres asistentes:
  · /api/semi/*    → asistente semiautomático
  · /api/auto/*    → asistente automático (Moodle)
  · /api/update/*  → asistente de actualización
  · /api/          → endpoints compartidos (colecciones, job status, modelos)
"""

import asyncio
import mimetypes
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

# En algunos Windows/Python el registro de mimetypes no conoce .webp
# (StaticFiles lo serviría como text/plain) — se registra explícito.
mimetypes.add_type("image/webp", ".webp")

from pydantic import BaseModel
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    BackgroundTasks, Query, Request, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from settings import (
    settings, validate_settings, AVAILABLE_EMBEDDING_MODELS,
    STAGE_CATALOGS, get_strategy_info, strategy_is_usable, get_model_dimensions,
)

from definitions import (get_service,
    AutoPreviewRequest, AutoVectorizeRequest,
    UpdateVectorizeRequest,
    _sanitize_filename,
    get_moodle_user_fullname, get_moodle_course_name,
)
from core import (
    SemiAutoCore, AutoCore, UpdateCore,
    get_job, job_store,
)
import rag_pipeline
import credentials
import db
import auth
import rds
import assistants
from strategies import registry as strategy_registry
from strategies.runtime_config import get_runtime_config
from qdrant_admin import CollectionCreateRequest, CollectionUpdateRequest, get_qdrant_admin


# ═══════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando Sistema de Vectorización v2...")
    validate_settings()
    if db.check_connection():
        print("✅ Postgres conectado")
        auth.bootstrap_admin()
        credentials.load_all_into_settings()
    else:
        print("⚠️ Postgres no disponible al arrancar — el login y las funciones de Asistentes/RD fallarán hasta que se restablezca la conexión.")
    # Pre-carga el modelo por defecto al arrancar
    get_service()
    print("✅ Sistema listo")
    yield
    print("👋 Cerrando sistema...")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rutas públicas (sin sesión): login, estáticos, chat público de
# asistentes, y documentación/health. Todo lo demás requiere sesión. ──
_PUBLIC_PATH_PREFIXES = ("/login", "/logout", "/img", "/asistente", "/docs", "/redoc", "/openapi.json", "/api/health")


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    path = request.url.path
    if path == "/" or path.startswith(_PUBLIC_PATH_PREFIXES):
        return await call_next(request)
    if not request.session.get("user"):
        if path.startswith("/api/"):
            return JSONResponse({"detail": "No autenticado. Inicia sesión en /login."}, status_code=401)
        return RedirectResponse("/login")
    return await call_next(request)


# NOTA de orden: Starlette apila los middlewares registrados con
# add_middleware() de modo que el ÚLTIMO agregado queda MÁS externo (se
# ejecuta primero). SessionMiddleware debe ejecutarse ANTES que auth_gate
# (para que request.session ya exista cuando auth_gate lo lee), así que
# se registra DESPUÉS en el código — verificado en vivo: registrarlo antes
# rompía con "SessionMiddleware must be installed to access request.session".
app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY, same_site="lax")


Path("frontend").mkdir(exist_ok=True)
templates = Jinja2Templates(directory="frontend")

Path("img").mkdir(exist_ok=True)
app.mount("/img", StaticFiles(directory="img"), name="img")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _save_upload_to_temp(upload: UploadFile, content: bytes) -> tuple:
    """
    Guarda un UploadFile en temp_docs/ con nombre sanitizado para el sistema de archivos.
    """
    original_name = upload.filename
    safe_name = _sanitize_filename(original_name)
    dest = settings.TEMP_DIR / safe_name
    dest.write_bytes(content)
    return dest, original_name


def _resolve_model(model_name: Optional[str]) -> str:
    """
    Retorna el model_name validado. Si es None usa el default del .env.
    Lanza HTTP 400 si el nombre no está en el catálogo.
    """
    key = model_name or settings.EMBEDDING_MODEL
    if key not in AVAILABLE_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Modelo '{key}' no está en el catálogo de modelos en español. "
                f"Usa GET /api/models para ver los disponibles."
            ),
        )
    return key


# ═══════════════════════════════════════════════════════════════════
# INTERFAZ WEB + LOGIN
# ═══════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root(request: Request):
    return RedirectResponse("/gestor" if request.session.get("user") else "/login")


@app.get("/gestor", response_class=HTMLResponse, tags=["Interfaz"])
async def home(request: Request):
    return templates.TemplateResponse("interfaz.html", {"request": request, "user": request.session.get("user")})


@app.get("/login", response_class=HTMLResponse, tags=["Interfaz"])
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/gestor")
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/login", tags=["Interfaz"])
async def login_submit(request: Request, req: LoginRequest):
    user = await db.run(auth.authenticate, req.username, req.password)
    if not user:
        raise HTTPException(401, "Usuario o contraseña incorrectos.")
    request.session["user"] = {"id": user["id"], "username": user["username"], "is_admin": user["is_admin"]}
    return {"success": True}


@app.post("/logout", tags=["Interfaz"])
async def logout(request: Request):
    request.session.clear()
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════
# USUARIOS (solo administradores)
# ═══════════════════════════════════════════════════════════════════

class CreateUserRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False


@app.get("/api/usuarios", tags=["Usuarios"])
async def list_usuarios(user=Depends(auth.require_admin)):
    return {"usuarios": await db.run(auth.list_users)}


@app.post("/api/usuarios", status_code=201, tags=["Usuarios"])
async def create_usuario(req: CreateUserRequest, user=Depends(auth.require_admin)):
    try:
        return await db.run(auth.create_user, req.username, req.password, req.is_admin)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/usuarios/{user_id}", tags=["Usuarios"])
async def delete_usuario(user_id: int, user=Depends(auth.require_admin)):
    if user_id == user["id"]:
        raise HTTPException(400, "No puedes eliminar tu propio usuario mientras tienes sesión activa.")
    await db.run(auth.delete_user, user_id)
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════
# COMPARTIDOS
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["Sistema"])
async def health():
    return {
        "status":        "healthy",
        "version":       settings.API_VERSION,
        "default_model": settings.EMBEDDING_MODEL,
        "docs":          f"http://{settings.API_HOST}:{settings.API_PORT}/docs",
    }


@app.get("/api/models", tags=["Sistema"])
async def list_models():
    """
    Lista el catálogo completo de modelos de embeddings disponibles,
    todos especializados o validados para texto en español.

    Retorna:
      · default      — modelo activo según .env
      · models[]     — catálogo con nombre, dimensiones y descripción
    """
    return {
        "default": settings.EMBEDDING_MODEL,
        "models": [
            {
                "name":        name,
                "dimensions":  info["dimensions"],
                "description": info["description"],
                "use_prefix":  info["use_prefix"],
            }
            for name, info in AVAILABLE_EMBEDDING_MODELS.items()
        ],
    }


@app.get("/api/collections", tags=["Compartido"])
async def list_collections(
    detail: bool = Query(
        False,
        description=(
            "Si es true, retorna el listado enriquecido (esquema, puntos, "
            "estado, descripción) desde el gestor de administración de "
            "Qdrant en vez de solo los nombres."
        ),
    ),
):
    """Lista todas las colecciones disponibles en Qdrant."""
    if detail:
        from qdrant_admin import get_qdrant_admin

        collections = [c.model_dump() for c in get_qdrant_admin().list_collections()]
        links = await db.run(
            db.fetch_all,
            "SELECT c.id AS link_id, c.qdrant_collection_name, c.rd_id, c.moodle_courseid, r.nombre AS rd_nombre "
            "FROM colecciones_rd c JOIN rds r ON r.id = c.rd_id "
            "WHERE c.qdrant_collection_name IS NOT NULL",
        )
        links_by_name = {l["qdrant_collection_name"]: l for l in links}
        for c in collections:
            link = links_by_name.get(c["name"])
            c["link_id"] = link["link_id"] if link else None
            c["rd_id"] = link["rd_id"] if link else None
            c["rd_nombre"] = link["rd_nombre"] if link else None
            c["moodle_courseid"] = link["moodle_courseid"] if link else None
        return {"collections": collections}
    service = get_service()
    return {"collections": service.list_collections()}


# ═══════════════════════════════════════════════════════════════════
# GESTIÓN QDRANT — administración de colecciones (dense y dense+sparse)
# Absorbe el proyecto standalone "gestion quadrant": mismas 8 rutas,
# apuntando a qdrant_admin.QdrantAdminManager en vez de un microservicio
# aparte, con soporte adicional para el esquema híbrido.
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/collections", status_code=201, tags=["Gestión Qdrant"])
async def qdrant_create_collection(req: CollectionCreateRequest):
    """Crea una colección en Qdrant. `vector_schema`: "hybrid" (dense+sparse,
    requerido para el pipeline RAG configurable) o "legacy" (vector único,
    compatible con el flujo anterior). `rd_id`+`moodle_courseid` son
    obligatorios: la colección queda asignada a esa RD/curso en Postgres
    (colecciones_rd) — sin eso ningún asistente podría encontrarla.

    Para colecciones híbridas, el tamaño de vector denso NO se recibe del
    cliente: si la RD ya tiene un embedding maestro fijado, se deriva de
    ahí (ignora lo que venga en `dense_size`/`dense_strategy`, todas las
    colecciones de una RD deben ser intercambiables entre sí); si es la
    primera colección real de la RD, `dense_strategy` es obligatorio y
    queda fijado como el embedding maestro de esa RD de ahí en adelante.

    Si el RD/curso no existe, ya tiene otra colección asignada, o falla
    fijar el embedding maestro, la colección recién creada en Qdrant se
    revierte para no dejar huérfanos."""
    dense_size = req.dense_size
    lock_in_embedding: Optional[tuple] = None  # (dense_strategy, sparse_strategy) a fijar tras crear

    if req.vector_schema == "hybrid":
        rd = await db.run(rds.get_rd, req.rd_id)
        if not rd:
            raise HTTPException(400, f"La RD {req.rd_id} no existe")
        if rd.get("dense_strategy"):
            dense_size = get_model_dimensions(rd["dense_strategy"])
        else:
            if not req.dense_strategy:
                raise HTTPException(
                    400,
                    "Esta RD todavía no tiene un embedding maestro fijado — es su primera "
                    "colección, indica 'dense_strategy' para fijarlo.",
                )
            try:
                get_strategy_info("dense", req.dense_strategy)
            except ValueError as e:
                raise HTTPException(400, str(e))
            dense_size = get_model_dimensions(req.dense_strategy)
            sparse_strategy = req.sparse_strategy or settings.SPARSE_EMBEDDING_STRATEGY
            lock_in_embedding = (req.dense_strategy, sparse_strategy)

    try:
        result = get_qdrant_admin().create_collection(
            name=req.name,
            description=req.description,
            vector_schema=req.vector_schema,
            dense_size=dense_size,
            distance=req.distance,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

    try:
        await db.run(rds.link_collection, req.name, req.rd_id, req.moodle_courseid)
        if lock_in_embedding:
            await db.run(rds.set_embedding, req.rd_id, *lock_in_embedding)
    except ValueError as e:
        try:
            get_qdrant_admin().delete_collection(req.name, force=True)
        except Exception:
            pass
        raise HTTPException(400, str(e))

    return result


@app.get("/api/collections/{collection_name}", tags=["Gestión Qdrant"])
async def qdrant_get_collection(collection_name: str):
    """Información detallada de una colección (esquema, puntos, estado)."""
    try:
        return get_qdrant_admin().get_collection_info(collection_name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.patch("/api/collections/{collection_name}", tags=["Gestión Qdrant"])
async def qdrant_update_collection(collection_name: str, req: CollectionUpdateRequest):
    """Actualiza la descripción de una colección."""
    try:
        return get_qdrant_admin().update_collection(collection_name, req.description)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


class CollectionRdReassignRequest(BaseModel):
    rd_id: int
    moodle_courseid: int


@app.patch("/api/collections/{collection_name}/rd", tags=["Gestión Qdrant"])
async def qdrant_reassign_collection_rd(collection_name: str, req: CollectionRdReassignRequest):
    """Reasigna a qué RD/courseid pertenece una colección ya creada."""
    if not get_qdrant_admin().collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe")
    try:
        return await db.run(rds.reassign_collection, collection_name, req.rd_id, req.moodle_courseid)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/collections/{collection_name}", tags=["Gestión Qdrant"])
async def qdrant_delete_collection(
    collection_name: str,
    force: bool = Query(False, description="Forzar eliminación aunque tenga vectores"),
):
    """Elimina una colección de Qdrant y su vínculo con la RD (si tenía)."""
    try:
        result = get_qdrant_admin().delete_collection(collection_name, force=force)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    await db.run(rds.unlink_collection, collection_name)
    return result


@app.post("/api/collections/{collection_name}/clear", tags=["Gestión Qdrant"])
async def qdrant_clear_collection(collection_name: str):
    """Elimina todos los vectores de una colección, conservando su
    configuración (esquema legacy o híbrido)."""
    try:
        return get_qdrant_admin().clear_collection(collection_name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/collections/{collection_name}/exists", tags=["Gestión Qdrant"])
async def qdrant_collection_exists(collection_name: str):
    """Verifica si una colección existe."""
    return {
        "collection_name": collection_name,
        "exists": get_qdrant_admin().collection_exists(collection_name),
    }


@app.get("/api/collections/{collection_name}/documents", tags=["Gestión Qdrant"])
async def qdrant_list_documents(collection_name: str):
    """Lista los documentos únicos vectorizados en una colección."""
    try:
        return get_qdrant_admin().list_documents_in_collection(collection_name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/collections/{collection_name}/documents/{filename:path}", tags=["Gestión Qdrant"])
async def qdrant_delete_document(collection_name: str, filename: str):
    """Elimina todos los chunks de un documento (por filename) de una colección."""
    try:
        return get_qdrant_admin().delete_document_by_filename(collection_name, filename)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


# ═══════════════════════════════════════════════════════════════════
# RD (aulas)  /api/rds/*
# ═══════════════════════════════════════════════════════════════════

class RdCreateRequest(BaseModel):
    nombre: str
    moodle_courseid: int
    moodle_course_url: Optional[str] = None


class RdUpdateRequest(BaseModel):
    nombre: Optional[str] = None
    moodle_courseid: Optional[int] = None
    moodle_course_url: Optional[str] = None
    # Embedding maestro de la RD — solo se puede cambiar mientras no tenga
    # colecciones reales creadas (ver rds.update_rd).
    dense_strategy: Optional[str] = None
    sparse_strategy: Optional[str] = None


@app.get("/api/rds", tags=["RD"])
async def list_rds_endpoint():
    """Todas las RD, con conteo de colecciones y asistentes asignados."""
    return {"rds": await db.run(rds.list_rds)}


@app.get("/api/rds/{rd_id}", tags=["RD"])
async def get_rd_endpoint(rd_id: int):
    """Detalle de una RD: sus colecciones y asistentes asignados."""
    detail = await db.run(rds.get_rd_detail, rd_id)
    if not detail:
        raise HTTPException(404, f"La RD {rd_id} no existe")
    return detail


@app.post("/api/rds", status_code=201, tags=["RD"])
async def create_rd_endpoint(req: RdCreateRequest):
    return await db.run(rds.create_rd, req.nombre, req.moodle_courseid, req.moodle_course_url)


@app.patch("/api/rds/{rd_id}", tags=["RD"])
async def update_rd_endpoint(rd_id: int, req: RdUpdateRequest):
    fields = req.model_dump(exclude_unset=True)
    strategy_kwargs = {k: fields[k] for k in ("dense_strategy", "sparse_strategy") if k in fields}
    try:
        return await db.run(
            rds.update_rd, rd_id, req.nombre, req.moodle_courseid, req.moodle_course_url, **strategy_kwargs
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/rds/{rd_id}", tags=["RD"])
async def delete_rd_endpoint(rd_id: int):
    try:
        await db.run(rds.delete_rd, rd_id)
        return {"success": True}
    except ValueError as e:
        raise HTTPException(400, str(e))


class RdCourseidRequest(BaseModel):
    moodle_courseid: int


@app.post("/api/rds/{rd_id}/courseids", status_code=201, tags=["RD"])
async def add_rd_courseid_endpoint(rd_id: int, req: RdCourseidRequest):
    """Registra un courseid más bajo esta RD (todavía sin colección)."""
    try:
        return await db.run(rds.add_courseid, rd_id, req.moodle_courseid)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/rds/courseids/{entry_id}", tags=["RD"])
async def remove_rd_courseid_endpoint(entry_id: int):
    """Quita un courseid de la lista de una RD (solo si no tiene colección asignada)."""
    try:
        await db.run(rds.remove_courseid_entry, entry_id)
        return {"success": True}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/moodle/curso/{courseid}", tags=["RD"])
async def moodle_curso_endpoint(courseid: int):
    """Nombre real del curso de Moodle — paso 2 del wizard de asistentes,
    para confirmar el Course ID antes de asociarlo a una RD."""
    loop = asyncio.get_event_loop()
    fullname = await loop.run_in_executor(None, get_moodle_course_name, courseid)
    if not fullname:
        raise HTTPException(404, f"Moodle no reconoce el Course ID {courseid}.")
    return {"id": courseid, "fullname": fullname}


# ═══════════════════════════════════════════════════════════════════
# ASISTENTES (CRUD)  /api/asistentes/*
# ═══════════════════════════════════════════════════════════════════

_ASISTENTE_STRATEGY_FIELDS = (
    "etl_document_strategy",
    "etl_audio_strategy",
    "contextual_strategy",
    "rerank_strategy",
    "generation_strategy",
)


class AsistenteCreateRequest(BaseModel):
    nombre: str
    rd_id: int
    prompt_maestro: Optional[str] = None
    # Config RAG propia del asistente — None/omitido = usa el valor por
    # defecto del sistema. etl_document/etl_audio/contextual también se
    # consultan automáticamente al vectorizar (ver core._build_pipeline_config).
    # dense/sparse NO van aquí: son el embedding maestro de la RD
    # (rds.dense_strategy/sparse_strategy) — todas las colecciones de una
    # RD deben compartirlo, así que el asistente solo lo hereda/muestra.
    etl_document_strategy: Optional[str] = None
    etl_audio_strategy: Optional[str] = None
    contextual_strategy: Optional[str] = None
    rerank_strategy: Optional[str] = None
    generation_strategy: Optional[str] = None


class AsistenteUpdateRequest(BaseModel):
    nombre: Optional[str] = None
    rd_id: Optional[int] = None
    prompt_maestro: Optional[str] = None
    activo: Optional[bool] = None
    # Si el campo no viene en el body, la etapa queda intacta; si viene como
    # null explícito, esa etapa vuelve a usar el valor por defecto (ver
    # assistants.update_asistente — usa model_dump(exclude_unset=True)).
    etl_document_strategy: Optional[str] = None
    etl_audio_strategy: Optional[str] = None
    contextual_strategy: Optional[str] = None
    rerank_strategy: Optional[str] = None
    generation_strategy: Optional[str] = None


@app.get("/api/asistentes", tags=["Asistentes"])
async def list_asistentes_endpoint():
    return {"asistentes": await db.run(assistants.list_asistentes)}


@app.get("/api/asistentes/{asistente_id}", tags=["Asistentes"])
async def get_asistente_endpoint(asistente_id: int):
    a = await db.run(assistants.get_asistente, asistente_id)
    if not a:
        raise HTTPException(404, f"El asistente {asistente_id} no existe")
    return a


@app.post("/api/asistentes", status_code=201, tags=["Asistentes"])
async def create_asistente_endpoint(req: AsistenteCreateRequest, user=Depends(auth.get_current_user)):
    strategy_kwargs = {field: getattr(req, field) for field in _ASISTENTE_STRATEGY_FIELDS}
    try:
        return await db.run(
            assistants.create_asistente,
            req.nombre,
            req.rd_id,
            req.prompt_maestro,
            user["id"],
            **strategy_kwargs,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/api/asistentes/{asistente_id}", tags=["Asistentes"])
async def update_asistente_endpoint(asistente_id: int, req: AsistenteUpdateRequest):
    fields = req.model_dump(exclude_unset=True)
    kwargs = {k: fields[k] for k in _ASISTENTE_STRATEGY_FIELDS if k in fields}
    try:
        return await db.run(
            assistants.update_asistente,
            asistente_id,
            req.nombre,
            req.rd_id,
            req.prompt_maestro,
            req.activo,
            **kwargs,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/asistentes/{asistente_id}", tags=["Asistentes"])
async def delete_asistente_endpoint(asistente_id: int):
    try:
        await db.run(assistants.delete_asistente, asistente_id)
        return {"success": True}
    except ValueError as e:
        raise HTTPException(404, str(e))


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE PÚBLICO (sin login)  /asistente/{token}
# ═══════════════════════════════════════════════════════════════════

@app.get("/asistente/{token}", response_class=HTMLResponse, tags=["Asistente público"])
async def asistente_chat_page(token: str, request: Request, courseid: int = Query(...), userid: int = Query(...)):
    asistente = await db.run(assistants.get_asistente_by_token, token)
    if not asistente:
        return templates.TemplateResponse(
            "asistente_chat.html",
            {"request": request, "error": "Este asistente no existe o no está activo."},
            status_code=404,
        )

    collection_name = await db.run(rds.resolve_collection, asistente["rd_id"], courseid)
    if not collection_name:
        return templates.TemplateResponse(
            "asistente_chat.html",
            {
                "request": request,
                "error": f"No hay ninguna colección asignada a este curso (id {courseid}) en la RD de este asistente.",
            },
            status_code=404,
        )

    loop = asyncio.get_event_loop()
    fullname = await loop.run_in_executor(None, get_moodle_user_fullname, userid)

    sesion = await db.run(
        assistants.get_or_create_sesion,
        asistente["id"], courseid, userid, None, fullname, collection_name,
    )
    historial = await db.run(assistants.get_mensajes, sesion["id"])

    return templates.TemplateResponse(
        "asistente_chat.html",
        {
            "request": request,
            "error": None,
            "token": token,
            "asistente_nombre": asistente["nombre"],
            "sesion_id": sesion["id"],
            "saludo_nombre": fullname or "",
            "historial": historial,
            "courseid": courseid,
            "userid": userid,
        },
    )


class AsistenteMensajeRequest(BaseModel):
    sesion_id: int
    pregunta: str


@app.post("/asistente/{token}/mensaje", tags=["Asistente público"])
async def asistente_enviar_mensaje(token: str, req: AsistenteMensajeRequest):
    asistente = await db.run(assistants.get_asistente_by_token, token)
    if not asistente:
        raise HTTPException(404, "Este asistente no existe o no está activo.")

    sesion = await db.run(assistants.get_sesion, req.sesion_id)
    if not sesion or sesion["asistente_id"] != asistente["id"]:
        raise HTTPException(404, "Sesión no válida para este asistente.")

    if not req.pregunta or not req.pregunta.strip():
        raise HTTPException(400, "La pregunta no puede estar vacía.")

    # Se obtiene el historial ANTES de guardar la pregunta actual, para no
    # duplicarla (memoria de hasta 30 pares pregunta/respuesta).
    history = await db.run(assistants.get_recent_history, req.sesion_id)
    await db.run(assistants.save_mensaje, req.sesion_id, "user", req.pregunta)

    # dense/sparse son el embedding maestro de la RD (no del asistente):
    # todas las colecciones de la RD deben compartirlo para que un mismo
    # asistente pueda responder con cualquiera de ellas indistintamente.
    rd = await db.run(rds.get_rd_for_collection, sesion["qdrant_collection_name"]) or {}

    try:
        result = await rag_pipeline.answer_query(
            collection_name=sesion["qdrant_collection_name"],
            query=req.pregunta,
            dense_strategy_name=rd.get("dense_strategy"),
            sparse_strategy_name=rd.get("sparse_strategy"),
            rerank_strategy_name=asistente.get("rerank_strategy"),
            generation_strategy_name=asistente.get("generation_strategy"),
            extra_system_prompt=asistente.get("prompt_maestro"),
            history=history,
        )
    except Exception as e:
        raise HTTPException(502, f"Error generando la respuesta: {e}")

    await db.run(
        assistants.save_mensaje,
        req.sesion_id, "assistant", result["answer"], result["sources"], result["config_used"],
    )

    return result


class AsistenteNuevaConversacionRequest(BaseModel):
    courseid: int
    userid: int


@app.post("/asistente/{token}/nueva-conversacion", tags=["Asistente público"])
async def asistente_nueva_conversacion(token: str, req: AsistenteNuevaConversacionRequest):
    """Crea una sesión nueva (sin arrastrar la memoria de la anterior) para
    el mismo asistente/curso/usuario — botón "Nueva conversación"."""
    asistente = await db.run(assistants.get_asistente_by_token, token)
    if not asistente:
        raise HTTPException(404, "Este asistente no existe o no está activo.")

    collection_name = await db.run(rds.resolve_collection, asistente["rd_id"], req.courseid)
    if not collection_name:
        raise HTTPException(404, f"No hay ninguna colección asignada al curso {req.courseid} en esta RD.")

    loop = asyncio.get_event_loop()
    fullname = await loop.run_in_executor(None, get_moodle_user_fullname, req.userid)

    sesion = await db.run(
        assistants.get_or_create_sesion,
        asistente["id"], req.courseid, req.userid, None, fullname, collection_name,
        True,
    )
    return {"sesion_id": sesion["id"], "saludo_nombre": fullname or ""}


@app.get("/asistente/{token}/conversaciones", tags=["Asistente público"])
async def asistente_listar_conversaciones(token: str, courseid: int = Query(...), userid: int = Query(...)):
    """Sidebar de historial: todas las conversaciones de este alumno con
    este asistente en este curso, más recientes primero, con vista previa."""
    asistente = await db.run(assistants.get_asistente_by_token, token)
    if not asistente:
        raise HTTPException(404, "Este asistente no existe o no está activo.")
    sesiones = await db.run(assistants.list_sesiones, asistente["id"], courseid, userid)
    return {"conversaciones": sesiones}


@app.get("/asistente/{token}/conversaciones/{sesion_id}", tags=["Asistente público"])
async def asistente_ver_conversacion(
    token: str, sesion_id: int, courseid: int = Query(...), userid: int = Query(...)
):
    """Mensajes de una conversación anterior — valida que pertenezca a
    este asistente/curso/usuario antes de devolverlos (mismo criterio de
    acceso que el resto de la ruta pública: el token + los parámetros)."""
    asistente = await db.run(assistants.get_asistente_by_token, token)
    if not asistente:
        raise HTTPException(404, "Este asistente no existe o no está activo.")
    sesion = await db.run(assistants.get_sesion, sesion_id)
    if (
        not sesion
        or sesion["asistente_id"] != asistente["id"]
        or sesion["moodle_courseid"] != courseid
        or sesion["moodle_userid"] != userid
    ):
        raise HTTPException(404, "Conversación no encontrada.")
    historial = await db.run(assistants.get_mensajes, sesion_id)
    return {"sesion_id": sesion_id, "historial": historial}


@app.get("/api/jobs/{job_id}", tags=["Compartido"])
async def job_status(job_id: str):
    """
    Estado de un job:
    - "pendiente_confirmacion" → mostrar resumen al usuario
    - "procesando"             → mostrar spinner
    - "completado"             → mostrar resultado
    - "error"                  → mostrar error
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return {
        "job_id":  job_id,
        "status":  job["status"],
        "message": job.get("message"),
        "result":  job.get("result"),
    }


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE SEMIAUTOMÁTICO  /api/semi/*
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/semi/upload", tags=["Semiautomático"])
async def semi_upload(
    collection_name: str = Query(..., description="Colección destino"),
    model_name: Optional[str] = Query(
        None,
        description=(
            "Modelo de embeddings en español a usar. "
            "Consulta GET /api/models para ver el catálogo. "
            "Si no se indica, se usa el modelo por defecto del servidor."
        ),
    ),
    source_type: Optional[str] = Query(
        None,
        description=(
            "'curso_propio' (default) o 'bibliografia'. Solo tiene efecto "
            "si la colección destino es híbrida — determina la prioridad "
            "que el tutor le da a este contenido en /api/rag/answer."
        ),
    ),
    files: List[UploadFile] = File(...),
):
    """
    Paso 1 y 2 del asistente semiautomático.
    Sube archivos PDF/DOCX, los guarda en temp y los valida.
    Retorna job_id + resumen de validación por archivo.
    Usuario debe confirmar antes de vectorizar.
    """
    if not files:
        raise HTTPException(400, "No se enviaron archivos")
    if len(files) > 20:
        raise HTTPException(400, "Máximo 20 archivos por lote")

    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe en Qdrant")

    # Validación de compatibilidad de dimensiones antes de procesar archivos
    try:
        service.validate_model_for_collection(collection_name)
    except ValueError as e:
        raise HTTPException(422, str(e))

    files_data = []
    errors = []

    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            errors.append({"filename": upload.filename, "error": f"Extensión {ext} no permitida"})
            continue

        content = await upload.read()
        if len(content) > settings.MAX_FILE_SIZE:
            errors.append({"filename": upload.filename, "error": "Archivo demasiado grande (máx 50 MB)"})
            continue

        temp_path, original_name = _save_upload_to_temp(upload, content)
        files_data.append({"filename": original_name, "path": temp_path})

    if not files_data and errors:
        raise HTTPException(400, detail={"errors": errors})

    result = await SemiAutoCore.validate_uploads(
        files_data, collection_name, model_name=resolved_model, source_type=source_type
    )
    result["upload_errors"] = errors
    return result


@app.post("/api/semi/confirm/{job_id}", tags=["Semiautomático"])
async def semi_confirm(job_id: str, background_tasks: BackgroundTasks):
    """
    Paso 3: el usuario confirma → vectorización en background.
    La interfaz consulta GET /api/jobs/{job_id} hasta status = 'completado'.
    El modelo usado es el que se guardó al hacer el upload.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    if job["status"] != "pendiente_confirmacion":
        raise HTTPException(400, f"El job está en estado '{job['status']}', no puede confirmarse")

    job["status"] = "procesando"
    background_tasks.add_task(SemiAutoCore.confirm_and_vectorize, job_id)

    return {
        "job_id":          job_id,
        "status":          "procesando",
        "embedding_model": job["context"].get("model_name", settings.EMBEDDING_MODEL),
    }


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE AUTOMÁTICO  /api/auto/*
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/auto/preview", tags=["Automático"])
async def auto_preview(
    curid: int = Query(..., description="ID del curso en Moodle"),
    collection_name: str = Query(..., description="Colección destino"),
    model_name: Optional[str] = Query(
        None,
        description="Modelo de embeddings a usar. Consulta GET /api/models.",
    ),
):
    """
    Consulta Moodle con el curid y lista los módulos con sus PDFs.
    Marca cuáles ya están vectorizados en la colección (duplicado).
    Usuario selecciona cuáles quiere vectorizar.
    """
    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe en Qdrant")

    try:
        service.validate_model_for_collection(collection_name)
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        return AutoCore.preview_course(curid, collection_name, model_name=resolved_model)
    except Exception as e:
        raise HTTPException(502, f"Error consultando Moodle: {e}")


@app.post("/api/auto/download-validate", tags=["Automático"])
async def auto_download_validate(
    req: AutoVectorizeRequest,
    model_name: Optional[str] = Query(
        None,
        description="Modelo de embeddings a usar. Consulta GET /api/models.",
    ),
):
    """
    Descarga los PDFs seleccionados a temp, los valida
    y retorna job_id + resumen.
    Usuario debe confirmar antes de vectorizar.
    """
    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(req.collection_name):
        raise HTTPException(404, f"Colección '{req.collection_name}' no existe en Qdrant")

    if not req.selected_filenames:
        raise HTTPException(400, "No se seleccionaron archivos")

    try:
        service.validate_model_for_collection(req.collection_name)
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        return await AutoCore.download_and_validate(
            curid=req.curid,
            collection_name=req.collection_name,
            selected_filenames=req.selected_filenames,
            model_name=resolved_model,
            source_type=req.source_type,
            file_source_types=req.file_source_types,
        )
    except Exception as e:
        raise HTTPException(502, f"Error en descarga/validación: {e}")


@app.post("/api/auto/confirm/{job_id}", tags=["Automático"])
async def auto_confirm(job_id: str, background_tasks: BackgroundTasks):
    """
    Usuario confirma → vectorización en background.
    Consultar GET /api/jobs/{job_id} para el estado.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    if job["status"] != "pendiente_confirmacion":
        raise HTTPException(400, f"El job está en estado '{job['status']}'")

    job["status"] = "procesando"
    background_tasks.add_task(AutoCore.confirm_and_vectorize, job_id)

    return {
        "job_id":          job_id,
        "status":          "procesando",
        "embedding_model": job["context"].get("model_name", settings.EMBEDDING_MODEL),
    }


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE DE ACTUALIZACIÓN  /api/update/*
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/update/list", tags=["Actualización"])
async def update_list(
    collection_name: str = Query(..., description="Colección a consultar"),
    model_name: Optional[str] = Query(None, description="Modelo de embeddings a usar."),
):
    """
    Lista los documentos vectorizados en la colección.
    Usuario selecciona cuál quiere reemplazar.
    """
    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe en Qdrant")

    docs = UpdateCore.list_vectorized(collection_name, model_name=resolved_model)
    return {
        "collection":      collection_name,
        "embedding_model": resolved_model,
        "documents":       docs,
    }


@app.post("/api/update/prepare-manual", tags=["Actualización"])
async def update_prepare_manual(
    collection_name: str = Query(...),
    filename_to_replace: str = Query(..., description="Filename del doc a reemplazar en Qdrant"),
    model_name: Optional[str] = Query(None, description="Modelo de embeddings a usar."),
    source_type: Optional[str] = Query(None, description="'curso_propio' (default) o 'bibliografia'."),
    file: UploadFile = File(...),
):
    """
    Modo manual: el usuario sube el archivo nuevo.
    Se guarda en temp, se valida y se retorna job_id + resumen.
    """
    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe en Qdrant")

    try:
        service.validate_model_for_collection(collection_name)
    except ValueError as e:
        raise HTTPException(422, str(e))

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Extensión {ext} no permitida")

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(400, "Archivo demasiado grande (máx 50 MB)")

    temp_path, original_name = _save_upload_to_temp(file, content)

    try:
        return await UpdateCore.prepare_manual(
            file_path=temp_path,
            original_filename=original_name,
            filename_to_replace=filename_to_replace,
            collection_name=collection_name,
            model_name=resolved_model,
            source_type=source_type,
        )
    except ValueError as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Error validando archivo: {e}")


@app.post("/api/update/prepare-moodle", tags=["Actualización"])
async def update_prepare_moodle(
    collection_name: str = Query(...),
    filename_to_replace: str = Query(..., description="Filename del doc a reemplazar en Qdrant"),
    curid: int = Query(...),
    moodle_filename: str = Query(..., description="Filename del PDF en Moodle"),
    model_name: Optional[str] = Query(None, description="Modelo de embeddings a usar."),
    source_type: Optional[str] = Query(None, description="'curso_propio' (default) o 'bibliografia'."),
):
    """
    Modo Moodle: descarga el archivo desde Moodle,
    lo guarda en temp, lo valida y retorna job_id + resumen.
    """
    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe en Qdrant")

    try:
        service.validate_model_for_collection(collection_name)
    except ValueError as e:
        raise HTTPException(422, str(e))

    try:
        return await UpdateCore.prepare_from_moodle(
            curid=curid,
            moodle_filename=moodle_filename,
            filename_to_replace=filename_to_replace,
            collection_name=collection_name,
            model_name=resolved_model,
            source_type=source_type,
        )
    except ValueError as e:
        msg = str(e)
        code = 404 if "no encontrado en curid" in msg else 422
        raise HTTPException(code, msg)
    except Exception as e:
        print("ERROR REAL EN PREPARE_MOODLE:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error interno procesando archivo: {str(e)}"
        )


@app.get("/api/update/preview-modules", tags=["Actualización"])
async def update_preview_modules(
    curid: int = Query(...),
    collection_name: str = Query(...),
    model_name: Optional[str] = Query(None, description="Modelo de embeddings a usar."),
):
    """
    Lista los módulos y PDFs del curid para que el usuario
    elija cuál descargará como reemplazo (modo Moodle de actualización).
    """
    resolved_model = _resolve_model(model_name)
    try:
        return AutoCore.preview_course(curid, collection_name, model_name=resolved_model)
    except Exception as e:
        raise HTTPException(502, f"Error consultando Moodle: {e}")


@app.post("/api/update/confirm/{job_id}", tags=["Actualización"])
async def update_confirm(job_id: str, background_tasks: BackgroundTasks):
    """
    El usuario confirma la actualización → background task:
    elimina el doc anterior y vectoriza el nuevo con el modelo del job.
    Consultar GET /api/jobs/{job_id} para el estado.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    if job["status"] != "pendiente_confirmacion":
        raise HTTPException(400, f"El job está en estado '{job['status']}'")

    job["status"] = "procesando"
    background_tasks.add_task(UpdateCore.confirm_and_vectorize, job_id)

    return {
        "job_id":          job_id,
        "status":          "procesando",
        "embedding_model": job["context"].get("model_name", settings.EMBEDDING_MODEL),
    }


# ═══════════════════════════════════════════════════════════════════
# BÚSQUEDA (utilitario)
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/search", tags=["Búsqueda"])
async def search(
    query: str = Query(...),
    collection_name: str = Query(...),
    top_k: int = Query(3, ge=1, le=20),
    model_name: Optional[str] = Query(
        None,
        description=(
            "Modelo para vectorizar la consulta. Debe coincidir con el modelo "
            "usado al vectorizar los documentos de esta colección."
        ),
    ),
):
    """
    Búsqueda semántica en una colección.
    El modelo usado para la query debe ser el mismo con el que se
    vectorizaron los documentos (mismas dimensiones).
    """
    resolved_model = _resolve_model(model_name)
    service = get_service(resolved_model)

    if not service.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe")

    try:
        service.validate_model_for_collection(collection_name)
    except ValueError as e:
        raise HTTPException(422, str(e))

    from qdrant_admin import DENSE_VECTOR_NAME, get_qdrant_admin

    vector = await service.embed([query], is_query=True)
    query_kwargs = {"collection_name": collection_name, "query": vector[0], "limit": top_k}
    if get_qdrant_admin().get_vector_schema(collection_name) == "hybrid":
        # Colecciones híbridas usan vectores con nombre (dense/sparse) — sin
        # `using`, Qdrant no sabe contra cuál vector buscar y responde 400.
        query_kwargs["using"] = DENSE_VECTOR_NAME
    hits = service._client.query_points(**query_kwargs).points
    return {
        "query":           query,
        "collection":      collection_name,
        "embedding_model": resolved_model,
        "results": [
            {
                "score":           round(h.score, 4),
                "text":            h.payload.get("text"),
                "filename":        h.payload.get("filename"),
                "course_name":     h.payload.get("course_name"),
                "chunk":           h.payload.get("chunk"),
                "embedding_model": h.payload.get("embedding_model"),
            }
            for h in hits
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN RAG (Strategy Pattern)  /api/config/*
# ═══════════════════════════════════════════════════════════════════

_CONFIG_STAGES = list(STAGE_CATALOGS.keys())


class StageConfigRequest(BaseModel):
    strategy_name: str


# ── Claves de API (registradas ANTES de /api/config/{stage}: Starlette
# resuelve por orden de registro, y "credentials" coincidiría con el
# parámetro {stage} si el genérico se registrara primero) ──

class SetCredentialRequest(BaseModel):
    value: str


class TestCredentialRequest(BaseModel):
    value: Optional[str] = None


@app.get("/api/config/credentials", tags=["Configuración RAG"])
async def get_credentials():
    """Estado de las credenciales (API keys) configuradas. Nunca expone el
    valor completo de un secreto — solo si está configurado y sus últimos
    caracteres (masked_value)."""
    return {"credentials": credentials.list_credentials()}


@app.post("/api/config/credentials/{field}", tags=["Configuración RAG"])
async def set_credential_endpoint(field: str, req: SetCredentialRequest):
    """Guarda una credencial: actualiza settings en memoria (efecto
    inmediato) y persiste en .env (sobrevive un reinicio del servidor)."""
    try:
        return credentials.set_credential(field, req.value)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/config/credentials/{field}/test", tags=["Configuración RAG"])
async def test_credential_endpoint(field: str, req: TestCredentialRequest):
    """Prueba la conexión real contra el proveedor correspondiente, usando
    `value` si se manda (prueba temporal, sin guardar) o el valor ya
    guardado en settings si no."""
    try:
        return await credentials.test_credential(field, req.value)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/config/{stage}", tags=["Configuración RAG"])
async def get_stage_config(stage: str):
    """
    Selección actual y catálogo completo de una etapa del pipeline RAG.
    `stage` ∈ etl_document | etl_audio | contextual | dense | sparse |
    rerank | generation.
    """
    if stage not in _CONFIG_STAGES:
        raise HTTPException(404, f"Etapa '{stage}' desconocida. Válidas: {_CONFIG_STAGES}")

    current = strategy_registry.resolve_strategy_name(stage)
    catalog = STAGE_CATALOGS[stage]

    return {
        "stage": stage,
        "current": current,
        "options": [
            {
                "name": name,
                **info,
                "usable": strategy_is_usable(stage, name),
            }
            for name, info in catalog.items()
        ],
    }


@app.post("/api/config/{stage}", tags=["Configuración RAG"])
async def set_stage_config(stage: str, req: StageConfigRequest):
    """Cambia la estrategia activa de una etapa (override en memoria).
    Afecta automáticamente la próxima vectorización de los 3 mecanismos
    contra colecciones híbridas, y las próximas consultas de /api/rag/*."""
    if stage not in _CONFIG_STAGES:
        raise HTTPException(404, f"Etapa '{stage}' desconocida. Válidas: {_CONFIG_STAGES}")

    try:
        get_strategy_info(stage, req.strategy_name)  # valida que exista
    except ValueError as e:
        raise HTTPException(400, str(e))

    get_runtime_config().set(stage, req.strategy_name)

    return {
        "stage": stage,
        "current": req.strategy_name,
        "message": f"Estrategia de '{stage}' actualizada a '{req.strategy_name}'",
    }


# ═══════════════════════════════════════════════════════════════════
# RAG: recuperación híbrida + reranking + generación  /api/rag/*
# ═══════════════════════════════════════════════════════════════════

class RagAnswerRequest(BaseModel):
    collection_name: str
    query: str
    dense_strategy: Optional[str] = None
    sparse_strategy: Optional[str] = None
    rerank_strategy: Optional[str] = None
    generation_strategy: Optional[str] = None
    top_n: Optional[int] = None
    asistente_id: Optional[int] = None


@app.get("/api/rag/retrieve", tags=["RAG"])
async def rag_retrieve(
    query: str = Query(...),
    collection_name: str = Query(...),
    top_n: int = Query(5, ge=1, le=20),
    dense_strategy: Optional[str] = Query(None),
    sparse_strategy: Optional[str] = Query(None),
    rerank_strategy: Optional[str] = Query(None),
):
    """
    Recuperación híbrida + reranking + boosting a `curso_propio`, sin
    generación (útil para depurar/previsualizar qué se recuperaría).
    Requiere una colección híbrida (vectores nombrados dense+sparse) —
    usa Gestión Qdrant para crear una si la colección es legacy.
    """
    from qdrant_admin import get_qdrant_admin

    admin = get_qdrant_admin()
    if not admin.collection_exists(collection_name):
        raise HTTPException(404, f"Colección '{collection_name}' no existe")
    if admin.get_vector_schema(collection_name) != "hybrid":
        raise HTTPException(
            422,
            f"La colección '{collection_name}' es legacy (vector único). "
            f"La búsqueda híbrida requiere una colección con esquema 'hybrid' "
            f"— créala o elige otra desde Gestión Qdrant.",
        )

    try:
        candidates = await rag_pipeline.retrieve_rerank_boost(
            collection_name=collection_name,
            query=query,
            dense_strategy_name=dense_strategy,
            sparse_strategy_name=sparse_strategy,
            rerank_strategy_name=rerank_strategy,
            top_n=top_n,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        "query": query,
        "collection": collection_name,
        "results": candidates,
    }


@app.post("/api/rag/answer", tags=["RAG"])
async def rag_answer(req: RagAnswerRequest):
    """
    Pipeline completo del tutor: recuperación híbrida → rerank → boost →
    generación con el system prompt fijo. Usado por "Búsqueda semántica" y
    por "Prueba de Asistente" (si `asistente_id` viene informado, se
    agrega su `prompt_maestro` — misma lógica que la ruta pública).
    """
    from qdrant_admin import get_qdrant_admin

    admin = get_qdrant_admin()
    if not admin.collection_exists(req.collection_name):
        raise HTTPException(404, f"Colección '{req.collection_name}' no existe")
    if admin.get_vector_schema(req.collection_name) != "hybrid":
        raise HTTPException(
            422,
            f"La colección '{req.collection_name}' es legacy (vector único). "
            f"La generación con el tutor requiere una colección con esquema "
            f"'hybrid' — créala o elige otra desde Gestión Qdrant.",
        )

    extra_system_prompt = None
    dense_strategy = req.dense_strategy
    sparse_strategy = req.sparse_strategy
    rerank_strategy = req.rerank_strategy
    generation_strategy = req.generation_strategy
    if req.asistente_id is not None:
        asistente = await db.run(assistants.get_asistente, req.asistente_id)
        if not asistente:
            raise HTTPException(404, f"El asistente {req.asistente_id} no existe")
        vinculo = await db.run(
            db.fetch_one,
            "SELECT 1 FROM colecciones_rd WHERE rd_id = %s AND qdrant_collection_name = %s",
            (asistente["rd_id"], req.collection_name),
        )
        if not vinculo:
            raise HTTPException(
                400,
                f"La colección '{req.collection_name}' no está vinculada a la RD de este asistente.",
            )
        extra_system_prompt = asistente.get("prompt_maestro")
        # La config propia del asistente aplica salvo que el request la
        # override explícitamente (así "Prueba de Asistente" prueba
        # exactamente lo que respondería en público). dense/sparse son el
        # embedding maestro de la RD, no del asistente.
        rd = await db.run(rds.get_rd, asistente["rd_id"]) or {}
        dense_strategy = dense_strategy or rd.get("dense_strategy")
        sparse_strategy = sparse_strategy or rd.get("sparse_strategy")
        rerank_strategy = rerank_strategy or asistente.get("rerank_strategy")
        generation_strategy = generation_strategy or asistente.get("generation_strategy")

    try:
        return await rag_pipeline.answer_query(
            collection_name=req.collection_name,
            query=req.query,
            dense_strategy_name=dense_strategy,
            sparse_strategy_name=sparse_strategy,
            rerank_strategy_name=rerank_strategy,
            generation_strategy_name=generation_strategy,
            top_n=req.top_n,
            extra_system_prompt=extra_system_prompt,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Error generando respuesta: {e}")


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🌐  SISTEMA DE VECTORIZACIÓN DOCUMENTAL  v2")
    print("=" * 65)
    print(f"📍 Interfaz : http://localhost:{settings.API_PORT}/")
    print(f"📖 API Docs : http://localhost:{settings.API_PORT}/docs")
    print(f"🗄️  Qdrant  : {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"🤖 Modelo   : {settings.EMBEDDING_MODEL}")
    print("=" * 65 + "\n")
    uvicorn.run(
        "app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )
