"""
app.py
======
FastAPI — un solo puerto (8100), tres asistentes:
  · /api/semi/*    → asistente semiautomático
  · /api/auto/*    → asistente automático (Moodle)
  · /api/update/*  → asistente de actualización
  · /api/          → endpoints compartidos (colecciones, job status, modelos)
"""

import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel
from fastapi import (
    FastAPI, UploadFile, File, HTTPException,
    BackgroundTasks, Query, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from settings import (
    settings, validate_settings, AVAILABLE_EMBEDDING_MODELS,
    STAGE_CATALOGS, get_strategy_info, strategy_is_usable,
)

from definitions import (get_service,
    AutoPreviewRequest, AutoVectorizeRequest,
    UpdateVectorizeRequest,
    _sanitize_filename,
)
from core import (
    SemiAutoCore, AutoCore, UpdateCore,
    get_job, job_store,
)
import rag_pipeline
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

Path("frontend").mkdir(exist_ok=True)
templates = Jinja2Templates(directory="frontend")


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
# INTERFAZ WEB
# ═══════════════════════════════════════════════════════════════════

@app.get("/vectorizacion", response_class=HTMLResponse, tags=["Interfaz"])
async def home(request: Request):
    return templates.TemplateResponse("interfaz.html", {"request": request})


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

        return {"collections": get_qdrant_admin().list_collections()}
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
    compatible con el flujo anterior)."""
    try:
        return get_qdrant_admin().create_collection(
            name=req.name,
            description=req.description,
            vector_schema=req.vector_schema,
            dense_size=req.dense_size,
            distance=req.distance,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


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


@app.delete("/api/collections/{collection_name}", tags=["Gestión Qdrant"])
async def qdrant_delete_collection(
    collection_name: str,
    force: bool = Query(False, description="Forzar eliminación aunque tenga vectores"),
):
    """Elimina una colección de Qdrant."""
    try:
        return get_qdrant_admin().delete_collection(collection_name, force=force)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


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
        files_data, collection_name, model_name=resolved_model
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

    vector = await service.embed([query], is_query=True)
    hits = service._client.query_points(
        collection_name=collection_name,
        query=vector[0],
        limit=top_k,
    ).points
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
    generación con el system prompt fijo. Usado por "Pruebas del LLM".
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

    try:
        return await rag_pipeline.answer_query(
            collection_name=req.collection_name,
            query=req.query,
            dense_strategy_name=req.dense_strategy,
            sparse_strategy_name=req.sparse_strategy,
            rerank_strategy_name=req.rerank_strategy,
            generation_strategy_name=req.generation_strategy,
            top_n=req.top_n,
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
