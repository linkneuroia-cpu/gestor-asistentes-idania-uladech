"""
core.py
=======
Flujo de negocio de los tres asistentes.
Cada clase orquesta su propio pipeline usando las funciones
y servicios definidos en definitions.py.

Clases:
  · SemiAutoCore   — asistente semiautomático (upload manual)
  · AutoCore       — asistente automático (Moodle + curid)
  · UpdateCore     — asistente de actualización (Qdrant → reemplazar)
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from settings import settings

from definitions import (
    get_service,
    validate_document,
    vectorize_file_with_pages,
    get_course_resources,
    download_pdf_to_temp,
    download_youtube_audio,
    cleanup_temp_file,
    classify_file_type,
    DocumentInfo,
    ValidationSummary,
)


# ═══════════════════════════════════════════════════════════════════
# ESTADO DE JOBS EN MEMORIA
# job_store[job_id] = {
#   "status":  "procesando" | "pendiente_confirmacion" | "completado" | "error",
#   "message": str,
#   "result":  dict | None,
#   "context": dict,   ← datos que necesita la confirmación
# }
# ═══════════════════════════════════════════════════════════════════

job_store: Dict[str, Dict[str, Any]] = {}


def _new_job(context: dict = None) -> str:
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status":  "procesando",
        "message": None,
        "result":  None,
        "context": context or {},
    }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return job_store.get(job_id)


def _complete_job(job_id: str, result: dict) -> None:
    job_store[job_id]["status"]  = "completado"
    job_store[job_id]["result"]  = result
    job_store[job_id]["message"] = f"✅ {result.get('vectors_stored', 0)} vectores almacenados"


def _fail_job(job_id: str, error: str) -> None:
    job_store[job_id]["status"]  = "error"
    job_store[job_id]["message"] = f"❌ {error}"


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE SEMIAUTOMÁTICO
# ═══════════════════════════════════════════════════════════════════

class SemiAutoCore:
    """
    Flujo:
      1. Usuario sube PDFs/DOCX y elige un modelo de embeddings.
      2. Se guardan en temp_docs/.
      3. Se valida estructura + check duplicado por filename.
      4. Se devuelve resumen de validación al usuario.
      5. Usuario confirma → se vectoriza con el mismo modelo + se limpia temp.
    """

    @staticmethod
    async def validate_uploads(
        files_data: List[Dict[str, Any]],
        collection_name: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recibe lista de dicts: {"filename": str, "path": Path}
        (archivos ya guardados en temp por el endpoint).
        Retorna un job_id con el contexto y la lista de ValidationSummary.
        """
        service = get_service(model_name)
        summaries = []

        for fd in files_data:
            file_path: Path = fd["path"]
            filename: str   = fd["filename"]

            try:
                val = await validate_document(str(file_path))
                dup = service.check_duplicate(filename, collection_name)

                summaries.append(ValidationSummary(
                    filename=filename,
                    file_type=val["file_type"],
                    total_pages=val["total_pages"],
                    total_images=val["total_images"],
                    total_tables=val["total_tables"],
                    duration_seconds=val.get("duration_seconds"),
                    has_non_textual=(val["total_images"] > 0 or val["total_tables"] > 0),
                    warning_message=val["warning_message"],
                    is_duplicate=dup["exists"],
                    existing_chunks=dup["total_chunks"],
                ))
            except Exception as e:
                summaries.append(ValidationSummary(
                    filename=filename,
                    file_type=classify_file_type(filename),
                    total_pages=0,
                    total_images=0,
                    total_tables=0,
                    has_non_textual=False,
                    warning_message=f"Error al validar: {e}",
                ))
              
        job_id = _new_job(context={
            "mode":            "semi",
            "collection_name": collection_name,
            "model_name":      model_name or settings.EMBEDDING_MODEL,
            "files": [
                {"filename": fd["filename"], "path": str(fd["path"])}
                for fd in files_data
            ],
            "summaries": [s.dict() for s in summaries],
        })

        job_store[job_id]["status"] = "pendiente_confirmacion"

        return {
            "job_id":        job_id,
            "embedding_model": model_name or settings.EMBEDDING_MODEL,
            "summaries":     [s.dict() for s in summaries],
        }

    @staticmethod
    async def confirm_and_vectorize(job_id: str) -> None:
        """
        Tarea en background: vectoriza todos los archivos del job
        usando el modelo guardado en el context.
        Si es duplicado → elimina primero, luego vectoriza.
        Al terminar cada archivo → limpia temp.
        """
        job = job_store.get(job_id)
        if not job:
            return

        job["status"] = "procesando"
        ctx = job["context"]
        collection_name: str = ctx["collection_name"]
        model_name: str      = ctx.get("model_name", settings.EMBEDDING_MODEL)

        # Recupera (o crea) el servicio con el modelo guardado en el context
        service = get_service(model_name)
        results = []

        for fd in ctx["files"]:
            file_path = Path(fd["path"])
            filename  = fd["filename"]

            saved_val = next(
                (s for s in ctx["summaries"] if s["filename"] == filename), {}
            )
            total_pages = saved_val.get("total_pages", 0)

            try:
                dup = service.check_duplicate(filename, collection_name)
                if dup["exists"]:
                    service.delete_document(filename, collection_name)
                    print(f"♻️ Reemplazando: {filename}")

                result = await vectorize_file_with_pages(
                    file_path=file_path,
                    collection_name=collection_name,
                    course_name="modo manual",
                    total_pages=total_pages,
                    service=service,
                    original_filename=filename,
                )
                results.append({"filename": filename, **result})

            except Exception as e:
                print(f"❌ Error vectorizando {filename}: {e}")
                results.append({"filename": filename, "success": False, "error": str(e)})
            finally:
                cleanup_temp_file(file_path)

        _complete_job(job_id, {
            "vectors_stored":  sum(r.get("vectors_stored", 0) for r in results),
            "embedding_model": model_name,
            "files":           results,
        })


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE AUTOMÁTICO
# ═══════════════════════════════════════════════════════════════════

class AutoCore:
    """
    Flujo:
      1. Usuario ingresa curid y elige un modelo de embeddings.
      2. Se consulta Moodle API → lista de módulos con sus PDFs.
      3. Se muestra al usuario (con flag de duplicado por filename).
      4. Usuario selecciona cuáles vectorizar.
      5. Se descargan a temp → validar → resumen → usuario confirma.
      6. Vectorizar con el modelo elegido → limpiar temp.
    """

    @staticmethod
    def preview_course(
        curid: int,
        collection_name: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Consulta Moodle y retorna módulos con sus PDFs, incluyendo si cada PDF ya existe en la colección (duplicado).
        """
        service = get_service(model_name)
        resources = get_course_resources(curid)

        modules = []
        for resource in resources:
            module_name = resource.get("name", "Sin nombre")
            files = []
            for cf in resource.get("contentfiles", []):
                filename = cf["filename"]
                dup = service.check_duplicate(filename, collection_name)
                files.append(DocumentInfo(
                    filename=filename,
                    filesize=cf.get("filesize"),
                    fileurl=cf.get("fileurl"),
                    course_name=module_name,
                    file_type=classify_file_type(filename),
                    is_duplicate=dup["exists"],
                    existing_chunks=dup["total_chunks"],
                ).dict())

            if files:
                modules.append({
                    "module_name": module_name,
                    "resource_id": resource.get("id"),
                    "files": files,
                })

        return {
            "curid":   curid,
            "modules": modules,
            "total_files": sum(len(m["files"]) for m in modules),
        }

    @staticmethod
    async def download_and_validate(
        curid: int,
        collection_name: str,
        selected_filenames: List[str],
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Descarga los PDFs seleccionados a temp_docs/,
        los valida y retorna resumen + job_id para confirmación.
        """
        service = get_service(model_name)
        resources = get_course_resources(curid)

        file_map: Dict[str, Dict] = {}
        for resource in resources:
            module_name = resource.get("name", "")
            for cf in resource.get("contentfiles", []):
                file_map[cf["filename"]] = {
                    "fileurl":     cf["fileurl"],
                    "course_name": module_name,
                    "source":      cf.get("source"),
                }

        summaries = []
        downloaded: List[Dict] = []

        for filename in selected_filenames:
            if filename not in file_map:
                summaries.append(ValidationSummary(
                    filename=filename,
                    total_pages=0, total_images=0, total_tables=0,
                    has_non_textual=False,
                    warning_message=f"No encontrado en curid {curid}",
                ).dict())
                continue

            info = file_map[filename]
            try:
                if info.get("source") == "youtube":
                    temp_path = download_youtube_audio(info["fileurl"], filename)
                else:
                    temp_path = download_pdf_to_temp(info["fileurl"], filename)
                val = await validate_document(str(temp_path))
                dup = service.check_duplicate(filename, collection_name)

                summaries.append(ValidationSummary(
                    filename=filename,
                    file_type=val["file_type"],
                    total_pages=val["total_pages"],
                    total_images=val["total_images"],
                    total_tables=val["total_tables"],
                    duration_seconds=val.get("duration_seconds"),
                    has_non_textual=(
                        val["total_images"] > 0 or val["total_tables"] > 0
                    ),
                    warning_message=val["warning_message"],
                    is_duplicate=dup["exists"],
                    existing_chunks=dup["total_chunks"],
                ).dict())

                downloaded.append({
                    "filename":    filename,
                    "path":        str(temp_path),
                    "course_name": info["course_name"],
                    "total_pages": val["total_pages"],
                })

            except Exception as e:
                summaries.append(ValidationSummary(
                    filename=filename,
                    total_pages=0, total_images=0, total_tables=0,
                    has_non_textual=False,
                    warning_message=f"Error al descargar/validar: {e}",
                ).dict())

        job_id = _new_job(context={
            "mode":            "auto",
            "collection_name": collection_name,
            "model_name":      model_name or settings.EMBEDDING_MODEL,
            "files":           downloaded,
            "summaries":       summaries,
        })
        job_store[job_id]["status"] = "pendiente_confirmacion"

        return {
            "job_id":          job_id,
            "embedding_model": model_name or settings.EMBEDDING_MODEL,
            "summaries":       summaries,
        }

    @staticmethod
    async def confirm_and_vectorize(job_id: str) -> None:
        """
        Background task: vectoriza los archivos descargados en temp
        usando el modelo guardado en el context.
        Si duplicado → elimina primero. Al terminar → limpia temp.
        """
        job = job_store.get(job_id)
        if not job:
            return

        job["status"] = "procesando"
        ctx = job["context"]
        collection_name = ctx["collection_name"]
        model_name: str = ctx.get("model_name", settings.EMBEDDING_MODEL)

        service = get_service(model_name)
        results = []

        for fd in ctx["files"]:
            file_path   = Path(fd["path"])
            filename    = fd["filename"]
            course_name = fd.get("course_name", "")
            total_pages = fd.get("total_pages", 0)

            try:
                dup = service.check_duplicate(filename, collection_name)
                if dup["exists"]:
                    service.delete_document(filename, collection_name)
                    print(f"♻️ Reemplazando: {filename}")

                result = await vectorize_file_with_pages(
                    file_path=file_path,
                    collection_name=collection_name,
                    course_name=course_name,
                    total_pages=total_pages,
                    service=service,
                    original_filename=filename,
                )
                results.append({"filename": filename, **result})

            except Exception as e:
                print(f"❌ Error vectorizando {filename}: {e}")
                results.append({"filename": filename, "success": False, "error": str(e)})
            finally:
                cleanup_temp_file(file_path)

        _complete_job(job_id, {
            "vectors_stored":  sum(r.get("vectors_stored", 0) for r in results),
            "embedding_model": model_name,
            "files":           results,
        })


# ═══════════════════════════════════════════════════════════════════
# ASISTENTE DE ACTUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════

class UpdateCore:
    """
    Flujo:
      1. Listar documentos vectorizados en la colección (desde Qdrant).
      2. Usuario selecciona cuál reemplazar y elige el modelo.
      3a. Modo manual: sube el archivo nuevo → temp.
      3b. Modo Moodle: ingresa curid → lista módulos → selecciona → descarga → temp.
      4. Validar + resumen.
      5. Usuario confirma → delete_old → vectorize_new (mismo modelo) → limpiar temp.
    """

    @staticmethod
    def list_vectorized(
        collection_name: str,
        model_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lista los documentos vectorizados en la colección."""
        service = get_service(model_name)
        if not service.collection_exists(collection_name):
            return []
        return service.list_documents_in_collection(collection_name)

    @staticmethod
    async def prepare_manual(
        file_path: Path,
        filename_to_replace: str,
        collection_name: str,
        original_filename: str = "",
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        El archivo nuevo ya fue guardado en temp por el endpoint.
        Valida y prepara el job para confirmación.
        """
        service = get_service(model_name)

        stored_filename = original_filename if original_filename else file_path.name

        if stored_filename != filename_to_replace:
            dup_new = service.check_duplicate(stored_filename, collection_name)
            if dup_new["exists"]:
                cleanup_temp_file(file_path)
                raise ValueError(
                    f"El archivo '{stored_filename}' ya existe en la colección '{collection_name}'. "
                    f"No se puede usar ese nombre como reemplazo porque crearía un duplicado. "
                    f"Use el mismo nombre que el archivo a reemplazar ('{filename_to_replace}') "
                    f"o un nombre que no exista en la colección."
                )

        val = await validate_document(str(file_path))

        job_id = _new_job(context={
            "mode":                "update",
            "collection_name":     collection_name,
            "model_name":          model_name or settings.EMBEDDING_MODEL,
            "filename_to_replace": filename_to_replace,
            "file": {
                "filename":    stored_filename,
                "path":        str(file_path),
                "course_name": "modo manual",
                "total_pages": val["total_pages"],
            },
        })
        job_store[job_id]["status"] = "pendiente_confirmacion"

        return {
            "job_id":          job_id,
            "embedding_model": model_name or settings.EMBEDDING_MODEL,
            "summary": ValidationSummary(
                filename=stored_filename,
                file_type=val["file_type"],
                total_pages=val["total_pages"],
                total_images=val["total_images"],
                total_tables=val["total_tables"],
                duration_seconds=val.get("duration_seconds"),
                has_non_textual=(val["total_images"] > 0 or val["total_tables"] > 0),
                warning_message=val["warning_message"],
                is_duplicate=True,
                existing_chunks=service.check_duplicate(
                    filename_to_replace, collection_name
                )["total_chunks"],
            ).dict(),
        }

    @staticmethod
    async def prepare_from_moodle(
        curid: int,
        moodle_filename: str,
        filename_to_replace: str,
        collection_name: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Descarga desde Moodle el archivo seleccionado,
        lo guarda en temp y prepara el job para confirmación.
        """
        service = get_service(model_name)
        resources = get_course_resources(curid)

        file_info = None
        course_name_found = ""
        for resource in resources:
            for cf in resource.get("contentfiles", []):
                if cf["filename"] == moodle_filename:
                    file_info = cf
                    course_name_found = resource.get("name", "")
                    break
            if file_info:
                break

        if not file_info:
            raise ValueError(
                f"'{moodle_filename}' no encontrado en curid {curid}"
            )

        if file_info.get("source") == "youtube":
            temp_path = download_youtube_audio(file_info["fileurl"], moodle_filename)
        else:
            temp_path = download_pdf_to_temp(file_info["fileurl"], moodle_filename)

        val = await validate_document(str(temp_path))
        dup = service.check_duplicate(filename_to_replace, collection_name)

        if moodle_filename != filename_to_replace:
            dup_new = service.check_duplicate(moodle_filename, collection_name)
            if dup_new["exists"]:
                cleanup_temp_file(temp_path)
                raise ValueError(
                    f"El archivo '{moodle_filename}' ya existe en la colección '{collection_name}'. "
                    f"No se puede usar ese nombre como reemplazo porque crearía un duplicado. "
                    f"Seleccione un archivo de Moodle con un nombre distinto, o uno cuyo nombre "
                    f"coincida con el archivo a reemplazar ('{filename_to_replace}')."
                )

        job_id = _new_job(context={
            "mode":                "update",
            "collection_name":     collection_name,
            "model_name":          model_name or settings.EMBEDDING_MODEL,
            "filename_to_replace": filename_to_replace,
            "file": {
                "filename":    moodle_filename,
                "path":        str(temp_path),
                "course_name": course_name_found,
                "total_pages": val["total_pages"],
            },
        })
        job_store[job_id]["status"] = "pendiente_confirmacion"

        return {
            "job_id":          job_id,
            "embedding_model": model_name or settings.EMBEDDING_MODEL,
            "summary": ValidationSummary(
                filename=moodle_filename,
                file_type=val["file_type"],
                total_pages=val["total_pages"],
                total_images=val["total_images"],
                total_tables=val["total_tables"],
                duration_seconds=val.get("duration_seconds"),
                has_non_textual=(val["total_images"] > 0 or val["total_tables"] > 0),
                warning_message=val["warning_message"],
                is_duplicate=dup["exists"],
                existing_chunks=dup["total_chunks"],
            ).dict(),
        }

    @staticmethod
    async def confirm_and_vectorize(job_id: str) -> None:
        """
        Background task:
        1. Elimina el documento anterior de Qdrant.
        2. Vectoriza el nuevo desde temp usando el modelo del context.
        3. Limpia temp.
        """
        job = job_store.get(job_id)
        if not job:
            return

        job["status"] = "procesando"
        ctx = job["context"]
        collection_name     = ctx["collection_name"]
        model_name: str     = ctx.get("model_name", settings.EMBEDDING_MODEL)
        filename_to_replace = ctx["filename_to_replace"]
        fd                  = ctx["file"]
        file_path           = Path(fd["path"])

        service = get_service(model_name)

        try:
            del_result = service.delete_document(filename_to_replace, collection_name)
            print(f"🗑️ Eliminado anterior: {del_result}")

            result = await vectorize_file_with_pages(
                file_path=file_path,
                collection_name=collection_name,
                course_name=fd.get("course_name", "modo manual"),
                total_pages=fd.get("total_pages", 0),
                service=service,
                original_filename=fd.get("filename", ""),
            )
            _complete_job(job_id, {**result, "embedding_model": model_name})

        except Exception as e:
            print(f"❌ Error en actualización: {e}")
            _fail_job(job_id, str(e))
        finally:
            cleanup_temp_file(file_path)
