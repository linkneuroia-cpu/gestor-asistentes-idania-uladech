"""
ledger.py
=========
Cola de trabajo de indexación por archivo de curso (`curso_archivos`):
un registro por archivo visto en el LMS, que permite detectar altas,
modificaciones y borrados SIN reprocesar todo el curso en cada pasada.

Flujo (llamado a demanda — sin job periódico, ver nota abajo):
  1. Lista los archivos actuales del curso vía la API del LMS.
  2. Altas: archivo visto que no está en la tabla -> 'nuevo'.
  3. Modificaciones: compara metadatos baratos que expone el LMS
     (`contenthash` de Moodle — un SHA1 nativo sobre los bytes del
     archivo, o `timemodified` si el primero no viene) contra lo
     guardado la última vez -> si difieren, 'posible_actualizacion'.
     OJO: esto NO confirma que el texto cambió — muchos LMS marcan
     "modificado" por cambios de visibilidad/permisos que no tocan el
     contenido. La confirmación real (vale la pena gastar cómputo en
     re-embeder) sigue siendo el `document_hash` que ya calcula
     `definitions.extract_normalize_and_hash()` DESPUÉS de re-extraer
     el texto — este módulo solo decide qué archivos vale la pena
     mandar a esa extracción, no reemplaza esa verificación.
  4. Borrados: cualquier archivo que estaba activo mano no apareció en
     esta pasada -> 'eliminado' (por ausencia, no por evento explícito).

Esta tabla es la cola de trabajo para el pipeline de indexación
(Semiautomático/Automático/Actualización deciden qué reprocesar
consultando su estado) — pero el job periódico que la recorra
automáticamente NO está implementado todavía a pedido explícito: se
deja este módulo listo para que ese job (cron/APScheduler/etc.) lo
llame cuando se decida construirlo. Moodle/Canvas/Google Drive exponen
esta metadata con confiabilidad distinta — los nombres de campo usados
acá (`contenthash`, `timemodified`) son específicos de Moodle; portar
a otro LMS requiere ajustar `definitions.get_course_resources()` y la
comparación de abajo.
"""
from typing import Any, Dict, List, Optional

import db

ESTADOS = ("nuevo", "sin_cambios", "posible_actualizacion", "eliminado", "vectorizado")


def sync_curso_archivos(collection_name: str, curid: int) -> Dict[str, Any]:
    """Compara la lista actual de archivos del curso (Moodle) contra lo
    guardado en `curso_archivos` para esta colección, y actualiza la
    tabla: altas, posibles modificaciones, y bajas por ausencia. Retorna
    un resumen + la lista de archivos que quedaron en cola de trabajo
    (nuevo/posible_actualizacion) para que el pipeline los procese."""
    from definitions import get_course_resources

    resources = get_course_resources(curid)
    vistos: List[str] = []
    resumen = {"nuevos": 0, "posibles_actualizaciones": 0, "sin_cambios": 0, "eliminados": 0}

    for resource in resources:
        for f in resource.get("contentfiles", []):
            filename = f.get("filename")
            if not filename:
                continue
            if filename in vistos:
                # Mismo archivo enlazado en otra sección/semana (p.ej. un
                # libro de bibliografía usado en varias semanas) — ya se
                # procesó en esta misma pasada. Sin esto, cada aparición
                # pisaba el UPDATE anterior con su propio timemodified
                # (que difiere unos segundos entre enlaces aunque el
                # archivo real no cambió), marcando falsos positivos de
                # "posible_actualizacion" en cada sync.
                continue
            vistos.append(filename)
            contenthash = f.get("contenthash")
            timemodified = f.get("timemodified")

            existing = db.fetch_one(
                "SELECT * FROM curso_archivos WHERE qdrant_collection_name=%s AND filename=%s",
                (collection_name, filename),
            )
            if not existing:
                db.execute(
                    "INSERT INTO curso_archivos "
                    "(qdrant_collection_name, moodle_courseid, resource_id, filename, fileurl, "
                    "lms_contenthash, lms_timemodified, estado, last_seen_at) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,'nuevo',now())",
                    (collection_name, curid, resource.get("id"), filename, f.get("fileurl"), contenthash, timemodified),
                )
                resumen["nuevos"] += 1
                continue

            # Prioriza contenthash (bytes reales del archivo) sobre
            # timemodified (puede cambiar sin tocar el archivo).
            if contenthash:
                cambio = contenthash != existing["lms_contenthash"]
            else:
                cambio = bool(timemodified) and timemodified != existing["lms_timemodified"]

            if cambio:
                nuevo_estado = "posible_actualizacion"
                resumen["posibles_actualizaciones"] += 1
            elif existing["estado"] == "eliminado":
                # Reapareció un archivo que se había marcado eliminado.
                nuevo_estado = "nuevo"
                resumen["nuevos"] += 1
            else:
                nuevo_estado = existing["estado"] if existing["estado"] != "posible_actualizacion" else existing["estado"]
                resumen["sin_cambios"] += 1

            db.execute(
                "UPDATE curso_archivos SET resource_id=%s, fileurl=%s, lms_contenthash=%s, "
                "lms_timemodified=%s, estado=%s, last_seen_at=now(), updated_at=now() "
                "WHERE id=%s",
                (resource.get("id"), f.get("fileurl"), contenthash, timemodified, nuevo_estado, existing["id"]),
            )

    # Borrados por ausencia: activos de esta colección que no aparecieron
    # en la pasada actual.
    if vistos:
        placeholders = ",".join(["%s"] * len(vistos))
        eliminados = db.fetch_all(
            f"UPDATE curso_archivos SET estado='eliminado', updated_at=now() "
            f"WHERE qdrant_collection_name=%s AND estado != 'eliminado' AND filename NOT IN ({placeholders}) "
            f"RETURNING id",
            (collection_name, *vistos),
        )
    else:
        eliminados = db.fetch_all(
            "UPDATE curso_archivos SET estado='eliminado', updated_at=now() "
            "WHERE qdrant_collection_name=%s AND estado != 'eliminado' RETURNING id",
            (collection_name,),
        )
    resumen["eliminados"] = len(eliminados)

    cola = db.fetch_all(
        "SELECT * FROM curso_archivos WHERE qdrant_collection_name=%s "
        "AND estado IN ('nuevo','posible_actualizacion') ORDER BY filename",
        (collection_name,),
    )
    return {"resumen": resumen, "cola": cola}


def marcar_vectorizado(collection_name: str, filename: str, document_hash: Optional[str] = None) -> None:
    """Llamado después de que el pipeline efectivamente (re)vectorizó un
    archivo — cierra el ciclo: sale de la cola de trabajo."""
    db.execute(
        "UPDATE curso_archivos SET estado='vectorizado', document_hash=%s, last_indexed_at=now(), updated_at=now() "
        "WHERE qdrant_collection_name=%s AND filename=%s",
        (document_hash, collection_name, filename),
    )


def list_curso_archivos(collection_name: str) -> List[Dict[str, Any]]:
    return db.fetch_all(
        "SELECT * FROM curso_archivos WHERE qdrant_collection_name=%s ORDER BY filename",
        (collection_name,),
    )
