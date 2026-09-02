"""
Administración de colecciones Qdrant con soporte para vectores nombrados
(dense + sparse) y búsqueda híbrida con fusión RRF en una sola llamada.

Extiende el patrón del proyecto standalone "gestion quadrant" (antes un
micro-servicio aparte, ahora absorbido aquí) para colecciones "hibridas"
usadas por el pipeline RAG configurable (ver strategies/), manteniendo
compatibilidad total con las colecciones "legacy" (un solo vector denso sin
nombre) que ya usa VectorizationService en definitions.py.

Convención de nombres de vector: literalmente "dense" y "sparse" en toda
colección híbrida creada por este módulo.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from settings import settings

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

VectorSchema = Literal["legacy", "hybrid"]


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class CollectionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    vector_schema: VectorSchema = "hybrid"
    # dense_size: solo se usa tal cual para colecciones "legacy". Para
    # "hybrid", se ignora y se deriva del embedding maestro de la RD (ver
    # app.py qdrant_create_collection) — o de `dense_strategy` si es la
    # primera colección real de esa RD.
    dense_size: int = settings.DEFAULT_VECTOR_SIZE
    distance: str = settings.DEFAULT_DISTANCE
    # Solo necesario/usado cuando la RD todavía no tiene un embedding
    # maestro fijado (su primera colección híbrida) — fija el de ahí en
    # adelante para toda la RD. Se ignora si la RD ya tiene uno.
    dense_strategy: Optional[str] = None
    sparse_strategy: Optional[str] = None
    # RD (aula) + curso de Moodle al que queda asignada esta colección. Course
    # ID siempre es obligatorio; RD es opcional — sin RD la colección queda
    # "normal" (independiente, ningún asistente puede encontrarla). No se
    # guardan en el payload de Qdrant, se persisten en Postgres
    # (colecciones_rd) desde el endpoint de app.py.
    rd_id: Optional[int] = None
    moodle_courseid: int


class CollectionUpdateRequest(BaseModel):
    description: Optional[str] = None


class CollectionStats(BaseModel):
    name: str
    description: Optional[str] = None
    vector_schema: VectorSchema
    vector_size: int
    distance: str
    points_count: int
    indexed_vectors_count: int
    status: str
    created_at: Optional[str] = None


# ============================================================================
# ÍNDICES DE PAYLOAD POR DEFECTO
# ============================================================================
# Superset de los índices del gestor standalone original, más los campos
# nuevos que trae el pipeline RAG (source_type, embedding_model).

DEFAULT_PAYLOAD_INDEXES = {
    "document_hash": PayloadSchemaType.KEYWORD,
    "filename": PayloadSchemaType.KEYWORD,
    "filename_normalized": PayloadSchemaType.KEYWORD,
    "format": PayloadSchemaType.KEYWORD,
    "file_type": PayloadSchemaType.KEYWORD,
    "chunk": PayloadSchemaType.INTEGER,
    "total_pages": PayloadSchemaType.INTEGER,
    "date": PayloadSchemaType.KEYWORD,
    "source_type": PayloadSchemaType.KEYWORD,
    "embedding_model": PayloadSchemaType.KEYWORD,
}

_DISTANCE_MAP = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


def _distance_metric(distance: str) -> Distance:
    return _DISTANCE_MAP.get(distance.lower(), Distance.COSINE)


# ============================================================================
# GESTOR DE ADMINISTRACIÓN DE QDRANT (colecciones legacy + híbridas)
# ============================================================================

class QdrantAdminManager:
    _instance = None
    _client: Optional[QdrantClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            print(f"Conectando a Qdrant (admin): {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            self._client = QdrantClient(
                url=f"{settings.QDRANT_PROTOCOL}://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
                api_key=settings.QDRANT_API_KEY or None,
                timeout=60,
            )
            print("Cliente Qdrant (admin) inicializado")

    # ========================================================================
    # CRUD DE COLECCIONES
    # ========================================================================

    def collection_exists(self, name: str) -> bool:
        return name in [c.name for c in self._client.get_collections().collections]

    def get_vector_schema(self, name: str) -> VectorSchema:
        """Detecta el esquema de vectores de una colección existente
        inspeccionando su configuración real (no solo la metadata), para que
        funcione incluso con colecciones creadas fuera de este módulo."""
        info = self._client.get_collection(name)
        vectors_config = info.config.params.vectors
        return "hybrid" if isinstance(vectors_config, dict) else "legacy"

    def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        vector_schema: VectorSchema = "hybrid",
        dense_size: int = settings.DEFAULT_VECTOR_SIZE,
        distance: str = settings.DEFAULT_DISTANCE,
    ) -> Dict[str, Any]:
        if self.collection_exists(name):
            raise ValueError(f"La colección '{name}' ya existe")

        distance_metric = _distance_metric(distance)
        metadata = {
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vector_schema": vector_schema,
        }

        if vector_schema == "hybrid":
            self._client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(size=dense_size, distance=distance_metric),
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    ),
                },
                metadata=metadata,
            )
        else:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dense_size, distance=distance_metric),
                metadata=metadata,
            )

        for field, schema in DEFAULT_PAYLOAD_INDEXES.items():
            try:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:
                print(f"⚠️ Índice '{field}' no creado: {e}")

        return {
            "success": True,
            "collection_name": name,
            "vector_schema": vector_schema,
            "message": f"Colección '{name}' creada correctamente ({vector_schema})",
        }

    def list_collections(self) -> List[CollectionStats]:
        collections = self._client.get_collections().collections
        return [self.get_collection_info(col.name) for col in collections]

    def get_collection_info(self, name: str) -> CollectionStats:
        if not self.collection_exists(name):
            raise ValueError(f"La colección '{name}' no existe")

        info = self._client.get_collection(name)
        metadata = info.config.metadata or {}
        vectors_config = info.config.params.vectors

        if isinstance(vectors_config, dict):
            detected_schema: VectorSchema = "hybrid"
            vp = vectors_config.get(DENSE_VECTOR_NAME) or next(iter(vectors_config.values()))
        else:
            detected_schema = "legacy"
            vp = vectors_config

        if isinstance(vp, dict):
            vector_size = vp.get("size")
            distance = vp.get("distance")
        else:
            vector_size = vp.size
            distance = vp.distance

        distance_value = distance if isinstance(distance, str) else distance.name

        return CollectionStats(
            name=name,
            description=metadata.get("description"),
            vector_schema=metadata.get("vector_schema", detected_schema),
            vector_size=vector_size,
            distance=distance_value,
            points_count=info.points_count or 0,
            indexed_vectors_count=info.indexed_vectors_count or 0,
            status=info.status.name,
            created_at=metadata.get("created_at"),
        )

    def update_collection(self, name: str, description: Optional[str]) -> Dict[str, Any]:
        if not self.collection_exists(name):
            raise ValueError(f"La colección '{name}' no existe")

        info = self._client.get_collection(name)
        existing_metadata = info.config.metadata or {}
        metadata = {
            **existing_metadata,
            "description": description,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self._client.update_collection(collection_name=name, metadata=metadata)

        return {
            "success": True,
            "collection_name": name,
            "message": f"Colección '{name}' actualizada correctamente",
        }

    def delete_collection(self, name: str, force: bool = False) -> Dict[str, Any]:
        if not self.collection_exists(name):
            raise ValueError(f"La colección '{name}' no existe")

        info = self._client.get_collection(name)
        if (info.points_count or 0) > 0 and not force:
            raise ValueError(
                f"La colección '{name}' tiene vectores. Use force=true."
            )

        self._client.delete_collection(name)

        return {
            "success": True,
            "collection_name": name,
            "message": f"Colección '{name}' eliminada correctamente",
        }

    def clear_collection(self, name: str) -> Dict[str, Any]:
        """Elimina y recrea la colección conservando exactamente su
        configuración de vectores (legacy o híbrida) y su metadata."""
        if not self.collection_exists(name):
            raise ValueError(f"La colección '{name}' no existe")

        info = self._client.get_collection(name)
        metadata = info.config.metadata
        vectors_config = info.config.params.vectors
        sparse_config = info.config.params.sparse_vectors

        self._client.delete_collection(name)
        self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
            metadata=metadata,
        )

        return {
            "success": True,
            "collection_name": name,
            "message": f"Colección '{name}' limpiada correctamente",
        }

    # ========================================================================
    # GESTIÓN DE DOCUMENTOS
    # ========================================================================

    def delete_document_by_filename(
        self,
        collection_name: str,
        filename: str,
    ) -> Dict[str, Any]:
        """Elimina todos los puntos (chunks) que pertenecen a un documento
        específico basándose en el campo 'filename' del payload."""
        if not self.collection_exists(collection_name):
            raise ValueError(f"La colección '{collection_name}' no existe")

        scroll_result = self._client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )

        points_to_delete = scroll_result[0]
        total_points = len(points_to_delete)

        if total_points == 0:
            raise ValueError(
                f"No se encontraron puntos con filename='{filename}' "
                f"en la colección '{collection_name}'"
            )

        self._client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
        )

        return {
            "success": True,
            "collection_name": collection_name,
            "filename": filename,
            "deleted_points": total_points,
            "message": f"Eliminados {total_points} chunks del documento '{filename}'",
        }

    def list_documents_in_collection(self, collection_name: str) -> Dict[str, Any]:
        """Lista todos los documentos únicos en una colección, agrupados por
        filename, con información de chunks."""
        if not self.collection_exists(collection_name):
            raise ValueError(f"La colección '{collection_name}' no existe")

        all_points = []
        offset = None

        while True:
            scroll_result = self._client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = scroll_result
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        documents: Dict[str, Dict[str, Any]] = {}
        for point in all_points:
            payload = point.payload or {}
            filename = payload.get("filename", "unknown")

            if filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "document_hash": payload.get("document_hash"),
                    "format": payload.get("format"),
                    "total_pages": payload.get("total_pages"),
                    "total_chunks": payload.get("total_chunks"),
                    "date": payload.get("date"),
                    "source_type": payload.get("source_type"),
                    "chunks_count": 0,
                }

            documents[filename]["chunks_count"] += 1

        return {
            "collection_name": collection_name,
            "total_documents": len(documents),
            "total_points": len(all_points),
            "documents": list(documents.values()),
        }

    # ========================================================================
    # UPSERT HÍBRIDO
    # ========================================================================

    def upsert_hybrid_points(
        self,
        *,
        collection_name: str,
        points: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Hace upsert de puntos con vectores nombrados dense+sparse.

        `points`: lista de dicts con las claves:
            id             - str | int, id único del punto
            dense_vector   - List[float]
            sparse_indices - List[int]
            sparse_values  - List[float]
            payload        - Dict[str, Any]
        """
        if not self.collection_exists(collection_name):
            raise ValueError(f"La colección '{collection_name}' no existe")

        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            structs = [
                PointStruct(
                    id=p["id"],
                    vector={
                        DENSE_VECTOR_NAME: p["dense_vector"],
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=p["sparse_indices"],
                            values=p["sparse_values"],
                        ),
                    },
                    payload=p["payload"],
                )
                for p in batch
            ]
            self._client.upsert(collection_name=collection_name, points=structs, wait=True)
            total += len(structs)

        return {
            "success": True,
            "collection_name": collection_name,
            "points_upserted": total,
        }

    # ========================================================================
    # BÚSQUEDA HÍBRIDA (dense + sparse, fusión RRF en una sola llamada)
    # ========================================================================

    def hybrid_search(
        self,
        *,
        collection_name: str,
        dense_query: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        limit: int = 50,
        prefetch_limit: int = 100,
        query_filter: Optional[Filter] = None,
    ) -> List[Any]:
        """Recupera candidatos fusionando dense + sparse (Reciprocal Rank
        Fusion) en una única llamada a Qdrant vía Prefetch + FusionQuery."""
        response = self._client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points


# ============================================================================
# SINGLETON
# ============================================================================

_admin_manager: Optional[QdrantAdminManager] = None


def get_qdrant_admin() -> QdrantAdminManager:
    global _admin_manager
    if _admin_manager is None:
        _admin_manager = QdrantAdminManager()
    return _admin_manager
