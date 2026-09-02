-- ============================================================================
-- db_migration_curso_archivos.sql
-- Cola de trabajo de indexación: un registro por archivo de curso visto en
-- el LMS. Ver ledger.py — detección de altas/modificaciones/bajas, sin job
-- periódico todavía (a pedido explícito, queda listo para engancharlo).
-- ============================================================================
BEGIN;

CREATE TABLE IF NOT EXISTS curso_archivos (
    id                      SERIAL PRIMARY KEY,
    qdrant_collection_name  VARCHAR(255) NOT NULL,
    moodle_courseid         INTEGER NOT NULL,
    resource_id             INTEGER,
    filename                VARCHAR(500) NOT NULL,
    fileurl                 TEXT,
    lms_contenthash         VARCHAR(64),
    lms_timemodified        BIGINT,
    document_hash           VARCHAR(64),
    estado                  VARCHAR(30) NOT NULL DEFAULT 'nuevo',
    last_seen_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_indexed_at         TIMESTAMPTZ,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (qdrant_collection_name, filename)
);
CREATE INDEX IF NOT EXISTS idx_curso_archivos_collection ON curso_archivos(qdrant_collection_name);
CREATE INDEX IF NOT EXISTS idx_curso_archivos_estado ON curso_archivos(estado);

COMMIT;
