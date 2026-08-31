-- ============================================================================
-- db_migration_courseids.sql
-- Migración: RD como plantillas replicadas en varios courseid + config RAG
-- por asistente. Aplicar UNA VEZ sobre una base ya creada con db_schema.sql.
-- ============================================================================

BEGIN;

-- Permite registrar un courseid bajo una RD antes de tener colección asignada
ALTER TABLE colecciones_rd ALTER COLUMN qdrant_collection_name DROP NOT NULL;

-- Un (RD, courseid) solo puede resolver a una colección — evita ambigüedad
-- en resolve_collection() (usada por la ruta pública del asistente)
ALTER TABLE colecciones_rd ADD CONSTRAINT colecciones_rd_rd_courseid_uniq UNIQUE (rd_id, moodle_courseid);

-- Config RAG propia por asistente (NULL = usa la configuración global, igual que hoy)
ALTER TABLE asistentes ADD COLUMN dense_strategy VARCHAR(100);
ALTER TABLE asistentes ADD COLUMN sparse_strategy VARCHAR(100);
ALTER TABLE asistentes ADD COLUMN rerank_strategy VARCHAR(100);
ALTER TABLE asistentes ADD COLUMN generation_strategy VARCHAR(100);

-- Backfill: asegura que el courseid original de cada RD ya exista en la lista
INSERT INTO colecciones_rd (rd_id, moodle_courseid)
SELECT id, moodle_courseid FROM rds r
WHERE NOT EXISTS (
  SELECT 1 FROM colecciones_rd c WHERE c.rd_id = r.id AND c.moodle_courseid = r.moodle_courseid
);

COMMIT;
