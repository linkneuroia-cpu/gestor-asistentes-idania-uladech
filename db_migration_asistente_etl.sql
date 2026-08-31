-- ============================================================================
-- db_migration_asistente_etl.sql
-- Migración: config RAG 100% por asistente — agrega ETL/Contextual (antes
-- solo dense/sparse/rerank/generation eran configurables por asistente).
-- Aplicar UNA VEZ sobre una base ya migrada con db_migration_courseids.sql.
-- ============================================================================

BEGIN;

ALTER TABLE asistentes ADD COLUMN etl_document_strategy VARCHAR(100);
ALTER TABLE asistentes ADD COLUMN etl_audio_strategy VARCHAR(100);
ALTER TABLE asistentes ADD COLUMN contextual_strategy VARCHAR(100);

COMMIT;
