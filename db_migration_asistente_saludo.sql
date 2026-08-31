-- ============================================================================
-- db_migration_asistente_saludo.sql
-- Migración: mensaje de bienvenida configurable por asistente (antes fijo:
-- "soy {nombre}. Pregúntame lo que necesites sobre el curso.").
-- Aplicar UNA VEZ sobre una base ya migrada con db_migration_rd_embedding.sql.
-- ============================================================================

BEGIN;

ALTER TABLE asistentes ADD COLUMN mensaje_bienvenida TEXT;

COMMIT;
