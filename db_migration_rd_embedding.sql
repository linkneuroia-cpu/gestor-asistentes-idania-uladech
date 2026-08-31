-- ============================================================================
-- db_migration_rd_embedding.sql
-- Migración: el embedding denso/disperso pasa a fijarse por RD (todas sus
-- colecciones deben ser intercambiables entre sí), no por asistente.
-- Aplicar UNA VEZ sobre una base ya migrada con db_migration_asistente_etl.sql.
-- ============================================================================

BEGIN;

ALTER TABLE rds ADD COLUMN dense_strategy VARCHAR(100);
ALTER TABLE rds ADD COLUMN sparse_strategy VARCHAR(100);

COMMIT;
