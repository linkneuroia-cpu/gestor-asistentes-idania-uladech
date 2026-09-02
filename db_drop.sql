-- ============================================================================
-- db_drop.sql
-- Elimina TODAS las tablas creadas por db_schema.sql, en orden inverso de
-- dependencias (FKs). Irreversible: borra también todos los datos.
-- ============================================================================

BEGIN;

DROP TABLE IF EXISTS asistente_mensajes;
DROP TABLE IF EXISTS asistente_sesiones;
DROP TABLE IF EXISTS asistentes;
DROP TABLE IF EXISTS colecciones_rd;
DROP TABLE IF EXISTS curso_archivos;
DROP TABLE IF EXISTS configuracion;
DROP TABLE IF EXISTS rds;
DROP TABLE IF EXISTS usuarios;

COMMIT;
