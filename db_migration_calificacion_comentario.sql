-- ============================================================================
-- db_migration_calificacion_comentario.sql
-- Comentario opcional del alumno al calificar una respuesta con 👎 ("no me
-- ayudó"). `calificacion` (1/-1/NULL) ya existe desde
-- db_migration_calidad_sesiones.sql — esta migración solo agrega el texto
-- libre asociado.
-- ============================================================================
BEGIN;

ALTER TABLE asistente_mensajes ADD COLUMN IF NOT EXISTS calificacion_comentario TEXT;

COMMIT;
