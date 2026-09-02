-- ============================================================================
-- db_migration_calidad_sesiones.sql
-- Tiempo de respuesta y calificación por mensaje; sesiones renombrables y
-- agrupables en carpetas.
-- ============================================================================
BEGIN;

ALTER TABLE asistente_mensajes ADD COLUMN IF NOT EXISTS tiempo_respuesta_ms INTEGER;
-- Calificación de la respuesta (solo mensajes 'assistant'): 1 = útil,
-- -1 = no útil, NULL = sin calificar todavía.
ALTER TABLE asistente_mensajes ADD COLUMN IF NOT EXISTS calificacion SMALLINT;

-- ── carpetas: agrupan conversaciones de un mismo asistente (una carpeta
-- por asistente+alumno en la práctica, ya que la sidebar es por alumno) ──
CREATE TABLE IF NOT EXISTS asistente_carpetas (
    id              SERIAL PRIMARY KEY,
    asistente_id    INTEGER NOT NULL REFERENCES asistentes(id) ON DELETE CASCADE,
    moodle_userid   INTEGER NOT NULL,
    nombre          VARCHAR(150) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_asistente_carpetas_lookup ON asistente_carpetas(asistente_id, moodle_userid);

ALTER TABLE asistente_sesiones ADD COLUMN IF NOT EXISTS nombre VARCHAR(150);
ALTER TABLE asistente_sesiones ADD COLUMN IF NOT EXISTS carpeta_id INTEGER REFERENCES asistente_carpetas(id) ON DELETE SET NULL;

COMMIT;
