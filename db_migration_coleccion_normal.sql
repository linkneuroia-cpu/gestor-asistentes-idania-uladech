-- ============================================================================
-- db_migration_coleccion_normal.sql
-- Permite colecciones "normales" (sin RD) y guarda la métrica de distancia
-- como parte del embedding maestro de la RD (junto a dense_strategy/
-- sparse_strategy) para que las colecciones "hija" también la hereden.
-- Además: límite configurable de actividades por sesión, mensaje de
-- reencuentro (al reabrir una conversación con historial), y tokens
-- consumidos por respuesta.
-- ============================================================================
BEGIN;

ALTER TABLE colecciones_rd ALTER COLUMN rd_id DROP NOT NULL;
ALTER TABLE asistentes ALTER COLUMN rd_id DROP NOT NULL;
ALTER TABLE rds ADD COLUMN IF NOT EXISTS distance VARCHAR(20);
ALTER TABLE asistentes ADD COLUMN IF NOT EXISTS max_actividades_sesion INTEGER NOT NULL DEFAULT 10;
ALTER TABLE asistentes ADD COLUMN IF NOT EXISTS mensaje_reencuentro TEXT;
-- courseid propio, solo para asistentes "normal" (rd_id NULL) — resuelven
-- su única colección directamente por courseid, sin pasar por una RD.
ALTER TABLE asistentes ADD COLUMN IF NOT EXISTS moodle_courseid INTEGER;
ALTER TABLE asistente_mensajes ADD COLUMN IF NOT EXISTS tokens_consumidos INTEGER;

COMMIT;
