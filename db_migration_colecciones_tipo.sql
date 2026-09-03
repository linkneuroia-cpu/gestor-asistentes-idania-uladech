-- ============================================================================
-- db_migration_colecciones_tipo.sql
-- 1) Renombra colecciones_rd -> colecciones: la tabla ya no es solo "de RD"
--    (rd_id es NULL para las colecciones normales desde
--    db_migration_coleccion_normal.sql).
-- 2) Persiste el esquema de vectores con el que se creó cada colección
--    ('hybrid' = dense+sparse, 'legacy' = un solo vector denso). Hasta ahora
--    ese valor viajaba en el request (CollectionCreateRequest.vector_schema)
--    y se perdía: link_collection() nunca lo guardaba.
-- Idempotente: se puede correr dos veces sin romper nada.
-- ============================================================================
BEGIN;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = current_schema() AND table_name = 'colecciones_rd')
       AND NOT EXISTS (SELECT 1 FROM information_schema.tables
                       WHERE table_schema = current_schema() AND table_name = 'colecciones') THEN
        ALTER TABLE colecciones_rd RENAME TO colecciones;
        ALTER INDEX  IF EXISTS idx_colecciones_rd_rd_id RENAME TO idx_colecciones_rd_id;
        ALTER SEQUENCE IF EXISTS colecciones_rd_id_seq  RENAME TO colecciones_id_seq;
    END IF;
END $$;

-- Esquema de vectores de la colección en Qdrant. NULL = colecciones creadas
-- antes de esta migración (el valor real se lee en vivo de Qdrant).
ALTER TABLE colecciones ADD COLUMN IF NOT EXISTS vector_schema VARCHAR(10);

COMMIT;
