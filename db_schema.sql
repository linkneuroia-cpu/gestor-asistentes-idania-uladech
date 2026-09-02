-- ============================================================================
-- db_schema.sql
-- Modelo de tablas del gestor RAG ULADECH Católica (base: U_F1_Profundizacion)
-- Ejecutar de una sola vez para crear todo el esquema desde cero.
-- Para revertir, ver db_drop.sql (elimina todo en orden inverso de FKs).
-- ============================================================================

BEGIN;

-- ── usuarios: login del gestor (/gestor) ───────────────────────────────────
CREATE TABLE usuarios (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(80) NOT NULL UNIQUE,
    password_hash   VARCHAR(200) NOT NULL,   -- formato "salt_hex:hash_hex" (pbkdf2_hmac sha256)
    is_admin        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login      TIMESTAMPTZ
);

-- ── rds: aulas RD (14 iniciales, editable) ─────────────────────────────────
CREATE TABLE rds (
    id                  SERIAL PRIMARY KEY,
    nombre              VARCHAR(300) NOT NULL,
    moodle_courseid     INTEGER NOT NULL,
    moodle_course_url   VARCHAR(500),
    -- Embedding maestro (colección "madre"/padre): dense_strategy/
    -- sparse_strategy/distance quedan fijados con la primera colección real
    -- de la RD y todas las siguientes (las "hijas") lo heredan.
    dense_strategy      VARCHAR(100),
    sparse_strategy     VARCHAR(100),
    distance            VARCHAR(20),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── colecciones_rd: lista de courseid de cada RD (plantilla replicada en
-- varios cursos) y, opcionalmente, qué colección de Qdrant resuelve cada uno.
-- qdrant_collection_name puede ser NULL: un courseid puede registrarse bajo
-- una RD antes de tener colección asignada. (rd_id, moodle_courseid) es
-- único: cada courseid resuelve a una sola colección, sin ambigüedad.
-- rd_id puede ser NULL: colecciones "normales", independientes de cualquier
-- RD (no las encuentra ningún asistente, solo se administran a mano).
CREATE TABLE colecciones_rd (
    id                      SERIAL PRIMARY KEY,
    qdrant_collection_name  VARCHAR(255) UNIQUE,
    rd_id                   INTEGER REFERENCES rds(id) ON DELETE RESTRICT,
    moodle_courseid         INTEGER NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (rd_id, moodle_courseid)
);
CREATE INDEX idx_colecciones_rd_rd_id ON colecciones_rd(rd_id);

-- ── asistentes: chatbots públicos, uno por RD (o varios por RD) ───────────
-- Config RAG 100% propia de este asistente, tanto para VECTORIZAR (etl_*,
-- contextual — consultadas automáticamente por Semiautomático/Automático/
-- Actualización vía core._build_pipeline_config) como para RESPONDER
-- (dense/sparse/rerank/generation). NULL en cualquiera = usa el valor por
-- defecto del sistema (config congelada en Postgres o default de .env).
-- rd_id puede ser NULL: asistente "normal", independiente de cualquier RD
-- (usa una sola colección "normal" propia, resuelta directamente por
-- courseid en vez de RD+courseid).
CREATE TABLE asistentes (
    id                      SERIAL PRIMARY KEY,
    nombre                  VARCHAR(200) NOT NULL,
    rd_id                   INTEGER REFERENCES rds(id) ON DELETE RESTRICT,
    moodle_courseid         INTEGER,  -- solo para asistentes "normal" (rd_id NULL): su único curso/colección
    prompt_maestro          TEXT,
    mensaje_bienvenida      TEXT,  -- NULL = "soy {nombre}. Pregúntame lo que necesites sobre el curso."
    mensaje_reencuentro     TEXT,  -- se muestra al reabrir una conversación con historial (no se guarda)
    max_actividades_sesion  INTEGER NOT NULL DEFAULT 10,  -- tope de preguntas por sesión
    token                   VARCHAR(64) NOT NULL UNIQUE,
    activo                  BOOLEAN NOT NULL DEFAULT TRUE,
    etl_document_strategy   VARCHAR(100),
    etl_audio_strategy      VARCHAR(100),
    contextual_strategy     VARCHAR(100),
    dense_strategy          VARCHAR(100),
    sparse_strategy         VARCHAR(100),
    rerank_strategy         VARCHAR(100),
    generation_strategy     VARCHAR(100),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by              INTEGER REFERENCES usuarios(id) ON DELETE SET NULL
);
CREATE INDEX idx_asistentes_rd_id ON asistentes(rd_id);
CREATE INDEX idx_asistentes_token ON asistentes(token);

-- ── asistente_carpetas: agrupan conversaciones de un mismo alumno con un
-- mismo asistente, para organizar la sidebar de historial ─────────────────
CREATE TABLE asistente_carpetas (
    id              SERIAL PRIMARY KEY,
    asistente_id    INTEGER NOT NULL REFERENCES asistentes(id) ON DELETE CASCADE,
    moodle_userid   INTEGER NOT NULL,
    nombre          VARCHAR(150) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_asistente_carpetas_lookup ON asistente_carpetas(asistente_id, moodle_userid);

-- ── asistente_sesiones: una conversación = (asistente, curso, usuario Moodle) ─
CREATE TABLE asistente_sesiones (
    id                      SERIAL PRIMARY KEY,
    asistente_id            INTEGER NOT NULL REFERENCES asistentes(id) ON DELETE CASCADE,
    moodle_courseid         INTEGER NOT NULL,
    moodle_userid           INTEGER NOT NULL,
    moodle_username         VARCHAR(150),
    moodle_fullname         VARCHAR(300),
    qdrant_collection_name  VARCHAR(255) NOT NULL,
    nombre                  VARCHAR(150),  -- NULL = usa el nombre por defecto (vista previa del primer mensaje)
    carpeta_id              INTEGER REFERENCES asistente_carpetas(id) ON DELETE SET NULL,
    started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_asistente_sesiones_lookup ON asistente_sesiones(asistente_id, moodle_courseid, moodle_userid);

-- ── asistente_mensajes: historial completo de preguntas/respuestas ────────
CREATE TABLE asistente_mensajes (
    id                    SERIAL PRIMARY KEY,
    sesion_id             INTEGER NOT NULL REFERENCES asistente_sesiones(id) ON DELETE CASCADE,
    tiempo_respuesta_ms   INTEGER,  -- solo 'assistant': duración total de la respuesta
    calificacion          SMALLINT,  -- solo 'assistant': 1 útil, -1 no útil, NULL sin calificar
    calificacion_comentario TEXT,  -- solo cuando calificacion = -1: problema opcional indicado por el alumno
    rol                 VARCHAR(10) NOT NULL CHECK (rol IN ('user', 'assistant')),
    tokens_consumidos   INTEGER,  -- solo en mensajes 'assistant': tokens totales de la llamada al LLM
    contenido           TEXT NOT NULL,
    fuentes_json        JSONB,
    config_usado_json   JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_asistente_mensajes_sesion ON asistente_mensajes(sesion_id, created_at);

-- ── curso_archivos: cola de trabajo de indexación (ver ledger.py) — un
-- registro por archivo de curso visto en el LMS, para detectar altas,
-- modificaciones (por metadata del LMS) y bajas (por ausencia) sin
-- reprocesar todo el curso en cada pasada. Sin job periódico todavía
-- (a pedido explícito) — se llama a demanda.
CREATE TABLE curso_archivos (
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
CREATE INDEX idx_curso_archivos_collection ON curso_archivos(qdrant_collection_name);
CREATE INDEX idx_curso_archivos_estado ON curso_archivos(estado);

-- ── configuracion: TODA la configuración mutable del gestor (clave/valor) ──
-- Reemplaza runtime_config.json (selección de estrategia activa por etapa)
-- y la persistencia en .env de las API keys (credentials.py).
CREATE TABLE configuracion (
    clave       VARCHAR(100) PRIMARY KEY,
    valor       TEXT,
    es_secreto  BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;

-- ============================================================================
-- Seed: las 14 RD iniciales dadas por el usuario
-- ============================================================================
BEGIN;

INSERT INTO rds (nombre, moodle_courseid, moodle_course_url) VALUES
('RD-PRE-DERECHO-BL-INTRODUCCIÓN A LAS CIENCIAS JURÍDICAS', 1371, 'https://campus.uladech.edu.pe/course/view.php?id=1371'),
('RD-PRE-DERECHO-VI-INTRODUCCIÓN A LAS CIENCIAS JURÍDICAS', 1374, 'https://campus.uladech.edu.pe/course/view.php?id=1374'),
('RD-PRE-EDUCACION-VI-FILOSOFIA DE LA EDUCACION', 533, 'https://campus.uladech.edu.pe/course/view.php?id=533'),
('RD-PRE-ENFERMERIA-PR-INTRODUCCION A LA ENFERMERIA', 192, 'https://campus.uladech.edu.pe/course/view.php?id=192'),
('RD-PRE-FORMACION GENERAL-BL-COMPETENCIAS COMUNICATIVAS PARA EL APRENDIZAJE UNIVERSITARIO', 1394, 'https://campus.uladech.edu.pe/course/view.php?id=1394'),
('RD-PRE-FORMACION GENERAL-BL-PENSAMIENTO LÓGICO Y MATEMÁTICO', 1392, 'https://campus.uladech.edu.pe/course/view.php?id=1392'),
('RD-PRE-FORMACION GENERAL-BL-COMPETENCIAS DIGITALES PARA LA VIDA UNIVERSITARIA', 1389, 'https://campus.uladech.edu.pe/course/view.php?id=1389'),
('RD-PRE-FORMACION GENERAL-VI-COMPETENCIAS DIGITALES PARA LA VIDA UNIVERSITARIA', 1398, 'https://campus.uladech.edu.pe/course/view.php?id=1398'),
('RD-PRE-FORMACION GENERAL-VI-APRENDIZAJE AUTÓNOMO Y ESTRATEGIAS UNIVERSITARIAS', 1400, 'https://campus.uladech.edu.pe/course/view.php?id=1400'),
('RD-PRE-FORMACION GENERAL-VI-COMPETENCIAS COMUNICATIVAS PARA EL APRENDIZAJE UNIVERSITARIO', 1399, 'https://campus.uladech.edu.pe/course/view.php?id=1399'),
('RD-PRE-FORMACION GENERAL-VI-DOCTRINA SOCIAL DE LA IGLESIA', 1401, 'https://campus.uladech.edu.pe/course/view.php?id=1401'),
('RD-PRE-FORMACION GENERAL-VI-PENSAMIENTO LÓGICO Y MATEMÁTICO', 1396, 'https://campus.uladech.edu.pe/course/view.php?id=1396'),
('RD-PRE-ODONTOLOGIA-PR-INTRODUCCION A LA ODONTOLOGIA', 173, 'https://campus.uladech.edu.pe/course/view.php?id=173'),
('RD-PRE-PSICOLOGIA-BL-INTRODUCCION A LA PSICOLOGIA', 948, 'https://campus.uladech.edu.pe/course/view.php?id=948');

-- Registra el courseid original de cada RD en su lista de cursos (sin
-- colección asignada todavía) — así "Gestión RD" ya lo muestra desde el inicio.
INSERT INTO colecciones_rd (rd_id, moodle_courseid)
SELECT id, moodle_courseid FROM rds;

COMMIT;
