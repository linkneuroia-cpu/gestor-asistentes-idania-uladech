#!/usr/bin/env bash
# Sube el código del proyecto a producción (10.0.0.92), sin tocar el .env
# real que ya esté en el servidor (nunca se sobreescribe un secreto de
# producción desde acá). Usa rsync si está disponible; si no (p.ej. Git
# Bash en Windows, donde no viene instalado y no hay pacman a mano), cae a
# tar+scp — mismo resultado neto para un deploy, sin depender de rsync.
set -euo pipefail
cd "$(dirname "$0")/.."  # raíz del proyecto

if [ ! -f deploy/.env.deploy ]; then
  echo "Falta deploy/.env.deploy — copiá deploy/.env.deploy.example y completá los datos reales." >&2
  exit 1
fi
# shellcheck disable=SC1091
source deploy/.env.deploy

# DEPLOY_PATH trae un espacio ("Funcionalidad 1") — se escapa para que
# rsync lo pase bien al shell remoto sobre ssh.
REMOTE_DIR_ESCAPED="${DEPLOY_PATH// /\\ }"
REMOTE_TARGET="${DEPLOY_USER}@${DEPLOY_HOST}:${REMOTE_DIR_ESCAPED}/"

HAS_KEY=0
ls "${HOME}"/.ssh/id_* >/dev/null 2>&1 && HAS_KEY=1

if command -v rsync >/dev/null 2>&1; then
  RSYNC_OPTS=(-avz --delete
    --exclude=.git/ --exclude=venv/ --exclude=__pycache__/ --exclude=temp_docs/
    --exclude=.env --exclude=deploy/.env.deploy --exclude='*.log' --exclude=.claude/
    --exclude=runtime_config.json --exclude='*.dump'
    --exclude='3_wowdash-tailwind-css-admin-dashboard-template-*.zip'
  )
  if [ "$HAS_KEY" = 1 ]; then
    echo "🔑 Subiendo con rsync (clave SSH local)..."
    rsync "${RSYNC_OPTS[@]}" ./ "${REMOTE_TARGET}"
  else
    echo "⚠️  Subiendo con rsync + contraseña (deploy/.env.deploy) — considerá configurar una clave SSH."
    command -v sshpass >/dev/null 2>&1 || { echo "Falta 'sshpass'." >&2; exit 1; }
    sshpass -p "${DEPLOY_PASSWORD}" rsync "${RSYNC_OPTS[@]}" -e ssh ./ "${REMOTE_TARGET}"
  fi
else
  echo "⚠️  'rsync' no está disponible acá — subiendo por tar+scp (mismo resultado, sin sincronización incremental)."
  TARBALL="$(mktemp -t gestor_deploy_XXXXXX).tar.gz"
  tar --exclude='.git' --exclude='venv' --exclude='__pycache__' --exclude='temp_docs' \
      --exclude='.env' --exclude='deploy/.env.deploy' --exclude='*.log' --exclude='.claude' \
      --exclude='runtime_config.json' --exclude='*.dump' --exclude='*.pyc' \
      --exclude='3_wowdash-tailwind-css-admin-dashboard-template-*.zip' \
      -czf "$TARBALL" .

  REMOTE_TMP="/tmp/gestor_deploy_$$.tar.gz"
  if [ "$HAS_KEY" = 1 ]; then
    scp -o BatchMode=yes "$TARBALL" "${DEPLOY_USER}@${DEPLOY_HOST}:${REMOTE_TMP}"
    ssh -o BatchMode=yes "${DEPLOY_USER}@${DEPLOY_HOST}" \
      "cd ${REMOTE_DIR_ESCAPED} && tar -xzf ${REMOTE_TMP} && rm -f ${REMOTE_TMP}"
  else
    echo "⚠️  Subiendo con contraseña (deploy/.env.deploy) — considerá configurar una clave SSH."
    command -v sshpass >/dev/null 2>&1 || { echo "Falta 'sshpass'." >&2; rm -f "$TARBALL"; exit 1; }
    sshpass -p "${DEPLOY_PASSWORD}" scp "$TARBALL" "${DEPLOY_USER}@${DEPLOY_HOST}:${REMOTE_TMP}"
    sshpass -p "${DEPLOY_PASSWORD}" ssh "${DEPLOY_USER}@${DEPLOY_HOST}" \
      "cd ${REMOTE_DIR_ESCAPED} && tar -xzf ${REMOTE_TMP} && rm -f ${REMOTE_TMP}"
  fi
  rm -f "$TARBALL"
fi

echo "✅ Código subido a ${DEPLOY_HOST}:${DEPLOY_PATH}"
echo "⚠️  El .env de producción NO se tocó. Si es la primera vez que se despliega, hay que crearlo a mano en el servidor antes de correr deploy.sh."
