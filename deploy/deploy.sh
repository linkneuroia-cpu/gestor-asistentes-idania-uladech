#!/usr/bin/env bash
# Levanta (o reconstruye) el contenedor en producción. Corré deploy/upload.sh
# antes de este script para que el código remoto esté al día. Corta con un
# mensaje claro si falta el .env de producción — nunca arranca el
# contenedor sin secretos reales.
#
# El usuario de despliegue no está en el grupo "docker" en el servidor —
# usa sudo (con la misma contraseña de login) para los comandos docker. No
# se modifica sudoers ni ninguna config del servidor para evitar esto.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env.deploy ]; then
  echo "Falta deploy/.env.deploy — copiá .env.deploy.example y completá los datos reales." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .env.deploy

REMOTE_PATH_ESCAPED="${DEPLOY_PATH// /\\ }"
REMOTE_CMD="cd ${REMOTE_PATH_ESCAPED} && test -f .env || { echo 'Falta .env en el servidor — creá uno con los secretos reales de producción antes de desplegar.' >&2; exit 1; } && sudo -S docker compose up -d --build"

if ls "${HOME}"/.ssh/id_* >/dev/null 2>&1; then
  echo "🔑 Desplegando con clave SSH (sudo pide la contraseña de deploy/.env.deploy solo para docker)..."
  ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "$REMOTE_CMD" <<< "${DEPLOY_PASSWORD}"
else
  echo "⚠️  Desplegando con contraseña (deploy/.env.deploy) — considerá configurar una clave SSH."
  command -v sshpass >/dev/null 2>&1 || { echo "Falta 'sshpass'." >&2; exit 1; }
  sshpass -p "${DEPLOY_PASSWORD}" ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "$REMOTE_CMD" <<< "${DEPLOY_PASSWORD}"
fi

echo "✅ Deploy disparado en ${DEPLOY_HOST}. Revisá logs con: deploy/connect.sh, luego 'sudo docker compose logs -f' en la carpeta del proyecto."
