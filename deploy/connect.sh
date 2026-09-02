#!/usr/bin/env bash
# Conexión SSH al servidor de producción (10.0.0.92). Prefiere una clave
# SSH local si existe (más seguro); si no, cae a contraseña vía sshpass
# (leída de deploy/.env.deploy, nunca hardcodeada acá).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env.deploy ]; then
  echo "Falta deploy/.env.deploy — copiá .env.deploy.example y completá los datos reales." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .env.deploy

if ls "${HOME}"/.ssh/id_* >/dev/null 2>&1; then
  echo "🔑 Usando clave SSH local..."
  exec ssh "${DEPLOY_USER}@${DEPLOY_HOST}"
fi

echo "⚠️  No se encontró una clave SSH local — conectando con contraseña (deploy/.env.deploy)."
echo "    Considerá configurar 'ssh-copy-id ${DEPLOY_USER}@${DEPLOY_HOST}' para no depender de la contraseña."
command -v sshpass >/dev/null 2>&1 || {
  echo "Falta 'sshpass'. Instalalo (apt install sshpass / brew install hudochenkov/sshpass/sshpass) o configurá una clave SSH." >&2
  exit 1
}
exec sshpass -p "${DEPLOY_PASSWORD}" ssh "${DEPLOY_USER}@${DEPLOY_HOST}"
