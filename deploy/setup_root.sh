#!/usr/bin/env bash
# Instala deploy/nginx_educaia.conf en el servidor compartido de
# educaia.uladech.edu.pe. Este archivo nginx es COMPARTIDO con las
# funcionalidades f2 y f3 (ver comentario de cabecera del propio .conf) —
# este script solo toca su contenido para agregar/actualizar el location
# de f1, sin modificar los bloques de f2/f3 (deploy/nginx_educaia.conf ya
# los trae copiados tal cual estaban desplegados).
#
# Hace backup del archivo vivo antes de reemplazarlo, y valida con
# `nginx -t` ANTES de recargar — si la validación falla, no recarga y deja
# el backup como red de seguridad para restaurar a mano.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env.deploy ]; then
  echo "Falta deploy/.env.deploy — copiá .env.deploy.example y completá los datos reales." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .env.deploy

REMOTE_TMP="/tmp/educaia_nginx_$$.conf"

echo "📤 Subiendo deploy/nginx_educaia.conf..."
if ls "${HOME}"/.ssh/id_* >/dev/null 2>&1; then
  scp -o BatchMode=yes nginx_educaia.conf "${DEPLOY_USER}@${DEPLOY_HOST}:${REMOTE_TMP}"
else
  command -v sshpass >/dev/null 2>&1 || { echo "Falta 'sshpass'." >&2; exit 1; }
  sshpass -p "${DEPLOY_PASSWORD}" scp nginx_educaia.conf "${DEPLOY_USER}@${DEPLOY_HOST}:${REMOTE_TMP}"
fi

# Todo el trabajo que requiere root va en UN solo `sudo -S bash -c ...`, así
# alcanza con pasar la contraseña una vez (sudo cachea la sesión adentro de
# ese proceso hijo) en vez de arriesgarse a que varias invocaciones de
# `sudo -S` sueltas se peleen por leer la misma línea de stdin.
REMOTE_CMD="sudo -S bash -c '
set -e
BACKUP=/etc/nginx/conf.d/educaia.conf.bak.\$(date +%Y%m%d%H%M%S)
cp /etc/nginx/conf.d/educaia.conf \"\$BACKUP\"
echo \"🗄️  Backup: \$BACKUP\"
cp ${REMOTE_TMP} /etc/nginx/conf.d/educaia.conf
rm -f ${REMOTE_TMP}
if nginx -t; then
  systemctl reload nginx
  echo \"✅ nginx recargado con el location de /f1/.\"
else
  echo \"❌ nginx -t falló — NO se recargó. Backup disponible en \$BACKUP para restaurar a mano.\" >&2
  exit 1
fi
'"

echo "🔧 Aplicando en el servidor (requiere sudo, misma contraseña de deploy/.env.deploy)..."
if ls "${HOME}"/.ssh/id_* >/dev/null 2>&1; then
  ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "$REMOTE_CMD" <<< "${DEPLOY_PASSWORD}"
else
  sshpass -p "${DEPLOY_PASSWORD}" ssh "${DEPLOY_USER}@${DEPLOY_HOST}" "$REMOTE_CMD" <<< "${DEPLOY_PASSWORD}"
fi
