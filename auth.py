"""
auth.py
=======
Login del gestor (/gestor) contra la tabla `usuarios` en Postgres.
Contraseñas con hashlib.pbkdf2_hmac (sin dependencias nuevas de hashing) +
sal propia por usuario. Sesión vía cookie firmada (SessionMiddleware de
Starlette, montada en app.py) — sin JWT ni tokens de API para esto, es
solo para el panel interno.
"""
import hashlib
import hmac
import secrets
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

import db
from settings import settings

_PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"{salt}:{digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, expected_hex = password_hash.split(":", 1)
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return hmac.compare_digest(digest.hex(), expected_hex)


def bootstrap_admin() -> None:
    """Si la tabla `usuarios` está vacía, crea el admin inicial desde
    ADMIN_USERNAME/ADMIN_PASSWORD (.env). Idempotente: no hace nada si ya
    hay al menos un usuario."""
    existing = db.fetch_one("SELECT id FROM usuarios LIMIT 1")
    if existing:
        return
    db.execute(
        "INSERT INTO usuarios (username, password_hash, is_admin) VALUES (%s, %s, TRUE)",
        (settings.ADMIN_USERNAME, hash_password(settings.ADMIN_PASSWORD)),
    )
    print(f"👤 Usuario admin inicial creado: '{settings.ADMIN_USERNAME}' (cambia ADMIN_PASSWORD en .env o la contraseña desde el gestor)")


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = db.fetch_one("SELECT * FROM usuarios WHERE username = %s", (username,))
    if not user or not verify_password(password, user["password_hash"]):
        return None
    db.execute("UPDATE usuarios SET last_login = now() WHERE id = %s", (user["id"],))
    return user


def create_user(username: str, password: str, is_admin: bool = False) -> Dict[str, Any]:
    existing = db.fetch_one("SELECT id FROM usuarios WHERE username = %s", (username,))
    if existing:
        raise ValueError(f"El usuario '{username}' ya existe")
    return db.execute_returning(
        "INSERT INTO usuarios (username, password_hash, is_admin) VALUES (%s, %s, %s) "
        "RETURNING id, username, is_admin, created_at",
        (username, hash_password(password), is_admin),
    )


def list_users() -> list:
    return db.fetch_all("SELECT id, username, is_admin, created_at, last_login FROM usuarios ORDER BY id")


def delete_user(user_id: int) -> None:
    db.execute("DELETE FROM usuarios WHERE id = %s", (user_id,))


# ── Dependencias FastAPI ────────────────────────────────────────────────

def get_current_user(request: Request) -> Dict[str, Any]:
    """Dependencia para proteger rutas del gestor. Lanza 401 si no hay
    sesión activa (el frontend redirige a /login)."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="No autenticado. Inicia sesión en /login.")
    return user


def require_admin(request: Request) -> Dict[str, Any]:
    user = get_current_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Requiere permisos de administrador.")
    return user
