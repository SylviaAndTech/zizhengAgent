"""
登录鉴权：账号密码登录 + 服务端session token，走 Authorization: Bearer <token> 请求头
（不用cookie——前端和后端在开发环境是两个不同origin，跨源cookie在纯HTTP下受SameSite/Secure
策略限制很难可靠工作；token方案不受这个限制，前端存localStorage即可）。

密码哈希用标准库hashlib的PBKDF2-HMAC-SHA256，不引入bcrypt/passlib这类新依赖——PBKDF2-SHA256
是OWASP认可的密码哈希算法，迭代次数按OWASP 2023年的建议（60万次）设置。

现在只有一个固定账号（admin/Wzlg123456），但整个设计按多用户来做：db.py里的User是一张真正的
用户表，不是写死判断用户名密码，以后要加新用户，直接insert一行、不用改任何代码。

AUTH_ENABLED环境变量（默认true）可以整体关掉登录校验，测试阶段用；上线前要确认这个开关是开着的。
"""
import hashlib
import os
import secrets
from datetime import datetime, timedelta

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").strip().lower() not in ("false", "0", "no")
SESSION_TTL_HOURS = int(os.environ.get("AUTH_SESSION_TTL_HOURS", "168"))  # 默认7天过期
PBKDF2_ITERATIONS = 600_000

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = os.environ.get("AUTH_DEFAULT_ADMIN_PASSWORD", "Wzlg123456")


def hash_password(password: str, salt: bytes | None = None) -> str:
    """格式: pbkdf2_sha256$<迭代次数>$<salt的hex>$<哈希结果的hex>——把迭代次数存进哈希字符串里，
    以后想提高迭代次数时，旧密码的hash仍然可以按各自存的迭代次数正确校验，不用强制全员改密码。"""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algo, iterations, salt_hex, hash_hex = password_hash.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        # 用secrets.compare_digest而不是==，避免哈希比较耗时差异被用来猜测密码（时序攻击）
        return secrets.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


def create_session_token(db, user) -> str:
    from db import UserSession
    token = secrets.token_urlsafe(32)
    db.add(UserSession(
        token=token, user_id=user.id,
        expires_at=datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    ))
    db.commit()
    return token


def get_user_from_token(db, token: str):
    from db import UserSession, User
    if not token:
        return None
    session = db.query(UserSession).filter(UserSession.token == token).first()
    if not session or session.expires_at < datetime.utcnow():
        return None
    return db.query(User).filter(User.id == session.user_id, User.is_active.is_(True)).first()


def delete_session_token(db, token: str):
    from db import UserSession
    if token:
        db.query(UserSession).filter(UserSession.token == token).delete()
        db.commit()
