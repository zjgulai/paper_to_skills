import base64 as _b64
import binascii as _binascii
import hashlib
import hmac
import json
import os
import secrets as _secrets
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import urlencode as _urlencode

import requests
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

DB_PATH = os.environ.get("P2S_DB_PATH", "/opt/paper2skills/service/reports.db")
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET", "")
JWT_SECRET = os.environ.get("JWT_SECRET", _secrets.token_hex(32))
SITE_URL = os.environ.get("SITE_URL", "https://skills.lute-tlz-dddd.top")

router = APIRouter()


def _jwt_encode(payload: dict) -> str:
    header = _b64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b"=").decode()
    body = _b64.urlsafe_b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).rstrip(b"=").decode()
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64.urlsafe_b64encode(sig).rstrip(b'=').decode()}"


def _jwt_decode(token: str) -> dict | None:
    try:
        header, body, sig = token.split(".")
        expected = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
        actual = _b64.urlsafe_b64decode(sig + "==")
        if not hmac.compare_digest(expected, actual):
            return None
        payload = json.loads(_b64.urlsafe_b64decode(body + "=="))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except (ValueError, json.JSONDecodeError, _binascii.Error, UnicodeDecodeError):
        return None


@router.get("/auth/login")
async def auth_login():
    if not FEISHU_APP_ID:
        return JSONResponse({"error": "Feishu OAuth not configured. Set FEISHU_APP_ID and FEISHU_APP_SECRET."}, status_code=501)
    params = _urlencode({
        "app_id": FEISHU_APP_ID,
        "redirect_uri": f"{SITE_URL}/auth/callback",
        "response_type": "code",
        "state": _secrets.token_urlsafe(8),
    })
    return RedirectResponse(f"https://open.feishu.cn/open-apis/authen/v1/authorize?{params}")


@router.get("/auth/callback")
async def auth_callback(code: str = "", state: str = ""):
    if not code:
        return HTMLResponse("<script>window.location='/'</script>")
    token_resp = requests.post("https://open.feishu.cn/open-apis/authen/v1/oidc/access_token", json={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": FEISHU_APP_ID,
        "client_secret": FEISHU_APP_SECRET,
    }, timeout=10)
    token_data = token_resp.json().get("data", {})
    user_token = token_data.get("access_token", "")
    if not user_token:
        return HTMLResponse("<script>alert('登录失败，请重试');window.location='/'</script>")
    user_resp = requests.get(
        "https://open.feishu.cn/open-apis/authen/v1/user_info",
        headers={"Authorization": f"Bearer {user_token}"},
        timeout=10,
    )
    user_data = user_resp.json().get("data", {})
    feishu_id = user_data.get("open_id", "")
    name = user_data.get("name", "用户")
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO users (feishu_id, name, org, tier, created_at, last_seen, usage_count)
            VALUES (?,?,?,?,?,?,0)
            ON CONFLICT(feishu_id) DO UPDATE SET name=excluded.name, last_seen=excluded.last_seen
            """,
            (feishu_id, name, "", "free", now, now),
        )
        conn.commit()
    jwt = _jwt_encode({"sub": feishu_id, "name": name, "tier": "free", "exp": int(time.time()) + 7 * 86400})
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie("p2s_auth", jwt, max_age=7 * 86400, httponly=True, samesite="lax", secure=SITE_URL.startswith("https"))
    return resp


@router.get("/auth/me")
async def auth_me(request: Request):
    token = request.cookies.get("p2s_auth", "")
    payload = _jwt_decode(token) if token else None
    if not payload:
        return JSONResponse({"feishu_id": None, "name": None, "tier": "free"})
    return JSONResponse({"feishu_id": payload["sub"], "name": payload["name"], "tier": payload.get("tier", "free")})


@router.post("/auth/logout")
async def auth_logout():
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie("p2s_auth")
    return resp
