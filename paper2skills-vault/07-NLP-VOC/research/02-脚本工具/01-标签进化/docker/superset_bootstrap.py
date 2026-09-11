"""Phase 7 D2 — Superset bootstrap (idempotent)

Drives Superset via REST API to bring it from blank state to "voc_bi connected
+ 6 views registered as datasets + sync queries enabled".

Steps (each idempotent):
  1. Login as admin → get JWT + CSRF
  2. Register voc_bi postgres database (skip if exists)
  3. Register 6 BI views as datasets (skip existing)
  4. Disable async to make SQL Lab work without celery results backend
  5. Smoke-test query against v_dept_kpi

Run after `docker compose up -d` + `superset fab create-admin` + `superset init`.

Usage:
  python superset_bootstrap.py
  python superset_bootstrap.py --reset    # wipe all datasets + recreate
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import urllib.request
import urllib.parse
import urllib.error


SUPERSET_URL = "http://localhost:8088"
ADMIN_USER = "admin"
ADMIN_PASS = "voc_admin_2026"

VOC_BI_DB_NAME = "voc_bi"
VOC_BI_SQLALCHEMY_URI = "postgresql+psycopg2://voc_bi:voc_bi_dev_2026@host.docker.internal:5432/voc_bi"

VIEWS = [
    "v_review_overview",
    "v_label_with_dept",
    "v_dept_topic_summary",
    "v_label_brand",
    "v_global_top_tags",
    "v_dept_kpi",
]


def _request(method: str, path: str, *, headers: dict, body: dict | None = None) -> tuple[int, dict | str]:
    url = f"{SUPERSET_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp else ""
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def login() -> tuple[str, str]:
    print("⏳ Login as admin", file=sys.stderr)
    code, resp = _request(
        "POST", "/api/v1/security/login",
        headers={"Content-Type": "application/json"},
        body={"username": ADMIN_USER, "password": ADMIN_PASS, "provider": "db", "refresh": True},
    )
    if code != 200 or not isinstance(resp, dict) or "access_token" not in resp:
        raise RuntimeError(f"login failed: {code} {resp}")
    token = resp["access_token"]
    print(f"   access_token: {token[:30]}...", file=sys.stderr)

    code, resp = _request(
        "GET", "/api/v1/security/csrf_token/",
        headers={"Authorization": f"Bearer {token}"},
    )
    csrf = resp["result"] if isinstance(resp, dict) and "result" in resp else ""
    return token, csrf


def auth_headers(token: str, csrf: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-CSRFToken": csrf,
        "Referer": SUPERSET_URL + "/",
        "Content-Type": "application/json",
    }


def list_databases(token: str, csrf: str) -> list[dict]:
    _, resp = _request("GET", "/api/v1/database/?q=(page_size:100)", headers=auth_headers(token, csrf))
    return resp.get("result", []) if isinstance(resp, dict) else []


def list_datasets(token: str, csrf: str) -> list[dict]:
    _, resp = _request("GET", "/api/v1/dataset/?q=(page_size:100)", headers=auth_headers(token, csrf))
    return resp.get("result", []) if isinstance(resp, dict) else []


def upsert_voc_bi_db(token: str, csrf: str) -> int:
    existing = list_databases(token, csrf)
    for db in existing:
        if db.get("database_name") == VOC_BI_DB_NAME:
            print(f"   voc_bi DB already exists (id={db['id']})", file=sys.stderr)
            db_id = db["id"]
            _, _ = _request(
                "PUT", f"/api/v1/database/{db_id}",
                headers=auth_headers(token, csrf),
                body={"allow_run_async": False},
            )
            return db_id

    print(f"⏳ Create voc_bi database", file=sys.stderr)
    code, resp = _request(
        "POST", "/api/v1/database/",
        headers=auth_headers(token, csrf),
        body={
            "database_name": VOC_BI_DB_NAME,
            "sqlalchemy_uri": VOC_BI_SQLALCHEMY_URI,
            "expose_in_sqllab": True,
            "allow_ctas": False,
            "allow_cvas": False,
            "allow_dml": False,
            "allow_run_async": False,
        },
    )
    if code not in (200, 201) or not isinstance(resp, dict) or "id" not in resp:
        raise RuntimeError(f"db create failed: {code} {resp}")
    print(f"   ✅ created id={resp['id']}", file=sys.stderr)
    return resp["id"]


def upsert_dataset(token: str, csrf: str, db_id: int, view_name: str) -> int:
    existing = list_datasets(token, csrf)
    for ds in existing:
        if ds.get("table_name") == view_name and ds.get("database", {}).get("id") == db_id:
            return ds["id"]

    code, resp = _request(
        "POST", "/api/v1/dataset/",
        headers=auth_headers(token, csrf),
        body={"database": db_id, "schema": "public", "table_name": view_name},
    )
    if code in (200, 201) and isinstance(resp, dict) and "id" in resp:
        return resp["id"]
    raise RuntimeError(f"dataset {view_name} failed: {code} {resp}")


def smoke_test_query(token: str, csrf: str, db_id: int) -> None:
    print(f"⏳ Smoke test: SELECT FROM v_dept_kpi", file=sys.stderr)
    code, resp = _request(
        "POST", "/api/v1/sqllab/execute/",
        headers=auth_headers(token, csrf),
        body={
            "database_id": db_id,
            "sql": "SELECT dept_owner, total_label_hits FROM v_dept_kpi ORDER BY total_label_hits DESC LIMIT 7;",
            "schema": "public",
            "tab": "bootstrap_smoke",
            "client_id": "bootstrap1",
        },
    )
    if isinstance(resp, dict) and "data" in resp:
        print(f"   ✅ {len(resp['data'])} rows returned", file=sys.stderr)
        for r in resp["data"]:
            print(f"     {r}", file=sys.stderr)
    else:
        raise RuntimeError(f"smoke query failed: {code} {resp}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 D2 Superset bootstrap")
    ap.add_argument("--reset", action="store_true",
                    help="(future) wipe + recreate all datasets")
    args = ap.parse_args(argv)

    if args.reset:
        print("⚠️ --reset not implemented yet (use Superset UI delete)", file=sys.stderr)

    token, csrf = login()
    db_id = upsert_voc_bi_db(token, csrf)

    print(f"⏳ Register {len(VIEWS)} views as datasets", file=sys.stderr)
    ids = {}
    for v in VIEWS:
        try:
            ids[v] = upsert_dataset(token, csrf, db_id, v)
            print(f"   {v} → id={ids[v]}", file=sys.stderr)
        except Exception as e:
            print(f"   {v} → ERR: {e}", file=sys.stderr)

    smoke_test_query(token, csrf, db_id)

    print(f"\n✅ Bootstrap complete", file=sys.stderr)
    print(f"   Superset: {SUPERSET_URL}", file=sys.stderr)
    print(f"   Login: {ADMIN_USER} / {ADMIN_PASS}", file=sys.stderr)
    print(f"   Database id: {db_id}", file=sys.stderr)
    print(f"   Datasets: {ids}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
