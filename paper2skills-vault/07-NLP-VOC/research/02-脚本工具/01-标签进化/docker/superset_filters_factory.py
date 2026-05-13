"""Phase 7 D4 — Superset native filter factory (idempotent)

Adds dashboard-level filters to enable interactive slicing:

  Overview dashboard (id=1):
    - data_source filter (multi-select)
    - product_line filter (multi-select)
    - proxy_nps filter (multi-select)
    Filters target v_review_overview, so chartsInScope = [4, 5] (pie charts).
    Charts 1, 2 (v_dept_kpi) and chart 3 (v_global_top_tags) are pre-aggregated
    and cannot be sliced by these filters.

  Per-dept dashboards (id=2..8, dataset v_dept_topic_summary):
    - polarity filter (single-select, multi)

Filters are added via PUT /api/v1/dashboard/{id} updating json_metadata's
`native_filter_configuration` array. The schema was probed by adding one filter
to dashboard 3 manually and verifying with GET (see D4.1 in progress report).

Idempotent: detects existing filters by name + skips.

Usage:
  python superset_filters_factory.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


SUPERSET_URL = "http://localhost:8088"
ADMIN_USER = "admin"
ADMIN_PASS = "voc_admin_2026"


DATASET_REVIEW = 1
DATASET_DEPT_TOPIC = 3

DEPT_DASHBOARDS = [
    {"id": 2, "dept": "全球客服与体验中心", "chart_name": "全球客服与体验中心 · Top 10 话题（按命中量）"},
    {"id": 3, "dept": "产品中心/品线", "chart_name": "产品中心/品线 · Top 10 话题（按命中量）"},
    {"id": 4, "dept": "供应链中心", "chart_name": "供应链中心 · Top 10 话题（按命中量）"},
    {"id": 5, "dept": "品牌市场中心", "chart_name": "品牌市场中心 · Top 10 话题（按命中量）"},
    {"id": 6, "dept": "电商运营部", "chart_name": "电商运营部 · Top 10 话题（按命中量）"},
    {"id": 7, "dept": "品控部", "chart_name": "品控部 · Top 10 话题（按命中量）"},
    {"id": 8, "dept": "质量与法规部", "chart_name": "质量与法规部 · Top 10 话题（按命中量）"},
]

OVERVIEW_CHART_NAMES = [
    "Overview · 数据源分布",
    "Overview · Proxy NPS 分布",
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
    code, resp = _request(
        "POST", "/api/v1/security/login",
        headers={"Content-Type": "application/json"},
        body={"username": ADMIN_USER, "password": ADMIN_PASS, "provider": "db", "refresh": True},
    )
    if code != 200 or "access_token" not in resp:
        raise RuntimeError(f"login: {code} {resp}")
    token = resp["access_token"]
    code2, csrf_resp = _request("GET", "/api/v1/security/csrf_token/",
                                 headers={"Authorization": f"Bearer {token}"})
    return token, csrf_resp.get("result", "") if isinstance(csrf_resp, dict) else ""


def auth_headers(token: str, csrf: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-CSRFToken": csrf,
        "Referer": SUPERSET_URL + "/",
        "Content-Type": "application/json",
    }


def build_filter(filter_id: str, name: str, dataset_id: int, column: str,
                 chart_ids: list[int], description: str) -> dict:
    return {
        "id": filter_id,
        "name": name,
        "type": "NATIVE_FILTER",
        "filterType": "filter_select",
        "targets": [{"datasetId": dataset_id, "column": {"name": column}}],
        "defaultDataMask": {"filterState": {"value": []}},
        "controlValues": {
            "multiSelect": True,
            "enableEmptyFilter": False,
            "defaultToFirstItem": False,
            "inverseSelection": False,
            "searchAllOptions": False,
        },
        "cascadeParentIds": [],
        "scope": {"rootPath": ["ROOT_ID"], "excluded": []},
        "description": description,
        "chartsInScope": chart_ids,
        "tabsInScope": [],
        "isInstant": True,
    }


def get_dashboard(token: str, csrf: str, dashboard_id: int) -> dict:
    _, resp = _request("GET", f"/api/v1/dashboard/{dashboard_id}",
                       headers=auth_headers(token, csrf))
    return resp.get("result", {}) if isinstance(resp, dict) else {}


def put_dashboard_metadata(token: str, csrf: str, dashboard_id: int, metadata: dict) -> None:
    code, resp = _request(
        "PUT", f"/api/v1/dashboard/{dashboard_id}",
        headers=auth_headers(token, csrf),
        body={"json_metadata": json.dumps(metadata, ensure_ascii=False)},
    )
    if code not in (200, 201):
        raise RuntimeError(f"PUT dashboard {dashboard_id} json_metadata failed: {code} {resp}")


def upsert_filters(token: str, csrf: str, dashboard_id: int,
                    desired_filters: list[dict], label: str) -> int:
    """Replace native_filter_configuration with desired list (idempotent by id).

    Returns count of filters in final config.
    """
    d = get_dashboard(token, csrf, dashboard_id)
    meta_raw = d.get("json_metadata", "{}") or "{}"
    try:
        meta = json.loads(meta_raw)
    except json.JSONDecodeError:
        meta = {}

    existing = meta.get("native_filter_configuration", []) or []
    existing_by_id = {f.get("id"): f for f in existing if isinstance(f, dict)}

    final = []
    desired_ids = {f["id"] for f in desired_filters}

    for f in desired_filters:
        if f["id"] in existing_by_id:
            print(f"   {label}: filter '{f['name']}' already present (kept)", file=sys.stderr)
        else:
            print(f"   {label}: adding filter '{f['name']}'", file=sys.stderr)
        final.append(f)

    for fid, fobj in existing_by_id.items():
        if fid not in desired_ids:
            print(f"   {label}: preserving unrelated filter '{fobj.get('name')}'", file=sys.stderr)
            final.append(fobj)

    meta["native_filter_configuration"] = final
    put_dashboard_metadata(token, csrf, dashboard_id, meta)
    return len(final)


def list_chart_ids(token: str, csrf: str) -> dict[str, int]:
    _, resp = _request("GET", "/api/v1/chart/?q=(page_size:100)",
                       headers=auth_headers(token, csrf))
    if not isinstance(resp, dict):
        return {}
    return {c["slice_name"]: c["id"] for c in resp.get("result", []) if "slice_name" in c}


def configure_overview(token: str, csrf: str, chart_ids_by_name: dict[str, int]) -> None:
    overview_ids = [chart_ids_by_name[n] for n in OVERVIEW_CHART_NAMES if n in chart_ids_by_name]
    if not overview_ids:
        print("   ⚠ Overview pie charts not found, skipping Overview filter setup", file=sys.stderr)
        return
    filters = [
        build_filter("NATIVE_FILTER-data-source", "数据源",
                     DATASET_REVIEW, "data_source", overview_ids,
                     "按数据来源（amazon / trustpilot / zendesk / momcozy / reddit）切片"),
        build_filter("NATIVE_FILTER-product-line", "产品线",
                     DATASET_REVIEW, "product_line", overview_ids,
                     "按产品品线（吸奶器 / 内衣服饰 / 等）切片"),
        build_filter("NATIVE_FILTER-proxy-nps", "Proxy NPS",
                     DATASET_REVIEW, "proxy_nps", overview_ids,
                     "按 Proxy NPS（promoter / passive / detractor）切片"),
    ]
    print(f"⏳ Overview dashboard (id=1, scope={overview_ids})", file=sys.stderr)
    n = upsert_filters(token, csrf, 1, filters, "Overview")
    print(f"   → {n} filter(s) total\n", file=sys.stderr)


def configure_dept_dashboards(token: str, csrf: str, chart_ids_by_name: dict[str, int]) -> None:
    for dd in DEPT_DASHBOARDS:
        chart_id = chart_ids_by_name.get(dd["chart_name"])
        if chart_id is None:
            print(f"   ⚠ Dept chart '{dd['chart_name']}' not found, skipping", file=sys.stderr)
            continue
        filters = [
            build_filter("NATIVE_FILTER-polarity", "情感极性",
                         DATASET_DEPT_TOPIC, "polarity", [chart_id],
                         "按标签极性（正向 / 负向 / 中性）切片"),
        ]
        label = f"Dept {dd['dept']}"
        print(f"⏳ {label} dashboard (id={dd['id']}, chart={chart_id})", file=sys.stderr)
        n = upsert_filters(token, csrf, dd["id"], filters, label)
        print(f"   → {n} filter(s) total\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 D4 Superset filter factory")
    args = ap.parse_args(argv)

    print("⏳ Login", file=sys.stderr)
    token, csrf = login()
    print("   ✅ token obtained\n", file=sys.stderr)

    chart_ids = list_chart_ids(token, csrf)
    print(f"   📊 found {len(chart_ids)} charts\n", file=sys.stderr)

    configure_overview(token, csrf, chart_ids)
    configure_dept_dashboards(token, csrf, chart_ids)

    print("\n✅ Done. Filters applied to 1 overview + 7 dept dashboards.", file=sys.stderr)
    print(f"   Browse: {SUPERSET_URL}/dashboard/list/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
