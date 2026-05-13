"""Phase 7 D3 — Superset chart + dashboard factory (idempotent)

Creates a curated set of charts and 8 dashboards (1 overview + 7 dept) via
Superset REST API. Replaces the static D10 HTML dashboard with interactive,
filterable Superset views backed by live voc_bi.

Charts created:
  Overview level (read v_dept_kpi / v_global_top_tags / v_review_overview):
    - chart_dept_total_hits          horizontal bar: dept × total_hits
    - chart_dept_polarity_breakdown  stacked bar: dept × {neg/pos/neu}
    - chart_top_global_tags          table: Top-30 全局标签
    - chart_source_distribution      pie: data_source 占比
    - chart_nps_distribution         pie: proxy_nps 占比

  Per-dept (read v_dept_topic_summary, filtered by dept_owner):
    - chart_dept_<dept>_topic_top10  horizontal bar: tag × hit_count

Dashboards:
  overview                        含 5 个总览图表
  dept_全球客服与体验中心 / 产品中心/品线 / ...   含该部门 topic_top10 + 通用 KPI 卡片

Idempotent: skips charts/dashboards that already exist.

Run after Phase 7 D2 (superset_bootstrap.py) completes.

Usage:
  python superset_charts_factory.py
  python superset_charts_factory.py --reset    # delete all existing first
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


SUPERSET_URL = "http://localhost:8088"
ADMIN_USER = "admin"
ADMIN_PASS = "voc_admin_2026"

DEPARTMENTS = [
    "全球客服与体验中心", "产品中心/品线", "供应链中心",
    "品牌市场中心", "电商运营部", "品控部", "质量与法规部",
]

DATASET_ID = {
    "v_review_overview": 1,
    "v_label_with_dept": 2,
    "v_dept_topic_summary": 3,
    "v_label_brand": 4,
    "v_global_top_tags": 5,
    "v_dept_kpi": 6,
}


def _request(method: str, path: str, *, headers: dict, body: dict | None = None) -> tuple[int, Any]:
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
    code2, resp2 = _request("GET", "/api/v1/security/csrf_token/",
                             headers={"Authorization": f"Bearer {token}"})
    csrf = resp2.get("result", "") if isinstance(resp2, dict) else ""
    return token, csrf


def auth_headers(token: str, csrf: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-CSRFToken": csrf,
        "Referer": SUPERSET_URL + "/",
        "Content-Type": "application/json",
    }


def list_charts(token: str, csrf: str) -> list[dict]:
    _, resp = _request("GET", "/api/v1/chart/?q=(page_size:200)", headers=auth_headers(token, csrf))
    return resp.get("result", []) if isinstance(resp, dict) else []


def list_dashboards(token: str, csrf: str) -> list[dict]:
    _, resp = _request("GET", "/api/v1/dashboard/?q=(page_size:50)", headers=auth_headers(token, csrf))
    return resp.get("result", []) if isinstance(resp, dict) else []


def _params_dist_bar_horizontal(metric_label: str, groupby: str, limit: int = 7) -> dict:
    """Horizontal bar (dist_bar) sorted by metric desc."""
    return {
        "viz_type": "dist_bar",
        "metrics": [
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": metric_label, "type": "BIGINT"},
                "aggregate": "SUM",
                "label": metric_label,
            }
        ],
        "groupby": [groupby],
        "row_limit": limit,
        "order_desc": True,
        "show_legend": False,
        "bar_stacked": False,
        "show_bar_value": True,
        "color_scheme": "supersetColors",
        "extra_form_data": {},
        "adhoc_filters": [],
    }


def _params_pie(metric_label: str, groupby: str, limit: int = 10) -> dict:
    # Use COUNT(*) instead of COUNT(review_id): v_review_overview is a pre-aggregated
    # view that doesn't expose review_id. SQL-form metric avoids the column dependency.
    return {
        "viz_type": "pie",
        "metric": {
            "expressionType": "SQL",
            "sqlExpression": "COUNT(*)",
            "label": metric_label,
        },
        "groupby": [groupby],
        "row_limit": limit,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "show_labels": True,
        "label_type": "key_percent",
        "donut": True,
        "innerRadius": 30,
        "outerRadius": 70,
        "adhoc_filters": [],
    }


def _params_table(columns: list[str], order_by: str | None, limit: int = 50) -> dict:
    p = {
        "viz_type": "table",
        "all_columns": columns,
        "row_limit": limit,
        "include_search": True,
        "show_cell_bars": True,
        "color_pn": True,
        "adhoc_filters": [],
    }
    if order_by:
        p["timeseries_limit_metric"] = {
            "expressionType": "SIMPLE",
            "column": {"column_name": order_by, "type": "BIGINT"},
            "aggregate": "SUM",
            "label": order_by,
        }
    return p


def _params_dept_topic_bar(dept: str, limit: int = 10) -> dict:
    """Per-dept horizontal bar of top-N tags by hit_count."""
    return {
        "viz_type": "dist_bar",
        "metrics": [
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": "hit_count", "type": "BIGINT"},
                "aggregate": "SUM",
                "label": "hit_count",
            }
        ],
        "groupby": ["tag_cn"],
        "row_limit": limit,
        "order_desc": True,
        "show_legend": False,
        "show_bar_value": True,
        "color_scheme": "supersetColors",
        "adhoc_filters": [
            {
                "expressionType": "SIMPLE",
                "subject": "dept_owner",
                "operator": "==",
                "comparator": dept,
                "clause": "WHERE",
            }
        ],
        "extra_form_data": {},
    }


def _params_stacked_polarity() -> dict:
    return {
        "viz_type": "dist_bar",
        "metrics": [
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": "hits_negative", "type": "BIGINT"},
                "aggregate": "SUM",
                "label": "负向",
            },
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": "hits_positive", "type": "BIGINT"},
                "aggregate": "SUM",
                "label": "正向",
            },
            {
                "expressionType": "SIMPLE",
                "column": {"column_name": "hits_neutral", "type": "BIGINT"},
                "aggregate": "SUM",
                "label": "中性",
            },
        ],
        "groupby": ["dept_owner"],
        "row_limit": 7,
        "order_desc": True,
        "show_legend": True,
        "bar_stacked": True,
        "show_bar_value": False,
        "color_scheme": "googleCategory10c",
        "adhoc_filters": [],
    }


def chart_specs() -> list[dict]:
    specs: list[dict] = []

    specs.append({
        "tag": "overview_dept_total_hits",
        "name": "Overview · 7 部门标签命中数",
        "datasource_id": DATASET_ID["v_dept_kpi"],
        "datasource_type": "table",
        "viz_type": "dist_bar",
        "params": _params_dist_bar_horizontal("total_label_hits", "dept_owner"),
    })

    specs.append({
        "tag": "overview_dept_polarity",
        "name": "Overview · 部门极性分布",
        "datasource_id": DATASET_ID["v_dept_kpi"],
        "datasource_type": "table",
        "viz_type": "dist_bar",
        "params": _params_stacked_polarity(),
    })

    specs.append({
        "tag": "overview_top_global_tags",
        "name": "Overview · Top-30 全局标签",
        "datasource_id": DATASET_ID["v_global_top_tags"],
        "datasource_type": "table",
        "viz_type": "table",
        "params": _params_table(
            columns=["tag_id", "tag_cn", "tag_en", "dept_owner", "polarity", "hit_count", "avg_confidence"],
            order_by="hit_count",
            limit=30,
        ),
    })

    specs.append({
        "tag": "overview_source_pie",
        "name": "Overview · 数据源分布",
        "datasource_id": DATASET_ID["v_review_overview"],
        "datasource_type": "table",
        "viz_type": "pie",
        "params": _params_pie("评论数", "data_source"),
    })

    specs.append({
        "tag": "overview_nps_pie",
        "name": "Overview · Proxy NPS 分布",
        "datasource_id": DATASET_ID["v_review_overview"],
        "datasource_type": "table",
        "viz_type": "pie",
        "params": _params_pie("评论数", "proxy_nps"),
    })

    for dept in DEPARTMENTS:
        specs.append({
            "tag": f"dept_{dept}_topic_top10",
            "name": f"{dept} · Top 10 话题（按命中量）",
            "datasource_id": DATASET_ID["v_dept_topic_summary"],
            "datasource_type": "table",
            "viz_type": "dist_bar",
            "params": _params_dept_topic_bar(dept, limit=10),
        })

    return specs


def upsert_chart(token: str, csrf: str, spec: dict, existing_by_name: dict[str, int]) -> int:
    name = spec["name"]
    if name in existing_by_name:
        return existing_by_name[name]
    body = {
        "slice_name": name,
        "datasource_id": spec["datasource_id"],
        "datasource_type": spec["datasource_type"],
        "viz_type": spec["viz_type"],
        "params": json.dumps(spec["params"], ensure_ascii=False),
    }
    code, resp = _request(
        "POST", "/api/v1/chart/",
        headers=auth_headers(token, csrf),
        body=body,
    )
    if code in (200, 201) and isinstance(resp, dict) and "id" in resp:
        return resp["id"]
    raise RuntimeError(f"chart create '{name}' failed: {code} {resp}")


def upsert_dashboard(
    token: str, csrf: str, title: str, slug: str,
    existing_by_title: dict[str, int],
) -> int:
    if title in existing_by_title:
        return existing_by_title[title]
    code, resp = _request(
        "POST", "/api/v1/dashboard/",
        headers=auth_headers(token, csrf),
        body={
            "dashboard_title": title,
            "slug": slug,
            "published": True,
            "json_metadata": json.dumps({"native_filter_configuration": []}),
        },
    )
    if code in (200, 201) and "id" in resp:
        return resp["id"]
    raise RuntimeError(f"dashboard create '{title}' failed: {code} {resp}")


def _build_position_json(chart_ids: list[int], chart_slugs: list[str]) -> dict:
    """Build Superset dashboard position_json: a tree with ROOT → GRID → ROW → CHART(s).

    Superset positions charts in a hierarchical layout structure. Minimum viable
    layout: 1 ROOT, 1 GRID inside, 1 ROW per chart, each ROW contains 1 CHART.
    """
    children: list[str] = []
    chart_blocks: dict[str, dict] = {}
    row_blocks: dict[str, dict] = {}

    for cid, slug in zip(chart_ids, chart_slugs):
        row_id = f"ROW-{cid}"
        chart_id = f"CHART-{cid}"
        children.append(row_id)
        row_blocks[row_id] = {
            "type": "ROW",
            "id": row_id,
            "children": [chart_id],
            "parents": ["ROOT_ID", "GRID_ID"],
            "meta": {"background": "BACKGROUND_TRANSPARENT"},
        }
        chart_blocks[chart_id] = {
            "type": "CHART",
            "id": chart_id,
            "children": [],
            "parents": ["ROOT_ID", "GRID_ID", row_id],
            "meta": {
                "width": 12,
                "height": 50,
                "chartId": cid,
                "sliceName": slug,
            },
        }

    layout = {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {
            "type": "ROOT",
            "id": "ROOT_ID",
            "children": ["GRID_ID"],
        },
        "GRID_ID": {
            "type": "GRID",
            "id": "GRID_ID",
            "children": children,
            "parents": ["ROOT_ID"],
        },
    }
    layout.update(row_blocks)
    layout.update(chart_blocks)
    return layout


def attach_charts_to_dashboard(
    token: str, csrf: str, dashboard_id: int, chart_ids: list[int], chart_slugs: list[str],
) -> None:
    """Attach charts to dashboard via 2 calls (Superset's actual mechanism):

    1. PUT /api/v1/chart/{id} with {"dashboards": [dashboard_id]} establishes the
       canonical chart→dashboard ownership link.
    2. PUT /api/v1/dashboard/{id} with position_json sets the visual layout.

    Skipping step 1 results in "无图表定义" empty chart placeholders.
    """
    if not chart_ids:
        return
    for cid in chart_ids:
        code, resp = _request(
            "PUT", f"/api/v1/chart/{cid}",
            headers=auth_headers(token, csrf),
            body={"dashboards": [dashboard_id]},
        )
        if code not in (200, 201):
            raise RuntimeError(f"PUT chart {cid}.dashboards failed: {code} {resp}")

    layout = _build_position_json(chart_ids, chart_slugs)
    code, resp = _request(
        "PUT", f"/api/v1/dashboard/{dashboard_id}",
        headers=auth_headers(token, csrf),
        body={"position_json": json.dumps(layout)},
    )
    if code not in (200, 201):
        raise RuntimeError(f"PUT dashboard {dashboard_id}.position_json failed: {code} {resp}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 7 D3 Superset chart + dashboard factory")
    ap.add_argument("--reset", action="store_true",
                    help="(future) delete all existing charts/dashboards first")
    args = ap.parse_args(argv)

    print("⏳ Login as admin", file=sys.stderr)
    token, csrf = login()

    if args.reset:
        print("⚠️ --reset not implemented (use UI)", file=sys.stderr)

    print("⏳ Listing existing charts", file=sys.stderr)
    existing_charts = {c["slice_name"]: c["id"] for c in list_charts(token, csrf)}
    print(f"   existing: {len(existing_charts)}", file=sys.stderr)

    print("⏳ Creating charts", file=sys.stderr)
    chart_ids: dict[str, int] = {}
    for spec in chart_specs():
        try:
            cid = upsert_chart(token, csrf, spec, existing_charts)
            chart_ids[spec["tag"]] = cid
            existing_charts[spec["name"]] = cid
            print(f"   {spec['tag']} → id={cid}", file=sys.stderr)
        except Exception as e:
            print(f"   {spec['tag']} → ERR: {str(e)[:200]}", file=sys.stderr)

    print(f"\n📊 Total charts: {len(chart_ids)} / {len(chart_specs())}", file=sys.stderr)

    print("\n⏳ Listing existing dashboards", file=sys.stderr)
    existing_dashboards = {d["dashboard_title"]: d["id"] for d in list_dashboards(token, csrf)}

    overview_id = upsert_dashboard(
        token, csrf, "VOC Overview · 全局总览", "voc-overview", existing_dashboards,
    )
    overview_chart_tags = [
        ("overview_dept_total_hits", "Overview · 7 部门标签命中数"),
        ("overview_dept_polarity", "Overview · 部门极性分布"),
        ("overview_top_global_tags", "Overview · Top-30 全局标签"),
        ("overview_source_pie", "Overview · 数据源分布"),
        ("overview_nps_pie", "Overview · Proxy NPS 分布"),
    ]
    overview_pairs = [
        (chart_ids.get(tag), name)
        for tag, name in overview_chart_tags
        if chart_ids.get(tag)
    ]
    if overview_pairs:
        attach_charts_to_dashboard(
            token, csrf, overview_id,
            [cid for cid, _ in overview_pairs],
            [name for _, name in overview_pairs],
        )
    existing_dashboards["VOC Overview · 全局总览"] = overview_id
    print(f"   overview dashboard → id={overview_id} ({len(overview_pairs)} charts)", file=sys.stderr)

    for dept in DEPARTMENTS:
        title = f"VOC · {dept}"
        d_id = upsert_dashboard(token, csrf, title, f"voc-dept-{dept}", existing_dashboards)
        existing_dashboards[title] = d_id
        tag = f"dept_{dept}_topic_top10"
        chart_id = chart_ids.get(tag)
        chart_name = f"{dept} · Top 10 话题（按命中量）"
        if chart_id:
            attach_charts_to_dashboard(token, csrf, d_id, [chart_id], [chart_name])
        print(f"   {title} → id={d_id} ({1 if chart_id else 0} charts)", file=sys.stderr)

    print(f"\n✅ Done", file=sys.stderr)
    print(f"   Charts: {len(chart_ids)}", file=sys.stderr)
    print(f"   Dashboards: 1 overview + 7 dept = 8", file=sys.stderr)
    print(f"   Browse: {SUPERSET_URL}/dashboard/list/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
