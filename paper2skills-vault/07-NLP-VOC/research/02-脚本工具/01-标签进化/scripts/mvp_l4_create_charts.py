#!/usr/bin/env python3
"""L4.3 · 5 Superset charts via API."""
import json, sys, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://voc.lute-tlz-dddd.top"
USER, PASS = "admin", "voc_admin_2026"

s = requests.Session(); s.verify = False
r = s.post(f"{URL}/api/v1/security/login",
           json={"username": USER, "password": PASS, "provider": "db", "refresh": True})
token = r.json()["access_token"]
s.headers.update({"Authorization": f"Bearer {token}"})
csrf = s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken": csrf, "Referer": URL})

# Datasets registered in step 2
DS = {
    "v_atomic_indicator_score": 7,
    "v_aipl_node_score":        8,
    "v_country_dept_health":    9,
    "v_proxy_nps_calibrated":  10,
}

charts = []

# -------- Chart 1: KPI 总览 (单卡 big_number = 总 reviews) --------
charts.append({
    "slice_name": "VOC D-Health · 总 reviews 数",
    "viz_type": "big_number_total",
    "datasource_id": DS["v_proxy_nps_calibrated"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_proxy_nps_calibrated"]}__table',
        "viz_type": "big_number_total",
        "metric": {
            "expressionType": "SQL",
            "label": "总 reviews",
            "sqlExpression": "COUNT(DISTINCT review_id)",
        },
        "adhoc_filters": [],
        "header_font_size": 0.4,
        "subheader_font_size": 0.15,
        "y_axis_format": "SMART_NUMBER",
    },
})

# -------- Chart 2: KPI · 平均 SAT 分 --------
charts.append({
    "slice_name": "VOC D-Health · 平均 SAT 分",
    "viz_type": "big_number_total",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "big_number_total",
        "metric": {
            "expressionType": "SQL",
            "label": "avg SAT score",
            "sqlExpression": "ROUND(AVG(sat_score)::numeric, 2)",
        },
        "adhoc_filters": [],
        "y_axis_format": ".2f",
    },
})

# -------- Chart 3: KPI · 校准 NPS --------
charts.append({
    "slice_name": "VOC D-Health · 校准 NPS",
    "viz_type": "big_number_total",
    "datasource_id": DS["v_proxy_nps_calibrated"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_proxy_nps_calibrated"]}__table',
        "viz_type": "big_number_total",
        "metric": {
            "expressionType": "SQL",
            "label": "NPS 校准",
            "sqlExpression": "ROUND(AVG(nps_score_calibrated)::numeric, 4)",
        },
        "adhoc_filters": [],
        "y_axis_format": ".4f",
    },
})

# -------- Chart 4: AIPL 节点 by country (echarts_timeseries_bar) --------
charts.append({
    "slice_name": "VOC D-Health · AIPL 节点健康度 by country",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_aipl_node_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_aipl_node_score"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["country"],
        "columns": ["aipl_node"],
        "metrics": [{
            "expressionType": "SQL",
            "label": "avg node_score",
            "sqlExpression": "AVG(node_score)",
        }],
        "adhoc_filters": [{
            "expressionType": "SIMPLE", "subject": "country",
            "operator": "IS NOT NULL", "comparator": None, "clause": "WHERE",
        }],
        "row_limit": 50,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "x_axis_label": "country",
        "y_axis_label": "AIPL node score",
    },
})

# -------- Chart 5: SAT Top10 + Bottom10 --------
charts.append({
    "slice_name": "VOC D-Health · SAT 排行 Top/Bottom 10",
    "viz_type": "table",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "table",
        "groupby": ["atomic_indicator_id"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",     "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "score_w",  "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
            {"expressionType": "SQL", "label": "pct_neg",  "sqlExpression": "ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [{
            "expressionType": "SIMPLE", "subject": "total_hits",
            "operator": ">", "comparator": "0", "clause": "HAVING",
            "sqlExpression": "SUM(total_hits) > 0",
        }],
        "row_limit": 50,
        "table_timestamp_format": "smart_date",
        "include_search": True,
    },
})

# -------- Chart 6: Country × Dept 热力 --------
charts.append({
    "slice_name": "VOC D-Health · Country × Dept 负向率热力",
    "viz_type": "heatmap",
    "datasource_id": DS["v_country_dept_health"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_country_dept_health"]}__table',
        "viz_type": "heatmap",
        "all_columns_x": "country",
        "all_columns_y": "dept_owner",
        "metric": {
            "expressionType": "SQL",
            "label": "avg pct_negative",
            "sqlExpression": "AVG(pct_negative)",
        },
        "adhoc_filters": [{
            "expressionType": "SIMPLE", "subject": "label_hits",
            "operator": ">", "comparator": "0", "clause": "WHERE",
        }],
        "linear_color_scheme": "fire",
        "xscale_interval": -1, "yscale_interval": -1,
        "canvas_image_rendering": "pixelated",
        "normalize_across": "heatmap",
        "left_margin": "auto", "bottom_margin": "auto",
        "y_axis_bounds": [None, None], "y_axis_format": ".2f",
        "row_limit": 100,
    },
})

# -------- Chart 7: NPS by source (raw vs cal) --------
charts.append({
    "slice_name": "VOC D-Health · 校准前后 NPS by source",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_proxy_nps_calibrated"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_proxy_nps_calibrated"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["data_source"],
        "metrics": [
            {"expressionType": "SQL", "label": "NPS 原始",  "sqlExpression": "AVG(nps_score_raw)"},
            {"expressionType": "SQL", "label": "NPS 校准",  "sqlExpression": "AVG(nps_score_calibrated)"},
        ],
        "adhoc_filters": [],
        "row_limit": 20,
        "show_legend": True,
        "x_axis_label": "data_source",
        "y_axis_label": "NPS",
        "color_scheme": "supersetColors",
    },
})

print(f"creating {len(charts)} charts...")
created = []
for c in charts:
    body = {
        "slice_name": c["slice_name"],
        "viz_type":   c["viz_type"],
        "datasource_id":   c["datasource_id"],
        "datasource_type": c["datasource_type"],
        "params": json.dumps(c["params"]),
    }
    r = s.post(f"{URL}/api/v1/chart/", json=body)
    try:
        rj = r.json()
    except Exception:
        rj = {"raw": r.text[:300]}
    cid = rj.get("id")
    print(f"  [{r.status_code}] {c['slice_name']}  id={cid}")
    if cid:
        created.append({"id": cid, "name": c["slice_name"], "viz": c["viz_type"]})
    else:
        print(f"    ERR: {json.dumps(rj)[:500]}")

print()
print("=== CREATED ===")
print(json.dumps(created, ensure_ascii=False, indent=2))

with open("/tmp/mvp_l4_chart_ids.json", "w") as f:
    json.dump(created, f, ensure_ascii=False, indent=2)
