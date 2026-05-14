#!/usr/bin/env python3
"""L5.3.1 · 15 D-Diag charts via Superset API.

3 themes × 5 charts:
  Theme 1 (Product · 产品力诊断):  P-1~P-5
  Theme 2 (Service · 服务力诊断):  S-1~S-5
  Theme 3 (Brand   · 内容/品牌力):  B-1~B-5

Datasets reused from L4:
  7 = v_atomic_indicator_score
  8 = v_aipl_node_score
  9 = v_country_dept_health
 10 = v_proxy_nps_calibrated
"""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://voc.lute-tlz-dddd.top"
USER, PASS = "admin", "voc_admin_2026"

s = requests.Session(); s.verify = False
tok = s.post(f"{URL}/api/v1/security/login",
             json={"username": USER, "password": PASS, "provider": "db", "refresh": True}
             ).json()["access_token"]
s.headers.update({"Authorization": f"Bearer {tok}"})
csrf = s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken": csrf, "Referer": URL})

DS = {
    "v_atomic_indicator_score": 7,
    "v_aipl_node_score":        8,
    "v_country_dept_health":    9,
    "v_proxy_nps_calibrated":  10,
}

charts = []

# ============================================================================
# THEME 1 · 产品力诊断 (P-1 ~ P-5) — dept_owner = 产品中心
# ============================================================================

# P-1 · 产品中心 SAT 失血 Top 10（按体量 × score 加权）
charts.append({
    "theme": "Product",
    "code":  "P-1",
    "slice_name": "VOC D-Diag Product · P-1 产品中心 SAT 失血 Top 10",
    "viz_type": "table",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "table",
        "groupby": ["atomic_indicator_id"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "score_w",
             "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
            {"expressionType": "SQL", "label": "pct_neg",
             "sqlExpression": "ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "LIKE", "comparator": "SAT_L1_%", "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "total_hits",
             "operator": ">", "comparator": "0", "clause": "HAVING",
             "sqlExpression": "SUM(total_hits) > 0"},
        ],
        "row_limit": 10,
        "order_desc": True,
        "include_search": True,
        "table_timestamp_format": "smart_date",
    },
})

# P-2 · 产品中心 × 国家 负向率 热力
charts.append({
    "theme": "Product",
    "code":  "P-2",
    "slice_name": "VOC D-Diag Product · P-2 产品中心 × 国家 负向率热力",
    "viz_type": "heatmap",
    "datasource_id": DS["v_country_dept_health"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_country_dept_health"]}__table',
        "viz_type": "heatmap",
        "all_columns_x": "country",
        "all_columns_y": "month",
        "metric": {"expressionType": "SQL", "label": "pct_negative",
                   "sqlExpression": "AVG(pct_negative)"},
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "dept_owner",
             "operator": "==", "comparator": "产品中心", "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "label_hits",
             "operator": ">", "comparator": "0", "clause": "WHERE"},
        ],
        "linear_color_scheme": "fire",
        "xscale_interval": -1, "yscale_interval": -1,
        "canvas_image_rendering": "pixelated",
        "normalize_across": "heatmap",
        "left_margin": "auto", "bottom_margin": "auto",
        "y_axis_bounds": [None, None], "y_axis_format": ".2f",
        "row_limit": 200,
    },
})

# P-3 · 产品中心 × product_line  体量 + 负向率交叉
charts.append({
    "theme": "Product",
    "code":  "P-3",
    "slice_name": "VOC D-Diag Product · P-3 产品中心 × 品类 体量+负向率",
    "viz_type": "table",
    "datasource_id": DS["v_country_dept_health"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_country_dept_health"]}__table',
        "viz_type": "table",
        "groupby": ["country"],  # column-grain not available; we collapse by country here
        "metrics": [
            {"expressionType": "SQL", "label": "reviews",
             "sqlExpression": "SUM(reviews)"},
            {"expressionType": "SQL", "label": "label_hits",
             "sqlExpression": "SUM(label_hits)"},
            {"expressionType": "SQL", "label": "hits_negative",
             "sqlExpression": "SUM(hits_negative)"},
            {"expressionType": "SQL", "label": "pct_negative",
             "sqlExpression": "ROUND((100.0 * SUM(hits_negative) / NULLIF(SUM(label_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "dept_owner",
             "operator": "==", "comparator": "产品中心", "clause": "WHERE"},
        ],
        "row_limit": 50,
        "order_desc": True,
        "include_search": False,
    },
})

# P-4 · L1 节点（使用层）所有 SAT 表现
charts.append({
    "theme": "Product",
    "code":  "P-4",
    "slice_name": "VOC D-Diag Product · P-4 L1 使用层 SAT 全量表现",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["atomic_indicator_id"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "neg_pct",
             "sqlExpression": "ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "LIKE", "comparator": "SAT_L1_%", "clause": "WHERE"},
        ],
        "row_limit": 30,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "x_axis_label": "SAT_L1_xx",
        "y_axis_label": "hits / neg_pct",
        "bar_stacked": False,
    },
})

# P-5 · 时间趋势 month × 负向率（产品中心）
charts.append({
    "theme": "Product",
    "code":  "P-5",
    "slice_name": "VOC D-Diag Product · P-5 产品中心 月度负向率趋势",
    "viz_type": "line",
    "datasource_id": DS["v_country_dept_health"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_country_dept_health"]}__table',
        "viz_type": "line",
        "groupby": ["country"],
        "metrics": [{"expressionType": "SQL", "label": "pct_negative",
                     "sqlExpression": "AVG(pct_negative)"}],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "dept_owner",
             "operator": "==", "comparator": "产品中心", "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "label_hits",
             "operator": ">", "comparator": "0", "clause": "WHERE"},
        ],
        "granularity_sqla": "month",
        "time_grain_sqla": "P1M",
        "row_limit": 5000,
        "show_legend": True,
        "color_scheme": "supersetColors",
        "x_axis_label": "month",
        "y_axis_label": "pct_negative (%)",
        "show_brush": "auto",
    },
})

# ============================================================================
# THEME 2 · 服务力诊断 (S-1 ~ S-5)
#   focus depts: 品质管理中心 · 仓储物流部 · 全球客服中心
# ============================================================================

# S-1 · 三服务部门 × 国家  负向率热力
charts.append({
    "theme": "Service",
    "code":  "S-1",
    "slice_name": "VOC D-Diag Service · S-1 服务三部门 × 国家 负向率热力",
    "viz_type": "heatmap",
    "datasource_id": DS["v_country_dept_health"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_country_dept_health"]}__table',
        "viz_type": "heatmap",
        "all_columns_x": "country",
        "all_columns_y": "dept_owner",
        "metric": {"expressionType": "SQL", "label": "pct_negative",
                   "sqlExpression": "AVG(pct_negative)"},
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "dept_owner",
             "operator": "IN", "comparator": ["品质管理中心","仓储物流部","全球客服中心"],
             "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "label_hits",
             "operator": ">", "comparator": "0", "clause": "WHERE"},
        ],
        "linear_color_scheme": "fire",
        "xscale_interval": -1, "yscale_interval": -1,
        "canvas_image_rendering": "pixelated",
        "normalize_across": "heatmap",
        "left_margin": "auto", "bottom_margin": "auto",
        "y_axis_format": ".2f",
        "row_limit": 200,
    },
})

# S-2 · L3 节点（服务层）SAT 失血 Top 10
charts.append({
    "theme": "Service",
    "code":  "S-2",
    "slice_name": "VOC D-Diag Service · S-2 L3 服务层 SAT 失血 Top 10",
    "viz_type": "table",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "table",
        "groupby": ["atomic_indicator_id"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "score_w",
             "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
            {"expressionType": "SQL", "label": "pct_neg",
             "sqlExpression": "ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "LIKE", "comparator": "SAT_L3_%", "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "total_hits",
             "operator": ">", "comparator": "0", "clause": "HAVING",
             "sqlExpression": "SUM(total_hits) > 0"},
        ],
        "row_limit": 10,
        "order_desc": True,
        "include_search": True,
    },
})

# S-3 · 数据源 × 国家 detractor 倾向矩阵 (校准 NPS)
charts.append({
    "theme": "Service",
    "code":  "S-3",
    "slice_name": "VOC D-Diag Service · S-3 数据源 × 国家 NPS 校准矩阵",
    "viz_type": "heatmap",
    "datasource_id": DS["v_proxy_nps_calibrated"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_proxy_nps_calibrated"]}__table',
        "viz_type": "heatmap",
        "all_columns_x": "country",
        "all_columns_y": "data_source",
        "metric": {"expressionType": "SQL", "label": "nps_calibrated",
                   "sqlExpression": "AVG(nps_score_calibrated)"},
        "adhoc_filters": [],
        "linear_color_scheme": "fire",
        "xscale_interval": -1, "yscale_interval": -1,
        "canvas_image_rendering": "pixelated",
        "normalize_across": "heatmap",
        "left_margin": "auto", "bottom_margin": "auto",
        "y_axis_format": ".2f",
        "row_limit": 200,
    },
})

# S-4 · 仓储物流部 SAT 明细排行
charts.append({
    "theme": "Service",
    "code":  "S-4",
    "slice_name": "VOC D-Diag Service · S-4 仓储物流 SAT 明细排行",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["atomic_indicator_id"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "score_w",
             "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            # SAT_L3 物流相关 SAT 是 SAT_L3_01（物流），SAT_L3_03（包装），SAT_L3_06（配送），SAT_L3_14
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "IN",
             "comparator": ["SAT_L3_01","SAT_L3_03","SAT_L3_06","SAT_L3_14","SAT_L3_04","SAT_L3_07"],
             "clause": "WHERE"},
        ],
        "row_limit": 30,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "x_axis_label": "SAT_L3 物流类",
        "y_axis_label": "hits / score_w",
        "bar_stacked": False,
    },
})

# S-5 · 品质问题分布饼图（SAT_L1_01 / L1_02 等核心质量类切分）
charts.append({
    "theme": "Service",
    "code":  "S-5",
    "slice_name": "VOC D-Diag Service · S-5 核心质量 SAT 体量分布",
    "viz_type": "pie",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "pie",
        "groupby": ["atomic_indicator_id"],
        "metric": {"expressionType": "SQL", "label": "hits",
                   "sqlExpression": "SUM(total_hits)"},
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "IN",
             "comparator": ["SAT_L1_01","SAT_L1_02","SAT_L1_03","SAT_L1_04","SAT_L1_11","SAT_L1_13","SAT_L1_16"],
             "clause": "WHERE"},
        ],
        "row_limit": 20,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "donut": True,
        "label_type": "key_percent",
    },
})

# ============================================================================
# THEME 3 · 内容/品牌力诊断 (B-1 ~ B-5)
# ============================================================================

# B-1 · 品牌市场中心 SAT 表现 Top 10
charts.append({
    "theme": "Brand",
    "code":  "B-1",
    "slice_name": "VOC D-Diag Brand · B-1 品牌市场中心 SAT Top 10",
    "viz_type": "table",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "table",
        "groupby": ["atomic_indicator_id"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "score_w",
             "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
            {"expressionType": "SQL", "label": "pct_neg",
             "sqlExpression": "ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            # 品牌相关 SAT 主要分布在 L4（品牌层）+ L3_17（内容/社群）
            {"expressionType": "SQL", "subject": "atomic_indicator_id",
             "operator": "==", "comparator": None, "clause": "WHERE",
             "sqlExpression": "(atomic_indicator_id LIKE 'SAT_L4_%' OR atomic_indicator_id IN ('SAT_L3_17'))"},
            {"expressionType": "SIMPLE", "subject": "total_hits",
             "operator": ">", "comparator": "0", "clause": "HAVING",
             "sqlExpression": "SUM(total_hits) > 0"},
        ],
        "row_limit": 10,
        "order_desc": True,
        "include_search": True,
    },
})

# B-2 · L4 节点 SAT × 国家 分布
charts.append({
    "theme": "Brand",
    "code":  "B-2",
    "slice_name": "VOC D-Diag Brand · B-2 L4 品牌层 SAT × 国家",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["country"],
        "columns": ["atomic_indicator_id"],
        "metrics": [{"expressionType": "SQL", "label": "score_w",
                     "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"}],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "LIKE", "comparator": "SAT_L4_%", "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "country",
             "operator": "IS NOT NULL", "comparator": None, "clause": "WHERE"},
        ],
        "row_limit": 50,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "x_axis_label": "country",
        "y_axis_label": "score_w (品牌层 SAT)",
    },
})

# B-3 · 品牌提及 (L4_01) 正负声量趋势 by data_source
charts.append({
    "theme": "Brand",
    "code":  "B-3",
    "slice_name": "VOC D-Diag Brand · B-3 SAT_L4_01 品牌声量趋势 by source",
    "viz_type": "line",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "line",
        "groupby": [],
        "metrics": [
            {"expressionType": "SQL", "label": "pos",
             "sqlExpression": "SUM(pos)"},
            {"expressionType": "SQL", "label": "neg",
             "sqlExpression": "SUM(neg)"},
            {"expressionType": "SQL", "label": "neu",
             "sqlExpression": "SUM(neu)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "==", "comparator": "SAT_L4_01", "clause": "WHERE"},
        ],
        "granularity_sqla": "month",
        "time_grain_sqla": "P1M",
        "row_limit": 5000,
        "show_legend": True,
        "color_scheme": "supersetColors",
        "x_axis_label": "month",
        "y_axis_label": "声量",
        "show_brush": "auto",
    },
})

# B-4 · 内容/社群 (SAT_L3_17) 健康度 × 国家
charts.append({
    "theme": "Brand",
    "code":  "B-4",
    "slice_name": "VOC D-Diag Brand · B-4 SAT_L3_17 内容/社群 × 国家",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_atomic_indicator_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_atomic_indicator_score"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["country"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "score_w",
             "sqlExpression": "ROUND((SUM(sat_score * total_hits) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
            {"expressionType": "SQL", "label": "neg_pct",
             "sqlExpression": "ROUND((100.0 * SUM(neg) / NULLIF(SUM(total_hits)::numeric, 0))::numeric, 2)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "atomic_indicator_id",
             "operator": "==", "comparator": "SAT_L3_17", "clause": "WHERE"},
            {"expressionType": "SIMPLE", "subject": "country",
             "operator": "IS NOT NULL", "comparator": None, "clause": "WHERE"},
        ],
        "row_limit": 30,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "x_axis_label": "country",
        "y_axis_label": "hits / score_w",
    },
})

# B-5 · 品牌 vs 产品 双轴对比（节点体量 vs 节点得分）
charts.append({
    "theme": "Brand",
    "code":  "B-5",
    "slice_name": "VOC D-Diag Brand · B-5 节点体量 vs 节点得分 双轴",
    "viz_type": "dist_bar",
    "datasource_id": DS["v_aipl_node_score"],
    "datasource_type": "table",
    "params": {
        "datasource": f'{DS["v_aipl_node_score"]}__table',
        "viz_type": "dist_bar",
        "groupby": ["aipl_node"],
        "metrics": [
            {"expressionType": "SQL", "label": "hits",
             "sqlExpression": "SUM(total_hits)"},
            {"expressionType": "SQL", "label": "node_score",
             "sqlExpression": "ROUND(AVG(node_score)::numeric, 2)"},
        ],
        "adhoc_filters": [
            {"expressionType": "SIMPLE", "subject": "aipl_node",
             "operator": "IS NOT NULL", "comparator": None, "clause": "WHERE"},
        ],
        "row_limit": 10,
        "color_scheme": "supersetColors",
        "show_legend": True,
        "x_axis_label": "AIPL 节点",
        "y_axis_label": "体量 / 得分",
    },
})


# ----------------------------------------------------------------------------
# Create charts
# ----------------------------------------------------------------------------
print(f"creating {len(charts)} charts...")
created = []
for c in charts:
    body = {
        "slice_name": c["slice_name"],
        "viz_type":   c["viz_type"],
        "datasource_id":   c["datasource_id"],
        "datasource_type": c["datasource_type"],
        "params": json.dumps(c["params"], ensure_ascii=False),
    }
    r = s.post(f"{URL}/api/v1/chart/", json=body)
    try:
        rj = r.json()
    except Exception:
        rj = {"raw": r.text[:300]}
    cid = rj.get("id")
    print(f"  [{r.status_code}] {c['code']} {c['slice_name']}  id={cid}")
    if cid:
        created.append({"id": cid, "code": c["code"], "theme": c["theme"],
                        "name": c["slice_name"], "viz": c["viz_type"]})
    else:
        print(f"    ERR: {json.dumps(rj, ensure_ascii=False)[:600]}")

print()
print("=== CREATED ===")
print(json.dumps(created, ensure_ascii=False, indent=2))

with open("/tmp/mvp_l5_chart_ids.json", "w") as f:
    json.dump(created, f, ensure_ascii=False, indent=2)
