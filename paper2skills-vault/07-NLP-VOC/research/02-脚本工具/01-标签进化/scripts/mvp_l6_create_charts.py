#!/usr/bin/env python3
"""L6.1 · 8 D-Action charts via Superset API.

Strategy: 8 charts = 7 dept-specific + 1 master cross-dept.
Each chart is a virtual_dataset table query against v_label_with_dept (dataset 2)
with priority_score = hit_negative * pct_negative / 100 (per user choice A).

Reusable: each dept chart is linked to its own dashboard (path A) AND to the
unified D-Action overview dashboard (path B).

Datasets used:
    2 = v_label_with_dept (has dept_owner / biz_action / strategy_pkg / sentiment_preset)
"""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://voc.lute-tlz-dddd.top"
USER, PASS = "admin", "voc_admin_2026"

s = requests.Session(); s.verify = False
tok = s.post(f"{URL}/api/v1/security/login",
             json={"username":USER,"password":PASS,"provider":"db","refresh":True}
             ).json()["access_token"]
s.headers.update({"Authorization": f"Bearer {tok}"})
csrf = s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken": csrf, "Referer": URL})

DATASET_ID = 2  # v_label_with_dept

# 7 departments (matching dim_tag.dept_owner exactly)
DEPTS = [
    "产品中心",
    "全球客服中心",
    "仓储物流部",
    "品牌市场中心",
    "品质管理中心",
    "电商运营部",
    "法务合规部",
]

# Common metrics for action queue
def action_metrics():
    return [
        {"expressionType":"SQL","label":"hit_count",
         "sqlExpression":"COUNT(*)"},
        {"expressionType":"SQL","label":"hit_negative",
         "sqlExpression":"SUM(CASE WHEN sentiment_preset='negative' THEN 1 ELSE 0 END)"},
        {"expressionType":"SQL","label":"distinct_reviews",
         "sqlExpression":"COUNT(DISTINCT review_id)"},
        {"expressionType":"SQL","label":"pct_negative",
         "sqlExpression":"ROUND(100.0 * SUM(CASE WHEN sentiment_preset='negative' THEN 1 ELSE 0 END) / NULLIF(COUNT(*)::numeric, 0), 2)"},
        {"expressionType":"SQL","label":"priority_score",
         "sqlExpression":"ROUND((SUM(CASE WHEN sentiment_preset='negative' THEN 1 ELSE 0 END) * (100.0 * SUM(CASE WHEN sentiment_preset='negative' THEN 1 ELSE 0 END) / NULLIF(COUNT(*)::numeric, 0)) / 100.0)::numeric, 1)"},
    ]

charts = []

# -----------------------------------------------------------------------------
# 7 dept-specific D-Action charts
# -----------------------------------------------------------------------------
for dept in DEPTS:
    charts.append({
        "code": f"DA-{dept}",
        "dept": dept,
        "slice_name": f"VOC D-Action · {dept} 行动队列 Top 10",
        "viz_type": "table",
        "datasource_id": DATASET_ID,
        "datasource_type": "table",
        "params": {
            "datasource": f"{DATASET_ID}__table",
            "viz_type": "table",
            "groupby": ["tag_cn", "biz_action", "strategy_pkg"],
            "metrics": action_metrics(),
            "adhoc_filters": [
                {"expressionType":"SIMPLE","subject":"dept_owner",
                 "operator":"==","comparator":dept,"clause":"WHERE"},
                {"expressionType":"SIMPLE","subject":"biz_action",
                 "operator":"IS NOT NULL","comparator":None,"clause":"WHERE"},
                {"expressionType":"SQL","subject":"biz_action",
                 "operator":"==","comparator":None,"clause":"WHERE",
                 "sqlExpression":"biz_action <> ''"},
            ],
            "row_limit": 10,
            "order_by_cols": [json.dumps(["priority_score", False])],
            "order_desc": True,
            "include_search": True,
            "show_cell_bars": True,
            "table_timestamp_format": "smart_date",
            "color_pn": True,
            "page_length": 10,
        },
    })

# -----------------------------------------------------------------------------
# Master cross-department D-Action (Top 20)
# -----------------------------------------------------------------------------
charts.append({
    "code": "DA-MASTER",
    "dept": "ALL",
    "slice_name": "VOC D-Action · 全部门 行动总队列 Top 20",
    "viz_type": "table",
    "datasource_id": DATASET_ID,
    "datasource_type": "table",
    "params": {
        "datasource": f"{DATASET_ID}__table",
        "viz_type": "table",
        "groupby": ["dept_owner", "tag_cn", "biz_action", "strategy_pkg"],
        "metrics": action_metrics(),
        "adhoc_filters": [
            {"expressionType":"SIMPLE","subject":"dept_owner",
             "operator":"IS NOT NULL","comparator":None,"clause":"WHERE"},
            {"expressionType":"SQL","subject":"dept_owner",
             "operator":"==","comparator":None,"clause":"WHERE",
             "sqlExpression":"dept_owner NOT IN ('', '未分类')"},
            {"expressionType":"SIMPLE","subject":"biz_action",
             "operator":"IS NOT NULL","comparator":None,"clause":"WHERE"},
            {"expressionType":"SQL","subject":"biz_action",
             "operator":"==","comparator":None,"clause":"WHERE",
             "sqlExpression":"biz_action <> ''"},
        ],
        "row_limit": 20,
        "order_by_cols": [json.dumps(["priority_score", False])],
        "order_desc": True,
        "include_search": True,
        "show_cell_bars": True,
        "page_length": 20,
    },
})

# -----------------------------------------------------------------------------
# Create charts
# -----------------------------------------------------------------------------
print(f"creating {len(charts)} D-Action charts...")
created = []
for c in charts:
    body = {
        "slice_name":      c["slice_name"],
        "viz_type":        c["viz_type"],
        "datasource_id":   c["datasource_id"],
        "datasource_type": c["datasource_type"],
        "params":          json.dumps(c["params"], ensure_ascii=False),
    }
    r = s.post(f"{URL}/api/v1/chart/", json=body)
    try:
        rj = r.json()
    except Exception:
        rj = {"raw": r.text[:300]}
    cid = rj.get("id")
    print(f"  [{r.status_code}] {c['code']:<22} {c['slice_name']}  id={cid}")
    if cid:
        created.append({"id": cid, "code": c["code"], "dept": c["dept"],
                        "name": c["slice_name"], "viz": c["viz_type"]})
    else:
        print(f"    ERR: {json.dumps(rj, ensure_ascii=False)[:600]}")

print()
print("=== CREATED ===")
print(json.dumps(created, ensure_ascii=False, indent=2))

with open("/tmp/mvp_l6_chart_ids.json","w") as f:
    json.dump(created, f, ensure_ascii=False, indent=2)
