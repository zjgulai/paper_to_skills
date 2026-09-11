#!/usr/bin/env python3
"""L6 fix · Filter out '【待填写】' placeholder + use SQL filter for biz_action.

Root cause:
  1. dim_tag.biz_action sometimes literally contains '【待填写】' (15 product
     tags), which my original `biz_action <> ''` filter does NOT catch.
  2. Superset's order_by_cols on a metric label sometimes does not correctly
     sort when ties exist — but server-side ORDER BY metric is robust.

Fix:
  - Replace adhoc_filters to use ONE SQL clause that filters BOTH null/empty/
    placeholder/inline.
  - Keep `order_desc=True` + `timeseries_limit_metric=priority_score` (table
    viz uses this).
"""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL="https://voc.lute-tlz-dddd.top"
s=requests.Session(); s.verify=False
tok=s.post(f"{URL}/api/v1/security/login",json={"username":"admin","password":"voc_admin_2026","provider":"db","refresh":True}).json()["access_token"]
s.headers.update({"Authorization":f"Bearer {tok}"})
csrf=s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken":csrf,"Referer":URL})

charts = json.load(open("/tmp/mvp_l6_chart_ids.json"))

# Updated filter: drop empty + null + 【待填写】 placeholder
ACTION_OK = {
    "expressionType":"SQL","subject":"biz_action",
    "operator":"==","comparator":None,"clause":"WHERE",
    "sqlExpression":"biz_action IS NOT NULL AND biz_action <> '' AND biz_action NOT LIKE '%待填写%'",
}

priority_metric = {
    "expressionType":"SQL","label":"priority_score",
    "sqlExpression":"ROUND((SUM(CASE WHEN sentiment_preset='negative' THEN 1 ELSE 0 END) * (100.0 * SUM(CASE WHEN sentiment_preset='negative' THEN 1 ELSE 0 END) / NULLIF(COUNT(*)::numeric, 0)) / 100.0)::numeric, 1)",
}

for c in charts:
    cid = c["id"]
    cur = s.get(f"{URL}/api/v1/chart/{cid}").json().get("result",{})
    params = json.loads(cur.get("params","{}"))

    # Rebuild adhoc_filters: keep dept-specific filter + replace biz_action filters
    new_filters = []
    for f in params.get("adhoc_filters",[]):
        if f.get("subject") == "biz_action":
            continue  # drop old biz_action filters
        new_filters.append(f)
    new_filters.append(ACTION_OK)

    # ALSO drop dept_owner filter for MASTER chart 42 (already in groupby)
    if c["dept"] == "ALL":
        # ensure dept_owner NOT IN ('','未分类') stays
        pass

    params["adhoc_filters"] = new_filters
    # Use timeseries_limit_metric as the server-side ORDER BY for table viz
    params["timeseries_limit_metric"] = priority_metric
    params["order_desc"] = True
    # remove the old order_by_cols (it was JSON-string-wrapped, causing issues)
    params.pop("order_by_cols", None)

    r = s.put(f"{URL}/api/v1/chart/{cid}",
              json={"params": json.dumps(params, ensure_ascii=False)})
    print(f"  [{r.status_code}] chart {cid:>3} ({c['code']}) updated")
