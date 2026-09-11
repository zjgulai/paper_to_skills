#!/usr/bin/env python3
"""L6.3 · Path B · Unified D-Action overview dashboard.

Layout:
    ROW_MASTER : MASTER (id 42, Top 20 across all depts, full width, h=80)
    ROW_DEPT_1 : 产品中心 (35) | 品牌市场中心 (38) | 品质管理中心 (39)    [4w each]
    ROW_DEPT_2 : 仓储物流部 (37) | 全球客服中心 (36)                       [6w each]
    ROW_DEPT_3 : 电商运营部 (40) | 法务合规部 (41)                          [6w each]

Charts are reused — they're already linked to dept dashboards (path A).
Path B adds them ALSO to dashboard 13.
"""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL="https://voc.lute-tlz-dddd.top"
s=requests.Session(); s.verify=False
tok=s.post(f"{URL}/api/v1/security/login",json={"username":"admin","password":"voc_admin_2026","provider":"db","refresh":True}).json()["access_token"]
s.headers.update({"Authorization": f"Bearer {tok}"})
csrf=s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken":csrf,"Referer":URL})

charts = json.load(open("/tmp/mvp_l6_chart_ids.json"))
C = {c["dept"]: c["id"] for c in charts}

# Layout helpers
def chart_node(node_id, chart_id, parent_chain, w, h):
    return {"type":"CHART","id":node_id,"children":[],"parents":parent_chain,
            "meta":{"width":w,"height":h,"chartId":chart_id,"uuid":""}}

ROOT = ["ROOT_ID","GRID_ID"]

position = {
    "DASHBOARD_VERSION_KEY":"v2",
    "ROOT_ID": {"type":"ROOT","id":"ROOT_ID","children":["GRID_ID"]},
    "HEADER_ID":{"type":"HEADER","id":"HEADER_ID",
                 "meta":{"text":"VOC 深度分析 · D-Action 7 部门行动总队列"},
                 "children":[]},
    "GRID_ID": {"type":"GRID","id":"GRID_ID",
                "children":["ROW_MASTER","ROW_DEPT_1","ROW_DEPT_2","ROW_DEPT_3"],
                "parents":["ROOT_ID"]},

    "ROW_MASTER": {"type":"ROW","id":"ROW_MASTER",
                   "children":["CHART_DA_MASTER"],
                   "parents":ROOT,"meta":{"background":"BACKGROUND_TRANSPARENT"}},
    "CHART_DA_MASTER":   chart_node("CHART_DA_MASTER",   C["ALL"],
                                     ROOT+["ROW_MASTER"], 12, 80),

    "ROW_DEPT_1": {"type":"ROW","id":"ROW_DEPT_1",
                   "children":["CHART_DA_PROD","CHART_DA_BRAND","CHART_DA_QUAL"],
                   "parents":ROOT,"meta":{"background":"BACKGROUND_TRANSPARENT"}},
    "CHART_DA_PROD":  chart_node("CHART_DA_PROD",  C["产品中心"],
                                  ROOT+["ROW_DEPT_1"], 4, 60),
    "CHART_DA_BRAND": chart_node("CHART_DA_BRAND", C["品牌市场中心"],
                                  ROOT+["ROW_DEPT_1"], 4, 60),
    "CHART_DA_QUAL":  chart_node("CHART_DA_QUAL",  C["品质管理中心"],
                                  ROOT+["ROW_DEPT_1"], 4, 60),

    "ROW_DEPT_2": {"type":"ROW","id":"ROW_DEPT_2",
                   "children":["CHART_DA_LOG","CHART_DA_CS"],
                   "parents":ROOT,"meta":{"background":"BACKGROUND_TRANSPARENT"}},
    "CHART_DA_LOG": chart_node("CHART_DA_LOG", C["仓储物流部"],
                                ROOT+["ROW_DEPT_2"], 6, 60),
    "CHART_DA_CS":  chart_node("CHART_DA_CS",  C["全球客服中心"],
                                ROOT+["ROW_DEPT_2"], 6, 60),

    "ROW_DEPT_3": {"type":"ROW","id":"ROW_DEPT_3",
                   "children":["CHART_DA_EC","CHART_DA_LEGAL"],
                   "parents":ROOT,"meta":{"background":"BACKGROUND_TRANSPARENT"}},
    "CHART_DA_EC":    chart_node("CHART_DA_EC",    C["电商运营部"],
                                  ROOT+["ROW_DEPT_3"], 6, 60),
    "CHART_DA_LEGAL": chart_node("CHART_DA_LEGAL", C["法务合规部"],
                                  ROOT+["ROW_DEPT_3"], 6, 60),
}

body = {
    "dashboard_title": "VOC · 深度分析 D-Action 7 部门行动总队列",
    "slug":            "voc-deep-action-overview",
    "published":       True,
    "position_json":   json.dumps(position),
    "css":             "",
    "json_metadata":   json.dumps({
        "color_scheme":"supersetColors","refresh_frequency":0,
        "expanded_slices":{},"label_colors":{},"shared_label_colors":{},
        "timed_refresh_immune_slices":[],"default_filters":"{}",
    }),
}
r=s.post(f"{URL}/api/v1/dashboard/", json=body)
print(f"[{r.status_code}] {r.text[:300]}")
dash_id = r.json().get("id")
print(f"\nD-Action overview dashboard id={dash_id}")
print(f"URL: {URL}/superset/dashboard/{dash_id}/")

# Link all 8 charts ADDITIONALLY to this dashboard (preserving existing links)
print("\nlinking 8 charts to D-Action overview...")
for c in charts:
    cid = c["id"]
    cur = s.get(f"{URL}/api/v1/chart/{cid}").json().get("result",{})
    cur_dashes = [d.get("id") for d in cur.get("dashboards",[]) if d.get("id")]
    new_dashes = list(set(cur_dashes + [dash_id]))
    cr = s.put(f"{URL}/api/v1/chart/{cid}", json={"dashboards": new_dashes})
    print(f"  chart {cid} ({c['code']}) ←→ dashboards {sorted(new_dashes)} : {cr.status_code}")

with open("/tmp/mvp_l6_overview_id.json","w") as f:
    json.dump({"dashboard_id":dash_id,"slug":"voc-deep-action-overview",
               "chart_ids":[c["id"] for c in charts]}, f, ensure_ascii=False, indent=2)
