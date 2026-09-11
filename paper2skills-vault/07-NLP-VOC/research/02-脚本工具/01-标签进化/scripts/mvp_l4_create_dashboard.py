#!/usr/bin/env python3
"""L4.3 Step 4 · Dashboard assembly."""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://voc.lute-tlz-dddd.top"
s = requests.Session(); s.verify = False
token = s.post(f"{URL}/api/v1/security/login",
               json={"username":"admin","password":"voc_admin_2026","provider":"db","refresh":True}).json()["access_token"]
s.headers.update({"Authorization": f"Bearer {token}"})
csrf = s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken": csrf, "Referer": URL})

CHART_IDS = json.load(open("/tmp/mvp_l4_chart_ids.json"))
ID_TOTAL, ID_SAT, ID_NPS = 13, 14, 15
ID_AIPL, ID_TBL, ID_HEAT, ID_NPSBAR = 16, 17, 18, 19

def chart_node(node_id, chart_id, parent, w, h):
    return {
        "type": "CHART",
        "id":   node_id,
        "children": [],
        "parents": parent,
        "meta": {
            "width":  w, "height": h,
            "chartId": chart_id,
            "uuid": "",
        },
    }

position = {
    "DASHBOARD_VERSION_KEY": "v2",
    "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
    "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": ["ROW_KPI", "ROW_AIPL", "ROW_HEAT", "ROW_NPS"], "parents": ["ROOT_ID"]},

    "ROW_KPI": {"type": "ROW", "id": "ROW_KPI",
                "children": ["CHART_TOTAL", "CHART_SAT", "CHART_NPS"],
                "parents": ["ROOT_ID", "GRID_ID"], "meta": {"background": "BACKGROUND_TRANSPARENT"}},
    "CHART_TOTAL": chart_node("CHART_TOTAL", ID_TOTAL, ["ROOT_ID","GRID_ID","ROW_KPI"], 4, 20),
    "CHART_SAT":   chart_node("CHART_SAT",   ID_SAT,   ["ROOT_ID","GRID_ID","ROW_KPI"], 4, 20),
    "CHART_NPS":   chart_node("CHART_NPS",   ID_NPS,   ["ROOT_ID","GRID_ID","ROW_KPI"], 4, 20),

    "ROW_AIPL": {"type": "ROW", "id": "ROW_AIPL",
                 "children": ["CHART_AIPL", "CHART_TBL"],
                 "parents": ["ROOT_ID", "GRID_ID"], "meta": {"background": "BACKGROUND_TRANSPARENT"}},
    "CHART_AIPL": chart_node("CHART_AIPL", ID_AIPL, ["ROOT_ID","GRID_ID","ROW_AIPL"], 6, 50),
    "CHART_TBL":  chart_node("CHART_TBL",  ID_TBL,  ["ROOT_ID","GRID_ID","ROW_AIPL"], 6, 50),

    "ROW_HEAT": {"type": "ROW", "id": "ROW_HEAT",
                 "children": ["CHART_HEAT"],
                 "parents": ["ROOT_ID", "GRID_ID"], "meta": {"background": "BACKGROUND_TRANSPARENT"}},
    "CHART_HEAT": chart_node("CHART_HEAT", ID_HEAT, ["ROOT_ID","GRID_ID","ROW_HEAT"], 12, 60),

    "ROW_NPS": {"type": "ROW", "id": "ROW_NPS",
                "children": ["CHART_NPSBAR"],
                "parents": ["ROOT_ID", "GRID_ID"], "meta": {"background": "BACKGROUND_TRANSPARENT"}},
    "CHART_NPSBAR": chart_node("CHART_NPSBAR", ID_NPSBAR, ["ROOT_ID","GRID_ID","ROW_NPS"], 12, 40),

    "HEADER_ID": {"type": "HEADER", "id": "HEADER_ID",
                  "meta": {"text": "VOC 深度分析 · D-Health"}, "children": []},
}

body = {
    "dashboard_title": "VOC · 深度分析 D-Health",
    "slug": "voc-deep-health",
    "published": True,
    "position_json": json.dumps(position),
    "css": "",
    "json_metadata": json.dumps({
        "color_scheme": "supersetColors",
        "refresh_frequency": 0,
        "expanded_slices": {},
        "label_colors": {},
        "shared_label_colors": {},
        "timed_refresh_immune_slices": [],
        "default_filters": "{}",
    }),
}

r = s.post(f"{URL}/api/v1/dashboard/", json=body)
print(f"[{r.status_code}] {r.text[:600]}")
if r.status_code in (200, 201):
    rj = r.json()
    dash_id = rj.get("id")
    print(f"\nDASHBOARD ID = {dash_id}")
    print(f"URL          = {URL}/superset/dashboard/{dash_id}/")

    chart_ids = [ID_TOTAL, ID_SAT, ID_NPS, ID_AIPL, ID_TBL, ID_HEAT, ID_NPSBAR]
    print(f"\nlinking {len(chart_ids)} charts to dashboard...")
    for cid in chart_ids:
        cr = s.put(f"{URL}/api/v1/chart/{cid}", json={"dashboards": [dash_id]})
        print(f"  chart {cid} -> dashboard {dash_id} : {cr.status_code}")

    json.dump({"dashboard_id": dash_id, "chart_ids": chart_ids},
              open("/tmp/mvp_l4_dashboard_ids.json", "w"))
