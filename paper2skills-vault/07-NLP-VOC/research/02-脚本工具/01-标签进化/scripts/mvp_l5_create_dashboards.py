#!/usr/bin/env python3
"""L5.3.2 · 3 D-Diag dashboards via Superset API.

Themes:
  D-Diag-Product (slug=voc-deep-diag-product)  → 5 charts P-1..P-5
  D-Diag-Service (slug=voc-deep-diag-service)  → 5 charts S-1..S-5
  D-Diag-Brand   (slug=voc-deep-diag-brand)    → 5 charts B-1..B-5
"""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = "https://voc.lute-tlz-dddd.top"
s = requests.Session(); s.verify = False
tok = s.post(f"{URL}/api/v1/security/login",
             json={"username":"admin","password":"voc_admin_2026","provider":"db","refresh":True}
             ).json()["access_token"]
s.headers.update({"Authorization": f"Bearer {tok}"})
csrf = s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken": csrf, "Referer": URL})

charts = json.load(open("/tmp/mvp_l5_chart_ids.json"))

def chart_node(node_id, chart_id, parent, w, h):
    return {
        "type": "CHART",
        "id":   node_id,
        "children": [],
        "parents": parent,
        "meta": {"width": w, "height": h, "chartId": chart_id, "uuid": ""},
    }

THEMES = [
    {
        "key":"Product",
        "title": "VOC · 深度分析 D-Diag-Product 产品力诊断",
        "slug":  "voc-deep-diag-product",
        "header_text": "D-Diag · 产品力诊断 (产品中心 18.56% 负向 / SAT_L1_01 0.25 分 / 吸奶器 35.64% 负向)",
    },
    {
        "key":"Service",
        "title": "VOC · 深度分析 D-Diag-Service 服务力诊断",
        "slug":  "voc-deep-diag-service",
        "header_text": "D-Diag · 服务力诊断 (品质 78.71% 极端 · US 品质 90.09% · UK 品质 76.24%)",
    },
    {
        "key":"Brand",
        "title": "VOC · 深度分析 D-Diag-Brand 品牌内容力诊断",
        "slug":  "voc-deep-diag-brand",
        "header_text": "D-Diag · 品牌内容力诊断 (内容 L3_17 68.91 高 · 品牌 L4_01 36.62 + 26.66% 负向 · ES NPS -0.13)",
    },
]

# Layout per theme (5 charts):
# Row1: code-1 (table/heatmap, full)             12 wide × 50 high
# Row2: code-2 (heatmap or table)                 6 wide × 50 high
#       code-3 (table or heatmap)                 6 wide × 50 high
# Row3: code-4 (dist_bar / pie)                   6 wide × 50 high
#       code-5 (line / dist_bar)                  6 wide × 50 high
# Total height ~150 grid units

def position_for(theme_key, ids):
    c1, c2, c3, c4, c5 = ids
    return {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {"type":"ROOT","id":"ROOT_ID","children":["GRID_ID"]},
        "GRID_ID": {"type":"GRID","id":"GRID_ID",
                    "children":["ROW_1","ROW_2","ROW_3"],"parents":["ROOT_ID"]},
        "ROW_1": {"type":"ROW","id":"ROW_1",
                  "children":[f"CHART_{theme_key}_1"],
                  "parents":["ROOT_ID","GRID_ID"],
                  "meta":{"background":"BACKGROUND_TRANSPARENT"}},
        f"CHART_{theme_key}_1": chart_node(f"CHART_{theme_key}_1", c1,
                                            ["ROOT_ID","GRID_ID","ROW_1"], 12, 50),
        "ROW_2": {"type":"ROW","id":"ROW_2",
                  "children":[f"CHART_{theme_key}_2", f"CHART_{theme_key}_3"],
                  "parents":["ROOT_ID","GRID_ID"],
                  "meta":{"background":"BACKGROUND_TRANSPARENT"}},
        f"CHART_{theme_key}_2": chart_node(f"CHART_{theme_key}_2", c2,
                                            ["ROOT_ID","GRID_ID","ROW_2"], 6, 50),
        f"CHART_{theme_key}_3": chart_node(f"CHART_{theme_key}_3", c3,
                                            ["ROOT_ID","GRID_ID","ROW_2"], 6, 50),
        "ROW_3": {"type":"ROW","id":"ROW_3",
                  "children":[f"CHART_{theme_key}_4", f"CHART_{theme_key}_5"],
                  "parents":["ROOT_ID","GRID_ID"],
                  "meta":{"background":"BACKGROUND_TRANSPARENT"}},
        f"CHART_{theme_key}_4": chart_node(f"CHART_{theme_key}_4", c4,
                                            ["ROOT_ID","GRID_ID","ROW_3"], 6, 50),
        f"CHART_{theme_key}_5": chart_node(f"CHART_{theme_key}_5", c5,
                                            ["ROOT_ID","GRID_ID","ROW_3"], 6, 50),
        "HEADER_ID": {"type":"HEADER","id":"HEADER_ID",
                      "meta":{"text": f"VOC 深度分析 · D-Diag-{theme_key}"},
                      "children":[]},
    }

result = {"dashboards": []}
for t in THEMES:
    ids = [c["id"] for c in charts if c["theme"] == t["key"]]
    print(f"\nTheme {t['key']}: charts {ids}  ({len(ids)} charts)")
    if len(ids) != 5:
        print(f"  ! expected 5 charts, got {len(ids)}; skipping")
        continue

    pos = position_for(t["key"], ids)
    body = {
        "dashboard_title": t["title"],
        "slug":            t["slug"],
        "published":       True,
        "position_json":   json.dumps(pos),
        "css":             "",
        "json_metadata":   json.dumps({
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
    print(f"  [{r.status_code}] dashboard create  {t['title']}")
    if r.status_code in (200, 201):
        rj = r.json()
        dash_id = rj.get("id")
        result["dashboards"].append({
            "id": dash_id, "theme": t["key"], "title": t["title"], "slug": t["slug"],
            "chart_ids": ids,
        })
        print(f"  → dashboard id={dash_id}  url={URL}/superset/dashboard/{dash_id}/")

        # Link charts to dashboard
        for cid in ids:
            cr = s.put(f"{URL}/api/v1/chart/{cid}", json={"dashboards":[dash_id]})
            print(f"    chart {cid} → dashboard {dash_id} : {cr.status_code}")
    else:
        print(f"  ERR: {r.text[:600]}")

print()
print("=== ALL DASHBOARDS CREATED ===")
print(json.dumps(result, ensure_ascii=False, indent=2))

with open("/tmp/mvp_l5_dashboard_ids.json","w") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
