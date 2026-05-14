#!/usr/bin/env python3
"""L6.2 · Path A · Append D-Action chart to each of 7 dept dashboards.

Each dept dashboard currently has structure:
    ROOT_ID → GRID_ID → ROW-7 → CHART-7 (the dept Top 10 topic chart)

We append:
    ROOT_ID → GRID_ID → [ROW-7 (existing), ROW_DA (new) → CHART_DA (new)]
"""
import json, urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL="https://voc.lute-tlz-dddd.top"
s=requests.Session(); s.verify=False
tok=s.post(f"{URL}/api/v1/security/login",json={"username":"admin","password":"voc_admin_2026","provider":"db","refresh":True}).json()["access_token"]
s.headers.update({"Authorization":f"Bearer {tok}"})
csrf=s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken":csrf,"Referer":URL})

# Map: dept_owner → (dashboard_id, chart_id)
charts = json.load(open("/tmp/mvp_l6_chart_ids.json"))
DEPT_TO_CHART = {c["dept"]: c["id"] for c in charts if c["dept"] != "ALL"}

# Map: dashboard_id → dept_owner (from pre-snapshot)
DASHBOARD_TO_DEPT = {
    2: "全球客服中心",
    3: "产品中心",
    4: "仓储物流部",
    5: "品牌市场中心",
    6: "电商运营部",
    7: "品质管理中心",
    8: "法务合规部",
}

results = []
for dash_id, dept in DASHBOARD_TO_DEPT.items():
    chart_id = DEPT_TO_CHART[dept]
    print(f"\n=== Dashboard {dash_id} ({dept}) ← chart {chart_id} ===")

    # Fetch current dashboard
    cur = s.get(f"{URL}/api/v1/dashboard/{dash_id}").json().get("result",{})
    pos = json.loads(cur.get("position_json","{}"))

    # Sanity check: GRID_ID must exist
    if "GRID_ID" not in pos:
        print(f"  ! GRID_ID missing, skipping")
        continue

    # Generate unique new component ids using chart_id to avoid collision
    new_row_id   = f"ROW-DA-{chart_id}"
    new_chart_id = f"CHART-DA-{chart_id}"

    # Already appended? skip
    if new_row_id in pos:
        print(f"  · already has {new_row_id}, skipping")
        results.append({"dashboard_id":dash_id,"dept":dept,
                        "chart_id":chart_id,"status":"already_present"})
        continue

    # Append new row to GRID_ID children
    grid_children = pos["GRID_ID"].get("children", [])
    if new_row_id not in grid_children:
        grid_children.append(new_row_id)
    pos["GRID_ID"]["children"] = grid_children

    # Add the new ROW node
    pos[new_row_id] = {
        "type": "ROW",
        "id":   new_row_id,
        "children": [new_chart_id],
        "parents": ["ROOT_ID","GRID_ID"],
        "meta": {"background":"BACKGROUND_TRANSPARENT"},
    }
    # Add the new CHART node (full width, height ~60)
    pos[new_chart_id] = {
        "type": "CHART",
        "id":   new_chart_id,
        "children": [],
        "parents": ["ROOT_ID","GRID_ID",new_row_id],
        "meta": {"width": 12, "height": 70, "chartId": chart_id, "uuid": ""},
    }

    body = {"position_json": json.dumps(pos)}
    r = s.put(f"{URL}/api/v1/dashboard/{dash_id}", json=body)
    print(f"  [{r.status_code}] dashboard updated")

    # Link chart to dashboard so it appears in chart→dashboard relationships
    cur_dashes = [d.get("id") for d in cur.get("dashboards",[]) if d.get("id")]
    new_dashes = list(set(cur_dashes + [dash_id]))
    cr = s.put(f"{URL}/api/v1/chart/{chart_id}", json={"dashboards": new_dashes})
    print(f"  [{cr.status_code}] chart {chart_id} ←→ dashboards {new_dashes}")

    results.append({"dashboard_id":dash_id,"dept":dept,
                    "chart_id":chart_id,"status":"appended"})

print()
print("=== L6 PATH A SUMMARY ===")
print(json.dumps(results, ensure_ascii=False, indent=2))
with open("/tmp/mvp_l6_path_a_results.json","w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
