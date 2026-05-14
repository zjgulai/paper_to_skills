#!/usr/bin/env python3
"""L6 一键回滚 · 删除 8 chart + 1 dashboard + 还原 7 dept dashboard 的 position_json。

注意：path A 修改了 7 个现有 dept dashboard 的 position_json（追加 ROW_DA-{cid}）。
回滚时使用 pre_l6_snapshot.json 中保存的原 position_json 还原即可。
"""
import json, urllib3, requests, os, sys
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL="https://voc.lute-tlz-dddd.top"
s=requests.Session(); s.verify=False
tok=s.post(f"{URL}/api/v1/security/login",json={"username":"admin","password":"voc_admin_2026","provider":"db","refresh":True}).json()["access_token"]
s.headers.update({"Authorization":f"Bearer {tok}"})
csrf=s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken":csrf,"Referer":URL})

# Restore 7 dept dashboard position_json from pre-L6 snapshot
HERE=os.path.dirname(os.path.abspath(__file__))
pre=json.load(open(os.path.join(HERE,"pre_l6_snapshot.json")))
DEPT_IDS=[2,3,4,5,6,7,8]
for d in pre["dashboards"]:
    if d["id"] in DEPT_IDS:
        body={"position_json": d["position_json"]}
        r=s.put(f"{URL}/api/v1/dashboard/{d['id']}", json=body)
        print(f"RESTORE dashboard {d['id']} ({d['title']}): [{r.status_code}]")

# Delete D-Action overview dashboard 13
r=s.delete(f"{URL}/api/v1/dashboard/13")
print(f"DELETE dashboard 13 (D-Action overview): [{r.status_code}]")

# Delete 8 charts (35-42)
for cid in range(35,43):
    r=s.delete(f"{URL}/api/v1/chart/{cid}")
    print(f"DELETE chart {cid}: [{r.status_code}]")
