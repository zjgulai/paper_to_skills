#!/usr/bin/env python3
"""L5 一键回滚 · 删除 15 chart + 3 dashboard，0 dataset / 0 view 影响。"""
import urllib3, requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
URL="https://voc.lute-tlz-dddd.top"
s=requests.Session(); s.verify=False
tok=s.post(f"{URL}/api/v1/security/login",
          json={"username":"admin","password":"voc_admin_2026",
                "provider":"db","refresh":True}).json()["access_token"]
s.headers.update({"Authorization":f"Bearer {tok}"})
csrf=s.get(f"{URL}/api/v1/security/csrf_token/").json()["result"]
s.headers.update({"X-CSRFToken":csrf,"Referer":URL})

# Step 1: dashboards first (linked charts will unlink automatically)
for did in (10,11,12):
    r=s.delete(f"{URL}/api/v1/dashboard/{did}")
    print(f"DELETE dashboard {did}: [{r.status_code}]")

# Step 2: charts
for cid in range(20,35):
    r=s.delete(f"{URL}/api/v1/chart/{cid}")
    print(f"DELETE chart {cid}: [{r.status_code}]")
