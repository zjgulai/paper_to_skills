#!/bin/bash
SKUS_JSON="[{\"id\":\"SKU-001\",\"name\":\"硅胶婴儿餐具\",\"dos\":0,\"acos\":0}]"
curl -s -X POST "https://skills.lute-tlz-dddd.top/api/daily-inspect" \
  -H "Content-Type: application/json" \
  -d "{\"secret\":\"p2s_inspect_2026\",\"skus\":$SKUS_JSON}" >> /var/log/p2s-inspect.log 2>&1
