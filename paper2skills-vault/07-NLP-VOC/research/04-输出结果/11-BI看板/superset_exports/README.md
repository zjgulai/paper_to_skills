# Superset Dashboard Exports — Phase 7 D3

8 dashboards exported from Superset 4.1.1 via `/api/v1/dashboard/export/` API.

## Files

| File | Dashboard | Charts |
|---|---|---:|
| `dashboard_1.zip` | VOC Overview · 全局总览 | 5 |
| `dashboard_2.zip` | VOC · 客服部 | 1 |
| `dashboard_3.zip` | VOC · 产品研发部 | 1 |
| `dashboard_4.zip` | VOC · 国际物流部 | 1 |
| `dashboard_5.zip` | VOC · 市场部 | 1 |
| `dashboard_6.zip` | VOC · 电商运营部 | 1 |
| `dashboard_7.zip` | VOC · 品控部 | 1 |
| `dashboard_8.zip` | VOC · 质量与法规部 | 1 |

## Import

```bash
# 1. Ensure voc_bi database + 6 datasets exist (run superset_bootstrap.py)
cd paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/docker
python3 superset_bootstrap.py

# 2. Import dashboards via Superset UI
# Settings → Import Dashboards → upload each ZIP
# OR via CLI:
for f in dashboard_*.zip; do
  superset import-dashboards -p "$f"
done
```

## Regenerate

```bash
# Idempotent: re-run factory to recreate all charts + dashboards from scratch
python3 superset_charts_factory.py
```

## Export Date

2026-05-10 (Phase 7 D3)

## Dependencies

- Superset 4.1.1
- voc_bi database (postgres) with 6 views (v_review_overview, v_label_with_dept, v_dept_topic_summary, v_label_brand, v_global_top_tags, v_dept_kpi)
- Dataset IDs 1-6 registered
