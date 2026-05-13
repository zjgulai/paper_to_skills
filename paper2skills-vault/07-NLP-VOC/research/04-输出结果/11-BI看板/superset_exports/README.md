# Superset Dashboard Exports — Phase 7 D4

8 dashboards exported from Superset 4.1.1 via `/api/v1/dashboard/export/` API.

**D4 update**: All dashboards now include native filter configuration.

## Files

| File | Dashboard | Charts | Filters |
|---|---|---:|---|
| `dashboard_1.zip` | VOC Overview · 全局总览 | 5 | 数据源 / 产品线 / Proxy NPS（针对 v_review_overview 上的 2 张饼图） |
| `dashboard_2.zip` | VOC · 全球客服与体验中心 | 1 | 情感极性 |
| `dashboard_3.zip` | VOC · 产品中心/品线 | 1 | 情感极性 |
| `dashboard_4.zip` | VOC · 供应链中心 | 1 | 情感极性 |
| `dashboard_5.zip` | VOC · 品牌市场中心 | 1 | 情感极性 |
| `dashboard_6.zip` | VOC · 电商运营部 | 1 | 情感极性 |
| `dashboard_7.zip` | VOC · 品控部 | 1 | 情感极性 |
| `dashboard_8.zip` | VOC · 质量与法规部 | 1 | 情感极性 |

## Import

```bash
cd paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/docker
python3 superset_bootstrap.py

for f in dashboard_*.zip; do
  superset import-dashboards -p "$f"
done
```

## Regenerate

```bash
python3 superset_charts_factory.py
python3 superset_filters_factory.py
```

## Export Date

2026-05-11 (Phase 7 D4)

## Dependencies

- Superset 4.1.1
- voc_bi database (postgres) with 6 views (v_review_overview, v_label_with_dept, v_dept_topic_summary, v_label_brand, v_global_top_tags, v_dept_kpi)
- Dataset IDs 1-6 registered
