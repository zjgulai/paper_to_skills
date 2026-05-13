"""Phase 6 D10 BI Dashboard Generator (Static HTML)

Aggregates the 14 MAA+AGRS dept JSON files (produced by C.1/C.2 using D9 data)
plus summary stats from phase6_d9_filtered.jsonl into a single self-contained HTML
dashboard. Usable offline, shareable via file://, no backend needed.

Design:
  - Single HTML file, CSS inline, Chart.js via CDN (graceful degrade: tables still work)
  - Left sidebar: Overview tab + 7 dept tabs
  - Overview: dept volume donut, sentiment bar, global top-10 tags
  - Per dept: summary card + SRAC bar chart + Top-10 table + AGRS sample reviews

Usage:
  python bi_dashboard_generator.py \
    --reports-dir <vault>/04-输出结果/10-周报/2026-W19-d9 \
    --pred-jsonl <vault>/04-输出结果/unified_labeling/phase6_d9_filtered.jsonl \
    --output <vault>/04-输出结果/bi-dashboard/dashboard-2026-W19.html
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEPARTMENTS = [
    "全球客服与体验中心", "产品中心/品线", "供应链中心",
    "品牌市场中心", "电商运营部", "品控部", "质量与法规部",
]


def load_dept_maa(reports_dir: Path, dept: str) -> dict[str, Any] | None:
    p = reports_dir / f"{dept}_MAA.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_dept_agrs(reports_dir: Path, dept: str) -> dict[str, Any] | None:
    p = reports_dir / f"{dept}_AGRS.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def compute_global_stats(pred_jsonl: Path) -> dict[str, Any]:
    source_counter: Counter[str] = Counter()
    platform_counter: Counter[str] = Counter()
    rating_counter: Counter[str] = Counter()
    sentiment_counter: Counter[str] = Counter()
    nps_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    n_records = 0
    n_with_labels = 0

    with pred_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_records += 1
            source_counter[r.get("data_source") or "unknown"] += 1
            platform_counter[r.get("platform") or "unknown"] += 1
            rating = r.get("rating")
            if rating is not None:
                rating_counter[str(int(rating)) if isinstance(rating, (int, float)) else str(rating)] += 1
            pol = r.get("proxy_nps")
            if pol:
                nps_counter[str(pol)] += 1
            sp = r.get("sentiment_polarity")
            if isinstance(sp, (int, float)):
                if sp >= 0.3:
                    sentiment_counter["positive"] += 1
                elif sp <= -0.3:
                    sentiment_counter["negative"] += 1
                else:
                    sentiment_counter["neutral"] += 1
            labels = r.get("labels") or []
            if labels:
                n_with_labels += 1
                for lbl in labels:
                    if isinstance(lbl, dict):
                        tid = lbl.get("tag_id")
                        if tid:
                            tag_counter[str(tid)] += 1

    return {
        "n_records": n_records,
        "n_with_labels": n_with_labels,
        "label_coverage": (n_with_labels / max(n_records, 1)) if n_records else 0,
        "source_dist": dict(source_counter.most_common()),
        "platform_dist": dict(platform_counter.most_common()),
        "rating_dist": dict(rating_counter.most_common()),
        "sentiment_dist": dict(sentiment_counter.most_common()),
        "nps_dist": dict(nps_counter.most_common()),
        "top_tags": dict(tag_counter.most_common(30)),
    }


def render_html(
    global_stats: dict[str, Any],
    depts: dict[str, dict[str, Any]],
    generated_at: str,
) -> str:
    def esc(s: Any) -> str:
        return html_lib.escape(str(s), quote=True)

    def fmt_num(n: int) -> str:
        return f"{n:,}"

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    dept_nav = "\n".join(
        f'<button class="tab-btn" data-tab="{esc(d)}" onclick="showTab(\'{esc(d)}\')">{esc(d)}</button>'
        for d in DEPARTMENTS
    )

    overview_cards = f"""
    <div class="card-grid">
      <div class="card"><div class="card-label">总评论数</div><div class="card-value">{fmt_num(global_stats["n_records"])}</div></div>
      <div class="card"><div class="card-label">有标签评论</div><div class="card-value">{fmt_num(global_stats["n_with_labels"])} ({fmt_pct(global_stats["label_coverage"])})</div></div>
      <div class="card"><div class="card-label">数据源</div><div class="card-value">{len(global_stats["source_dist"])} 个</div></div>
      <div class="card"><div class="card-label">标签种类</div><div class="card-value">{fmt_num(len(global_stats["top_tags"]))}+</div></div>
    </div>
    """

    source_rows = "\n".join(
        f"<tr><td>{esc(k)}</td><td class='right'>{fmt_num(v)}</td><td class='right'>{fmt_pct(v/global_stats['n_records'])}</td></tr>"
        for k, v in global_stats["source_dist"].items()
    )
    nps_rows = "\n".join(
        f"<tr><td>{esc(k)}</td><td class='right'>{fmt_num(v)}</td></tr>"
        for k, v in global_stats["nps_dist"].items()
    )
    sentiment_rows = "\n".join(
        f"<tr><td>{esc(k)}</td><td class='right'>{fmt_num(v)}</td></tr>"
        for k, v in global_stats["sentiment_dist"].items()
    )

    top_tag_rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{esc(tid)}</td><td class='right'>{fmt_num(n)}</td></tr>"
        for i, (tid, n) in enumerate(list(global_stats["top_tags"].items())[:20])
    )

    source_dist = global_stats["source_dist"]
    source_data_js = json.dumps(
        {"labels": list(source_dist.keys()), "data": list(source_dist.values())},
        ensure_ascii=False,
    )
    sentiment_data_js = json.dumps(
        {"labels": list(global_stats["sentiment_dist"].keys()),
         "data": list(global_stats["sentiment_dist"].values())},
        ensure_ascii=False,
    )

    dept_pages_html = []
    for dept, data in depts.items():
        maa = data.get("maa") or {}
        agrs = data.get("agrs") or {}
        maa_items = maa.get("items") or []
        agrs_groups = agrs.get("groups") or []

        srac_rows_html = []
        srac_chart_labels = []
        srac_chart_totals = []
        srac_chart_colors = []
        for i, item in enumerate(maa_items, 1):
            srac = item.get("srac") or {}
            pol = item.get("sentiment_polarity") or "中性"
            color = "#e74c3c" if pol == "负向" else ("#2ecc71" if pol == "正向" else "#95a5a6")
            srac_chart_labels.append(f"{item.get('tag_cn') or item.get('tag_id')}")
            srac_chart_totals.append(srac.get("total", 0))
            srac_chart_colors.append(color)
            srac_rows_html.append(
                f"<tr>"
                f"<td>{i}</td>"
                f"<td>{esc(item.get('tag_cn'))} <code>{esc(item.get('tag_id'))}</code></td>"
                f"<td><span class='badge badge-{pol}'>{esc(pol)}</span></td>"
                f"<td class='right'>{fmt_num(item.get('hit_count') or 0)}</td>"
                f"<td class='right'>{srac.get('severity', 0)}</td>"
                f"<td class='right'>{srac.get('reach', 0)}</td>"
                f"<td class='right'>{srac.get('actionability', 0)}</td>"
                f"<td class='right'>{srac.get('confidence', 0)}</td>"
                f"<td class='right score'><strong>{srac.get('total', 0)}</strong></td>"
                f"</tr>"
            )

        srac_data_js = json.dumps(
            {
                "labels": srac_chart_labels,
                "data": srac_chart_totals,
                "colors": srac_chart_colors,
            },
            ensure_ascii=False,
        )

        agrs_html = []
        for g in agrs_groups[:10]:
            top = g.get("top_sentences") or []
            pol_counts = g.get("distribution") or {}
            agrs_html.append(f"""
<div class="agrs-group">
  <h4>{esc(g.get("group_label") or g.get("group_key"))}</h4>
  <p class="group-meta">
    评论数 <strong>{fmt_num(g.get("n_reviews") or 0)}</strong> ·
    平均极性 <strong>{g.get("avg_polarity", 0):+.2f}</strong> ·
    主导 <strong>{esc(g.get("dominant_sentiment"))}</strong> ·
    正/中/负: {pol_counts.get("positive", 0)}/{pol_counts.get("neutral", 0)}/{pol_counts.get("negative", 0)}
  </p>
  <p class="group-summary">{esc(g.get("aggregate_summary"))}</p>
  <details>
    <summary>代表评论 ({len(top)} 条)</summary>
    <ul class="sentences">
      {"".join(f'<li><span class="sent-label sent-{esc(s.get("label",""))}">{esc(s.get("label",""))}</span> <code>{esc(s.get("review_id",""))}</code> — {esc(s.get("sentence",""))}</li>' for s in top[:5])}
    </ul>
  </details>
</div>
""")

        total_topics = len(maa_items)
        neg_topics = sum(1 for i in maa_items if i.get("sentiment_polarity") == "负向")
        pos_topics = sum(1 for i in maa_items if i.get("sentiment_polarity") == "正向")
        neu_topics = total_topics - neg_topics - pos_topics
        if total_topics > 0:
            top_total = maa_items[0].get("srac", {}).get("total", 0)
            bot_total = maa_items[-1].get("srac", {}).get("total", 0)
            spread = top_total - bot_total
        else:
            spread = 0

        dept_pages_html.append(f"""
<section id="tab-{esc(dept)}" class="tab-page" style="display:none;">
  <h2>{esc(dept)} — Top 10 行动建议</h2>

  <div class="card-grid">
    <div class="card"><div class="card-label">Top 话题数</div><div class="card-value">{total_topics}</div></div>
    <div class="card card-neg"><div class="card-label">负向话题</div><div class="card-value">{neg_topics}</div></div>
    <div class="card card-pos"><div class="card-label">正向话题</div><div class="card-value">{pos_topics}</div></div>
    <div class="card"><div class="card-label">中性话题</div><div class="card-value">{neu_topics}</div></div>
    <div class="card"><div class="card-label">SRAC 区分度</div><div class="card-value">{spread:.2f}</div></div>
  </div>

  <h3>SRAC 排序</h3>
  <div class="chart-container"><canvas id="srac-chart-{esc(dept)}" height="360"></canvas></div>

  <h3>Top 10 行动建议（SRAC 四维评分）</h3>
  <div class="table-wrap">
    <table class="srac-table">
      <thead>
        <tr><th>#</th><th>标签</th><th>极性</th><th class="right">命中</th>
          <th class="right">Sev</th><th class="right">Reach</th>
          <th class="right">Act</th><th class="right">Conf</th>
          <th class="right">Total</th></tr>
      </thead>
      <tbody>{"".join(srac_rows_html)}</tbody>
    </table>
  </div>

  <h3>AGRS 分组摘要</h3>
  {"".join(agrs_html) if agrs_html else "<p>（无分组数据）</p>"}

  <script>
    (function() {{
      var data = {srac_data_js};
      var ctx = document.getElementById('srac-chart-{esc(dept)}');
      if (ctx && window.Chart) {{
        new Chart(ctx.getContext('2d'), {{
          type: 'bar',
          data: {{
            labels: data.labels,
            datasets: [{{
              label: 'SRAC Total',
              data: data.data,
              backgroundColor: data.colors,
            }}],
          }},
          options: {{
            indexAxis: 'y',
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
              x: {{ beginAtZero: true, max: 10, title: {{ display: true, text: 'SRAC Total' }} }},
            }},
          }},
        }});
      }}
    }})();
  </script>
</section>
""")

    overview_charts_script = f"""
<script>
(function() {{
  if (!window.Chart) return;

  var srcData = {source_data_js};
  var srcCtx = document.getElementById('overview-source-chart');
  if (srcCtx) {{
    new Chart(srcCtx.getContext('2d'), {{
      type: 'doughnut',
      data: {{
        labels: srcData.labels,
        datasets: [{{ data: srcData.data, backgroundColor: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#95a5a6'] }}],
      }},
      options: {{ plugins: {{ legend: {{ position: 'right' }} }} }},
    }});
  }}

  var sentData = {sentiment_data_js};
  var sentCtx = document.getElementById('overview-sentiment-chart');
  if (sentCtx && sentData.labels.length > 0) {{
    new Chart(sentCtx.getContext('2d'), {{
      type: 'pie',
      data: {{
        labels: sentData.labels,
        datasets: [{{ data: sentData.data, backgroundColor: ['#2ecc71', '#95a5a6', '#e74c3c'] }}],
      }},
      options: {{ plugins: {{ legend: {{ position: 'right' }} }} }},
    }});
  }}
}})();
</script>
"""

    html_tpl = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>VOC BI Dashboard — 2026-W19</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, "PingFang SC", "Segoe UI", sans-serif; margin: 0; color: #333; background: #f5f7fa; }}
    .app {{ display: flex; min-height: 100vh; }}
    aside {{ width: 220px; background: #2c3e50; color: #ecf0f1; padding: 20px 0; position: sticky; top: 0; height: 100vh; overflow-y: auto; }}
    aside h1 {{ margin: 0 0 20px 20px; font-size: 16px; color: #ecf0f1; }}
    aside .gen {{ margin: 0 0 16px 20px; font-size: 11px; color: #95a5a6; }}
    .tab-btn {{ display: block; width: 100%; padding: 12px 20px; background: transparent; border: none; color: #bdc3c7; text-align: left; cursor: pointer; font-size: 14px; border-left: 3px solid transparent; }}
    .tab-btn:hover {{ background: #34495e; color: #fff; }}
    .tab-btn.active {{ background: #34495e; color: #fff; border-left-color: #3498db; }}
    main {{ flex: 1; padding: 24px 32px; max-width: 1400px; }}
    h2 {{ margin-top: 0; color: #2c3e50; }}
    h3 {{ color: #34495e; margin-top: 28px; }}
    h4 {{ margin: 12px 0 4px; color: #2c3e50; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #fff; padding: 16px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #3498db; }}
    .card-neg {{ border-left-color: #e74c3c; }}
    .card-pos {{ border-left-color: #2ecc71; }}
    .card-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
    .card-value {{ font-size: 22px; font-weight: 600; margin-top: 4px; color: #2c3e50; }}
    .chart-container {{ background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 24px; }}
    .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; text-align: left; }}
    th {{ background: #34495e; color: #fff; font-weight: 500; }}
    .right {{ text-align: right; }}
    .score {{ color: #e67e22; }}
    code {{ background: #ecf0f1; padding: 1px 6px; border-radius: 3px; font-size: 11px; }}
    .badge {{ padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }}
    .badge-正向 {{ background: #d5f5e3; color: #27ae60; }}
    .badge-负向 {{ background: #fadbd8; color: #c0392b; }}
    .badge-中性 {{ background: #ecf0f1; color: #7f8c8d; }}
    .agrs-group {{ background: #fff; padding: 14px 18px; margin-bottom: 12px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .group-meta {{ font-size: 12px; color: #7f8c8d; margin: 4px 0; }}
    .group-summary {{ font-size: 13px; color: #2c3e50; margin: 6px 0 8px; }}
    .sentences {{ list-style: none; padding: 0; font-size: 12px; }}
    .sentences li {{ padding: 6px 0; border-bottom: 1px dashed #ecf0f1; }}
    .sent-label {{ display: inline-block; min-width: 54px; padding: 0 4px; border-radius: 3px; font-size: 10px; text-transform: uppercase; }}
    .sent-positive {{ background: #d5f5e3; color: #27ae60; }}
    .sent-negative {{ background: #fadbd8; color: #c0392b; }}
    .sent-neutral {{ background: #ecf0f1; color: #7f8c8d; }}
    details summary {{ cursor: pointer; color: #3498db; font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>VOC BI 2026-W19</h1>
      <div class="gen">生成于 {esc(generated_at)}</div>
      <button class="tab-btn active" data-tab="overview" onclick="showTab('overview')">📊 总览</button>
      {dept_nav}
    </aside>
    <main>
      <section id="tab-overview" class="tab-page">
        <h2>总览</h2>
        {overview_cards}
        <div class="chart-row">
          <div class="chart-container">
            <h3 style="margin:0">数据源分布</h3>
            <canvas id="overview-source-chart" height="260"></canvas>
          </div>
          <div class="chart-container">
            <h3 style="margin:0">情感分布（全量）</h3>
            <canvas id="overview-sentiment-chart" height="260"></canvas>
          </div>
        </div>
        <h3>全局 Top-20 标签</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th>#</th><th>Tag ID</th><th class="right">命中数</th></tr></thead>
            <tbody>{top_tag_rows}</tbody>
          </table>
        </div>
        <h3>数据源详情</h3>
        <div class="table-wrap">
          <table>
            <thead><tr><th>来源</th><th class="right">记录数</th><th class="right">占比</th></tr></thead>
            <tbody>{source_rows}</tbody>
          </table>
        </div>
        <div class="chart-row">
          <div class="table-wrap">
            <h3 style="margin:16px">Proxy NPS</h3>
            <table>
              <thead><tr><th>类别</th><th class="right">数量</th></tr></thead>
              <tbody>{nps_rows}</tbody>
            </table>
          </div>
          <div class="table-wrap">
            <h3 style="margin:16px">情感分布</h3>
            <table>
              <thead><tr><th>类别</th><th class="right">数量</th></tr></thead>
              <tbody>{sentiment_rows}</tbody>
            </table>
          </div>
        </div>
      </section>
      {"".join(dept_pages_html)}
    </main>
  </div>
  {overview_charts_script}
  <script>
    function showTab(tab) {{
      document.querySelectorAll('.tab-page').forEach(function(p) {{ p.style.display = 'none'; }});
      document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      var page = document.getElementById('tab-' + tab);
      if (page) page.style.display = 'block';
      var btn = document.querySelector('.tab-btn[data-tab="' + tab + '"]');
      if (btn) btn.classList.add('active');
    }}
    showTab('overview');
  </script>
</body>
</html>
"""
    return html_tpl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6 D10 BI Dashboard Generator")
    ap.add_argument("--reports-dir", required=True, type=Path)
    ap.add_argument("--pred-jsonl", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)

    if not args.reports_dir.is_dir():
        print(f"❌ reports dir not found: {args.reports_dir}", file=sys.stderr); return 2
    if not args.pred_jsonl.is_file():
        print(f"❌ pred jsonl not found: {args.pred_jsonl}", file=sys.stderr); return 2

    print(f"⏳ Loading dept reports from {args.reports_dir}", file=sys.stderr)
    depts: dict[str, dict[str, Any]] = {}
    for dept in DEPARTMENTS:
        maa = load_dept_maa(args.reports_dir, dept)
        agrs = load_dept_agrs(args.reports_dir, dept)
        if maa or agrs:
            depts[dept] = {"maa": maa, "agrs": agrs}
        else:
            print(f"  ⚠️  missing data for {dept}", file=sys.stderr)

    print(f"⏳ Computing global stats from {args.pred_jsonl}", file=sys.stderr)
    global_stats = compute_global_stats(args.pred_jsonl)
    print(f"   n_records={global_stats['n_records']:,}", file=sys.stderr)

    generated_at = datetime.now().isoformat(timespec="seconds")
    html = render_html(global_stats, depts, generated_at)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    size_kb = args.output.stat().st_size / 1024
    print(f"✅ Dashboard: {args.output} ({size_kb:.1f} KB)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
