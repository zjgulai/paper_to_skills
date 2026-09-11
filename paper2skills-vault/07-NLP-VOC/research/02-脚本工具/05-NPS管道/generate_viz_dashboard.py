"""VOC 指标看板可视化

从 dashboard_and_insights.json 读取数据，生成 matplotlib 图表。
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUTPUT_BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/labeling-latest")


def load_data():
    with open(OUTPUT_BASE / "dashboard_and_insights.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_funnel(ax, funnel):
    """AIPL 旅程漏斗"""
    nodes = ["A", "I", "P1", "P2", "L1", "L2", "L3"]
    counts = [funnel.get(n, {}).get("count", 0) for n in nodes]
    colors = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6", "#1abc9c"]

    bars = ax.barh(nodes[::-1], counts[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"{count:,}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("VOC 条数", fontsize=10)
    ax.set_title("AIPL 旅程漏斗", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.15)


def plot_nps_gauge(ax, nps_data):
    """Proxy NPS 仪表盘"""
    proxy_nps = nps_data["proxy_nps"]
    promoters = nps_data["promoters"]
    detractors = nps_data["detractors"]
    passives = nps_data["passives"]
    total = promoters + detractors + passives

    # 绘制 NPS 弧形刻度
    theta = np.linspace(0, np.pi, 100)
    r = 1.0
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # 颜色分区
    ax.fill_between(x[:33], y[:33], 0, color="#e74c3c", alpha=0.3)
    ax.fill_between(x[33:66], y[33:66], 0, color="#f1c40f", alpha=0.3)
    ax.fill_between(x[66:], y[66:], 0, color="#2ecc71", alpha=0.3)

    # 指针
    nps_norm = (proxy_nps + 100) / 200  # -100~100 → 0~1
    needle_angle = np.pi * (1 - nps_norm)
    nx = 0.8 * np.cos(needle_angle)
    ny = 0.8 * np.sin(needle_angle)
    ax.plot([0, nx], [0, ny], "k-", linewidth=3)
    ax.plot(0, 0, "ko", markersize=10)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # 文字
    color = "#2ecc71" if proxy_nps > 0 else "#e74c3c"
    ax.text(0, -0.1, f"Proxy NPS: {proxy_nps}", ha="center", fontsize=18,
            fontweight="bold", color=color)
    ax.text(0, -0.25, f"推荐者 {promoters:,} ({promoters/total*100:.1f}%)", ha="center", fontsize=9, color="#2ecc71")
    ax.text(0, -0.35, f"贬损者 {detractors:,} ({detractors/total*100:.1f}%)", ha="center", fontsize=9, color="#e74c3c")
    ax.set_title("Proxy NPS", fontsize=12, fontweight="bold")


def plot_top_tags(ax, tag_dist):
    """Top 标签命中分布"""
    tags = list(tag_dist.keys())[:15]
    counts = list(tag_dist.values())[:15]

    bars = ax.barh(tags[::-1], counts[::-1], color="#3498db", edgecolor="white", height=0.7)
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{count:,}", va="center", fontsize=8)

    ax.set_xlabel("命中次数", fontsize=10)
    ax.set_title("Top 15 标签命中分布", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.2)


def plot_drivers(ax, drivers):
    """Promoter vs Detractor 驱动因素"""
    detractors = drivers["top_detractor_themes"][:5]
    promoters = drivers["top_promoter_themes"][:5]

    # Detractor themes (negative sentiment)
    det_names = [d["theme"] for d in detractors]
    det_rates = [d["mention_rate"] * 100 for d in detractors]

    # Promoter themes (positive sentiment)
    pro_names = [p["theme"] for p in promoters]
    pro_rates = [p["mention_rate"] * 100 for p in promoters]

    y_pos = np.arange(len(det_names))
    ax.barh(y_pos - 0.2, det_rates, 0.4, label="Detractor 驱动", color="#e74c3c", alpha=0.8)
    ax.barh(y_pos + 0.2, pro_rates, 0.4, label="Promoter 驱动", color="#2ecc71", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"D:{d}\nP:{p}" for d, p in zip(det_names, pro_names)], fontsize=8)
    ax.set_xlabel("提及率 (%)", fontsize=10)
    ax.set_title("Top 5 驱动因素对比", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)


def plot_persona(ax, persona_insights):
    """画像渗透率 + NPS"""
    names = []
    penetrations = []
    nps_values = []

    for name, info in persona_insights.items():
        if not name:
            continue
        names.append(name.replace("_", " ").title())
        penetrations.append(info["penetration"] * 100)
        nps_values.append(info["proxy_nps"]["proxy_nps"])

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, penetrations, width, label="渗透率 (%)", color="#3498db", alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, nps_values, width, label="Proxy NPS", color="#e67e22", alpha=0.8)

    ax.set_ylabel("渗透率 (%)", fontsize=10, color="#3498db")
    ax2.set_ylabel("Proxy NPS", fontsize=10, color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_title("画像渗透率 vs NPS", fontsize=12, fontweight="bold")

    # 添加数值标签
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, color="#3498db")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.0f}", ha="center", fontsize=8, color="#e67e22")

    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)


def plot_brand(ax, brand_data):
    """品牌提及分布"""
    brands = list(brand_data["brand_distribution"].keys())[:8]
    counts = list(brand_data["brand_distribution"].values())[:8]

    colors = plt.cm.Set3(np.linspace(0, 1, len(brands)))
    wedges, texts, autotexts = ax.pie(counts, labels=brands, autopct="%1.1f%%",
                                       colors=colors, startangle=90,
                                       textprops={"fontsize": 9})
    ax.set_title("品牌提及分布 (Top 8)", fontsize=12, fontweight="bold")


def plot_source_comparison(ax, source_dashboards):
    """各数据源指标对比"""
    sources = list(source_dashboards.keys())
    coverages = [source_dashboards[s]["tag_coverage"]["coverage_rate"] * 100 for s in sources]
    conflict_rates = [source_dashboards[s]["tag_coverage"].get("conflict_rate", 0) for s in sources]

    x = np.arange(len(sources))
    width = 0.35

    bars1 = ax.bar(x - width/2, coverages, width, label="覆盖率 (%)", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, conflict_rates, width, label="冲突率 (%)", color="#e74c3c", alpha=0.8)

    ax.set_ylabel("百分比 (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([s.title() for s in sources], fontsize=10)
    ax.set_title("各数据源指标对比", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)


def main():
    data = load_data()
    global_d = data["global_dashboard"]
    sources = data["source_dashboards"]

    # 构建全局标签分布
    global_tag_dist = {}
    for s_name, s_data in sources.items():
        for tag, count in s_data.get("tag_coverage", {}).items():
            if tag not in ("total_voc", "matched_voc", "unmatched_voc", "coverage_rate", "unique_tags_matched"):
                global_tag_dist[tag] = global_tag_dist.get(tag, 0) + count

    # 如果没有全局标签分布，从 source 汇总
    if not global_tag_dist:
        # 从 summary.json 汇总
        for s_name in sources:
            summary_path = OUTPUT_BASE / s_name / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    s_summary = json.load(f)
                for tag, count in s_summary.get("tag_distribution", {}).items():
                    global_tag_dist[tag] = global_tag_dist.get(tag, 0) + count

    # 创建大图 (3行2列)
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle("VOC Proxy NPS × AIPL 全旅程指标看板", fontsize=18, fontweight="bold", y=0.98)

    # 1. Proxy NPS 仪表盘
    ax1 = fig.add_subplot(3, 2, 1)
    plot_nps_gauge(ax1, global_d["proxy_nps"])

    # 2. AIPL 旅程漏斗
    ax2 = fig.add_subplot(3, 2, 2)
    plot_funnel(ax2, global_d["aipl_funnel"])

    # 3. 驱动因素对比
    ax3 = fig.add_subplot(3, 2, 3)
    plot_drivers(ax3, global_d["driver_analysis"])

    # 4. Top 标签分布
    ax4 = fig.add_subplot(3, 2, 4)
    plot_top_tags(ax4, dict(sorted(global_tag_dist.items(), key=lambda x: -x[1])))

    # 5. 画像洞察
    ax5 = fig.add_subplot(3, 2, 5)
    plot_persona(ax5, global_d["persona_insights"])

    # 6. 品牌分析
    ax6 = fig.add_subplot(3, 2, 6)
    plot_brand(ax6, global_d["brand_analysis"])

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_BASE / "viz_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"看板图表已保存: {output_path}")

    # 第二页：各数据源详细对比
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle("各数据源详细对比", fontsize=16, fontweight="bold", y=0.98)

    # 1. 覆盖率/冲突率对比
    ax1 = fig2.add_subplot(2, 2, 1)
    plot_source_comparison(ax1, sources)

    # 2. 各数据源 NPS
    ax2 = fig2.add_subplot(2, 2, 2)
    source_names = [s.title() for s in sources.keys()]
    source_nps = [sources[s]["proxy_nps"]["proxy_nps"] for s in sources]
    colors = ["#2ecc71" if n > 0 else "#e74c3c" for n in source_nps]
    bars = ax2.bar(source_names, source_nps, color=colors, edgecolor="white", alpha=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Proxy NPS", fontsize=10)
    ax2.set_title("各数据源 Proxy NPS", fontsize=12, fontweight="bold")
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}", ha="center", fontsize=10, fontweight="bold")

    # 3. 各数据源 AIPL 分布热力图
    ax3 = fig2.add_subplot(2, 2, 3)
    nodes = ["A", "I", "P1", "P2", "L1", "L2", "L3"]
    matrix = []
    for s_name in sources:
        funnel = sources[s_name]["aipl_funnel"]
        row = [funnel.get(n, {}).get("count", 0) for n in nodes]
        matrix.append(row)

    im = ax3.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks(range(len(nodes)))
    ax3.set_xticklabels(nodes)
    ax3.set_yticks(range(len(sources)))
    ax3.set_yticklabels([s.title() for s in sources])
    ax3.set_title("AIPL 节点分布热力图", fontsize=12, fontweight="bold")

    for i in range(len(sources)):
        for j in range(len(nodes)):
            text = ax3.text(j, i, matrix[i][j], ha="center", va="center",
                           color="white" if matrix[i][j] > max(max(row) for row in matrix) / 2 else "black",
                           fontsize=9)
    plt.colorbar(im, ax=ax3)

    # 4. 品牌提及对比
    ax4 = fig2.add_subplot(2, 2, 4)
    brand_counts = {}
    for s_name, s_data in sources.items():
        for brand, count in s_data.get("brand_analysis", {}).get("brand_distribution", {}).items():
            brand_counts[brand] = brand_counts.get(brand, 0) + count

    top_brands = sorted(brand_counts.items(), key=lambda x: -x[1])[:8]
    brand_names = [b[0] for b in top_brands]
    brand_vals = [b[1] for b in top_brands]
    ax4.barh(brand_names[::-1], brand_vals[::-1], color="#9b59b6", alpha=0.8)
    ax4.set_xlabel("提及次数", fontsize=10)
    ax4.set_title("品牌提及 Top 8", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path2 = OUTPUT_BASE / "viz_dashboard_sources.png"
    plt.savefig(output_path2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"数据源对比图表已保存: {output_path2}")

    print("\n✓ 可视化看板生成完成")


if __name__ == "__main__":
    main()
