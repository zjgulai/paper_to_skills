#!/usr/bin/env python3
"""
知识图谱可视化脚本
支持导出为 Gephi/D3.js 格式

Usage:
    python 知识图谱可视化脚本.py --format gephi --output graph
    python 知识图谱可视化脚本.py --format d3 --output graph.json
    python 知识图谱可视化脚本.py --viz --output graph.html
"""

import sys
import json
import argparse
from pathlib import Path

# 添加 skills_graph_analyzer 到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                       'paper2skills-skills/paper-skills-graph/scripts'))

from skills_graph_analyzer import SkillsGraph


def export_gephi(graph: SkillsGraph, output_prefix: str):
    """导出为 Gephi 兼容的 CSV 格式"""
    import pandas as pd

    # 节点表
    nodes_data = []
    for node in graph.nodes.values():
        nodes_data.append({
            'Id': node.id,
            'Label': node.id.replace('Skill-', ''),
            'Domain': node.domain,
            'Difficulty': node.difficulty,
            'BusinessValue': node.business_value,
            'InDegree': sum(1 for e in graph.edges if e.target == node.id),
            'OutDegree': sum(1 for e in graph.edges if e.source == node.id)
        })

    nodes_df = pd.DataFrame(nodes_data)
    nodes_file = f"{output_prefix}_nodes.csv"
    nodes_df.to_csv(nodes_file, index=False, encoding='utf-8')
    print(f"节点表已导出: {nodes_file}")

    # 边表
    edges_data = []
    for edge in graph.edges:
        edges_data.append({
            'Source': edge.source,
            'Target': edge.target,
            'Type': 'Directed',
            'Relationship': edge.edge_type,
            'Weight': edge.weight
        })

    edges_df = pd.DataFrame(edges_data)
    edges_file = f"{output_prefix}_edges.csv"
    edges_df.to_csv(edges_file, index=False, encoding='utf-8')
    print(f"边表已导出: {edges_file}")

    # 生成 Gephi 导入说明
    readme = f"""
Gephi 导入步骤:
1. 打开 Gephi
2. 数据实验室 -> 导入电子表格
3. 先导入节点表: {nodes_file}
4. 再导入边表: {edges_file}
5. 概览页面选择布局算法 (推荐: Force Atlas 2)
6. 根据 Domain 字段着色
7. 根据 Weight 调整边粗细
"""
    readme_file = f"{output_prefix}_README.txt"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"导入说明: {readme_file}")


def export_d3_json(graph: SkillsGraph, output_file: str):
    """导出为 D3.js 力导向图 JSON"""
    # 节点
    nodes = []
    domain_colors = {
        'causal_inference': '#3498db',      # 蓝色
        'ab_testing': '#2ecc71',            # 绿色
        'time_series': '#f39c12',           # 橙色
        'supply_chain': '#9b59b6',          # 紫色
        'recommendation': '#e74c3c',        # 红色
        'growth_model': '#1abc9c',          # 青色
        'nlp_voc': '#ff69b4',               # 粉色
        'unknown': '#95a5a6'                # 灰色
    }

    for node in graph.nodes.values():
        nodes.append({
            'id': node.id,
            'name': node.id.replace('Skill-', ''),
            'domain': node.domain,
            'difficulty': node.difficulty,
            'businessValue': node.business_value,
            'group': list(domain_colors.keys()).index(node.domain)
                        if node.domain in domain_colors else 7,
            'color': domain_colors.get(node.domain, '#95a5a6')
        })

    # 边
    links = []
    for edge in graph.edges:
        links.append({
            'source': edge.source,
            'target': edge.target,
            'type': edge.edge_type,
            'value': edge.weight
        })

    data = {
        'nodes': nodes,
        'links': links
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"D3.js 数据已导出: {output_file}")

    # 生成 D3.js HTML 模板
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Paper2Skills 知识图谱</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #graph {{ width: 100vw; height: 100vh; }}
        .node {{ stroke: #fff; stroke-width: 1.5px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .node text {{ font-size: 10px; pointer-events: none; }}
        #legend {{
            position: absolute; top: 10px; left: 10px;
            background: rgba(255,255,255,0.9); padding: 10px;
            border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .legend-item {{ display: flex; align-items: center; margin: 5px 0; }}
        .legend-color {{ width: 15px; height: 15px; margin-right: 8px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="legend">
        <h3>领域分类</h3>
        <div class="legend-item"><div class="legend-color" style="background:#3498db"></div>因果推断</div>
        <div class="legend-item"><div class="legend-color" style="background:#2ecc71"></div>A/B实验</div>
        <div class="legend-item"><div class="legend-color" style="background:#f39c12"></div>时间序列</div>
        <div class="legend-item"><div class="legend-color" style="background:#9b59b6"></div>供应链</div>
        <div class="legend-item"><div class="legend-color" style="background:#e74c3c"></div>推荐系统</div>
        <div class="legend-item"><div class="legend-color" style="background:#1abc9c"></div>增长模型</div>
        <div class="legend-item"><div class="legend-color" style="background:#ff69b4"></div>NLP-VOC</div>
    </div>
    <script>
        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#graph").append("svg")
            .attr("width", width)
            .attr("height", height);

        d3.json("{output_file}").then(data => {{
            const simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2));

            const link = svg.append("g")
                .selectAll("line")
                .data(data.links)
                .enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", d => Math.sqrt(d.value * 5));

            const node = svg.append("g")
                .selectAll("g")
                .data(data.nodes)
                .enter().append("g")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            node.append("circle")
                .attr("class", "node")
                .attr("r", d => 5 + d.businessValue * 2)
                .attr("fill", d => d.color);

            node.append("text")
                .attr("dx", 12)
                .attr("dy", ".35em")
                .text(d => d.name);

            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
            }});

            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
            }}

            function dragged(event, d) {{
                d.fx = event.x; d.fy = event.y;
            }}

            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null; d.fy = null;
            }}
        }});
    </script>
</body>
</html>"""

    html_file = output_file.replace('.json', '.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    print(f"D3.js 可视化页面: {html_file}")


def generate_graph_summary(graph: SkillsGraph) -> str:
    """生成图谱摘要报告"""
    lines = []
    lines.append("# 知识图谱摘要报告\n")
    lines.append(f"生成时间: 2026-04-04\n")
    lines.append(f"技能总数: {len(graph.nodes)}")
    lines.append(f"关系总数: {len(graph.edges)}\n")

    # 领域分布
    lines.append("## 领域分布\n")
    domain_counts = {}
    for node in graph.nodes.values():
        domain_counts[node.domain] = domain_counts.get(node.domain, 0) + 1

    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count + "░" * (10 - count)
        lines.append(f"- {domain:20s} {bar} {count}")

    lines.append("")

    # 中心性分析
    lines.append("## 中心性分析\n")
    in_degrees = {}
    out_degrees = {}
    for edge in graph.edges:
        in_degrees[edge.target] = in_degrees.get(edge.target, 0) + 1
        out_degrees[edge.source] = out_degrees.get(edge.source, 0) + 1

    # 高入度技能 (被依赖多)
    lines.append("### 核心基础技能 (高被依赖)\n")
    top_in = sorted(in_degrees.items(), key=lambda x: -x[1])[:5]
    for skill, degree in top_in:
        lines.append(f"- {skill}: 被 {degree} 个技能依赖")

    lines.append("")

    # 高出度技能 (延伸多)
    lines.append("### 核心发展技能 (高延伸)\n")
    top_out = sorted(out_degrees.items(), key=lambda x: -x[1])[:5]
    for skill, degree in top_out:
        lines.append(f"- {skill}: 延伸出 {degree} 个技能")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='知识图谱可视化工具')
    parser.add_argument('--format', choices=['gephi', 'd3'], required=True,
                       help='导出格式')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件前缀或路径')
    parser.add_argument('--summary', action='store_true',
                       help='同时生成摘要报告')

    args = parser.parse_args()

    # 构建图谱
    print("正在构建知识图谱...")
    vault_path = '/Users/pray/project/paper_to_skills/paper2skills-vault'
    graph = SkillsGraph(vault_path)
    graph.build_graph()
    print(f"构建完成: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")

    # 导出
    if args.format == 'gephi':
        export_gephi(graph, args.output)
    elif args.format == 'd3':
        export_d3_json(graph, args.output)

    # 生成摘要
    if args.summary:
        summary = generate_graph_summary(graph)
        summary_file = f"{args.output}_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"摘要报告: {summary_file}")

    print("\n完成!")


if __name__ == '__main__':
    main()
