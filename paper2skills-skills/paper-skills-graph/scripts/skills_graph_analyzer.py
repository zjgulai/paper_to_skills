#!/usr/bin/env python3
"""
Skills Graph Analyzer
基于已有 Skill 卡片构建知识图谱，分析知识缺口并推荐新选题。

Usage:
    python skills_graph_analyzer.py --analyze          # 完整图谱分析
    python skills_graph_analyzer.py --skill Skill-Name # 分析特定技能
    python skills_graph_analyzer.py --gaps             # 仅显示知识缺口
    python skills_graph_analyzer.py --visualize        # 生成可视化图谱
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SkillNode:
    """技能图谱节点"""
    id: str
    domain: str
    difficulty: int = 0  # 1-5
    business_value: int = 0  # 1-5
    prerequisites: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=list)
    combinable: List[str] = field(default_factory=list)


@dataclass
class SkillEdge:
    """技能图谱边"""
    source: str
    target: str
    edge_type: str  # 'prerequisite', 'extension', 'combinable'
    weight: float = 1.0


class SkillsGraph:
    """技能图谱类"""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.nodes: Dict[str, SkillNode] = {}
        self.edges: List[SkillEdge] = []
        self.domain_mapping = {
            '01-因果推断': 'causal_inference',
            '02-A_B实验': 'ab_testing',
            '03-时间序列': 'time_series',
            '04-供应链': 'supply_chain',
            '05-推荐系统': 'recommendation',
            '06-增长模型': 'growth_model',
            '07-NLP-VOC': 'nlp_voc',
        }

    def parse_skill_file(self, file_path: Path) -> Optional[SkillNode]:
        """解析单个 Skill 文件"""
        content = file_path.read_text(encoding='utf-8')

        # 提取 skill 名称
        skill_name = file_path.stem

        # 提取领域
        domain = 'unknown'
        for cn_name, en_name in self.domain_mapping.items():
            if cn_name in str(file_path):
                domain = en_name
                break

        # 提取难度和业务价值（从星级评分）
        difficulty = self._extract_star_rating(content, '实施难度')
        business_value = self._extract_star_rating(content, '业务价值|商业价值|优先级')

        # 提取技能关系
        prerequisites = self._extract_section_items(content, '前置技能')
        extensions = self._extract_section_items(content, '延伸技能')
        combinable = self._extract_section_items(content, '可组合技能')

        return SkillNode(
            id=skill_name,
            domain=domain,
            difficulty=difficulty,
            business_value=business_value,
            prerequisites=prerequisites,
            extensions=extensions,
            combinable=combinable
        )

    def _extract_star_rating(self, content: str, pattern: str) -> int:
        """提取星级评分 (⭐)"""
        matches = re.findall(rf'{pattern}.*?([⭐]{{1,5}})', content)
        if matches:
            return matches[0].count('⭐')
        return 0

    def _extract_section_items(self, content: str, section_name: str) -> List[str]:
        """提取特定部分的列表项"""
        # 找到技能关联部分 - 匹配直到下一个 ## 开头的标题或文件结束
        section_pattern = r'## [④4]\.?\s*技能关联.*?(?=\n## |\Z)'
        section_match = re.search(section_pattern, content, re.DOTALL)

        if not section_match:
            return []

        section_content = section_match.group(0)

        # 在部分内查找特定子部分
        subsection_pattern = rf'### {section_name}.*?(?=###|\n## |\Z)'
        subsection_match = re.search(subsection_pattern, section_content, re.DOTALL)

        if not subsection_match:
            return []

        subsection_content = subsection_match.group(0)

        # 提取列表项 - 匹配多种格式
        items = []

        # 格式 1: - **技能名**：描述
        pattern1 = r'[-*]\s*\*\*([^*]+?)\*\*\s*[:：]'
        for match in re.finditer(pattern1, subsection_content):
            item = match.group(1).strip()
            if item and len(item) < 100:  # 过滤过长的文本
                items.append(item)

        # 格式 2: - **技能名** (无冒号描述)
        pattern2 = r'[-*]\s*\*\*([^*]+?)\*\*\s*$'
        for match in re.finditer(pattern2, subsection_content, re.MULTILINE):
            item = match.group(1).strip()
            if item and len(item) < 100 and item not in items:
                items.append(item)

        # 格式 3: - 技能名：描述 (无加粗)
        pattern3 = r'[-*]\s*([^\n*]+?)\s*[:：]\s*'
        for match in re.finditer(pattern3, subsection_content):
            item = match.group(1).strip()
            # 过滤掉纯描述性文字
            if item and len(item) < 50 and not item.startswith('http') and item not in items:
                items.append(item)

        return items

    def build_graph(self):
        """构建完整图谱"""
        # 扫描所有 Skill 文件
        skill_files = list(self.vault_path.glob('*/Skill-*.md'))
        skill_files.extend(self.vault_path.glob('*/Skill-*/**/*.md'))

        print(f"发现 {len(skill_files)} 个 Skill 文件")

        # 解析所有节点
        for file_path in skill_files:
            node = self.parse_skill_file(file_path)
            if node:
                self.nodes[node.id] = node

        print(f"成功解析 {len(self.nodes)} 个技能节点")

        # 构建边
        self._build_edges()

    def _build_edges(self):
        """根据节点关系构建边"""
        for node in self.nodes.values():
            # 前置技能边 (技能 -> 前置技能)
            for prereq in node.prerequisites:
                self.edges.append(SkillEdge(
                    source=node.id,
                    target=prereq,
                    edge_type='prerequisite',
                    weight=1.0
                ))

            # 延伸技能边 (技能 -> 延伸技能)
            for ext in node.extensions:
                self.edges.append(SkillEdge(
                    source=node.id,
                    target=ext,
                    edge_type='extension',
                    weight=0.8
                ))

            # 可组合技能边 (双向)
            for combo in node.combinable:
                self.edges.append(SkillEdge(
                    source=node.id,
                    target=combo,
                    edge_type='combinable',
                    weight=0.6
                ))

    def analyze_centrality(self) -> Dict:
        """分析节点中心性"""
        # 计算入度 (被依赖次数)
        in_degree = defaultdict(int)
        for edge in self.edges:
            in_degree[edge.target] += 1

        # 计算出度 (延伸数量)
        out_degree = defaultdict(int)
        for edge in self.edges:
            if edge.edge_type in ['extension', 'prerequisite']:
                out_degree[edge.source] += 1

        # 找出核心技能 (高入度)
        core_skills = sorted(
            [(node_id, count) for node_id, count in in_degree.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        # 找出潜力技能 (高出度且高价值)
        potential_skills = []
        for node_id, node in self.nodes.items():
            if node.business_value >= 4 and out_degree[node_id] == 0:
                potential_skills.append((node_id, node.business_value))

        potential_skills.sort(key=lambda x: x[1], reverse=True)

        return {
            'core_skills': core_skills,
            'potential_skills': potential_skills,
            'in_degree': dict(in_degree),
            'out_degree': dict(out_degree)
        }

    def _is_likely_skill_reference(self, text: str) -> bool:
        """判断文本是否像是一个 skill 引用（而非描述性文字）"""
        # 如果包含 Skill- 前缀，肯定是 skill 引用
        if 'Skill-' in text:
            return True

        # 过滤掉组合建议（包含 + 号）
        if ' + ' in text or '+' in text:
            return False

        # 如果文本很长或包含明显描述性词汇，可能不是 skill 引用
        descriptive_words = ['基础', '熟练', '理解', '掌握', '了解', '熟悉', 'python', 'pandas', 'sql']
        text_lower = text.lower()
        for word in descriptive_words:
            if word in text_lower:
                return False

        # 较短的、包含专业术语的文本更可能是 skill 引用
        technical_terms = ['Model', 'Forest', 'Estimation', 'Prediction', 'Forecasting',
                          'Uplift', 'Causal', 'Bandit', 'Matrix', 'Churn', 'LTV',
                          'Recommendation', 'Inventory', 'Optimization']
        for term in technical_terms:
            if term in text or term.lower() in text_lower:
                return True

        # 默认：不确定时视为非 skill 引用
        return False

    def _normalize_skill_name(self, text: str) -> str:
        """规范化 skill 名称用于匹配"""
        # 移除 Skill- 前缀（如果存在）
        text = re.sub(r'^Skill-', '', text, flags=re.IGNORECASE)
        # 转换为小写
        text = text.lower()
        # 统一连字符和空格
        text = text.replace('-', ' ')
        # 移除多余空格
        text = ' '.join(text.split())
        return text

    def _skill_exists(self, skill_name: str, all_skill_ids: Set[str]) -> bool:
        """检查 skill 是否存在（支持模糊匹配）"""
        normalized_input = self._normalize_skill_name(skill_name)

        for skill_id in all_skill_ids:
            normalized_id = self._normalize_skill_name(skill_id)
            # 完全匹配或包含匹配
            if normalized_input == normalized_id:
                return True
            # 如果输入包含在 skill_id 中，或 skill_id 包含输入
            if normalized_input in normalized_id or normalized_id in normalized_input:
                return True

        return False

    def find_knowledge_gaps(self) -> List[Dict]:
        """发现知识缺口"""
        gaps = []

        # 1. 前置缺口：边指向不存在的节点（且看起来像是 skill 引用）
        all_skill_ids = set(self.nodes.keys())
        for edge in self.edges:
            if not self._skill_exists(edge.target, all_skill_ids):
                # 只报告那些看起来像是 skill 引用的缺失
                if self._is_likely_skill_reference(edge.target):
                    gaps.append({
                        'type': 'missing_prerequisite',
                        'source_skill': edge.source,
                        'missing_skill': edge.target,
                        'priority': 'high',
                        'description': f"{edge.source} 依赖的 {edge.target} 尚未建立"
                    })

        # 2. 延伸缺口：高价值技能无延伸
        centrality = self.analyze_centrality()
        out_degree = centrality['out_degree']

        for node_id, node in self.nodes.items():
            if node.business_value >= 4 and out_degree.get(node_id, 0) == 0:
                gaps.append({
                    'type': 'missing_extension',
                    'skill': node_id,
                    'domain': node.domain,
                    'priority': 'high',
                    'description': f"高价值技能 {node_id} 缺少延伸方向"
                })

        # 3. 孤岛技能：无任何关联
        connected_skills = set()
        for edge in self.edges:
            connected_skills.add(edge.source)
            connected_skills.add(edge.target)

        for node_id in self.nodes:
            if node_id not in connected_skills:
                gaps.append({
                    'type': 'isolated_skill',
                    'skill': node_id,
                    'priority': 'medium',
                    'description': f"{node_id} 是孤立技能，无关联"
                })

        # 4. 领域间桥梁缺口
        domain_connections = self._analyze_domain_connections()
        for domain_a, domain_b in self._get_domain_pairs():
            if not domain_connections.get((domain_a, domain_b), []):
                gaps.append({
                    'type': 'missing_bridge',
                    'domain_a': domain_a,
                    'domain_b': domain_b,
                    'priority': 'medium',
                    'description': f"{domain_a} 与 {domain_b} 之间缺少桥梁连接"
                })

        return sorted(gaps, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])

    def _analyze_domain_connections(self) -> Dict[Tuple[str, str], List[str]]:
        """分析领域间的连接"""
        connections = defaultdict(list)

        for edge in self.edges:
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)

            if source_node and target_node:
                if source_node.domain != target_node.domain:
                    key = tuple(sorted([source_node.domain, target_node.domain]))
                    connections[key].append(f"{edge.source} -> {edge.target}")

        return connections

    def _get_domain_pairs(self) -> List[Tuple[str, str]]:
        """获取所有领域对"""
        domains = list(self.domain_mapping.values())
        pairs = []
        for i, a in enumerate(domains):
            for b in domains[i+1:]:
                pairs.append((a, b))
        return pairs

    def recommend_topics(self) -> List[Dict]:
        """基于缺口推荐选题"""
        gaps = self.find_knowledge_gaps()
        recommendations = []

        for gap in gaps:
            if gap['type'] == 'missing_prerequisite':
                recommendations.append({
                    'priority': 'P0',
                    'topic': f"基础: {gap['missing_skill']}",
                    'type': '前置技能填补',
                    'gap_match': gap['source_skill'],
                    'search_keywords': self._generate_search_keywords(gap['missing_skill']),
                    'rationale': f"{gap['source_skill']} 的前置基础，必须先掌握"
                })

            elif gap['type'] == 'missing_extension':
                skill = gap['skill']
                domain = gap['domain']
                recommendations.append({
                    'priority': 'P0',
                    'topic': f"延伸: {skill} 的高级应用",
                    'type': '技能延伸拓展',
                    'gap_match': skill,
                    'search_keywords': self._generate_extension_keywords(skill, domain),
                    'rationale': f"高价值技能 {skill} 需要更多延伸应用"
                })

            elif gap['type'] == 'missing_bridge':
                recommendations.append({
                    'priority': 'P1',
                    'topic': f"跨领域: {gap['domain_a']} + {gap['domain_b']}",
                    'type': '跨领域融合',
                    'gap_match': f"{gap['domain_a']}-{gap['domain_b']}",
                    'search_keywords': f"{gap['domain_a']} {gap['domain_b']} cross-domain",
                    'rationale': f"连接两个领域，创造新的应用场景"
                })

        return recommendations

    def _generate_search_keywords(self, skill_name: str) -> str:
        """生成搜索关键词"""
        # 移除 Skill- 前缀
        clean_name = skill_name.replace('Skill-', '').replace('-', ' ')
        return f"{clean_name} tutorial survey"

    def _generate_extension_keywords(self, skill_name: str, domain: str) -> str:
        """生成延伸方向关键词"""
        clean_name = skill_name.replace('Skill-', '').replace('-', ' ')

        # 根据领域生成相关关键词
        domain_keywords = {
            'causal_inference': 'dynamic pricing personalization',
            'ab_testing': 'sequential testing bandit',
            'time_series': 'uncertainty quantification transformer',
            'supply_chain': 'multi-echelon reinforcement learning',
            'recommendation': 'causal debiased LLM',
            'growth_model': 'churn LTV segmentation',
            'nlp_voc': 'sentiment aspect LLM'
        }

        related = domain_keywords.get(domain, 'advanced applications')
        return f"{clean_name} {related}"

    def generate_report(self) -> str:
        """生成完整的分析报告"""
        centrality = self.analyze_centrality()
        gaps = self.find_knowledge_gaps()
        recommendations = self.recommend_topics()

        report = f"""# Skills Graph 分析报告

## 1. 图谱概览

- **节点总数**: {len(self.nodes)} 个技能
- **边总数**: {len(self.edges)} 条关系
- **领域分布**:
"""

        # 领域分布
        domain_counts = defaultdict(int)
        for node in self.nodes.values():
            domain_counts[node.domain] += 1

        for domain, count in sorted(domain_counts.items()):
            report += f"  - {domain}: {count} 个\n"

        # 中心性分析
        report += f"""
## 2. 中心性分析

### 核心基础技能 (高被依赖数)
| 排名 | 技能 | 被依赖数 |
|-----|------|---------|
"""
        for i, (skill, count) in enumerate(centrality['core_skills'][:5], 1):
            report += f"| {i} | {skill} | {count} |\n"

        report += f"""
### 潜力延伸技能 (高价值无延伸)
| 排名 | 技能 | 业务价值 | 推荐延伸方向 |
|-----|------|---------|------------|
"""
        for i, (skill, value) in enumerate(centrality['potential_skills'][:5], 1):
            stars = '⭐' * value
            node = self.nodes.get(skill)
            keywords = self._generate_extension_keywords(skill, node.domain) if node else ''
            report += f"| {i} | {skill} | {stars} | {keywords} |\n"

        # 知识缺口
        report += f"""
## 3. 知识缺口

### 🔴 高优先级缺口

"""
        high_gaps = [g for g in gaps if g['priority'] == 'high']
        for i, gap in enumerate(high_gaps[:5], 1):
            report += f"""#### 缺口 {i}: {gap['type']}
- **描述**: {gap['description']}
"""
            if 'missing_skill' in gap:
                report += f"- **缺失技能**: {gap['missing_skill']}\n"
            if 'skill' in gap:
                report += f"- **相关技能**: {gap['skill']}\n"
            report += "\n"

        # 推荐选题
        report += f"""
## 4. 推荐选题列表

| 优先级 | 选题 | 类型 | 搜索关键词 |
|-------|------|------|-----------|
"""
        for rec in recommendations[:10]:
            report += f"| {rec['priority']} | {rec['topic']} | {rec['type']} | `{rec['search_keywords']}` |\n"

        report += f"""
## 5. 行动建议

1. **立即行动**: 优先填补 {len(high_gaps)} 个高优先级缺口
2. **本周计划**: 基于延伸缺口搜索 3-5 篇候选论文
3. **本月目标**: 建立跨领域桥梁，完成 1 个跨领域 skill
"""

        return report

    def export_graph_json(self, output_path: str):
        """导出图谱为 JSON 格式"""
        graph_data = {
            'nodes': [
                {
                    'id': node.id,
                    'domain': node.domain,
                    'difficulty': node.difficulty,
                    'business_value': node.business_value
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'source': edge.source,
                    'target': edge.target,
                    'type': edge.edge_type,
                    'weight': edge.weight
                }
                for edge in self.edges
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        print(f"图谱已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Skills Graph Analyzer')
    parser.add_argument('--analyze', action='store_true', help='运行完整分析')
    parser.add_argument('--skill', type=str, help='分析特定技能')
    parser.add_argument('--gaps', action='store_true', help='仅显示知识缺口')
    parser.add_argument('--export', type=str, help='导出图谱 JSON 文件路径')
    parser.add_argument('--vault', type=str,
                        default='/Users/pray/project/paper_to_skills/paper2skills-vault',
                        help='Vault 路径')

    args = parser.parse_args()

    # 初始化图谱
    graph = SkillsGraph(args.vault)
    graph.build_graph()

    if args.analyze or not any([args.skill, args.gaps, args.export]):
        # 完整分析
        report = graph.generate_report()
        print(report)

        # 保存报告
        report_path = Path('skills_graph_report.md')
        report_path.write_text(report, encoding='utf-8')
        print(f"\n报告已保存到: {report_path}")

    if args.skill:
        # 分析特定技能
        if args.skill in graph.nodes:
            node = graph.nodes[args.skill]
            print(f"\n=== Skill: {args.skill} ===")
            print(f"领域: {node.domain}")
            print(f"难度: {'⭐' * node.difficulty}")
            print(f"业务价值: {'⭐' * node.business_value}")
            print(f"前置技能: {node.prerequisites}")
            print(f"延伸技能: {node.extensions}")
            print(f"可组合: {node.combinable}")
        else:
            print(f"未找到技能: {args.skill}")

    if args.gaps:
        # 仅显示缺口
        gaps = graph.find_knowledge_gaps()
        print(f"\n=== 发现 {len(gaps)} 个知识缺口 ===\n")
        for gap in gaps:
            print(f"[{gap['priority'].upper()}] {gap['type']}")
            print(f"  {gap['description']}\n")

    if args.export:
        graph.export_graph_json(args.export)


if __name__ == '__main__':
    main()
