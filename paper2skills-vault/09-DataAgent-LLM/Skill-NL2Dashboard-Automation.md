# Skill Card: NL2Dashboard Automation（自然语言→仪表盘）

> **领域**: 09-DataAgent-LLM | **类型**: 综合萃取

---

## ① 算法原理

将自然语言分析需求自动转化为 BI 仪表盘（图表+指标卡片+筛选器）。NL→结构化查询→Chart DSL→渲染。核心：意图解析（trend/comparison/distribution/ranking）+ 自动图表类型选择。

"过去 30 天各渠道 ROAS 趋势"→解析=trend + line_chart + group_by(channel) + metric(ROAS) + time_range(30d)。

---

## ② 母婴出海应用案例

运营说"给我看吸奶器在美国和德国的周销量对比，加趋势线"→自动生成双折线图+趋势线+自动洞察："德国周均增长 8%，美国 3%。德国增速更快。"

年化：节省 BI 开发人力 **15-25 万元**。

---

## ③ 代码模板

```python
def nl_to_chart_spec(query: str) -> dict:
    spec = {'chart_type': 'line', 'metrics': [], 'dimensions': [], 'filters': {}}
    q = query.lower()
    if '对比' in q or 'comparison' in q: spec['chart_type'] = 'bar'
    if '趋势' in q or 'trend' in q: spec['chart_type'] = 'line'
    if '分布' in q or 'distribution' in q: spec['chart_type'] = 'histogram'
    if 'roas' in q: spec['metrics'].append('ROAS')
    if '销量' in q or 'sales' in q: spec['metrics'].append('sales')
    if '国家' in q or 'country' in q: spec['dimensions'].append('country')
    return spec

r = nl_to_chart_spec("吸奶器在美国和德国的周销量对比趋势")
print(f"Chart: {r['chart_type']}, metrics={r['metrics']}, dims={r['dimensions']}")
assert r['chart_type'] == 'line' and 'sales' in r['metrics']
print("[✓] NL2Dashboard 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-RAG-Enhanced-Data-Analysis]] | [[Skill-Data-to-Dashboard]]
- **组合**：[[Skill-SQL-Agent-Text-to-SQL]]

---

- **可组合**：[[Skill-Root-Cause-Analysis-Agent]] / [[Skill-Data-to-Dashboard-Multi-Agent-Visualization]]

## ⑤ 商业价值：15-25 万元 | **难度**：⭐⭐⭐☆☆ | **优先级**：⭐⭐⭐☆☆
