"""
Data-to-Dashboard Multi-Agent Visualization — 自然语言驱动 BI 仪表盘生成
paper2skills-code: 09-DataAgent-LLM | 母婴出海跨境电商
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataSource:
    name: str
    schema: dict[str, str]
    sample_data: list[dict]


@dataclass
class ChartSpec:
    chart_type: str         # bar / line / pie / scatter / table
    title: str
    x_field: str
    y_field: str
    color_field: str = ""
    aggregation: str = "sum"  # sum / avg / count / max / min
    filters: dict = field(default_factory=dict)


@dataclass
class Dashboard:
    title: str
    description: str
    charts: list[ChartSpec]
    sql_queries: list[str]
    insights: list[str]


class IntentParser:
    """解析用户自然语言 -> 仪表盘意图"""
    KEYWORDS = {
        "趋势": ("line", "date"),
        "对比": ("bar", "category"),
        "占比": ("pie", "category"),
        "分布": ("scatter", "value"),
        "明细": ("table", "all"),
    }

    def parse(self, query: str, data_source: DataSource) -> dict:
        chart_type = "bar"
        for kw, (ct, _) in self.KEYWORDS.items():
            if kw in query:
                chart_type = ct
                break
        numeric_fields = [k for k, v in data_source.schema.items()
                          if v in ("float", "int")]
        text_fields = [k for k, v in data_source.schema.items()
                       if v == "str"]
        date_fields = [k for k, v in data_source.schema.items()
                       if v == "date"]
        return {
            "chart_type": chart_type,
            "x_field": date_fields[0] if date_fields and "趋势" in query
                       else (text_fields[0] if text_fields else "category"),
            "y_field": numeric_fields[0] if numeric_fields else "value",
            "raw_query": query,
        }


class SQLGenerator:
    """生成对应分析 SQL"""
    def generate(self, intent: dict, table: str) -> str:
        x = intent["x_field"]
        y = intent["y_field"]
        agg = "SUM"
        if intent["chart_type"] == "line":
            return (f"SELECT {x}, {agg}({y}) AS metric "
                    f"FROM {table} GROUP BY {x} ORDER BY {x}")
        elif intent["chart_type"] == "pie":
            return (f"SELECT {x}, {agg}({y}) AS metric "
                    f"FROM {table} GROUP BY {x} ORDER BY metric DESC LIMIT 10")
        else:
            return (f"SELECT {x}, {agg}({y}) AS metric "
                    f"FROM {table} GROUP BY {x} ORDER BY metric DESC LIMIT 20")


class InsightGenerator:
    """从数据生成文字洞察"""
    def generate(self, data: list[dict], metric_field: str) -> list[str]:
        if not data:
            return ["数据为空，无法生成洞察"]
        values = [row.get(metric_field, 0) for row in data]
        max_val = max(values)
        min_val = min(values)
        avg_val = sum(values) / len(values)
        max_key = data[values.index(max_val)].get(list(data[0].keys())[0], "N/A")
        insights = [
            f"最高值 {max_val:.0f}（{max_key}），高于均值 {(max_val/avg_val-1)*100:.0f}%",
            f"最低值 {min_val:.0f}，与最高值相差 {(max_val-min_val)/avg_val*100:.0f}%",
            f"平均值 {avg_val:.0f}，数据共 {len(data)} 条记录",
        ]
        return insights


class DataToDashboardAgent:
    """Data-to-Dashboard 主 Agent（自然语言 -> 完整仪表盘规格）"""
    def __init__(self):
        self.parser = IntentParser()
        self.sql_gen = SQLGenerator()
        self.insight_gen = InsightGenerator()

    def generate(self, queries: list[str], data_source: DataSource) -> Dashboard:
        charts = []
        sqls = []
        all_insights = []
        for q in queries:
            intent = self.parser.parse(q, data_source)
            chart = ChartSpec(
                chart_type=intent["chart_type"],
                title=q,
                x_field=intent["x_field"],
                y_field=intent["y_field"],
            )
            sql = self.sql_gen.generate(intent, data_source.name)
            insights = self.insight_gen.generate(data_source.sample_data, intent["y_field"])
            charts.append(chart)
            sqls.append(sql)
            all_insights.extend(insights[:1])
        return Dashboard(
            title=f"{data_source.name} 分析仪表盘",
            description=f"基于 {len(queries)} 个分析维度自动生成",
            charts=charts, sql_queries=sqls, insights=all_insights,
        )


def run_dashboard_demo():
    ds = DataSource(
        name="baby_store_sales",
        schema={"date": "date", "category": "str", "revenue": "float",
                "orders": "int", "return_rate": "float"},
        sample_data=[
            {"date": "2026-06-01", "category": "奶粉", "revenue": 12000, "orders": 80},
            {"date": "2026-06-02", "category": "湿巾", "revenue": 3500, "orders": 120},
            {"date": "2026-06-03", "category": "玩具", "revenue": 5200, "orders": 60},
        ],
    )
    agent = DataToDashboardAgent()
    dashboard = agent.generate(
        ["分析各品类营收对比", "查看每日营收趋势", "各品类占比"],
        ds,
    )
    print(f"=== {dashboard.title} ==={dashboard.description}")
    for i, (chart, sql) in enumerate(zip(dashboard.charts, dashboard.sql_queries)):
        print(f"图表 {i+1}: [{chart.chart_type}] {chart.title}")
        print(f"  SQL: {sql}")
    print(f"洞察:" + "".join(f"  - {ins}" for ins in dashboard.insights))
    print("✅ 仪表盘生成完成")
if __name__ == "__main__":
    run_dashboard_demo()
