"""
Data-to-Dashboard Mock Test — 无需API Key验证框架逻辑
"""
import os, sys
os.environ["OPENAI_API_KEY"] = "sk-fake-key-for-mock-testing"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from dashboard_agent import DataToDashboardAgent, Insight, ChartSpec


class MockResp:
    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]

class MockCC:
    def __init__(self, client): self.client = client
    def create(self, **kw): return self.client._create(**kw)

class MockChat:
    def __init__(self, cc): self.completions = cc

class MockClient:
    """模拟OpenAI客户端"""
    def __init__(self):
        self.call_count = 0
        self._cc = MockCC(self)
        self._chat = MockChat(self._cc)

    @property
    def chat(self): return self._chat

    def _create(self, **kw):
        self.call_count += 1
        content = kw['messages'][-1]['content']

        # 注意：匹配顺序很重要！更具体的条件放前面
        if 'SELECTED' in content or '保留' in content:
            return MockResp("SELECTED: 1, 2, 3, 4, 5")
        elif 'matplotlib' in content or '代码' in content:
            return MockResp("""```python
import matplotlib.pyplot as plt
df_grouped = df.groupby('platform')['revenue'].sum()
plt.figure(figsize=(8, 5))
df_grouped.plot(kind='bar')
plt.title('Revenue by Platform')
plt.tight_layout()
plt.savefig('/tmp/d2d_chart_bar.png')
plt.close()
```""")
        elif 'DOMAIN:' in content and '领域标签' in content:
            # 领域检测：要求同时匹配 DOMAIN: 和 领域标签（避免和概念提取混淆）
            return MockResp("DOMAIN: e-commerce-sales")
        elif 'metrics' in content and 'dimensions' in content:
            # 概念提取：要求同时匹配 metrics 和 dimensions
            return MockResp('{"metrics": ["sales_qty", "revenue", "roi"], "dimensions": ["platform", "sku"], "time_fields": ["date"], "id_fields": []}')
        elif 'trend' in content:
            return MockResp("INSIGHT: Amazon平台GMV呈稳步上升趋势\nCONFIDENCE: 0.85\nSUPPORTING: data")
        elif 'distribution' in content:
            return MockResp("INSIGHT: SKU销量呈帕累托分布\nCONFIDENCE: 0.90\nSUPPORTING: data")
        elif 'correlation' in content:
            return MockResp("INSIGHT: 广告spend与ROI弱负相关\nCONFIDENCE: 0.70\nSUPPORTING: data")
        elif 'anomaly' in content:
            return MockResp("INSIGHT: Shopify退货率异常升高至40%\nCONFIDENCE: 0.95\nSUPPORTING: data")
        elif 'composition' in content:
            return MockResp("INSIGHT: Amazon占GMV的45%\nCONFIDENCE: 0.88\nSUPPORTING: data")
        return MockResp("Mock response")


def create_test_df():
    np.random.seed(42)
    return pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=30),
        'platform': np.random.choice(['Amazon', 'Shopify', 'SHEIN'], 30),
        'sku': np.random.choice(['Bottle', 'Diaper', 'Stroller'], 30),
        'sales_qty': np.random.poisson(50, 30),
        'revenue': np.random.normal(500, 150, 30).round(2),
        'ad_spend': np.random.normal(100, 30, 30).round(2),
        'returns': np.random.poisson(3, 30),
    })


def test_domain_detection():
    print("\n[Test] 领域检测 Agent...")
    agent = DataToDashboardAgent()
    agent.client = MockClient()
    df = create_test_df()
    domain = agent._detect_domain(df, "电商销售数据")
    assert domain == "e-commerce-sales", f"期望 e-commerce-sales，实际 {domain}"
    print(f"  OK 领域检测: {domain}")


def test_concept_extraction():
    print("\n[Test] 概念提取 Agent...")
    agent = DataToDashboardAgent()
    agent.client = MockClient()
    df = create_test_df()
    concepts = agent._extract_concepts(df, "e-commerce-sales")
    assert 'metrics' in concepts and len(concepts['metrics']) > 0
    print(f"  OK 概念提取: {concepts}")


def test_multi_perspective():
    print("\n[Test] 多角度分析 Agent...")
    agent = DataToDashboardAgent()
    agent.client = MockClient()
    df = create_test_df()
    concepts = {"metrics": ["revenue"], "dimensions": ["platform"], "time_fields": ["date"]}
    insights = agent._multi_perspective_analysis(df, concepts)
    assert len(insights) > 0, f"期望至少1个洞察，实际 {len(insights)}"
    print(f"  OK 多角度分析: 生成 {len(insights)} 个洞察")
    for i, ins in enumerate(insights[:3], 1):
        print(f"    {i}. [{ins.perspective}] {ins.description[:40]}... ({ins.confidence})")


def test_self_reflection():
    print("\n[Test] 自反思 Agent...")
    agent = DataToDashboardAgent()
    agent.client = MockClient()
    insights = [
        Insight("trend", "GMV上升", 0.85),
        Insight("distribution", "帕累托", 0.90),
        Insight("correlation", "弱负相关", 0.70),
        Insight("anomaly", "退货率异常", 0.95),
        Insight("composition", "Amazon占比", 0.88),
        Insight("trend", "低质量趋势", 0.40),
    ]
    df = pd.DataFrame({'a': [1, 2, 3]})
    refined = agent._self_reflection(insights, df)
    assert len(refined) <= 5
    print(f"  OK 自反思: {len(insights)} -> {len(refined)} 个")


def test_chart_candidates():
    print("\n[Test] 图表候选生成...")
    agent = DataToDashboardAgent()
    agent.client = MockClient()
    insight = Insight("distribution", "SKU销量帕累托", 0.90)
    df = create_test_df()
    candidates = agent._generate_chart_candidates(insight, df)
    assert len(candidates) > 0
    print(f"  OK 图表候选: {len(candidates)} 个")


def test_expert_consensus():
    print("\n[Test] 专家共识...")
    agent = DataToDashboardAgent()
    insight = Insight("trend", "GMV上升", 0.85)
    valid_code = """
df_grouped = df.groupby('platform')['revenue'].sum()
plt.figure(figsize=(8, 5))
df_grouped.plot(kind='bar')
plt.title('Revenue')
plt.tight_layout()
plt.savefig('/tmp/d2d_chart_bar.png')
plt.close()
"""
    invalid_code = "raise ValueError('test')"
    candidates = [
        ChartSpec(insight, "bar", "Title", "X", "Y", valid_code),
        ChartSpec(insight, "line", "Title", "X", "Y", invalid_code),
    ]
    df = pd.DataFrame({'platform': ['A', 'B'], 'revenue': [500, 300]})
    best = agent._expert_consensus(candidates, df)
    assert best is not None
    print(f"  OK 专家共识: 选中 {best.chart_type} (评分:{best.score})")


def test_end_to_end():
    print("\n[Test] 端到端 Mock 测试...")
    agent = DataToDashboardAgent()
    agent.client = MockClient()
    df = create_test_df()
    result = agent.generate_dashboard(df, "母婴出海电商数据")
    assert 'domain' in result
    assert 'insights' in result and len(result['insights']) > 0
    assert 'charts' in result and len(result['charts']) > 0
    assert 'summary' in result
    print(f"  OK 端到端: 领域={result['domain']}, 洞察={len(result['insights'])}, 图表={len(result['charts'])}")
    print(f"\n  摘要预览:\n  {result['summary'][:200]}...")


def run_all_tests():
    print("=" * 60)
    print("Data-to-Dashboard Mock 测试套件")
    print("=" * 60)
    tests = [
        test_domain_detection,
        test_concept_extraction,
        test_multi_perspective,
        test_self_reflection,
        test_chart_candidates,
        test_expert_consensus,
        test_end_to_end,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
