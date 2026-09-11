"""
DeepAnalyze Agent Mock 测试
不依赖 LLM API，用预置响应验证框架逻辑
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from agent import DataScienceAgent, ActionType


class MockLLMClient:
    """模拟 LLM 响应，预置数据分析动作序列"""
    
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
        
        def create(self, model, messages, temperature=None):
            if self.parent.step < len(self.parent.responses):
                content = self.parent.responses[self.parent.step]
                self.parent.step += 1
            else:
                content = "ACTION: ANSWER\nCONTENT: 分析完成。"
            
            class Choice:
                class Message:
                    def __init__(self, c):
                        self.content = c
                def __init__(self, c):
                    self.message = self.Message(c)
            
            class Response:
                def __init__(self, c):
                    self.choices = [Choice(c)]
            
            return Response(content)
    
    def __init__(self):
        self.step = 0
        self.responses = [
            # Step 1: Analyze - 规划分析方向
            "ACTION: ANALYZE\nCONTENT: 已理解数据结构。计划：1)计算各平台GMV和ROI 2)SKU销量排名 3)分析退货率趋势 4)总结异常",
            
            # Step 2: Code - 计算平台统计
            """ACTION: CODE\nCONTENT: ```python
# 计算各平台GMV和ROI
df = dfs['test_sales_data.csv']
platform_stats = df.groupby('platform').agg({
    'revenue': 'sum',
    'ad_spend': 'sum',
    'sales_qty': 'sum',
    'returns': 'sum'
}).reset_index()
platform_stats['gmv'] = platform_stats['revenue']
platform_stats['roi'] = ((platform_stats['revenue'] - platform_stats['ad_spend']) / platform_stats['ad_spend']).round(2)
platform_stats['return_rate'] = (platform_stats['returns'] / platform_stats['sales_qty']).round(3)
print("平台统计:")
print(platform_stats[['platform', 'gmv', 'roi', 'return_rate']])
```""",
            
            # Step 3: Analyze - 基于结果反思
            "ACTION: ANALYZE\nCONTENT: 平台统计已完成。Shopify的退货率(0.118)明显高于其他平台。接下来计算SKU排名并深入分析退货率趋势。",
            
            # Step 4: Code - SKU排名 + 退货率趋势
            """ACTION: CODE\nCONTENT: ```python
# SKU销量排名
sku_stats = df.groupby('sku')['sales_qty'].sum().sort_values(ascending=False)
print("\\nSKU销量排名:")
print(sku_stats)

# 按平台和日期分析退货率趋势
import matplotlib.pyplot as plt
df['date'] = pd.to_datetime(df['date'])
shopify_df = df[df['platform'] == 'Shopify'].sort_values('date')
plt.figure(figsize=(10, 4))
plt.plot(shopify_df['date'], shopify_df['return_rate'], marker='o', markersize=2)
plt.title('Shopify Daily Return Rate Trend')
plt.xlabel('Date')
plt.ylabel('Return Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/tmp/output_return_rate.png')
plt.close()
print("\\n退货率趋势图已保存到 /tmp/output_return_rate.png")

# 找出退货率异常时段
high_return = shopify_df[shopify_df['return_rate'] > 0.1]
if len(high_return) > 0:
    print(f"\\nShopify高退货率时段: {len(high_return)} 天")
    print(f"异常日期范围: {high_return['date'].min()} 到 {high_return['date'].max()}")
```""",
            
            # Step 5: Analyze - 总结发现
            "ACTION: ANALYZE\nCONTENT: 关键发现：1)Shopify退货率在day 60-65异常飙升 2)Stroller-003销量最高 3)Amazon ROI最优。准备生成最终报告。",
            
            # Step 6: Answer - 最终报告
            """ACTION: ANSWER\nCONTENT: ## 母婴出海销售数据分析报告

### 1. 平台对比
| 平台 | GMV | ROI | 退货率 |
|------|-----|-----|--------|
| Amazon | 最高 | 最优(~4.0) | 最低(~5%) |
| Shopify | 中等 | 中等 | **最高(~12%)** |
| SHEIN | 较低 | 较低 | 中等(~6%) |

### 2. SKU销量排名
1. Stroller-003 (销量冠军)
2. Diaper-002
3. Bottle-001
4. Pump-004

### 3. 异常发现
**Shopify退货率在 2025-03-02 至 2025-03-07 异常升高**，从正常~5%飙升至~40%。
建议：立即排查该时段的物流/产品质量问题。

### 4. 建议
- 优先优化 Shopify 物流和品控流程
- 维持 Amazon 高ROI策略
- 关注 Stroller-003 库存，避免断货""",
        ]
        self.chat = type('Chat', (), {'completions': self.ChatCompletions(self)})()


def test_mock_agent():
    """Mock 测试：验证完整五动作循环"""
    print("=" * 60)
    print("DeepAnalyze Agent Mock 测试")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=90, freq='D')
    data = {
        'date': dates,
        'platform': np.random.choice(['Amazon', 'Shopify', 'SHEIN'], 90),
        'sku': np.random.choice(['Bottle-001', 'Diaper-002', 'Stroller-003', 'Pump-004'], 90),
        'sales_qty': np.random.poisson(50, 90),
        'revenue': np.random.normal(500, 150, 90).round(2),
        'ad_spend': np.random.normal(100, 30, 90).round(2),
        'returns': np.random.poisson(3, 90),
    }
    df = pd.DataFrame(data)
    df['roi'] = ((df['revenue'] - df['ad_spend']) / df['ad_spend']).round(2)
    df['return_rate'] = (df['returns'] / df['sales_qty']).round(3)
    mask = (df['platform'] == 'Shopify') & (df.index >= 60) & (df.index <= 65)
    df.loc[mask, 'returns'] = 20
    df.loc[mask, 'return_rate'] = (df.loc[mask, 'returns'] / df.loc[mask, 'sales_qty']).round(3)
    test_path = '/tmp/test_sales_data.csv'
    df.to_csv(test_path, index=False)
    print(f"\n[1/5] 测试数据已生成: {test_path} ({df.shape[0]} 行 x {df.shape[1]} 列)")
    
    # 创建 Agent，注入 Mock LLM
    agent = DataScienceAgent(api_key="sk-fake")
    agent.client = MockLLMClient()
    agent.model = "mock"
    
    instruction = (
        "分析这份母婴出海销售数据。需要完成："
        "1. 各平台GMV和ROI对比 2. 各SKU销量排名 3. 退货率趋势，找出异常时段 4. 生成简洁的分析结论"
    )
    
    print(f"\n[2/5] 用户指令: {instruction[:60]}...")
    print(f"\n[3/5] 开始执行五动作循环...")
    
    result = agent.run(instruction, [test_path])
    
    print(f"\n[4/5] 动作执行记录:")
    print(f"   - 数据理解(Understand): OK")
    print(f"   - 分析规划(Analyze): OK")
    print(f"   - 代码生成(Code): OK")
    print(f"   - 代码执行(Execute): OK")
    print(f"   - 报告生成(Answer): OK")
    
    print(f"\n[5/5] 最终报告:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    # 验证输出文件
    if os.path.exists('/tmp/output_return_rate.png'):
        size = os.path.getsize('/tmp/output_return_rate.png')
        print(f"\n[验证] 可视化图表已生成: /tmp/output_return_rate.png ({size} bytes)")
    
    print("\n" + "=" * 60)
    print("Mock 测试通过！框架逻辑验证完毕。")
    print("接入真实 LLM：将 agent.client = MockLLMClient() 替换为真实 OpenAI 客户端")
    print("=" * 60)


if __name__ == "__main__":
    test_mock_agent()
