"""
OpenAI API 接入示例

演示如何将自迭代 LLM Agent 与 OpenAI API 集成,
用于生产环境的电商文案优化。

依赖:
    pip install openai

环境变量:
    OPENAI_API_KEY: OpenAI API 密钥

用法:
    python examples/openai_integration.py
"""

import os
import random
from openai import OpenAI

# 将上级目录加入路径以导入模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from self_improving_llm_agent import CopyOptimizationAgent, IntelligenceExtractionAgent


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def gpt4_generate(prompt: str, temperature: float = 0.7) -> str:
    """调用 GPT-4o 生成文本"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[Error] OpenAI API 调用失败: {e}")
        return ""


def gpt4_evaluate(eval_prompt: str, output: str) -> str:
    """调用 GPT-4o 进行评估（使用更低 temperature 保证稳定性）"""
    full_prompt = f"{eval_prompt}\n\n待评估输出: {output[:500]}"
    return gpt4_generate(full_prompt, temperature=0.2)


def demo_copy_optimization():
    """演示: 电商文案优化"""
    print("=" * 60)
    print("Demo: 电商文案自迭代优化 (OpenAI GPT-4o)")
    print("=" * 60)

    agent = CopyOptimizationAgent(
        generate_llm=lambda p: gpt4_generate(p, temperature=0.8),
        evaluator_llm=gpt4_evaluate,
        refine_llm=lambda p: gpt4_generate(p, temperature=0.5)
    )

    # 模拟 5 轮不同产品的文案生成和反馈
    products = [
        ("Momcozy S12 Pro 吸奶器", "职场妈妈", "professional"),
        ("Hegen 奶瓶 240ml", "新手妈妈", "warm"),
        ("贝亲婴儿湿巾 80片", "价格敏感妈妈", "friendly"),
    ]

    for i, (product, persona, tone) in enumerate(products):
        print(f"\n--- 产品 {i+1}: {product} ---")
        copy = agent.generate_copy(product, persona, tone)
        print(f"生成文案:\n{copy[:200]}...")

        # 模拟 CTR 反馈（实际应从 Amazon Advertising API 获取）
        ctr = random.uniform(0.01, 0.06)
        reflection = agent.record_ctr(
            context=f"产品: {product}\n目标用户: {persona}\n语气: {tone}",
            copy_text=copy,
            ctr=ctr
        )
        print(f"CTR: {ctr:.3f} | 反思: {reflection.improvement_hint[:80]}...")

    # 查看管线状态
    status = agent.get_dpo_status()
    print(f"\n管线状态:")
    print(f"  执行次数: {status['executions']}")
    print(f"  反思次数: {status['reflections']}")
    print(f"  DPO 数据对: {status['preference_pairs']}")
    print(f"  Top 失败模式: {status['top_failure_modes']}")

    # 导出 DPO 训练数据
    agent.export_dpo_data("/tmp/dpo_copy_data.jsonl")
    print(f"\nDPO 训练数据已导出至 /tmp/dpo_copy_data.jsonl")


def demo_intelligence_extraction():
    """演示: 竞品情报萃取"""
    print("\n" + "=" * 60)
    print("Demo: 竞品情报自萃取 (OpenAI GPT-4o)")
    print("=" * 60)

    agent = IntelligenceExtractionAgent(
        generate_llm=lambda p: gpt4_generate(p, temperature=0.3),
        evaluator_llm=gpt4_evaluate,
        refine_llm=lambda p: gpt4_generate(p, temperature=0.4)
    )

    # 模拟竞品页面内容
    raw_contents = [
        "New Release: Momcozy M5 Wearable Breast Pump - $179.99. Features: ultra-quiet 40dB, 28mm flange, 4 modes, 300min battery.",
        "BabyBuddha Breast Pump Kit - $129.99. Compact design, 3 suction levels, USB-C charging, hospital-grade suction.",
    ]

    for i, content in enumerate(raw_contents):
        print(f"\n--- 竞品 {i+1} ---")
        extraction = agent.extract_intelligence(content)
        print(f"萃取结果:\n{extraction[:300]}...")

        # 模拟准确率反馈（实际应通过与人工标注对比计算）
        accuracy = random.uniform(0.6, 0.95)
        reflection = agent.record_accuracy(
            context=f"原始内容: {content[:200]}",
            extraction=extraction,
            accuracy=accuracy
        )
        print(f"准确率: {accuracy:.2f} | 反思: {reflection.improvement_hint[:80]}...")

    status = agent.get_dpo_status()
    print(f"\n管线状态:")
    print(f"  执行次数: {status['executions']}")
    print(f"  DPO 数据对: {status['preference_pairs']}")

    agent.export_dpo_data("/tmp/dpo_intel_data.jsonl")
    print(f"\nDPO 训练数据已导出至 /tmp/dpo_intel_data.jsonl")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("错误: 请设置 OPENAI_API_KEY 环境变量")
        print("示例: export OPENAI_API_KEY=sk-xxx")
        sys.exit(1)

    demo_copy_optimization()
    demo_intelligence_extraction()
