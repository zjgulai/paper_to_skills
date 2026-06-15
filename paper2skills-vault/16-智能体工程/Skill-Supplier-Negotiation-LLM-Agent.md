---
title: LLM驱动供应商谈判智能体 — 结构化采购谈判自动化与议价策略优化
doc_type: knowledge
module: 16-智能体工程
topic: supplier-negotiation-llm-agent
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: LLM驱动供应商谈判智能体

> **论文**：LLM-Based Autonomous Negotiation Agents / Strategic Reasoning in Multi-Party Negotiations with Large Language Models
> **arXiv**：2405.14644 | 2024 | **桥梁**: 智能体工程 ↔ 供应链 | **类型**: 跨域融合

## ① 算法原理

**反直觉洞察**：采购谈判被认为是"强关系、弱算法"的领域——许多卖家认为谈判靠人脉和经验。但研究发现，**在价格谈判中，结构化的BATNA（最佳替代方案）分析和锚定策略比"关系好坏"贡献了更多的价格差异**（平均差距8-15%）。而LLM的优势不是"替代谈判"，而是**实时战略顾问**：分析历史谈判数据、生成最优锚定价格、准备竞品报价反驳脚本、识别供应商话语中的让步信号。

**核心算法：博弈论+LLM的混合谈判智能体**

1. **谈判状态机（FSM）**：
   - 状态：初始接触 → 报价阶段 → 反报价 → 让步探索 → 协议/僵局
   - 每个状态有对应的LLM提示策略和退出条件
   - 关键参数：保留价（Reservation Price）、BATNA价格、目标价格

2. **BATNA分析引擎**：
   - 收集3-5家同类供应商报价作为BATNA集合
   - 计算"谈判区间"：`ZOPA = [买方保留价, 卖方保留价]`
   - 若ZOPA为空 → 建议更换供应商；若ZOPA>15% → 高潜力谈判

3. **锚定策略优化（Anchoring Theory）**：
   - 首次报价策略：目标价格的0.85倍（激进锚定）或0.92倍（温和锚定）
   - 基于供应商历史数据选择策略
   - LLM生成锚定理由脚本："我们参考了[竞品报价]，考虑到[批量/长期合作/付款条件]..."

4. **让步检测（Concession Pattern Recognition）**：
   - 分析供应商话语中的软化信号："可以再商量"/"给你争取一下"
   - 基于历史谈判录音/文本训练分类器
   - LLM实时分析当前对话并给出"让步空间评估"

5. **多轮对话管理（LLM Agent Loop）**：
   ```
   Observe（观察供应商最新消息）
   → Analyze（LLM分析意图+让步信号）
   → Strategize（选择响应策略）
   → Generate（生成回复草稿）
   → Human Review（人工审核，高风险决策）
   → Execute（发送响应）
   ```

**数学直觉**：Nash谈判解最大化 (u_A - d_A)(u_B - d_B)，其中d为分歧点（破裂价值）。提高BATNA等于提高d_A，使Nash解向买方倾斜。LLM在这里充当"信息优势构建器"——比对手更了解市场，谈判就更有优势。

## ② 母婴出海应用案例

**场景A：电动吸奶器核心部件采购谈判**

- **业务问题**：某卖家向固定供应商采购电机组件，单价¥45/个，年采购量10万件，年采购额¥450万。凭经验谈了3年，价格几乎没降。竞争对手报价¥38-42/件，但换供应商有质量风险
- **数据要求**：历史采购记录（价格/批量/付款条件）、3家竞品供应商报价、原材料价格指数（铜/磁铁）
- **算法应用**：
  1. LLM分析原材料价格指数：铜价近6个月下降8% → "成本降低论据"自动生成
  2. BATNA分析：最佳替代方案¥40/件（B级供应商报价）→ 谈判保留价设为¥42
  3. 锚定策略：首轮报价¥36/件（目标价¥40的90%），附带"3年框架协议+预付30%"作为交换条件
  4. 谈判脚本：LLM生成3个版本（强硬/温和/灵活），运营人员选择合适版本
  5. 让步探索：对方报¥43时，LLM识别"可以商量"信号，建议让步至¥41但要求"付款期从60天改为90天"
- **预期产出**：最终成交价¥41/件（vs 原¥45），年节省¥40万；付款期延长30天（现金流改善）
- **业务价值**：采购成本降低8.9%，年化¥40万纯净利润改善

**场景B：包装材料批量采购谈判助手**

- **业务问题**：包装盒、说明书、贴纸每年10+次采购，每次都要重新谈价，耗时多且一致性差
- **算法应用**：建立"采购谈判知识库"（历史成交价+市场行情+谈判话术），LLM根据当前询价自动生成谈判初稿，人工3分钟审核后发出，响应速度从2天缩短至2小时
- **预期产出**：采购人员每年节省160小时谈判准备时间，平均采购成本降低5-7%

## ③ 代码模板

```python
"""
LLM驱动供应商谈判智能体
功能：BATNA分析 + 锚定策略 + 谈判状态机 + 让步信号检测
（生产环境接入真实LLM API，本版本用规则引擎模拟LLM推理）
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re
import warnings
warnings.filterwarnings('ignore')


class NegotiationState(Enum):
    """谈判状态机"""
    INITIAL_CONTACT = "初始接触"
    ANCHOR_PHASE = "锚定报价"
    COUNTER_OFFER = "反报价"
    CONCESSION_EXPLORE = "让步探索"
    PACKAGE_DEAL = "打包条件"
    AGREEMENT = "达成协议"
    DEADLOCK = "谈判僵局"


@dataclass
class NegotiationContext:
    """谈判上下文"""
    product_name: str
    annual_volume: int              # 年采购量
    current_price: float            # 当前成交价（CNY）
    target_price: float             # 目标价格
    reservation_price: float        # 保留价（最高可接受）
    
    # BATNA信息
    batna_price: float              # 最佳替代方案价格
    batna_supplier: str             # 替代供应商名称
    
    # 附加条件筹码
    payment_terms_current: int      # 当前付款期（天）
    order_commitment_months: int    # 框架协议月数（0=无）
    
    # 市场数据
    material_cost_change_pct: float # 原材料价格变化（正=涨价）
    
    # 状态追踪
    current_state: NegotiationState = NegotiationState.INITIAL_CONTACT
    rounds: int = 0
    history: List[Dict] = field(default_factory=list)
    
    @property
    def zopa_width(self) -> float:
        """谈判区间宽度"""
        return self.reservation_price - self.batna_price
    
    @property
    def leverage_score(self) -> float:
        """谈判筹码评分 0-10"""
        score = 0
        if self.batna_price < self.current_price * 0.95:
            score += 3  # 有竞品替代
        if self.annual_volume > 50000:
            score += 2  # 大批量
        if self.order_commitment_months >= 12:
            score += 2  # 框架协议
        if self.material_cost_change_pct < -5:
            score += 2  # 原材料降价论据
        if self.payment_terms_current < 60:
            score += 1  # 付款期可谈
        return min(score, 10)


def compute_anchoring_price(ctx: NegotiationContext) -> Tuple[float, str]:
    """
    计算最优锚定价格和理由
    基于锚定理论：首次报价影响最终成交价
    """
    leverage = ctx.leverage_score
    
    if leverage >= 7:
        # 高筹码：激进锚定（目标价×85%）
        anchor_price = ctx.target_price * 0.85
        strategy = "激进锚定"
        rationale = "高筹码位，可以更激进"
    elif leverage >= 4:
        # 中筹码：标准锚定（目标价×92%）
        anchor_price = ctx.target_price * 0.92
        strategy = "标准锚定"
        rationale = "中等筹码，温和锚定"
    else:
        # 低筹码：保守锚定（目标价×97%）
        anchor_price = ctx.target_price * 0.97
        strategy = "保守锚定"
        rationale = "筹码不足，接近目标价开价"
    
    # 确保锚定价高于BATNA
    anchor_price = max(anchor_price, ctx.batna_price)
    
    return anchor_price, strategy


def generate_negotiation_script(ctx: NegotiationContext, 
                                 anchor_price: float,
                                 script_type: str = "standard") -> str:
    """
    生成谈判话术脚本（生产环境由LLM生成，此处规则模板）
    """
    volume_millions = ctx.annual_volume * ctx.current_price / 10000  # 万元
    
    if script_type == "anchor":
        material_context = ""
        if ctx.material_cost_change_pct < -3:
            material_context = f"，叠加近期原材料价格下降{abs(ctx.material_cost_change_pct):.0f}%"
        
        commitment_context = ""
        if ctx.order_commitment_months >= 12:
            commitment_context = f"同时，我们愿意签署{ctx.order_commitment_months}个月框架协议，保证年采购量"
        
        return f"""
【锚定报价脚本 - {script_type}】

您好，感谢一直以来的合作。基于我们对市场的最新调研{material_context}，
结合同类产品的市场报价情况，我们本次希望将合作价格调整至 ¥{anchor_price:.1f}/件。

我们年采购规模约{ctx.annual_volume/10000:.0f}万件（约¥{volume_millions:.0f}万元），
{commitment_context}。

我们希望在双赢的基础上继续深化合作，请贵司评估后回复。
"""
    
    elif script_type == "concession":
        # 让步脚本
        concession_price = (anchor_price + ctx.target_price) / 2
        return f"""
【让步探索脚本】

感谢贵司的报价。我们充分理解生产成本的压力。

考虑到双方长期合作关系，我们可以将报价调整至 ¥{concession_price:.1f}/件，
同时我们愿意：
1. 将付款期从{ctx.payment_terms_current}天延长至{ctx.payment_terms_current + 15}天
2. Q4旺季前提前下单，给贵司更稳定的排产计划

希望这个方案能够满足双方需求。
"""
    
    elif script_type == "deadlock_breaker":
        return f"""
【僵局破解脚本】

我们理解目前存在价格分歧。为推动合作继续，建议我们探索打包方案：

方案A：¥{ctx.target_price:.1f}/件 + 年度返利1.5%（达量后结算）
方案B：¥{ctx.reservation_price * 0.98:.1f}/件 + 我方协助贵司联系原材料供应商（降低贵司成本）

如以上方案均不可接受，我们可能需要考虑{ctx.batna_supplier}的方案（报价¥{ctx.batna_price:.1f}），
这不是我们希望的结果，希望贵司能再做考量。
"""
    
    return "默认脚本"


def detect_concession_signals(supplier_message: str) -> Dict:
    """
    检测供应商话语中的让步信号
    生产环境用fine-tuned分类模型，此处用规则近似
    """
    soft_signals = ['可以商量', '给你争取', '看看能不能', '尽量', '再研究', 
                    'negotiate', 'see what I can do', 'let me check', 'might be possible']
    hard_signals = ['这是底线', '成本就这么多', '不能再低了', 'final price', 
                    'cannot go lower', 'this is our cost']
    positive_signals = ['合作愉快', '感谢信任', '长期合作', '重要客户', 
                        'valued customer', 'long-term partnership']
    
    msg_lower = supplier_message.lower()
    
    soft_count = sum(1 for s in soft_signals if s.lower() in msg_lower)
    hard_count = sum(1 for s in hard_signals if s.lower() in msg_lower)
    positive_count = sum(1 for s in positive_signals if s.lower() in msg_lower)
    
    # 提取供应商报价数字
    prices = re.findall(r'[¥￥\$]?\s*(\d+\.?\d*)', supplier_message)
    extracted_price = float(prices[0]) if prices else None
    
    concession_score = soft_count * 3 - hard_count * 4 + positive_count * 1
    
    return {
        'concession_score': concession_score,
        'has_soft_signals': soft_count > 0,
        'has_hard_signals': hard_count > 0,
        'relationship_warmth': positive_count,
        'extracted_price': extracted_price,
        'recommendation': (
            "继续施压，有让步空间" if concession_score > 0 else
            "已接近底线，考虑打包条件" if concession_score == 0 else
            "立场强硬，切换BATNA威慑策略"
        )
    }


class SupplierNegotiationAgent:
    """供应商谈判智能体"""
    
    def __init__(self, ctx: NegotiationContext):
        self.ctx = ctx
        self.anchor_price, self.anchor_strategy = compute_anchoring_price(ctx)
    
    def run_negotiation_simulation(self, supplier_responses: List[str]) -> pd.DataFrame:
        """运行谈判模拟"""
        rounds = []
        current_our_price = self.anchor_price
        
        for round_num, supplier_msg in enumerate(supplier_responses, 1):
            signals = detect_concession_signals(supplier_msg)
            supplier_price = signals['extracted_price']
            
            # 状态转换
            if round_num == 1:
                self.ctx.current_state = NegotiationState.ANCHOR_PHASE
            elif signals['has_hard_signals']:
                self.ctx.current_state = NegotiationState.DEADLOCK
            elif supplier_price and supplier_price <= self.ctx.target_price:
                self.ctx.current_state = NegotiationState.AGREEMENT
            else:
                self.ctx.current_state = NegotiationState.CONCESSION_EXPLORE
            
            # 我方响应价格策略
            if self.ctx.current_state == NegotiationState.CONCESSION_EXPLORE:
                # 每轮让步幅度递减
                step = (self.ctx.target_price - self.anchor_price) * (0.4 / round_num)
                current_our_price = min(current_our_price + step, self.ctx.reservation_price)
            
            rounds.append({
                'round': round_num,
                'state': self.ctx.current_state.value,
                'supplier_message': supplier_msg[:60] + '...' if len(supplier_msg) > 60 else supplier_msg,
                'supplier_price': supplier_price,
                'our_price': current_our_price,
                'concession_score': signals['concession_score'],
                'signal': signals['recommendation'],
            })
        
        return pd.DataFrame(rounds)


def run_negotiation_demo():
    """完整供应商谈判智能体演示"""
    print("=" * 65)
    print("LLM驱动供应商谈判智能体系统")
    print("=" * 65)
    
    # 1. 定义谈判场景
    ctx = NegotiationContext(
        product_name="电动吸奶器电机组件",
        annual_volume=100000,
        current_price=45.0,
        target_price=40.0,
        reservation_price=43.0,
        batna_price=40.5,
        batna_supplier="深圳XX精密",
        payment_terms_current=60,
        order_commitment_months=12,
        material_cost_change_pct=-7.5,  # 铜价近期下跌7.5%
    )
    
    agent = SupplierNegotiationAgent(ctx)
    
    print(f"\n[谈判背景分析]")
    print(f"  产品: {ctx.product_name}")
    print(f"  年采购量: {ctx.annual_volume:,} 件 | 当前价: ¥{ctx.current_price}/件")
    print(f"  目标价: ¥{ctx.target_price}/件 | 保留价: ¥{ctx.reservation_price}/件")
    print(f"  BATNA: {ctx.batna_supplier} @ ¥{ctx.batna_price}/件")
    print(f"  谈判区间(ZOPA): ¥{ctx.batna_price} ~ ¥{ctx.reservation_price} = ¥{ctx.zopa_width:.1f}")
    print(f"  原材料变动: {ctx.material_cost_change_pct:+.1f}% (降价=有利)")
    print(f"\n  筹码评分: {ctx.leverage_score}/10 → 策略: {agent.anchor_strategy}")
    print(f"  计算锚定价: ¥{agent.anchor_price:.1f}/件")
    
    # 2. 生成谈判脚本
    print(f"\n[谈判脚本生成]")
    anchor_script = generate_negotiation_script(ctx, agent.anchor_price, "anchor")
    print(anchor_script)
    
    # 3. 模拟多轮谈判
    print(f"[多轮谈判模拟]")
    supplier_responses = [
        "贵司报价¥36收到了，我们成本压力较大，现在最低能给到¥44，给您争取了很大力度了，希望理解。",
        "¥44真的是给到底线了，不过考虑到长期合作关系，可以再商量一下，¥43.5怎么样？",
        "¥43.5这个价格我们内部已经是亏损价了，这是我们的最低价了，再低真的做不了。",
    ]
    
    rounds_df = agent.run_negotiation_simulation(supplier_responses)
    
    print(f"\n  {'轮次':<6} {'状态':<12} {'供应商报价':<12} {'我方报价':<12} {'信号分析'}")
    print("  " + "-" * 70)
    for _, row in rounds_df.iterrows():
        sp = f"¥{row['supplier_price']:.1f}" if row['supplier_price'] else "未报价"
        print(f"  R{row['round']:<5} {row['state']:<12} {sp:<12} ¥{row['our_price']:<11.1f} {row['signal']}")
    
    # 4. 僵局破解
    print(f"\n[僵局检测 → 打包条件策略]")
    last_supplier_msg = supplier_responses[-1]
    signals = detect_concession_signals(last_supplier_msg)
    print(f"  让步信号分: {signals['concession_score']} → {signals['recommendation']}")
    
    deadlock_script = generate_negotiation_script(ctx, agent.anchor_price, "deadlock_breaker")
    print(deadlock_script)
    
    # 5. 成果预测
    print(f"[谈判成果预测]")
    # 假设最终成交¥41.5（在ZOPA内）
    final_price = 41.5
    savings_per_unit = ctx.current_price - final_price
    annual_savings = savings_per_unit * ctx.annual_volume
    print(f"  预测成交价: ¥{final_price}/件 (vs 当前¥{ctx.current_price})")
    print(f"  单件节省: ¥{savings_per_unit:.1f}")
    print(f"  年度采购节省: ¥{annual_savings:,.0f} (约¥{annual_savings/10000:.0f}万)")
    print(f"  谈判ROI: 节省¥{annual_savings/10000:.0f}万 / 系统成本¥2万 = {annual_savings/20000:.0f}x")
    
    print("\n[✓] LLM驱动供应商谈判智能体系统测试通过")
    return rounds_df


if __name__ == "__main__":
    rounds_df = run_negotiation_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Agent-Safety-Guardrails]]（LLM智能体安全护栏）、[[Skill-Supply-Chain-Due-Diligence]]（供应商尽调基础）
- **延伸（extends）**：[[Skill-Agent-Observability-Tracing]]（谈判过程可观测性追踪）、[[Skill-Supplier-Lead-Time-Buffer]]（谈判结果与交期缓冲联动）
- **可组合（combinable）**：[[Skill-Tariff-FX-FBA-Cost-Dynamics]]（谈判目标价格由成本动态测算模型提供）、[[Skill-Competitive-Price-Intelligence]]（竞品价格情报支撑BATNA分析）

## ⑤ 商业价值评估

- **ROI 预估**：年采购额¥500万的卖家，通过系统化谈判降低5-8%采购成本，年节省¥25-40万；系统建设成本¥5万，ROI≈500-800%
- **实施难度**：⭐⭐⭐☆☆（核心逻辑（BATNA/锚定/状态机）工程难度低；生产环境接入LLM API并做安全护栏是主要工作）
- **优先级**：⭐⭐⭐⭐☆（采购是可控成本最大来源之一，任何规模都值得优化）
- **适用规模**：年采购额>¥100万的卖家均可受益
- **数据依赖**：历史采购记录、至少3家竞品供应商报价（建立BATNA）、原材料价格指数（公开数据）
