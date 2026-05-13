"""
InstructUIE — 统一信息抽取框架
基于论文: Wang et al. "InstructUIE: Multi-task Instruction Tuning for Unified Information Extraction"

核心能力:
1. 统一框架 — 所有 IE 任务(实体识别NER/关系抽取RE/事件抽取EE)统一为 seq2seq 生成
2. 指令+选项机制 — Task Instruction + Options + Text → 结构化输出
3. 辅助任务 — span extraction, entity typing 等增强结构理解
4. 零样本泛化 — 新标签体系无需重训, 通过指令适配

母婴电商场景: 评论多任务统一抽取 (实体+关系+情感+事件)
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IEInstruction:
    """信息抽取指令模板"""
    task_type: str           # NER / RE / EE / SENTIMENT
    instruction: str         # 任务描述
    options: List[str]       # 候选标签/选项
    text: str                # 输入文本
    output_format: str       # 输出格式描述

    def to_prompt(self) -> str:
        """将指令转换为 LLM 可用的 prompt"""
        options_str = "\n".join(f"  - {opt}" for opt in self.options)
        return f"""Task: {self.task_type}
Instruction: {self.instruction}
Options:
{options_str}

Text: {self.text}

Output format: {self.output_format}

Answer:"""


@dataclass
class ExtractedStructure:
    """抽取的结构化结果"""
    task_type: str
    entities: List[Dict]     # [{text, type, start, end}, ...]
    relations: List[Dict]    # [{head, relation, tail}, ...]
    events: List[Dict]       # [{trigger, type, arguments}, ...]
    sentiment: Optional[Dict] = None


class InstructUIEBuilder:
    """
    InstructUIE 指令构建器

    将不同 IE 任务统一为文本生成格式,
    核心思想: P(output | input, instruction, options)
    """

    # 预定义的任务指令模板
    TEMPLATES = {
        'NER': {
            'instruction': 'Identify and extract all named entities from the text.',
            'output_format': 'entity1 [TYPE1], entity2 [TYPE2], ...'
        },
        'RE': {
            'instruction': 'Extract all relationships between entities in the text.',
            'output_format': '(head_entity, relation, tail_entity); ...'
        },
        'EE': {
            'instruction': 'Identify events and their arguments from the text.',
            'output_format': 'Event: trigger [TYPE] -> arg1 [ROLE1], arg2 [ROLE2], ...'
        },
        'SENTIMENT': {
            'instruction': 'Analyze the sentiment of the text towards specific aspects.',
            'output_format': 'aspect1: sentiment1 (positive/negative/neutral); ...'
        },
        'UNIFIED': {
            'instruction': 'Extract all entities, relations, and sentiments from the text.',
            'output_format': 'Entities: ... | Relations: ... | Sentiments: ...'
        }
    }

    def __init__(self, domain: str = 'maternal_baby'):
        self.domain = domain
        self.domain_options = self._load_domain_options()

    def _load_domain_options(self) -> Dict[str, Dict[str, List[str]]]:
        """加载领域特定的候选标签"""
        return {
            'maternal_baby': {
                'NER': ['PRODUCT', 'BRAND', 'ATTRIBUTE', 'USER_GROUP', 'SCENARIO'],
                'RE': ['has_attribute', 'positive_for', 'negative_for', 'compare_with',
                       'complement_of', 'alternative_to'],
                'EE': ['PURCHASE', 'RETURN', 'COMPLAINT', 'RECOMMENDATION'],
                'SENTIMENT': ['quality', 'price', 'logistics', 'packaging',
                             'safety', 'usability', 'appearance'],
            }
        }

    def build_ner_instruction(self, text: str,
                              entity_types: Optional[List[str]] = None) -> IEInstruction:
        """构建 NER 任务指令"""
        options = entity_types or self.domain_options[self.domain]['NER']
        template = self.TEMPLATES['NER']
        return IEInstruction(
            task_type='NER',
            instruction=template['instruction'],
            options=options,
            text=text,
            output_format=template['output_format']
        )

    def build_re_instruction(self, text: str,
                             relation_types: Optional[List[str]] = None) -> IEInstruction:
        """构建关系抽取任务指令"""
        options = relation_types or self.domain_options[self.domain]['RE']
        template = self.TEMPLATES['RE']
        return IEInstruction(
            task_type='RE',
            instruction=template['instruction'],
            options=options,
            text=text,
            output_format=template['output_format']
        )

    def build_sentiment_instruction(self, text: str,
                                    aspects: Optional[List[str]] = None) -> IEInstruction:
        """构建方面情感分析任务指令"""
        options = aspects or self.domain_options[self.domain]['SENTIMENT']
        template = self.TEMPLATES['SENTIMENT']
        return IEInstruction(
            task_type='SENTIMENT',
            instruction=template['instruction'],
            options=options,
            text=text,
            output_format=template['output_format']
        )

    def build_unified_instruction(self, text: str) -> IEInstruction:
        """构建统一抽取任务指令 (多任务合并)"""
        # 合并所有选项
        all_options = []
        for task_opts in self.domain_options[self.domain].values():
            all_options.extend(task_opts)

        template = self.TEMPLATES['UNIFIED']
        return IEInstruction(
            task_type='UNIFIED',
            instruction=template['instruction'],
            options=list(set(all_options)),
            text=text,
            output_format=template['output_format']
        )

    def build_auxiliary_span_instruction(self, text: str,
                                         entity_type: str) -> IEInstruction:
        """
        辅助任务: 实体跨度抽取

        帮助模型学习 "哪些文本片段是实体" 的通用能力,
        提升主任务的实体边界识别精度。
        """
        return IEInstruction(
            task_type='AUX_SPAN',
            instruction=f'Extract all text spans that represent a {entity_type} entity.',
            options=[entity_type],
            text=text,
            output_format='span1 [start:end], span2 [start:end], ...'
        )

    def build_auxiliary_typing_instruction(self, text: str,
                                           spans: List[str]) -> IEInstruction:
        """
        辅助任务: 实体类型分类

        给定实体跨度, 判断其类型。
        帮助模型学习实体类型之间的区分特征。
        """
        options = self.domain_options[self.domain]['NER']
        spans_text = "; ".join(f'"{s}"' for s in spans)
        return IEInstruction(
            task_type='AUX_TYPING',
            instruction=f'Classify the type of each given entity span: {spans_text}',
            options=options,
            text=text,
            output_format='span1 [TYPE1], span2 [TYPE2], ...'
        )


class SimpleInstructUIEEngine:
    """
    简化的 InstructUIE 推理引擎

    演示核心流程, 生产环境应替换为基于 Flan-T5 或 LLM 的实现。
    这里使用规则模拟 LLM 生成, 展示指令→输出的映射逻辑。
    """

    def __init__(self, builder: InstructUIEBuilder):
        self.builder = builder

    def predict(self, instruction: IEInstruction) -> str:
        """
        模拟 LLM 生成输出

        实际部署中, 这里应调用:
        - Flan-T5 (InstructUIE 原论文 backbone)
        - GPT-4 / Claude 等通用 LLM
        - 自研微调模型
        """
        text = instruction.text.lower()

        if instruction.task_type == 'NER':
            return self._simulate_ner(text, instruction.options)
        elif instruction.task_type == 'RE':
            return self._simulate_re(text, instruction.options)
        elif instruction.task_type == 'SENTIMENT':
            return self._simulate_sentiment(text, instruction.options)
        elif instruction.task_type == 'UNIFIED':
            ner = self._simulate_ner(text, self.builder.domain_options['maternal_baby']['NER'])
            sentiment = self._simulate_sentiment(text, self.builder.domain_options['maternal_baby']['SENTIMENT'])
            return f"Entities: {ner} | Sentiments: {sentiment}"
        else:
            return self._simulate_ner(text, instruction.options)

    def _simulate_ner(self, text: str, options: List[str]) -> str:
        """规则模拟 NER (仅用于演示)"""
        results = []

        # 产品词典
        products = ['spectra s1', 'medela pump', '吸奶器', '储奶袋', '温奶器', '奶瓶']
        brands = ['spectra', 'medela', 'lansinoh', 'avent', 'dr brown']
        attributes = ['静音', '便携', '防胀气', '双边', '电动']

        for prod in products:
            if prod in text:
                results.append(f"{prod} [PRODUCT]")
        for brand in brands:
            if brand in text:
                results.append(f"{brand} [BRAND]")
        for attr in attributes:
            if attr in text:
                results.append(f"{attr} [ATTRIBUTE]")

        return ", ".join(results) if results else "None"

    def _simulate_re(self, text: str, options: List[str]) -> str:
        """规则模拟关系抽取"""
        relations = []

        if '比' in text or 'vs' in text or 'compared to' in text:
            relations.append("(product1, compare_with, product2)")
        if '还买了' in text or 'also bought' in text or '搭配' in text:
            relations.append("(product1, complement_of, product2)")
        if '不好' in text or '差' in text or '慢' in text:
            relations.append("(product, negative_for, aspect)")
        if '好' in text or '推荐' in text or '不错' in text:
            relations.append("(product, positive_for, aspect)")

        return "; ".join(relations) if relations else "None"

    def _simulate_sentiment(self, text: str, aspects: List[str]) -> str:
        """规则模拟方面情感分析"""
        sentiments = []

        # 简单规则匹配
        positive_words = ['好', '不错', '推荐', '满意', '喜欢', 'good', 'great', 'love']
        negative_words = ['差', '不好', '失望', '慢', '漏', '坏', 'bad', 'slow', 'poor']

        aspect_keywords = {
            'quality': ['质量', '做工', '品质', 'quality', 'durable'],
            'price': ['价格', '贵', '便宜', '划算', 'price', 'expensive'],
            'logistics': ['物流', '快递', '配送', 'shipping', 'delivery'],
            'packaging': ['包装', '破损', 'packaging', 'box'],
            'safety': ['安全', '放心', '有机', 'safe', 'organic'],
            'usability': ['方便', '好用', '简单', 'easy', 'convenient'],
            'appearance': ['外观', '颜色', '好看', 'cute', 'design'],
        }

        for aspect, keywords in aspect_keywords.items():
            for kw in keywords:
                if kw in text:
                    # 检查附近情感词
                    pos_count = sum(1 for w in positive_words if w in text)
                    neg_count = sum(1 for w in negative_words if w in text)
                    if pos_count > neg_count:
                        sentiments.append(f"{aspect}: positive")
                    elif neg_count > pos_count:
                        sentiments.append(f"{aspect}: negative")
                    else:
                        sentiments.append(f"{aspect}: neutral")
                    break

        return "; ".join(sentiments) if sentiments else "None"


# ============================================
# 母婴电商评论抽取示例
# ============================================

def run_maternal_baby_extraction():
    """运行母婴电商评论统一抽取示例"""
    print("=" * 70)
    print("InstructUIE — 统一信息抽取框架")
    print("母婴电商评论多任务抽取演示")
    print("=" * 70)

    # 初始化
    builder = InstructUIEBuilder(domain='maternal_baby')
    engine = SimpleInstructUIEEngine(builder)

    # 测试评论数据
    reviews = [
        {
            'id': 'R001',
            'text': 'Spectra S1 吸奶器非常好用，静音效果很好，晚上不会吵醒宝宝。价格有点贵但值得。',
            'lang': 'zh'
        },
        {
            'id': 'R002',
            'text': 'Compared to Medela Pump, Spectra S1 is much quieter and more portable. Love the built-in battery!',
            'lang': 'en'
        },
        {
            'id': 'R003',
            'text': '买了吸奶器之后又买了储奶袋和温奶器，搭配使用很方便。物流太慢了等了一周。',
            'lang': 'zh'
        },
        {
            'id': 'R004',
            'text': 'Dr Brown 奶瓶防胀气效果不错，但是包装破损了，客服态度很好给换了新的。',
            'lang': 'zh'
        },
    ]

    print("\n[1] 单任务抽取演示\n")

    for review in reviews[:2]:
        print(f"评论 {review['id']}: {review['text'][:50]}...")

        # NER
        ner_inst = builder.build_ner_instruction(review['text'])
        ner_result = engine.predict(ner_inst)
        print(f"  [NER]      {ner_result}")

        # 情感
        sent_inst = builder.build_sentiment_instruction(review['text'])
        sent_result = engine.predict(sent_inst)
        print(f"  [Sentiment] {sent_result}")

        # 关系
        re_inst = builder.build_re_instruction(review['text'])
        re_result = engine.predict(re_inst)
        print(f"  [RE]       {re_result}")
        print()

    print("\n[2] 统一多任务抽取 (一次调用)\n")

    for review in reviews:
        unified_inst = builder.build_unified_instruction(review['text'])
        unified_result = engine.predict(unified_inst)
        print(f"评论 {review['id']}:")
        print(f"  输入: {review['text'][:60]}...")
        print(f"  输出: {unified_result}")
        print()

    print("\n[3] 辅助任务演示\n")

    # Span extraction 辅助任务
    text = 'Spectra S1 吸奶器静音效果很好'
    span_inst = builder.build_auxiliary_span_instruction(text, 'PRODUCT')
    print(f"辅助任务 (Span Extraction): {text}")
    print(f"  Prompt:\n{span_inst.to_prompt()[:200]}...")
    print()

    # Typing 辅助任务
    typing_inst = builder.build_auxiliary_typing_instruction(text, ['Spectra S1', '吸奶器', '静音'])
    print(f"辅助任务 (Entity Typing): {text}")
    print(f"  Prompt:\n{typing_inst.to_prompt()[:200]}...")

    print("\n" + "=" * 70)


def demonstrate_zero_shot_adaptation():
    """
    演示零样本标签适配

    业务场景: 业务新增 "环保认证" 标签, 无需重新训练模型,
    只需在指令选项中加入新标签即可。
    """
    print("\n" + "=" * 70)
    print("场景演示: 零样本新标签适配")
    print("=" * 70)

    builder = InstructUIEBuilder(domain='maternal_baby')
    engine = SimpleInstructUIEEngine(builder)

    # 原始标签体系
    original_aspects = ['quality', 'price', 'logistics', 'packaging', 'safety']
    print(f"原始标签体系: {original_aspects}")

    text = "这款产品通过了欧盟环保认证，材质很安全，就是价格有点高。"
    inst_original = builder.build_sentiment_instruction(text, original_aspects)
    result_original = engine.predict(inst_original)
    print(f"原始体系输出: {result_original}")

    # 新增 "环保认证" 标签 (零样本)
    new_aspects = original_aspects + ['eco_certification']
    print(f"\n新增标签后: {new_aspects}")

    inst_new = IEInstruction(
        task_type='SENTIMENT',
        instruction='Analyze sentiment including eco-certification concerns.',
        options=new_aspects,
        text=text,
        output_format='aspect: sentiment; ...'
    )

    print(f"\n新标签适配无需重训模型!")
    print(f"只需修改指令中的 options 列表即可。")
    print(f"模型通过指令理解新标签语义, 利用预训练知识进行零样本推断。")


def demonstrate_instruction_design_principles():
    """演示指令设计的核心原则"""
    print("\n" + "=" * 70)
    print("InstructUIE 指令设计原则")
    print("=" * 70)

    principles = """
    1. 指令必须包含任务语义 (Task Instruction)
       → 让模型理解 "要抽什么"
       例: "Extract all product attributes mentioned in the review"

    2. 选项必须约束输出空间 (Options)
       → 防止模型 hallucination, 限定合法输出范围
       例: Options: [quality, price, safety, logistics]

    3. 输出格式必须结构化 (Output Format)
       → 确保输出可被程序解析
       例: "aspect: sentiment (positive/negative/neutral)"

    4. 辅助任务增强通用能力
       → Span extraction 学习 "什么是实体"
       → Entity typing 学习 "实体类型区分"
       → 辅助任务与主任务联合训练, 提升泛化

    5. 多任务指令混合训练
       → NER + RE + EE + Sentiment 统一格式
       → 模型学习跨任务共享的结构知识
       → 零样本场景利用指令迁移能力
    """
    print(principles)


if __name__ == '__main__':
    # 主流程
    run_maternal_baby_extraction()

    # 场景演示
    demonstrate_zero_shot_adaptation()
    demonstrate_instruction_design_principles()

    print("\n" + "=" * 70)
    print("生产环境部署建议:")
    print("  1. 使用 Flan-T5-XL 作为 backbone (InstructUIE 原论文)")
    print("  2. 在 IE INSTRUCTIONS (32数据集) 上微调")
    print("  3. 收集领域评论数据继续微调 (domain adaptation)")
    print("  4. 指令模板版本化管理, 便于 A/B 测试")
    print("  5. 输出增加 Pydantic schema 校验, 确保结构化")
    print("=" * 70)
