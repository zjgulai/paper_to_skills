"""Zendesk 极简规则打标器（Phase 4 T3.3）

针对 Zendesk 客服工单的极简快速匹配规则：
- 适用场景：文本长度 < 30 字符的短工单
- 规则数：12 条
- 匹配方式：关键词子串匹配（无复杂否定检测，因文本极短）
- 输出格式：与通用标签器一致

标签空间（8维中的 问题类型 + 情感）：
- 问题类型：退货/换货/退款/物流/缺陷/缺失/尺码/取消/追踪/咨询
- 情感：正面感谢/中性咨询/负面投诉
"""

from typing import Optional


# ── 12 条极简规则定义 ──────────────────────────────────────────────

ZENDESK_RULES: list[dict] = [
    {
        "tag_id": "TAG_ZEN_R001",
        "tag_en": "return_request",
        "tag_cn": "退货请求",
        "aipl": "L3",
        "sentiment": "negative",
        "category": "售后",
        "keywords": [
            "return", "want to return", "return item", "send back",
            "returning", "return this", "return the", "return my",
            # 中文
            "退货", "想退货", "申请退货", "退回", "退回去",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R002",
        "tag_en": "exchange_request",
        "tag_cn": "换货请求",
        "aipl": "L3",
        "sentiment": "negative",
        "category": "售后",
        "keywords": [
            "exchange", "swap", "replace", "replacement", "exchanging",
            "swap for", "replace with", "get a replacement",
            # 中文
            "换货", "换一件", "更换", "换尺码", "换颜色",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R003",
        "tag_en": "refund_request",
        "tag_cn": "退款请求",
        "aipl": "P2",
        "sentiment": "negative",
        "category": "售后",
        "keywords": [
            "refund", "money back", "get refund", "full refund",
            "partial refund", "refund please", "request refund",
            # 中文
            "退款", "申请退款", "退钱", "全额退款",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R004",
        "tag_en": "shipping_inquiry",
        "tag_cn": "配送咨询",
        "aipl": "P2",
        "sentiment": "neutral",
        "category": "物流",
        "keywords": [
            "where is my", "where's my", "wheres my", "order status",
            "shipping status", "delivery status", "hasn't arrived",
            "hasnt arrived", "not arrived yet", "when will it arrive",
            "shipping delay", "delayed delivery",
            # 中文
            "快递", "物流", "发货", "送到", "配送", "什么时候到",
        ],
        "min_confidence": 0.70,
    },
    {
        "tag_id": "TAG_ZEN_R005",
        "tag_en": "defective_product",
        "tag_cn": "产品缺陷",
        "aipl": "L1",
        "sentiment": "negative",
        "category": "质量",
        "keywords": [
            "broken", "damaged", "doesn't work", "doesnt work",
            "not working", "defective", "faulty", "stopped working",
            "wont turn on", "won't turn on", "not functioning",
            "cracked", "torn", "leaking", "leak",
            # 中文
            "坏了", "不工作", "故障", " defective", "破损",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R006",
        "tag_en": "missing_parts",
        "tag_cn": "缺少配件",
        "aipl": "L1",
        "sentiment": "negative",
        "category": "配件",
        "keywords": [
            "missing", "not included", "parts missing", "missing part",
            "no adapter", "no cable", "no charger", "no manual",
            "incomplete", "not in the box", "box is empty",
            # 中文
            "缺配件", "少配件", "没收到", "缺少", "漏发",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R007",
        "tag_en": "size_issue",
        "tag_cn": "尺码问题",
        "aipl": "L1",
        "sentiment": "negative",
        "category": "尺码",
        "keywords": [
            "too small", "too big", "too large", "too tight",
            "too loose", "wrong size", "size wrong", "doesn't fit",
            "doesnt fit", "not my size", "runs small", "runs large",
            "fit issue", "sizing problem",
            # 中文
            "尺码不对", "太小", "太大", "不合身", "尺码偏小", "尺码偏大",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R008",
        "tag_en": "unwanted_order",
        "tag_cn": "不想要",
        "aipl": "A3",
        "sentiment": "negative",
        "category": "购买意愿",
        "keywords": [
            "don't want", "dont want", "changed mind", "no longer need",
            "not needed", "wrong order", "ordered by mistake",
            "accidental order", "didn't mean to", "didnt mean to",
            # 中文
            "不想要", "不要了", "买错了", "下单错了", "不想要了",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R009",
        "tag_en": "cancel_order",
        "tag_cn": "取消订单",
        "aipl": "A3",
        "sentiment": "negative",
        "category": "售后",
        "keywords": [
            "cancel", "cancel order", "cancel my order", "stop order",
            "stop shipment", "hold order", "do not ship", "don't ship",
            # 中文
            "取消订单", "取消", "不要发货", "停发",
        ],
        "min_confidence": 0.75,
    },
    {
        "tag_id": "TAG_ZEN_R010",
        "tag_en": "tracking_request",
        "tag_cn": "追踪查询",
        "aipl": "P2",
        "sentiment": "neutral",
        "category": "物流",
        "keywords": [
            "tracking", "track order", "track my", "tracking number",
            "where is", "where's the", "package location",
            "shipping update", "delivery update",
            # 中文
            "查物流", "追踪", "单号", "查询",
        ],
        "min_confidence": 0.70,
    },
    {
        "tag_id": "TAG_ZEN_R011",
        "tag_en": "positive_feedback",
        "tag_cn": "正面反馈",
        "aipl": "L3",
        "sentiment": "positive",
        "category": "满意度",
        "keywords": [
            "thank you", "thanks", "great", "love it", "excellent",
            "amazing", "perfect", "wonderful", "awesome", "fantastic",
            "happy with", "very happy", "so happy", "best purchase",
            "highly recommend", "would recommend",
            # 中文
            "感谢", "谢谢", "好评", "满意", "很喜欢",
        ],
        "min_confidence": 0.70,
    },
    {
        "tag_id": "TAG_ZEN_R012",
        "tag_en": "general_inquiry",
        "tag_cn": "一般咨询",
        "aipl": "A1",
        "sentiment": "neutral",
        "category": "咨询",
        "keywords": [
            "question", "help", "how to", "how do", "information",
            "need info", "wondering", "curious about", "can you tell me",
            "what is", "what's the", "whats the", "do you have",
            "is this", "does this", "can i",
            # 中文
            "请问", "咨询", "问下", "怎么", "如何",
        ],
        "min_confidence": 0.65,
    },
]


# ── 规则引擎 ───────────────────────────────────────────────────────

def apply_zendesk_rules(text: str, max_length: int = 50) -> list[dict]:
    """应用 Zendesk 极简规则

    Args:
        text: 工单文本
        max_length: 最大适用长度，超过则返回空（交由通用标签器处理）

    Returns: 标签列表（流水线标准格式）
    """
    if len(text.strip()) > max_length:
        return []

    text_lower = text.lower()
    labels: list[dict] = []
    matched_rules: set[str] = set()

    for rule in ZENDESK_RULES:
        tag_id = rule["tag_id"]
        if tag_id in matched_rules:
            continue

        # 关键词匹配：多词短语优先，短词用整词边界
        matched_kw = None
        for kw in rule["keywords"]:
            kw_lower = kw.lower()
            if " " in kw_lower or len(kw_lower) >= 5:
                # 多词或长词：子串匹配
                if kw_lower in text_lower:
                    matched_kw = kw
                    break
            else:
                # 短词：整词边界（CJK字符用子串匹配，英文用整词边界）
                import re
                has_cjk = any('\u4e00' <= c <= '\u9fff' for c in kw_lower)
                if has_cjk:
                    if kw_lower in text_lower:
                        matched_kw = kw
                        break
                else:
                    pattern = re.compile(r'\b' + re.escape(kw_lower) + r'\b')
                    if pattern.search(text_lower):
                        matched_kw = kw
                        break

        if matched_kw:
            sentiment = rule["sentiment"]
            confidence = rule["min_confidence"]
            # 关键词越长置信度越高
            kw_len = len(matched_kw.split())
            confidence = min(confidence + kw_len * 0.02, 0.9)

            labels.append({
                "tag_id": tag_id,
                "tag_en": rule["tag_en"],
                "tag_cn": rule["tag_cn"],
                "aipl_node": rule["aipl"],
                "sentiment_preset": sentiment,
                "sentiment_calibrated": 1.0 if sentiment == "positive" else (-1.0 if sentiment == "negative" else 0.0),
                "confidence": round(confidence, 2),
                "source": "zendesk_minimal_rule",
                "matched_keyword": matched_kw,
                "category": rule["category"],
            })
            matched_rules.add(tag_id)

    return labels


def should_use_minimal_rules(text: str, source: Optional[str] = None) -> bool:
    """判断是否应使用极简规则

    触发条件：
    1. 数据源为 zendesk 或包含 zendesk
    2. 文本长度 < 50 字符
    3. 非空文本
    """
    text = text.strip()
    if not text or len(text) > 50:
        return False
    if source and "zendesk" in source.lower():
        return True
    # 无 source 信息时，仅按长度判断
    return len(text) <= 50


# ── 冲突解决 ───────────────────────────────────────────────────────

def resolve_zendesk_conflicts(labels: list[dict]) -> list[dict]:
    """解决 Zendesk 标签冲突

    规则：
    1. 退货(R001) + 换货(R002) + 退款(R003) 互斥 → 按优先级保留：退款 > 退货 > 换货
    2. 配送咨询(R004) + 追踪查询(R010) 可共存
    3. 取消订单(R009) 与 退货(R001) 同时命中 → 保留取消订单（更早阶段）
    """
    tag_ids = {lbl["tag_id"] for lbl in labels}

    # 售后互斥组
    after_sales = {"TAG_ZEN_R001", "TAG_ZEN_R002", "TAG_ZEN_R003"}
    matched_after = tag_ids & after_sales
    if len(matched_after) > 1:
        # 优先级：退款 > 退货 > 换货
        priority = {"TAG_ZEN_R003": 3, "TAG_ZEN_R001": 2, "TAG_ZEN_R002": 1}
        keep_id = max(matched_after, key=lambda x: priority.get(x, 0))
        to_remove = matched_after - {keep_id}
        labels = [lbl for lbl in labels if lbl["tag_id"] not in to_remove]

    # 取消订单 vs 退货：保留取消订单
    if "TAG_ZEN_R009" in tag_ids and "TAG_ZEN_R001" in tag_ids:
        labels = [lbl for lbl in labels if lbl["tag_id"] != "TAG_ZEN_R001"]

    return labels


# ── 自证测试 ───────────────────────────────────────────────────────

def _test():
    """Zendesk 极简规则自证测试"""
    print("=" * 70)
    print("Zendesk 极简规则自证测试")
    print("=" * 70)

    test_cases = [
        # ── 售后请求 ──
        ("I want to return this", ["TAG_ZEN_R001"], "退货请求"),
        ("Please exchange for a new one", ["TAG_ZEN_R002"], "换货请求"),
        ("Need a refund please", ["TAG_ZEN_R003"], "退款请求"),
        ("cancel my order", ["TAG_ZEN_R009"], "取消订单"),

        # ── 物流 ──
        ("Where is my order?", ["TAG_ZEN_R004", "TAG_ZEN_R010"], "配送咨询+追踪"),
        ("Tracking number please", ["TAG_ZEN_R010"], "追踪查询"),

        # ── 产品问题 ──
        ("Pump is broken", ["TAG_ZEN_R005"], "产品缺陷"),
        ("Missing adapter", ["TAG_ZEN_R006"], "缺少配件"),
        ("Too small for me", ["TAG_ZEN_R007"], "尺码问题"),

        # ── 其他 ──
        ("I don't want it anymore", ["TAG_ZEN_R008"], "不想要"),
        ("Thank you so much!", ["TAG_ZEN_R011"], "正面反馈"),
        ("How do I use this?", ["TAG_ZEN_R012"], "一般咨询"),

        # ── 中文 ──
        ("想退货", ["TAG_ZEN_R001"], "中文-退货"),
        ("快递什么时候到", ["TAG_ZEN_R004"], "中文-配送咨询"),
        ("好评", ["TAG_ZEN_R011"], "中文-好评"),
        ("怎么使用", ["TAG_ZEN_R012"], "中文-咨询"),

        # ── 互斥 ──
        ("return and refund", ["TAG_ZEN_R003"], "互斥-退款优先"),
        ("cancel order and return", ["TAG_ZEN_R009"], "互斥-取消优先"),

        # ── 长度过滤 ──
        ("This is a very long message that definitely exceeds the fifty character limit for minimal rules", [], "超长文本过滤"),
    ]

    passed = 0
    failed = 0

    for text, expected_ids, desc in test_cases:
        labels = apply_zendesk_rules(text)
        labels = resolve_zendesk_conflicts(labels)
        actual_ids = sorted([l["tag_id"] for l in labels])
        expected = sorted(expected_ids)

        status = "PASS" if actual_ids == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {desc}")
        if status == "FAIL":
            print(f"    文本: '{text[:50]}...' ({len(text)} chars)")
            print(f"    期望: {expected}")
            print(f"    实际: {actual_ids}")
            for lbl in labels:
                print(f"      -> {lbl['tag_id']} ({lbl['tag_cn']}): kw={lbl.get('matched_keyword')}, conf={lbl['confidence']}")

    print(f"\n测试结果: {passed}/{passed + failed} 通过 ({passed / (passed + failed) * 100:.1f}%)")

    # 规则覆盖审计
    print("\n--- 规则覆盖审计 ---")
    print(f"  总规则数: {len(ZENDESK_RULES)}")
    for rule in ZENDESK_RULES:
        print(f"    {rule['tag_id']} [{rule['category']}] {rule['tag_cn']} ({rule['sentiment']}) - {len(rule['keywords'])} 关键词")

    # 长度分布审计
    print("\n--- 触发长度审计 ---")
    lengths = []
    for text, _, _ in test_cases:
        if len(text) <= 50:
            lengths.append(len(text))
    if lengths:
        print(f"  测试样本平均长度: {sum(lengths)/len(lengths):.1f} 字符")
        print(f"  最短: {min(lengths)}, 最长: {max(lengths)}")

    # 互斥组审计
    print("\n--- 互斥组审计 ---")
    print("  售后互斥: R001(退货) < R002(换货) < R003(退款)")
    print("  取消优先: R009(取消) > R001(退货)")

    print("=" * 70)
    return passed, failed


if __name__ == "__main__":
    _test()
