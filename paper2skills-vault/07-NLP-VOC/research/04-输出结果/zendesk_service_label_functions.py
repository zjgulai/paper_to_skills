"""Zendesk Service Label Functions (Auto-Generated)

为 Zendesk 工单设计的 AIPL 售中/售后标签体系。
每个函数遵循: (text) -> (matched: bool, confidence: float)

Usage:
    from zendesk_service_label_functions import apply_all
    results = apply_all("I want to return my order")
"""


def lf_Order_Placement(text: str) -> tuple[bool, float]:
    """Label Function: 下单/购买 -> P1/neutral
    
    触发关键词: "place order", "placed an order", "order number", "order id", "checkout", "cart", "purchase", "bought", "buying"
    """
    text_lower = text.lower()
    
    keywords = ["place order", "placed an order", "order number", "order id", "checkout", "cart", "purchase", "bought", "buying"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Payment_Issue(text: str) -> tuple[bool, float]:
    """Label Function: 支付/账单问题 -> P2/negative
    
    触发关键词: "payment", "pay", "charged", "charge", "billing", "invoice", "credit card", "paypal", "transaction", "money was taken"
    """
    text_lower = text.lower()
    
    keywords = ["payment", "pay", "charged", "charge", "billing", "invoice", "credit card", "paypal", "transaction", "money was taken"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Shipping_and_Delivery(text: str) -> tuple[bool, float]:
    """Label Function: 物流/配送 -> P2/neutral
    
    触发关键词: "shipping", "shipment", "deliver", "delivery", "tracking", "track", "package", "parcel", "carrier", "fedex", "ups", "usps", "dhl", "arrived", "received", "estimated delivery"
    """
    text_lower = text.lower()
    
    keywords = ["shipping", "shipment", "deliver", "delivery", "tracking", "track", "package", "parcel", "carrier", "fedex", "ups", "usps", "dhl", "arrived", "received", "estimated delivery"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Delivery_Problem(text: str) -> tuple[bool, float]:
    """Label Function: 配送问题 -> P2/negative
    
    触发关键词: "not delivered", "missing package", "wrong address", "damaged package", "late delivery", "delayed", "lost package", "stolen", "not arrived", "hasn't arrived"
    """
    text_lower = text.lower()
    
    keywords = ["not delivered", "missing package", "wrong address", "damaged package", "late delivery", "delayed", "lost package", "stolen", "not arrived", "hasn't arrived"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Return_Request(text: str) -> tuple[bool, float]:
    """Label Function: 退货/换货 -> L1/negative
    
    触发关键词: "return", "returning", "return policy", "send back", "exchange", "swap", "wrong item", "not fit", "doesn't fit", "too small", "too large", "too big", "size"
    """
    text_lower = text.lower()
    
    keywords = ["return", "returning", "return policy", "send back", "exchange", "swap", "wrong item", "not fit", "doesn't fit", "too small", "too large", "too big", "size"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Refund_Request(text: str) -> tuple[bool, float]:
    """Label Function: 退款请求 -> L2/negative
    
    触发关键词: "refund", "money back", "get my money", "reimburse", "full refund", "partial refund", "chargeback"
    """
    text_lower = text.lower()
    
    keywords = ["refund", "money back", "get my money", "reimburse", "full refund", "partial refund", "chargeback"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Warranty_Claim(text: str) -> tuple[bool, float]:
    """Label Function: 质保/维修/换新 -> L2/negative
    
    触发关键词: "warranty", "defective", "broken", "not working", "stopped working", "malfunction", "repair", "replacement", "replace", "faulty", "damaged", "doesn't work"
    """
    text_lower = text.lower()
    
    keywords = ["warranty", "defective", "broken", "not working", "stopped working", "malfunction", "repair", "replacement", "replace", "faulty", "damaged", "doesn't work"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Customer_Service(text: str) -> tuple[bool, float]:
    """Label Function: 客服体验 -> L3/neutral
    
    触发关键词: "customer service", "support", "help", "assistance", "response", "reply", "contacted", "reach out", "chat", "ticket", "escalate", "supervisor", "manager"
    """
    text_lower = text.lower()
    
    keywords = ["customer service", "support", "help", "assistance", "response", "reply", "contacted", "reach out", "chat", "ticket", "escalate", "supervisor", "manager"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_Product_Inquiry(text: str) -> tuple[bool, float]:
    """Label Function: 产品咨询/使用指导 -> I/neutral
    
    触发关键词: "how to", "instructions", "manual", "guide", "setup", "install", "configure", "use", "using", "usage", "compatible", "fit", "work with", "does it work"
    """
    text_lower = text.lower()
    
    keywords = ["how to", "instructions", "manual", "guide", "setup", "install", "configure", "use", "using", "usage", "compatible", "fit", "work with", "does it work"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0

def lf_General_Feedback(text: str) -> tuple[bool, float]:
    """Label Function: 一般反馈/感谢 -> L3/positive
    
    触发关键词: "thank", "thanks", "appreciate", "grateful", "great service", "good experience", "satisfied", "disappointed", "frustrated", "unhappy"
    """
    text_lower = text.lower()
    
    keywords = ["thank", "thanks", "appreciate", "grateful", "great service", "good experience", "satisfied", "disappointed", "frustrated", "unhappy"]
    
    for kw in keywords:
        if kw in text_lower:
            return True, 0.75
    
    return False, 0.0


# ── 注册表 ────────────────────────────────────────────────────────

LABEL_FUNCTION_REGISTRY = {
    "Order_Placement": lf_Order_Placement,  # 下单/购买
    "Payment_Issue": lf_Payment_Issue,  # 支付/账单问题
    "Shipping_and_Delivery": lf_Shipping_and_Delivery,  # 物流/配送
    "Delivery_Problem": lf_Delivery_Problem,  # 配送问题
    "Return_Request": lf_Return_Request,  # 退货/换货
    "Refund_Request": lf_Refund_Request,  # 退款请求
    "Warranty_Claim": lf_Warranty_Claim,  # 质保/维修/换新
    "Customer_Service": lf_Customer_Service,  # 客服体验
    "Product_Inquiry": lf_Product_Inquiry,  # 产品咨询/使用指导
    "General_Feedback": lf_General_Feedback,  # 一般反馈/感谢
}


def apply_all(text: str) -> dict[str, tuple[bool, float]]:
    """对单条文本应用全部 Zendesk 标签函数"""
    results = {}
    for tag_name, lf in LABEL_FUNCTION_REGISTRY.items():
        matched, conf = lf(text)
        if matched:
            results[tag_name] = (matched, conf)
    return results
