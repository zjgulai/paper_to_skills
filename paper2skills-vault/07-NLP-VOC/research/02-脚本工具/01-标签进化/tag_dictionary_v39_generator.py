"""标签字典 v3.9 生成器

将 Phase 3 P2 的 10 个 Zendesk Service 标签（TAG_SRV_01~10）同步到 v3.8 字典，生成 v3.9。

v3.8 → v3.9 变更：
- 新增 10 个 TAG_SRV 标签行（插在 BRAND 标签之前）
- 通用标签主表：255 → 265 行
- 所有新增标签字段补全（36 列）
"""

import json
from pathlib import Path

import pandas as pd

# ── 路径 ───────────────────────────────────────────────
BASE = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本")
V38_PATH = BASE / "tag_dictionary_v3.8.xlsx"
V39_PATH = BASE / "tag_dictionary_v3.9.xlsx"

SRV_JSON = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/08-辅助数据/zendesk_service_labels.json")

# ── 读取 TAG_SRV 定义 ─────────────────────────────────
with open(SRV_JSON) as f:
    SRV_TAGS = json.load(f)

# ── 标签命中次数（从 phase4_labeled.jsonl 统计）───────
SRV_HIT_COUNTS = {
    "TAG_SRV_01": 6178,
    "TAG_SRV_02": 3252,
    "TAG_SRV_03": 6533,
    "TAG_SRV_04": 290,
    "TAG_SRV_05": 2645,
    "TAG_SRV_06": 1534,
    "TAG_SRV_07": 5671,
    "TAG_SRV_08": 3533,
    "TAG_SRV_09": 9090,
    "TAG_SRV_10": 2592,
}

ZENDESK_TOTAL = 47204


# ── 字段推导规则 ─────────────────────────────────────

def derive_fields(tag: dict) -> dict:
    """根据 TAG_SRV 标签定义推导 36 个字典字段"""
    tag_id = tag["tag_id"]
    tag_en = tag["tag_en"]
    tag_cn = tag["tag_cn"]
    aipl = tag["aipl"]
    sentiment = tag["sentiment"]
    keywords = tag["keywords"]
    hit_count = SRV_HIT_COUNTS.get(tag_id, 0)
    coverage_in_zendesk = hit_count / ZENDESK_TOTAL * 100

    # 关键词字符串
    kw_str = "; ".join(keywords)

    # ── 标签主题推导 ──
    theme_map = {
        "Order_Placement": "订单履约",
        "Payment_Issue": "订单履约",
        "Shipping_and_Delivery": "物流时效",
        "Delivery_Problem": "物流时效",
        "Return_Request": "客服售后",
        "Refund_Request": "客服售后",
        "Warranty_Claim": "客服售后",
        "Customer_Service": "客服售后",
        "Product_Inquiry": "产品咨询",
        "General_Feedback": "用户满意度",
    }
    theme = theme_map.get(tag_en, "客服售后")

    # ── 主责部门推导 ──
    dept_map = {
        "Order_Placement": "电商运营部",
        "Payment_Issue": "电商运营部",
        "Shipping_and_Delivery": "供应链中心",
        "Delivery_Problem": "供应链中心",
        "Return_Request": "全球客服与体验中心",
        "Refund_Request": "全球客服与体验中心",
        "Warranty_Claim": "全球客服与体验中心",
        "Customer_Service": "全球客服与体验中心",
        "Product_Inquiry": "全球客服与体验中心",
        "General_Feedback": "全球客服与体验中心",
    }
    main_dept = dept_map.get(tag_en, "全球客服与体验中心")

    # ── 协同部门推导 ──
    collab_map = {
        "Order_Placement": "物流运营部",
        "Payment_Issue": "财务部门",
        "Shipping_and_Delivery": "物流运营部",
        "Delivery_Problem": "物流运营部",
        "Return_Request": "电商运营部; 供应链中心",
        "Refund_Request": "电商运营部; 财务部门",
        "Warranty_Claim": "品控部; 产品中心/品线",
        "Customer_Service": "品控部",
        "Product_Inquiry": "产品中心/品线; 电商运营部",
        "General_Feedback": "品牌市场中心; 产品中心/品线",
    }
    collab_dept = collab_map.get(tag_en, "")

    # ── 原子指标推导 ──
    metric_map = {
        "Order_Placement": "order_placement_rate",
        "Payment_Issue": "payment_issue_rate",
        "Shipping_and_Delivery": "shipping_inquiry_rate",
        "Delivery_Problem": "delivery_problem_rate",
        "Return_Request": "return_request_rate",
        "Refund_Request": "refund_request_rate",
        "Warranty_Claim": "warranty_claim_rate",
        "Customer_Service": "customer_service_contact_rate",
        "Product_Inquiry": "product_inquiry_rate",
        "General_Feedback": "general_feedback_rate",
    }
    metric = metric_map.get(tag_en, f"{tag_en.lower()}_rate")

    # ── MetricDirection ──
    metric_dir = "negative" if sentiment == "negative" else "positive"

    # ── Proxy NPS 贡献 ──
    nps_map = {
        "positive": "Promoter驱动",
        "negative": "Detractor驱动",
        "neutral": "中性-Passive",
    }
    nps_contrib = nps_map.get(sentiment, "中性-Passive")

    # ── 策略包推导 ──
    strategy_map = {
        "Order_Placement": "订单流程优化包",
        "Payment_Issue": "支付体验优化包",
        "Shipping_and_Delivery": "物流时效提升包",
        "Delivery_Problem": "末端配送治理包",
        "Return_Request": "退货流程优化包",
        "Refund_Request": "退款时效优化包",
        "Warranty_Claim": "质保服务优化包",
        "Customer_Service": "客服响应优化包",
        "Product_Inquiry": "产品信息完善包",
        "General_Feedback": "用户满意度提升包",
    }
    strategy = strategy_map.get(tag_en, "客服售后优化包")

    # ── 故事线关联推导 ──
    story_map = {
        "Order_Placement": "订单转化效率",
        "Payment_Issue": "支付成功率",
        "Shipping_and_Delivery": "物流履约效率",
        "Delivery_Problem": "末端配送体验",
        "Return_Request": "退货处理效率",
        "Refund_Request": "退款处理时效",
        "Warranty_Claim": "质保服务体验",
        "Customer_Service": "客服响应质量",
        "Product_Inquiry": "产品信息触达率",
        "General_Feedback": "用户满意度闭环",
    }
    story = story_map.get(tag_en, "客服售后体验")

    # ── 业务动作/责任部门 ──
    action_map = {
        "Order_Placement": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Payment_Issue": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Shipping_and_Delivery": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Delivery_Problem": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Return_Request": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Refund_Request": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Warranty_Claim": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Customer_Service": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "Product_Inquiry": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
        "General_Feedback": f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环",
    }
    action = action_map.get(tag_en, f"{main_dept}：围绕「{tag_cn}」主题做专项优化和闭环")

    # ── 标签定义 ──
    definition_map = {
        "Order_Placement": "用户在Zendesk工单中表达下单/购买/购物车相关意图",
        "Payment_Issue": "用户在Zendesk工单中表达支付/账单/扣款相关问题",
        "Shipping_and_Delivery": "用户在Zendesk工单中查询物流/配送/追踪信息",
        "Delivery_Problem": "用户在Zendesk工单中反馈配送延迟/丢件/破损等问题",
        "Return_Request": "用户在Zendesk工单中表达退货/换货/尺码不合适等意图",
        "Refund_Request": "用户在Zendesk工单中申请退款/退钱/chargeback",
        "Warranty_Claim": "用户在Zendesk工单中提出质保/维修/换新/产品故障等诉求",
        "Customer_Service": "用户在Zendesk工单中提及客服/支持/回复/升级等体验",
        "Product_Inquiry": "用户在Zendesk工单中咨询产品使用/安装/兼容性等问题",
        "General_Feedback": "用户在Zendesk工单中表达感谢/满意/失望/沮丧等一般反馈",
    }
    definition = definition_map.get(tag_en, f"用户在Zendesk工单中表达{tag_cn}相关意图")

    # ── 风险等级推导 ──
    risk_map = {
        "Order_Placement": "低风险",
        "Payment_Issue": "中风险",
        "Shipping_and_Delivery": "低风险",
        "Delivery_Problem": "中风险",
        "Return_Request": "中风险",
        "Refund_Request": "中风险",
        "Warranty_Claim": "中风险",
        "Customer_Service": "低风险",
        "Product_Inquiry": "低风险",
        "General_Feedback": "低风险",
    }
    risk = risk_map.get(tag_en, "中风险")

    # ── 问题诊断 ──
    if coverage_in_zendesk > 15:
        diagnosis = f"Zendesk内覆盖率高({coverage_in_zendesk:.1f}%)；需验证是否过度召回"
    elif coverage_in_zendesk < 1:
        diagnosis = f"Zendesk内覆盖率低({coverage_in_zendesk:.1f}%)；关键词可能过于严格"
    else:
        diagnosis = f"Zendesk内覆盖率适中({coverage_in_zendesk:.1f}%)"

    # ── 优化建议 ──
    if coverage_in_zendesk > 15:
        opt = f"验证关键词是否过于宽泛，考虑增加区分度 | 检查与相似标签的重叠 | 补充否定词检测"
    elif coverage_in_zendesk < 1:
        opt = f"扩展关键词覆盖更多同义表达 | 检查是否有遗漏的常用短语 | 考虑放宽匹配条件"
    else:
        opt = f"保持当前关键词覆盖 | 定期基于新工单扩展同义词 | 监控覆盖率变化趋势"

    # ── 合理性评分 ──
    # 基于命中率和覆盖率的综合评分
    # 命中率高且覆盖率适中 → 高评分
    # 参考已有标签范围 39.5-95.0，均值 66.5
    base_score = 60.0
    hit_bonus = min(hit_count / 1000, 15)  # 命中越多加分越多，上限 15
    coverage_penalty = 0
    if coverage_in_zendesk > 20:
        coverage_penalty = 5  # 过高覆盖率可能过度召回
    elif coverage_in_zendesk < 1:
        coverage_penalty = 3  # 过低覆盖率可能遗漏

    score = base_score + hit_bonus - coverage_penalty
    score = min(max(score, 45.0), 85.0)

    # ── 优先级 ──
    if sentiment == "negative" and coverage_in_zendesk > 5:
        priority = "P0"
    elif coverage_in_zendesk > 10:
        priority = "P1"
    else:
        priority = "P2"

    # ── 优化优先级 ──
    opt_priority = "P1" if coverage_in_zendesk < 1 or coverage_in_zendesk > 20 else "P2"

    # ── 适用VOC载体 ──
    voc_carrier = "Zendesk工单"

    # ── 适用产品品线 ──
    product_line = "通用"

    # ── 适用用户画像 ──
    persona = "通用"

    # ── 是否通用标签 ──
    is_general = "是"

    # ── 品类特异性指数 ──
    specificity = 0.0

    # ── 共性/特性分类 ──
    common_type = "强共性标签"

    # ── 主导品类 ──
    dominant_category = "通用"

    # ── v3.6_AIPL动态规则 ──
    aipl_rule = f"消费者旅程中自然触发，关键词匹配:{tag_en}"

    # ── v3.6_安全等级 ──
    safety = "低-常规监控"

    # ── 备注 ──
    note = "来源：Zendesk Service Label Functions (Phase 3 P2)，v3.9补齐"

    # ── 审核状态 ──
    audit_status = "已通过"

    return {
        "标签ID": tag_id,
        "AIPL节点": aipl,
        "标签主题": theme,
        "VOC标签（中文）": tag_cn,
        "VOC标签（英文）": tag_en,
        "英文关键词/典型表达": kw_str,
        "消费者习惯关键词/原话短语": kw_str,
        "标签定义": definition,
        "情感极性": "负向" if sentiment == "negative" else ("正向" if sentiment == "positive" else "中性"),
        "是否AI可抽取": "是",
        "来源类型": "补齐",
        "适用产品品线": product_line,
        "适用VOC载体": voc_carrier,
        "适用用户画像": persona,
        "对应原子指标": metric,
        "MetricDirection": metric_dir,
        "Proxy NPS贡献": nps_contrib,
        "是否通用标签": is_general,
        "故事线关联": story,
        "策略包": strategy,
        "业务动作/责任部门": action,
        "主责部门": main_dept,
        "协同部门": collab_dept,
        "默认优先级": priority,
        "备注": note,
        "合理性评分": round(score, 1),
        "风险等级": risk,
        "问题诊断": diagnosis,
        "品类特异性指数": specificity,
        "共性/特性分类": common_type,
        "主导品类": dominant_category,
        "优化建议": opt,
        "优化优先级": opt_priority,
        "审核状态": audit_status,
        "v3.6_AIPL动态规则": aipl_rule,
        "v3.6_安全等级": safety,
    }


# ── 主流程 ───────────────────────────────────────────

def main():
    # 读取 v3.8
    xls = pd.ExcelFile(V38_PATH)
    sheet_names = xls.sheet_names
    print(f"Sheets: {sheet_names}")

    # 读取所有 sheet
    sheets = {}
    for name in sheet_names:
        sheets[name] = pd.read_excel(xls, sheet_name=name)
        print(f"  {name}: {len(sheets[name])} rows")

    df_main = sheets["01_通用标签主表"]
    print(f"\nOriginal main sheet: {len(df_main)} rows")

    # 生成 10 个 TAG_SRV 行
    new_rows = []
    for tag in SRV_TAGS:
        row = derive_fields(tag)
        new_rows.append(row)
        print(f"  {tag['tag_id']}: {tag['tag_cn']} | AIPL={tag['aipl']} | sentiment={tag['sentiment']} | score={row['合理性评分']}")

    df_new = pd.DataFrame(new_rows)

    # 插入位置：在 BRAND 标签之前（按 tag_id 字母顺序，TAG_SRV < BRAND）
    brand_mask = df_main["标签ID"].str.startswith("BRAND", na=False)
    brand_start = brand_mask.idxmax() if brand_mask.any() else len(df_main)

    # 分段拼接：前面部分 + 新行 + BRAND 及后面部分
    df_before = df_main.iloc[:brand_start].copy()
    df_after = df_main.iloc[brand_start:].copy()
    df_main_v39 = pd.concat([df_before, df_new, df_after], ignore_index=True)

    print(f"\nNew main sheet: {len(df_main_v39)} rows ({len(df_before)} + {len(df_new)} + {len(df_after)})")

    # 更新 sheets
    sheets["01_通用标签主表"] = df_main_v39

    # 更新 08_映射关系表：为每个新标签添加映射行
    if "08_映射关系表" in sheets:
        df_map = sheets["08_映射关系表"]
        print(f"\nOriginal mapping table: {len(df_map)} rows")

        # 获取映射表的列名
        map_cols = list(df_map.columns)
        print(f"Mapping columns: {map_cols}")

        # 为每个 TAG_SRV 标签创建映射行
        map_rows = []
        next_map_id = len(df_map) + 1
        for tag in SRV_TAGS:
            row = derive_fields(tag)
            map_row = {col: None for col in map_cols}
            map_row["映射ID"] = f"MAP_{next_map_id:04d}"
            map_row["VOC标签（中文）"] = row["VOC标签（中文）"]
            map_row["VOC标签（英文）"] = tag["tag_en"]
            map_row["标签主题"] = row["标签主题"]
            map_row["AIPL节点"] = row["AIPL节点"]
            map_row["对应原子指标"] = row["对应原子指标"]
            map_row["MetricDirection"] = row["MetricDirection"]
            map_row["情感极性"] = row["情感极性"]
            map_row["sentiment"] = tag["sentiment"]
            map_row["risk_type"] = row["标签主题"]
            next_map_id += 1
            map_rows.append(map_row)

        df_map_new = pd.concat([df_map, pd.DataFrame(map_rows)], ignore_index=True)
        sheets["08_映射关系表"] = df_map_new
        print(f"New mapping table: {len(df_map_new)} rows (+{len(map_rows)})")

    # 更新 09_存量标签归档：添加说明
    if "09_存量标签归档" in sheets:
        df_archive = sheets["09_存量标签归档"]
        print(f"\nArchive sheet: {len(df_archive)} rows (unchanged)")

    # 保存 v3.9
    print(f"\nSaving to {V39_PATH}...")
    with pd.ExcelWriter(V39_PATH, engine="openpyxl") as writer:
        for name in sheet_names:
            sheets[name].to_excel(writer, sheet_name=name, index=False)
            print(f"  Written: {name} ({len(sheets[name])} rows)")

    print(f"\nDone! v3.9 saved: {V39_PATH}")

    # 验证
    xls_v39 = pd.ExcelFile(V39_PATH)
    df_verify = pd.read_excel(xls_v39, sheet_name="01_通用标签主表")
    srv_mask = df_verify["标签ID"].str.startswith("TAG_SRV", na=False)
    print(f"\nVerification:")
    print(f"  Total rows: {len(df_verify)}")
    print(f"  TAG_SRV tags: {srv_mask.sum()}")
    print(f"  BRAND tags: {df_verify['标签ID'].str.startswith('BRAND', na=False).sum()}")
    print(f"  TAG_GEN tags: {df_verify['标签ID'].str.startswith('TAG_GEN', na=False).sum()}")
    print(f"  TAG_ZEN tags: {df_verify['标签ID'].str.startswith('TAG_ZEN', na=False).sum()}")
    print(f"  TAG_DEF tags: {df_verify['标签ID'].str.startswith('TAG_DEF', na=False).sum()}")

    # 显示新增 TAG_SRV 标签的合理性评分
    print(f"\nTAG_SRV 合理性评分分布:")
    for _, row in df_verify[srv_mask].iterrows():
        print(f"  {row['标签ID']}: {row['VOC标签（中文）']:<12} | 评分={row['合理性评分']} | 主责={row['主责部门']}")


if __name__ == "__main__":
    main()
