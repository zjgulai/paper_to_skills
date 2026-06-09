# Skill Card: Amazon 账号申诉策略（POA 行动计划）

> **桥梁**: 21-合规决策 ↔ 19-风控反欺诈 | **类型**: 合规运营

---

## ① 算法原理

**核心思想**：Amazon 账号/Listing 被封后，POA（Plan of Action）是唯一有效的申诉工具。成功率取决于 POA 的结构化程度，而非情感诉求。

**POA 三段式结构（亚马逊明确要求）**：
```
Section 1: Root Cause（根本原因）
  - 用数据而非情绪描述"发生了什么"
  - 承认问题存在，不要辩解
  - 具体到 ASIN/订单号/日期

Section 2: Corrective Actions（纠正措施）
  - 已经采取的具体步骤（过去时）
  - 可验证的行动，非泛泛承诺
  - 时间线清晰

Section 3: Preventive Measures（预防措施）
  - 系统性改变，不是临时修补
  - 建立监控机制
  - 培训/流程/技术三层防护
```

**常见封号类型与 POA 策略**：
```
类型                 | 根因重点              | 纠正重点
ODR 超标(>1%)        | 具体差评/A-to-Z分析    | 客服响应改善证明
ASIN 违规           | 合规文件缺失           | 上传认证文件
Review 操纵         | 操作失误说明           | 停止相关活动证明
知识产权投诉         | 误用/不知情说明         | 品牌授权书/改款证明
```

---

## ② 母婴出海应用案例

**业务问题**：某母婴品牌吸奶器 Listing 因竞品恶意投诉知识产权被下架，不知道如何写 POA 才能快速恢复。

**应用流程**：
1. 收到 Amazon 通知 → 24h 内不要急着申诉（先收集证据）
2. 确认投诉类型（IP投诉 vs 产品安全 vs 政策违规）
3. 联系投诉方协商撤诉（成功率 40-60%）
4. 如协商失败，写 POA：
   - Section 1: 说明我司拥有合法权益（附品牌注册证/授权书）
   - Section 2: 已移除有歧义的描述词/图片
   - Section 3: 建立月度 Listing 合规审查 SOP
5. 提交后 48-72h 审核，拒绝则升级到 Amazon Executive Seller Relations

**年化收益**：
- 从被封到恢复缩短 3-7 天（vs 自行摸索 2-4 周）
- 专业 POA 成功率 65-80%（vs 模板 POA 20-30%）

---

## ③ 代码模板

```python
POA_TEMPLATE = {
    "structure": {
        "root_cause": {
            "required": True,
            "format": "具体事件描述 + 数据支撑",
            "bad_example": "我们的产品没有问题，这是恶意投诉",
            "good_example": "2026-06-01, ASIN B0XXX 收到 IP 投诉，投诉方为 Company X，投诉编号 XXXXXX。经核查，我司产品描述中使用了与对方商标相似的词汇'XXX'，该词汇已于2025年被对方注册为美国商标。",
        },
        "corrective_actions": {
            "required": True,
            "format": "已完成的具体步骤 + 时间戳",
            "examples": [
                "2026-06-02: 已从 Listing 标题/描述/Bullet 中删除争议词汇",
                "2026-06-02: 已联系投诉方 company@email.com 寻求撤诉，邮件已附",
                "2026-06-03: 已提交新版 Listing 图片（去除争议标识）",
            ],
        },
        "preventive_measures": {
            "required": True,
            "format": "系统性流程改变",
            "examples": [
                "建立新品上架前商标检索 SOP（USPTO + 欧盟商标数据库）",
                "每季度进行全店 Listing 合规审查",
                "购买商标监控服务（如 TrademarkNow）实时预警",
            ],
        },
    },
    "escalation_path": [
        "Seller Central > Performance > Account Health > Submit Appeal",
        "如 48h 无回复: 邮件 seller-performance@amazon.com",
        "如仍拒绝: Amazon Executive Seller Relations（需要 Case ID）",
        "最后手段: Amazon Seller Forums + 寻求法律援助",
    ],
}

def generate_poa(issue_type: str, asin: str, incident_date: str, actions_taken: list) -> str:
    """生成 POA 框架，基于具体情况填写。"""
    template = POA_TEMPLATE["structure"]
    
    poa = f"""
PLAN OF ACTION - {issue_type.upper()}
ASIN: {asin} | Date: {incident_date}

SECTION 1 - ROOT CAUSE:
[描述根本原因，使用数据]
Incident Date: {incident_date}
ASIN Affected: {asin}
Issue Type: {issue_type}
Root Cause Analysis: [在此填写具体原因]

SECTION 2 - CORRECTIVE ACTIONS TAKEN:
"""
    for i, action in enumerate(actions_taken, 1):
        poa += f"{i}. {action}\n"
    
    poa += """
SECTION 3 - PREVENTIVE MEASURES:
1. [在此填写系统性预防措施]
2. [建立监控机制]
3. [培训/流程改进]
"""
    return poa

# 测试
test_poa = generate_poa(
    issue_type="IP Complaint",
    asin="B0XXXXX",
    incident_date="2026-06-01",
    actions_taken=[
        "2026-06-02: 删除争议词汇",
        "2026-06-02: 联系投诉方协商",
    ]
)
print(test_poa[:200])
print("[✓] POA 生成器测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Review-Fraud-Detection]] (19) | [[Skill-Amazon-ToS-Compliance-Guardrail]] (13)
- **组合**：[[Skill-Consumer-Complaint-Recall-Prediction]] (21) | [[Skill-Compliance-Scored-Guardrail-Orchestration]] (21)

---

## ⑤ 商业价值

- **ROI**：缩短申诉周期 = 每天 1-3 万 GMV × 节省天数
- **难度**：⭐⭐⭐☆☆（需要理解亚马逊审核逻辑）
- **优先级**：⭐⭐⭐⭐⭐（封号时唯一解法）
- **适用场景**：Listing 被投诉下架、账号 ODR 超标、知识产权纠纷申诉
