# 21-合规决策

**领域代码**: `compliance`  
**创建日期**: 2026-05-25  
**触发来源**: Sprint 4 业务完整性审计，识别 WF-D 选品扫描最大盲区之一

## 领域定位

与 `13-广告分析` 中 `Skill-Amazon-ToS-Compliance-Guardrail`（运营期合规）**互补**，本领域聚焦**进入前的品类级合规预筛**：

| 阶段 | 归属领域 | 代表 Skill |
|---|---|---|
| 选品前（进入决策） | `21-合规决策` ← **本领域** | Category-Compliance-Prescan |
| 运营期（内容合规） | `13-广告分析` | Amazon-ToS-Compliance-Guardrail |

## 核心主题

- 品类级合规风险预筛（FDA/CPSC/CE/REACH）
- 历史召回率分析与风险等级映射
- 合规成本估算（认证费用 × 时间 × 通过率）
- 合规护城河策略（高合规门槛 = 竞争壁垒）

## 业务背景

baby sterilizer 案例验证：
- UV-C 消毒器品类有 FDA Class 2 大规模召回历史（BigTree 2025-08 / Uvlizer 2026-04）
- 高合规门槛既是陷阱（进入成本高）也是护城河（竞争者少）
- 进入决策前必须量化合规风险等级和成本，否则 WF-D 选品分析缺失关键维度

## 当前 Skill 清单

> Sprint 4 待萃取（2026-05-25 起）

| Skill | 状态 | 论文关键词 |
|---|---|---|
| Skill-Category-Compliance-Prescan | ⬜ 待萃取 | `product safety compliance FDA recall prediction 2024` |
