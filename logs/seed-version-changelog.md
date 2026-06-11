---
doc_type: seed-version-changelog
description: 智能体报告页 SEED_REPORTS 版本变更记录。每次修改 SEED_REPORTS 内容后必须更新此文件并同步更新 build_playbook.py 中的 SEED_VERSION 常量。
---

# 智能体报告种子数据版本记录

## 版本命名规则

`v{YYYYMMDD}-r{迭代号}`

- 日期：种子数据生成/更新的日期
- 迭代号：同一日期内的第N次迭代（r1、r2...）

## 版本历史

### v20260611-r2（当前）

**生效文件**：`build_playbook.py` → `SEED_VERSION = 'v20260611-r2'`  
**种子数据条数**：24条（每个 Agent 各2条）  
**更新内容**：
- Round 1（12条）：手工构造，真实业务参数
  - 供应链哨兵：340件库存/28日销/21天周期/Amazon FBA
  - 动态定价顾问：$19.99/$7.80成本/$15-22竞品/BSR#234
  - P&L透视镜：月销$32400/COGS$9200/FBA$5800/广告$6500/退货4%
  - 广告归因侦探：Amazon SP/$12400/ACoS 18%
  - 竞品雷达站：3个ASIN/7天/全维度监控
  - Listing医生：89字符标题/硅胶碗/silicone baby plate
  - VOC解码器：147条英文评论/1-3星52条
  - 客服分诊台：63条Amazon工单/24小时SLA
  - 账号风险卫士：黄色预警+warning邮件
  - 品牌合规卫士：含clinically proven/prevents colic文案
  - 选品雷达：硅胶婴儿餐具/US/$5-20k
  - TikTok内容官：硅胶餐具套装/痛点反转/3条/周

- Round 2（12条）：Python 精确复现计算引擎后执行的真实输出
  - 供应链哨兵：1250件库存/65日销/35天周期/FBA+海外仓混合
  - 动态定价顾问：$34.99/$12.50成本/$28-42竞品/BSR#87
  - P&L透视镜：月销$58600/低退货2.8%/广告$8800
  - 广告归因侦探：TikTok Ads/$9500/ROAS 4x
  - 竞品雷达站：4个ASIN/14天/价格+BSR
  - Listing医生：186字符优质标题+5条详细Bullet
  - VOC解码器：15条真实英文评论（B0CXYZ1234）
  - 客服分诊台：10条高压工单（含A-to-Z威胁+安全投诉）
  - 账号风险卫士：review pattern警告+2个ASIN排查
  - 品牌合规卫士：含clinically tested/hypoallergenic/pediatrician
  - 选品雷达：婴儿安全防护角/DE/>$20k
  - TikTok内容官：BLW辅食场景/教程风格/5条/周

**修复 Bug**：
- 修复 localStorage 旧版 `'[]'` 导致种子数据不展示
- 修复 `exportReports` JS 语法错误（f-string `\n` 问题）

---

### v20260611-r1（已废弃）

**种子数据条数**：12条（每个 Agent 各1条）  
**废弃原因**：`loadReports()` 逻辑 bug 导致空 localStorage 时不注入种子数据

---

## 操作规范

### 何时必须升级版本

1. 修改了 `SEED_REPORTS` 中任何一条数据的内容、输入参数或结果
2. 新增或删除了种子数据条目
3. 修改了 `renderReports()` 的渲染逻辑（影响展示效果）

### 升级步骤

1. 修改 `build_playbook.py` 中 `SEED_REPORTS` 的内容
2. 更新 `SEED_VERSION = 'v{YYYYMMDD}-r{N}'`
3. 在本文件顶部新增版本条目，描述变更内容
4. git commit 时注明 `chore(seed): bump SEED_VERSION to v...`

### 不需要升级版本的情况

- 只修改了 UI 样式（颜色、间距等）
- 只修改了 `exportReports()`、`deleteReport()` 等工具函数
- 只修改了 `report-filter-bar` 的筛选逻辑（不影响数据）

---

*最后更新：2026-06-11*
