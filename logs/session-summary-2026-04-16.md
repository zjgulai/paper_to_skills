# Session Summary - 2026-04-16

## 会话目标

继续执行 **paper2skills Skill Graph 闭环补全计划**，完成 **A/B 实验设计基础**（AB Experimental Design Basics）技能的萃取与落地。

---

## 核心产出

### 1. 新增 Skill 卡片
- **文件**: `paper2skills-vault/02-A_B实验/Skill-AB-Experimental-Design.md`
- **内容**:
  - 算法原理：连续/二分类指标样本量计算、Relative Lift 的 Delta Method 修正（Zhou et al. 2023）、Power 与 MDE 互算、CUPED 方差缩减
  - 业务应用：Momcozy 落地页转化率优化实验设计、大促前实验可行性快速评估
  - 代码模板与运行方式
  - 技能关联：前置 Demand Forecasting / Uplift Modeling，延伸 MAB / Thompson Sampling / TJAP
  - 业务价值评估：⭐⭐⭐⭐⭐（5/5）

### 2. 代码模板修复与验证
- **文件**: `paper2skills-code/ab_testing/experimental_design/design.py`
- **修复项**:
  1. **`power_analysis` / `mde_calculator` 公式修正** — 移除了错误的 `n_eff = n_per_group * ratio / (1 + ratio)`，改为直接使用 `sigma_eff = std * np.sqrt(1 + 1/ratio)` 作为分母，确保回验 Power 精确等于 80.00%。
  2. **`ABTestDesigner` 二分类验证修正** — 将 `std = np.sqrt(baseline * (1 - baseline))` 改为使用 pooled proportion `p_pool` 计算 `std_pool`，与 `sample_size_binary` 保持一致。
  3. **`stratified_allocation` 索引错位修复** — 原实现通过 `assignments.extend()` 按 groupby 顺序填充，但 DataFrame 分组顺序与原始索引不一致，导致某些 stratum 出现 100% C 或 100% T。修复为使用 `result.loc[group.index, "assignment"] = ...` 按索引精确赋值，各分层内比例恢复至 ~50/50。
  4. **演示数据量调整** — 分层演示的 `n_users` 从 5000 提升至 20000，保证 32 个 stratum 内有足够样本支撑 1:1 随机化。

### 3. 同步状态更新
- **文件**: `paper2skills-vault/07-资源库/sync_status.json`
- **变更**: 新增 `Skill-AB-Experimental-Design.md` 的 vault/github 同步记录（`code_path`: `paper2skills-code/ab_testing/experimental_design/`）

### 4. Skill 关联图谱更新
- **文件**: `paper2skills-vault/00-项目管理/Skill关联图谱.md`
- **变更**:
  - 02-A/B实验 领域技能数 2 → 3
  - 链路 4「实验与优化」从「薄弱/缺失」更新为「已补齐基础」，在图中加入 `A/B Exp Design` 节点
  - 技能缺口热力图：02-A/B实验 总计缺口从 **8** 降至 **4**
  - 跨领域组合推荐新增「实验严谨加速（A/B Exp Design + CUPED）」和「在线实验优化（A/B Exp Design + Thompson Sampling）」
  - P1 优先级清单中标记 A/B 实验设计基础为已完成

---

## 运行验证结果

```bash
cd paper2skills-code/ab_testing/experimental_design
python3 design.py
```

关键输出确认：
- 场景 1 回验 Power: **80.00%** ✅
- 场景 2 分层分配前 6 组比例: 0.491/0.509 ~ 0.518/0.482 ✅
- 场景 3 CUPED 方差缩减比例: **29.3%**（等效样本量缩减至 1.4 倍）✅

---

## 任务状态

- [x] 萃取 A/B 实验设计基础技能（Task #8 已完成）

---

## 待办 / 下一步

基于计划，剩余关键缺口：
1. **iReFeed 需求优先级排序** — VOC→产品决策链断点，需尽快补齐
2. **Learning to Rank** — 05-推荐系统缺口，REVISION 已有搜索意图但缺少排序决策层
3. 继续重建 `知识图谱架构与分类体系.md`（尚未更新）

---

## Git 状态快照

```
M paper2skills-vault/07-资源库/sync_status.json
M paper2skills-vault/00-项目管理/Skill关联图谱.md
?? logs/session-summary-2026-04-16.md
?? paper2skills-vault/02-A_B实验/Skill-AB-Experimental-Design.md
```

（未要求提交，仅记录状态）
