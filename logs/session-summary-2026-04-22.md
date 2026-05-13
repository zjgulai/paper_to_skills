# Session Summary: AdaNEN 流式分类器技能萃取

**日期**: 2026-04-22  
**主题**: AdaNEN (Adaptive Neural Ensemble Network) 流式 VOC 分类器技能萃取  
**技能总数**: 43 → 44 (NLP-VOC: 22 → 23)

---

## 背景

执行 plan `voc-cozy-comet.md` Phase 2 深度论文挖掘。基于 ACM TKDD 2024 论文 *A Novel Neural Ensemble Architecture for On-the-fly Classification of Evolving Text Streams* (Daniele Falcao 等, Amazon)，将 AdaNEN 萃取为独立 Skill。

---

## 核心交付物

### 1. 技能卡片
- **路径**: `paper2skills-vault/07-NLP-VOC/Skill-AdaNEN-Streaming-Classifier.md`
- **结构**: 5 模块完整卡片（算法原理 → 母婴应用 → 代码模板 → 技能关联 → ROI评估）
- **关键洞察**: InsightNet 负责"已知标签静态分类"，AdaNEN 负责"分布漂移自适应"，OpenCML 负责"新类别发现"——三者形成三层防御

### 2. 代码模板
- **路径**: `paper2skills-code/nlp_voc/streaming_voc_classifier/streaming_classifier.py`
- **规模**: ~540 行，7 个类
- **核心类**:
  - `PrototypeClassifier` — 类内均值原型 + 最近原型分类
  - `DriftDetector` — 基于样本到原型平均距离的相对变化率检测漂移
  - `EnsembleClassifier` — 动态权重集成（`weight = accuracy × decay^(age/100)`）
  - `AdaNENClassifier` — 流式入口，滑动窗口处理

### 3. 图谱更新（3处）

| 文件 | 变更 |
|------|------|
| `Skill关联图谱.md` v2.3→v2.4 | AdaNEN 插入 VOC 链路图；新增3个跨领域组合（流式分类自适应、开放世界流式分类、实时情绪监控） |
| `VOC决策智能桥接算法-完整图谱.md` | 基础VOC层添加 AdaNEN 流式漂移检测；技能萃取清单添加 P6 |
| `VOC用户画像萃取体系-完整图谱.md` | ABSA 下方添加 AdaNEN 分布监控模块，联动检测情感分布漂移 |

### 4. 调研报告更新
- **路径**: `research/autotag-paper-survey-phase2.md`
- AdaNEN 条目标记为"已萃取"，添加产出链接

---

## Bug 修复记录

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 漂移检测始终返回 0.000 | 复合 bug：① 用 new_clf 检测（距离天然小）② min_window_size=20 > val集样本数(6) ③ update_reference 用 X_val（样本量小） | ① 改用 latest_clf ② 动态 min_window_size = max(5, int(window_size × val_ratio / 2)) ③ update_reference 改用 X_train |
| 2 | 窗口7-9假阳性漂移 | prune 后 `members[-1]` 不是最新分类器，导致用旧分类器检测 | `max(members, key=lambda m: m.birth_time)` |
| 3 | 新成员未经验证即被 prune | `_prune_members` 在 `update_weights` 之前调用 | 提取 `prune_if_needed()`，在权重更新后调用 |
| 4 | 空原型崩溃 | `predict()` 在 prototypes 为空时 `min()` 抛异常 | 添加空检查并抛出 `RuntimeError` |

---

## 自我审计结果

| 审计项 | 结果 |
|--------|------|
| 代码质量 | ✅ 3场景测试通过（突变/渐进/无漂移） |
| 漂移检测准确性 | ✅ 突变场景正确识别窗口6、11；无漂移基线误报率 8.3% |
| 图谱一致性 | ✅ 4处引用全部对齐 |
| Skill卡片完整性 | ✅ 5模块结构，代码路径引用正确 |

---

## 后续建议

1. **InsightNet 稳定后接入**：AdaNEN 依赖高质量 embedding 输入，InsightNet 产出可直接作为 feature_dim 输入
2. **漂移阈值调参工具**：当前阈值 0.25 (abrupt) / 0.2 (gradual) 为经验值，建议后续提供基于历史数据的自动阈值选择
3. **周期性漂移场景**：当前仅验证 abrupt/gradual，季节性 periodic 漂移需保留更多历史分类器（提高 max_members）
