# 验证报告：CausalRAG Skill 萃取

**论文**: CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation  
**arXiv**: 2503.19878 | ACL 2025 Findings  
**验证时间**: 2026-05-19  
**验证人**: Sisyphus-Junior (Claude Code)

---

## 1. 代码验证

### 测试运行结果

```
$ python3 paper2skills-code/08-知识图谱/causal_rag_2025/model.py

CausalRAG 模型自测
论文: CausalRAG - Integrating Causal Graphs into RAG
arXiv: 2503.19878 | ACL 2025 Findings

测试1: 因果知识图谱构建验证
  图谱节点数: 32
  图谱边数: 16
  根因节点数: 16
  叶子结果节点数: 16
✅ 测试1 通过

测试2: 因果路径追踪验证
  查询: 「机器停止运行闪红灯」
  找到因果链: 2 条
  最高置信度链:
    [root_cause] 底盘传感器被灰尘遮挡
    [symptom] 机器判断遇到悬崖，停止运行
  置信度: 62.5%
✅ 测试2 通过

测试3: 因果增强上下文生成
  查询: 「电池频繁回充」
  生成上下文长度: 360 字符
✅ 测试3 通过

测试4: 多故障场景查询测试（4 种故障类型）
✅ 测试4 通过

测试5: 因果图结构完整性验证
  边数: 16，悬空引用: 0 ✓
  正向/反向索引一致性: 通过 ✓
  因果强度范围: [0,1] ✓
✅ 测试5 通过

测试结果: 5/5 通过
🎉 所有测试通过！CausalRAG 模型验证成功
```

### 修复记录

**问题**: 初版使用 `re.findall(r'\w+', text)` 进行 token 级匹配，中文字符串无空格分隔导致整串变成单一 token，Jaccard 相似度始终为0，测试2和3失败。

**修复**: 改为**字符级 bigram Jaccard 相似度**，将中文文本按相邻2字符组合为特征集，正确计算中文语义相似度。修复后测试2从0/1变为找到2条有效因果链。

---

## 2. Skill Card 质量评估

依据 MasterPrompt.md 5维度评分标准（总分10分，≥7分合格）：

| 维度 | 权重 | 得分 | 说明 |
|------|------|------|------|
| ① 算法原理 | 25% | 9/10 | 非复制重述，含 bigram Jaccard 和因果链置信度数学公式，说明了"中文无空格"等关键假设 |
| ② 应用案例 | 25% | 9/10 | 两个具体场景（扫地机器人排障 + 母婴产品不良反应归因），含量化数据（退货率/月损失额），强相关母婴出海 |
| ③ 代码模板 | 25% | 9/10 | 完整可运行（5/5测试通过），含扩展 Embedding 替换示例，900行高质量代码 |
| ④ 技能关联 | 10% | 8/10 | 关联3个前置技能（GraphRAG/KG-Auto/KGQA），2个延伸方向，有逻辑图谱 |
| ⑤ 商业价值 | 15% | 9/10 | ROI 量化到具体金额（120-200万/年），引用论文实验数据（AF+35%），5星优先级有具体依据 |

**综合评分：9.0/10 ✅（超过7分合格线）**

---

## 3. 论文核心算法对齐验证

| 论文关键概念 | Skill Card 覆盖 | 代码实现 |
|-------------|----------------|----------|
| 因果节点抽取（Causal Node Extraction） | ✅ 场景一描述 | ✅ `CausalGraphBuilder.build_from_texts()` |
| 有向因果图（Causal Graph）| ✅ 算法原理 | ✅ `CausalGraph` + `CausalEdge` 数据结构 |
| 因果路径追踪（Causal Tracing）| ✅ 核心思想 | ✅ `CausalTracer.trace_backward()` BFS 逆向遍历 |
| 因果摘要生成（Causal Summary）| ✅ 三步流程 | ✅ `CausalRAG.generate_context()` |
| 逆向溯源（Backward Tracing）| ✅ 数学直觉 | ✅ `_effect_to_causes` 反向索引 |
| 置信度评分（Chain Confidence）| ✅ 公式推导 | ✅ `_compute_chain_confidence()` 几何均值+惩罚 |

---

## 4. 产出文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `paper2skills-code/08-知识图谱/causal_rag_2025/model.py` | ✅ 生成并验证 | 897行，5/5测试通过 |
| `paper2skills-vault/08-知识图谱/Skill-CausalRAG-Knowledge-Retrieval.md` | ✅ 生成 | 完整5模块 Skill Card |
| `paper2skills-vault/papers/08-知识图谱/causal_rag_2025/verification_report.md` | ✅ 本文件 | 验证报告 |

---

## 5. 已知局限与改进建议

1. **因果抽取基于规则**：当前使用正则匹配触发词，对于隐式因果（"A 是 B"类语句）无法识别。生产环境建议替换为 LLM 抽取（GPT-4/Claude 对因果三元组的精度可达 85%+）

2. **语义匹配使用 bigram**：bigram Jaccard 对长文本精度有限，建议替换为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 等中文 Embedding 模型

3. **因果强度默认 0.8**：实际应根据频次、工程师确认度标注真实因果强度，会显著提升链置信度的区分度

4. **扩展方向**：支持跨文档的因果链合并（同一现象在多份文档中有不同描述的节点合并去重）
