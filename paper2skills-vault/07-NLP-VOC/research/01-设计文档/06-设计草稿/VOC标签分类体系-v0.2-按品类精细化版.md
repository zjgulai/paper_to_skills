# VOC 标签分类体系 v0.2 - 按品类精细化版

> 适用场景：母婴非食品类跨境电商 VOC 智能洞察
> 数据基础：431 SKU，4 大 VOC 品线，6 大市场（美/西/加/德/英/法）
> 升级焦点：AIPL 锚定 + 渠道权重 + 画像推导（三大优先级）
> 关联文档：
> - [v0.1 基础版](VOC标签分类体系-v0.1.md)
> - [AIPL 缺口分析](VOC标签分类体系-AIPL缺口分析与v0.2升级方案.md)

---

## 执行摘要

本文档基于 **431 个 SKU** 的真实产品主数据，将 v0.1 的通用标签体系升级为**按品线精细化**的 v0.2 版本。

**核心升级**：
- 每个标签配置 **AIPL 节点锚定**（默认节点 + 上下文动态调整）
- 每个标签配置 **渠道权重**（4 路数据源差异化）
- 每个标签配置 **画像推导信号**（命中该标签时如何影响用户画像）
- 按 **4 大 VOC 品线** 拆分标签子集，避免品线交叉污染

**品线结构**：

| VOC 品线 | SKU 数 | 占比 | 核心子品类 |
|----------|--------|------|-----------|
| **A. 吸奶器** | 129 | 30% | 穿戴式吸奶器(38) + 配件(91) |
| **B. 内衣服饰** | 117 | 27% | 文胸(83) + 内裤/运动/塑身(34) |
| **C. 喂养电器** | 19 | 4% | 暖奶器(10) + 消毒器(8) + 调奶器/辅食机(5) |
| **D. 智能母婴电器** | 15 | 3% | 婴儿监视器(6) + 配件(20) + 加湿器(4) |
| **其他/边缘** | ~151 | 35% | 宠物用品、按摩仪等 |

*注：本文档聚焦 A/B/C/D 四大核心品线，边缘品类使用通用标签子集。*

---

## 一、v0.2 标签字典模板（按品线精细化版）

每个标签必须包含以下全部字段：

```yaml
标签ID: {品线缩写}-{维度}-{序号}
标签名称: {中文名}
适用品线: [品线A, 品线B, ...]      # v0.2 新增：品线过滤
适用市场: [US, ES, CA, DE, UK, FR]  # v0.2 新增：市场范围

# === Layer 1: AIPL 锚定 ===
默认AIPL节点: {A|I|P1|P2|L1|L2|L3}
AIPL调整规则:
  - 条件: {逻辑表达式}
    节点: {调整后节点}
    权重倍率: {float}

# === Layer 1: 渠道权重 ===
渠道权重:
  return_note: {float}     # 退货留言
  ticket: {float}         # 客服工单
  review: {float}         # 商品评论
  trustpilot: {float}     # Trustpilot

# === Layer 0: 语义本体 ===
语义本体映射:
  英文: [关键词列表]
  西班牙文: [关键词列表]
  法文: [关键词列表]
  德文: [关键词列表]

触发规则:
  - NLP置信度: > {阈值}
  - 语义模式: [必须共现的词模式]
  - 排除模式: [否定/例外模式]

# === Layer 2: 画像推导 ===
画像推导信号:
  - 条件: {命中此标签 + 其他条件}
    原子标签加分: {标签名: 权重}
    推导画像加分: {画像名: 权重}

# === Layer 3: 业务闭环 ===
业务闭环:
  策略包: {策略包名}
  主责部门: {部门名}
  默认优先级: {P0|P1|P2|P3}
  优先级升级规则:
    - 条件: {聚合条件}
      升级: {目标优先级}
      附加动作: {自动化动作}

# === Layer 4: 动态监控 ===
漂移监控:
  监控指标: {指标名}
  告警阈值: {阈值}
  自动动作: {动作描述}

版本: v0.2
```

---

## 二、品线 A：吸奶器（129 SKU）- 核心品线

> 子品类：一体穿戴式(19) / 分离穿戴式(13) / 双边穿戴式(6) / 手动(2) / 电动双边(1) / 配件(91)
> 市场：美/西/加/德/英/法
> 特殊属性：功能性极强、配件生态复杂、竞品对比频繁、情感强度高

### 2.1 吸奶器 VOC 特殊性分析

**为什么吸奶器的标签需要独立设计？**

1. **功能性极强**：吸奶器是"医疗器械级"消费品， suction strength / noise level / battery life 是核心决策因素，VOC 中功能性描述占比超过 60%
2. **配件生态复杂**：一台吸奶器有 5-15 个配件（导管、阀门、隔膜、乳罩、连接器、奶瓶等），配件问题经常"嫁祸"给主机
3. **竞品对比频繁**：Spectra / Medela / Willow / Elvie / Lansinoh 是高频对比品牌，用户常做 A vs B 决策
4. **情感强度高**：与母乳喂养直接相关，失败体验带来的挫败感极强，负面 VOC 的情感强度通常比其他品类高 30-50%
5. **使用场景复杂**：夜间（噪音敏感）、公司背奶（便携+隐蔽）、外出（续航+便携）

### 2.2 吸奶器标签子集（30 个核心标签）

#### A1. 吸力性能（Suction Performance）- 6 标签

```yaml
标签ID: A-SUC-001
标签名称: 吸力太强
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: P1
AIPL调整规则:
  - 条件: source_type == "review" AND 购买后天数 < 14
    节点: L1
    权重倍率: 1.3
  - 条件: text 包含 "first time" OR "new to pumping"
    节点: L1
    权重倍率: 1.2

渠道权重:
  return_note: 1.5
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["too strong", "too much suction", "painful suction", "hurts", "nipple pain", "too powerful"]
  西班牙文: ["demasiada succión", "duele", "dolor de pezón", "muy potente"]
  法文: ["trop forte succion", "fait mal", "douleur du mamelon", "trop puissante"]
  德文: ["zu stark", "schmerzhaft", "Schmerzen", "zu viel Saugen"]

触发规则:
  - NLP置信度: > 0.80
  - 语义模式: [吸力词] + [过强词] + [负面情感/身体反应]
  - 排除模式: ["not too strong", "just right", "perfect strength"]

画像推导信号:
  - 条件: 命中此标签 + 首次购买
    原子标签加分: {first_time_parent: +2, anxiety_driven: +1}
    推导画像加分: {systematic_planner: +1}
  - 条件: 命中此标签 + 提及 " flange "
    原子标签加分: {research_driven: +1}

业务闭环:
  策略包: 吸力体验优化包
  主责部门: 产品中心
  默认优先级: P1
  优先级升级规则:
    - 条件: 同SKU 7天内 ≥5例
      升级: P0
      附加动作: 触发吸力档位校准流程

VOC示例:
  - "The suction is way too strong even on the lowest setting. My nipples were so sore after one use."
  - "Demasiada succión, tuve que devolverlo." (吸力太强，不得不退货)
  - 西班牙市场: suction pain 的提及率比美国高 20%（文化差异：西语用户更直接表达身体不适）
```

```yaml
标签ID: A-SUC-002
标签名称: 吸力不足
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "return" OR "refund"
    节点: L1
    权重倍率: 1.5
  - 条件: text 包含 "compared with" AND 竞品提及
    节点: I
    权重倍率: 1.2

渠道权重:
  return_note: 1.5
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["weak suction", "not enough suction", "doesn't empty", "low suction power", "poor suction"]
  西班牙文: ["succión débil", "no succiona bien", "baja potencia"]
  法文: ["succion faible", "pas assez puissante", "n'aspire pas bien"]
  德文: ["schwacher Sog", "nicht stark genug", "zu wenig"]

触发规则:
  - NLP置信度: > 0.75
  - 语义模式: [吸力词] + [不足词] + [负面情感]
  - 排除模式: ["not weak", "strong enough", "good suction"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "supply" OR "low supply"
    原子标签加分: {anxiety_driven: +2}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 核心体验改良包
  主责部门: 产品中心
  默认优先级: P0
  优先级升级规则:
    - 条件: 同型号 30天内 ≥3例
      升级: P0
      附加动作: 立即启动吸力测试复核

VOC示例:
  - "Suction is weak compared to my Spectra. Takes 30 minutes to get half the amount."
  - "No succiona como mi antigua Medela, muy decepcionada." (吸力不如我的旧Medela，很失望)
```

```yaml
标签ID: A-SUC-003
标签名称: 吸力模式不适
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.3
  ticket: 1.2
  review: 1.0
  trustpilot: 0.7

语义本体映射:
  英文: ["mode", "pattern", "cycle", "rhythm", "uncomfortable pattern", "weird rhythm"]
  西班牙文: ["modo", "patrón", "ritmo"]
  法文: ["mode", "rythme", "pattern"]
  德文: ["Modus", "Rhythmus", "Muster"]

业务闭环:
  策略包: 核心体验改良包
  主责部门: 产品中心
  默认优先级: P2
```

#### A2. 噪音控制（Noise Control）- 4 标签

```yaml
标签ID: A-NOI-001
标签名称: 噪音太大
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "night" OR "sleeping" OR "baby sleeping"
    节点: L1
    权重倍率: 1.5
    备注: 夜间噪音是最高频痛点
  - 条件: text 包含 "office" OR "work" OR "pumping at work"
    节点: L1
    权重倍率: 1.3
    备注: 公司背奶场景的噪音尴尬
  - 条件: source_type == "review" AND text 包含 "compared"
    节点: I
    权重倍率: 1.0

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["loud", "noisy", "wakes baby", "too loud", "can hear it", "embarrassing noise"]
  西班牙文: ["ruidoso", "muy ruidoso", "despierta al bebé", "ruido molesto"]
  法文: ["bruyant", "trop bruyant", "réveille bébé"]
  德文: ["laut", "zu laut", "weckt das Baby"]

触发规则:
  - NLP置信度: > 0.75
  - 语义模式: [噪音词] + [负面情感/场景词]
  - 排除模式: ["not loud", "quiet", "silent", "barely hear"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "night" OR "sleeping"
    原子标签加分: {nighttime_user: +2, quiet_seeker: +2}
    推导画像加分: {systematic_planner: +1}
  - 条件: 命中此标签 + 提及 "office" OR "work"
    原子标签加分: {working_mom: +2, quiet_seeker: +2}
    推导画像加分: {community_driven: +1}

业务闭环:
  策略包: 静音体验优化包
  主责部门: 产品中心
  默认优先级: P1
  优先级升级规则:
    - 条件: 同型号 "噪音" 标签提及率 > 15%
      升级: P0
      附加动作: 启动降噪技术升级项目

VOC示例:
  - "It's so loud I can't use it when the baby is sleeping. Had to switch back to my manual pump at night."
  - "Too noisy for the office, everyone can hear it. So embarrassing."
  - 德国市场: 德文用户对噪音的容忍度最低，"zu laut" 的提及率比英语市场高 35%
```

```yaml
标签ID: A-NOI-002
标签名称: 静音好评
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L3
AIPL调整规则:
  - 条件: text 包含 "compared with" AND 竞品提及
    节点: I
    权重倍率: 1.2
    备注: 对比场景下静音是差异化卖点

渠道权重:
  return_note: 0.0      # 退货留言中不会出现静音好评
  ticket: 0.0
  review: 1.2
  trustpilot: 1.5       # 第三方平台好评可信度高

语义本体映射:
  英文: ["quiet", "silent", "barely hear", "discreet", "whisper quiet"]
  西班牙文: ["silencioso", "quieto", "casi no se oye"]
  法文: ["silencieux", "discret", "on n'entend presque rien"]
  德文: ["leise", "still", "kaum hörbar"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "office" OR "night"
    原子标签加分: {quiet_seeker: +2}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 差异化卖点强化包
  主责部门: 品牌营销部
  默认优先级: P3
```

#### A3. 穿戴舒适度（Wearable Comfort）- 5 标签

```yaml
标签ID: A-WEA-001
标签名称: 穿戴不适
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "first time wearing" OR "tried on"
    节点: P1
    权重倍率: 1.0
    备注: 试穿就发现不适 → 购买前评估失败

渠道权重:
  return_note: 1.5
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["uncomfortable", "hurts to wear", "too tight", "pinches", "digging in", "bulky"]
  西班牙文: ["incómodo", "aprieta", "molesto", "demasiado grande"]
  法文: ["inconfortable", "serre", "fait mal", "trop volumineux"]
  德文: ["unbequem", "drückt", "zu eng", "sperrig"]

触发规则:
  - NLP置信度: > 0.75

画像推导信号:
  - 条件: 命中此标签 + 提及 "flange" OR "shield"
    原子标签加分: {research_driven: +1}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 穿戴体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

```yaml
标签ID: A-WEA-002
标签名称: 隐蔽性不足
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "work" OR "office" OR "public"
    节点: L1
    权重倍率: 1.3
    备注: 外出/工作场景的隐蔽性需求更高

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["bulky", "noticeable", "shows through", "not discreet", "awkward shape", "weird lumps"]
  西班牙文: ["se nota", "voluminoso", "forma rara"]
  法文: ["visible", "encombrant", "forme bizarre"]
  德文: ["auffällig", "sperrig", "sieht komisch aus"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "work" OR "office"
    原子标签加分: {working_mom: +2}
    推导画像加分: {community_driven: +1}

业务闭环:
  策略包: 便携体验优化包
  主责部门: 产品中心
  默认优先级: P2
```

```yaml
标签ID: A-WEA-003
标签名称: 固定不稳
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.5
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["falls out", "doesn't stay", "slips", "loose", "keeps detaching", "pops off"]
  西班牙文: ["se cae", "no se mantiene", "se desprende"]
  法文: ["tombe", "ne tient pas", "se détache"]
  德文: ["fällt raus", "hält nicht", "rutscht"]

业务闭环:
  策略包: 核心体验改良包
  主责部门: 产品中心
  默认优先级: P1
```

#### A4. 续航与充电（Battery & Charging）- 3 标签

```yaml
标签ID: A-BAT-001
标签名称: 续航不足
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "travel" OR "flight" OR "commute"
    节点: L1
    权重倍率: 1.3
    备注: 外出场景续航需求更高

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["battery dies", "doesn't last", "short battery life", "needs charging often", "dead by noon"]
  西班牙文: ["batería dura poco", "se descarga rápido"]
  法文: ["batterie faible", "ne dure pas", "se décharge vite"]
  德文: ["Akku hält nicht", "schnell leer", "kurze Akkulaufzeit"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "travel" OR "commute"
    原子标签加分: {portable_needer: +2}
    推导画像加分: {community_driven: +1}

业务闭环:
  策略包: 续航体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

#### A5. 清洗便捷性（Cleaning Ease）- 3 标签

```yaml
标签ID: A-CLE-001
标签名称: 清洗困难
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2
AIPL调整规则:
  - 条件: text 包含 "too many parts" OR "so many pieces"
    节点: L1
    权重倍率: 1.2
    备注: 配件太多 → 首购体验即受挫

渠道权重:
  return_note: 1.2
  ticket: 1.0
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["hard to clean", "too many parts", "difficult to assemble", "mold", "milk residue", "tricky to wash"]
  西班牙文: ["difícil de limpiar", "demasiadas piezas", "moho"]
  法文: ["difficile à nettoyer", "trop de pièces", "moisissure"]
  德文: ["schwer zu reinigen", "zu viele Teile", "Schimmel"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "exhausted" OR "tired" OR "no time"
    原子标签加分: {anxiety_driven: +1, working_mom: +1}

业务闭环:
  策略包: 清洗体验优化包
  主责部门: 产品中心
  默认优先级: P2
```

#### A6. 配件相关（Accessories）- 4 标签

```yaml
标签ID: A-ACC-001
标签名称: 配件耐用性差
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2
AIPL调整规则:
  - 条件: text 包含 "first week" OR "broke after"
    节点: L1
    权重倍率: 1.3
    备注: 短期内损坏 → 首购体验问题

渠道权重:
  return_note: 1.2
  ticket: 1.3        # 配件问题常通过客服解决
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["broke", "cracked", "wore out", "tore", "doesn't fit anymore", "stretched out"]
  西班牙文: ["se rompió", "agrietado", "desgastado"]
  法文: ["cassé", "fissuré", "usé"]
  德文: ["gebrochen", "gerissen", "verschlissen"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "expensive" OR "costly"
    原子标签加分: {price_sensitive: +2}
    推导画像加分: {quality_explorer: +1}

业务闭环:
  策略包: 配件质量优化包
  主责部门: 供应链管理部
  默认优先级: P1
```

```yaml
标签ID: A-ACC-002
标签名称: 配件价格贵
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2

渠道权重:
  return_note: 0.8
  ticket: 1.2
  review: 1.0
  trustpilot: 1.0

语义本体映射:
  英文: ["expensive accessories", "overpriced parts", "costly replacements", "constantly buying"]
  西班牙文: ["accesorios caros", "repuestos costosos"]
  法文: ["accessoires chers", "pièces coûteuses"]
  德文: ["teures Zubehör", "Ersatzteile zu teuer"]

画像推导信号:
  - 条件: 命中此标签
    原子标签加分: {price_sensitive: +2}
    推导画像加分: {community_driven: +1}

业务闭环:
  策略包: 配件定价优化包
  主责部门: 财务/定价部
  默认优先级: P2
```

#### A7. 漏奶问题（Leakage）- 2 标签

```yaml
标签ID: A-LEA-001
标签名称: 漏奶
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "ruined" OR "stained" OR "clothes"
    节点: L1
    权重倍率: 1.4
    备注: 漏奶导致衣物损坏 → 情感强度高

渠道权重:
  return_note: 1.5
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["leaks", "leaking", "spills", "milk everywhere", "soaked", "drips"]
  西班牙文: ["gotea", "pierde leche", "se derrama"]
  法文: ["fuit", "perd du lait", "dégouline"]
  德文: ["undicht", "leckt", "Milch überall"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "work" OR "office"
    原子标签加分: {working_mom: +2, anxiety_driven: +1}

业务闭环:
  策略包: 密封性能优化包
  主责部门: 产品中心
  默认优先级: P0
  优先级升级规则:
    - 条件: 同型号 7天内 ≥3例
      升级: P0
      附加动作: 启动密封设计复核

VOC示例:
  - "Leaks constantly! Ruined two work shirts. So embarrassing at the office."
  - "Gotea todo el tiempo, perdí leche preciosa." (一直漏奶，浪费了珍贵的母乳)
```

#### A8. 智能/App 功能（Smart Features）- 2 标签

```yaml
标签ID: A-APP-001
标签名称: App连接/功能问题
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2

渠道权重:
  return_note: 1.0
  ticket: 1.5        # App问题常通过客服解决
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["app doesn't work", "can't connect", "bluetooth issues", "sync problems", "crashes"]
  西班牙文: ["app no funciona", "no se conecta", "problemas Bluetooth"]
  法文: ["application ne marche pas", "problème de connexion"]
  德文: ["App funktioniert nicht", "Verbindungsprobleme"]

业务闭环:
  策略包: App体验优化包
  主责部门: 软件开发部
  默认优先级: P2
```

#### A9. 尺寸/法兰适配（Sizing & Fit）- 1 标签

```yaml
标签ID: A-SIZ-001
标签名称: 法兰尺寸不适
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: P1
AIPL调整规则:
  - 条件: text 包含 "ordered" AND ("wrong size" OR "wrong flange")
    节点: P1
    权重倍率: 1.2
    备注: 购买前尺码选择错误
  - 条件: source_type == "return_note"
    节点: L1
    权重倍率: 1.3

渠道权重:
  return_note: 1.5
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["wrong flange size", "flange too small", "flange too big", "need different size", "nipple doesn't fit"]
  西班牙文: ["talla incorrecta", "brida demasiado pequeña", "no encaja"]
  法文: ["mauvaise taille", "trop petit", "trop grand"]
  德文: ["falsche Größe", "zu klein", "zu groß"]

画像推导信号:
  - 条件: 命中此标签 + 首次购买
    原子标签加分: {first_time_parent: +2, research_driven: +1}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 尺码指导优化包
  主责部门: 品牌营销部 + 产品中心
  默认优先级: P1
```

### 2.3 吸奶器标签子集汇总

| 标签组 | 标签数 | 高频标签 | P0标签 |
|--------|--------|----------|--------|
| 吸力性能 | 3 | 吸力不足、吸力太强 | 吸力不足 |
| 噪音控制 | 2 | 噪音太大 | - |
| 穿戴舒适度 | 3 | 穿戴不适、隐蔽性不足 | - |
| 续航充电 | 1 | 续航不足 | - |
| 清洗便捷 | 1 | 清洗困难 | - |
| 配件相关 | 2 | 配件耐用性差、配件价格贵 | - |
| 漏奶问题 | 1 | 漏奶 | 漏奶 |
| 智能/App | 1 | App连接问题 | - |
| 尺寸法兰 | 1 | 法兰尺寸不适 | - |
| **合计** | **15** | | **2个P0** |

---

## 三、品线 B：内衣服饰（117 SKU）

> 子品类：文胸(83) / 内裤(10) / 运动(11) / 塑身(3) / 家居服(3) / 袜子(4) / 其他(3)
> 市场：美/西/加/德/英/法
> 特殊属性：舒适性主导、尺码敏感性极高、孕期身体变化、清洗频率高

### 3.1 内衣服饰 VOC 特殊性分析

**为什么内衣服饰的标签需要独立设计？**

1. **尺码敏感性极高**：孕期/哺乳期身体变化大，同一用户在不同孕期阶段需要不同尺码，尺码选择困难是退货的首要原因
2. **长时间穿戴**：哺乳期每天穿戴 12+ 小时，舒适度要求比日常内衣更高
3. **单手操作需求**：哺乳文胸需要一手抱娃一手穿脱，操作便利性直接影响使用体验
4. **清洗频率高**：哺乳期分泌物多，需要频繁清洗，变形/褪色/起球问题突出
5. **外观与功能的平衡**：既要实用（哺乳开口、支撑）又要美观（颜色、款式）
6. **AIPL 节点分布不同**：内衣服饰的 P1（评估）和 L1（首购使用）阶段占比远高于其他品线，因为试穿/首穿即决定满意度

### 3.2 内衣服饰标签子集（20 个核心标签）

#### B1. 尺码合身（Sizing & Fit）- 6 标签

```yaml
标签ID: B-SIZ-001
标签名称: 尺码偏大
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: P1
AIPL调整规则:
  - 条件: source_type == "return_note"
    节点: L1
    权重倍率: 1.5
  - 条件: text 包含 "ordered" AND ("size down" OR "smaller size")
    节点: P1
    权重倍率: 1.2

渠道权重:
  return_note: 1.5        # 尺码问题是退货头号原因
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["too big", "runs large", "size down", "loose", "baggy", "gaping", "roomy"]
  西班牙文: ["demasiado grande", "talla grande", "suelto"]
  法文: ["trop grand", "taille large", "ample"]
  德文: ["zu groß", "fällt groß aus", "weit"]

触发规则:
  - NLP置信度: > 0.75

画像推导信号:
  - 条件: 命中此标签 + 提及 "pregnant" OR "pregnancy"
    原子标签加分: {first_time_parent: +1}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 尺码体验优化包
  主责部门: 产品中心
  默认优先级: P1
  优先级升级规则:
    - 条件: 同SKU 30天内 "尺码" 相关退货率 > 20%
      升级: P0
      附加动作: 启动尺码表复核

VOC示例:
  - "Ordered my pre-pregnancy size but it's way too big. The band keeps riding up."
  - "Demasiado grande, tuve que pedir una talla menos." (太大了，不得不换小一码)
```

```yaml
标签ID: B-SIZ-002
标签名称: 尺码偏小
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: P1
AIPL调整规则:
  - 条件: source_type == "return_note"
    节点: L1
    权重倍率: 1.5

渠道权重:
  return_note: 1.5
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["too small", "runs small", "size up", "tight", "squeezing", "digging in"]
  西班牙文: ["demasiado pequeño", "talla pequeña", "apretado"]
  法文: ["trop petit", "taille petit", "serré"]
  德文: ["zu klein", "fällt klein aus", "eng"]

业务闭环:
  策略包: 尺码体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

```yaml
标签ID: B-SIZ-003
标签名称: 尺码表不清
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: P1
AIPL调整规则:
  - 条件: text 包含 "confused" OR "not sure" OR "unclear"
    节点: P1
    权重倍率: 1.2

渠道权重:
  return_note: 1.0
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["sizing chart confusing", "not sure what size", "between sizes", "size guide unclear"]
  西班牙文: ["tabla de tallas confusa", "no sé qué talla"]
  法文: ["guide des tailles confus", "pas sûr de la taille"]
  德文: ["Größentabelle verwirrend", "unsicher welche Größe"]

画像推导信号:
  - 条件: 命中此标签
    原子标签加分: {first_time_parent: +2, anxiety_driven: +1}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 尺码指导优化包
  主责部门: 品牌营销部
  默认优先级: P2
```

#### B2. 支撑与承托（Support）- 3 标签

```yaml
标签ID: B-SUP-001
标签名称: 支撑力不足
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "heavy" OR "large breasts" OR "DDD" OR "F cup"
    节点: L1
    权重倍率: 1.3
    备注: 大胸用户对支撑力要求更高

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["no support", "sagging", "not enough support", "flimsy", "no lift", "bounces"]
  西班牙文: ["sin soporte", "no sostiene", "se cae"]
  法文: ["pas de soutien", "trop mou", "pas de maintien"]
  德文: ["keine Stütze", "zu weich", "hält nicht"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "large" OR "heavy"
    原子标签加分: {quality_explorer: +1}

业务闭环:
  策略包: 支撑体验优化包
  主责部门: 产品中心
  默认优先级: P2
```

#### B3. 哺乳便利性（Nursing Convenience）- 4 标签

```yaml
标签ID: B-NUR-001
标签名称: 单手操作困难
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "one hand" OR "holding baby"
    节点: L1
    权重倍率: 1.3
    备注: 一手抱娃一手操作的场景

渠道权重:
  return_note: 1.3
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["hard to open", "clasp difficult", "need two hands", "fumbling", "awkward to undo"]
  西班牙文: ["difícil de abrir", "complicado", "necesito dos manos"]
  法文: ["difficile à ouvrir", "besoin de deux mains"]
  德文: ["schwer zu öffnen", "zwei Hände nötig"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "baby" OR "crying"
    原子标签加分: {first_time_parent: +1, anxiety_driven: +1}

业务闭环:
  策略包: 哺乳体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

```yaml
标签ID: B-NUR-002
标签名称: 哺乳开口设计问题
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["clip broke", "clasp came off", "opening too small", "hard to access", "poor design"]
  西班牙文: ["clip se rompió", "apertura pequeña", "difícil acceso"]
  法文: ["clip cassé", "ouverture trop petite"]
  德文: ["Clip gebrochen", "Öffnung zu klein"]

业务闭环:
  策略包: 哺乳体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

#### B4. 面料舒适（Fabric Comfort）- 3 标签

```yaml
标签ID: B-FAB-001
标签名称: 面料不舒适
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "itchy" OR "scratchy" OR "rash"
    节点: L1
    权重倍率: 1.4
    备注: 皮肤过敏 → 安全相关

渠道权重:
  return_note: 1.3
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["itchy", "scratchy", "rough", "uncomfortable fabric", "irritating", "stiff"]
  西班牙文: ["pica", "áspero", "irrita"]
  法文: ["démange", "rugueux", "irrite"]
  德文: ["juckt", "kratzt", "unangenehm"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "sensitive skin" OR "allergy"
    原子标签加分: {anxiety_driven: +2}
    推导画像加分: {quality_explorer: +1}

业务闭环:
  策略包: 面料体验优化包
  主责部门: 产品中心
  默认优先级: P2
  优先级升级规则:
    - 条件: 提及 "rash" OR "allergic"
      升级: P0
      附加动作: 触发材质安全检测
```

#### B5. 洗后问题（Post-Wash Issues）- 2 标签

```yaml
标签ID: B-WAS-001
标签名称: 洗后变形
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2
AIPL调整规则:
  - 条件: text 包含 "after one wash" OR "first wash"
    节点: L1
    权重倍率: 1.3
    备注: 首次清洗即变形 → 首购体验

渠道权重:
  return_note: 1.2
  ticket: 1.0
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["shrunk", "stretched out", "lost shape", "warped", "misshapen after wash"]
  西班牙文: ["encogió", "perdió forma", "deformado"]
  法文: ["a rétréci", "a perdu sa forme", "déformé"]
  德文: ["eingegangen", "verzogen", "Form verloren"]

业务闭环:
  策略包: 耐洗性优化包
  主责部门: 供应链管理部
  默认优先级: P2
```

#### B6. 外观满意（Aesthetics）- 2 标签

```yaml
标签ID: B-AES-001
标签名称: 外观不满意
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.0
  ticket: 0.8
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["ugly", "frumpy", "grandma style", "looks cheap", "not cute", "boring color"]
  西班牙文: ["feo", "anticuado", "parece barato"]
  法文: ["moche", "ringard", "pas joli"]
  德文: ["hässlich", "altmodisch", "sieht billig aus"]

画像推导信号:
  - 条件: 命中此标签
    原子标签加分: {quality_explorer: +1}

业务闭环:
  策略包: 外观设计优化包
  主责部门: 产品中心
  默认优先级: P3
```

### 3.3 内衣服饰标签子集汇总

| 标签组 | 标签数 | 高频标签 | P0标签 |
|--------|--------|----------|--------|
| 尺码合身 | 3 | 尺码偏大、尺码偏小 | - |
| 支撑承托 | 1 | 支撑力不足 | - |
| 哺乳便利 | 2 | 单手操作困难 | - |
| 面料舒适 | 1 | 面料不舒适 | 皮肤过敏→P0 |
| 洗后问题 | 1 | 洗后变形 | - |
| 外观满意 | 1 | 外观不满意 | - |
| **合计** | **9** | | **1个潜在P0** |

---

## 四、品线 C：喂养电器（19 SKU）

> 子品类：暖奶器(10) + 消毒器(8) + 调奶器(5) + 辅食机(5)
> 市场：美/西/加/德/英/法
> 特殊属性：加热精准性要求高、材质安全敏感、操作便捷性关键

### 4.1 喂养电器 VOC 特殊性分析

**为什么喂养电器的标签需要独立设计？**

1. **加热/消毒精准性**：温度不准可能导致奶质破坏（营养流失）或烫伤，是安全性问题
2. **材质安全（加热后）**：加热后是否有异味/化学物质释放，是母婴特有的安全关注点
3. **操作便捷性**：夜间使用场景需要简单操作（半睡半醒状态），按键复杂度直接影响体验
4. **容量限制**：一次能处理多少奶瓶/配件，影响使用效率
5. **清洗维护**：水垢、残留奶渍的清洁问题
6. **AIPL 节点集中**：L1（首购使用）和 L2（持续使用）阶段占绝对主导，因为购买前的评估通常较少涉及具体使用细节

### 4.2 喂养电器标签子集（12 个核心标签）

#### C1. 加热性能（Heating Performance）- 4 标签

```yaml
标签ID: C-HEA-001
标签名称: 温度不准
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "too hot" OR "burned" OR "scalding"
    节点: L1
    权重倍率: 1.5
    备注: 温度过高 → 安全风险
  - 条件: text 包含 "nutrients" OR "destroyed"
    节点: L1
    权重倍率: 1.3
    备注: 营养破坏 → 健康风险

渠道权重:
  return_note: 1.5
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["temperature inaccurate", "too hot", "too cold", "not 98 degrees", "scalding", "lukewarm"]
  西班牙文: ["temperatura incorrecta", "demasiado caliente", "templado"]
  法文: ["température inexacte", "trop chaud", "tiede"]
  德文: ["Temperatur ungenau", "zu heiß", "zu kalt"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "baby" AND ("burned" OR "too hot")
    原子标签加分: {anxiety_driven: +2}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 温控安全优化包
  主责部门: 产品中心
  默认优先级: P0
  优先级升级规则:
    - 条件: 提及 "burned" OR "scalded" OR "baby cried"
      升级: P0
      附加动作: 立即启动温度传感器校准

VOC示例:
  - "The temperature is all over the place. Sometimes scalding hot, sometimes barely warm. Not safe for baby."
  - "La temperatura no es consistente, tengo que probar siempre antes." (温度不一致，每次都要先试)
```

```yaml
标签ID: C-HEA-002
标签名称: 加热不均匀
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["uneven heating", "hot spots", "cold spots", "not evenly warm"]
  西班牙文: ["calentamiento desigual", "puntos calientes"]
  法文: ["chauffage inégal", "points chauds"]
  德文: ["ungleichmäßige Erwärmung", "heiße Stellen"]

业务闭环:
  策略包: 加热体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

```yaml
标签ID: C-HEA-003
标签名称: 加热速度慢
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "crying" OR "hungry" OR "waiting"
    节点: L1
    权重倍率: 1.3
    备注: 宝宝哭闹等待 → 情感强度高

渠道权重:
  return_note: 1.2
  ticket: 1.0
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["takes too long", "slow", "waiting forever", "impatient", "hungry baby"]
  西班牙文: ["tarda mucho", "lento", "bebe hambriento"]
  法文: ["trop long", "lent", "bébé affamé"]
  德文: ["dauert zu lange", "langsam", "hungriges Baby"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "night" OR "3am"
    原子标签加分: {nighttime_user: +2, anxiety_driven: +1}

业务闭环:
  策略包: 加热效率优化包
  主责部门: 产品中心
  默认优先级: P2
```

#### C2. 操作便捷（Ease of Use）- 3 标签

```yaml
标签ID: C-EAS-001
标签名称: 操作复杂
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "middle of the night" OR "3am" OR "exhausted"
    节点: L1
    权重倍率: 1.4
    备注: 夜间疲惫状态下的操作困难

渠道权重:
  return_note: 1.2
  ticket: 1.3        # 操作问题常通过客服指导解决
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["complicated", "too many buttons", "confusing", "not intuitive", "hard to figure out"]
  西班牙文: ["complicado", "muchos botones", "confuso"]
  法文: ["compliqué", "trop de boutons", "pas intuitif"]
  德文: ["kompliziert", "zu viele Knöpfe", "nicht intuitiv"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "husband" OR "grandma"
    原子标签加分: {first_time_parent: +1}
    备注: 家人帮忙使用时操作困难

业务闭环:
  策略包: 操作体验优化包
  主责部门: 产品中心
  默认优先级: P2
```

#### C3. 材质安全（Material Safety）- 2 标签

```yaml
标签ID: C-MAT-001
标签名称: 加热后异味
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "plastic smell" OR "chemical" OR "worried"
    节点: L1
    权重倍率: 1.5
    备注: 化学物质担忧 → 安全敏感

渠道权重:
  return_note: 1.3
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["plastic smell", "chemical odor", "weird smell", "tastes funny", "concerned about plastic"]
  西班牙文: ["olor a plástico", "olor químico", "sabe raro"]
  法文: ["odeur de plastique", "odeur chimique", "goût bizarre"]
  德文: ["Plastikgeruch", "Chemiegeruch", "schmeckt komisch"]

画像推导信号:
  - 条件: 命中此标签
    原子标签加分: {anxiety_driven: +2, quality_explorer: +1}

业务闭环:
  策略包: 材质安全优化包
  主责部门: 产品中心 + 品质管理中心
  默认优先级: P0
  优先级升级规则:
    - 条件: 同型号 30天内 ≥2例
      升级: P0
      附加动作: 启动材质安全检测
```

#### C4. 容量与效率（Capacity）- 2 标签

```yaml
标签ID: C-CAP-001
标签名称: 容量不足
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.2
  ticket: 1.0
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["too small", "only fits one", "not enough space", "have to do multiple batches"]
  西班牙文: ["demasiado pequeño", "solo cabe uno", "varias tandas"]
  法文: ["trop petit", "ne tient qu'un", "plusieurs cycles"]
  德文: ["zu klein", "passt nur einer", "mehrere Durchgänge"]

业务闭环:
  策略包: 容量规划优化包
  主责部门: 产品中心
  默认优先级: P2
```

#### C5. 清洗维护（Cleaning）- 1 标签

```yaml
标签ID: C-CLE-001
标签名称: 清洗困难/水垢问题
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2

渠道权重:
  return_note: 1.0
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["hard to clean", "limescale", "mineral buildup", "discolored", "stains"]
  西班牙文: ["difícil de limpiar", "cal", "manchas"]
  法文: ["difficile à nettoyer", "calcaire", "taches"]
  德文: ["schwer zu reinigen", "Kalk", "Verfärbungen"]

业务闭环:
  策略包: 清洗体验优化包
  主责部门: 产品中心
  默认优先级: P3
```

### 4.3 喂养电器标签子集汇总

| 标签组 | 标签数 | 高频标签 | P0标签 |
|--------|--------|----------|--------|
| 加热性能 | 3 | 温度不准 | 温度不准 |
| 操作便捷 | 1 | 操作复杂 | - |
| 材质安全 | 1 | 加热后异味 | 加热后异味 |
| 容量效率 | 1 | 容量不足 | - |
| 清洗维护 | 1 | 清洗困难 | - |
| **合计** | **7** | | **2个P0** |

---

## 五、品线 D：智能母婴电器（15 SKU）

> 子品类：婴儿监视器(6) + 配件(20) + 加湿器(4)
> 市场：美/西/加/德/英/法
> 特殊属性：技术性能主导、隐私安全敏感、安装复杂度、夜间使用场景

### 5.1 智能母婴电器 VOC 特殊性分析

**为什么智能母婴电器的标签需要独立设计？**

1. **技术性能主导**：画质、连接稳定性、延迟是核心体验，与传统母婴产品的"舒适度"维度完全不同
2. **隐私安全敏感**：摄像头类产品，家长对数据隐私高度关注（尤其是云端存储、第三方访问）
3. **安装复杂度**：需要配置 WiFi、安装支架、下载 App，技术门槛高于其他母婴产品
4. **夜间使用场景**：夜视效果、红外灯是否刺眼、是否干扰宝宝睡眠
5. **多设备管理**：部分家庭多个摄像头，App 管理复杂度
6. **AIPL 节点特殊**：P1（评估）阶段技术参数对比极多，L1（首购使用）阶段安装问题集中爆发

### 5.2 智能母婴电器标签子集（10 个核心标签）

#### D1. 画质与显示（Video Quality）- 2 标签

```yaml
标签ID: D-VID-001
标签名称: 画质不清晰
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "night vision" OR "in the dark"
    节点: L1
    权重倍率: 1.3
    备注: 夜视画质问题

渠道权重:
  return_note: 1.3
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["blurry", "grainy", "poor video", "can't see clearly", "pixelated", "low resolution"]
  西班牙文: ["borroso", "granulado", "no se ve bien"]
  法文: ["flou", "granuleux", "mauvaise qualité"]
  德文: ["unscharf", "verpixelt", "schlechte Qualität"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "night" OR "dark"
    原子标签加分: {nighttime_user: +1, quality_explorer: +1}

业务闭环:
  策略包: 画质体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

```yaml
标签ID: D-VID-002
标签名称: 夜视效果差
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "can't see baby" OR "completely dark"
    节点: L1
    权重倍率: 1.4
    备注: 夜间看不到宝宝 → 安全相关

渠道权重:
  return_note: 1.3
  ticket: 1.3
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["night vision bad", "can't see in dark", "infrared too bright", "glow wakes baby"]
  西班牙文: ["visión nocturna mala", "no veo en la oscuridad"]
  法文: ["vision nocturne mauvaise", "ne voit pas dans le noir"]
  德文: ["Nachtsicht schlecht", "sieht nicht im Dunkeln"]

业务闭环:
  策略包: 夜视体验优化包
  主责部门: 产品中心
  默认优先级: P1
```

#### D2. 连接稳定性（Connection）- 2 标签

```yaml
标签ID: D-CON-001
标签名称: 连接不稳定/断连
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "missed" OR "didn't notify" OR "alert failed"
    节点: L1
    权重倍率: 1.5
    备注: 漏报/未提醒 → 安全风险

渠道权重:
  return_note: 1.3
  ticket: 1.5        # 连接问题常需客服技术支持
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["keeps disconnecting", "connection drops", "unreliable", "lags", "delayed", "won't connect"]
  西班牙文: ["se desconecta", "conexión inestable", "retraso"]
  法文: ["se déconnecte", "connexion instable", "retard"]
  德文: ["verliert Verbindung", "Verbindung bricht", "verzögert"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "router" OR "WiFi"
    原子标签加分: {research_driven: +1}
    备注: 用户已尝试排查网络问题

业务闭环:
  策略包: 连接稳定性优化包
  主责部门: 软件开发部
  默认优先级: P0
  优先级升级规则:
    - 条件: 提及 "missed" OR "didn't alert" OR "baby cried"
      升级: P0
      附加动作: 启动连接协议紧急review
```

#### D3. App 体验（App Experience）- 2 标签

```yaml
标签ID: D-APP-001
标签名称: App体验差
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2

渠道权重:
  return_note: 1.0
  ticket: 1.5        # App问题主要通过客服解决
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["app crashes", "app is terrible", "UI confusing", "hard to navigate", "too many ads"]
  西班牙文: ["app se cierra", "app es terrible", "interfaz confusa"]
  法文: ["application plante", "mauvaise application", "interface confuse"]
  德文: ["App stürzt ab", "App ist schlecht", "Oberfläche verwirrend"]

业务闭环:
  策略包: App体验优化包
  主责部门: 软件开发部
  默认优先级: P2
```

#### D4. 隐私安全（Privacy）- 2 标签

```yaml
标签ID: D-PRI-001
标签名称: 隐私安全担忧
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: P1
AIPL调整规则:
  - 条件: text 包含 "hacked" OR "stranger" OR "someone else"
    节点: P1
    权重倍率: 1.5
    备注: 被入侵/陌生人访问 → 极端安全事件

渠道权重:
  return_note: 1.2
  ticket: 1.3
  review: 1.0
  trustpilot: 1.2       # Trustpilot上隐私担忧更容易传播

语义本体映射:
  英文: ["privacy concerns", "don't trust", "cloud storage", "who can see", "hacked", "data breach"]
  西班牙文: ["preocupación privacidad", "no confío", "almacenamiento nube"]
  法文: ["préoccupation vie privée", "stockage cloud", "piraté"]
  德文: ["Datenschutzbedenken", "Cloud-Speicher", "gehackt"]

画像推导信号:
  - 条件: 命中此标签
    原子标签加分: {anxiety_driven: +2, research_driven: +1}
    推导画像加分: {systematic_planner: +1}

业务闭环:
  策略包: 隐私安全优化包
  主责部门: 信息安全部 + 法务合规部
  默认优先级: P0
  优先级升级规则:
    - 条件: 提及 "hacked" OR "stranger" OR "breach"
      升级: P0
      附加动作: 立即启动安全事件响应

VOC示例:
  - "I don't feel comfortable with cloud storage. Who has access to footage of my baby?"
  - "Heard about baby monitors getting hacked. This one says 'encrypted' but I'm still worried."
  - 德国市场: 德文用户对隐私的敏感度最高，"Datenschutz" 相关提及率比美国高 40%
```

#### D5. 安装便捷（Installation）- 1 标签

```yaml
标签ID: D-INS-001
标签名称: 安装复杂
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.2
  ticket: 1.5        # 安装问题常通过客服指导
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["hard to install", "complicated setup", "mounting difficult", "instructions unclear"]
  西班牙文: ["difícil de instalar", "configuración complicada"]
  法文: ["difficile à installer", "installation compliquée"]
  德文: ["schwer zu installieren", "Setup kompliziert"]

业务闭环:
  策略包: 安装体验优化包
  主责部门: 产品中心
  默认优先级: P2
```

#### D6. 加湿性能（Humidification）- 1 标签（仅加湿器）

```yaml
标签ID: D-HUM-001
标签名称: 加湿效果不佳
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.3
  ticket: 1.1
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["not humid enough", "too much mist", "leaks water", "mold", "no difference"]
  西班牙文: ["no humedece suficiente", "gotea", "moho"]
  法文: ["pas assez d'humidité", "fuit", "moisissure"]
  德文: ["nicht feucht genug", "leckt", "Schimmel"]

业务闭环:
  策略包: 加湿体验优化包
  主责部门: 产品中心
  默认优先级: P2
```

### 5.3 智能母婴电器标签子集汇总

| 标签组 | 标签数 | 高频标签 | P0标签 |
|--------|--------|----------|--------|
| 画质显示 | 2 | 画质不清晰、夜视效果差 | - |
| 连接稳定 | 1 | 连接不稳定 | 连接不稳定 |
| App体验 | 1 | App体验差 | - |
| 隐私安全 | 1 | 隐私安全担忧 | 隐私安全担忧 |
| 安装便捷 | 1 | 安装复杂 | - |
| 加湿性能 | 1 | 加湿效果不佳 | - |
| **合计** | **7** | | **2个P0** |

---

## 六、跨品线通用标签

以下标签适用于全部品线，但渠道权重和 AIPL 锚定可能因品线而异。

### 6.1 物流维度（Logistics）- 5 标签

```yaml
标签ID: GEN-LOG-001
标签名称: 物流延迟
适用品线: [全部]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1
AIPL调整规则:
  - 条件: text 包含 "before baby arrived" OR "needed it sooner"
    节点: L1
    权重倍率: 1.4
    备注: 预产期前未送达 → 情感强度高
  - 条件: source_type == "return_note"
    节点: L1
    权重倍率: 1.0
    备注: 退货留言中物流延迟很少见

渠道权重:
  return_note: 0.5        # 退货通常不因为物流延迟
  ticket: 1.3
  review: 1.0
  trustpilot: 1.2        # 第三方平台物流抱怨更常见

语义本体映射:
  英文: ["late delivery", "took forever", "shipping delayed", "arrived late"]
  西班牙文: ["entrega tardía", "llegó tarde", "envío retrasado"]
  法文: ["livraison tardive", "arrivé en retard"]
  德文: ["verspätete Lieferung", "kam zu spät"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "urgent" OR "needed it"
    原子标签加分: {anxiety_driven: +1}

业务闭环:
  策略包: 物流体验优化包
  主责部门: 仓储物流部
  默认优先级: P2
```

```yaml
标签ID: GEN-LOG-002
标签名称: 包装破损
适用品线: [全部]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L1

渠道权重:
  return_note: 1.0
  ticket: 1.2
  review: 1.0
  trustpilot: 0.8

语义本体映射:
  英文: ["package damaged", "box crushed", "arrived broken", "poor packaging"]
  西班牙文: ["paquete dañado", "caja aplastada"]
  法文: ["colis endommagé", "boîte écrasée"]
  德文: ["Paket beschädigt", "Karton zerdrückt"]

业务闭环:
  策略包: 包装优化包
  主责部门: 仓储物流部
  默认优先级: P1
```

### 6.2 售后维度（Service）- 3 标签

```yaml
标签ID: GEN-SER-001
标签名称: 客服响应慢
适用品线: [全部]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2
AIPL调整规则:
  - 条件: text 包含 "urgent" OR "emergency" OR "baby needs"
    节点: L2
    权重倍率: 1.3

渠道权重:
  return_note: 0.0        # 退货留言不涉及客服响应
  ticket: 1.0            # 工单本身就是客服渠道
  review: 1.2            # 评论中抱怨客服 → 公开传播
  trustpilot: 1.5        # Trustpilot 客服抱怨影响品牌

语义本体映射:
  英文: ["slow response", "no reply", "waiting for days", "customer service terrible"]
  西班牙文: ["respuesta lenta", "sin respuesta", "servicio terrible"]
  法文: ["réponse lente", "pas de réponse", "service horrible"]
  德文: ["langsame Antwort", "keine Antwort", "schlechter Service"]

画像推导信号:
  - 条件: 命中此标签 + 提及 "refund" OR "return"
    原子标签加分: {price_sensitive: +1}

业务闭环:
  策略包: 客服体验优化包
  主责部门: 客户服务部
  默认优先级: P1
```

```yaml
标签ID: GEN-SER-002
标签名称: 退换货困难
适用品线: [全部]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L2

渠道权重:
  return_note: 1.0
  ticket: 1.5            # 退换货问题主要通过客服
  review: 1.2
  trustpilot: 1.2

语义本体映射:
  英文: ["hard to return", "refund refused", "complicated return", "restocking fee"]
  西班牙文: ["difícil devolver", "reembolso rechazado"]
  法文: ["difficile de retourner", "remboursement refusé"]
  德文: ["Rückgabe schwierig", "Erstattung verweigert"]

业务闭环:
  策略包: 售后流程优化包
  主责部门: 客户服务部
  默认优先级: P1
```

### 6.3 价格维度（Price）- 2 标签

```yaml
标签ID: GEN-PRI-001
标签名称: 价格过高
适用品线: [全部]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: I
AIPL调整规则:
  - 条件: text 包含 "after buying" OR "not worth"
    节点: L1
    权重倍率: 1.2
    备注: 购买后觉得不值
  - 条件: text 包含 "compared with" AND 竞品提及
    节点: I
    权重倍率: 1.3

渠道权重:
  return_note: 0.8
  ticket: 0.5
  review: 1.0
  trustpilot: 1.2

语义本体映射:
  英文: ["overpriced", "too expensive", "not worth the price", "cheaper alternatives"]
  西班牙文: ["caro", "sobreprecio", "no vale"]
  法文: ["trop cher", "surfait", "pas worth"]
  德文: ["zu teuer", "überbewertet", "nicht wert"]

画像推导信号:
  - 条件: 命中此标签
    原子标签加分: {price_sensitive: +2}
    推导画像加分: {community_driven: +1}

业务闭环:
  策略包: 定价策略优化包
  主责部门: 财务/定价部
  默认优先级: P3
```

### 6.4 推荐意愿（Recommendation）- 1 标签

```yaml
标签ID: GEN-REC-001
标签名称: 推荐意愿
适用品线: [全部]
适用市场: [US, ES, CA, DE, UK, FR]

默认AIPL节点: L3
AIPL调整规则:
  - 条件: text 包含 "would not recommend" OR "don't recommend"
    节点: L3
    权重倍率: 1.0
    备注: 否定推荐 → Proxy NPS Detractor
  - 条件: text 包含 "highly recommend" OR "definitely recommend"
    节点: L3
    权重倍率: 1.2

渠道权重:
  return_note: 0.0
  ticket: 0.0
  review: 1.2
  trustpilot: 1.5        # 第三方平台推荐意愿可信度最高

语义本体映射:
  英文: ["recommend", "would recommend", "tell friends", "spread the word"]
  西班牙文: ["recomendar", "lo recomiendo"]
  法文: ["recommander", "je recommande"]
  德文: ["empfehlen", "ich empfehle"]

触发规则:
  - NLP置信度: > 0.85
  - 语义模式: [推荐词]
  - 排除模式: ["would not recommend", "don't recommend", "can't recommend", "not recommend"]  # 否定词过滤

画像推导信号:
  - 条件: 正面推荐 + 提及 "friend" OR "sister" OR "mom group"
    原子标签加分: {community_driven: +2}
    推导画像加分: {community_driven: +2}

业务闭环:
  策略包: 口碑传播激励包
  主责部门: 品牌营销部
  默认优先级: P3
```

---

## 七、安全标签（按品线 + 市场监管差异化）

### 7.1 安全标签设计原则

1. **品线差异化**：不同品线的安全风险点完全不同
2. **市场监管差异化**：同一安全风险在不同市场的监管要求不同
3. **P0 强制升级**：任何安全相关标签默认 P0，不受数量阈值限制

### 7.2 按品线的安全标签

#### 吸奶器安全标签

```yaml
标签ID: A-SAF-001
标签名称: 材质过敏反应
适用品线: [吸奶器]
适用市场: [US, ES, CA, DE, UK, FR]
P0触发条件: 任何提及

监管映射:
  美国: CPSIA 邻苯二甲酸盐限制
  欧盟: REACH SVHC 清单
  德国: LFGB 食品接触材料
  英国: UK REACH
  加拿大: CCPSA
  法国: 同欧盟 REACH

语义本体映射:
  英文: ["rash", "allergic", "skin reaction", "nipple burning", "redness"]
  西班牙文: ["erupción", "alérgica", "reacción"]
  法文: ["éruption", "allergique", "rougeur"]
  德文: ["Ausschlag", "allergisch", "Rötung"]
```

#### 内衣服饰安全标签

```yaml
标签ID: B-SAF-001
标签名称: 绳带安全风险
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]
P0触发条件: 儿童服饰 + 颈部/腰部绳带

监管映射:
  美国: CPSC 16 CFR Part 1120 (儿童上衣绳带禁令)
  欧盟: EN 14682 (童装绳索和抽绳安全)
  英国: BS EN 14682
  其他: 同欧盟标准

语义本体映射:
  英文: ["drawstring", "cord", "strangulation risk", "got caught"]
  西班牙文: ["cordón", "riesgo estrangulación"]
  法文: ["cordons", "risque étranglement"]
  德文: ["Kordel", "Strangulationsgefahr"]
```

```yaml
标签ID: B-SAF-002
标签名称: 小部件脱落风险
适用品线: [内衣服饰]
适用市场: [US, ES, CA, DE, UK, FR]
P0触发条件: 任何提及

监管映射:
  美国: CPSC 小部件法规 (16 CFR 1501)
  欧盟: EN 71-1 (小部件测试)
  英国: BS EN 71-1

语义本体映射:
  英文: ["button fell off", "choking hazard", "small parts", "decoration detached"]
  西班牙文: ["botón se cayó", "piezas pequeñas"]
  法文: ["bouton tombé", "petites pièces"]
  德文: ["Knopf abgefallen", "Kleinteile"]
```

#### 喂养电器安全标签

```yaml
标签ID: C-SAF-001
标签名称: 烫伤风险
适用品线: [喂养电器]
适用市场: [US, ES, CA, DE, UK, FR]
P0触发条件: 任何提及

监管映射:
  美国: CPSC 烫伤警告要求
  欧盟: EN 60335-2-15 (液体加热器安全)
  英国: BS EN 60335-2-15

语义本体映射:
  英文: ["burned", "scalding", "too hot", "baby burned", "temperature too high"]
  西班牙文: ["quemadura", "demasiado caliente"]
  法文: ["brûlure", "trop chaud"]
  德文: ["Verbrennung", "zu heiß"]
```

#### 智能母婴电器安全标签

```yaml
标签ID: D-SAF-001
标签名称: 数据隐私泄露
适用品线: [智能母婴电器]
适用市场: [US, ES, CA, DE, UK, FR]
P0触发条件: 任何提及

监管映射:
  美国: COPPA (儿童在线隐私保护法)
  欧盟: GDPR
  英国: UK GDPR
  德国: BDSG (联邦数据保护法)
  加州: CCPA

语义本体映射:
  英文: ["hacked", "unauthorized access", "data breach", "stranger talking", "privacy violation"]
  西班牙文: ["hackeado", "acceso no autorizado", "violación privacidad"]
  法文: ["piraté", "accès non autorisé", "violation vie privée"]
  德文: ["gehackt", "unautorisierter Zugriff", "Datenschutzverletzung"]
```

### 7.3 安全标签跨市场监管矩阵

| 安全标签 | 美国 CPSC | 欧盟 CE/REACH | 英国 UKCA | 加拿大 CCPSA | 德国 LFGB | 法国/西班牙 |
|----------|-----------|---------------|-----------|-------------|-----------|------------|
| 材质过敏 | CPSIA | REACH SVHC | UK REACH | CCPSA | LFGB | 同欧盟 |
| 绳带风险 | 16 CFR 1120 | EN 14682 | BS EN 14682 | 同美国 | 同欧盟 | 同欧盟 |
| 小部件 | 16 CFR 1501 | EN 71-1 | BS EN 71-1 | 同美国 | 同欧盟 | 同欧盟 |
| 烫伤风险 | CPSC 警告 | EN 60335 | BS EN 60335 | 同美国 | 同欧盟 | 同欧盟 |
| 数据隐私 | COPPA/CCPA | GDPR | UK GDPR | PIPEDA | BDSG | GDPR |

---

## 八、渠道权重总矩阵（按品线）

### 8.1 吸奶器品线渠道权重

| 标签组 | 退货留言 | 客服工单 | 商品评论 | Trustpilot |
|--------|----------|----------|----------|------------|
| 吸力性能 | 1.5 | 1.3 | 1.0 | 0.8 |
| 噪音控制 | 1.3 | 1.1 | 1.0 | 0.8 |
| 穿戴舒适 | 1.5 | 1.2 | 1.0 | 0.8 |
| 续航充电 | 1.3 | 1.1 | 1.0 | 0.8 |
| 清洗便捷 | 1.2 | 1.0 | 1.0 | 0.8 |
| 配件相关 | 1.2 | 1.3 | 1.0 | 0.8 |
| 漏奶问题 | 1.5 | 1.3 | 1.0 | 0.8 |
| 智能/App | 1.0 | 1.5 | 1.0 | 0.8 |
| 尺寸法兰 | 1.5 | 1.3 | 1.0 | 0.8 |
| 物流通用 | 0.5 | 1.3 | 1.0 | 1.2 |
| 售后通用 | 0.0 | 1.0 | 1.2 | 1.5 |
| 价格通用 | 0.8 | 0.5 | 1.0 | 1.2 |
| 推荐意愿 | 0.0 | 0.0 | 1.2 | 1.5 |

### 8.2 内衣服饰品线渠道权重

| 标签组 | 退货留言 | 客服工单 | 商品评论 | Trustpilot |
|--------|----------|----------|----------|------------|
| 尺码合身 | **1.5** | 1.2 | 1.0 | 0.8 |
| 支撑承托 | 1.3 | 1.1 | 1.0 | 0.8 |
| 哺乳便利 | 1.3 | 1.2 | 1.0 | 0.8 |
| 面料舒适 | 1.3 | 1.3 | 1.0 | 0.8 |
| 洗后问题 | 1.2 | 1.0 | 1.0 | 0.8 |
| 外观满意 | 1.0 | 0.8 | 1.0 | 0.8 |

**关键差异**：内衣服饰的退货留言权重显著高于其他品线，因为尺码问题是退货的头号原因。

### 8.3 喂养电器品线渠道权重

| 标签组 | 退货留言 | 客服工单 | 商品评论 | Trustpilot |
|--------|----------|----------|----------|------------|
| 加热性能 | 1.5 | 1.3 | 1.0 | 0.8 |
| 操作便捷 | 1.2 | **1.3** | 1.0 | 0.8 |
| 材质安全 | 1.3 | 1.3 | 1.0 | 0.8 |
| 容量效率 | 1.2 | 1.0 | 1.0 | 0.8 |
| 清洗维护 | 1.0 | 1.1 | 1.0 | 0.8 |

**关键差异**：操作便捷性在客服工单中的权重最高（1.3），因为操作问题常通过客服指导解决。

### 8.4 智能母婴电器品线渠道权重

| 标签组 | 退货留言 | 客服工单 | 商品评论 | Trustpilot |
|--------|----------|----------|----------|------------|
| 画质显示 | 1.3 | 1.2 | 1.0 | 0.8 |
| 连接稳定 | 1.3 | **1.5** | 1.0 | 0.8 |
| App体验 | 1.0 | **1.5** | 1.0 | 0.8 |
| 隐私安全 | 1.2 | 1.3 | 1.0 | **1.2** |
| 安装便捷 | 1.2 | **1.5** | 1.0 | 0.8 |

**关键差异**：智能母婴电器的客服工单权重显著高于其他品线，因为技术问题（连接/App/安装）高度依赖客服支持。

---

## 九、AIPL 锚定规则总结

### 9.1 各品线的 AIPL 节点分布特征

| 品线 | 主导 AIPL 节点 | 原因 |
|------|---------------|------|
| **吸奶器** | L1(首购使用) 40% / P1(评估) 30% / I(兴趣) 20% | 功能性极强，使用后才能验证 |
| **内衣服饰** | P1(评估) 35% / L1(首购使用) 35% | 试穿/首穿即决定满意度 |
| **喂养电器** | L1(首购使用) 50% / L2(持续使用) 30% | 购买前评估较少，使用后发现问题 |
| **智能母婴电器** | P1(评估) 40% / L1(首购使用) 35% | 技术参数对比多 + 安装问题集中 |

### 9.2 通用 AIPL 锚定规则

```
# 按数据源类型的默认 AIPL 节点（当 NLP 未匹配到任何标签时）
return_note  →  L1    # 退货意味着已购买使用
review       →  L1    # 评论通常是使用后的反馈
ticket       →  L2    # 客服工单通常是持续使用中的问题
trustpilot   →  L3    # Trustpilot 多为总结性评价（含推荐意愿）

# 按关键词的 AIPL 节点调整
包含 "searching" / "looking for" / "researching"     → A
包含 "compared" / "vs" / "better than" / "worse than" → I
包含 "deciding" / "hesitating" / "not sure"          → P1
包含 "ordered" / "bought" / "purchased"              → P2
包含 "arrived" / "received" / "first time using"     → L1
包含 "after using" / "month later" / "wear and tear" → L2
包含 "recommend" / "would buy again" / "told friends" → L3
```

---

## 十、画像推导信号总结

### 10.1 吸奶器品线画像推导

| 命中标签 | 推导信号 | 原子标签加分 | 推导画像 |
|----------|----------|-------------|----------|
| 噪音太大 + 提及"night" | 夜间用户 + 静音需求 | nighttime_user+2, quiet_seeker+2 | systematic_planner |
| 噪音太大 + 提及"office" | 职场妈妈 + 静音需求 | working_mom+2, quiet_seeker+2 | community_driven |
| 吸力不足 + 提及"supply" | 焦虑型 + 追奶需求 | anxiety_driven+2 | systematic_planner |
| 漏奶 + 提及"work" | 职场妈妈 + 焦虑型 | working_mom+2, anxiety_driven+1 | community_driven |
| 配件贵 | 价格敏感 + 配件依赖 | price_sensitive+2 | community_driven |
| 续航不足 + 提及"travel" | 便携需求 + 外出场景 | portable_needer+2 | community_driven |

### 10.2 内衣服饰品线画像推导

| 命中标签 | 推导信号 | 原子标签加分 | 推导画像 |
|----------|----------|-------------|----------|
| 尺码表不清 | 新手妈妈 + 焦虑型 | first_time_parent+2, anxiety_driven+1 | systematic_planner |
| 面料不舒适 + 过敏 | 敏感型 + 品质追求 | anxiety_driven+2, quality_explorer+1 | quality_explorer |
| 单手操作困难 + 提及"baby" | 新手妈妈 + 焦虑型 | first_time_parent+1, anxiety_driven+1 | systematic_planner |
| 洗后变形 | 高频使用者 + 实用型 | working_mom+1 | quality_explorer |

### 10.3 喂养电器品线画像推导

| 命中标签 | 推导信号 | 原子标签加分 | 推导画像 |
|----------|----------|-------------|----------|
| 温度不准 + 提及"baby burned" | 焦虑型 + 安全敏感 | anxiety_driven+2 | systematic_planner |
| 加热慢 + 提及"3am" | 夜间用户 + 焦虑型 | nighttime_user+2, anxiety_driven+1 | systematic_planner |
| 操作复杂 + 提及"husband" | 多人使用家庭 | first_time_parent+1 | community_driven |
| 加热后异味 | 焦虑型 + 品质追求 | anxiety_driven+2, quality_explorer+1 | quality_explorer |

### 10.4 智能母婴电器品线画像推导

| 命中标签 | 推导信号 | 原子标签加分 | 推导画像 |
|----------|----------|-------------|----------|
| 隐私担忧 | 焦虑型 + 研究型 | anxiety_driven+2, research_driven+1 | systematic_planner |
| 连接不稳定 + 提及"router" | 技术型 + 已排查 | research_driven+1 | systematic_planner |
| 画质差 + 提及"night" | 夜间用户 + 品质追求 | nighttime_user+1, quality_explorer+1 | quality_explorer |

---

## 十一、实施路径

### Phase 1：吸奶器核心标签 v0.2 升级（Week 1-2）

**原因**：吸奶器是最大品线（129 SKU，30%），且 VOC 情感强度最高，业务价值最大。

| 任务 | 工作量 | 产出 |
|------|--------|------|
| 15个吸奶器标签的 AIPL 锚定配置 | 2天 | 吸奶器标签字典 v0.2 |
| 渠道权重矩阵配置 | 1天 | 权重配置文件 |
| 画像推导规则配置 | 2天 | 推导规则库 |
| 多语言语义映射完善 | 3天 | 4语言关键词表 |
| 业务闭环（策略包/部门/优先级） | 2天 | 业务路由规则 |

### Phase 2：内衣服饰标签 v0.2 升级（Week 3）

| 任务 | 工作量 | 产出 |
|------|--------|------|
| 9个内衣服饰标签的 v0.2 配置 | 2天 | 内衣服饰标签字典 |
| 尺码相关标签的渠道权重调优 | 1天 | 尺码权重专项配置 |

### Phase 3：喂养电器 + 智能母婴电器（Week 4）

| 任务 | 工作量 | 产出 |
|------|--------|------|
| 7个喂养电器标签 v0.2 配置 | 2天 | 喂养电器标签字典 |
| 7个智能母婴电器标签 v0.2 配置 | 2天 | 智能母婴电器标签字典 |
| 安全标签的品线差异化配置 | 1天 | 安全标签监管映射表 |

### Phase 4：通用标签 + 整合测试（Week 5）

| 任务 | 工作量 | 产出 |
|------|--------|------|
| 11个通用标签的跨品线配置 | 2天 | 通用标签字典 |
| 统一萃取引擎的品线过滤接入 | 2天 | 品线过滤模块 |
| 端到端测试（吸奶器品线试点） | 1天 | 测试报告 |

---

## 十二、版本记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v0.1 | 2026-04-23 | 基础版，7维度通用标签体系 |
| v0.2 | 2026-04-23 | 按品线精细化版，4大品线56个核心标签，含AIPL锚定、渠道权重、画像推导 |

---

**文档版本**: v0.2  
**创建日期**: 2026-04-23  
**覆盖SKU**: 431  
**核心标签数**: 56（品线专属45 + 通用11）  
**覆盖市场**: 美/西/加/德/英/法
