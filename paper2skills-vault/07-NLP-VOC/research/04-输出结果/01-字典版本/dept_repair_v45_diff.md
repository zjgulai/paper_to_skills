# tag_dictionary v4.4 → v4.5 修复 diff 报告

- 生成时间：2026-05-14 18:22:54
- 源文件：tag_dictionary_v4.4.xlsx
- 目标文件：tag_dictionary_v4.5.xlsx

## 修复汇总

| 修复类型 | 总数 |
|---|---:|
| 业务动作/责任部门 | 140 |
| 策略包 | 86 |
| cleanup_tag_cn | 56 |
| tag_cn | 2 |
| 故事线关联 | 2 |
| 主责部门 | 2 |

## 分 Sheet 修复明细

| Sheet | 修复类型 | 行数 |
|---|---|---:|
| 01_通用标签主表 | tag_cn | 2 |
| 01_通用标签主表 | 主责部门 | 2 |
| 01_通用标签主表 | 故事线关联 | 2 |
| 01_通用标签主表 | 策略包 | 2 |
| 02_吸奶器 | cleanup_tag_cn | 4 |
| 02_吸奶器 | 业务动作/责任部门 | 14 |
| 02_吸奶器 | 策略包 | 10 |
| 03_内衣服饰 | cleanup_tag_cn | 6 |
| 03_内衣服饰 | 业务动作/责任部门 | 16 |
| 03_内衣服饰 | 策略包 | 10 |
| 04_家居家纺 | cleanup_tag_cn | 6 |
| 04_家居家纺 | 业务动作/责任部门 | 17 |
| 04_家居家纺 | 策略包 | 11 |
| 05_母婴综合护理 | cleanup_tag_cn | 12 |
| 05_母婴综合护理 | 业务动作/责任部门 | 31 |
| 05_母婴综合护理 | 策略包 | 19 |
| 06_喂养电器 | cleanup_tag_cn | 7 |
| 06_喂养电器 | 业务动作/责任部门 | 14 |
| 06_喂养电器 | 策略包 | 7 |
| 07_智能母婴电器 | cleanup_tag_cn | 21 |
| 07_智能母婴电器 | 业务动作/责任部门 | 48 |
| 07_智能母婴电器 | 策略包 | 27 |

## 修复样例（前 30 条）

### 1. 02_吸奶器 row 70 · cleanup_tag_cn
- tag_id: `TAG_NEW_001`
- tag_cn: `吸奶器`
- 原值: `'[吸奶器] breast pump'`
- 新值: `吸奶器`

### 2. 02_吸奶器 row 71 · cleanup_tag_cn
- tag_id: `TAG_NEW_002`
- tag_cn: `吸奶器`
- 原值: `'[吸奶器] air ultra'`
- 新值: `吸奶器`

### 3. 02_吸奶器 row 72 · cleanup_tag_cn
- tag_id: `TAG_NEW_003`
- tag_cn: `吸奶器`
- 原值: `'[吸奶器] ultra slim'`
- 新值: `吸奶器`

### 4. 02_吸奶器 row 73 · cleanup_tag_cn
- tag_id: `TAG_NEW_004`
- tag_cn: `吸奶器`
- 原值: `'[吸奶器] slim breast'`
- 新值: `吸奶器`

### 5. 03_内衣服饰 row 43 · cleanup_tag_cn
- tag_id: `TAG_NEW_005`
- tag_cn: `收腹带`
- 原值: `'[收腹带] postpartum belly'`
- 新值: `收腹带`

### 6. 03_内衣服饰 row 44 · cleanup_tag_cn
- tag_id: `TAG_NEW_006`
- tag_cn: `托腹带`
- 原值: `'[托腹带] belly band'`
- 新值: `托腹带`

### 7. 03_内衣服饰 row 45 · cleanup_tag_cn
- tag_id: `TAG_NEW_007`
- tag_cn: `收腹带`
- 原值: `'[收腹带] belly wrap'`
- 新值: `收腹带`

### 8. 03_内衣服饰 row 46 · cleanup_tag_cn
- tag_id: `TAG_NEW_008`
- tag_cn: `哺乳背心`
- 原值: `'[哺乳背心] tank top'`
- 新值: `哺乳背心`

### 9. 03_内衣服饰 row 47 · cleanup_tag_cn
- tag_id: `TAG_NEW_009`
- tag_cn: `哺乳背心`
- 原值: `'[哺乳背心] nursing tank'`
- 新值: `哺乳背心`

### 10. 03_内衣服饰 row 48 · cleanup_tag_cn
- tag_id: `TAG_NEW_010`
- tag_cn: `收腹带`
- 原值: `'[收腹带] ergowrap postpartum'`
- 新值: `收腹带`

### 11. 04_家居家纺 row 37 · cleanup_tag_cn
- tag_id: `TAG_NEW_011`
- tag_cn: `夹腿枕`
- 原值: `'[夹腿枕] knee pillow'`
- 新值: `夹腿枕`

### 12. 04_家居家纺 row 38 · cleanup_tag_cn
- tag_id: `TAG_NEW_012`
- tag_cn: `U型哺乳枕`
- 原值: `'[U型哺乳枕] nursing pillow'`
- 新值: `U型哺乳枕`

### 13. 04_家居家纺 row 39 · cleanup_tag_cn
- tag_id: `TAG_NEW_014`
- tag_cn: `记忆棉哺乳枕`
- 原值: `'[记忆棉哺乳枕] memory foam'`
- 新值: `记忆棉哺乳枕`

### 14. 04_家居家纺 row 40 · cleanup_tag_cn
- tag_id: `TAG_NEW_015`
- tag_cn: `腰凳背带`
- 原值: `'[腰凳背带] baby carrier'`
- 新值: `腰凳背带`

### 15. 04_家居家纺 row 41 · cleanup_tag_cn
- tag_id: `TAG_NEW_018`
- tag_cn: `腰凳背带`
- 原值: `'[腰凳背带] hip seat'`
- 新值: `腰凳背带`

### 16. 04_家居家纺 row 42 · cleanup_tag_cn
- tag_id: `TAG_NEW_019`
- tag_cn: `W型孕妇枕`
- 原值: `'[W型孕妇枕] pregnancy pillow'`
- 新值: `W型孕妇枕`

### 17. 05_母婴综合护理 row 41 · cleanup_tag_cn
- tag_id: `TAG_NEW_022`
- tag_cn: `沐浴炸弹`
- 原值: `'[沐浴炸弹] bath bombs'`
- 新值: `沐浴炸弹`

### 18. 05_母婴综合护理 row 42 · cleanup_tag_cn
- tag_id: `TAG_NEW_023`
- tag_cn: `应急垫`
- 原值: `'[应急垫] play mat'`
- 新值: `应急垫`

### 19. 05_母婴综合护理 row 43 · cleanup_tag_cn
- tag_id: `TAG_NEW_024`
- tag_cn: `婴儿无袖睡袋`
- 原值: `'[婴儿无袖睡袋] sleeping bag'`
- 新值: `婴儿无袖睡袋`

### 20. 05_母婴综合护理 row 44 · cleanup_tag_cn
- tag_id: `TAG_NEW_025`
- tag_cn: `一次性防溢乳垫`
- 原值: `'[一次性防溢乳垫] nursing pads'`
- 新值: `一次性防溢乳垫`

### 21. 05_母婴综合护理 row 45 · cleanup_tag_cn
- tag_id: `TAG_NEW_026`
- tag_cn: `沐浴炸弹`
- 原值: `'[沐浴炸弹] bath bomb'`
- 新值: `沐浴炸弹`

### 22. 05_母婴综合护理 row 46 · cleanup_tag_cn
- tag_id: `TAG_NEW_027`
- tag_cn: `应急垫`
- 原值: `'[应急垫] non slip'`
- 新值: `应急垫`

### 23. 05_母婴综合护理 row 47 · cleanup_tag_cn
- tag_id: `TAG_NEW_029`
- tag_cn: `应急垫`
- 原值: `'[应急垫] baby play'`
- 新值: `应急垫`

### 24. 05_母婴综合护理 row 48 · cleanup_tag_cn
- tag_id: `TAG_NEW_031`
- tag_cn: `沐浴片`
- 原值: `'[沐浴片] individually wrapped'`
- 新值: `沐浴片`

### 25. 05_母婴综合护理 row 49 · cleanup_tag_cn
- tag_id: `TAG_NEW_032`
- tag_cn: `可水洗防溢乳垫`
- 原值: `'[可水洗防溢乳垫] breast pads'`
- 新值: `可水洗防溢乳垫`

### 26. 05_母婴综合护理 row 50 · cleanup_tag_cn
- tag_id: `TAG_NEW_034`
- tag_cn: `妈咪包`
- 原值: `'[妈咪包] diaper bag'`
- 新值: `妈咪包`

### 27. 05_母婴综合护理 row 51 · cleanup_tag_cn
- tag_id: `TAG_NEW_037`
- tag_cn: `防挤压支架`
- 原值: `'[防挤压支架] pouch holder'`
- 新值: `防挤压支架`

### 28. 05_母婴综合护理 row 52 · cleanup_tag_cn
- tag_id: `TAG_NEW_040`
- tag_cn: `吸奶器背心`
- 原值: `'[吸奶器背心] soutien gorge'`
- 新值: `吸奶器背心`

### 29. 06_喂养电器 row 41 · cleanup_tag_cn
- tag_id: `TAG_NEW_041`
- tag_cn: `湿巾加热盒`
- 原值: `'[湿巾加热盒] wipe warmer'`
- 新值: `湿巾加热盒`

### 30. 06_喂养电器 row 42 · cleanup_tag_cn
- tag_id: `TAG_NEW_042`
- tag_cn: `奶瓶沥干支架`
- 原值: `'[奶瓶沥干支架] drying rack'`
- 新值: `奶瓶沥干支架`

## 增量修复：08_映射关系表（10 行批量回填）

修复方法：以 `VOC标签（中文）` 为键，从 `01_通用标签主表` 反查回填。

| Sheet | 修复类型 | 行数 |
|---|---|---:|
| 08_映射关系表 | strategy_package | 10 |
| 08_映射关系表 | 主责部门 | 10 |
| 08_映射关系表 | 协同部门 | 10 |
| 08_映射关系表 | 默认优先级 | 10 |
| 08_映射关系表 | 故事线关联 | 10 |
| 08_映射关系表 | 是否Promoter信号（由情感极性推导） | 10 |
| 08_映射关系表 | 是否Detractor信号（由情感极性推导） | 10 |
| 08_映射关系表 | 可识别性/归因清晰度/故事线支撑度/业务动作明确度（默认「中」）| 4 × 10 |

修复涉及 10 行：MAP_0393 - MAP_0402（即 TAG_SRV_01-10 在映射表中的镜像）。

| 映射ID | tag_cn | 主责部门 |
|---|---|---|
| MAP_0393 | 下单/购买 | 电商运营部 |
| MAP_0394 | 支付/账单问题 | 电商运营部 |
| MAP_0395 | 物流/配送 | 仓储物流部 |
| MAP_0396 | 配送问题 | 仓储物流部 |
| MAP_0397 | 退货/换货 | 全球客服中心 |
| MAP_0398 | 退款请求 | 全球客服中心 |
| MAP_0399 | 质保/维修/换新 | 全球客服中心 |
| MAP_0400 | 客服体验 | 全球客服中心 |
| MAP_0401 | 产品咨询/使用指导 | 全球客服中心 |
| MAP_0402 | 一般反馈/感谢 | 全球客服中心 |

## 终版完整性审计（v4.5）

| Sheet | 行数 | 关键字段数 | 空值 | 占位符 | 脏数据 | 状态 |
|---|---:|---:|---:|---:|---:|:---:|
| 01_通用标签主表 | 267 | 8 | 0 | 0 | 0 | ✅ |
| 02_吸奶器 | 82 | 9 | 0 | 0 | 0 | ✅ |
| 03_内衣服饰 | 57 | 9 | 0 | 0 | 0 | ✅ |
| 04_家居家纺 | 52 | 9 | 0 | 0 | 0 | ✅ |
| 05_母婴综合护理 | 70 | 9 | 0 | 0 | 0 | ✅ |
| 06_喂养电器 | 53 | 9 | 0 | 0 | 0 | ✅ |
| 07_智能母婴电器 | 64 | 9 | 0 | 0 | 0 | ✅ |
| 08_映射关系表 | 402 | 8 | 0 | 0 | 0 | ✅ |
| **合计** | **1,047** | — | **0** | **0** | **0** | **✅ 全部 PASS** |

## 已知质量问题（留待 v5.0 主版本优化）

- **业务动作多样性偏低**：6 品线 sheet 中业务动作 unique 比例 28%-80% 不均。LLM 在批量推断时对相似语义标签倾向给同模板。最严重案例：117 行重复「产品中心：围绕"产品核心性能"主题做专项优化和闭环」。
- **策略包分布过度集中**：「核心体验改良包」占 328/645 (51%)，未充分利用 49 闭集词表。
- **业务侧建议**：基于本表生成部门工单时，对 unique business_action 做去重 + 人工 review，避免向各部门下发同质化任务。

## 修复脚本

[repair_dict_v45.py](../../02-脚本工具/01-标签进化/scripts/repair_dict_v45.py) - 支持 sample / spot / batch 三模式，可复用于 v4.6 及后续版本。

## 09_存量标签归档（本次不修）

- 客服组别 181/457 空（39.6%）—— 历史数据无源
- 是否使用中 276/457 空（60.4%）—— 历史数据无源

09 sheet 定位为「v3.0 → v3.9 转录的归档参考」，不影响当前生产，本次未修。如需修复需要原始客服系统数据回溯。
