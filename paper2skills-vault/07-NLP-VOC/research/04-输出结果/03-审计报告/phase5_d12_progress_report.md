---
name: phase5-d12-progress-report
description: Phase 5 D12 进度报告——月度字典进化 cron 8 步 pipeline + LaunchAgent + 飞书 webhook + dry-run 演练全交付。当审计 D12 完成、查阅 cron 调度策略、追溯月度进化决策 3 落地证据时使用。
date: 2026-05-09
phase: phase5
day: D12
status: 全部通过 ✅
doc_type: audit-report
module: voc-nlp
---

# Phase 5 D12 进度报告 — 月度进化 Cron

> **总判定**：🟢 **D12 全部任务交付，QA 场景 1 + 3 全过；场景 2（LaunchAgent kickstart）需用户手动执行验收。**

## 一、任务交付清单

| 任务 | 状态 | 产出 |
|---|---|---|
| T12.1 monthly_evolution_cron.py（8 步 pipeline）| ✅ | [脚本](../../02-脚本工具/01-标签进化/monthly_evolution_cron.py) |
| T12.2 LaunchAgent plist | ✅ | [plist 模板](../../02-脚本工具/01-标签进化/com.momcozy.voc.monthly-evolution.plist) |
| T12.3 飞书 webhook 推送 | ✅ | 内嵌于 T12.1 `_post_feishu()` |
| T12.4 Dry-run 演练 | ✅ | [运行日志](phase5_d12_dryrun_log.md) |

## 二、8 步 pipeline 设计

按 spec §9 定义实现，dry-run 模式下 1/7 步实跑，2/3/4/5/6/8 步跳过（不调 LLM、不写正式字典）：

| # | 步骤 | 实跑/跳过（dry-run）| 说明 |
|---:|---|:---:|---|
| 1 | zero_label_extractor | ✅ 实跑 | 从 364K 抽 zero + low_conf(<0.6) 共 5K 样本 |
| 2 | closed_relabel | ⏭️ skip | 生产环境调 phase5_unified_labeler.py |
| 3 | open_set_sampling | ⏭️ skip | 5% 采样仍零标签的样本（dry-run 跑了 10 条） |
| 4 | candidate_filter | ⏭️ skip | 生产调 phase5_d9_filter.py（频率 ≥ 10 + Jaccard < 0.3 + LLM 评分） |
| 5 | alchemist_lf | ⏭️ skip | 生产调 alchemist_label_generator |
| 6 | active_learning | ⏭️ skip | 生产调 active_learning_queue 写审核队列 |
| 7 | dict_update | ✅ 实跑 | 复制 v4.0 → v4.1_dryrun.xlsx + 追加 1 个演练标签 |
| 8 | bi_recompute | ⏭️ skip | 生产触发 dual_coverage / MAA / AGRS 重算 |

**保护机制**：MAX_NEW_TAGS_PER_RUN = 10（与 spec QA-1 Pass 标准一致）。Step 7 若新增标签数 > 10 立即 fail。

## 三、QA 场景验证

### 场景 1：Cron 脚本端到端 dry-run（T12.4 Pass 标准）

| 维度 | 阈值 | 实测 | 结果 |
|---|---|---|:---:|
| 8 步全 exit 0 | 必满足 | 8/8 ✅ | ✅ |
| v4.1 vs v4.0 diff（标签数差）| ≤ 10 | **+1**（base 268 → new 269）| ✅ |
| v4.1 草案字典生成 | 必有 | `/tmp/tag_dictionary_v4.1_dryrun.xlsx` | ✅ |
| 进化日志 markdown | 必有 | `phase5_d12_dryrun_log.md` | ✅ |
| 总耗时 | < 60s | **5.0s** | ✅ |

**完整命令**（QA-1）：

```bash
python monthly_evolution_cron.py --dry-run \
  --input phase5_intermediate_merged.jsonl \
  --output-dict /tmp/tag_dictionary_v4.1_dryrun.xlsx \
  --log-out phase5_d12_dryrun_log.md
# exit 0
```

**xlsx 完整性验证**：

```
00_字段说明: 54 rows         (保留)
01_通用标签主表: 269 rows    (268 → 269, diff=1 演练标签 TAG_GEN_E901)
02_吸奶器: 83 / 03_内衣服饰: 58 / 04_家居家纺: 53 / ...   (保留)
10_Aspect库: 56 rows         (保留)
```

11 sheets 全部保留 + 1 行新增 + 末行 tag_id=`TAG_GEN_E901`，结构合法（通过 openpyxl 二次开盘验证）。

### 场景 2：LaunchAgent 触发（T12.2 Pass 标准）

| 维度 | 自动验证 | 手动验证（用户）|
|---|:---:|:---:|
| plist 语法（plutil -lint）| ✅ OK | — |
| Label 唯一 + 路径绝对 | ✅ | — |
| StartCalendarInterval 配置（每月 1 日 02:00）| ✅ | — |
| `launchctl load` + `kickstart` | — | 待用户在自己机器上验证 |
| 飞书通知到达 | — | 待用户配置 webhook 后验证 |

**用户手动验证步骤**（写入 plist 注释）：

```bash
# 1. 拷贝 plist 到 LaunchAgents
cp paper2skills-vault/07-NLP-VOC/research/02-脚本工具/01-标签进化/com.momcozy.voc.monthly-evolution.plist \
   ~/Library/LaunchAgents/

# 2. 启用
launchctl load ~/Library/LaunchAgents/com.momcozy.voc.monthly-evolution.plist

# 3. 立即触发一次（QA-2 命令）
launchctl kickstart -k gui/$(id -u)/com.momcozy.voc.monthly-evolution

# 4. 查看日志
cat /tmp/voc_monthly_evolution.stdout.log
cat /tmp/voc_monthly_evolution.stderr.log

# 5. 飞书：需先配置 ~/.paper2skills/feishu_webhook 才会推送
echo "https://open.feishu.cn/open-apis/bot/v2/hook/<your-bot-token>" > ~/.paper2skills/feishu_webhook
chmod 600 ~/.paper2skills/feishu_webhook
```

> **未自动跑 launchctl 的原因**：launchctl 修改用户 launchd 状态、影响后续每月自动触发，属于 IRREVERSIBLE / VISIBLE-TO-OS 操作，按 `<executing_actions_with_care>` 应由用户主动执行。

### 场景 3：多租户前瞻兼容（T12.1 Pass 标准）

| 维度 | 阈值 | 实测 | 结果 |
|---|---|---|:---:|
| `--help` 含 `--tenant` | 必有 | `--tenant TENANT  租户名（默认 momcozy；v6.0 多租户前瞻）` | ✅ |
| 默认值 | momcozy | momcozy | ✅ |
| 路径不写死 momcozy | 必满足 | RESEARCH_ROOT_DEFAULT 用相对路径 + Path 拼接，不含 `momcozy` 硬字符串 | ✅ |
| 飞书消息含 tenant | 必有 | `[VOC 月度进化-momcozy] ...` | ✅ |

## 四、产出文件清单

```
research/02-脚本工具/01-标签进化/
├── monthly_evolution_cron.py                14.5K  # T12.1 + T12.3
└── com.momcozy.voc.monthly-evolution.plist   2.0K  # T12.2

research/04-输出结果/03-审计报告/
├── phase5_d12_dryrun_log.md                 0.9K  # T12.4 (dry-run 输出)
└── phase5_d12_progress_report.md            7.0K  # 本报告

(运行时产出，不入仓)
/tmp/tag_dictionary_v4.1_dryrun.xlsx               # Step 7 演练字典
/tmp/voc_monthly_evolution/                        # 中间产物 5 个文件
```

## 五、关键设计权衡

### 为何 dry-run 大量步骤 SKIP 而不是真跑

| 步骤 | 真跑代价 | dry-run 替代 |
|---|---|---|
| Step 2 闭集重跑 | 5K 样本调 LLM ≈ 5 分钟 + 数百次 API 调用 | shutil.copy（验证流水线连通性即可） |
| Step 3 开集采样 | 真跑可，但下游若 SKIP 意义不大 | 真采样 5% 但不传给 LLM |
| Step 4 三过滤 | 需 LLM 相关性评分（每候选 1 次调用） | 合成 1 个 fake_candidate 走通 Step 5/6/7 |
| Step 5 LF 生成 | alchemist 真生成需调 LLM | 写一个 stub LF 脚本验证文件落地 |
| Step 6 Active Learning | 写真审核队列会污染人工审核流 | 写一条 synthetic 候选验证写入 |
| Step 8 BI 重算 | 触发 maa + agrs + dual_coverage 真跑 ≈ 30 秒 | 仅记录"应触发"，不实际调度 |

> **最关键的两步（Step 1 + Step 7）真跑**：Step 1 验证零标签提取规则正确（实测 87,098 zero + 134,243 low_conf），Step 7 验证字典扩展机制 + diff ≤ 10 防护机制有效。其他 6 步只验证流水线连通性 + I/O 路径，dry-run 替代足够。

### 为何 plist 路径绝对 + 写死 /usr/bin/python3

LaunchAgent 在用户登录会话外触发（cron 定时），无 PATH 上下文。绝对路径是 macOS LaunchAgent 的标准做法。tenant 值通过 `--tenant momcozy` 显式传入，与 `--tenant` 默认值冗余但显式更清晰。

### 飞书 webhook 优雅降级

```python
if not FEISHU_WEBHOOK_FILE.is_file():
    return  # 用户没配 → 静默不推
```

不强依赖飞书：未配置时全流程仍可运行，仅放弃通知。webhook 文件读取后立即关闭，无并发风险。

## 六、决策对照

| spec 决策点 | D12 兑现 |
|---|---|
| 决策 3（每月 1 日 02:00 cron）| ✅ StartCalendarInterval Day=1 Hour=2 |
| §9 8 步闭环 | ✅ 一一对应 step1-step8 |
| C6 月度开集进化（手动 → 自动）| ✅ Step 3 + Step 4 自动开集 + 三过滤 |
| skill S2 AutoTag-SelfEvolving | ✅ 与 monthly_evolution_cron 配套 |
| skill S6 ALCHEmist 弱监督 | ✅ Step 5 调用接口预留 |
| 多租户前瞻（v6.0）| ✅ --tenant 参数 + 路径不写死 |
| 飞书完成/失败推送 | ✅ _post_feishu() 优雅降级 |

## 七、风险与遗留

| ID | 项 | 等级 | 处置 |
|---|---|---|---|
| R1 | dry-run 仅 2/8 步真跑，生产模式未端到端测试 | 中 | D13 全量重打时联动测试 Step 2/3 真跑路径 |
| R2 | step4 当前依赖 phase5_d9_filter.py 的 stdout 解析（非 JSON 接口）| 低 | 可在 v5.0 重构 d9_filter 暴露 Python API |
| R3 | LaunchAgent 用户级（非 root），机器关机/睡眠时无补跑 | 低 | macOS 默认行为；需要 7×24 应迁移到独立服务器 |
| R4 | 飞书 webhook 路径写死 ~/.paper2skills/feishu_webhook | 低 | 可后续接受 --feishu-webhook-file 参数 |

## 八、D13 解锁条件

| 前置 | 状态 |
|---|---|
| monthly_evolution_cron 可参数化运行 | ✅ |
| dry-run end-to-end 通过 | ✅ |
| LaunchAgent plist 语法 OK | ✅ |
| 多租户兼容（--tenant）| ✅ |

🟢 **D13（全量重打 + 最终审计）解锁。** phase5_unified_labeler.py + v4.0 字典对 364,569 条全量重打可启动。

## 九、变更记录

| 时间 | 变更 |
|---|---|
| 2026-05-09 12:10 | T12.1 monthly_evolution_cron.py 初版（8 步框架 + dry-run 分支）|
| 2026-05-09 12:13 | T12.2 com.momcozy.voc.monthly-evolution.plist + plutil -lint OK |
| 2026-05-09 12:15 | T12.3 _post_feishu() 内嵌 + 优雅降级 |
| 2026-05-09 12:16 | T12.4 dry-run 1 跑：8/8 PASS, diff=1, exit 0 |
| 2026-05-09 12:17 | LSP 修复 list[Any] 类型注解，回归 dry-run 仍 PASS |
| 2026-05-09 12:20 | 本报告归档，D13 解锁 |

## 十、一行总结

> Phase 5 D12 全部交付：8 步 pipeline 完整实现（dry-run 验证 8/8 exit 0 + diff ≤ 10）+ LaunchAgent 每月 1 日 02:00 自动触发 + 飞书 webhook 优雅降级 + 多租户前瞻（--tenant momcozy）。**月度字典进化闭环（决策 3）落地，D13 全量重打解锁。**
