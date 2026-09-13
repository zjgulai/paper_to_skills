# dsh-overseas-skills · 产品形态与接入缝调查报告

> 调查对象：`/Users/lute/project/Magpie-Horch/packages/capabilities/dsh-overseas-skills`
> 调查时间：本次会话（下述所有数字均为**现场实测**，非文档转述；凡与文档不符处已逐条标注）
> 调查方法：读源码 + 跑 `node --test`（32/32 通过）+ 起真实 `buildOrgTree` 打数 + 探活本机 `http://127.0.0.1:43120` 两条路由 + 读 `~/.dsh/profiles/desktop/package.json` 挂载实况

---

## 0. 一句话结论

这不是一个「技能列表页」，而是一个**组织归位可视化器**：它把一个扁平的技能目录，投影到《AI组织变革》材料的
组织骨架上（4 面 / 8 责任域 / 50 岗位），并且**刻意把「归位（技能属于谁）」与「接线（preset 真挂了谁）」分成两件事显示**。
论文衍生卡（`p2s-*`，1338 张）**已经有自己的兄弟页面**（`packages/surfaces/dsh-algo-skills-local`），
复用的是**同一套组织骨架与同一份层头像字节**——这决定了新内容管线应该接在哪里（见 §E）。

---

## A. 包体解剖

### A.1 `package.json`

| 字段 | 值 |
| --- | --- |
| `name` | `dsh-overseas-skills` |
| `version` | `0.1.0` |
| `type` | `module`（ESM） |
| `main` | `lib/index.js` |
| `exports` | `"."` → `./lib/index.js`；`"./client"` → `./lib/client.js`；`"./package.json"` |
| `luteOrigin` / `luteOwner` / `lutePublish` | `self` / `lute` / `false`（本仓自研，不外发） |
| `license` | MIT |
| **dependencies** | **无运行时依赖**（`dependencies` 字段不存在） |
| `devDependencies` | `@types/node@^22.20.2`, `typescript@5.6.3`, `react@18.3.1`, `@types/react@18.3.12` |
| `scripts.typecheck` | `tsc -p tsconfig.json --pretty false` |
| `scripts.test` | `node --test test/*.spec.mjs` |
| `scripts.icons` | `node scripts/gen-layer-icons.mjs` |
| `scripts.role-map` | `python3 scripts/build_role_map.py` |

`dsh` 块（挂载声明）：

```json
"dsh": {
  "bundle":  { "patch": "./cordis.patch.yml" },
  "client":  { "platform": "web",
               "inject": ["@deepseek-ai/dsh-client-runtime",
                          "@deepseek-ai/dsh-client-locale",
                          "@deepseek-ai/dsh-client-ui-slots"] }
}
```

**关键点：`dsh.bundle` 声明 + `cordis.patch.yml` 两件套齐备**，这是 LUTE 平台 §1.5 的防白屏硬规则（缺一条新装 bundle 会触发启动白屏）。

`files` 白名单（发布时真正打包的 10 项）——**这是一个重要的接入约束**：

```
lib/index.js  lib/catalog.js  lib/client.js  cordis.patch.yml
lib/templates.js  lib/layer-icons.js  lib/role-map.js  lib/org-tree.js
lib/preset-roles.js  manifest/role-assignments.json
```

注意 `manifest/` 目录下**只有 `role-assignments.json` 一个文件**在白名单里。
`lib/catalog.js` 之所以存在，正是因为宿主是已安装的 npm 包、**运行期读不到 `manifest/skills.json`**
——这条理由写在 `scripts/build_role_map.py` 的模块 docstring 里（第 5–7 行）：

> 宿主是已安装的 npm 包，`manifest/` 不在 `package.json` 的 `files` 白名单里（与 `lib/catalog.js`
> 由 manifest 生成同理）。manifest 是**事实的家**（可读、可 diff、可复核），lib 里这份是**构建产物**。

### A.2 目录布局（实测体量：42 MB）

| 目录 | 体量 | 内容 |
| --- | --- | --- |
| `lib/` | 1.5 MB | 9 个源文件 + 5 个 `*.bak-*` / `*.orig-*` / `*.pre-*` 备份 |
| `manifest/` | 1.2 MB | 11 个 JSON + 3 个 `.bak-lieflat-1789205601` |
| `scripts/` | 320 K | **26 个 `.py`/`.mjs`/`.sh`** + 若干 mapping JSON + `__pycache__/` |
| `presets/` | 8 K | `preset-skills.json`、`workflows.json` |
| `eval/` | 220 K | 4 份路由基准 + 3 份 LLM picks + `runs/`（15 次历史） |
| `test/` | 44 K | 6 个 `*.spec.mjs` |
| `staging/` | **12 MB** | `81-skills/`（82 个目录）、`third-party/simplify-codebase/`、`icon-preview/`（2×561 KB HTML）、`translations/`（30 份 `.body.md`）、4 份 import report |
| `docs/` | 268 K | 26 篇 `.md` + `adr/`（0001–0008）+ `momcozy-geo/`（9 个 HTML/JSON/CSV） |
| `backup/` | 436 K | `pre-81/`（**23 个技能目录**，每个一份 `SKILL.md`） |

`lib/` 逐文件职责（无 `.bak` 的行）：

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `lib/index.js` | 387 | **Host**：注册 6 条 HTTP 路由 + loopback 栅栏 |
| `lib/client.js` | 912 | **Client bundle**：`settings.section` 页面 + 输入框胶囊（`ovp*`），全 `dsw-*` 语义 Token |
| `lib/catalog.js` | 1 行（585 KB） | **构建产物**：`CATEGORIES`/`SKILLS`/`CATEGORIES_FS`/`SKILLS_FS` 四个常量，由 `scripts/build_preset_catalog.py` 生成 |
| `lib/role-map.js` | 270 | **构建产物**：`ROLE_ASSIGNMENTS` + `ROLE_ASSIGNMENT_META`，由 `scripts/build_role_map.py` 生成（`--check` 供门禁） |
| `lib/org-tree.js` | 217 | **纯函数**：四层树构建 + 三态判定（`buildOrgTree` / `wiringStatus`），**无 IO** |
| `lib/preset-roles.js` | 63 | **有 IO**：读 `~/.dsh/.agent-presets/agt-*/manifest.json` 的 `x_lute` |
| `lib/templates.js` | 242 | 提示词模板三层引擎（L1 人工 30 条 → L2 解析 SKILL.md「## 输入」 → L3 通用） |
| `lib/layer-icons.js` | 52 | **构建产物**：12 枚层头像 data URI（4 面 + 8 责任域） |
| `lib/host-util.js` | 101 | `errorMessage` / `isValidSkillName` / `rebuildFrontmatter` |
| `lib/globals.d.ts` | — | 类型声明 |

`lib/` 里还躺着 5 个历史副本：`catalog.js.bak-a2`（624 KB）、`client.js.bak-cn-slash`、`client.js.bak-contract`、
`client.js.orig-skillbar`、`client.js.pre-fullstack-20260907`、`templates.js.bak-contract`（见 §F 风险 R5）。

### A.3 挂载方式与**路由路径（精确）**

**`cordis.patch.yml`（全文 4 行）**：

```yaml
- insert:
    - id: dsh-overseas-skills
      name: dsh-overseas-skills
```

插件 id 与包名同名（`dsh-overseas-skills`），在 profile 的 composition 树内唯一。

**profile 实况**（`~/.dsh/profiles/desktop/package.json`，实测）：

```json
"dependencies": { "dsh-overseas-skills": "file:/Users/lute/project/Magpie-Horch/packages/capabilities/dsh-overseas-skills", ... }
"dsh.profile.bundles": [ ..., "dsh-overseas-skills", "dsh-algo-skills-local", "dsh-preset-lint-local", "dsh-overseas-tools", ... ]
```

**注册的路由（`lib/index.js` 第 24 行 `const BASE = "/api/dsh-overseas-skills"`）——六条，全部 `kind: "exact"`**：

| # | 路径 | 方法 | 处理函数 | 用途 |
| --- | --- | --- | --- | --- |
| 1 | `/api/dsh-overseas-skills/list` | GET | `handleList` | 场景两级 + 细分扁平视图（222/223 条，供胶囊每 2 s 轮询） |
| 2 | `/api/dsh-overseas-skills/fullstack-list` | GET | `handleFullstackList` | AI 全栈 30 条（第二页共用组件） |
| 3 | `/api/dsh-overseas-skills/org` | GET | `buildOrgPayload` | **四层树负载**（唯一承载 `tree` 的路由），30 s 缓存 |
| 4 | `/api/dsh-overseas-skills/toggle` | POST | `handleToggle` | 写 `SKILL.md` frontmatter 的 `disable-model-invocation` |
| 5 | `/api/dsh-overseas-skills/prompt-template` | GET | `getPromptTemplate` | `?name=&title=` → 结构化引导模板 |
| 6 | `/api/dsh-overseas-skills/credential` | GET/POST | `handleCredentialDescribe` / `handleCredentialSet` | 外部工具凭据（`overseas_exa` / `overseas_jungle_scout` / `overseas_klaviyo` 白名单） |

**信任栅栏（每条路由第一件事）**：`isLoopbackRequest(req)` —— 要求 **socket 的 `remoteAddress` 是回环**
**并且** `Host` 头解析后也是回环主机名；`X-Forwarded-For` **永不采信**。不满足 → `401 {"error":"unauthorized"}`；
方法不符 → `405`。凭据值**落库后绝不回显**（POST 只回 `{ok, ref, configured}`）。

**缓存语义**：`ORG_TTL_MS = 30_000`（模块级 `orgCache`）。理由写在源码注释里——
岗位头像 50 枚约 190 KB，而 `/list` 会被胶囊每 2 秒轮询一次，把头像塞进 `/list` 等于每秒陪跑 95 KB，
所以 `/org` 单独一条路由、由设置页**只取一次**。

**Slot 注册（`lib/client.js` 尾部）**：

| slot | id | order | 说明 |
| --- | --- | --- | --- |
| `settings.section` | `overseas-skills` | 26 | 走四层下钻（`<OverseasSkillsPage endpoint="/list" org />`） |
| `settings.section` | `fullstack-skills` | 27 | **不带 `org` 属性** → 保持原分组视图 |

同一 bundle 另有一段 `no-hero-capsules.spec.mjs` 明确锁定的**反向契约**：插件**不再向 `conversation.input.dock`
注册任何面**（R1 决策），但「技能中心两页仍在注册」也必须成立（移除不得连带打掉插件本体）。

**实测探活**：`curl -H 'Host: 127.0.0.1:43120' http://127.0.0.1:43120/api/dsh-overseas-skills/org` → **HTTP 200 / 355,750 bytes**。

---

## B. 数据模型 —— manifest 逐份 schema 与精确计数

### B.1 `manifest/role-assignments.json` — **归位判定的唯一事实源**（198,243 B）

顶层三键：`_meta` / `coverage` / `skills`。

**`_meta`（全文）**：

| 键 | 值（要点） |
| --- | --- |
| `purpose` | 「出海技能（222）+ AI全栈（29）→ 《AI组织变革》50 岗位的归位判定；页面『出海技能』四层下钻的数据源。」 |
| `vocabulary` | 岗位/责任名取自材料 `docs/05-agents/role-catalog.json` 与 `docs/04-organization/organization-graph.json`（**151 条责任名，一名唯一属一岗**） |
| `relation_to_skill_map` | 与 `scripts/role-presets/skill-map.json` 的关系：**该文件是「接线」**（责任名→技能供给，喂 50 个 preset）；**本文件是「归位」**（技能→岗位，喂页面）。`source=skill-map` 的条目即从前者继承 |
| `generated` | `2026-09-12 十批语义判定 + 机械校验（.scratch/overseas-skills-refactor/validate-assignments.py）` |

**`coverage`（实测）**：`{ "skills": 253, "with_roles": 217, "without_roles": 36, "role_ids_with_supply": 49 }`

> ⚠️ **口径提醒**：`coverage` 的 253 是 **223 出海 + 30 AI全栈**，而页面四层树只喂 `SKILLS`（223 条）。
> 所以页面上「未归岗」是 **14** 条，不是 36 条。`_meta.purpose` 里写的「222 + 29」也已过期 1–2 条。

**`skills`：dict，253 条，键 = 技能英文名（kebab-case）**。

**per-skill 记录形状**（字段出现频次 = 253/253，除非另注）：

| 字段 | 类型 | 含义 | 实测分布 |
| --- | --- | --- | --- |
| `catalog` | `"overseas"` \| `"fs"` | 归属哪本目录 | overseas 223 / fs 30 |
| `scenario` | string | L1 场景 key | overseas 8 个 key（a-market 49 … h-enable 30）；fs 8 个 `fs-*` |
| `sub` | string | **细分场景 key**（L2 of taxonomy） | 如 `a5-supplier`、`h2-agent-skill` |
| `roles` | array | 归位到的岗位（可为空数组） | **总数 329 条挂载** |
| `no_role_kind` | `null` \| enum | 无岗**分型**，无岗时必填 | 217 个 `null`；`TOOL_ONLY` 14 / `OUT_OF_SCOPE` 1 / `GENERIC_METHOD` 21 |
| `no_role_reason` | `null` \| string | 无岗的判定理由（散文） | 与 `no_role_kind` 同非空 |
| `dropped` | array（**仅 20 条有**） | 既有先验被**显式删除**的记录 | 20 条，涉及 20 个技能，共 24 条删除记录 |

**`roles[]` 每条的形状**（**329 条**，每个字段都 100% 存在）：

| 字段 | 类型 | 实测 |
| --- | --- | --- |
| `id` | `AGT-NNN` | 岗位 id |
| `responsibility` | string | **中文责任名**，必须逐字属于该岗三条责任之一 |
| `source` | `"skill-map"` \| `"assigned"` | `skill-map` **197** / `assigned` **132** |
| `confidence` | `"high"` \| `"medium"` \| `"low"` | **high 101 / medium 180 / low 48** |
| `evidence` | `{from_skill, from_role}` | **两条逐字证据**，见下 |
| `note` | string（**233/329 有**，96 条无） | 自由说明 |

**`evidence` 是本表最值得抄的设计**（329/329 都有两个键）：

- `from_skill`：**该技能文本中的连续子串**（如 `"订单自动路由"`、`"REST/GraphQL API"`）
- `from_role`：**该岗位 mission / artifact / 责任名中的连续子串**

ADR-0044 把这条写成硬约束：「每条『技能 → 岗位』必须附 `from_skill` 与 `from_role`；
校验脚本断言二者都是**原文连续子串**、责任名逐字属于该岗三条之一、岗位 id 在 50 个之内、
同岗不重复挂载、**≥3 岗不得标 `high`**，并且**既有映射被删除时必须写明理由（静默删除直接判错）**」。

**一条真实记录（原样，供接入时对照）**：

```json
"dropshipping-supplier-integrator": {
  "catalog": "overseas", "scenario": "a-market", "sub": "a5-supplier",
  "roles": [
    { "id": "AGT-015", "responsibility": "订单协调", "source": "assigned",
      "confidence": "medium",
      "evidence": { "from_skill": "订单自动路由", "from_role": "订单协调" } },
    { "id": "AGT-047", "responsibility": "接口契约", "source": "assigned",
      "confidence": "high",
      "evidence": { "from_skill": "REST/GraphQL API", "from_role": "接口契约" },
      "note": "技能定义供应商系统对接方式（API/CSV/EDI）与订单、库存接口要求，属集成契约与业务工具实现" }
  ],
  "no_role_kind": null, "no_role_reason": null,
  "dropped": [ { "id": "AGT-014", "reason": "…AGT-014 三条责任（供应商评估/产能调查/OEM协作）无对应工作" } ]
}
```

**一条无岗记录**：

```json
"social-network-mapper": {
  "catalog": "overseas", "scenario": "a-market", "sub": "a2-competitor",
  "roles": [], "no_role_kind": "TOOL_ONLY",
  "no_role_reason": "技能是以 Twitter/X 互动数据构建关系图谱并跑社群检测的工具步骤（networkx），由 org-structure-research 等调研任务调用，本身不产出任何岗位三条责任所要求的工作产物"
}
```

### B.2 `manifest/taxonomy-v3.json`（21,750 B）—— 分类骨架的权威

| 键 | 类型 | 计数 |
| --- | --- | --- |
| `version` | int | `3` |
| `scenarios` | array | **8** |
| `mapping` | dict | **253**（技能名 → 细分场景 key） |
| `overseasNames` | array | **223** |
| `fullstackNames` | array | **30** |

**8 场景 → 28 细分**（实测，非文档的「29」）：

| key | title | subs |
| --- | --- | --- |
| `a-market` | A 市场与选品 | 5 |
| `b-product` | B 产品与Listing | 4 |
| `c-content-brand` | C 内容与品牌 | 3 |
| `d-traffic` | D 流量与获客 | 3 |
| `e-sales` | E 销售与转化 | 4 |
| `f-fulfillment` | F 履约与售后 | 3 |
| `g-insight` | G 数据与决策 | 3 |
| `h-enable` | H 组织与工具 | 3 |
| | **合计** | **28** |

每个 scene 元素形状：`{ key, title, subs: [{ key, title }] }`。
`mapping` 样例：`"anysearch" → "a1-market-trend"`、`"alibaba-amazon-market-intel" → "a1-market-trend"`。

> **本表是 join 的中枢**：`mapping[skillName] = subKey`，再由 `scenario_by_sub[subKey] = sceneKey`。
> `build_preset_catalog.py` 就是靠这两跳把 `category`（大场景）与 `subcategory`（细分）写进 `catalog.js`；
> **未在 `mapping` 里的技能会被丢弃并列入 `unmapped`，脚本随即 `sys.exit(1)`**（第 288–290 行）。

### B.3 五本技能目录 manifest —— 精确计数与 join 键

| 文件 | 体量 | `skills` | `categories` | 其它 | 行字段（join 键加粗） |
| --- | --- | --- | --- | --- | --- |
| `manifest/skills.json` | 73,900 B | **131** | **12** | — | **`name`** / `title` / `category` / `categoryTitle` / `toolBacked` / `importable` |
| `manifest/81-skills.json` | 27,472 B | **52** | **3** | `overrides` **29** | **`name`** / `title` / `category` / `categoryTitle` / `toolBacked` / `summaryZh` / `toolGap` / `icon` |
| `manifest/marketing-skills.json` | 11,444 B | **38** | **3** | — | **`name`** / `title` / `category` / `categoryTitle` / `toolBacked` / `importable` / `summaryZh` / `toolGap` |
| `manifest/fullstack-skills.json` | 9,445 B | **30** | **8** | — | **`name`** / `title` / `category` / `categoryTitle` / `toolBacked` / `summaryZh` / `toolGap` / `icon` |
| `manifest/extra-skills.json` | 733 B | **2** | 0 | — | **`name`** / `title` / `category` / `categoryTitle` / `toolBacked` / `summaryZh` / `toolGap` / `icon` |

**join 键一律是 `name`（技能英文名 = `~/.dsh/skills/<name>/` 目录名 = SKILL.md frontmatter 的 `name`）。**

**合并口径（`build_preset_catalog.py` 的确定性顺序，重复即丢弃）**：
`skills.json`(131) → `marketing-skills.json`(38) → `81-skills.json`(52) → `extra-skills.json`(2，且 `if any(x["name"]==s["name"] …) continue`)
→ 对全部行施加 `81-skills.json.overrides`（29 条，覆盖 `title`/`toolBacked`/`summaryZh`）
→ 施加 `scripts/legacy-summaries.json` 的人工精修
→ 按 `taxonomy.mapping` 归位 → 写入 `lib/catalog.js`。

**实测产物**：`SKILLS` **223** 条（不是 131+38+52+2 = 223 ✓ **恰好吻合**，说明无重名丢弃）；
`CATEGORIES` 8 组 / 28 `subs`；`SKILLS_FS` **30** 条 / `CATEGORIES_FS` 8 组。

**分场景实测分布**：`a-market 49 / b-product 16 / c-content-brand 37 / d-traffic 24 / e-sales 30 / f-fulfillment 12 / g-insight 25 / h-enable 30` = **223**。

> ⚠️ `manifest/skills.json` 自己的 12 个 `category`（`sourcing`/`research-selection`/`design`/`content-gtm`/`seo-ads`/`store-ops`/`shipping-tariff`/`analytics-finance`/`crm-retention`/`productivity`/`agent-tools`/`other`）
> 与 `81-skills.json` 的 3 个（`knowledge-engineering`/`skill-engineering`/`ecommerce-analytics`）、`marketing-skills.json` 的 3 个
> 在 v3 之后**已全部被 `taxonomy` 的 8 场景覆盖**——源码注释原文：「v3：分类完全由 taxonomy 决定，marketing 旧分类不再插入」。
> 旧 `category` 只作为**来源档案**保留，页面不再使用。

### B.4 图标 manifest：`*-fs` 变体是什么，怎么同步

| 文件 | 体量 | 键数 | 键集 |
| --- | --- | --- | --- |
| `manifest/skill-icons.json` | 335,155 B | **84** | 技能英文名（81 系 + 2 自定义 + 1） |
| `manifest/skill-icons-fs.json` | 116,525 B | **30** | AI 全栈技能英文名 |
| `manifest/category-icons.json` | 121,896 B | **33** | 分类 key（12 旧分类 + 8 场景 + 7 `preset-*` + 若干） |
| `manifest/category-icons-fs.json` | 28,743 B | **8** | `fs-clarify` / `fs-spec` / `fs-architecture` / `fs-implement` / `fs-quality` / `fs-infra` / `fs-collab` / `fs-writing` |

**`-fs` = AI 全栈（full-stack）第二页面专用的那一档。实测 `skill-icons` 与 `skill-icons-fs` 键集交集 = 0，`category-icons` 与 `category-icons-fs` 交集 = 0** —— 两档是**不重叠的两份图**，不是同一份的镜像。

**生成方式（单一脚本、双档同跑）**：`scripts/assign_lute_icons.py` 从
`~/.dsh/skills/lute-brand-icons/assets/manifest.json`（品牌清单，176 条目）按
`sk-fs-<name>` / `fs-cat-<key>` 前缀取图，**在同一次运行里**分别 `json.dump` 出这两个 `-fs` 文件
（脚本第 83–96 行）。即：**同步不是靠「两边对齐」，而是靠「一次运行同时产两份」**。

**运行时怎么被读**：`build_preset_catalog.py` 把 `skill-icons.json` 的图按 `name` 贴到海外行；
`skill-icons-fs.json` 的图按 `name` 贴到 FS 行，FS 行的兜底链是
`skill_svg_fs[name] → fs_cat_icon[category] → cat_icon_fs[category]`。
`verify_static.mjs` 断言「每一行最终都能拿到一个 `data:image/svg+xml;base64,` 图标，否则报 `emptyIcon`」。

**已固化的防抹逻辑**（`docs/recent-changes-2026-09-08.md`）：`assign_lute_icons.py` 从「全量重写」改为
**「保留非 81 系自定义图标」合并写**，否则重跑会抹掉手工加的两枚图标。

---

## C. 页面的层语义

### C.1 四层是什么、节点类型有哪些

树形（`lib/org-tree.js` 第 4 行原话）：
**L1 场景 → L2 面 → L3 责任域 → L4 岗位 → 技能卡**。

| 层 | 节点 id 形状 | 来源（**事实的家**） | 数量 |
| --- | --- | --- | --- |
| L1 场景 | `a-market` … `h-enable` | `manifest/taxonomy-v3.json:scenarios` → `CATEGORIES` | 8 |
| L2 面（plane） | `PLN-MGT` / `PLN-OPS` / `PLN-CTL` / `PLN-PLT` | **已安装 preset 的 `x_lute.plane`** | 4 |
| L3 责任域（domain） | `DOM-01` … `DOM-08` | **已安装 preset 的 `x_lute.domain`** | 8 |
| L4 岗位（role） | `AGT-001` … `AGT-050` | `~/.dsh/.agent-presets/agt-NNN/manifest.json` | **50** |
| 卡（card） | 技能英文名 | `lib/catalog.js` 的 `SKILLS` | **223** |

节点类型共 **5 种**，其中 3 种有专门的「兜底位置」：`scenarios[].unassigned`（未归岗卡）、
`stats.zeroCardRoles`（零卡岗位）、`tree.wiredOnlyByRole`（preset 挂了但未归本岗的卡）。

**三条设计约束（源码注释里明说是「来自真实的坑」）**：

1. **归位 ≠ 接线**。卡挂在本岗下是「归位」（输入 `assignment`）；preset 是否真的挂载了它是「接线」（`roles[].wired`）。
2. **一技多岗**。同一技能可在多个岗位下出现（岗位视图 = 工具箱），**卡片数 ≠ 行数**。
3. **零卡岗位与未归岗技能不许消失**。「它们没有『位置』，不给位置就等于不存在。」

**`PLANE_ORDER` 硬编码**（`["PLN-MGT","PLN-OPS","PLN-CTL","PLN-PLT"]`）——面的顺序是代码常量，不是数据。
排序函数 `byPlaneThenDomainThenOrder`：面序 → `domain.id` 字典序 → `x_lute.order`。

### C.2 责任域 vs 平面 vs 岗位 —— 三者关系

一句话：**平面是责任域的父容器；责任域是岗位的父容器；岗位是唯一能被「接线」的东西。**

- **层级是严格的树**，不是图：`org-tree.js` 先按 `PLANE_ORDER` 取平面，再在平面内
  `[...new Set(planeRoles.map(r => r.domain.id))].sort()` 取责任域，再在域内按 `order` 排岗位。
- **一个责任域只属于一个平面**（`DOM-03 供应与履约` 恒在 `PLN-OPS` 下）；层级由**preset 的 `x_lute` 声明**决定，
  页面**不自己分类**。`preset-roles.js` 的注释把这条写死了：

  > 事实的家是**已安装的 preset**……本模块**不复制**任何组织骨架，也就不会与材料或生成器产生第二份事实。

- **页面上「面 / 责任域」是筛选轴，不是归属轴**：`org-tree.js` 第 143 行
  `if (node.cards.length > 0) nodes.push(node)` —— **空岗位不渲染，空域不渲染，空面不渲染**。
  所以树的形状随所选场景变化（8 个场景各有一棵完全不同的子树），而不是一棵固定的 4×8×50 骨架。
- **计数口径注意**：平面节点带 `roleCount`（该面在该场景下有卡的岗位数），域节点带 `total`，都是**行数**不是卡片数。

### C.3 「本岗已接线 / 接线到 AGT-0xx / 未接线」—— 精确三态

**定义处：`lib/org-tree.js` 第 203–217 行 `wiringStatus(skill, roleId, wiredIndex)`**（有单测锁定）。
`wiredIndex` 的构造在第 47–54 行：遍历 50 个岗位的 `role.wired`（= preset 的 `x_lute.skills.subset`），
建「技能名 → 挂载它的岗位 id 列表」的反向索引。

```js
export function wiringStatus(skill, roleId, wiredIndex) {
  const ids = wiredIndex?.[skill] ?? [];
  if (ids.includes(roleId)) return { kind: "self",  roleIds: [] };      // 本岗已接线
  if (ids.length > 0)      return { kind: "other", roleIds: ids };      // 接线到 AGT-0xx
  return                          { kind: "none",  roleIds: [] };       // 未接线
}
```

| 状态 | 判据（精确） | 页面徽标文案 | class |
| --- | --- | --- | --- |
| **本岗已接线** | 该技能名出现在**当前岗位**的 `x_lute.skills.subset` 里 | `本岗已接线` | `ovsBadgeSelf` |
| **接线到 AGT-0xx** | 该技能名**不在**当前岗位 subset 里，但**出现在 ≥1 个别的岗位** subset 里 | `接线到 ` + `roleIds.join("/")`（如 `接线到 AGT-021/AGT-022`） | `ovsBadgeOther` |
| **未接线** | 该技能名在**全部 50 个岗位**的 subset 里都不出现 | `未接线` | `ovsBadgeNone` |

- **判据在宿主侧算，客户端只做 4 行查表**。`lib/client.js` 第 466–479 行的注释明说：
  「接线三态：判据来自宿主（`node.wired` 与 `tree.wiredIndex`），这里只做查表。规范定义在
  `lib/org-tree.js` 的 `wiringStatus` 里，宿主侧有单测锁定；client bundle 是独立工厂，
  `require` 不到那个模块，因此这里只重复 4 行查表、不重复判据。」
  → **这是一处刻意的逻辑复制**：改判据必须同时改两处。
- **三态按「卡 × 岗位」算**（`org-tree.js` 第 205 行原话：「与 `buildOrgTree` 分开，因为它是**每张卡在每个岗位下**都要算一次的东西」）。
  **实测行级分布（在 223 卡 / 319 行的真实负载上）：`self = 124`、`other = 52`、`none = 143`。**
- 另有两枚**独立**徽标，别与三态混：`归位·本轮判定`（`roles[].source === "assigned"` 的那部分）与
  `跨 N 岗`（`roles.length > 1`）。
- ⚠️ **`stats.wiring` 里的 `toOtherRole` / `none` 恒为 0**（`org-tree.js` 第 195–196 行硬编码），
  与上面实测的 52 / 143 不符。**这是两个死字段**；页面不读它们（页面走 `wiringStatus`），
  但任何外部消费者读 `tree.stats.wiring` 会拿到错的数（见 §F 风险 R2）。

### C.4 「卡」怎么表示、UI 逐卡渲染什么

**card 的负载由 `lib/index.js: itemRecord()`（第 105–117 行）产生，共 9 个字段**：

```js
{
  name,          // 技能英文名（join 键 / toggle 的 key / 目录名）
  title,         // 中文标题（来自 manifest）
  icon,          // data URI（skill-icons.json 优先，回退扇区图标）
  description,   // 读 ~/.dsh/skills/<name>/SKILL.md 的 frontmatter description（>160 字符截断加 …）
  descriptionZh, // manifest 的 summaryZh；工具型且未安装时的兜底文案
  modelEnabled,  // 由 disable-model-invocation 反推（缺字段 = false）
  toolGap,       // 非空 → 渲染「需外部工具」橙色徽标
  installed,     // SKILL.md 是否存在
  template       // lib/templates.js 的三层结构化引导模板
}
```

**UI 逐卡渲染（`lib/client.js: renderSkillCardWith`，第 668–706 行）**：

```
.ovsCard
├── .ovsCardHead
│   ├── .ovsCardTitleWrap
│   │   ├── <img class="ovsCardIcon" src={it.icon}>        ← 空 icon 则不渲染
│   │   ├── <span class="ovsCardTitle" title={it.name}>{it.title}
│   │   └── <span class="ovsToolGap">需外部工具</span>       ← 仅 it.toolGap 非空
│   └── <button class="ovsSwitch" role="switch" aria-checked={it.modelEnabled}>
├── <p class="ovsCardDesc">{it.descriptionZh || it.description}
└── <div>{badges}</div>                                     ← 仅四层视图传入
```

- **`badges` 只在四层视图里传**。扁平视图与搜索路径调用单参版本 `renderSkillCard(it)`——
  注释里特意写明「`.map(renderSkillCard)` 时第二个实参是数组下标，所以对外只暴露单参版本」。
- **开关 = 模型调用开关**：开 → `disable-model-invocation: false`；关 → `true`；`user-invocable` **恒 `true`**
  （`/` 菜单始终全量可见）。默认策略 O1：导入技能默认 `disable-model-invocation: true`。
- **占位卡**：归位表里有、目录里没有的技能，不静默丢弃，渲染一张
  「目录中缺少这张卡 / 归位表里有它，但技能目录（catalog）里没有，请重跑目录生成脚本。」的 `ovsCard`。
- **没有链接字段**。卡片上没有 URL、没有跳转、没有「论文来源」。**这是本页与论文卡页面最大的形态差异**。

### C.5 能不能渲染**外部来源**（如论文衍生的 markdown）的卡？

**现状：不能，且不是缺一个字段的问题——是缺整条数据通路。**

页面渲染的卡**只来自 `lib/catalog.js` 的 `SKILLS` 常量**（`lib/index.js` 第 5 行 import，
`buildScenarios` / `buildFlatGroups` / `buildOrgTree` 三处都只遍历它）。
`SKILLS` 是 `build_preset_catalog.py` 从 **5 本 manifest + taxonomy** 生成的**构建产物**，
`manifest/` 除了 `role-assignments.json` 外都不在 `files` 白名单里，宿主运行期**读不到**。

**要让一张论文卡出现在这一页，最少要补齐的字段/通路（精确清单）**：

| # | 需要存在的东西 | 位置 | 现状 |
| --- | --- | --- | --- |
| 1 | `SKILLS` 数组里多一条记录，字段 ≥ `{name, title, category, categoryTitle, toolBacked, summaryZh, toolGap, icon, subcategory}` | `lib/catalog.js`（→ `manifest/*.json` 源） | ✗ 无 |
| 2 | `taxonomy.mapping[<name>] = <subKey>` | `manifest/taxonomy-v3.json` | ✗ 无 → **缺它则 `build_preset_catalog.py` 直接 `sys.exit(1)`** |
| 3 | `ROLE_ASSIGNMENTS[<name>] = {roles:[{id,source}], noRoleKind}` | `manifest/role-assignments.json` → `lib/role-map.js` | ✗ 无 |
| 4 | `~/.dsh/skills/<name>/SKILL.md` 且 frontmatter 有 `description` | 用户技能根 | ✗ 无（缺则 `installed:false`、`description:""`，卡仍渲染但无描述） |
| 5 | UI 侧**新增渲染字段**（`venue` / `evidence_grade` / `paper_id` / 代码可执行性） | `lib/index.js: itemRecord` + `lib/client.js: renderSkillCardWith` | ✗ 无任何位置可放 |
| 6 | **没有任何"来源"或"链接"字段** | — | ✗ 卡片 schema 里不存在 |

**结论**：**论文卡在本页是「外来物种」**。本页的卡 schema 里没有 `source` / `url` / `paper` / `venue` 任何一个字段，
也没有「卡类型」维度；`itemRecord` 的 9 个字段全部是「运营技能目录 + 开关状态」语义。
硬塞进来的成本 ≈ 重写 `itemRecord` + `renderSkillCardWith` + 扩 `taxonomy` 与 `role-assignments` 两本 manifest。

**但**——论文卡**已经有一个一模一样的页面了**，见 §E。正确答案不是「扩建这一页」，而是「在已有的兄弟页面上补字段」。

---

## D. 构建与校验

### D.1 `scripts/build_role_map.py` —— 输入 → 输出 → 校验

**69 行，无第三方依赖。**

| 项 | 内容 |
| --- | --- |
| 输入 | `manifest/role-assignments.json`（缺则 `sys.exit` 并提示先跑 `.scratch/overseas-skills-refactor/merge-assignments.py`） |
| 输出 | `lib/role-map.js` |
| 用法 1 | `python3 scripts/build_role_map.py` → 写文件，打印 `✓ 写入 lib/role-map.js（253 条技能）` |
| 用法 2 | `python3 scripts/build_role_map.py --check` → **只验证**，不一致则**非零退出**：`✗ lib/role-map.js 与 manifest/role-assignments.json 不一致：重跑 …` |

**投影是「全等」而不是「子集」**：`--check` 读回文件全文，与现场重新拼出来的 `text` **逐字符比较**
（第 63 行 `if open(OUT).read() != text`）。因此文件的每一字节（含注释头）都受锁定。

**投影时做的三件事**：

1. `roles` 只保留 `{id, source}`（**丢掉 `responsibility` / `confidence` / `evidence` / `note`**）——
   页面不需要，且 evidence 逐字证据会把 payload 撑大。
2. `no_role_kind` → JS 的 `noRoleKind`（`null` 或字符串）。
3. `coverage` → `ROLE_ASSIGNMENT_META`，**键名重映射**：`skills` / `withRoles` / `withoutRoles` / `roleIdsWithSupply`。
   注释：「数字由 manifest 决定，**不在这里重算**」。

**它自己"校验"什么**：只校验一致性，**不做语义校验**。真正的机械校验（`evidence` 是连续子串、
责任名逐字属该岗三条之一、同岗不重复、≥3 岗不得 `high`、删除须有理由）在
`.scratch/overseas-skills-refactor/validate-assignments.py`（**一次性脚本，不在包内**）。
**包内留守的语义校验只有 `test/role-map.spec.mjs` 的三条**（形状 / meta 计数 / `--check`）。

### D.2 `scripts/` 全表（26 个，一行一个用途）

**构建·生成（8）**

| 脚本 | 用途 |
| --- | --- |
| `build_role_map.py` | 编译 `manifest/role-assignments.json` → `lib/role-map.js`（`--check` 供门禁） |
| `build_preset_catalog.py` | 扫 7 个 preset 的 `skills/` → `presets/preset-skills.json`；合并 5 本 manifest + taxonomy → 重建 `lib/catalog.js`；`unmapped` 非空即 `exit 1` |
| `build_manifest.py` | 从 Accio 目录（`remote_skills_cache` + OCR 映射）生成 `manifest/skills.json` |
| `gen-layer-icons.mjs` | 从 `lute-brand-icons/assets/manifest.json` 烘 12 枚层头像 → `lib/layer-icons.js`（缺图即 `exit 1`，绝不写半份） |
| `gen_category_icons.py` | **DEPRECATED**（已被 `assign_lute_icons.py` 取代，保留作历史参考） |
| `gen_preset.mjs` | 生成业务工作流 agent presets |
| `gen_bmg_preset.mjs` | 从 `81-mapping.json` 重建「品牌营销增长官」preset 三件套 |
| `assign_lute_icons.py` | 分配 LUTE 头像：分类 25 枚 + 81 系技能 81 枚；**双档写** `category-icons{,-fs}.json` / `skill-icons{,-fs}.json`；合并写、不抹自定义图标 |

**导入（4，一次性幂等）**

| 脚本 | 用途 |
| --- | --- |
| `import-accio.mjs` | Accio 技能包 → `~/.dsh/skills`（工具型跳过、frontmatter 归一化、中文 title） |
| `import-81skills.mjs` | 81-Skills 中文目录 → DSH 技能并安装 |
| `import-fullstack.mjs` | 安装 mattpocock AI 全栈稳定集 29 个（译文取 `staging/translations/`，缺则英文回退） |
| `import-marketing-skills.mjs` | 导入 `/Users/lute/project/skills` 的营销技能 |

**数据加工（6，全部幂等）**

| 脚本 | 用途 |
| --- | --- |
| `apply_summaries.py` | 为 manifest 追加简明中文简介 `summaryZh`，并重建 `lib/catalog.js` |
| `optimize_data.py` | O1 默认关闭 + O2 互斥边界（description 尾部 + `whenToUse`）+ O3 环境说明块 + 描述压缩 |
| `normalize-zh.mjs` | 对照 zh-CN fork 的三项必采修正（只处理 AI 全栈 29 个技能，ADR-0008） |
| `soften-clarify.mjs` | F1：把 blanket「缺材料先追问」软化为分级规则 |
| `unify-directories.mjs` | 统一 `~/.dsh/skills` 目录形态（D2/D3 决策） |
| `unify-structure.mjs` | 把未被 81-Skills 替换的存量技能统一为 81 风格结构 |
| `unify-refine.mjs` | 精修层：用人工校准的路由边界（`unify-refine-batch{1,2,3}.json`）替换批量层生成的「## 何时不用」 |

**评测（5）**

| 脚本 | 用途 |
| --- | --- |
| `route_bench.mjs` | 词法快筛（v2 schema）：严格（精确技能）+ 宽松（同 cluster） |
| `route_bench_llm.mjs` | LLM 档路由评测聚合读 `eval/llm-picks.jsonl`；自动与上一轮 diff 输出 REGRESS / FIXED |
| `route_bench81.mjs` | 81 系路由词法快筛（词库 = catalog 终版） |
| `route_bench_fs.mjs` | AI 全栈词法路由（词库 = `*_FS` 双常量） |
| `route_bench_full.mjs` | 全目录词法路由基准（词库 = catalog 228 行） |

**门禁·验收（3 + 3 sh）**

| 脚本 | 用途 |
| --- | --- |
| `verify_static.mjs` | **静态闸门**（重建后 / 交付前必跑）：名字唯一 / 图标覆盖 / 悬空引用 / 分类存在性 |
| `verify-fullstack.mjs` | ① 30/30 安装 + frontmatter 解析 / 引号 / name / description |
| `patch-cn-slash.mjs` | 「斜杠命令全中文 + 结构化引导模板」补丁（幂等，锚点匹配，首次自动备份；`--restore` 可回滚） |
| `pipeline.sh` | **一键管线（闸门式）**：静态校验 → 分配头像 → 重建目录 → 同步 profile → 预设 lint；`--import` 含 81 转换 |
| `verify_p7.sh` | **重启后运行时验收**（只读 HTTP 检查，不改状态） |
| `preset_mount_probe.sh` | 扫 `~/.dsh/sessions` 的 `session.jsonl.zstd` 头行，统计引用某 preset 的会话 |

### D.3 `test/` 与 `eval/`

**`test/`：6 个 spec，共 32 个 test，实测 `node --test test/*.spec.mjs` → 32 pass / 0 fail / 0 skip（198 ms）。**

| 文件 | tests | 锁什么 |
| --- | --- | --- |
| `host-routes.spec.mjs` | 4 | 真跑 `apply()` 截下 `webServer.register` 的 handler 真调用：六条路径全注册 / 非回环 401 + 错方法 405 / **`/org` 负载数字必须与 manifest、catalog 对得上** / 30 s 缓存返回同一对象 |
| `org-tree.spec.mjs` | 9 | 四层计数 / 行数 > 卡片数 / 零卡岗位不许消失 / 未归岗按场景分桶带分型计数 / 词表外 id 绝不凭空造岗 / 归位来源分得清 / **`wiringStatus` 三态互斥** / `wiredOnlyByRole` 不混进归位名单 / 空输入不炸 |
| `overseas-skills.spec.mjs` | 12 | `errorMessage` 兜底 / `isValidSkillName` 拒路径穿越 / `rebuildFrontmatter` 7 条（开关语义、幂等、CRLF、无 frontmatter 返 null、**与抽取前内联实现逐字节一致**）/ `findCatalogInconsistencies` 2 条 |
| `role-map.spec.mjs` | 3 | 形状合法（`^AGT-\d{3}$` / source 白名单 / 无岗必给分型）/ meta 计数与 coverage 一致（`withRoles + withoutRoles === skills`）/ **真跑 `build_role_map.py --check`** |
| `layer-icons.spec.mjs` | 2 | 12 枚齐全且键就是 4 面 + 8 域 / **跨包逐字节**（见下） |
| `no-hero-capsules.spec.mjs` | 3 | 「捕获本身有效」（否则断言空转）/ R1 不再注册 `conversation.input.dock` / 两页仍在注册 |

**跑法**：`npm test`（= `node --test test/*.spec.mjs`）。**32 个全绿，0 skip**。
README 第 52 行写「（22 项）」——**已过期，实际 32**。

**`eval/`：路由精度基准，不是正确性测试。**

| 文件 | schema | cases |
| --- | --- | --- |
| `eval/routing-benchmark.json` | `routing-benchmark.v2` | **141**（字段：`query` / `expect` / `negative` / `cluster` / `difficulty`） |
| `eval/routing-benchmark-81.json` | `…-81.v1` | **22** |
| `eval/routing-benchmark-fs.json` | `…-fs.v1` | **29** |
| `eval/routing-benchmark-full.json` | `…-full.v1` | **228** |
| `eval/llm-picks.jsonl` / `-81` / `-full` | JSONL（每行一组 JSON 数组） | 10 / 1 / 1 行 |
| `eval/llm-cases-groups.json` | list | 7 组 |
| `eval/runs/` | — | **15 次历史留痕**（`<时间戳>-{lexical,llm,lexical-81,lexical-full}.json`） |

**门禁（`docs/iteration-runbook.md`）**：LLM 档**严格命中 < 95% → 不通过**；本轮**新增失败用例即回归**，必须修复或回滚文案。
词法档为秒级粗筛，语义型失败可预期，**不算门禁**。
> ⚠️ runbook「当前基线」段写「v2，70 条（49 basic + 21 boundary）」，**实测 141 条**——同一份文档内的数字也已过期。

### D.4 跨包逐字节测试 —— 「两个包各烘一份，靠跨包逐字节测试防漂移」

**存在，且就是 `test/layer-icons.spec.mjs` 的第 2 条。**

- **被防的是什么**：12 枚层头像（4 面 + 8 责任域）的 base64 SVG data URI。本包 `lib/layer-icons.js` 一份，
  `packages/surfaces/dsh-algo-skills-local/src/layer-icons.ts` 一份。**两个页面分属不同插件、各自要能独立渲染**，
  所以必须各烘一份；代价是同一枚图有两份字节。
- **断言方式**：`readAlgoIcons()` 用 `/"(PLN|DOM)-[A-Z0-9]+":\s*"([^"]+)"/g` 从隔壁包的 **TS 源**里
  抽字面量（不解析 TS），然后 `assert.deepEqual(drifted, [])`——**逐字节相等，差一个字符即红**。
- **测试自己写明了「不静默通过」**：
  > 拿不到隔壁包（单包签出、离线）时**跳过并说明**，不做静默通过。

  实现是 `{ skip: !existsSync(ALGO_ICONS) ? "隔壁包不可见（单包签出），本项跳过——不视为通过" : false }`。
  **实测本次 0 skip**，即跨包断言**真的跑了且过了**。
- **路径**：`test/../../../surfaces/dsh-algo-skills-local/src/layer-icons.ts`
  → `/Users/lute/project/Magpie-Horch/packages/surfaces/dsh-algo-skills-local/src/layer-icons.ts` ✓ 存在。
- **同源的第三处**：`scripts/gen-layer-icons.mjs` 里有 `LAYER_MAP`（brandId → 骨架键），
  注释「与算法技能页**逐字一致**」；隔壁包有对等的 `scripts/gen-layer-icons.mjs`，
  以及一个独立脚本 `scripts/verify-layer-icons.mjs`（`npm run verify:icons`）。

---

## E. 新内容类型的接入缝（论文衍生算法技能卡）

### E.0 先纠正一个前提：**这个页面已经存在**

`packages/surfaces/dsh-algo-skills-local`（v0.1.0）就是**论文 → 技能库的页面**。
它的 `package.json` description（原文）：

> LUTE algorithm-skills surface: the settings section that carries the **paper→skills library (1338 cards)**
> through **the same organization skeleton the role matrix uses** — 4 planes → 8 responsibility domains → 50 roles.
> The tree is built from **installed runtime state only** (each `p2s-*/SKILL.md` frontmatter plus each `agt-*`
> preset manifest), so the surface **owns no classification fact of its own**.

**实测探活：`GET /api/dsh-algo-skills/health` → HTTP 200**

```json
{"ok":true,"plugin":"algo-skills-local","root":"/Users/lute/.dsh/skills",
 "presetRoot":"/Users/lute/.dsh/.agent-presets",
 "totals":{"planes":4,"domainSlices":12,"distinctDomains":8,"roles":50,
           "skills":1338,"placed":1327,"unplaced":11,
           "wired":127,"wiredAnywhere":258,"emptyRoles":2},
 "issues":2}
```

**`~/.dsh/skills/p2s-*` 实测 1338 个目录**。已注册三条路由（`src/routes.ts` 的 `ROUTES` 常量）：

| 路由 | 方法 | 用途 |
| --- | --- | --- |
| `/api/dsh-algo-skills/tree` | GET | 整棵骨架 + 每张卡的 index row（一次请求渲染整页，**不扇出 1338 次**）；`DEFAULT_CACHE_TTL_MS = 15_000` |
| `/api/dsh-algo-skills/toggle` | POST | 唯一写操作，**双重收窄**：`isValidSkillName` **且** `P2S_NAME = /^p2s-[a-z0-9]+(?:-[a-z0-9]+)*$/` |
| `/api/dsh-algo-skills/health` | GET | 计数 + 漂移诊断，一行健康检查 |

**它的归位 join key 是**：`card.l3_business → role.responsibilities`（中文责任名）。源码原文：

> The join is `card.l3_business → role`. A card is placed by how it was [classified].

**它读的 p2s 卡 frontmatter（实测一份真卡 `~/.dsh/skills/p2s-3d-bin-packing-optimization/SKILL.md`）**：

```yaml
name: "p2s-3d-bin-packing-optimization"
title: "3D Bin Packing Optimization — 3D 装箱优化：…"
description: "触发词：装箱优化… 何时不用：… 安全边界：…"
l1_id: "PLN-OPS"          l1_plane:  "业务运营"
l2_id: "DOM-03"           l2_domain: "供应与履约"
l3_id: "DOM-03-056"       l3_business: "物流方案"
l3_all: "物流方案 / 仓储协作"
l1_l2_l3: "业务运营/供应与履约/物流方案"
p2s_card_id: "Skill-3D-Bin-Packing-Optimization"
p2s_src_domain: "18-物流履约"
user_summary: "把要发的箱子按尺寸和重量排进集装箱…"
user_try: "试试：黑五这批 35 种 SKU、55 个立方的货，能不能用 2 个 40HC 柜装完？…"
whenToUse: "…"   workflow: "…"   enabled: "true"
disable-model-invocation: "true"   user-invocable: "true"
```

**与 `dsh-overseas-skills` 的对照（同一套骨架，不同的数据源与归属）**：

| 维度 | `dsh-overseas-skills` | `dsh-algo-skills-local` |
| --- | --- | --- |
| 目录位置 | `packages/capabilities/` | `packages/surfaces/` |
| 卡数 | 223（+30 FS） | **1338** |
| 卡的来源 | `lib/catalog.js`（5 本 manifest 的构建产物） | **`~/.dsh/skills/p2s-*/SKILL.md` frontmatter（运行时扫盘）** |
| 归类事实在哪 | `manifest/taxonomy-v3.json` + `role-assignments.json`（**包自己拥有**） | **卡自己带的 `l1_*` / `l2_*` / `l3_business`（包不拥有任何分类事实）** |
| 归位 join | `role-assignments.json[skill] → roles[]` | **`card.l3_business` → `role.responsibilities`** |
| 层头像 | `lib/layer-icons.js` | `src/layer-icons.ts`（**逐字节相同**，跨包测试锁） |
| 组织骨架来源 | `~/.dsh/.agent-presets/agt-*/manifest.json` 的 `x_lute` | **同一份** |
| 路由前缀 | `/api/dsh-overseas-skills/{list,fullstack-list,org,toggle,prompt-template,credential}` | `/api/dsh-algo-skills/{tree,toggle,health}` |
| 构建 | 无构建（JS，`node --test`） | **有构建**（TS → `tsc -p tsconfig.build.json && tsdown`；`vitest`） |

### E.1 那么「加论文卡」到底该改哪里？两条路线

#### 路线 1（**推荐**）：在 `dsh-algo-skills-local` 上补字段

论文卡**已经在那里了**。要加的只是「论文来源 / venue / 证据等级 / 代码可执行性」这四个**呈现字段**。

| # | 改哪个文件 | 具体改什么 |
| --- | --- | --- |
| 1 | **`packages/surfaces/dsh-algo-skills-local/src/frontmatter.ts`** → `parseFields` 的调用点 | 该文件已是**通用** key→值 Map（只按第一个冒号切分、支持 JSON 引号化解码），**新增字段无需改它** |
| 2 | **`packages/surfaces/dsh-algo-skills-local/src/collect.ts`** → `parseSkill()`（第 170–192 行） | 这是**唯一需要改的解析点**。加 `fields.get('p2s_venue')` / `p2s_evidence_grade` / `p2s_code_runnable` / `p2s_paper_id` 等，塞进 `row` |
| 3 | **`packages/surfaces/dsh-algo-skills-local/src/wire.ts`** → `interface SkillRow`（第 14 行起） | 加四个可选字段（当前只有 `name/title/summary/modelEnabled/srcDomain/cardId/wired/wiredElsewhere`） |
| 4 | **`packages/surfaces/dsh-algo-skills-local/src/client/AlgoSkillsPage.tsx`** | 渲染这四枚字段（新徽标 / 新行） |
| 5 | **`packages/surfaces/dsh-algo-skills-local/src/client/algo-skills.module.css`** | 新徽标样式 |
| 6 | **`~/.dsh/skills/p2s-*/SKILL.md` 的 frontmatter（1338 份）** | **上游是 `paper2skills-vault` 的卡片 frontmatter**——那里**已经有** `paper_id` / `paper` / `venue` / `venue_tier` / `evidence_grade`（见本仓 `CLAUDE.md` 的 frontmatter v2 定义） |
| 7 | **`packages/surfaces/dsh-algo-skills-local/tests/collect.spec.ts`** | 加断言 |

**关键判断：这条路上「manifest」不是本包新增的文件，而是 `~/.dsh/skills/p2s-*/SKILL.md` 自己。**
——该插件刻意「owns no classification fact of its own」，把事实留在卡里（源码注释第 15–17 行：
「already carries its classification (`l1_plane` / `l2_domain` / `l3_business`). **The card IS the data**;
**no package reads another package's**」）。

**管线缺口（真正要做的活）**：`paper2skills-vault/<域>/Skill-*.md` 的 `venue` / `evidence_grade` / `paper_id`
**目前没有进 `~/.dsh/skills/p2s-*/SKILL.md` 的 frontmatter**——现有 p2s 卡只带 `p2s_card_id` 与 `p2s_src_domain`。
所以这不是「建一条新通路」，而是**把 `paper2skills-skills/paper-同步/scripts/sync.py` 的同步目标**（或 K1/K2 门禁产物的字段）**透传到 p2s 卡的 frontmatter**。

#### 路线 2（**不推荐**）：硬塞进 `dsh-overseas-skills`

若坚持让**同一页**同时显示运营技能与论文卡，则需改这 7 处（全部在 `packages/capabilities/dsh-overseas-skills/`）：

| # | 文件 | 改动 |
| --- | --- | --- |
| 1 | `manifest/taxonomy-v3.json` | `mapping` 加 1338 条 `p2s-*` → subKey；`overseasNames` 扩容。**不改则 `build_preset_catalog.py` 第 288 行 `sys.exit(1)`** |
| 2 | `manifest/<新兄弟 manifest>` | 新建 `manifest/paper-skills.json`（形状对齐 `skills.json`：`{categories, skills:[{name,title,category,categoryTitle,toolBacked,summaryZh,toolGap,icon}]}`）——**这就是「需要新增的 sibling manifest」** |
| 3 | `scripts/build_preset_catalog.py` | 加一段 `if os.path.isfile(PAPER)` 合并块（对齐现有 5 个 `if os.path.isfile(...)` 写法），并把它写进 `CATALOG` 头注释 |
| 4 | `manifest/role-assignments.json` | 1338 条归位记录（含 `evidence.from_skill`/`from_role` 逐字证据）——**这个成本极高** |
| 5 | `lib/index.js` | `itemRecord()` 加字段（第 105–117 行）+ 可能新增路由（如 `/paper-list`） |
| 6 | `lib/client.js` | `renderSkillCardWith`（第 668 行）加渲染分支 |
| 7 | `scripts/build_role_map.py` + `package.json:files` | 若新增 manifest 需被宿主读到，得再加白名单（或再编译一份 `lib/*.js`） |

**路线 2 的致命问题**：会把 1338 张卡的**名称空间灌进 `SKILLS` 常量**，
而该常量现在只有 223 条、被**每 2 秒轮询一次的 `/list`** 直接吐出去；
`/org` 的 30 s 缓存与「头像不进树节点」的优化也全部基于「50 岗位 × 几百张卡」的量级假设。
**并且会让两份组织骨架的投影产生两套 `wiredIndex` 语义**（论文卡的接线是 `p2s-*` 出现在 preset subset 里，
实测 `wiredAnywhere = 258`，其中 **163 个被接线的技能名根本不在海外 catalog 里**）。

### E.2 落地建议（一句话）

**不要在 `dsh-overseas-skills` 里造第二条通路。**
把 `venue` / `evidence_grade` / `paper_id` / 代码可执行性**写进 p2s 卡的 frontmatter**（源头是 `paper2skills-vault`），
再在 `packages/surfaces/dsh-algo-skills-local/src/collect.ts` 的 `parseSkill()` 与
`src/wire.ts` 的 `SkillRow` 上各加几个字段即可——**改动量约 4 个文件，且不动任何组织骨架、不动 manifest、不动跨包字节契约。**

---

## F. 已知问题、TODO 与漂移风险

### F.1 包内 `docs/` 已登记的

| 出处 | 内容 |
| --- | --- |
| `docs/maintenance-sop.md §7 待办留痕` | ① 「4 个加密技能（上市策略 / 市场可行性审计 / 竞品情报 / 电商季度战略）补齐 → `pipeline.sh --import`」；② 「DSH 升级后跑 §3」 |
| `docs/iteration-runbook.md` | 「81-Skills 增量」段的 4 个加密技能补齐流程（创作源在 `~/project/81-Skills/`，**仓库外**，2026-09-11 迁出） |
| `docs/phase3-tools.md` | 「**待办**：重启 → 用户在设置页填 Key → `exa_search` 实测 → **移除 3 个 Exa 徽标**（manifest/catalog 再生成 + 重启）」 |
| `docs/delivery-report-81.md` | 「待办：重启后按 `docs/fullstack-acceptance.md` 执行 P5 **人眼验收**」；「待办：重启 DSH Desktop + 刷新设置页，**人工核对 4 张新卡**」 |
| `docs/delivery-report-81.md` | **I3 子集开关语义**：「判定保持现状（预设内全部可调用），因 O1 默认模型关数据下改为『读文件标志』会反向禁用全部 12 预设技能，**回归风险高**——以文档明示为当前结论」 |
| `docs/ai-fullstack-analysis.md` | 「## 6. 执行 TODO（决策后定稿）」「## 8. 执行 TODO（定稿）」 |
| `docs/other-presets-plan.md` | `llm-wiki-fullstack` 曾 **缺 `includeDefaultRoots:false`（泄漏用户技能）**，已补配置 |
| `docs/audit-and-upgrade-plan.md` | 「**假设**『B 类替换 = 内容一致』……无错配证据，**低风险**」——即该等价关系**未被证明**，只是没有反证 |
| `docs/brand-marketing-growth-plan.md` | 「§10 风险与边界」；「**待办增量**（加密文件补齐后）：跑 P1/P2 增量（4 个 deferred）+ catalog 加 4 行 + 白名单自动生效」 |

### F.2 本调查**新发现**的漂移与缺陷（文档未登记）

| # | 级别 | 发现 | 证据 |
| --- | --- | --- | --- |
| **R1** | 🔴 **高** | **README 与页面现实的计数已漂移**。README 写「技能卡 222 / 未归岗 13 / 222 项 / 22 项测试」；**实测全部不一致** | `/org` 实返回 `cards:223, cardsUnassigned:14, rows:319, roles:50, rolesWithCards:47`；`node --test` → **32 tests**。README 第 36–39、52 行 + `_meta.purpose` 的「222 + 29」+ ADR-0044 的「251 条 / 82 条跨岗」均已过期 |
| **R2** | 🟠 中 | **`tree.stats.wiring` 里 `toOtherRole` / `none` 是死字段、恒为 0**，与真实值（52 / 143）矛盾 | `lib/org-tree.js` 第 195–196 行 `toOtherRole: 0, none: 0`；页面不读它们（走 `wiringStatus`），但任何外部消费者会拿到错的数 |
| **R3** | 🟠 中 | **`docs/` 全体指向已废弃的包路径**。`maintenance-sop.md`｜`iteration-runbook.md` 等都写 `~/project/Magpie-Horch/dsh-overseas-skills/`，实际是 `…/packages/capabilities/dsh-overseas-skills` | 逐字比对；`pipeline.sh` 等命令直接复制粘贴会失败 |
| **R4** | 🟠 中 | **`role-assignments.json` 的语义校验脚本不在包内**。`build_role_map.py --check` 只做一致性，不做「evidence 是连续子串 / 责任名逐字属该岗三条之一 / ≥3 岗不得 high」这些**真正的判定正确性**断言——那些只在 `.scratch/overseas-skills-refactor/validate-assignments.py`（一次性、未入库） | `.scratch/` 不在包内；包内留守的只有 `role-map.spec.mjs` 三条形状断言 |
| **R5** | 🟡 低 | **`lib/` 与 `manifest/` 里躺着 9 个未清理的历史副本**：`catalog.js.bak-a2`（624 KB）、`client.js.bak-cn-slash` / `.bak-contract` / `.orig-skillbar` / `.pre-fullstack-20260907`、`templates.js.bak-contract`、3 个 `.bak-lieflat-1789205601` | 占 ~1 MB；`files` 白名单未包含它们（不会发布），但**会让人误以为存在第二份事实源** |
| **R6** | 🟡 低 | **硬链接同步陷阱（文档已警示，但仍是最高频故障）**：pnpm 的 `file:` 安装是硬链接，而 `edit`/`write` 工具是**原子替换（新 inode）**→ 断链 → profile 副本停在旧内容、宿主热更检测不到。README 第 63–85 行给了 `-ef` 守卫 + `ln -f` 的绕法 | 若某天两文件变成同一 inode，`cat >` 会**自截断为 0 字节**（`docs/recent-changes` 记录过 `lib/catalog.js` 曾被「硬链接双杀」归零） |
| **R7** | 🟡 低 | **`eval/` 是死数据**。`routing-benchmark*.json` 最新修改 2026-09-06，`runs/` 最新 2026-09-06；而 `manifest/role-assignments.json` 是 2026-09-12。**路由基准没有跟上 223 条目录的现状**（`routing-benchmark-full.json` 只有 228 条，`route_bench_full.mjs` 注释仍写「catalog.js 228 行」） | 文件 mtime + 注释 |
| **R8** | 🟡 低 | **客户端刻意复制了宿主的三态判据**（`lib/client.js` 第 466–479 行 vs `lib/org-tree.js: wiringStatus`）。源码注释自认「client bundle 是独立工厂，`require` 不到那个模块，因此这里只重复 4 行查表」——**改判据必须同时改两处，且第二处没有测试** | `org-tree.spec.mjs` 只测宿主侧 |
| **R9** | 🟡 低 | `staging/` 12 MB 全部入库，含 82 个 81-Skills 源目录、第三方 `simplify-codebase`（含 1.6 MB `hero.png`、377 KB `template.html`）、2 份 561 KB 的 icon preview HTML、以及 `__pycache__/*.pyc`（二进制） | `staging/81-skills/*/scripts/__pycache__/` 有 6 个 `.pyc` |
| **R10** | 🟢 提示 | **`-fs` 两档图标的「同步」是靠单次运行同时产两份，没有任何测试锁定两者不互相污染**。跨包逐字节测试**只覆盖 12 枚层头像**，**不覆盖** `skill-icons{,-fs}.json` / `category-icons{,-fs}.json` | `layer-icons.spec.mjs` 只断言 `PLN-*`/`DOM-*` |

### F.3 三条最该先处理的风险（按「会造成静默错误」排序）

1. **R1 文档计数漂移** —— 它是**唯一会让人在错误的基线上做决策**的风险。本报告 §B 的所有数字都已按实测改写。
2. **R4 归位判定的语义校验不在包内** —— 页面全部数字的上游是「329 条挂载 + 逐字证据」，而这套判定**当前没有任何可重复执行的门禁**。`evidence` 一旦被改写，`--check` 只会说「lib 与 manifest 一致」，不会说「证据是假的」。
3. **R6 硬链接同步陷阱** —— 唯一被文档 + 历史事故双重证实的**生产事故**（`lib/catalog.js` 曾被归零）。任何新的接入工作都必须走 `pipeline.sh` 的 tmp+mv 原子替换，不能直接 `write`。

---

## 附录：一页速查

```
包         dsh-overseas-skills v0.1.0 · 无运行时依赖 · ESM · 42 MB
挂载       cordis.patch.yml insert{id: dsh-overseas-skills} · profile bundles 已含
路由       /api/dsh-overseas-skills/{list, fullstack-list, org, toggle, prompt-template, credential}
Slot       settings.section @overseas-skills(order 26) · @fullstack-skills(order 27)
           刻意不注册 conversation.input.dock（R1，有测试锁定）

事实的家    manifest/role-assignments.json  ← 253 条（overseas 223 + fs 30）/ 329 挂载
           manifest/taxonomy-v3.json        ← 8 场景 / 28 细分 / mapping 253
           ~/.dsh/.agent-presets/agt-*/     ← 50 岗位骨架（x_lute.plane/domain/order/skills.subset）
           ~/.dsh/skills/<name>/SKILL.md    ← description / 开关

构建产物    lib/catalog.js  ← build_preset_catalog.py   （SKILLS 223 / SKILLS_FS 30）
           lib/role-map.js ← build_role_map.py --check （全等投影）

页面实测    卡片 223 · 归位 209 · 未归岗 14 · 岗位 50（有卡 47）· 行数 319
           三态行级：本岗已接线 124 / 接线到别岗 52 / 未接线 143
           零卡岗位 AGT-013 / AGT-005 / AGT-049

测试       node --test test/*.spec.mjs  → 32 pass / 0 fail / 0 skip（198 ms）
跨包字节    test/layer-icons.spec.mjs 第 2 条 ⇄ packages/surfaces/dsh-algo-skills-local/src/layer-icons.ts

兄弟页     packages/surfaces/dsh-algo-skills-local  ← 论文卡（p2s-*，1338 张）已经在用同一套骨架
           /api/dsh-algo-skills/{tree,toggle,health} · join key = card.l3_business → role.responsibilities
```
