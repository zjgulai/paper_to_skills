# PHASE6 · F8 / S12 —— 消费口闸门（Q5）：不被契约引用的卡，不进模型目录

> **状态**：✅ 交付（2026-09-13）。板上卡 B18（S12），优先级已由 F7 从「过渡期口径」升为
> **「契约层存在的理由」**。
> **一句话**：契约层现在**真的拦住了东西** —— 实测硬拦模式下，模型目录里的 p2s 条目从
> **147 条降到 32 条**，被移走的 115 条**逐条**能在账上追到「它没有任何契约引用」。

---

## 1 闸门落在哪：不是新增一道门，是给一道已存在的门补判据

−0.95 那次参数移植发生在卡**被模型调用**的那一刻，不在卡**入库**的那一刻（入卡取弱门槛是 Q4 的既定决定）。
而消费口本来就有闸门：

```
~/.dsh/skills/ 下 1338 张 p2s 卡  全部 disable-model-invocation: true   ← 不进全局模型目录
        │
        └─→ 50 个岗位 preset 的 skill-subset 白名单（respectFileFlags: false ⇒ 岗位装配权威）
                    ↑
                    └── 这里就是模型目录的唯一入口。S12 在这里加一个判据。
```

**判据**：白名单生成条件从「归位到本岗」改为「归位到本岗 **且** 该卡已被某份契约引用」。

判据只有**一处**实现：`packages/capabilities/dsh-paper2skills/lib/contract-gate.js`。
两个消费者共用它（风险 N2：判据只能有一处）：

| 消费者 | 角色 |
|---|---|
| `scripts/check-contract-gate.mjs`（产品侧，dsh-paper2skills） | **独立核对器**：读**已生成**的 preset 实物，不读生成器的意图 |
| `scripts/role-presets/generate.mjs`（产品侧，仓库根） | **白名单生成器**：`P2S_CONTRACT_GATE = count \| enforce \| off` |

---

## 2 账（可复跑）

```sh
cd packages/capabilities/dsh-paper2skills
node scripts/check-contract-gate.mjs --json-out generated/contract-gate.json   # 退出码 0/1/2/3
```

| 项 | 实测 | 口径 |
|---|---|---|
| 契约 | **54 份**（A 40 / B 14） | `07-资源库/contracts/{A,B}/CTR-*.md` |
| 契约引用条目 | **146 条** → 绑定 141 / **待装线 5** / **无法解析 0** | 逐条解析，见 §3 |
| 已装线卡 | **1338 张** | 产品侧 `classification.json` |
| ├ **已挂契约** | **29** | 在白名单里 **且** 被契约引用 |
| ├ **待挂契约** | **114** | 在白名单里、**没有**契约引用 ← S12 新增的那一个归位态 |
| └ 未接线 | **1195** | 没进任何岗位白名单（与 S12 无关，是 S13/Q4 的账） |
| 白名单里的 p2s 条目（跨岗位去重） | **143**（按行计 147） | 已挂契约 29 / 待挂契约 114 |
| 平台原生技能 + pb 手册技能 | **115**（按行 292） | **不归本闸门管** |

**账是机器算的，不是抄的**：把契约目录换空、或把某份契约的 `cards` 摘掉，账立刻跟着变
（selftest 变异 4/9「删契约后 alpha 掉到待挂契约」钉住这件事）。

---

## 3 本批的主要发现：一份契约的 `cards` 字段里有**三个命名空间**

这是 S12 动工前没被写下来的东西。同一张卡在链路里有三种写法，闸门必须逐条判到三态之一：

| 写法 | 例 | 出自 | 落点 |
|---|---|---|---|
| **slug** | `p2s-ad-attribution-modeling` | `classification.json[].slug`，也是技能目录名 | ✅ **bound**（白名单认的就是它） |
| **已装线 id** | `Skill-Ad-Attribution-Modeling` | `classification.json[].id` | ✅ **bound**（经 `id → slug` 归一化） |
| **精选线 id** | `Skill-CrossLingual-Semantic-Alignment` | `card-classification.json`（146 张） | ⚠️ **pending-install**（S5 换底前它不在技能目录里） |
| 编造 | `p2s-不存在的卡` | — | 🔴 **unresolvable** → 判红，**不静默丢弃** |

实测分布：**116 条 slug + 5 条精选线 id**（S1 原始口径里 5 条 `Skill-*` 一度被我手工核数时
误记成「无法解析」；接上 `card-classification.json` 后判定为**待装线**）。

**5 条待装线引用**（契约没错，是卡还没装线 —— 这是 S5 的前置依赖，**不判红**）：

| 契约 | 卡 |
|---|---|
| CTR-A-006 | `Skill-MAS-Consumer-Behavior-Simulation` |
| CTR-A-027 | `Skill-Multi-Warehouse-Allocation-LLM` |
| CTR-A-046 | `Skill-CrossLingual-Semantic-Alignment` ← 该契约因此**暂时零绑定** |
| CTR-B-028 | `Skill-TJAP-跨市场品类组合定价` |
| CTR-B-035 | `Skill-LLM-Personalized-Marketing-Copy-Generation` |

> 反后门：这 5 条**不算「已挂契约」**（卡不在消费链里就是不在），但**也不算缺陷**。
> 判定拆成两态的理由是：若把它判红，就会制造一条**永远修不掉**的红灯（要等 S5）；
> 若把它算成 bound，闸门就会放行一张模型根本看不见的卡 —— 那是假绿。
> **换底后若 id 与 slug 对不上，闸门会立刻报 `REF_UNRESOLVABLE`**（这条也进了 selftest）。

---

## 4 验收判据 ③：对照测量（**这是本批最实的一段**）

> 板上原话：「**有一张卡的对照测量证明它真的不进模型目录**」。
> 换措辞的证明不算证明 —— 所以做的是**同一份生成器、同一个技能库、同一份材料，
> 只翻一个环境变量**，然后逐字节比对两臂产出。

```sh
node scripts/role-presets/measure-contract-gate.mjs      # 产出在临时目录，绝不写 ~/.dsh
```

| 判据 | 内容 | 实测 |
|---|---|---|
| **A** | 线上 `~/.dsh/.agent-presets` 与 count 臂**逐岗 subset 相同** | ✅ 50/50（证明线上模型目录就是 count 臂的产物，对照才有意义） |
| **B** | 待挂契约的卡在 enforce 臂缺席、count 臂在位 | ✅ **115 条 / 跨 47 个岗位** |
| **C** | 已挂契约的卡两臂**都在位** | ✅ 29 张一张没动（防「把整条白名单都砍了也算通过」） |
| **D** | 非 p2s 技能两臂完全一致 | ✅ 292 条不变 |
| **E** | **反向**：摘掉引用它的那份契约，该卡**也**必须消失 | ✅ 摘掉 `CTR-A-006-情景模拟.md` ⇒ `p2s-sc-whatif-scenario-analysis-engine` 从模型目录消失；对照卡 `p2s-agent-error-budget`（契约仍在）留在目录里 |
| **F** | `agent.cordis.yml` 的 skill-subset 行与 manifest 逐岗一致 | ✅ 账不是第二份事实源 |

**物证**（报告里必须能点出具体一张卡、具体一个岗位）：

```
卡：p2s-agentic-pnl-analyst        岗位：agt-001
  count 臂   agent.cordis.yml 里 有 该卡
  enforce 臂 agent.cordis.yml 里 无 该卡
  该卡的契约引用数：0（因此被移除）
```

**p2s 条目：147 → 32（−115）；非 p2s 条目：292 → 292。**
⇒ 收窄的是**未经契约引用的算法卡**，平台原生技能与手册技能不受影响。

⚠️ **这条测量能证明什么、不能证明什么**：它证明的是**模型目录的实物**（preset 的 subset 行）
按契约引用变化；**没有**在真实会话里跑「模型还会不会挑到这张卡」——那需要重启宿主，
不在本轮范围。这是「目录层的对照测量」，不是「模型行为的对照测量」。

---

## 5 过渡期口径 vs 硬拦：差多少，为什么现在不硬拦

板上 Q5 原文：**「过渡期不把白名单条件升级为硬拒绝，先以归位态可见 + 计数可查的方式跑一轮。」**

| 模式 | 白名单 | 谁来用 |
|---|---|---|
| `count`（**默认**） | 不变（147 条） | 线上。归位态写进 manifest 的 `x_lute.skills.contract_gate`，页面显示「待挂契约」 |
| `enforce` | 移除 115 条 → 32 条 | **对照测量的仪器**，以及未来真正的硬拦 |
| `off` | 不读契约 | 显式退出，收尾会打印警告（**不是默认值**） |

**为什么默认不硬拦**：54 份契约只覆盖 **1/8 条 FLOW**。现在硬拦等于让 **80%** 的模型目录
因为「剩下的 7 条 FLOW 还没写契约」而消失 —— 那不是闸门在起作用，是闸门在替一份未完成的
工作背锅。**但闸门本身已经是真的了**：`enforce` 一开，卡真的出目录（§4）。

---

## 6 页面：归位态的第 4 枚徽标

`dsh-overseas-skills` 的岗位树卡片上，新增一枚**正交于接线三态**的徽标：

```
接线三态（既有）：本岗已接线 / 接线到 N 岗 / 未接线     ← 回答「挂上了吗」
契约态（S12 新增）：待挂契约 / 闸门未记账              ← 回答「挂它算不算数」
```

**刻意不合并成一维**：一张卡可以「本岗已接线 **且** 待挂契约」—— 过渡期 114 张里的绝大多数
正是这个形态，合并后就答不了「挂了，但挂它算不算数」。

- `lib/preset-roles.js`：读 `x_lute.skills.contract_gate`；缺字段留 **null**（不是空数组）
- `lib/org-tree.js`：`contractGate.{mode, pendingIndex, pendingSkills, boundSkills, rowsPending, rowsBound, rolesReporting}` + `contractStatus()` 三态
- `lib/client.js`：`待挂契约` 徽标 + **`闸门未记账`** 徽标
- ⚠️ **「没记账」≠「记了是零」**：老版 manifest 没有这一段时页面必须说「闸门未记账」，
  否则会把「没法判」画成「都挂了契约」。`mode` 取 `count|enforce|null|mixed`，
  **`mixed`（一批 preset 混着两个版本）本身就是一条要报的缺陷**。

线上实测：`50 个岗位全部报账 / 待挂契约 115 行（114 张去重）/ 已挂契约 32 行（29 张）`。

---

## 7 线上 preset 重生成：**可证明是纯增量**

页面要读到 `contract_gate` 就得重跑生成器。重跑前做了备份并逐文件比对：

| 文件 | 结果 |
|---|---|
| `preset.yml` × 50 | **逐字节相同** |
| `agent.cordis.yml` × 50 | **逐字节相同**（含 `agt-033` 手插行的原位与上下文） |
| `manifest.json` × 50 | 唯一差异 = **新增** `x_lute.skills.contract_gate` |

---

## 8 本批被自己撞出来的三条缺陷（已进台账 #23–#25）

这一批的特殊之处：**三条都不是别人撞出来的，是新仪器自己在真数据上跑出来的。**

### #23 假绿：`x_lute.skills.subset` 与 `skills.subset`（🔴 最危险的一条）

`readPresetSubsets()` 第一版只读顶层 `manifest.skills.subset`，而实际落点是
**`x_lute.skills.subset`**（平台扩展命名空间）。读不到 ⇒ **50 个岗位全读空**，
而闸门**照样打出一张完整的账**：

```
已装线卡总数 1338 / 已挂契约 0 / 待挂契约 0 / 未接线 1338
```

**这张账看起来完全正常。** 之所以没有变成报告里的假数字，只因为动工前手工算过一遍
（29/114/1195）—— 也就是说，**发现它靠的是人肉交叉核对，不是判据**。

- ✅ 修：两个已知落点都试；**两个都拿不到 ⇒ exit 2**（不是「零接线」）
- ✅ 锁定：变异 9/9 双向（`x_lute` 下能读到；落在未知键下必须 `missing: true`）

### #24 删别人的数据：生成器静默擦掉手插的本机装配行

重生成时 `agt-033` 里按 ADR-0061 手插的 `product-kol-hunter` 行（本机产品装配，
不进仓库出货物）**被整文件重写静默删掉**。页面上的表现正是 AGENTS.md 记的那条
「产品卡点了打不开」。

**文件里其实有一句注释预告了这件事**：「运行 generate.mjs 会重写本文件，届时需重新插入本行」。
⇒ **写了警告不等于有了判据。**

- ✅ 已恢复：`agt-033/agent.cordis.yml` 与事故前**逐字节一致**（先 diff 定位、再按原位插回、再整文件 cmp）
- ✅ 已修：抽成 `scripts/role-presets/unmanaged-rows.mjs` —— 非生成器产出的行块**逐块带过**、
  **插回原位**（前驱行 id 决定位置，因为 Cordis 的行是**顺序敏感**的挂载次序）、
  位置退化时**点名报出**。8 条用例 / 变异 **3/3**。

### #25 假仪表：新门禁自己的 selftest 里三处摆设

三轮变异测试逐轮暴露：

| 轮次 | 抓住率 | 漏网的 |
|---|---|---|
| 第一轮 | 8/11 | `main()` 的 preset exit 2 守卫 / 契约目录 exit 2 守卫 / unresolvable 判红 |
| 第二轮 | 10/11 | classification 读空守卫 |
| 第三轮 | **13/13** | — |

漏网的都是同一形态：**判据在 `main()` 里，而 selftest 只测库函数**。
把守卫改成 `if (false)`，selftest 照样全绿。

- ✅ 修：改用**子进程跑真 CLI + 端到端夹具**，直接断言退出码；并加**反向控制**
  「干净夹具必须 exit 0」，防「什么输入都判红」的假仪表。
- 📌 **纪律（本批新增）**：新门禁的第一个 selftest 用例应当是「**把门禁自己改坏，看它报不报**」
  —— 而不是「跑通一遍」。

---

## 9 复算不出来的数

板上 S12 卡里记着「实测 1338 张 p2s 卡全部 `disable-model-invocation: true`，
只经 preset 白名单逐岗露出（**wired 127 / wiredAnywhere 258 / 未接线 978**）」。

| 说法 | 实测 | 结论 |
|---|---|---|
| wiredAnywhere **258** | **258** ✅ | 复现成功 = 白名单里全部 distinct 技能（p2s **143** + 平台原生 **115**） |
| wired **127** | **143** | ❌ 复算不出来 |
| 未接线 **978** | **1195** | ❌ 复算不出来；且 127 + 978 = **1105 ≠ 1338** —— 两个数既不等于实测，彼此也不构成一个划分 |

处置：报告与文档一律用**现算值 + 写明口径**（`1338 = 143 白名单 + 1195 未接线`）。
**手写的分型数字就是一张过期照片（风险 N6 的第 N 次命中）。**

---

## 10 复现（完整命令）

```sh
# ① 闸门自证：13 组 / 27 条断言，每份变异各自被对应用例打红
cd packages/capabilities/dsh-paper2skills
node scripts/check-contract-gate.mjs --selftest

# ② 真实对账（读线上 preset 实物）
node scripts/check-contract-gate.mjs --json-out generated/contract-gate.json

# ③ 硬拦判定（不改盘）：会移除哪些条目、跨多少岗位
node scripts/check-contract-gate.mjs --enforce

# ④ 对照测量（六条判据 A–F，产出在临时目录）
cd ../..
node scripts/role-presets/measure-contract-gate.mjs

# ⑤ 白名单生成器（count 默认 / enforce 硬拦 / off 显式退出）
node scripts/role-presets/generate.mjs --dry-run
P2S_CONTRACT_GATE=enforce node scripts/role-presets/generate.mjs --dry-run

# ⑥ 手插行保护
node --test scripts/role-presets/unmanaged-rows.test.mjs

# ⑦ 页面侧
cd packages/capabilities/dsh-overseas-skills && npx tsc --noEmit -p tsconfig.json && node --test test/*.spec.mjs
```

**本轮全部验证**：闸门 selftest 13/13 组 · 变异 13/13 抓住 · 对照测量 A–F 全绿 ·
`unmanaged-rows` 8 用例 / 变异 3/3 · `dsh-overseas-skills` **49/49** + tsc 干净 ·
线上 preset 重生成证明为**纯增量**（`agent.cordis.yml`/`preset.yml` 逐字节不变）。

---

## 11 未做与交接

| 项 | 目的地 |
|---|---|
| 其余 7 条 FLOW 的方案域 + 85 份契约 | S1 续批（闸门不用改，契约一多账自动变） |
| 硬拦的**实际启用** | 待 139 份契约收敛到足够覆盖后再决定（本轮只交仪器与测量） |
| 「模型还会不会挑到这张卡」的真实会话测量 | 需重启宿主，不在本轮范围 |
| 精选线 5 条 `pending-install` 引用的绑定 | S5（p2s 换底装精选线）；对不上时闸门会报 `REF_UNRESOLVABLE` |
| 入卡弱门槛（L3 属 A/B 类即放行） | S13 |
| `agt-033` 手插的 `product-kol-hunter` 行**位置** | 按 ADR-0061 应移到 profile 的 `cordis.patch.yml`；本轮只保证它不再被静默删除 |
