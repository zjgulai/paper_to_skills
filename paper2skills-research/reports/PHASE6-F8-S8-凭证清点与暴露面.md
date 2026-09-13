# PHASE6 S8 · 凭证清点与暴露面（2026-09-13）

> 本文件是 **B14（S8）** 的现场记录。**所有者已决定：轮换不进 PHASE6 本轮**，登记为独立安全条目。
> 本文件只做**鉴定与记账**：全程**只查元数据（路径 / 权限位 / mtime / 哈希 / git 跟踪状态）与引用关系，
> 未读取任何密钥内容**。唯一读到「内容」的地方是 `.pem` 的**首行**（`BEGIN …` 头），
> 那是判定「私钥 vs 证书」的唯一可靠依据（先例：本仓库 2026-09-13 由首行推翻「它是公钥」的旧记载）。

## 一、现场清点（2026-09-13 复测，命令可复跑）

### 1.1 `DDDD.pem` —— 腾讯云轻量服务器生产 SSH **私钥**，4 份

| # | 路径 | 权限 | mtime | sha256 前 12 |
|---|------|------|-------|--------------|
| 1 | `~/Downloads/DDDD.pem` | `600` | 2026-07-11 15:13 | `db144e603280` |
| 2 | `~/project/思维模型/DDDD.pem` | `600` | 2026-07-11 15:13 | `db144e603280` |
| 3 | `~/project/Agent/lute-momcozy-audit/DDDD.pem` | `600` | 2026-07-11 15:13 | `db144e603280` |
| 4 | **`~/project/paper_to_skills/DDDD.pem`** | `600` | **2026-09-12 03:07** | `db144e603280` |

四份哈希相同 ⇒ **同一把**。第 4 份的 mtime 是本仓库**首次纳入 git 的前一天**。

⚠️ **CLAUDE.md 旧记载「另有 3 份副本」不成立**：实测 **4 份**（3 份在别处 + 1 份在本仓库根目录）。
旧记载漏掉的正是**最危险的那一份**（它在一个后来变成 PUBLIC 存在的仓库工作区里）。

### 1.2 `ai_video.pem` —— **另一把**，10 份

| # | 路径 | 权限 | mtime | sha256 前 12 |
|---|------|------|-------|--------------|
| 1 | `~/project/VOA/ai_video.pem` | `600` | 2026-07-02 10:15 | `e45784bf8f80` |
| 2 | `~/project/Agent/Agent_agents/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 3 | `~/project/Agent/archives/Agent_agents-2026-06-04/untracked/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 4 | `~/project/Agent/product/ana_caijing/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 5 | `~/project/Agent/product/ai_product_select/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 6 | `~/project/Agent/product/data_achieve/meltwater/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 7 | `~/project/Agent/product/html_everything/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 8 | `~/project/Agent/product/ai_employ_platform/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 9 | `~/project/Agent/Brand_agent/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |
| 10 | `~/project/Agent/lute-momcozy-audit/ai_video.pem` | `600` | 2026-04-30 03:56 | `e45784bf8f80` |

十份哈希相同 ⇒ 同一把，**与 `DDDD.pem` 不是同一把**。
⚠️ 任务卡原写「散布 10+ 个位置」，**实测恰好 10**（`find` 覆盖 `~/Downloads ~/project ~/Desktop ~/Documents`，`maxdepth 5–6`）。
**用途仍未鉴定** —— 下面 §三 给出鉴定方法，本轮未执行（需连接其对应服务）。

### 1.3 全量扫描：所有 `.pem` / `.key` / `id_rsa*` 的 git 暴露面

扫描范围 `~/Downloads ~/project ~/Desktop`（`maxdepth 6`），对每个命中文件问三个问题：
① 它在某个 git 仓库里吗；② 它被跟踪了吗（`git ls-files --error-unmatch`）；③ 它被忽略了吗（`git check-ignore`）。

| 文件 | 所在仓库 | tracked | ignored |
|------|---------|---------|---------|
| `Downloads/DDDD.pem` | （不在仓库内） | — | — |
| `project/Agent/archives/…/untracked/ai_video.pem` | （不在仓库内） | — | — |
| `project/Agent/product/ana_caijing/ai_video.pem` | （不在仓库内） | — | — |
| `project/思维模型/DDDD.pem` | `思维模型` | no | ✅ |
| `project/Agent/Brand_agent/ai_video.pem` | `Brand_agent` | no | ✅ |
| `project/Agent/lute-momcozy-audit/{ai_video,DDDD}.pem` | `lute-momcozy-audit` | no | ✅ |
| `project/Agent/product/ai_employ_platform/ai_video.pem` | `ai_employ_platform` | no | ✅ |
| `project/Agent/product/ai_product_select/ai_video.pem` | `ai_product_select` | no | ✅ |
| `project/Agent/product/data_achieve/meltwater/ai_video.pem` | `meltwater` | no | ✅ |
| `project/Agent/product/html_everything/ai_video.pem` | `html_everything` | no | ✅ |
| `project/paper_to_skills/DDDD.pem` | `paper_to_skills` | no | ✅ |
| `project/VOA/ai_video.pem` | `VOA` | no | ✅ |
| **`project/Agent/Agent_agents/ai_video.pem`** | **`Agent_agents`** | no | **❌ 未被忽略** |
| `project/Agent/lute-momcozy-platform/config/ci-artifact-encryption-cert.pem` | `lute-momcozy-platform` | **YES** | ❌ 未被忽略 |

## 二、两条结论（一条真暴露面 · 一条「不是漏洞」）

### 2.1 🔴 真暴露面：`Agent_agents` 里那把私钥**未被忽略**

`~/project/Agent/Agent_agents/ai_video.pem` 所在仓库：

- 远端 = `https://github.com/zjgulai/AI_Company_Person.git`（**有远端**）；
- 该仓库**没有 `.gitignore`**（实测 `grep -iE "pem|\.key|secret" .gitignore` 无输出，文件不存在）；
- **当前 0 个 `.pem` 被跟踪**。

⇒ 这把生产私钥处于「**未跟踪但未被忽略**」态：**一次 `git add -A` 即可入库并推送**。
同一把钥匙在另外 9 个位置都被 `*.pem` 规则拦住了，**只有这一处没有**。
（本仓库刚发生过一次同族事故：`playbook/` 下的 DeepSeek key 与飞书 webhook 一路 commit 了 28 次，
每次门禁都是绿的 —— 见 CLAUDE.md §凭证事件。）

**最小、可逆、非破坏的修法**：给 `Agent_agents` 加一条 `*.pem` 忽略规则（不碰密钥本身）。
⚠️ 本轮**未执行**：跨仓库改动不在 PHASE6 授权范围内，登记待所有者一句话。

### 2.2 ✅ 一条「不是漏洞」（登记以防后人误修）

`lute-momcozy-platform` **确实**提交了 `config/ci-artifact-encryption-cert.pem`（`tracked=YES`），
但它的**首行是 `-----BEGIN CERTIFICATE-----`** —— 是**公开 X.509 证书**，不是私钥。

三重佐证（不是我一个人说它不是）：

1. `docs/superpowers/plans/2026-07-29-ci-minimum-permission-repair.md:61` —
   「Create: `config/ci-artifact-encryption-cert.pem` (**public certificate only**)」；
2. `README.md:107` — 「仓库仅提交 `config/ci-artifact-encryption-cert.pem` public certificate；
   对应 private key 不得写入仓库、artifact 或日志」；
3. `tests/tencent-workflow-contract.test.mjs` 有四条断言把「这个文件存在 / 且被 workflow 的
   `-recip` 引用」锁住。

**它不是泄露，不要删** —— 删了会让 CI 的 artifact 解密接收方失效，并打红该仓库自己的测试。
（本节之所以要写，是因为「扫到 `.pem` 就报」是一条会把公开证书也算成事故的假阳性规则；
本仓库已四次栽在「判据的适用范围被默认成了全体」上。）

## 三、仍在所有者手上的两件事（按不可颠倒的顺序）

### 3.1 `DDDD.pem` 与 `ai_video.pem` 的轮换

**顺序不可颠倒**（颠倒的代价不是立刻报错，而是「某天突然登不上」）：

1. 生成新密钥对；
2. 用**【旧密钥】登录**，把新公钥**追加**到服务器 `authorized_keys`；
3. 用**新密钥验证能登录**；
4. **才**移除旧公钥；
5. 销毁全部旧私钥副本 + 加 `~/.ssh/config` 条目。

⚠️ **跳过 1–4 直接删副本是危险的**：若服务器上仍留着这把公钥，删除本地副本不会立刻报错。
⚠️ 腾讯云轻量服务器有**控制台 VNC 兜底** —— 这是「可以放心轮换」的前提，不是「可以不轮换」的理由。
⚠️ **轮换必须由所有者执行或走正当凭证通道；agent 不读取、不使用这两把私钥。**

### 3.2 `ai_video.pem` 的用途鉴定（**2026-09-13 已做**，只查引用关系、未读密钥内容）

做法：在引用它的项目里搜 `ai_video.pem` 与 `ssh -i` / `scp -i`（**不打开密钥文件**）。

| 问题 | 实测答案 |
|---|---|
| 哪台主机 | **`ubuntu@101.34.52.232`** —— 腾讯云轻量服务器 |
| 哪个服务 | **BOS Web 生产部署**：`/opt/bos-web`（nginx + docker-compose） |
| 依据 | `VOA/docs/superpowers/plans/2026-07-02-bos-tencent-lighthouse-deployment.md` 里 **12+ 处** `ssh -i …/ai_video.pem ubuntu@101.34.52.232 …` 与 `scp -i …`（部署脚本原文） |
| 该仓库的暴露面 | `VOA` 远端 = `https://github.com/lillian-maker/BOS`；`ai_video.pem` **未被跟踪、且被忽略**（`git check-ignore` ✅） |

**结论（对轮换计划有直接影响）**：`ai_video.pem` **不是一把闲置的旧钥匙，而是一台在跑的服务器
（`101.34.52.232`）的部署私钥** —— 它应当与 `DDDD.pem` **同批、按同一顺序**轮换，且**优先级不低于**后者。

⚠️ **同时暴露了§1.3 清点的一个适用范围边界（登记不重判）**：那份部署文档把密钥路径写死为
**`/Users/ll/Documents/VOA/ai_video.pem`** —— 而本机**没有 `/Users/ll`**（`ls /Users` 只有 `lute` 与 `Shared`）。
⇒ 该路径属于**另一台机器**（或历史别名）。本报告的清点在 `/Users/lute` 下做（`maxdepth 5–6`），
**它回答的是「这台机器上有几份」，不是「一共存在几份」** —— 与 §四 那条推论同一条：
**先问仪器能不能看见，再下结论。**

`lute-momcozy-platform` 的 `.pem` **不在**此项内（那是公开证书，见 §2.2）。

## 四、扫描器的**适用范围**（这一节比上面任何一条都重要）

`paper2skills-skills/paper-维护/scripts/scan_secrets.py` **查不到这两把密钥中的任何一把** ——
不是它失灵，是它**只扫已入库的文件**（退出码 2 的语义正是「一个文件都没扫到 ≠ 干净」）。

⇒ 本仓库的凭证防线目前有**两个不重叠的覆盖面**：

| 仪器 | 覆盖面 | 盲区 |
|------|--------|------|
| `scan_secrets.py` | **已入库**文本里的 10 类凭证（13 条规则） | 工作区里**任何**未入库文件；二进制；`.pem` 类密钥文件本身 |
| `check_key_exposure.py`（**2026-09-13 已落地，见下**） | 工作区里的 `*.pem`/`*.key`/`id_rsa*` 等**及其 git 暴露面** | **不读内容**（只读首行判「私钥 vs 公开证书」）；不查已入库文本里的其他凭证类型（那是上一行的活） |

📌 **与铁律 1/2 同一条推论**：每次「某个东西不存在」的结论，都要先问「我用的仪器能看见它吗」。
`scan_secrets.py` 报 0 命中，回答的是「**已入库的东西**里没有凭证」，
**不是**「这台机器上没有凭证泄露风险」。

### 4.1 **待办已交付**：第二条扫描门禁 `check_key_exposure.py`（2026-09-13 主控落地）

判据「工作区里存在**未被 git 忽略**的 `*.pem`/`*.key`/`id_rsa*` ⇒ 判红；公开证书需显式白名单 + 理由」已实现，
接进验收面 **L14a / L14b**。

**五态（分开是必须的，否则会把公开证书算成事故）**

| 状态 | 含义 | 判罚 |
|---|---|---|
| `TRACKED` | 密钥**已入库** | **红** |
| `UNIGNORED` | 在 git 仓库里、未跟踪、**也未被忽略**（一次 `git add -A` 即入库） | **红**（正是 §2.1 那条真暴露面） |
| `PUBLIC_CERT` | 首行 `BEGIN CERTIFICATE` —— 公开证书 | 白名单里有理由 ⇒ 绿；**没有 ⇒ 红**（「无害」必须由人写下，不许扫描器默认放行） |
| `IGNORED` | 未跟踪且被忽略 | 绿，**一等输出**（不是「没问题」，是「防线有效」） |
| `NO_REPO` | 不在任何 git 仓库里 | 绿（不会被 git 带走），但登记「也没有忽略规则在保护它」 |

**⭐ 它在真语料上独立复现了本节 §1.3 / §2.1 的两条结论（零硬编码）**：

```
--sweep-roots ~/project/Agent     走过 39700 个文件 · 密钥形态文件 11 个
  ❌ [UNIGNORED]   …/Agent_agents/ai_video.pem                ← §2.1 那条真暴露面
  ✅ [PUBLIC_CERT] …/lute-momcozy-platform/config/ci-artifact-encryption-cert.pem
                   （已在白名单写明理由：README:107 + 计划文档 + 四条测试）  ← §2.2 那条「不是漏洞」
  ✅ [IGNORED] ×7（Brand_agent / lute-momcozy-audit ×2 等）
  ✅ [NO_REPO] ×1（archives/…/untracked/）
  广扫合计：判红 1 条 —— 与 S8 人工清点的结论**逐条一致**
```

**白名单**：`paper2skills-research/data/key-exposure-whitelist.json` —— 每条必须写 `reason` + `evidence` + `expires_when`
（永久豁免＝台账 #5 那种腐烂）。已用 `!` 规则**提升入库**（不入库＝新克隆没有它＝等于没交付）。
**这条白名单在真语料上被撞过一次**（momcozy 公开证书），不是只在构造样本上验过。

**⚠️ 两条口径（与 `scan_secrets.py` 不同，理由写在这里免得后人照抄）**

1. **「0 个密钥文件」在这里不是我这里的失败**。`scan_secrets.py` 的退出码 2 在那边是对的
   （「一个文件都没扫到」= **扫描目标本身是空的**）；但在这里，工作区没有密钥文件**正是期望结果**。
   故本门禁的「我到底测了没有」由**走过多少个文件**证明：`files_walked == 0` ⇒ **exit 2**。
2. **广扫与门禁必须分开**。`--check` 只扫**本仓库工作区**（确定性，进门禁）；
   `--sweep-roots` 扫任意目录（回答「这台机器上有几份」），**默认恒 exit 0**，`--strict` 才判红 ——
   因为它随环境漂移。**把两者混起来，就会造出一条「结果取决于谁在哪台机器上跑」的门禁。**
   ⚠️ 这条不是理论担忧：§3.2 实测到部署文档写死 `/Users/ll/…` 而本机没有 `/Users/ll` ⇒
   **清点回答的是「这台机器上有几份」，不是「一共存在几份」。**

**⚠️ 开发中它自己撞出的两个缺陷（都已修，登记以示同族）**

- **判据用错了执行位置**：第一版把 `git check-ignore` 的 `cwd` 写成**扫描根目录**，而不是**文件自己所在的仓库**。
  扫 `~/project/Agent` 时，那把 `Agent_agents/ai_video.pem` 直接返回 **128** ⇒ 门禁报「内部错误」。
  **那条 128 正是 S8 点名的那把私钥** —— 与「判据只认一种路径」同族。已改为每个文件各自的仓库里问，
  并显式分出 `NO_REPO` 态。
- **一个显示字段缺失把全部读数丢掉**：白名单生效后 `PUBLIC_CERT` 分支漏了 `why` 键 ⇒
  `print_report` 撞 `KeyError`，整个广扫**在中途崩掉**（退出码变成 1，而不是「判红 1 条」）。
  已修两处：两个分支都给 `why`，且**显示层用 `.get`** —— 字段缺失不该让**前面所有根目录的读数**一起丢。

**自检**：`--selftest` **9/9**（构造 git 仓库造出 TRACKED / UNIGNORED / IGNORED / PUBLIC_CERT 四态 +
白名单反向控制「写明理由才放行，不是一律放行」+ 空根目录必须报 `InputMissing`），
外加**三份「把门禁自己改坏」的变异**（TRACKED 判绿 / UNIGNORED 判绿 / 公开证书一律放行）。
⚠️ 变异探针第一版**自己写错了**：三个探针全放在 fixture 改写之后跑，于是「把 TRACKED 判成绿」
**没有对象可打**，探针误报「漏网」——**那是测试的缺陷不是判据的**。现改为
`_sabotage_probe(root, kind, want_state)`：**先断言干净扫描里确实存在该状态**（否则算「变异没施上力」），
**再断言打了变异后它确实由红变绿**。**变异必须改变行为，不是改变能不能跑。**
