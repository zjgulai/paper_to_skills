#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scan_secrets.py — 推送前的凭证扫描门禁（C-级硬拦截）

## 为什么有这个脚本

2026-09-13 首次推送到 GitHub 前做全库扫描，命中两类硬编码凭证：

  1. 一枚 DeepSeek key（`sk-` 前缀，此处**刻意不写全**，见下方「固件为什么要拼」）
     —— 嵌在 `playbook/` 下 6 个文件的**内嵌 Python 模板**里，3 种转义形态
        （裸双引号 / JSON 转义 \\" / HTML 实体 &quot;）各一份。
  2. 一个飞书机器人 webhook（`open.feishu.cn/open-apis/bot/v2/hook/<uuid>`）
     —— 已经在公开仓库 `main` 上裸奔了两个月，本次才被发现。

### ⚠️ 固件为什么要「拼」出来，而不是直接写字面量

自测固件里若出现**完整**的凭证字面量，会同时踩两个坑，实测都踩过：

  ① 本文件被自己扫出命中 → 门禁**恒红** → 而永远红的门禁等于没有门禁
     （大家会学会忽略它的输出，正如 ignore 一个 flaky test）。
     首版实测 12 处自命中。
  ② `git filter-repo --replace-text` 清理历史时，会把**固件里的真值一并改写**
     成 `***REMOVED-…***` —— 于是「用来抓该凭证的回归测试」被「清除该凭证的操作」
     打坏了，而且**自测从绿变红**这件事没有任何人预期到。
     这不是假想：本文件第 309 行就真实发生过。

故所有固件一律经 `_fx(*parts)` 拼接 —— 源码里不存在完整凭证，运行时构造出的
样本仍是真值。`--selftest` 用例 7 是**元级**用例，专门锁死「本文件自己必须干净」。

第 1 条是**这次推送会新造成**的泄露；第 2 条是**既成事实**。两者的共同点是：
`repo_health.py` 的 C1–C8、K1/K2 三个门禁**全都没有覆盖**——
它们查重复卡、frontmatter、路径、围栏、registry、门禁时效、卫生、段落完整性，
**没有一项查凭证**。所以仓库带着 5 个文件、6 处硬编码 key 一路 commit 了 28 次，
每一次门禁都是绿的。

这与本项目之前 4 次「提升来自修门禁而非改资产」同源：
**没有被测的东西不会自己变好，只会安静地烂下去。**

## 关键设计：三种转义形态必须都认

第一版扫描如果只写 `sk-[A-Za-z0-9]{20,}` 是能命中的，但如果只写
`api_key="sk-..."` 这种完整形态，就会漏掉 JSON 转义与 HTML 实体两种变体。
实测 `playbook-data.json` 里存的是 `api_key=\\"sk-...\\"`，
`Skill-RankGPT-*.html` 里存的是 `api_key=&quot;sk-...&quot;`。
→ 故本脚本**先**做转义归一化（把 \\" 与 &quot; 还原成 "），**再**匹配，
   这样同一个模式覆盖全部形态，不需要为每种转义各写一条规则。

## 与 .gitignore 的分工

`.gitignore` 按**文件名**拦（`*.pem` / `*.key`），拦不住「key 写在 .html 里」。
本脚本按**内容**拦。两者互补，缺一不可 —— 本次命中的 6 个文件全是 `.html` / `.json`，
`.gitignore` 对它们完全无效。

## 用法

    # 扫描全部 tracked 文件（推送前必跑）
    python3 paper2skills-skills/paper-维护/scripts/scan_secrets.py

    # 只扫指定路径
    python3 paper2skills-skills/paper-维护/scripts/scan_secrets.py --path playbook/

    # 自证：用构造样本证明每条检测器真的会报警
    python3 paper2skills-skills/paper-维护/scripts/scan_secrets.py --selftest

    # 输出 JSON（给 CI / pre-push hook 用）
    python3 paper2skills-skills/paper-维护/scripts/scan_secrets.py \\
        --json-out paper2skills-research/data/health/secrets.json

退出码：0 = 干净；1 = 命中凭证；2 = 脚本自身出错。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 检测规则
# --------------------------------------------------------------------------
# 说明：每条规则的 pattern 作用在**转义归一化之后**的文本上，因此只需考虑
#       最朴素的写法，不必为 \\" / &quot; / &#34; 各写一份。

RULES: list[tuple[str, re.Pattern[str], str]] = [
    # ⚠️ 顺序即优先级：**具体规则必须排在泛化规则之前**。
    #    ANTHROPIC_KEY 必须早于 OPENAI_STYLE_KEY —— 后者的 `sk-[A-Za-z0-9_-]{20,}`
    #    能把 `sk-ant-api03-...` 整条吞掉，于是 Anthropic 的 key 被报成 OpenAI 的
    #    （--selftest 用例 3 实测抓出）。同理 GENERIC 必须垫底。
    ("PRIVATE_KEY_BLOCK",
     re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
     "PEM/PKCS 私钥正文。任何形式的私钥都不得入库"),

    ("ANTHROPIC_KEY",
     re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
     "Anthropic API key"),

    ("OPENAI_STYLE_KEY",
     re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
     "OpenAI / DeepSeek / Moonshot 等 sk- 前缀 API key"),

    ("GITHUB_TOKEN",
     re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
     "GitHub personal access / OAuth / refresh token"),

    ("AWS_ACCESS_KEY",
     re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
     "AWS access key id"),

    ("HUGGINGFACE_TOKEN",
     re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
     "HuggingFace access token"),

    ("GOOGLE_API_KEY",
     re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"),
     "Google API key"),

    ("FEISHU_WEBHOOK",
     re.compile(r"open\.feishu\.cn/open-apis/bot/v2/hook/[0-9a-fA-F-]{20,}"),
     "飞书自定义机器人 webhook（拿到即可往群里发消息）"),

    ("SLACK_WEBHOOK",
     re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]{20,}"),
     "Slack incoming webhook"),

    ("DINGTALK_WEBHOOK",
     re.compile(r"oapi\.dingtalk\.com/robot/send\?access_token=[A-Za-z0-9]{20,}"),
     "钉钉机器人 webhook"),

    ("JWT",
     re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
     "JWT（可能内嵌可用的会话/服务凭证）"),

    # ⚠️ 用户名的长度下限必须小：`postgresql://<user>:<pass>@host` 里 user 只有 5 字符，
    #    写成 `[^\s"'<>]{10,}:[^\s"'<>]{6,}@` 会因前缀不足 10 而**整个漏掉**这条真凭证。
    #    （这是 --selftest 用例 3 实测抓出来的，不是理论担忧。）
    ("CONNECTION_STRING",
     re.compile(r"(?i)\b(?:mysql|postgres(?:ql)?|mongodb(?:\+srv)?|redis|amqp)://[^\s\"'<>/@]{1,64}:[^\s\"'<>/@]{4,}@"),
     "带密码的连接串"),

    ("GENERIC_SECRET_ASSIGNMENT",
     re.compile(r"(?i)\b(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|passwd|password)\b\s*[:=]\s*[\"'][^\"'\s]{16,}[\"']"),
     "疑似凭证赋值（变量名 + 16 字符以上的字面量）"),
]

# 归一化：把各种转义还原成朴素引号，让同一模式覆盖全部形态
ESCAPE_FORMS: list[tuple[str, str]] = [
    ('\\"', '"'),        # JSON / JS 字符串里的转义双引号
    ('&quot;', '"'),     # HTML 实体
    ('&#34;', '"'),
    ('&#x22;', '"'),
    ("\\'", "'"),        # 转义单引号
    ('&apos;', "'"),
]

def _fx(*parts: str) -> str:
    """把自测固件**拼**出来 —— 源码里不存在完整凭证。

    见模块 docstring「固件为什么要拼」。一句话：直接写字面量会让本文件被自己
    扫出命中（门禁恒红），而且 `git filter-repo` 清历史时会把固件真值一并改写掉，
    连带打坏回归测试。两件事都实测发生过。
    """
    return "".join(parts)


# 墓碑：`git filter-repo --replace-text` 把清掉的凭证替换成的标记。
# ⚠️ **它不是凭证，它是「这里曾经有过凭证、已被清掉」的物证。**
#    不归一化它，就会得到一个荒谬的读数：跑完清理之后，残留的墓碑本身被当成新凭证，
#    于是「清干净了」看起来比「没清」还红。2026-09-13 实测：本仓库历史里 3 份
#    `playbook/skills/*.html` 的旧版本命中 GENERIC_SECRET_ASSIGNMENT，命中的值就是
#    `***REMOVED-DEEPSEEK-KEY***`。
# ⚠️ 反向要求（漏洞 #10 的教训：**新增豁免本身就是一次放水**）：墓碑**旁边**的真凭证
#    必须照样命中。`--selftest` 用例 21 就是为此而设（墓碑 + 真 key 同块）。
TOMBSTONE_RE = re.compile(r"\*\*\*REMOVED-[A-Za-z0-9_.\-]*\*\*\*")


def tombstones(text: str) -> list[str]:
    """列出文本里的墓碑标记。

    单独暴露成函数（而不是让 `scan_text` 顺带返回）是为了让**调用方把它当读数报出来**：
    「这一处曾经有过凭证」是信息，不该被归一化悄悄吃掉。见 `check_history_secrets.py`。
    """
    return TOMBSTONE_RE.findall(text)


# 白名单：确认无害/已失效的值，写在这里而不是写宽规则
# ⚠️ 每加一条都必须写清「为什么它无害」，否则白名单就是下一个后门（漏洞 #10 的教训）。
# ⚠️ 尤其禁止把**规则级**的东西塞进来：把 PEM 私钥头加进白名单等于放行所有私钥。
#    宁可在源头改文本（见 .gitignore 里那句「此处刻意不写完整的 PEM 头字面量」）。
# ⚠️ 键值同样经 `_fx` 拼接：白名单条目本身也是「文件里出现完整凭证形态」。
ALLOWLIST: dict[str, str] = {
    # --- 占位示例，不是真凭证 ---
    _fx("sk-your-", "key-here"): "文档里的占位示例",
    _fx("sk-", "x" * 24): "文档里的占位示例",

    # --- 2026-09-13 推送前全量扫描的 3 条真命中，逐条判定无害 ---
    _fx("sk-fake-key-", "for-mock-testing"):
        "paper2skills-code/.../test_mock.py 的 mock 固件；值本身写着 fake，"
        "且该测试在断网语义下运行，不发真实请求",
    _fx("postgresql://", "voc_user", ":", "voc_pass", "@"):
        "Superset_BI_SOP.md 的操作示例，用户名/密码均为 voc_user/voc_pass 占位串，"
        "主机是 host.docker.internal（本地容器），不含任何真实库凭证",
    _fx("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.",
        "eyJzdWIiOiJwaG9uZSs3OTM0ODE4MTkxIiwiZXhwIjoxNjg0NDEyMDk4fQ.",
        "ad3t3S_Xj7YhoDDFZeW4BlVL4dNniMdfaXC1143fbzw"):
        "arXiv 2510.00615 论文全文表格里印着的示例 access_token —— 该论文已公开发表，"
        "此值随论文本身公开；且 exp=1684412098（2023-05）早已过期。"
        "它是论文内容，不是本项目的凭证",
}


def normalise(text: str) -> str:
    """把各种转义形态还原成朴素引号，使同一规则覆盖全部写法。"""
    out = text
    for src, dst in ESCAPE_FORMS:
        if src in out:
            out = out.replace(src, dst)
    return out


def is_probably_binary(data: bytes) -> bool:
    """NUL 字节是二进制的最可靠信号（比扩展名可靠）。"""
    return b"\x00" in data[:8192]


def scan_text(text: str) -> list[dict]:
    """对单份文本跑全部规则，返回命中列表（同一处凭证只报一条）。

    两个关键处理，都是 --selftest 逼出来的：

    1. **白名单先做精确替换，再匹配**。若在匹配之后才判 `value in ALLOWLIST`，
       那么同一处凭证会被 GENERIC_SECRET_ASSIGNMENT 再命中一次而绕开豁免
       （自测用例 5 实测失败）。先替换成哨兵，任何规则都看不到它。
    2. **重叠命中按规则优先级去重**。`api_key="sk-..."` 同时命中
       OPENAI_STYLE_KEY 与 GENERIC_SECRET_ASSIGNMENT —— 同一个密钥报两条，
       报告会虚高且难读（用例 2 实测得到 2 条而非 1 条）。
       ⚠️ 去重必须**按规则优先级遍历**，不能按命中位置排序后去重：
       GENERIC 的匹配从 `api_key=` 就开始，位置比 `sk-` 更靠前，
       按位置排会让泛化规则赢（用例 1 实测）。
    """
    norm = normalise(text)
    for allowed in ALLOWLIST:
        if allowed in norm:
            norm = norm.replace(allowed, "\x00ALLOWLISTED\x00")
    # 墓碑（filter-repo 的清理标记）先换成定长哨兵再匹配。
    # ⚠️ 必须**先替换再匹配**，与 ALLOWLIST 同理：若在匹配之后才判，同一处会被
    #    GENERIC_SECRET_ASSIGNMENT 从 `api_key=` 起头再命中一次（用例 21 覆盖）。
    if TOMBSTONE_RE.search(norm):
        norm = TOMBSTONE_RE.sub("\x00TOMBSTONE\x00", norm)

    kept: list[dict] = []
    for rule_id, pattern, desc in RULES:  # RULES 已按「具体 → 泛化」排序
        for m in pattern.finditer(norm):
            if any(not (m.end() <= k["_start"] or m.start() >= k["_end"]) for k in kept):
                continue  # 与更具体的命中重叠 → 让具体规则说话
            kept.append({
                "rule": rule_id,
                "desc": desc,
                "match_preview": _preview(m.group(0)),
                "line": norm.count("\n", 0, m.start()) + 1,
                "_start": m.start(),
                "_end": m.end(),
            })

    kept.sort(key=lambda h: h["_start"])
    for h in kept:
        h.pop("_start", None)
        h.pop("_end", None)
    return kept


def _preview(value: str, keep: int = 8) -> str:
    """只露头尾，避免扫描报告本身成为第二份泄露源。"""
    if len(value) <= keep * 2 + 3:
        return value
    return f"{value[:keep]}…{value[-4:]} (len={len(value)})"


def collect_files(root: Path, target: str | None) -> list[Path]:
    """默认取 git tracked 文件；--path 时递归取磁盘文件。"""
    if target:
        p = (root / target) if not os.path.isabs(target) else Path(target)
        if p.is_file():
            return [p]
        return sorted(f for f in p.rglob("*") if f.is_file())
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root,
            capture_output=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 不在 git 仓库里就退回全盘遍历（但跳过 .git）
        return sorted(
            f for f in root.rglob("*")
            if f.is_file() and ".git" not in f.parts
        )
    names = [n for n in out.split(b"\0") if n]
    return [root / n.decode("utf-8", "surrogateescape") for n in names]


def scan(root: Path, target: str | None = None) -> dict:
    files = collect_files(root, target)
    hits: list[dict] = []
    scanned = skipped = 0
    for f in files:
        try:
            data = f.read_bytes()
        except OSError:
            skipped += 1
            continue
        if is_probably_binary(data):
            # 二进制仍要查私钥头（PEM 是 ASCII，会出现在二进制容器里）
            if b"-----BEGIN" in data and b"PRIVATE KEY-----" in data:
                hits.append({
                    "file": str(f.relative_to(root)) if root in f.parents else str(f),
                    "rule": "PRIVATE_KEY_BLOCK",
                    "desc": "二进制文件内含 PEM 私钥块",
                    "match_preview": "-----BEGIN … PRIVATE KEY-----",
                    "line": None,
                })
            skipped += 1
            continue
        scanned += 1
        text = data.decode("utf-8", "ignore")
        for hit in scan_text(text):
            rel = str(f.relative_to(root)) if root in f.parents else str(f)
            hits.append({"file": rel, **hit})
    return {
        "files_total": len(files),
        "files_scanned_text": scanned,
        "files_skipped_binary": skipped,
        "rules": len(RULES),
        "findings": hits,
        "findings_count": len(hits),
        "clean": not hits,
    }


# --------------------------------------------------------------------------
# 自证
# --------------------------------------------------------------------------

def selftest() -> int:
    """用构造样本证明每条规则真的会报警，且不会误报干净文本。

    本仓库的规矩：门禁必须先自证可信，再拿去判别人（见 CLAUDE.md 门禁体系一节）。
    """
    failures: list[str] = []

    # --- 用例 1：与真实历史值**同形**的样本必须命中（这次事件的回归测试）---
    # ⚠️ 固件必须**拼出来**，不能写成整串字面量。
    #    写整串的话，本文件自己就会被自己扫出两条命中，于是门禁永远红着 ——
    #    而一个永远红的门禁等于没有门禁（大家会学会忽略它）。
    #
    #    ⚠️⚠️ 这里**真的被 filter-repo 打坏过一次**：hook 固件原先写成
    #    `<hook 路径> + "<uuid 第 1 段>" + ...`，随后跑 `git filter-repo --replace-text` 清历史时，
    #    这个「用来抓该 webhook 的固件」被替换成了 `***REMOVED-FEISHU-HOOK***` ——
    #    自测从绿变红，而没有任何人预期到「清凭证的操作」会连带打坏「抓凭证的测试」。
    #
    #    ⚠️⚠️⚠️ **2026-09-13 第二次修正：固件里放「真值」本身就是一个错误。**
    #    原先这里放的是**真实**的 key 与 hook 值（理由写的是「真实历史值必须命中」）。
    #    但规则是 `\bsk-[A-Za-z0-9_-]{20,}` 这种**形状**判据，它不知道也不需要知道值是不是真的：
    #    同形假值与真值的检出结果**逐字节等价**（把 fixture 换成同形假值后，
    #    下面 4 条用例与转义归一化用例全绿，实测）。
    #    而代价是实打实的：`_fx("sk-", "<值>")` 拆开只骗得过**模式扫描器**，
    #    骗不过任何一个读源码的人 —— 一行 `+` 就能还原文。
    #    于是「用来防泄露的固件机制」把真值以「一行可还原」的形式留在了**公开**仓库里，
    #    而 `scan_secrets.py` 自己报「✅ 未发现凭证」——因为拆开了，它按定义看不见。
    #    ⇒ 值维度改由 `data/exposed-credential-registry.json` + 判据 C 常驻守着
    #      （只存 len+sha256，命中即红、无豁免）。
    #      分工：**这里管形状，登记表管值。**
    real_key = _fx("sk-", "f" * 20 + "0" * 12)          # 同形假值：sk- + 32 位
    real_hook = _fx("open.feishu.cn/open-apis/bot/v2/hook/",
                    "00000000" + "-1111-2222" + "-3333-444444444444")
    for label, sample, expect in [
        ("裸双引号 key", f'api_key="{real_key}"', "OPENAI_STYLE_KEY"),
        ("JSON 转义 key", f'api_key=\\"{real_key}\\"', "OPENAI_STYLE_KEY"),
        ("HTML 实体 key", f'api_key=&quot;{real_key}&quot;', "OPENAI_STYLE_KEY"),
        ("飞书 webhook", f"const _H = 'https://{real_hook}';", "FEISHU_WEBHOOK"),
    ]:
        got = {h["rule"] for h in scan_text(sample)}
        if expect not in got:
            failures.append(f"用例1[{label}]：期望命中 {expect}，实得 {got or '空'}")

    # --- 用例 2：三种转义形态必须归一化成同一结果 ---
    forms = [
        f'api_key="{real_key}"',
        f'api_key=\\"{real_key}\\"',
        f'api_key=&quot;{real_key}&quot;',
    ]
    counts = {len(scan_text(f)) for f in forms}
    if counts != {1}:
        failures.append(f"用例2[转义归一化]：三种形态命中数应同为 1，实得 {counts}")

    # --- 用例 3：每类凭证各造一个真样本，必须全部命中 ---
    for label, sample, expect in [
        ("PEM 私钥", _fx("-----BEGIN RSA ", "PRIVATE KEY-----\nMIIEow==\n"), "PRIVATE_KEY_BLOCK"),
        ("GitHub token", _fx("token: ghp_", "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"), "GITHUB_TOKEN"),
        ("AWS key", _fx("aws_access_key_id = AKI", "AIOSFODNN7EXAMPLE"), "AWS_ACCESS_KEY"),
        ("HuggingFace", "hf_" + "a" * 34, "HUGGINGFACE_TOKEN"),
        ("Google", "AIza" + "B" * 35, "GOOGLE_API_KEY"),
        ("Anthropic", "sk-ant-" + "c" * 30, "ANTHROPIC_KEY"),
        ("Slack", _fx("https://hooks.slack.com/services/", "T000/B000/XXXXXXXXXXXXXXXXXXXX"), "SLACK_WEBHOOK"),
        ("钉钉", "https://oapi.dingtalk.com/robot/send?access_token=" + "d" * 30, "DINGTALK_WEBHOOK"),
        ("连接串", _fx("postgresql://", "admin", ":", "hunter2secret", "@db.internal:5432/prod"), "CONNECTION_STRING"),
        ("凭证赋值", _fx('client_secret = "', '8f3a9c2e1b7d4a6f0e5c', '"'), "GENERIC_SECRET_ASSIGNMENT"),
    ]:
        got = {h["rule"] for h in scan_text(sample)}
        if expect not in got:
            failures.append(f"用例3[{label}]：期望命中 {expect}，实得 {got or '空'}")

    # --- 用例 4：干净文本不得误报（防「规则太宽 → 满屏假阳性 → 没人再看报告」）---
    clean_samples = [
        "# 用环境变量读取凭证，不要硬编码\nimport os\nKEY = os.environ.get('DEEPSEEK_API_KEY', '')\n",
        "论文里 sample size = 12345，accuracy 92.2%，提升 15%。",
        "参见 https://open.feishu.cn/document/server-docs/im-v1/message/create 的官方文档。",
        "skill_audit.json 里记录了 146 张卡的通过率。",
        "api_key=__import__('os').environ.get('DEEPSEEK_API_KEY', '')",
        "hash = 'a3f5'  # 短标识，不是凭证",
    ]
    for i, s in enumerate(clean_samples):
        got = scan_text(s)
        if got:
            failures.append(f"用例4[干净样本{i}]误报：{[h['rule'] for h in got]}")

    # --- 用例 5：白名单只豁免**精确值**，不得顺手豁免同前缀的其他值 ---
    WL = _fx("sk-test-placeholder-", "value-1234567890")
    ALLOWLIST[WL] = "自测用占位符"
    try:
        if scan_text('api_key="' + WL + '"'):
            failures.append("用例5：白名单精确值未被豁免")
        if not scan_text('api_key="' + WL + 'XXXX"'):
            failures.append("用例5：白名单把同前缀的其他值也豁免了（白名单过宽）")
    finally:
        ALLOWLIST.pop(WL, None)

    # --- 用例 6：二进制里的私钥块必须被检出（不能被「跳过二进制」放过）---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "blob.bin"
        p.write_bytes(b"\x00\x01\x02" + b"-----BEGIN RSA " + b"PRIVATE KEY-----\nMIIE\n\x00\xff")
        res = scan(Path(td), "blob.bin")
        if res["findings_count"] != 1:
            failures.append(f"用例6：二进制内私钥未检出（实得 {res['findings_count']} 条）")

    # --- 用例 7（元级）：扫描器**自己**必须干净 ------------------------------
    # 为什么值得单列一条：门禁如果恒红，就没人会读它的输出 —— 与 flaky test 同一下场。
    # 实测踩过两次，都不是假想：
    #   ① 首版把固件写成完整字面量 → 本文件被自己扫出 **12 处**命中，
    #      于是 `scan_secrets.py`（不带 --path）**永远 exit 1**；
    #   ② `git filter-repo --replace-text` 把固件里的真值改写成 `***REMOVED-…***`，
    #      自测从绿变红。
    # 这条用例把「本文件干净」变成一条会失败的断言，而不是一句注释里的期望。
    self_path = Path(__file__).resolve()
    self_hits = scan_text(self_path.read_text(encoding="utf-8"))
    if self_hits:
        rules = sorted({h["rule"] for h in self_hits})
        failures.append(
            f"用例7：扫描器自身被扫出 {len(self_hits)} 处命中 {rules} —— "
            f"全库扫描会因此恒红；请把固件改为经 _fx() 拼接")

    # --- 用例 7b：**整个门禁家族**都必须干净，不只本文件 -----------------------
    # ⚠️ 2026-09-13 实测踩到：本文件干净了，而**同一轮新写的另外三个文件**（另两个扫描器的
    #    源码 + 两张判据表 + 那份安全报告）各自把字面量又写了一遍，`scan_secrets.py` 当场判红
    #    —— **#72「教训写在隔壁文件里＝没写」的第三次**，这次是**在写下这条教训的同一轮里**犯的。
    #    故把范围从「本文件」扩到「家族」，让复发当场被抓住，而不是靠人记得。
    fam = [
        Path(__file__).resolve().parent / "check_history_secrets.py",
        Path(__file__).resolve().parents[3] / "paper2skills-research" / "scripts"
        / "check_key_exposure.py",
        Path(__file__).resolve().parents[3] / "paper2skills-research" / "data"
        / "history-secrets-whitelist.json",
        Path(__file__).resolve().parents[3] / "paper2skills-research" / "data"
        / "exposed-credential-registry.json",
    ]
    for f in fam:
        if not f.exists():
            failures.append(f"用例7b：门禁家族文件缺失（{f.name}）—— 判据的输入不在，"
                            f"「没查」不许当成「干净」")
            continue
        hits = scan_text(f.read_text(encoding="utf-8"))
        if hits:
            rules = sorted({h["rule"] for h in hits})
            failures.append(
                f"用例7b：{f.name} 被扫出 {len(hits)} 处命中 {rules} —— "
                f"家族文件里不许出现凭证字面量（拼接口径见用例 7）")

    # --- 用例 8（元级）：全库扫描不得把「本文件」算成问题源 -------------------
    # 用例 7 只看本文件；这里确认「扫自己时也不会因为 ALLOWLIST 里留了完整字面量而漏报」。
    # （白名单条目本身若写字面量，虽然会被 scan_text 豁免，但 filter-repo 仍会改写它。）
    if re.search(r'ALLOWLIST[^}]*?"[^"]*://[^"]*:[^"]*@"', self_path.read_text(encoding="utf-8"), re.S):
        failures.append("用例8：ALLOWLIST 里仍留有完整的连接串字面量（应经 _fx 拼接）")

    # --- 用例 21：墓碑不得被当成凭证；但墓碑**旁边**的真凭证必须照样命中 --------
    # 为什么需要它：`git filter-repo --replace-text` 把清掉的凭证替换成
    # `***REMOVED-…***`。清完之后，**墓碑自己**长得就像一条赋值语句 ——
    # 实测 3 份 `playbook/skills/*.html` 的旧版本因此命中 GENERIC_SECRET_ASSIGNMENT。
    # 不处理它，会出现荒谬读数：清理做得越彻底，门禁越红。
    #
    # ⚠️ 反向控制是这条用例的**主体**，不是补充（漏洞 #10 的教训：**新增豁免本身就是一次放水**）。
    #    「墓碑被豁免」与「墓碑成了万能免检牌」只差一行代码 —— 后者才是真正的后门。
    tomb = _fx("***REMOVED-", "DEEPSEEK-KEY", "***")
    if scan_text(f'api_key="{tomb}"'):
        failures.append("用例21a：墓碑标记本身不得被判成凭证")
    if tombstones(f'api_key="{tomb}"') != [tomb]:
        failures.append("用例21b：墓碑必须同时作为**读数**被报出来（豁免≠看不见）")
    # 反向控制一：墓碑不能把同一行/同一块里的真凭证一起免掉
    realish = _fx("sk-", "z" * 32)
    mixed = scan_text(f'old="{tomb}"  new="' + realish + '"')
    if not any(h["rule"] == "OPENAI_STYLE_KEY" for h in mixed):
        failures.append("用例21c：墓碑**旁边**的真 key 必须照样命中（豁免不得成为免检牌）")
    # 反向控制二：墓碑形状不许被放宽成通配 —— 少一个星号就不是 filter-repo 的标记
    # （载荷必须 ≥16 字符，否则 GENERIC 规则本来就够不着，这条反向控制会变成**恒真摆设**）
    near = _fx("**REMOVED-", "X" * 20, "***")
    if not scan_text(f'api_key="{near}"'):
        failures.append("用例21d：墓碑模式必须精确（`**REMOVED-…` 不是墓碑，不得免检）")

    if failures:
        print("❌ --selftest 未通过：")
        for f in failures:
            print("   ·", f)
        return 1

    print("✅ --selftest 通过（8 组用例）")
    print(f"   规则数 {len(RULES)}；覆盖 3 种转义形态、10 类凭证、干净样本零误报、"
          f"白名单精确匹配、二进制内私钥检出、**扫描器自身干净**")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="推送前凭证扫描门禁")
    ap.add_argument("--path", help="只扫描该路径（默认扫描全部 git tracked 文件）")
    ap.add_argument("--root", default=".", help="仓库根目录")
    ap.add_argument("--json-out", help="把结果写成 JSON")
    ap.add_argument("--selftest", action="store_true", help="用构造样本自证检测器可信")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve()
    result = scan(root, args.path)

    # ⚠️ 「一个文件都没扫到」**不等于**「干净」。
    #    首次跑回归测试时 `--root` 指向一个空 git 仓库，git ls-files 返回空列表，
    #    脚本照样打印「✅ 未发现凭证」并 exit 0 —— 一个什么都没测的绿灯。
    #    这与 CLAUDE.md 记录的漏洞 #7（quote_check 把 NO_QUOTES 算作通过）同一病根：
    #    **没东西可查 ≠ 查过了没问题**。故这里必须硬失败，且用退出码 2 与「发现凭证」区分开。
    if result["files_total"] == 0:
        print("❌ 未扫描到任何文件 —— 这不是「干净」，是「没测」。")
        print(f"   root={root}  target={args.path or '(git tracked)'}")
        print("   若目标是空仓库或无 tracked 文件，请改用 --path 显式指定目录。")
        return 2

    print(f"扫描 {result['files_total']} 个文件"
          f"（文本 {result['files_scanned_text']} / 二进制 {result['files_skipped_binary']}）"
          f"，规则 {result['rules']} 条")

    if result["clean"]:
        print("✅ 未发现凭证")
    else:
        print(f"\n❌ 发现 {result['findings_count']} 处疑似凭证：\n")
        by_file: dict[str, list[dict]] = {}
        for h in result["findings"]:
            by_file.setdefault(h["file"], []).append(h)
        for f, hs in sorted(by_file.items()):
            print(f"  {f}")
            for h in hs:
                loc = f":{h['line']}" if h.get("line") else ""
                print(f"     [{h['rule']}]{loc}  {h['match_preview']}")
            print(f"     → {hs[0]['desc']}")
        print("\n处置：换成读环境变量 / DSH 凭据服务；")
        print("      已在服务端轮换过的历史值，在 ALLOWLIST 里登记并写明理由。")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON 已写入 {out}")

    return 0 if result["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
