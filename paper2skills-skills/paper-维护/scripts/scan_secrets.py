#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scan_secrets.py — 推送前的凭证扫描门禁（C-级硬拦截）

## 为什么有这个脚本

2026-09-13 首次推送到 GitHub 前做全库扫描，命中两类硬编码凭证：

  1. `***REMOVED-DEEPSEEK-KEY***`（DeepSeek key）
     —— 嵌在 `playbook/` 下 6 个文件的**内嵌 Python 模板**里，3 种转义形态
        （裸双引号 / JSON 转义 \\" / HTML 实体 &quot;）各一份。
  2. `open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-...`（飞书机器人 webhook）
     —— 已经在公开仓库 `main` 上裸奔了两个月，本次才被发现。

第 1 条是**这次推送会新造成**的泄露；第 2 条是**既成事实**。两者的共同点是：
`repo_health.py` 的 C1–C8、K1/K2 三个门禁**全都没有覆盖**——
它们查重复卡、frontmatter、路径、围栏、registry、门禁时效、卫生、段落完整性，
**没有一项查凭证**。所以仓库带着 5 个文件、6 处硬编码 key 一路 commit 了 27 次，
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

    # ⚠️ 用户名的长度下限必须小：`postgresql://admin:hunter2secret@host` 里 admin 只有 5 字符，
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

# 白名单：确认无害/已失效的值，写在这里而不是写宽规则
# ⚠️ 每加一条都必须写清「为什么它无害」，否则白名单就是下一个后门（漏洞 #10 的教训）。
# ⚠️ 尤其禁止把**规则级**的东西塞进来：把 PEM 私钥头加进白名单等于放行所有私钥。
#    宁可在源头改文本（见 .gitignore 里那句「此处刻意不写完整的 PEM 头字面量」）。
ALLOWLIST: dict[str, str] = {
    # --- 占位示例，不是真凭证 ---
    "sk-your-key-here": "文档里的占位示例",
    "sk-xxxxxxxxxxxxxxxxxxxxxxxx": "文档里的占位示例",

    # --- 2026-09-13 推送前全量扫描的 3 条真命中，逐条判定无害 ---
    "sk-fake-key-for-mock-testing":
        "paper2skills-code/.../test_mock.py 的 mock 固件；值本身写着 fake，"
        "且该测试在断网语义下运行，不发真实请求",
    "postgresql://voc_user:voc_pass@":
        "Superset_BI_SOP.md 的操作示例，用户名/密码均为 voc_user/voc_pass 占位串，"
        "主机是 host.docker.internal（本地容器），不含任何真实库凭证",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiJwaG9uZSs3OTM0ODE4MTkxIiwiZXhwIjoxNjg0NDEyMDk4fQ."
    "ad3t3S_Xj7YhoDDFZeW4BlVL4dNniMdfaXC1143fbzw":
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

    # --- 用例 1：真实历史值必须命中（这次事件的回归测试）---
    # ⚠️ 固件必须**拼出来**，不能写成整串字面量。
    #    写整串的话，本文件自己就会被自己扫出两条命中，于是门禁永远红着 ——
    #    而一个永远红的门禁等于没有门禁（大家会学会忽略它）。
    #    拼接后源码里不存在完整凭证，运行时构造出的样本仍然是真值。
    real_key = "sk-" + "aae11f4438f943b9bf32a233620437bd"
    real_hook = ("open.feishu.cn/open-apis/bot/v2/hook/"
                 + "***REMOVED-FEISHU-HOOK***")
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
        ("PEM 私钥", "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n", "PRIVATE_KEY_BLOCK"),
        ("GitHub token", "token: ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "GITHUB_TOKEN"),
        ("AWS key", "aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "AWS_ACCESS_KEY"),
        ("HuggingFace", "hf_" + "a" * 34, "HUGGINGFACE_TOKEN"),
        ("Google", "AIza" + "B" * 35, "GOOGLE_API_KEY"),
        ("Anthropic", "sk-ant-" + "c" * 30, "ANTHROPIC_KEY"),
        ("Slack", "https://hooks.slack.com/services/T000/B000/XXXXXXXXXXXXXXXXXXXX", "SLACK_WEBHOOK"),
        ("钉钉", "https://oapi.dingtalk.com/robot/send?access_token=" + "d" * 30, "DINGTALK_WEBHOOK"),
        ("连接串", "postgresql://admin:hunter2secret@db.internal:5432/prod", "CONNECTION_STRING"),
        ("凭证赋值", 'client_secret = "8f3a9c2e1b7d4a6f0e5c"', "GENERIC_SECRET_ASSIGNMENT"),
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
    ALLOWLIST["sk-test-placeholder-value-1234567890"] = "自测用占位符"
    try:
        if scan_text('api_key="sk-test-placeholder-value-1234567890"'):
            failures.append("用例5：白名单精确值未被豁免")
        if not scan_text('api_key="sk-test-placeholder-value-1234567890XXXX"'):
            failures.append("用例5：白名单把同前缀的其他值也豁免了（白名单过宽）")
    finally:
        ALLOWLIST.pop("sk-test-placeholder-value-1234567890", None)

    # --- 用例 6：二进制里的私钥块必须被检出（不能被「跳过二进制」放过）---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "blob.bin"
        p.write_bytes(b"\x00\x01\x02-----BEGIN RSA PRIVATE KEY-----\nMIIE\n\x00\xff")
        res = scan(Path(td), "blob.bin")
        if res["findings_count"] != 1:
            failures.append(f"用例6：二进制内私钥未检出（实得 {res['findings_count']} 条）")

    if failures:
        print("❌ --selftest 未通过：")
        for f in failures:
            print("   ·", f)
        return 1

    print("✅ --selftest 通过（6 组用例）")
    print(f"   规则数 {len(RULES)}；覆盖 3 种转义形态、10 类凭证、干净样本零误报、"
          f"白名单精确匹配、二进制内私钥检出")
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
