#!/usr/bin/env python3
"""凭证暴露面门禁 —— 补上 `scan_secrets.py` 结构性看不见的那一类。

## 为什么需要第二道扫描（适用范围，不是「再扫一遍」）

`paper-维护/scripts/scan_secrets.py` **只扫已入库（tracked）的文件**。它的退出码 2
（「一个文件都没扫到」）说的是这一点。因此它**回答不了**下面这个问题：

    工作区里有一把私钥，它「未被跟踪」但**也未被忽略** —— 一次 `git add -A` 就入库。

2026-09-13 实测（S8 报告 §1.3）：`~/project/Agent/Agent_agents/ai_video.pem` 所在仓库有 GitHub 远端、
**没有 `.gitignore`**，那把生产私钥正处于上述状态；同一把钥匙在另外 9 个位置都被 `*.pem` 规则拦住了。
**`scan_secrets.py` 对这两把（DDDD.pem / ai_video.pem）一把都查不到** —— 不是它失灵，
是它们从未入库，**不在它的适用范围里**。

⇒ 本脚本守的是那条缝：**「未被跟踪」被当成了「安全」**。

## 三个状态必须分开（否则会把公开证书算成事故）

| 状态 | 含义 | 判罚 |
|---|---|---|
| `TRACKED` | 密钥文件**已入库** | **红**（已在版本控制里，必须立刻处置） |
| `UNIGNORED` | 在仓库工作区里、未被跟踪、**也未被 git 忽略** | **红**（一次 `git add -A` 即入库 —— 本条正是 S8 实测的那条真暴露面） |
| `IGNORED` | 未被跟踪、**且**被忽略规则拦住 | 绿，**一等输出**（不是「没问题」，是「这道防线在工作」） |
| `PUBLIC_CERT` | 首行 `-----BEGIN CERTIFICATE-----`（**公开 X.509 证书，不是私钥**） | 绿，但**必须在白名单里逐条写明理由**，否则**红** |

⚠️ `PUBLIC_CERT` 这一态是本仓库既有教训的直接产物：`lute-momcozy-platform` 提交的
`config/ci-artifact-encryption-cert.pem` **首行是 `BEGIN CERTIFICATE`** —— 它是公开证书，
`README.md:107` 与四条测试都明写「只提交 public certificate」，删了会让 CI 解密接收方失效。
**「扫到 `.pem` 就报」是一条会把公开证书也算成事故的假阳性规则。**

## ⚠️ 两条口径（与 `scan_secrets.py` 不同，理由写在这里免得后人照抄）

1. **`0 个密钥文件` 不是我这里的失败**（`scan_secrets.py` 的退出码 2 在那边是对的，
   因为「一个文件都没扫到」意味着**扫描目标本身是空的**）。在这里，工作区里没有密钥文件
   **正是期望结果**。故本脚本的「我到底测了没有」由**走过多少个文件**来证明：
   `files_walked == 0` ⇒ **exit 2**（那说明根目录选错了，不是「干净」）。
2. **广扫与门禁必须分开**：`--check` 只扫**本仓库工作区**（确定性，可进门禁）；
   `--sweep-roots` 扫任意目录（回答「这台机器上有几份」），**默认恒 exit 0**，
   把结果当**一等读数**而不是判决 —— 因为它随环境漂移，跨机器结果不同。
   2026-09-13 实测的教训：S8 的清点在 `/Users/lute` 下做，而那份部署文档把密钥路径写死为
   `/Users/ll/Documents/VOA/ai_video.pem`（本机**没有** `/Users/ll`）⇒
   **那种清点回答的是「这台机器上有几份」，不是「一共存在几份」。**

## 退出码

`0` 全过 · `1` 判红 · `2` 输入没拿到（根目录走不到任何文件 ≠ 通过）· `3` 门禁内部错误（≠ 判红）

⚠️ **本脚本不读密钥内容**，只读**首行**（`BEGIN …` 头）—— 那是判定「私钥 vs 公开证书」的
唯一可靠依据（先例：本仓库 2026-09-13 由首行推翻了「DDDD.pem 是 SSH 公钥」的旧记载）。
其余只用元数据：路径 / 权限位 / 是否被 git 跟踪 / 是否被忽略。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# 与 scan_secrets.py 同族的密钥文件形态；**不含** .crt/.cer（那是证书的常规扩展名）
KEY_GLOBS = ("*.pem", "*.key", "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
             "*.p12", "*.pfx", "*.jks", "*.keystore")
PRIVATE_HEAD_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
CERT_HEAD_RE = re.compile(r"-----BEGIN CERTIFICATE-----")
DEFAULT_WHITELIST = HERE.parent / "data" / "key-exposure-whitelist.json"

SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache",
             ".pytest_cache", "dist", "build", ".next", ".turbo", ".idea"}


class InputMissing(Exception):
    """根目录走不到任何文件 —— 这是「没测到」，不是「干净」。"""


class GateError(Exception):
    """门禁自己坏了 —— 与「判红」必须分开。"""


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                           timeout=30)
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as exc:                       # git 不在 ⇒ 内部错误
        raise GateError(f"git 不可用：{exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GateError(f"git 超时：{' '.join(args)}") from exc


def repo_of(path: Path) -> Path | None:
    """文件**自己所在**的 git 仓库根；不在任何仓库里 ⇒ None。

    ⚠️ 这里踩过一个实测的坑（2026-09-13，第一次跑 `--sweep-roots` 就撞上）：
    第一版把 `cwd` 写成**扫描根目录**，于是 git 命令不是在**文件自己的仓库**里跑。
    扫 `~/project/Agent` 时，`Agent_agents/ai_video.pem` 的 `check-ignore` 直接返回 **128**，
    门禁报「内部错误」。那条 128 正是 S8 点名的那把私钥 —— **判据用错了执行位置**，
    与「判据只认一种路径」同族。现在每个文件各自的仓库里问，并显式处理「不在任何仓库里」。
    """
    code, out, _ = _git(["rev-parse", "--show-toplevel"], path.parent)
    if code != 0:
        return None
    top = out.strip().splitlines()
    return Path(top[-1]) if top else None


def _head_of(path: Path) -> str:
    """只读首行（≤64 字节）—— 判定私钥 vs 公开证书的唯一可靠依据。"""
    try:
        with path.open("rb") as fh:
            return fh.read(64).decode("utf-8", "replace").splitlines()[0].strip()
    except OSError:
        return ""


def walk_keys(root: Path) -> list[Path]:
    """列出 root 下所有密钥形态的文件（跳过常见构建/依赖目录）。"""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if any(Path(name).match(g) for g in KEY_GLOBS):
                out.append(Path(dirpath) / name)
    return sorted(out)


def classify(path: Path, repo: Path, whitelist: dict) -> dict:
    """把一个密钥文件判成五态之一。不读内容，只读首行与 git 元数据。

    `repo` 参数保留只为兼容调用口径，**实际一律用文件自己所在的仓库**（见 `repo_of`）。
    """
    head = _head_of(path)
    is_cert = bool(CERT_HEAD_RE.search(head))
    is_key = bool(PRIVATE_HEAD_RE.search(head))
    rel = _rel(path) if path.is_absolute() else str(path)

    rec = {"path": rel, "head": head[:48], "is_private_key": is_key, "is_public_cert": is_cert,
           "mode": None}
    try:
        rec["mode"] = oct(stat.S_IMODE(path.stat().st_mode))
    except OSError:
        rec["mode"] = None

    owner = repo_of(path)
    rec["repo"] = str(owner) if owner else None

    if is_cert:
        entry = whitelist.get(rel) or whitelist.get(Path(rel).name)
        rec["whitelisted"] = bool(entry)
        rec["whitelist_reason"] = (entry or {}).get("reason", "")
        rec["state"] = "PUBLIC_CERT"
        rec["ok"] = bool(entry)
        # ⚠️ `why` **两个分支都要给**：第一版只在「未白名单」分支给，于是白名单生效后
        #    `print_report` 撞 `KeyError: 'why'`，整个广扫**在中途崩掉**（退出码变成 1，
        #    而不是「判红 1 条」）—— 一条判据的显示字段缺失，把前面所有根目录的读数一起丢了。
        rec["why"] = (
            "公开证书（首行 BEGIN CERTIFICATE）且**已在白名单里写明理由**"
            f"：{(entry or {}).get('reason', '')[:80]}"
            if entry else
            "公开证书（首行 BEGIN CERTIFICATE）**未在白名单里写明理由** —— "
            "它可能确实无害，但「无害」必须由人写下理由，不能由扫描器默认放行")
        return rec

    if owner is None:
        rec.update({"tracked": False, "ignored": False})
        rec["state"] = "NO_REPO"
        rec["ok"] = True
        rec["why"] = ("**不在任何 git 仓库里** —— 不会被 git 带走（故不判红），"
                      "但也没有忽略规则在保护它：一旦那个目录以后被纳入 git，它立刻变成 UNIGNORED")
        return rec

    code_t, _, _ = _git(["ls-files", "--error-unmatch", "--", str(path)], owner)
    tracked = code_t == 0
    code_i, _, err_i = _git(["check-ignore", "-q", "--", str(path)], owner)
    # check-ignore: 0 = 被忽略，1 = 未被忽略，其他 = 出错
    if code_i not in (0, 1):
        raise GateError(f"git check-ignore 在 {owner} 里对 {rel} 返回 {code_i}"
                        f"（stderr: {err_i.strip()[:120]}）—— 无法判定是否被忽略")
    ignored = code_i == 0

    rec.update({"tracked": tracked, "ignored": ignored})
    if tracked:
        rec["state"] = "TRACKED"
        rec["ok"] = False
        rec["why"] = "密钥文件**已入库** —— 立刻处置（轮换 + 从历史里清除）"
    elif not ignored:
        rec["state"] = "UNIGNORED"
        rec["ok"] = False
        rec["why"] = ("在 git 仓库里、未被跟踪、**也未被忽略** —— 一次 `git add -A` 即入库"
                      "（S8 实测的真暴露面就是这一态）。最小修法是加一条忽略规则，**不要碰密钥本身**")
    else:
        rec["state"] = "IGNORED"
        rec["ok"] = True
        rec["why"] = "被忽略规则拦住 —— 这道防线在工作（**不是**「没问题」，是「防线有效」）"
    return rec


def scan(root: Path, whitelist: dict) -> dict:
    if not root.is_dir():
        raise InputMissing(f"根目录不存在：{root}")
    walked = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        walked += len(filenames)
    if walked == 0:
        raise InputMissing(f"{_rel(root)} 下走过 0 个文件 —— 根目录选错了（这**不是**「干净」）")
    keys = walk_keys(root)
    recs = [classify(p, root, whitelist) for p in keys]
    by = {}
    for r in recs:
        by[r["state"]] = by.get(r["state"], 0) + 1
    bad = [r for r in recs if not r["ok"]]
    return {"root": _rel(root), "files_walked": walked, "n_key_files": len(keys),
            "by_state": by, "records": recs, "n_red": len(bad)}


def load_whitelist(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"白名单读不动：{path}（{exc}）") from exc
    entries = doc.get("entries", doc)
    if not isinstance(entries, dict):
        raise GateError(f"白名单结构不对：{path} 顶层应为 {{\"entries\": {{路径: {{reason: …}}}}}}")
    return entries


def print_report(res: dict) -> None:
    print(f"根目录 {res['root']} · 走过 {res['files_walked']} 个文件 · "
          f"密钥形态文件 {res['n_key_files']} 个")
    order = ["TRACKED", "UNIGNORED", "PUBLIC_CERT", "IGNORED"]
    for st in order:
        n = res["by_state"].get(st, 0)
        if n:
            print(f"  {st:12s} {n}")
    for r in res["records"]:
        mark = "✅" if r["ok"] else "❌"
        print(f"{mark} [{r['state']}] {r['path']}  ({r['mode']})  首行 {r['head'][:34]!r}")
        # 显示层用 .get：字段缺失不该让**前面所有根目录的读数**一起丢掉
        # （实测过：一个 `why` 缺失让整个广扫在中途 KeyError 崩掉）
        print(f"      {r.get('why', '(无说明)')}")
    if res["n_red"] == 0:
        print("✅ 暴露面判据全过（工作区里没有被跟踪的密钥，也没有「未忽略」的密钥）")
    else:
        print(f"❌ 判红 {res['n_red']} 条")


# --------------------------------------------------------------------------
def selftest() -> int:
    """每条判据都配一份**能打红的构造样本** + 反向控制。

    ⚠️ 纪律（S12 台账 #25）：**新门禁的第一个用例应当是「把门禁自己改坏，看它报不报」**。
    这里用 `_sabotage` 参数直接注入三种错误判定，逐条验证它们会被抓住。
    """
    ok = True
    tmp = tempfile.mkdtemp(prefix="keyexp-")
    root = Path(tmp) / "repo"
    root.mkdir()
    (root / ".gitignore").write_text("*.pem\n*.key\n", encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for cmd in (["git", "init", "-q"], ["git", "add", ".gitignore"],
                ["git", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=root, capture_output=True, env=env)

    (root / "ignored-key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nxxxx\n",
                                          encoding="utf-8")
    (root / "unignored.key").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nxxxx\n",
                                        encoding="utf-8")
    (root / "cert.pem").write_text("-----BEGIN CERTIFICATE-----\nxxxx\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", "unignored.key"], cwd=root, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-qm", "oops"], cwd=root, capture_output=True, env=env)
    (root / ".gitignore").write_text("*.pem\n", encoding="utf-8")   # .key 不再被忽略

    res = scan(root, {})
    st = {Path(r["path"]).name: r["state"] for r in res["records"]}

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"{'✅' if good else '❌'} {label}" + (f"：{detail}" if detail else ""))

    check("构造样本：被忽略的私钥判 IGNORED（绿）",
          st.get("ignored-key.pem") == "IGNORED")
    check("构造样本：**已入库**的私钥判 TRACKED（红）",
          st.get("unignored.key") == "TRACKED",
          f"实得 {st.get('unignored.key')}")
    check("构造样本：公开证书未白名单 ⇒ 判红（PUBLIC_CERT 但 ok=False）",
          st.get("cert.pem") == "PUBLIC_CERT"
          and not [r for r in res["records"] if r["path"].endswith("cert.pem")][0]["ok"])

    # ⚠️ 变异探针必须在**它要打的那个状态还存在**的时候跑。第一版把三个探针全放在
    #    下面那段 fixture 改写之后，于是「把 TRACKED 判成绿」这条**没有对象可打**
    #    （那时 unignored.key 已被 git rm --cached 掉）—— 探针自己报「漏网」。
    #    那是我这个**测试**的缺陷，不是判据的：**变异必须证明它改变了真实取值。**
    print("--- 变异：把判定改坏，看它报不报（要求先证明变异施上了力）---")
    check("变异 [把 TRACKED 判成绿] 必须被抓住",
          _sabotage_probe(root, "tracked_ok", want_state="TRACKED") is True)

    # 第二条：未被跟踪 + 未被忽略（S8 那条真暴露面）—— 必须单独造，因为上面那张 .key 已入库
    (root / "unignored.key").unlink()
    subprocess.run(["git", "rm", "-q", "--cached", "unignored.key"], cwd=root,
                   capture_output=True, env=env)
    (root / "floating.key").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nxxxx\n",
                                       encoding="utf-8")
    res2 = scan(root, {})
    fl = [r for r in res2["records"] if r["path"].endswith("floating.key")]
    check("构造样本：未跟踪 + **未忽略** ⇒ 判 UNIGNORED（红）",
          bool(fl) and fl[0]["state"] == "UNIGNORED" and not fl[0]["ok"],
          f"实得 {fl[0]['state'] if fl else '（没扫到）'}")
    check("变异 [把 UNIGNORED 判成绿] 必须被抓住",
          _sabotage_probe(root, "unignored_ok", want_state="UNIGNORED") is True)

    # 白名单：写明理由后公开证书才放行
    res3 = scan(root, {"cert.pem": {"reason": "公开 X.509 证书，CI 解密接收方，见 README:107"}})
    ce = [r for r in res3["records"] if r["path"].endswith("cert.pem")][0]
    check("白名单：写明理由后公开证书放行（反向控制：不是一律放行）", ce["ok"] is True)
    check("变异 [把公开证书一律放行] 必须被抓住",
          _sabotage_probe(root, "cert_always_ok", want_state="PUBLIC_CERT") is True)

    # 输入没拿到：走过 0 个文件 ⇒ exit 2 语义
    empty = Path(tmp) / "empty"
    empty.mkdir()
    try:
        scan(empty, {})
        check("空根目录 ⇒ 必须报 InputMissing（不是「干净」）", False, "**没抓住**")
    except InputMissing:
        check("空根目录 ⇒ 必须报 InputMissing（不是「干净」）", True)

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _sabotage_probe(root: Path, kind: str, want_state: str) -> bool:
    """返回 True 表示「变异体确实漏掉了红」（即原判据有劲）。

    ⚠️ 两步都要断言，缺一不可（与 route_backlog / check_card_identity 同一条纪律）：
      ① **前提**：干净扫描里确实存在 `want_state` 这一态 —— 否则变异**没施上力**；
      ② **效果**：打了变异之后，那一态确实从红变绿 —— 否则变异没改变判定。
    第一版只做②，于是「变异没对象可打」被误报成「漏网」。**变异必须改变行为，不是改变能不能跑。**
    """
    real_classify = classify
    base = scan(root, {})
    if not any(r["state"] == want_state for r in base["records"]):
        print(f"   ⚠️ 变异 [{kind}] 没施上力：干净扫描里不存在 {want_state} 态")
        return False

    def mutant(path: Path, repo: Path, wl: dict) -> dict:
        r = real_classify(path, repo, wl)
        if kind == "tracked_ok" and r.get("tracked"):
            r["ok"] = True
        if kind == "unignored_ok" and r.get("state") == "UNIGNORED":
            r["ok"] = True
        if kind == "cert_always_ok" and r.get("state") == "PUBLIC_CERT":
            r["ok"] = True
        return r

    globals()["classify"] = mutant
    try:
        res = scan(root, {})
    finally:
        globals()["classify"] = real_classify
    # 变异体把目标态从红放绿 ⇒ 它改变了判定（说明原判据确实在拦它）
    return any(r["ok"] for r in res["records"] if r["state"] == want_state)


def main() -> int:
    ap = argparse.ArgumentParser(description="凭证暴露面门禁（未被忽略的密钥）")
    ap.add_argument("--check", action="store_true", help="扫本仓库工作区（确定性；进门禁用）")
    ap.add_argument("--root", default=str(REPO), help="被扫的根目录（默认本仓库）")
    ap.add_argument("--sweep-roots", nargs="*", default=None,
                    help="广扫若干目录：结果是一等读数，**默认恒 exit 0**（随环境漂移，不进门禁）")
    ap.add_argument("--strict", action="store_true", help="广扫时也按判红给退出码")
    ap.add_argument("--whitelist", default=str(DEFAULT_WHITELIST))
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    try:
        wl = load_whitelist(Path(args.whitelist))
        if args.sweep_roots is not None:
            roots = [Path(r).expanduser() for r in args.sweep_roots]
            docs, missing = [], []
            for r in roots:
                try:
                    docs.append(scan(r, wl))
                except InputMissing as exc:
                    missing.append(str(exc))
            for d in docs:
                print_report(d)
                print()
            if missing:
                print("⚠️ 走不到的根目录（登记，不当干净）：")
                for m in missing:
                    print(f"   - {m}")
            total = sum(d["n_red"] for d in docs)
            print(f"广扫合计：根目录 {len(docs)} 个 · 判红 {total} 条")
            if args.json_out:
                Path(args.json_out).write_text(
                    json.dumps({"sweep": docs, "missing_roots": missing},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            return 1 if (args.strict and total) else 0

        res = scan(Path(args.root), wl)
        print_report(res)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                           encoding="utf-8")
        return 1 if res["n_red"] else 0
    except InputMissing as exc:
        print(f"❓ 输入没拿到：{exc}")
        return 2
    except GateError as exc:
        print(f"💥 门禁内部错误：{exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
