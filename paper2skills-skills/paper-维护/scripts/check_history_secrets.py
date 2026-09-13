#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_history_secrets.py — 凭证扫描的**第三条缝：git 对象库**

## 为什么还需要第三个扫描器

本仓库已有两套凭证口径，它们覆盖的是**同一件事的两个切片**：

| 脚本 | 看哪里 | 问什么 |
|------|--------|--------|
| `scan_secrets.py` | **已 tracked 的文件内容** | 现在入库的东西里有没有凭证 |
| `check_key_exposure.py` | **工作区的路径** | 有没有「未跟踪但也**未忽略**」的密钥（一次 `git add -A` 即入库） |

两者都只看**此刻**。于是留下第三个切片，实测（2026-09-13 构造样本，命令可复跑）：
一份**曾经把私钥 commit 进去、后来又删掉**的仓库，两份门禁**同时报绿**——

    $ git ls-files                       # 工作区：干净
    README.md
    $ git cat-file -p <blob>             # 对象库：私钥原样还在（PEM 头 + base64，一字未改）
    $ python3 scan_secrets.py --root $T
    ✅ 未发现凭证                          exit=0
    $ python3 check_key_exposure.py --check --root $T
    ✅ 暴露面判据全过                       exit=0

私钥**没有被删除**，只是**不再出现在任何人的视野里**。`git push` 会把它一并推到远端，
`git cat-file` 一行就能取回。这与本仓库 2026-09-13 那次事件同源（凭证一路 commit 了 28 次、
每次门禁都是绿的），也是同一句自检口令的第三次变体：

> **「我这一侧的读数，凭什么是对象那一侧的事实？」**
> 交付≠接线 · 自述≠指纹 · 能扫到≠一共有 · 没入库≠安全 · **工作区里没有 ≠ 仓库里没有**

## 两条**互相独立**的判据（不许互相顶账）

| 判据 | 覆盖 | 看不见 |
|------|------|--------|
| **A 路径史** | 任何 ref 的历史上 `--diff-filter=A` 添加过密钥形态文件名（哪怕之后删了） | 把 key 粘进 `.html` / `.json` / 笔记本 |
| **B 内容** | 对象库里**任何** blob 的正文命中凭证规则 | 加过密/打包过的密钥（base64 段落不带 `BEGIN` 头） |

A 与 B **必须分开报**：只有 A 命中 ⇒ 历史上有过密钥文件；只有 B 命中 ⇒ 密钥被写进了普通文件。
**把两者合并成一个数，就再也回答不了「是哪一种」。**

## 覆盖可达与不可达两类对象 —— 这正是本门禁存在的理由

- `REACHABLE`：从某个 ref 可达 ⇒ **会被 push 出去**，是发布面。要 `git filter-repo` 重写历史。
- `DANGLING`：只在对象库里（reflog／stash 残骸、`reset --hard` 之后的旧提交）⇒ 不会被 push，
  但 `.git` 被拷走（本仓库就备份过 `.git`）时一样可复原。`git gc --prune=now` 即可清掉。

⚠️ **`--check` 的判据主体是 REACHABLE（跨克隆确定）**；DANGLING 是**本机事实**（新克隆没有它），
报告里单独成段并注明。把两类混成一个数，就会造出一条「结果取决于谁在哪台机器上跑」的门禁 ——
`check_key_exposure.py` 已经踩过同一个坑（那条 `--sweep-roots` 因此刻意不进验收面）。

## 判据只有一处实现

规则表、转义归一化、白名单、命中预览**全部 `import` 自 `scan_secrets.py`**，本文件不另写一份。
（S12 风险 N2 的纪律：**同一个判据不许有两处实现**，否则两边会各自漂移。）

## 用法

    # 门禁（验收面 L15a）：扫本仓库对象库
    python3 paper2skills-skills/paper-维护/scripts/check_history_secrets.py --check

    # 跨仓库读数（回答「我这台机器上有哪个仓库的历史里有过凭证」，默认恒 exit 0）
    python3 paper2skills-skills/paper-维护/scripts/check_history_secrets.py \
        --sweep-roots ~/project --max-depth 4 --strict

    python3 ... --selftest     # 构造仓库自证（含反向控制）
    python3 ... --mutate       # 把门禁自己改坏，端到端验证判据有劲

退出码：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 门禁内部错误（≠ 判红）**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scan_secrets as SS  # noqa: E402  —— 规则表只应有一处实现

REPO = HERE.parents[2]  # paper_to_skills/
DEFAULT_WHITELIST = (
    REPO / "paper2skills-research" / "data" / "history-secrets-whitelist.json"
)
DEFAULT_REGISTRY = (
    REPO / "paper2skills-research" / "data" / "exposed-credential-registry.json"
)

# 密钥形态文件名。与 .gitignore 的口径一致，但**问的是历史上有没有被 add 过**。
KEY_NAME_GLOBS = [
    "*.pem", "*.key", "*.p12", "*.pfx", "*.ppk", "*.jks", "*.keystore",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*",
    ".env", ".env.*",
]
_KEY_NAME_RE = re.compile(
    r"(?:^|/)("
    r"[^/]*\.(?:pem|key|p12|pfx|ppk|jks|keystore)"
    r"|id_(?:rsa|dsa|ecdsa|ed25519)"
    r"|\.env(?:\.[A-Za-z0-9_.-]+)?"
    r")$",
    re.I,
)
# 反向控制：这些**看起来像**密钥名但不是 —— 必须**不**被计入判据 A，
# 否则门禁会靠误报活着（`.env.example` / `id_rsa.pub` 是公开内容）。
_BENIGN_NAME_RE = re.compile(
    r"\.env\.(?:example|sample|template|dist|test)$|\.pub$", re.I
)

MIN_BLOB_BYTES = 1  # 跳过「大小 ≤ 该值」的 blob；0 = 不跳过任何东西


class InputMissing(Exception):
    """输入没拿到 —— 退出码 2。**不是「干净」。**"""


class GateError(Exception):
    """门禁内部错误 —— 退出码 3。**不是「判红」。**"""


# --------------------------------------------------------------------------
# git 访问
# --------------------------------------------------------------------------

def _git(repo: Path, *args: str, text: bool = True):
    r = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=text,
        errors=None if text else None,
    )
    return r


def find_repo(path: Path) -> Path | None:
    try:
        r = _git(path, "rev-parse", "--show-toplevel")
    except FileNotFoundError as e:  # pragma: no cover
        raise GateError(f"git 不可用：{e}")
    if r.returncode != 0:
        return None
    top = (r.stdout or "").strip()
    return Path(top) if top else None


def discover_repos(roots: list[Path], max_depth: int) -> list[Path]:
    """在 roots 下按深度找 git 仓库（只认工作区根，不重复认子目录）。"""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            raise InputMissing(f"--sweep-roots 里这个根不存在：{root}")
        for dirpath, dirnames, _ in _walk(root, max_depth):
            if ".git" in dirnames or (dirpath / ".git").exists():
                dirnames[:] = []  # 不再往下钻子目录里的仓库
                if dirpath in seen:
                    continue
                seen.add(dirpath)
                found.append(dirpath)
    return sorted(found)


def _walk(root: Path, max_depth: int):
    root = root.resolve()
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(d.iterdir())
        except (PermissionError, OSError):
            continue
        dirs = [e for e in entries if e.is_dir() and not e.is_symlink()]
        yield d, [x.name for x in dirs], entries
        if depth >= max_depth:
            continue
        for child in dirs:
            if child.name in (".git", "node_modules", ".venv", "venv", "__pycache__"):
                continue
            stack.append((child, depth + 1))


def rev_list_all(repo: Path) -> int:
    r = _git(repo, "rev-list", "--all", "--count")
    if r.returncode != 0:
        raise GateError(f"git rev-list 失败：{r.stderr.strip()[:200]}")
    return int((r.stdout or "0").strip() or 0)


def reachable_object_paths(repo: Path) -> dict[str, list[str]]:
    """sha → 该对象在可达树里的路径（blob/tree 才有）。"""
    r = _git(repo, "rev-list", "--all", "--objects")
    if r.returncode != 0:
        raise GateError(f"git rev-list --objects 失败：{r.stderr.strip()[:200]}")
    out: dict[str, list[str]] = {}
    for line in (r.stdout or "").splitlines():
        sha, _, path = line.partition(" ")
        if path:
            out.setdefault(sha, []).append(path)
    return out


def head_files(repo: Path) -> set[str]:
    r = _git(repo, "ls-tree", "-r", "--name-only", "HEAD")
    if r.returncode != 0:
        return set()
    return {ln for ln in (r.stdout or "").splitlines() if ln}


# --------------------------------------------------------------------------
# 判据 A：路径史
# --------------------------------------------------------------------------

def key_shaped(path: str) -> bool:
    """这个路径算不算「密钥形态」。

    两道过滤缺一不可：先按 glob 粗筛（git 的 pathspec 会把 `id_rsa.pub` 也带进来），
    再用 `_BENIGN_NAME_RE` 剔除**公开**的那几种。
    """
    if _BENIGN_NAME_RE.search(path):
        return False
    return bool(_KEY_NAME_RE.search(path))


def path_history(repo: Path, head: set[str]) -> list[dict]:
    """判据 A：任何 ref 的历史上 `--diff-filter=A` 添加过密钥形态文件。

    ⚠️ 覆盖**全部 ref**（`--all`），不只是 HEAD —— 只看 HEAD 就正好漏掉「加过又删了」，
    而那正是本门禁存在的理由。
    """
    args = [
        "log", "--all", "--diff-filter=A", "--name-only", "--format=%x01%H", "--",
        *KEY_NAME_GLOBS,
    ]
    r = _git(repo, *args)
    if r.returncode != 0:
        raise GateError(f"git log 失败：{r.stderr.strip()[:200]}")
    hits: list[dict] = []
    commit = ""
    for line in (r.stdout or "").splitlines():
        if line.startswith("\x01"):
            commit = line[1:].strip()
            continue
        p = line.strip()
        if not p or not key_shaped(p):
            continue
        hits.append({
            "commit": commit,
            "path": p,
            "still_present": p in head,
        })
    return hits


# --------------------------------------------------------------------------
# 判据 B：对象库内容
# --------------------------------------------------------------------------

def _read_exact(f, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        b = f.read(min(1 << 20, n - got))
        if not b:
            raise GateError("对象流在读完之前断了（git cat-file 提前退出）")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def iter_objects(repo: Path, max_blob_bytes: int):
    """流式产出对象库里的每一个对象。

    用 `--batch-all-objects`（**不需要喂 stdin**，因此没有经典的双管道死锁），
    它覆盖**可达 + 不可达**的全部对象 —— 后者只有它看得见。
    """
    p = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch-all-objects", "--batch", "--buffer"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stats = {"objects": 0, "blobs": 0, "blobs_scanned": 0, "blobs_skipped_big": 0,
             "blobs_binary": 0, "bytes_scanned": 0}
    try:
        f = p.stdout
        assert f is not None
        while True:
            header = f.readline()
            if not header:
                break
            stats["objects"] += 1
            parts = header.rstrip(b"\n").split()
            if len(parts) < 3:
                # `<sha> missing` 之类：**如实计数，不许静默跳过**
                stats.setdefault("objects_unreadable", 0)
                stats["objects_unreadable"] += 1
                continue
            sha = parts[0].decode()
            typ = parts[1].decode()
            size = int(parts[2])
            if typ != "blob":
                _read_exact(f, size)
                f.read(1)
                continue
            stats["blobs"] += 1
            if size > max_blob_bytes:
                _read_exact(f, size)
                f.read(1)
                stats["blobs_skipped_big"] += 1
                yield ("skip_big", sha, size, None, stats)
                continue
            body = _read_exact(f, size)
            f.read(1)
            if SS.is_probably_binary(body):
                stats["blobs_binary"] += 1
                yield ("binary", sha, size, None, stats)
                continue
            stats["blobs_scanned"] += 1
            stats["bytes_scanned"] += size
            yield ("blob", sha, size, body.decode("utf-8", "ignore"), stats)
    finally:
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
    err = (p.stderr.read() if p.stderr else b"") or b""
    if p.returncode not in (0, None) and err:
        raise GateError(f"git cat-file 退出码 {p.returncode}：{err.decode('utf-8', 'ignore')[:200]}")


def scan_repo(
    repo: Path,
    whitelist: dict | None = None,
    max_blob_bytes: int = 8 * 1024 * 1024,
    with_paths: bool = True,
    registry: list[dict] | None = None,
) -> dict:
    """扫描一个仓库的对象库。返回结构化读数（不打印、不决定退出码）。"""
    t0 = time.time()
    registry = registry if registry is not None else []
    wl = whitelist if whitelist is not None else load_whitelist(Path("/nonexistent"))
    wl_blobs: dict[str, dict] = wl["blobs"]
    wl_paths: dict[str, dict] = wl["paths"]
    wl_registry: dict[str, dict] = wl.get("registry", {})
    live_by_sha = {e["sha256"]: (e.get("measured_status") == "live") for e in registry}

    n_commits = rev_list_all(repo)
    reach = reachable_object_paths(repo)
    head = head_files(repo) if with_paths else set()

    path_hits = path_history(repo, head) if with_paths else []
    for h in path_hits:
        ent = wl_paths.get(h["path"])
        h["whitelisted"] = ent is not None
        h["reason"] = (ent or {}).get("reason", "")
        h["ok"] = ent is not None   # 有效性已在 load_whitelist 判定过一次，此处不重复判

    content_hits: list[dict] = []
    registry_hits: list[dict] = []
    big_at_key_path: list[dict] = []
    stats: dict = {}
    for kind, sha, size, text, stats in iter_objects(repo, max_blob_bytes):
        if kind == "skip_big":
            # 大 blob 未做内容扫描 —— **必须可见**，且若它的路径本来就是密钥形态，直接判红。
            for p in reach.get(sha, []):
                if key_shaped(p):
                    big_at_key_path.append({"sha": sha, "size": size, "path": p})
            continue
        if kind != "blob":
            continue
        found = SS.scan_text(text)
        reg = registry_scan(text, registry, stats)
        if reg:
            evaluated = []
            for r in reg:
                w = wl_registry.get(r["sha256"])
                live = live_by_sha.get(r["sha256"], False)
                refused = bool(w) and live
                evaluated.append({
                    **r,
                    # ⚠️ 规则：**仍然有效（live）的凭证不许开豁免**。
                    #    豁免的语义是「已知且**无可行动作**」；活凭证**有**可行动作（后台轮换），
                    #    把它豁免掉＝把一条真实暴露静默掉。
                    "waiver": None if refused else (w or None),
                    "waiver_refused": refused,
                    "ok": bool(w) and not live,
                })
            registry_hits.append({
                "sha": sha,
                "size": size,
                "paths": reach.get(sha, []),
                "reachable": sha in reach,
                "entries": evaluated,
                "ok": all(e["ok"] for e in evaluated),
            })
        if not found:
            continue
        ent = wl_blobs.get(sha)
        content_hits.append({
            "sha": sha,
            "size": size,
            "paths": reach.get(sha, []),
            "reachable": sha in reach,
            "rules": [{"rule": h["rule"], "preview": h["match_preview"],
                       "line": h["line"]} for h in found],
            "whitelisted": ent is not None,
            "reason": (ent or {}).get("reason", ""),
            "ok": ent is not None,   # 同上：不在两处判同一条
        })

    red_paths = [h for h in path_hits if not h["ok"]]
    red_content = [h for h in content_hits if not h["ok"]]
    red_registry = [h for h in registry_hits if not h["ok"]]
    waived_registry = [h for h in registry_hits if h["ok"]]
    red_big = big_at_key_path

    return {
        "repo": str(repo),
        "commits": n_commits,
        "objects": stats.get("objects", 0),
        "objects_unreadable": stats.get("objects_unreadable", 0),
        "blobs": stats.get("blobs", 0),
        "blobs_scanned": stats.get("blobs_scanned", 0),
        "blobs_skipped_big": stats.get("blobs_skipped_big", 0),
        "blobs_binary": stats.get("blobs_binary", 0),
        "bytes_scanned": stats.get("bytes_scanned", 0),
        "max_blob_bytes": max_blob_bytes,
        "registry_entries": len(registry),
        "c_skipped_groups": stats.get("c_skipped_groups", 0),
        "c_skipped_literals": stats.get("c_skipped_literals", 0),
        "path_hits": path_hits,
        "content_hits": content_hits,
        "registry_hits": registry_hits,
        "big_at_key_path": big_at_key_path,
        "red": {"paths": red_paths, "content": red_content, "big": red_big,
                "registry": red_registry},
        "waived_registry": waived_registry,
        "n_red": len(red_paths) + len(red_content) + len(red_big) + len(red_registry),
        "reachable_red": [h for h in red_content if h["reachable"]],
        "dangling_red": [h for h in red_content if not h["reachable"]],
        "elapsed_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------
# 判据 C：**拼接固件的还原扫描**
# --------------------------------------------------------------------------

# 字符串字面量。只认单双引号、不跨行 —— 够用且不会把 Markdown 正文吃进来。
_LITERAL_RE = re.compile(r"""(?<![A-Za-z0-9_])("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')""")
# 两个字面量之间只隔着这些 ⇒ 视为「可拼接的同组」（覆盖 `_fx(a, b)` 与 `a + b`）
_ADJACENT_RE = re.compile(r"^[\s+,()\[\]]*$")
# 一组里的字面量个数上限。超过它就不再逐边界算 —— 真实固件是 2–5 段；
# 而「一个上万条字符串的大列表」在逗号相邻规则下会变成**一个巨型组**。
# ⚠️ 跳过的组数必须报出来（跳过 ≠ 没这回事）。
_MAX_GROUP = 512


def _unquote(lit: str) -> str:
    return lit[1:-1]


def rejoined_candidates(text: str, wanted_len: int, stats: dict | None = None) -> list[str]:
    """产出所有**跨越字面量边界**的长度 `wanted_len` 的候选窗口。

    为什么需要它：模式扫描器**按定义**看不见被拆开的凭证 ——
    `_fx("sk-", "aae1…")` 里不存在任何连续的凭证形态。
    而「把真值拼出来当自测固件」正是本仓库 `scan_secrets.py` 的既有写法
    （其 docstring 明说「运行时构造出的样本仍然是真值」）。
    于是产生反直觉后果：**防泄露的固件机制，本身让真值以「一行可还原」的形式留在公开仓库里。**

    ## 适用范围（写清楚，免得被当成万能）
    本判据**只覆盖跨边界的那一种**：值的一部分在字面量 A、另一部分在字面量 B。
    值若**完整落在某一段字面量内部**，它就是一段连续文本 —— 那是**判据 B**（模式扫描）
    的辖区，不需要也不应该由这里重复覆盖。

    ## 为什么不是「把所有拼接结果都造出来」
    第一版就是那么写的（对每组做 O(k²) 个候选、每个候选再拼一次中间段 ⇒ 实际 O(k³)），
    在 124 MB 语料上实测 **9s → 162s**，直接超掉验收面能接受的时间。
    现改为**按边界取窗**：一个长度为 n 的值若要跨过边界 b，它必然落在
    `S[b-n+1 : b+n-1]` 里。于是每个边界只需 ~2n 次滑窗，
    **与组的大小成线性、与值长成线性**，且**不漏**（跨 3 段以上的值照样落在某个边界窗里）。

    本判据**零误报**：判据不是启发式，而是「候选窗的 sha256 与登记表相等」。
    """
    lits = [(m.start(), m.end(), _unquote(m.group(0))) for m in _LITERAL_RE.finditer(text)]
    if not lits:
        return []
    groups: list[list[tuple[int, int, str]]] = []
    cur = [lits[0]]
    for prev, nxt in zip(lits, lits[1:]):
        if _ADJACENT_RE.match(text[prev[1]:nxt[0]]):
            cur.append(nxt)
        else:
            groups.append(cur)
            cur = [nxt]
    groups.append(cur)

    n = wanted_len
    out: list[str] = []
    for g in groups:
        if len(g) < 2:
            continue                      # 单段组没有边界 ⇒ 归判据 B 管
        if len(g) > _MAX_GROUP:
            if stats is not None:
                stats["c_skipped_groups"] = stats.get("c_skipped_groups", 0) + 1
                stats["c_skipped_literals"] = stats.get("c_skipped_literals", 0) + len(g)
            continue
        parts = [x[2] for x in g]
        s = "".join(parts)
        pos = 0
        for p in parts[:-1]:
            pos += len(p)                 # 边界位置
            lo = max(0, pos - n + 1)
            hi = min(len(s), pos + n - 1)
            if hi - lo >= n:
                out.append(s[lo:hi])
    return out


def registry_scan(text: str, registry: list[dict], stats: dict | None = None) -> list[dict]:
    """判据 C：文本里有没有**可还原**的已知暴露值。命中即红，无豁免途径。"""
    if not registry:
        return []
    hits: list[dict] = []
    for ent in registry:
        n = int(ent["len"])
        want = ent["sha256"]
        for cand in rejoined_candidates(text, n, stats):
            raw = cand.encode("utf-8", "ignore")
            for s in range(0, len(raw) - n + 1):
                if hashlib.sha256(raw[s:s + n]).hexdigest() == want:
                    hits.append({"label": ent.get("label", "?"),
                                 "len": n,
                                 "sha256": want,
                                 "sha256_12": want[:12]})
                    break
            else:
                continue
            break
    return hits


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise GateError(f"已知暴露值登记表读不出来（{path}）：{e}")
    out = []
    for ent in raw.get("entries", []):
        if not ent.get("sha256") or not ent.get("len"):
            raise GateError(f"登记表条目缺 sha256/len：{ent!r}")
        out.append(ent)
    return out


# --------------------------------------------------------------------------
# 白名单
# --------------------------------------------------------------------------

def load_whitelist(path: Path) -> dict:
    """白名单：**每条必须写 `reason`，没有理由的条目不生效且判红**。

    ⚠️ 与 `check_key_exposure.py` 的白名单同一条纪律：**「无害」必须由人写下来**，
    不许扫描器默认放行。永久豁免＝台账 #5 那种腐烂，故 `expires_when` 是必填。

    ⚠️ **「生效不生效」只在本函数里判定一次**：无效条目**不登记**（于是消费方
    `ent is None` ⇒ 仍判红），消费方不再重复判一遍 reason。
    第一版把同一条判定写在两处（loader 标空 reason + 消费方 `bool(reason)`），
    结果是**没有任何单点变异能改变行为** —— `--mutate` 的 M4 当场翻不了面。
    那不只是可测性问题：同一判据两处实现正是 S12 风险 N2（两边各自漂移）。
    """
    out = {"blobs": {}, "paths": {}, "registry": {}, "problems": []}
    if not path.exists():
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        out["problems"].append(f"白名单读不出来（{path}）：{e}")
        return out
    for ent in raw.get("entries", []):
        kind = ent.get("kind")
        key = ent.get("key")
        if kind not in ("blob", "path", "registry") or not key:
            out["problems"].append(f"条目缺 kind/key：{ent!r}")
            continue
        if not ent.get("reason"):
            out["problems"].append(f"条目缺 reason（**不登记**，命中它的文件仍判红）：{kind}:{key}")
            continue
        if not ent.get("expires_when"):
            out["problems"].append(f"条目缺 expires_when（永久豁免即腐烂）：{kind}:{key}")
        if kind == "registry" and not ent.get("evidence"):
            out["problems"].append(f"登记表条目豁免缺 evidence（实测证据）：registry:{key}")
            continue
        out[{"blob": "blobs", "path": "paths", "registry": "registry"}[kind]][key] = ent
    return out


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

def print_report(res: dict, label: str = "") -> None:
    print(f"仓库 {res['repo']}{('  ' + label) if label else ''}")
    print(
        f"  提交 {res['commits']} · 对象 {res['objects']}"
        f"（不可读 {res['objects_unreadable']}）· blob {res['blobs']}"
        f" · 内容已扫 {res['blobs_scanned']}（{res['bytes_scanned'] / 1e6:.1f} MB）"
        f" · 二进制跳过 {res['blobs_binary']} · 超大跳过 {res['blobs_skipped_big']}"
        f" · {res['elapsed_s']}s"
    )
    if res["blobs_skipped_big"]:
        print(
            f"  ⚠️ 有 {res['blobs_skipped_big']} 份 blob 超过 {res['max_blob_bytes']} B "
            f"**未做内容扫描** —— 把它们打包进去的凭证，本门禁看不见（这是适用范围，不是绿灯）"
        )

    print(f"  判据 A（路径史）：命中 {len(res['path_hits'])} 条，其中判红 {len(res['red']['paths'])} 条")
    for h in res["red"]["paths"]:
        print(f"    ❌ [A] {h['path']}  ← 由 {h['commit'][:10]} 添加"
              f"{'（HEAD 仍在）' if h['still_present'] else '（**HEAD 已无此文件**）'}")
    for h in res["path_hits"]:
        if h["ok"]:
            print(f"    ✅ [A] {h['path']}（白名单：{h['reason'][:40]}）")

    print(f"  判据 B（对象内容）：命中 {len(res['content_hits'])} 条，其中判红 {len(res['red']['content'])} 条")
    for h in res["red"]["content"]:
        loc = ("REACHABLE" if h["reachable"] else "DANGLING")
        paths = " / ".join(h["paths"][:2]) or "（不可达，无路径）"
        rules = ",".join(sorted({r["rule"] for r in h["rules"]}))
        print(f"    ❌ [B/{loc}] {h['sha'][:12]} {paths}  ← {rules}")
        for r in h["rules"][:6]:
            # ⚠️ 行号必须打出来：没有行号的命中**没法裁定**（只能整文件拉黑），
            #    而「整文件拉黑」正是把门禁变成恒红的第一步。
            #    行号是相对 **blob 内容**的行号，不是工作区文件的行号 ——
            #    两者在「文件已被删掉 / 已被改过」时**必然不同**，别照抄。
            print(f"         {r['rule']} @L{r['line']}: {r['preview']}")
    for h in res["content_hits"]:
        if h["ok"]:
            print(f"    ✅ [B] {h['sha'][:12]}（白名单：{h['reason'][:40]}）")

    print(f"  判据 C（已知暴露值还原）：命中 {len(res['registry_hits'])} 条"
          f"（登记表 {res['registry_entries']} 条 · **无豁免途径**）")
    if res.get("c_skipped_groups"):
        print(f"  ⚠️ 判据 C 跳过了 {res['c_skipped_groups']} 个字面量组"
              f"（共 {res['c_skipped_literals']} 段，超过单组上限）—— 这些组里的跨段拼接没被检查")
    for h in res["red"]["registry"]:
        loc = ("REACHABLE" if h["reachable"] else "DANGLING")
        paths = " / ".join(h["paths"][:2]) or "（不可达，无路径）"
        print(f"    ❌ [C/{loc}] {h['sha'][:12]} {paths}")
        for e in h["entries"]:
            print(f"         可还原：{e['label']}（len={e['len']} · sha256 {e['sha256_12']}…）")
            if e["waiver_refused"]:
                print("           ⚠️ 白名单里有豁免条目但**被拒**：该值实测仍然有效 ⇒ 活凭证不许豁免")
            elif e["ok"]:
                print("           （此条已豁免，同块另一条仍判红）")

    for h in res["big_at_key_path"]:
        print(f"    ❌ [超大 blob × 密钥名路径] {h['sha'][:12]} {h['path']}（{h['size']} B）")

    # ⚠️ 已豁免的条目**每次运行都打出来**：豁免是「可见」不是「忘记」。
    for h in res.get("waived_registry", []):
        for e in h["entries"]:
            if e["ok"] and e["waiver"]:
                print(f"    ⚠️ [C/已豁免] {e['label']}")
                print(f"         理由：{e['waiver']['reason']}")
                print(f"         到期条件：{e['waiver']['expires_when']}")

    if res["dangling_red"]:
        print(
            f"  ⚠️ 其中 DANGLING {len(res['dangling_red'])} 条：只在**本机**对象库里"
            f"（新克隆没有它），不随 push 出去 —— 但 `.git` 被拷走即复原；`git gc --prune=now` 可清"
        )


def verdict(res: dict) -> int:
    """退出码。**覆盖率为零 ⇒ 2（没测）**，绝不等于通过。"""
    if res["commits"] == 0:
        raise InputMissing(f"{res['repo']} 一个提交都没有 —— 这不是「干净」，是「没测」")
    if res["blobs_scanned"] == 0:
        # ⚠️ 这里**不豁免**「blob 全是二进制」与「blob 全被大小阈值跳过」两种情形：
        #    判据 B 覆盖的是 0 个 blob，那就是「没测」。第一版把条件写成
        #    `... and res["blobs"] == 0`，于是「有 blob 但一个都没扫」被判成了干净
        #    —— 用例 6 当场抓住。**「我扫过」必须由「真的扫到内容的 blob 数」证明。**
        raise InputMissing(
            f"{res['repo']} 走过 {res['objects']} 个对象、blob {res['blobs']} 个，"
            f"但**内容扫描覆盖 0 个**（二进制 {res['blobs_binary']} / 超大跳过 "
            f"{res['blobs_skipped_big']}）—— 这不是「干净」，是「没测」"
        )
    return 1 if res["n_red"] else 0


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

ENV = {
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
}

# 固件一律**拼**出来 —— 源码里不存在完整凭证。
# （继承 scan_secrets.py 的理由：本文件自己被扫出命中 ⇒ 门禁恒红 ⇒ 恒红的门禁等于没有门禁。）
FAKE_PEM = SS._fx("-----BEGIN RSA ", "PRIVATE KEY-----") + "\n" + "ZmFrZWZha2U=\n" + \
    SS._fx("-----END RSA ", "PRIVATE KEY-----") + "\n"
FAKE_SK = SS._fx("api_key=\"", "sk-", "f" * 32, "\"")


def _init_repo(root: Path) -> None:
    env = {**os.environ, **ENV}
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, capture_output=True, env=env)
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, capture_output=True, env=env)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=root, capture_output=True, env=env)


def _g(root: Path, *args: str):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, env={**os.environ, **ENV})


def _mk(tmp: Path, name: str) -> Path:
    p = tmp / name
    _init_repo(p)
    return p


def selftest() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="histsec-"))

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"{'✅' if good else '❌'} {label}" + (f"：{detail}" if detail else ""))

    # ---- 用例 1：核心剧本 —— 曾入库、后删除 -------------------------------
    r1 = _mk(tmp, "deleted-key")
    (r1 / "cfg").mkdir()
    (r1 / "cfg" / "deploy.pem").write_text(FAKE_PEM, encoding="utf-8")
    _g(r1, "add", "-f", "cfg/deploy.pem")
    _g(r1, "commit", "-qm", "add key")
    _g(r1, "rm", "-q", "cfg/deploy.pem")
    _g(r1, "commit", "-qm", "remove key (working tree now clean)")
    res1 = scan_repo(r1)
    check("用例 1：曾入库后删除的私钥 ⇒ 判据 A 打红",
          len(res1["red"]["paths"]) == 1 and res1["red"]["paths"][0]["path"] == "cfg/deploy.pem")
    check("用例 1：同一份内容 ⇒ 判据 B 也打红（两条判据独立命中，不互相顶账）",
          len(res1["red"]["content"]) >= 1
          and any(r["rule"] == "PRIVATE_KEY_BLOCK"
                  for h in res1["red"]["content"] for r in h["rules"]))
    check("用例 1：覆盖率达标 ⇒ 退出码语义为 1（判红，不是 2）",
          verdict(res1) == 1)
    check("用例 1：这一条**同时**是那两个既有门禁的盲区（本门禁存在的全部理由）",
          not SS.scan_text("".join(
              (r1 / p).read_text(encoding="utf-8", errors="ignore")
              for p in _g(r1, "ls-files").stdout.split()
              if (r1 / p).is_file())))

    # ---- 用例 2：反向控制 —— 干净仓库必须绿 --------------------------------
    r2 = _mk(tmp, "clean")
    (r2 / "notes").mkdir()
    (r2 / "notes" / "readme.md").write_text("纯文档，没有任何凭证\n", encoding="utf-8")
    _g(r2, "add", "-A")
    _g(r2, "commit", "-qm", "docs")
    res2 = scan_repo(r2)
    check("用例 2（反向控制）：干净仓库 ⇒ 绿（否则门禁靠误报活着）",
          res2["n_red"] == 0 and verdict(res2) == 0)
    check("用例 2：干净仓库也必须真的扫过东西（覆盖率一等输出）",
          res2["blobs_scanned"] >= 2, f"已扫 {res2['blobs_scanned']}")

    # ---- 用例 3：只有判据 A 命中（内容无 BEGIN 头 ⇒ B 看不见）--------------
    r3 = _mk(tmp, "path-only")
    (r3 / "enc.key").write_bytes(b"\x00\x01binary-ish-content-no-begin-header\x02")
    _g(r3, "add", "-f", "enc.key")
    _g(r3, "commit", "-qm", "add opaque key")
    _g(r3, "rm", "-q", "enc.key")
    _g(r3, "commit", "-qm", "drop")
    res3 = scan_repo(r3)
    check("用例 3：内容判据看不见的密钥文件（无 BEGIN 头）⇒ 判据 A 单独抓住",
          len(res3["red"]["paths"]) == 1 and res3["n_red"] >= 1,
          f"A={len(res3['red']['paths'])} B={len(res3['red']['content'])}")

    # ---- 用例 4：只有判据 B 命中（key 粘进普通 .html）----------------------
    r4 = _mk(tmp, "content-only")
    (r4 / "playbook.html").write_text(f"<script>const cfg = {{{FAKE_SK}}};</script>\n",
                                      encoding="utf-8")
    _g(r4, "add", "-A")
    _g(r4, "commit", "-qm", "add html with key")
    _g(r4, "rm", "-q", "playbook.html")
    _g(r4, "commit", "-qm", "drop html")
    res4 = scan_repo(r4)
    check("用例 4：粘进 .html 的 key（路径无密钥名）⇒ 判据 B 单独抓住",
          len(res4["red"]["content"]) == 1 and not res4["red"]["paths"],
          f"A={len(res4['red']['paths'])} B={len(res4['red']['content'])}")
    check("用例 4：命中的是 sk- 类规则而不是泛化规则（优先级去重仍然生效）",
          any(r["rule"] == "OPENAI_STYLE_KEY"
              for h in res4["red"]["content"] for r in h["rules"]))
    check("用例 4：预览被打码，报告本身不成为第二份泄露源",
          all("f" * 32 not in r["preview"]
              for h in res4["red"]["content"] for r in h["rules"]))

    # ---- 用例 5：白名单（有理由才生效；无理由不生效）-----------------------
    sha4 = res4["red"]["content"][0]["sha"]
    wl_good = {"entries": [{"kind": "blob", "key": sha4, "reason": "构造样本：值本身写着 fake",
                            "evidence": "selftest 用例 5", "expires_when": "本用例结束时"}]}
    res5 = scan_repo(r4, whitelist=_wl(wl_good))
    check("用例 5：白名单**写明理由**后放行（反向控制：不是一律放行）",
          res5["n_red"] == 0, f"n_red={res5['n_red']}")
    wl_bad = {"entries": [{"kind": "blob", "key": sha4, "reason": "",
                           "expires_when": "never"}]}
    wl_bad = load_whitelist(_write_wl(tmp, wl_bad))
    res5b = scan_repo(r4, whitelist=wl_bad)
    check("用例 5（反后门）：白名单条目**缺 reason** ⇒ 不生效，仍然判红",
          res5b["n_red"] == 1 and wl_bad["problems"])

    # ---- 用例 6：覆盖率不够 ⇒ exit 2，不是 0 -------------------------------
    empty = tmp / "empty"
    empty.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=empty, capture_output=True,
                   env={**os.environ, **ENV})
    res6 = scan_repo(tf := r2, max_blob_bytes=MIN_BLOB_BYTES)  # 全跳过
    try:
        verdict(res6)
        check("用例 6：全部 blob 被跳过 ⇒ 必须报『没测』（不是「干净」）", False,
              f"**没抓住**：n_red={res6['n_red']} scanned={res6['blobs_scanned']}")
    except InputMissing:
        check("用例 6：全部 blob 被跳过 ⇒ 必须报『没测』（不是「干净」）", True)
    res6b = scan_repo(empty)
    try:
        verdict(res6b)
        check("用例 6b：空仓库（0 提交）⇒ 必须报『没测』", False, "**没抓住**")
    except InputMissing:
        check("用例 6b：空仓库（0 提交）⇒ 必须报『没测』", True)

    # ---- 用例 7：DANGLING（不可达）也必须被抓住，且分类正确 ----------------
    r7 = _mk(tmp, "dangling")
    (r7 / "id_rsa").write_text(FAKE_PEM, encoding="utf-8")
    _g(r7, "add", "-f", "id_rsa")
    _g(r7, "commit", "-qm", "oops")
    _g(r7, "reset", "-q", "--hard", "HEAD~1")
    _g(r7, "reflog", "expire", "--expire=now", "--all")
    res7 = scan_repo(r7)
    check("用例 7：reset --hard 之后的旧提交 ⇒ 仍判红（对象库还留着它）",
          res7["n_red"] >= 1, f"n_red={res7['n_red']}")
    check("用例 7：分类为 DANGLING（不随 push 出去，处置是 gc 不是 filter-repo）",
          bool(res7["dangling_red"]) or bool(res7["red"]["paths"]))

    # ---- 用例 8：元级 —— 本文件自己必须干净 --------------------------------
    own = Path(__file__).read_text(encoding="utf-8")
    check("用例 8（元级）：本门禁自身源码不含完整凭证", not SS.scan_text(own))

    # ---- 用例 9：判据 A 不得靠误报活着（反向控制）--------------------------
    r9 = _mk(tmp, "benign-names")
    (r9 / ".env.example").write_text("API_KEY=your-key-here\n", encoding="utf-8")
    (r9 / "id_rsa.pub").write_text("ssh-rsa AAAAB3NzaC1yc2EAAA fake\n", encoding="utf-8")
    _g(r9, "add", "-A")
    _g(r9, "commit", "-qm", "example config")
    res9 = scan_repo(r9)
    check("用例 9（反向控制）：`.env.example` 与 `id_rsa.pub` ⇒ 判据 A 不得命中",
          not res9["path_hits"], f"命中了 {[h['path'] for h in res9['path_hits']]}")
    # ---- 用例 10：判据 C —— 跨字面量拼接的可还原值 -------------------------
    # 用**同形假值**建固件（真值放进来就等于把要防的东西写进测试里），哈希现算。
    split_val = "sk-" + "d" * 28 + "9" * 4               # 35 字符，与登记表同形
    reg = [{"label": "构造样本：拼接固件里的假值", "len": len(split_val),
            "sha256": hashlib.sha256(split_val.encode()).hexdigest(),
            "measured_status": "dead"}]
    r10 = _mk(tmp, "rejoin")
    # 三段拼接 —— 单看任何一段都不是凭证形态
    (r10 / "fixture.py").write_text(
        'K = _fx("sk-", "' + "d" * 28 + '", "' + "9" * 4 + '")\n', encoding="utf-8")
    _g(r10, "add", "-A"); _g(r10, "commit", "-qm", "split fixture")
    res10 = scan_repo(r10, registry=reg)
    check("用例 10：判据 C 抓到**跨三段拼接**的可还原值",
          len(res10["red"]["registry"]) == 1, f"实得 {len(res10['red']['registry'])}")
    check("用例 10：同一份内容**判据 B 抓不到**（这正是 C 存在的理由）",
          not res10["red"]["content"],
          f"B 判红 {len(res10['red']['content'])} 条")
    check("用例 10：命中即红、退出码语义为 1", verdict(res10) == 1)

    # 反向控制：值不同 ⇒ 必须不报（否则判据靠误报活着）
    other = "sk-" + "d" * 28 + "8" * 4
    reg_other = [{**reg[0], "sha256": hashlib.sha256(other.encode()).hexdigest()}]
    res10b = scan_repo(r10, registry=reg_other)
    check("用例 10b（反向控制）：同形而值不同 ⇒ 判据 C 不得命中",
          res10b["n_red"] == 0 and verdict(res10b) == 0)

    # ---- 用例 11：豁免规则（dead 可豁免 / **live 不许豁免**）----------------
    dead_wl = _wl({"entries": [{"kind": "registry", "key": reg[0]["sha256"],
                                "reason": "构造样本：已实测失效",
                                "evidence": "selftest 用例 11", "expires_when": "用例结束"}]})
    res11 = scan_repo(r10, whitelist=dead_wl, registry=reg)
    check("用例 11：dead 条目写明理由+证据 ⇒ 豁免生效（绿）",
          res11["n_red"] == 0 and len(res11["waived_registry"]) == 1,
          f"n_red={res11['n_red']}")

    reg_live = [{**reg[0], "measured_status": "live"}]
    res11b = scan_repo(r10, whitelist=dead_wl, registry=reg_live)
    check("用例 11b（关键反向控制）：**live 条目即使写了豁免也照样判红**",
          res11b["n_red"] == 1 and res11b["red"]["registry"][0]["entries"][0]["waiver_refused"],
          f"n_red={res11b['n_red']}")
    check("用例 11b：豁免被拒这件事**被报出来**，不是静默忽略",
          any(e["waiver_refused"] for h in res11b["red"]["registry"] for e in h["entries"]))

    noev = _write_wl(tmp, {"entries": [{"kind": "registry", "key": reg[0]["sha256"],
                                        "reason": "缺 evidence", "expires_when": "x"}]})
    wl_noev = load_whitelist(noev)
    check("用例 11c：登记表类豁免缺 `evidence` ⇒ 不登记且有 problems（判红）",
          reg[0]["sha256"] not in wl_noev["registry"] and bool(wl_noev["problems"]))

    # ---- 用例 12：跳过的字面量组必须可见（不是静默）-----------------------
    r12 = _mk(tmp, "biggroup")
    body = "L = [" + ", ".join(f'"x{i:05d}"' for i in range(600)) + "]\n"
    (r12 / "big.py").write_text(body, encoding="utf-8")
    _g(r12, "add", "-A"); _g(r12, "commit", "-qm", "big literal list")
    res12 = scan_repo(r12, registry=reg)
    check("用例 12：超上限的字面量组 ⇒ 跳过并被计数（跳过≠没这回事）",
          res12["c_skipped_groups"] == 1 and res12["c_skipped_literals"] == 600,
          f"groups={res12['c_skipped_groups']} lits={res12['c_skipped_literals']}")

    # ---- 变异：把判据自己改坏，看它报不报 ----------------------------------
    # ⚠️ 每个变异必须打在**只有该判据会命中的那块语料**上。第一版把「判据 A 恒不命中」
    #    打在用例 1（A 与 B 同时命中）上，于是关掉 A 之后 B 仍然报红、探针误报「漏网」
    #    —— 与 check_key_exposure 那次同型：**变异必须打得动它要打的那一条。**
    #    这条纪律顺带把「两条判据互不顶账」也变成了可执行的断言。
    print("--- 变异：把门禁自己改坏（要求先证明变异施上了力）---")
    for label, target, res_obj, only in [
        ("判据 B 恒不命中", "content", res4, "B"),
        ("判据 A 恒不命中", "paths", res3, "A"),
    ]:
        got = _sabotage_probe(target, res_obj)
        check(f"变异 [{label}] 必须被抓住（该夹具上只有判据 {only} 会命中）", got is True)

    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _wl(spec: dict) -> dict:
    out = {"blobs": {}, "paths": {}, "registry": {}, "problems": []}
    key = {"blob": "blobs", "path": "paths", "registry": "registry"}
    for ent in spec.get("entries", []):
        if ent.get("reason"):
            out[key[ent["kind"]]][ent["key"]] = ent
    return out


def _write_wl(tmp: Path, spec: dict) -> Path:
    p = tmp / "wl.json"
    p.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return p


def _sabotage_probe(target: str, res_obj: dict) -> bool:
    """两道断言缺一不可（route_backlog / check_card_identity 同一条纪律）：
    ① 干净扫描里**确实存在**判红的对象 —— 否则变异没施上力；
    ② 打了变异之后它**确实由红变绿** —— 否则变异没改变判定。
    """
    if not res_obj["n_red"]:
        print(f"   ⚠️ 变异 [{target}] 没施上力：干净扫描里没有可打的红")
        return False
    real_path = path_history
    real_scan = SS.scan_text
    if target == "paths":
        globals()["path_history"] = lambda repo, head: []
    else:
        SS.scan_text = lambda text: []
    try:
        after = scan_repo(Path(res_obj["repo"]))
    finally:
        globals()["path_history"] = real_path
        SS.scan_text = real_scan
    return after["n_red"] == 0


# --------------------------------------------------------------------------
# 变异测试（源级、端到端）
# --------------------------------------------------------------------------

MUTATIONS = [
    # ⚠️ 锚点必须**拼**出来：若在变异表里写字面量，它自己就成了源码里的第二处出现
    #    —— 正是 S11「变异锚点被变异表自己抄一遍」那个坑，本文件的 `--mutate` 首跑就踩了。
    ("M1 判据 B 恒不命中（scan_text 变空）",
     "        found = SS." + "scan_text(text)",
     "        found = []",
     "content"),
    ("M2 判据 A 恒不命中（path_history 提前返回）",
     "    hits: list[dict] = []\n    commit = \"\"",
     "    return []\n    hits: list[dict] = []\n    commit = \"\"",
     "paths"),
    ("M3 覆盖率判据失效（0 blob 也算测过）",
     "    if res[\"blobs_scanned\"] == 0:",
     "    if False:",
     "coverage"),
    ("M5 活凭证也允许豁免（live 检查失效）",
     "                \"ok\": bool(w) and not live,",
     "                \"ok\": bool(w),",
     "live_waiver"),
    ("M4 白名单不看理由（缺 reason 也放行）",
     "        if not ent.get(\"reason\"):",
     "        if False:",
     "whitelist"),
]


def mutate() -> int:
    """把门禁自己改坏，跑**真 CLI + 端到端夹具**，要求端到端判据随之翻面。

    ⚠️ 台账 #25 的教训：判据在 `main()` 里而 selftest 只测库函数 ⇒ 把守卫改成 `if (false)`
    照样全绿。故这里**不开进程内后门**，而是把源码拷出去改、用子进程跑真 CLI。
    """
    ok = True
    src = Path(__file__).read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="histsec-mut-"))

    def check(label: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and good
        print(f"{'✅' if good else '❌'} {label}" + (f"：{detail}" if detail else ""))

    # 夹具：三类必须判红的仓库 + 一个「白名单缺 reason」的仓库
    fc = tmp / "fx-content"
    _init_repo(fc)
    (fc / "a.html").write_text(f"<b>{FAKE_SK}</b>", encoding="utf-8")
    _g(fc, "add", "-A"); _g(fc, "commit", "-qm", "k")
    _g(fc, "rm", "-q", "a.html"); _g(fc, "commit", "-qm", "d")

    fp = tmp / "fx-paths"
    _init_repo(fp)
    (fp / "s.pem").write_text("opaque\n", encoding="utf-8")
    _g(fp, "add", "-f", "s.pem"); _g(fp, "commit", "-qm", "k")
    _g(fp, "rm", "-q", "s.pem"); _g(fp, "commit", "-qm", "d")

    fe = tmp / "fx-empty"
    fe.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=fe, capture_output=True,
                   env={**os.environ, **ENV})

    fn = tmp / "fx-clean"          # 干净仓库：只给覆盖率变异用（变异前必须**没有**红）
    _init_repo(fn)
    (fn / "notes").mkdir()
    (fn / "notes" / "a.md").write_text("纯文档\n", encoding="utf-8")
    _g(fn, "add", "-A"); _g(fn, "commit", "-qm", "docs")

    fw = tmp / "fx-wl"
    _init_repo(fw)
    (fw / "b.html").write_text(f"<b>{FAKE_SK}</b>", encoding="utf-8")
    _g(fw, "add", "-A"); _g(fw, "commit", "-qm", "k")
    sha = _g(fw, "rev-parse", "HEAD:b.html").stdout.strip()
    blobsha = _g(fw, "hash-object", "b.html").stdout.strip()
    assert sha == blobsha, f"{sha} != {blobsha}"
    wl = tmp / "wl.json"
    wl.write_text(json.dumps({"entries": [
        {"kind": "blob", "key": blobsha, "reason": "", "expires_when": "never"}]}),
        encoding="utf-8")

    # ⚠️ 每个夹具的「变异前」退出码必须**正是该变异要打掉的那一条**判据给出的：
    #    content/paths/whitelist 用判红(1)，coverage 用「没测」(2) 且夹具本身**无红**
    #    —— 否则变异后即使判据失效也仍是 1，翻不了面。
    # 活凭证夹具：值与登记表同形，但 measured_status=live，且白名单里写了豁免
    fl = tmp / "fx-live"
    _init_repo(fl)
    live_val = "sk-" + "e" * 28 + "7" * 4
    (fl / "fx.py").write_text(
        'K = _fx("sk-", "' + "e" * 28 + '", "' + "7" * 4 + '")\n', encoding="utf-8")
    _g(fl, "add", "-A"); _g(fl, "commit", "-qm", "split")
    reg = tmp / "reg.json"
    reg.write_text(json.dumps({"entries": [
        {"label": "构造样本：live", "len": len(live_val),
         "sha256": hashlib.sha256(live_val.encode()).hexdigest(),
         "measured_status": "live"}]}), encoding="utf-8")
    wl2 = tmp / "wl-live.json"
    wl2.write_text(json.dumps({"entries": [
        {"kind": "registry", "key": hashlib.sha256(live_val.encode()).hexdigest(),
         "reason": "变异体应当把它放行", "evidence": "mutation fixture",
         "expires_when": "never"}]}), encoding="utf-8")

    cases = {
        "live_waiver": (["--registry", str(reg), "--whitelist", str(wl2), "--root", str(fl)], 1),
        "content": (["--registry", "/nonexistent", "--root", str(fc)], 1),
        "paths": (["--registry", "/nonexistent", "--root", str(fp)], 1),
        "coverage": (["--registry", "/nonexistent", "--max-blob-bytes", "1",
                      "--root", str(fn)], 2),
        "whitelist": (["--registry", "/nonexistent", "--whitelist", str(wl),
                       "--root", str(fw)], 1),
    }

    def run(script: Path, argv: list[str]) -> tuple[int, str]:
        r = subprocess.run([sys.executable, str(script), "--check", *argv],
                           capture_output=True, text=True,
                           env={**os.environ, **ENV})
        return r.returncode, (r.stdout + r.stderr).strip().splitlines()[-1:] and \
            (r.stdout + r.stderr).strip().splitlines()[-1] or ""

    for label, old, new, key in MUTATIONS:
        argv, want = cases[key]
        base, tail = run(Path(__file__), argv)
        check(f"{label}：变异**前**夹具实测退出码 {want}", base == want,
              f"实得 {base} · {tail[:80]}")
        n_anchor = src.count(old)
        if n_anchor != 1:
            check(f"{label}：变异锚点在源码里**恰好出现一次**（S11 的『锚点被变异表自己抄一遍』）",
                  False, f"出现 {n_anchor} 次：{old[:50]!r}")
            continue
        mut_dir = tmp / f"mut-{key}"
        mut_dir.mkdir(exist_ok=True)
        shutil.copy2(HERE / "scan_secrets.py", mut_dir / "scan_secrets.py")
        target = mut_dir / "check_history_secrets.py"
        target.write_text(src.replace(old, new, 1), encoding="utf-8")
        got, tail2 = run(target, argv)
        check(f"{label}：变异**后**端到端判据必须翻面（{want} → 0）", got == 0,
              f"实得 {got} · {tail2[:80]}")

    shutil.rmtree(tmp, ignore_errors=True)
    print("MUTATE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="凭证扫描第三缝：git 对象库")
    ap.add_argument("--check", action="store_true", help="扫本仓库对象库（确定性；进门禁用）")
    ap.add_argument("--root", default=None, help="被扫的仓库路径（默认本仓库）")
    ap.add_argument("--sweep-roots", nargs="*", default=None,
                    help="跨仓库读数（随环境漂移，默认恒 exit 0；--strict 才判红）")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-blob-bytes", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--whitelist", default=str(DEFAULT_WHITELIST))
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                    help="已知暴露值登记表（只存 len+sha256；判据 C 用，**无豁免途径**）")
    ap.add_argument("--strict", action="store_true", help="广扫时也按判红给退出码")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mutate", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.mutate:
        return mutate()
    if not args.check and args.sweep_roots is None:
        ap.print_help()
        return 3

    try:
        wl_path = Path(args.whitelist)
        wl = load_whitelist(wl_path)
        registry = load_registry(Path(args.registry))
        if wl["problems"]:
            print("⚠️ 白名单有问题（下列条目不生效）：")
            for p in wl["problems"]:
                print(f"   - {p}")
        results: list[dict] = []
        worst = 0
        if args.sweep_roots is not None:
            roots = [Path(r).expanduser() for r in args.sweep_roots]
            repos = discover_repos(roots, args.max_depth)
            if not repos:
                raise InputMissing(f"在 {[str(r) for r in roots]} 下没找到任何 git 仓库")
            print(f"广扫：{len(repos)} 个仓库")
            for r in repos:
                res = scan_repo(r, whitelist=wl, max_blob_bytes=args.max_blob_bytes,
                                registry=registry)
                results.append(res)
                if res["n_red"]:
                    print_report(res)
                else:
                    print(f"  ✅ {res['repo']}（blob {res['blobs']} · 已扫 "
                          f"{res['blobs_scanned']} · {res['elapsed_s']}s）")
            tot = sum(r["n_red"] for r in results)
            print(f"广扫合计：判红 {tot} 个仓库/条目"
                  f"（这是**读数**不是判决 —— 随环境漂移，默认不进门禁）")
            if wl["problems"]:
                tot += 1
            worst = 1 if (args.strict and tot) else 0
        else:
            root = Path(args.root).expanduser() if args.root else REPO
            repo = find_repo(root)
            if repo is None:
                raise InputMissing(f"{root} 不在任何 git 仓库里 —— 这不是「干净」，是「没测」")
            res = scan_repo(repo, whitelist=wl, max_blob_bytes=args.max_blob_bytes,
                            registry=registry)
            results.append(res)
            print_report(res)
            worst = verdict(res)
            if wl["problems"]:
                print("❌ 白名单里有不生效的条目（缺 reason / 缺 expires_when）⇒ 判红")
                worst = 1
            if res["n_red"] == 0 and not wl["problems"]:
                _w = sum(len([e for e in h["entries"] if e["ok"]]) for h in res["registry_hits"])
                print("✅ 对象库里没有凭证形态内容，历史上也没有添加过密钥形态文件，"
                      "已知暴露值也不可还原")
                if _w:
                    print(f"   ⚠️ 但仍有 {_w} 条已知暴露值处于**豁免**中（上方逐条列出理由与到期条件）")
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps({"results": results, "whitelist_problems": wl["problems"],
                            "registry_entries": len(registry)},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        return worst
    except InputMissing as e:
        print(f"❓ 输入没拿到：{e}")
        return 2
    except GateError as e:
        print(f"💥 门禁内部错误：{e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
