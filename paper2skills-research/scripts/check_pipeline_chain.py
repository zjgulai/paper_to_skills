#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 2 环（收割 → 过滤 → 打分 → 派生）**接线后的链条门禁**（PHASE6 #105 接线批交付）

为什么要有它 —— 两条都是本批实测撞出来的，且**没有任何既有门禁看得见**：

  ⚠️ **#113：`build_registry.py` 是死脚本，而它死了两天没人知道。**
  P1（`4dfa9f2`）给 `DECISIONS` 加的域名守卫写在了 `DECISIONS` **定义之前** ⇒
  模块级前向引用 ⇒ `NameError` at import ⇒ **加守卫的那天起，这个脚本就无法运行**。
  42 个脚本里只有它一个 import 就炸，而验收面 65 条门禁**没有一条跑过它**。
  ⇒ 「交付」不是「接线」（#11/#23/#71/#78/#105 同族），而**「接线」也不是「跑得起来」**。
  J1 就是这面镜子：把**规格块点名的**每个脚本真的 import 一次。

  ⚠️ **#114：那面镜子照出的第二层更危险 —— 它跑起来会毁数据。**
  盘上 `papers_registry.json` 是**唯一事实源**，PHASE3–PHASE5 由人逐条补过；
  而 `build_registry.py` 只是**一次性初始化器**，重跑会改写 **39/45** 条记录
  （venue_tier 校正 31 / 已交付卡回指 19 / note 口径修正 9 / **DOI 更正 8** / R4 降级 7），
  只有 6 条能被原样复现。⇒ **「跑不起来」掩盖了「跑起来会毁数据」**。
  J3 因此不看源码措辞，看**行为**：默认跑一遍，盘上那份必须**逐字节不变**。

  ⚠️ **#105 本身：接线的三条纪律必须能被打红。** J2 逐条喂反向控制 ——
  删掉过滤产物、把**未过滤池改名冒充**过滤产物、`unjudged_ids` 非空，
  三种都必须 exit 2；而 `exit 0` 或「静默回退到未过滤池」一律判红。

退出码：**0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 门禁内部错误（≠ 判红）**。
`--selftest` 用构造样本证明**每条判据都能失败**（含「干净夹具必须 exit 0」的反向控制）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "paper2skills-research" / "data"
VAULT = REPO / "paper2skills-vault"
SPEC_DOC = VAULT / "07-资源库" / "关键词库-v2.md"

#: 第 2 环 + 它的派生消费者。「点名的脚本」以规格块为准，这里只补**规格块之外**
#: 但确实读打分产物的那几个 —— 补表要连带补用例（别凭印象扩表）。
EXTRA_CHAIN = ["build_registry.py", "make_bundles.py", "dedup_check.py"]

#: 本门禁**不得改动**的真产物（自证用）。
FROZEN = [
    DATA / "arxiv_candidates.json",
    DATA / "arxiv_candidates_filtered.json",
    DATA / "recommendations.json",
    DATA / "recommendations.csv",
    DATA / "shortlist.md",
    VAULT / "07-资源库" / "papers_registry.json",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def spec_scripts(spec_path: Path = SPEC_DOC) -> list[str] | None:
    """规格块点名的 `.py`（唯一一处定义：规格块本身）。读不到 ⇒ None。"""
    if not spec_path.is_file():
        return None
    m = re.search(r"<!--\s*P2S-SPEC:pipeline(.*?)-->", spec_path.read_text(encoding="utf-8"), re.S)
    if not m:
        return None
    return sorted(set(re.findall(r"([A-Za-z0-9_./-]+\.py)", m.group(1))))


def _py(code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=str(cwd),
                          capture_output=True, text=True, timeout=300)


# ---------------------------------------------------------------------------
# J1 —— 点名的脚本必须**加载得起来**（不是「文件在」，是「import 不炸」）
# ---------------------------------------------------------------------------
def j1_importable(scripts: list[str], repo: Path) -> tuple[bool, list[str]]:
    """⚠️ 判据问的是「import 得起来吗」而不是「文件存在吗」 —— 存在但 import 就炸，
    正是 #113 的形态，而 `os.path.isfile` 对它**完全无感**。
    ⚠️ 只对**有 `if __name__ == "__main__"` 守卫**的脚本 import：没有守卫的脚本
    一 import 就会执行主逻辑（那个动作本身就是副作用，不该由门禁代做）。
    没有守卫的脚本在这里**报出来但不判红** —— 「未测」必须是一等读数，不能算通过。
    """
    errs, unshielded = [], []
    sdir = repo / "paper2skills-research" / "scripts"
    for rel in scripts:
        p = sdir / Path(rel).name
        if not p.is_file():
            errs.append(f"J1 规格块点名了 `{rel}`，但 {p} 不存在")
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            errs.append(f"J1 `{p.name}` 语法就不通过：{e}")
            continue
        shielded = any(isinstance(n, ast.If)
                       and getattr(getattr(n.test, "left", None), "id", None) == "__name__"
                       for n in tree.body)
        if not shielded:
            unshielded.append(p.name)
            continue
        r = _py(f"import {p.stem}", sdir)
        if r.returncode != 0:
            last = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
            errs.append(f"J1 `{p.name}` **文件在、但 import 就炸** "
                        f"（=死脚本，跑不到 main）：{last[-1] if last else '(无 stderr)'}")
    if unshielded:
        print(f"  ⚠️ 未测（没有 `__main__` 守卫，import 会执行主逻辑）：{unshielded}")
    return (not errs), errs


# ---------------------------------------------------------------------------
# J2 —— 第 2 环的三条纪律，逐条喂反向控制
# ---------------------------------------------------------------------------
def j2_wiring(repo: Path, tmp: Path) -> tuple[bool, list[str], dict]:
    """必须以**真脚本、真判定**跑，而不是读源码猜。"""
    errs, codes = [], {}
    sdir = repo / "paper2skills-research" / "scripts"
    runner = sdir / "rank_candidates.py"
    if not runner.is_file():
        return False, ["J2 `rank_candidates.py` 不存在"], {}

    def run(pool: Path | None) -> subprocess.CompletedProcess:
        argv = [sys.executable, str(runner), "--out-dir", str(tmp / "out")]
        if pool is not None:
            argv += ["--pool", str(pool)]
        return subprocess.run(argv, cwd=str(sdir), capture_output=True, text=True, timeout=600)

    # ① 过滤产物缺失 ⇒ exit 2（**不许静默回退到未过滤池**）
    r = run(tmp / "does-not-exist.json")
    codes["缺产物"] = r.returncode
    if r.returncode != 2:
        errs.append(f"J2 过滤产物不存在时必须 **exit 2**（没测到 ≠ 通过），实得 {r.returncode}"
                    + ("  ← 0 = **静默回退**，那会让「接线了」与「没接线」长得一模一样"
                       if r.returncode == 0 else ""))

    # ② **未过滤池改名冒充**过滤产物 ⇒ 仍须 exit 2（判据看自述块，不看文件名）
    raw = tmp / "arxiv_candidates_filtered.json"
    shutil.copyfile(repo / "paper2skills-research" / "data" / "arxiv_candidates.json", raw)
    r = run(raw)
    codes["冒充"] = r.returncode
    if r.returncode != 2:
        errs.append(f"J2 未过滤池**改名冒充**过滤产物时必须 exit 2（判据看 `filter` 自述块，"
                    f"不看文件名），实得 {r.returncode}")

    # ③ `unjudged_ids` 非空 ⇒ exit 2（「仪器瞎了」是危险性 3，不许算成通过）
    fp = repo / "paper2skills-research" / "data" / "arxiv_candidates_filtered.json"
    if fp.is_file():
        obj = json.loads(fp.read_text(encoding="utf-8"))
        obj["filter"]["unjudged_ids"] = ["dummy-1"]
        bad = tmp / "unjudged.json"
        bad.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        r = run(bad)
        codes["unjudged"] = r.returncode
        if r.returncode != 2:
            errs.append(f"J2 `unjudged_ids` 非空时必须 exit 2，实得 {r.returncode}")
        # ④ 反向控制：**同一份产物、只把 unjudged 清空** ⇒ 必须 exit 0。
        #    少了这条，③ 可能只是因为「这份夹具本来就跑不过」而红（红得不是地方）。
        obj["filter"]["unjudged_ids"] = []
        good = tmp / "ok.json"
        good.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        r = run(good)
        codes["干净夹具"] = r.returncode
        if r.returncode != 0:
            errs.append(f"J2 **反向控制**：干净夹具必须 exit 0（否则判据过严 ⇒ 会被绕过），"
                        f"实得 {r.returncode}\n       stderr: {r.stderr.strip()[-200:]}")
        # ⑤ 干净夹具真跑完之后，产物里必须带 `filter` 自述块 ——
        #    「接线」的证据是**产物里那一栏**，不是源码里出现过那个常量名。
        rec = tmp / "out" / "recommendations.json"
        if not rec.is_file():
            errs.append("J2 **端到端**：夹具跑完却没有产出 recommendations.json")
        else:
            d = json.loads(rec.read_text(encoding="utf-8"))
            flt = d.get("filter")
            if not isinstance(flt, dict):
                errs.append("J2 **端到端**：产物里没有 `filter` 自述块 ⇒ 下游无从知道"
                            "「过了哪一版词表、丢了哪些」")
            else:
                codes["产物条数"] = (len(d["items"]), flt.get("count"))
                if len(d["items"]) != flt.get("count"):
                    errs.append(f"J2 **端到端**：产物 items={len(d['items'])} "
                                f"与 filter.count={flt.get('count')} 不符")
                if not flt.get("dropped_ids") and flt.get("count_before", 0) > flt.get("count", 0):
                    errs.append("J2 **端到端**：明明丢了论文，`dropped_ids` 却是空的 ⇒ "
                                "「274 篇去哪了」在链上不可追溯")
    else:
        errs.append(f"J2 过滤产物不存在：{fp} —— 先跑 candidate_filter.py --apply")
    return (not errs), errs, codes


# ---------------------------------------------------------------------------
# J3 —— **一次性初始化器**默认不许改写唯一事实源
# ---------------------------------------------------------------------------
def j3_no_clobber(repo: Path) -> tuple[bool, list[str], dict]:
    """⚠️ 这条判据**看行为不看措辞**：默认跑一遍，盘上那份必须逐字节不变。

    为什么不是「源码里有没有写 `--force-reinit`」：那种判据只能证明**有人说了一句话**，
    证明不了**跑一遍会不会毁数据** —— 而 #114 的全部危险恰恰在于后者
    （源码读起来一直是对的，是**前向引用**让它跑不起来，从而没人发现它跑起来会做什么）。
    """
    errs = []
    br = repo / "paper2skills-research" / "scripts" / "build_registry.py"
    reg = repo / "paper2skills-vault" / "07-资源库" / "papers_registry.json"
    if not br.is_file() or not reg.is_file():
        return False, [f"J3 输入没拿到：{br.name} / {reg.name}"], {}
    before = _sha(reg)
    r = subprocess.run([sys.executable, str(br)], cwd=str(br.parent),
                       capture_output=True, text=True, timeout=600)
    after = _sha(reg)
    info = {"exit": r.returncode, "产物 sha 前 8": before[:8], "后 8": after[:8]}
    if before != after:
        errs.append(f"J3 **默认跑一遍就改写了唯一事实源**（sha {before[:8]} → {after[:8]}）—— "
                    f"它是**一次性初始化器**，实测会改写 39/45 条人工核对过的记录。"
                    f"默认必须只对账、不写盘。")
    if r.returncode not in (0, 1):
        errs.append(f"J3 默认跑一遍的退出码 {r.returncode} 不在 {{0 对账完成, 1 拒绝写盘}} 里"
                    f"（2/3 说明它连输入都没拿到）")
    if "--force-reinit" not in br.read_text(encoding="utf-8"):
        errs.append("J3 没有 `--force-reinit` 这道显式闸门 ⇒ 写盘无从留痕")
    # 覆盖账必须**现算**：若它被冻成一个常数，事实源长出新手工内容时它不会跟着变。
    if "overwrite_account" not in br.read_text(encoding="utf-8"):
        errs.append("J3 覆盖账不是现算的（找不到 `overwrite_account`）")
    return (not errs), errs, info


# ---------------------------------------------------------------------------
def check(repo: Path = REPO, spec_path: Path = SPEC_DOC) -> int:
    print(f"第 2 环接线门禁 · 仓库 {repo}")
    scripts = spec_scripts(spec_path)
    if scripts is None:
        print(f"🔴 规格块读不到（{spec_path}）—— 「没东西可查」不等于「查过了没问题」，exit 2")
        return 2
    if not scripts:
        print("🔴 规格块里一个 `.py` 都没点名 —— 空块等于没声明，exit 2")
        return 2
    frozen = {str(p): _sha(p) for p in FROZEN}

    errs: list[str] = []
    print(f"\n=== J1 规格块点名的脚本必须**加载得起来**（{len(scripts)} 个）===")
    ok1, e1 = j1_importable(scripts, repo)
    print(f"  {'✅' if ok1 else '🔴'} import 探针：{len(scripts) - len(e1)}/{len(scripts)} 通过")
    errs += e1

    with tempfile.TemporaryDirectory() as td:
        print(f"\n=== J2 第 2 环三条纪律（真脚本 · 真判定 · 反向控制）===")
        ok2, e2, codes = j2_wiring(repo, Path(td))
        print(f"  {'✅' if ok2 else '🔴'} 退出码：{codes}")
        errs += e2

    print(f"\n=== J3 一次性初始化器默认不许改写唯一事实源 ===")
    ok3, e3, info = j3_no_clobber(repo)
    print(f"  {'✅' if ok3 else '🔴'} {info}")
    errs += e3

    # --- 自证：本门禁**不得改动仓库真产物**（前后 sha 相等） -------------------
    print(f"\n=== 自证：本门禁没有改动仓库真产物 ===")
    drift = [k for k, v in frozen.items() if _sha(Path(k)) != v]
    for k in sorted(frozen):
        print(f"  {'✅' if str(k) not in drift else '🔴'} {Path(k).name} {_sha(Path(k))[:8]}")
    if drift:
        errs.append(f"自证失败：本门禁自身改动了真产物 {[Path(x).name for x in drift]} —— "
                    f"「门禁自己动过的东西」不能再被当成独立证据")

    if errs:
        print(f"\n🔴 {len(errs)} 条不成立：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print("\n✅ 第 2 环已接线：脚本加载得起来 · 三条纪律都能被打红 · "
          "初始化器默认不改写事实源 · 本门禁自身未改动真产物")
    return 0


# ---------------------------------------------------------------------------
def selftest() -> int:
    """每一条判据都要有**能失败**的证据，外加反向控制「干净夹具必须 exit 0」。"""
    ok = True

    def expect(cond: bool, desc: str) -> None:
        nonlocal ok
        ok &= cond
        print(f"  {'✅' if cond else '❌'} {desc}")

    print("=== 夹具：一个「干净」的子仓库 ===")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def mk_repo(name: str, *, build_registry_body: str, spec: str) -> Path:
            repo = tmp / name
            sd = repo / "paper2skills-research" / "scripts"
            sd.mkdir(parents=True)
            (repo / "paper2skills-research" / "data").mkdir(parents=True)
            (repo / "paper2skills-vault" / "07-资源库").mkdir(parents=True)
            for fn, body in {
                "arxiv_harvest.py": "# 收割\n",
                "candidate_filter.py": "# 过滤\n",
                "rank_candidates.py": "# 打分\n",
                "build_registry.py": build_registry_body,
                "make_bundles.py": "# 束\n",
                "dedup_check.py": "# 去重\n",
            }.items():
                (sd / fn).write_text(body, encoding="utf-8")
            sp = tmp / f"{name}-spec.md"
            sp.write_text(spec, encoding="utf-8")
            return repo

        SPEC_OK = ("<!-- P2S-SPEC:pipeline\n"
                   "  1 收割  arxiv_harvest.py\n"
                   "  2 过滤  candidate_filter.py        wired=true\n"
                   "  3 打分  rank_candidates.py\n-->\n")
        ALIVE = "if __name__ == '__main__':\n    pass\n"

        # ① J1 的正向：干净夹具必须**不报**（否则判据过严 ⇒ 会被绕过）
        clean = mk_repo("clean", build_registry_body=ALIVE, spec=SPEC_OK)
        ok1, e1 = j1_importable(["arxiv_harvest.py", "candidate_filter.py",
                                 "rank_candidates.py"], clean)
        expect(ok1, f"① 干净夹具 J1 必须通过（实得 {len(e1)} 条错）")

        # ② J1 的反向：#113 的真实形态 —— 文件在、import 就炸。
        dead = mk_repo("dead", build_registry_body=ALIVE + "\nGUARD = MISSING_NAME\n",
                       spec=SPEC_OK)
        ok2, e2 = j1_importable(["build_registry.py"], dead)
        expect((not ok2) and any("import 就炸" in x for x in e2),
               f"② **#113 的真实形态必须被打红**（文件在、import 就炸）→ {e2[:1]}")

        # ③ J1 的另一种漏法：文件**根本不存在** ⇒ 也要红（不是只有 import 才判）
        ok3, e3 = j1_importable(["no_such_thing.py"], dead)
        expect(not ok3, f"③ 点名的脚本不存在必须判红 → {e3[:1]}")

        # ④ J3 的反向：一个**默认就写盘**的初始化器必须被判红。
        CLOBBER = (ALIVE +
                   "from pathlib import Path\n"
                   "REG = Path(__file__).resolve().parents[2] / 'paper2skills-vault' / "
                   "'07-资源库' / 'papers_registry.json'\n"
                   "if __name__ == '__main__':\n"
                   "    REG.write_text('{\"records\": []}', encoding='utf-8')\n")
        clob = mk_repo("clobber", build_registry_body=CLOBBER, spec=SPEC_OK)
        reg = clob / "paper2skills-vault" / "07-资源库" / "papers_registry.json"
        reg.write_text('{"records": [1, 2, 3]}', encoding="utf-8")
        ok4, e4, _ = j3_no_clobber(clob)
        expect(not ok4, f"④ 默认就写盘的初始化器必须判红 → {e4[:1]}")

        # ⑤ J3 的正向：干净的、只对账的初始化器必须**不报**
        SAFE = (ALIVE +
                "from pathlib import Path\n"
                "REG = Path(__file__).resolve().parents[2] / 'paper2skills-vault' / "
                "'07-资源库' / 'papers_registry.json'\n"
                "def overwrite_account():\n    return REG.exists()\n"
                "if __name__ == '__main__':\n"
                "    print('--force-reinit' if overwrite_account() else 'no reg')\n")
        safe = mk_repo("safe", build_registry_body=SAFE, spec=SPEC_OK)
        (safe / "paper2skills-vault" / "07-资源库" / "papers_registry.json").write_text(
            '{"records": [1]}', encoding="utf-8")
        ok5, e5, _ = j3_no_clobber(safe)
        expect(ok5, f"⑤ 干净夹具 J3 必须通过（实得 {e5}）")

        # ⑥ J2 的反向：规格块读不到 / 空块 ⇒ **exit 2**（没测到 ≠ 通过）
        expect(spec_scripts(tmp / "nope.md") is None, "⑥a 规格文档不存在 ⇒ 读不到（exit 2）")
        empty = tmp / "empty.md"
        empty.write_text("<!-- P2S-SPEC:pipeline\n  # 什么都没点名\n-->\n", encoding="utf-8")
        expect(spec_scripts(empty) == [], "⑥b 空规格块 ⇒ 空列表（调用方必须 exit 2）")
        bad_block = tmp / "noblock.md"
        bad_block.write_text("# 只有散文，没有机读块\n", encoding="utf-8")
        expect(spec_scripts(bad_block) is None, "⑥c 没有机读块 ⇒ 读不到（exit 2）")

    print("✅ 自检通过：J1/J2/J3 各有能失败的反向控制" if ok else "❌ 自检失败")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="第 2 环接线门禁（PHASE6 #105）")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--spec", type=Path, default=SPEC_DOC)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return check(a.repo, a.spec)


if __name__ == "__main__":
    sys.exit(main())
