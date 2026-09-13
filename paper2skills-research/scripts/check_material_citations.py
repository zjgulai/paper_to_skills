#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""材料引文核验器 —— 契约里凡声称「材料『X』」的 X，必须真的在材料里，且**逐字**是对的。

## 这个东西为什么存在

S1 收口期抓到一条：**67 份契约（139 的 48%）引用「季度经营策略」**，而材料 0 命中。
追到源头是 **F7 的 v2 范本契约 `contracts/v2/B/CTR-B-023` 第 31 行** ——
范本把「材料已定名的经营节奏（「月度经营复盘」＝1 个自然月、**「季度经营策略」＝1 个自然季度**）」
写进了给全体撰写人照抄的样板块。**范本是设计上就会被抄的东西**，所以一个自造词两天长成 108 处。

这与 `撰写规范.md` §2 记的 `业务规则字典` 事件同型（自造词经两层传播后被当成材料词汇引用），
但规模大一个量级，且这一次**种子就在我们自己的模板里**。

## 它把「三种罪」分开 —— 这是本脚本存在的全部理由

第一版只问「这个词在不在材料里」，跑出 22 条「查无此词」。逐条追上下文后发现**大部分是仪器自己的假阳性**：
契约在写**自己的标签**（`外发材料只能写「非增量」`、`该对象的材料只能作「客户陈述」`）
或**自己的产物名**（`人工补录的材料清单`、`每一份材料登记成「对象×时间戳…」`），
`原材料成本冲击` 里的 `材料` 更是把整条引文劫持了。**照第一版报，会制造 20 条假红。**

所以现在分三类判定 + 两类罪：

| 判定 | 含义 |
|---|---|
| `attrib` | `材料` 确是引文来源 ⇒ 进入核验 |
| `non_attrib` | `材料` 只是句子里的普通名词（外发材料/原材料/的材料/每条材料）⇒ **不判** |
| `unclassified` | 限定语形态从没见过 ⇒ **不猜**，单独报出并 exit 3（漏测 ≠ 通过） |

| 罪 | 判据 |
|---|---|
| `missing` | 引文在材料里找不到 |
| `altered` | 能找到**近逐字**的原文但字面不同 —— 尤其**声称「原文」却在改字**时必须报 |

`altered` 会**回捞材料原句**打印出来，让「改了什么」当场可见，而不是只丢一句「不一致」。

## 三条纪律（都来自仓库既有台账）
1. **「材料里没有 X」之前先问「我用的仪器能看见 X 吗」**（铁律 1 的推广）⇒ 每次运行跑**活体探针**，
   阳性必命中、阴性必不命中，探针不通 ⇒ **exit 3**，绝不退化成 exit 0。
2. **「没扫到文件」不是「扫过了没问题」** ⇒ 材料根不存在/为空 ⇒ **exit 2**，不是 0。
   （`scan_secrets.py` 同一纪律；也是门禁缺陷 #42（J12 依赖本机 HOME）的形状。）
3. **豁免必须可见** ⇒ `non_attrib` 与 `v1 免检` 一样，**份数与名单是一等输出**，
   否则「我判它不算引文」就成了新后门。

## 退出码
  0 = 全部有据；1 = 有 missing/altered；2 = 材料没拿到（≠ 通过）；3 = 仪器自检或分类不过（≠ 判红）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

MATERIAL = "材料"
OPEN_Q = "「『\"“"
CLOSE_Q = "」』\"”"
PAIR_RE = re.compile(
    r"「([^」\n]{1,60})」|『([^』\n]{1,60})』|“([^”\n]{1,60})”|\"([^\"\n]{1,60})\""
)

# ---------------------------------------------------------------------------
# 左邻判定：`材料` 左边是什么，决定它是「引文来源」还是「句子里的普通名词」
# 黑名单来自 139 份契约的全量实测（左邻字符普查），不是设想：
#   外发材料 / 申诉材料 / 原材料 / 该对象的材料 / 每一份材料 / 每条材料 / 无材料级 / 在材料清单
# ---------------------------------------------------------------------------
LEFT_BLACK = set("发诉原的外条无在份级")
# 白名单：这些左边的「材料」实测全是引文来源（按材料/依据是材料/取自材料/与材料…一致/系材料…）
LEFT_WHITE = set("按据依见取齐是与自系即而以且把考为把如依")

# ---------------------------------------------------------------------------
# 限定语形态：白名单收录实测出现过的**归属**形态；黑名单收录实测出现过的**非归属**形态
# ---------------------------------------------------------------------------
RUN_ATTRIB = re.compile(
    r"^(?:"
    r"|已定(?:名)?的[^「『\"]{0,16}|已定(?:名)?[^「『\"]{0,16}"
    r"|原文[：:]?[^「『\"]{0,12}|口径(?:明写|记作)?[^「『\"]{0,12}"
    r"|的?经营节奏[^「『\"]{0,8}|通用(?:工单)?规则|里的|明写[^「『\"]{0,12}"
    r"|判断原则|同时给了它的降级条件[^「『\"]{0,24}"
    r"|只定名了[^「『\"]{0,16}|给[^「『\"]{0,24}|对[^「『\"]{0,24}|为该[^「『\"]{0,24}"
    r"|PB-\d+[^「『\"]{0,16}|§[^「『\"]{0,20}|把"
    r"|同向的二值口径（PB-\d+|记为"
    r")$"
)
RUN_NONATTRIB = re.compile(
    r"(?:只能|不得|只写|只取|不含|登记成|逐条|标注|并标|顶替|卡面举例|中写|清单标"
    r"|上标注|等级|成本冲击|例[（,，]|^标$|\|)"
)


def normalize(s: str) -> str:
    """markdown 强调符与空白抹掉，其余（含标点）保留 —— 抹多了会把不同的词抹成同一个。

    ⚠️ **必须用 `str.maketrans`，不能直接给 `str.translate` 传 `{c: None}`**：
    `str.translate` 拿 dict 表时是**按 ordinal 查键**的，字符键一个都命不中 ⇒
    整行翻译**静默什么都不做**，而代码看起来完全正确。实测代价：
    `材料原文「**能力新版本失败时回退版本并保留业务效果记录**」`（与材料逐字相同）
    被判成「声称引用却改了字」—— **仪器自己的假红**。
    selftest 用例⑩直接断言函数行为，不让它再静默退化。
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans("", "", "*_`>#|~"))
    return re.sub(r"\s+", "", s)


def strip_punct(s: str) -> str:
    """只留汉字/字母/数字 —— 用于「声称逐字引用」的对照（标点差异不算改字）。"""
    return re.sub(r"[^\w]", "", normalize(s))


# ---------------------------------------------------------------------------
# 活体探针（每次运行都跑）
# ---------------------------------------------------------------------------
PROBE_POSITIVE = ["月度经营复盘", "经营复盘", "其余 20% 渠道构成未明"]
PROBE_NEGATIVE = ["季度经营策略", "\u0000__本词必然不存在__\u0000"]


def load_corpus(root: Path) -> tuple[str, str, int]:
    parts: list[str] = []
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".md", ".json", ".txt", ".csv", ".yml", ".yaml"):
            continue
        # ⚠️ 材料目录**自己是个 git 仓库** —— `.git/` 下有 loose objects、refs、packed-refs。
        #    实测首版把这些也读进了语料（`grep -rl F\\.3` 命中过 `.git/objects/...`）。
        #    后果不是「多读了点字」而是**假绿**：一个材料正文里没有的词，可能因为存在于
        #    某个历史 blob 里而被判「有据」。判据读的必须是材料**内容**，不是它的仓库元数据。
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        try:
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        n += 1
    raw = "\n".join(parts)
    return normalize(raw), raw, n


def longest_hit(term: str, corpus: str) -> str:
    best = ""
    for i in range(len(term)):
        for j in range(len(term), i, -1):
            if j - i <= len(best):
                break
            if term[i:j] in corpus:
                best, _ = term[i:j], None
                break
    return best


def context_of(needle: str, raw: str, width: int = 70) -> str:
    """把材料里有这个片段的**原句**捞出来 —— 让「改了什么」当场可见。

    ⚠️ 首版只给 needle 去了标点、**没给草垛去**，于是 `…责任负荷，不将其…`（草垛有 `，`）
    永远找不到 `…责任负荷不将其…`（针已去标点）⇒ 回捞恒为空，
    `altered` 退化成「只说不一样、不说哪不一样」。selftest 用例③当场抓住。
    修法：草垛与针**同一套归一化**再比。
    """
    if not needle:
        return ""
    flat = strip_punct(raw)
    i = flat.find(strip_punct(needle))
    if i < 0:
        return ""
    return flat[max(0, i - width): i + len(strip_punct(needle)) + width]


def run_probes(corpus: str) -> list[str]:
    bad = []
    for t in PROBE_POSITIVE:
        if normalize(t) not in corpus:
            bad.append(f"阳性对照「{t}」没命中 —— 材料读错了，或归一化吃掉了真词")
    for t in PROBE_NEGATIVE:
        if normalize(t) in corpus:
            bad.append(f"阴性对照「{t}」竟然命中 —— 判据本身失效")
    return bad


def extract(files: list[Path]):
    attrib: dict[str, list] = {}
    non: dict[str, list] = {}
    unclass: dict[str, list] = {}
    for p in files:
        for ln_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for pm in PAIR_RE.finditer(line):
                term = next(g for g in pm.groups() if g is not None).strip()
                head = line[: pm.start()]
                seg = re.split(r"[。；\n]", head)[-1]
                pos = seg.rfind(MATERIAL)
                if pos < 0 or len(seg) - (pos + len(MATERIAL)) > 20:
                    continue
                left = seg[pos - 1] if pos > 0 else "␀"
                run = seg[pos + len(MATERIAL):]
                # ⚠️ 一行里有多个引号时，第 2 个引号的「head 片段」会把**第 1 个引号**也算进 run，
                #    实测 `为该格定的检查项里就有「外部承诺」与「恢复条件」` 会把 run 记成
                #    `为该格定的检查项里就有「外部承诺」与` ⇒ 形态白名单永远认不出它。
                #    材料引文只可能是**紧随材料的那一个**引号，故这里的 run 一旦含引号就跳过。
                if any(q in run for q in OPEN_Q + CLOSE_Q):
                    continue
                loc = (p.name, ln_no)
                if "\u4e00" <= left <= "\u9fff" and left not in LEFT_WHITE:
                    non.setdefault(term, []).append(loc)
                    continue
                if RUN_NONATTRIB.search(run):
                    non.setdefault(term, []).append(loc)
                    continue
                if RUN_ATTRIB.match(run.strip()) or run.strip() == "":
                    attrib.setdefault(term, []).append((loc, run.strip()))
                    continue
                unclass.setdefault(run.strip(), []).append((term, loc))
    return attrib, non, unclass


def verdict(term: str, corpus: str, raw: str) -> dict:
    t = normalize(term)
    if "…" in term or "..." in term:
        frags = [f for f in re.split(r"…|\.\.\.", t) if len(strip_punct(f)) >= 2]
        bad = [f for f in frags if strip_punct(f) and strip_punct(f) not in strip_punct(corpus)]
        return {"kind": "fragment", "ok": not bad, "bad_fragments": bad}
    if t in corpus:
        return {"kind": "exact", "ok": True}
    lh = longest_hit(t, corpus)
    if len(strip_punct(lh)) >= max(6, len(strip_punct(t)) // 2):
        return {"kind": "altered", "ok": False, "longest_hit": lh,
                "material_context": context_of(lh, raw)}
    return {"kind": "missing", "ok": False, "longest_hit": lh,
            "material_context": context_of(lh, raw) if len(strip_punct(lh)) >= 3 else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default=os.environ.get("P2S_MATERIAL_DIR",
                                                         "/Users/lute/project/AI组织变革"))
    ap.add_argument("--dir", default="paper2skills-vault/07-资源库/contracts")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show-nonattrib", action="store_true",
                    help="把被判为「非引文」的词与份数打出来（豁免必须可见）")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    root = Path(args.material)
    if not root.is_dir():
        print(f"❌ 材料根不存在：{root} —— 「没拿到输入」不等于「通过」", file=sys.stderr)
        return 2
    corpus, raw, nfiles = load_corpus(root)
    if nfiles == 0 or len(corpus) < 1000:
        print(f"❌ 材料读空了（{nfiles} 文件 / {len(corpus)} 字符）—— 同上，exit 2 不是 0", file=sys.stderr)
        return 2
    probes = run_probes(corpus)
    if probes:
        print("❌ 仪器自检不过（这是仪器的问题，不是被检对象的问题）：", file=sys.stderr)
        for b in probes:
            print("   -", b, file=sys.stderr)
        return 3

    base = Path(args.dir)
    files = sorted(list((base / "A").glob("CTR-*.md")) + list((base / "B").glob("CTR-*.md")))
    if not files:
        print(f"❌ 契约目录里没扫到文件：{base}/A|B", file=sys.stderr)
        return 2

    attrib, non, unclass = extract(files)
    print(f"材料根：{root}（{nfiles} 个文件）· 扫描 {len(files)} 份契约")
    print(f"活体探针：阳性 {len(PROBE_POSITIVE)}/{len(PROBE_POSITIVE)} 命中 · "
          f"阴性 {len(PROBE_NEGATIVE)}/{len(PROBE_NEGATIVE)} 未命中 ⇒ 仪器通过")
    print(f"引文归属：**{sum(len(v) for v in attrib.values())}** 条判为材料引文（{len(attrib)} 个不同词）· "
          f"**{sum(len(v) for v in non.values())}** 条判为「材料只是普通名词」（{len(non)} 个词，不计）")

    bad: dict[str, dict] = {}
    for term, locs in sorted(attrib.items()):
        v = verdict(term, corpus, raw)
        if not v["ok"]:
            bad[term] = {**v, "count": len(locs),
                         "files": sorted({f for (f, _), _ in locs}),
                         "runs": sorted({r for _, r in locs}),
                         "sample": locs[0][0]}

    if bad:
        print(f"\n❌ **材料引文有问题：{len(bad)} 个词**\n")
        for term, d in sorted(bad.items(), key=lambda kv: -kv[1]["count"]):
            tag = {"missing": "查无此词", "altered": "**声称引用却改了字**",
                   "fragment": "省略号片段有片段找不到"}[d["kind"]]
            print(f"  「{term}」 —— {d['count']} 处 / {len(d['files'])} 份契约 · {tag}")
            if d.get("longest_hit"):
                print(f"      最长可命中片段：『{d['longest_hit']}』"
                      f"（{len(strip_punct(d['longest_hit']))} 字）")
            if d.get("material_context"):
                print(f"      材料原句：…{d['material_context']}…")
            if d.get("bad_fragments"):
                print(f"      找不到的片段：{d['bad_fragments']}")
            print(f"      样例：{d['sample'][0]}:{d['sample'][1]}")

    if args.show_nonattrib:
        print(f"\n（参考）判为「非引文」的词 —— 豁免必须可见：")
        for term, locs in sorted(non.items(), key=lambda kv: -len(kv[1]))[:25]:
            print(f"    {len(locs):3d}  「{term}」")

    if unclass:
        print(f"\n⚠️ **{len(unclass)} 种限定语形态超出仪器的判定范围**（不是判红，是**没测到**）：")
        for run, items in sorted(unclass.items(), key=lambda kv: -len(kv[1]))[:20]:
            term, loc = items[0]
            print(f"    run=│{run}│ × {len(items)}  例：「{term}」 @ {loc[0]}:{loc[1]}")

    payload = {
        "material_root": str(root), "material_files": nfiles,
        "contracts_scanned": len(files),
        "attrib_citations": sum(len(v) for v in attrib.values()),
        "attrib_terms": len(attrib),
        "non_attrib_terms": {k: len(v) for k, v in non.items()},
        "unclassified": {k: len(v) for k, v in unclass.items()},
        "problems": bad,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")

    if unclass:
        print("\n⇒ exit 3：有引文形态没被测到。**先把它归类**（加进 RUN_ATTRIB 或 RUN_NONATTRIB），"
              "再跑 —— 漏测不许当成通过。")
        return 3
    if bad:
        print("\n⇒ exit 1：材料引文有问题。处置由人定（换材料真词 / 改标「业务侧默认＋替换条件」/"
              "改成逐字原文）。")
        return 1
    print("\n⇒ exit 0：全部材料引文有据。")
    return 0


def selftest() -> int:
    import tempfile
    cases = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mat = tmp / "mat"
        (mat / "docs").mkdir(parents=True)
        (mat / "docs" / "p.md").write_text(
            "# AGT-001\n业务技能：**月度经营复盘**。\n"
            "FLOW-08 经营复盘。\n"
            "若释放工时没有降低峰值或责任负荷，不将其直接换算为人员变化。\n"
            "企业自承「其余 20% 渠道构成未明」。\n" + "填充。" * 400, encoding="utf-8")
        corpus, raw, _ = load_corpus(mat)
        assert not run_probes(corpus)

        def cls(body: str):
            d = tmp / "c" / "A"
            d.mkdir(parents=True, exist_ok=True)
            (d / "CTR-A-999-样本.md").write_text(body, encoding="utf-8")
            return extract(sorted((d).glob("CTR-*.md")))

        # ① 归属：材料真有的词
        a, n, u = cls("> 来源＝材料「月度经营复盘」＝1 个自然月。")
        cases.append(("① 真引文 ⇒ 归 attrib 且 ok", a and all(verdict(t, corpus, raw)["ok"] for t in a)))
        # ② 自造词 ⇒ 必须 missing
        a, _, _ = cls("> 来源＝材料「季度经营策略」＝1 个自然季度。")
        cases.append(("② 材料没有的词 ⇒ missing", any(verdict(t, corpus, raw)["kind"] == "missing" for t in a)))
        # ③ 改字却称「原文」⇒ 必须 altered 并回捞材料原句
        a, _, _ = cls("材料 PB-008 原文「释放工时若未降低峰值或责任负荷，不将其直接换算为人员变化」")
        v = [verdict(t, corpus, raw) for t in a]
        cases.append(("③ 称「原文」而改字 ⇒ altered 并给出材料原句",
                      any(x["kind"] == "altered" and "没有降低峰值" in x.get("material_context", "") for x in v)))
        # ④ 外发材料只能写「X」⇒ 非引文（这是第一版制造 20 条假红的那一族）
        a, n, _ = cls("- 结论字段与外发材料，只能写站内对照结论并标注「非增量」。")
        cases.append(("④「外发材料…只能写『X』」⇒ non_attrib，不得判红", not a and bool(n)))
        # ⑤ 原材料成本冲击 ⇒ 非引文（子串劫持）
        a, n, _ = cls("同卡工具变量清单（竞争对手价格／汇率波动／原材料成本冲击，例「芯片短缺」）")
        cases.append(("⑤「原材料…」子串劫持 ⇒ non_attrib", not a and bool(n)))
        # ⑥ 省略号引文 ⇒ 按片段核验
        a, _, _ = cls("对齐材料「核对…商品主数据…宣称」的口径")
        v = [verdict(t, corpus, raw) for t in a]
        cases.append(("⑥ 省略号引文 ⇒ fragment 且片段各自核验", any(x["kind"] == "fragment" for x in v)))
        # ⑦ 未见过的限定语形态 ⇒ unclassified（漏测 ≠ 通过）
        a, n, u = cls("材料某种没见过的说法「月度经营复盘」")
        cases.append(("⑦ 未见过的限定语 ⇒ unclassified（不猜）", bool(u)))
        # ⑧ 材料根不存在 ⇒ exit 2
        old = sys.argv
        sys.argv = ["x", "--material", str(tmp / "nope")]
        try:
            rc = main()
        finally:
            sys.argv = old
        cases.append(("⑧ 材料根不存在 ⇒ exit 2", rc == 2))
        # ⑨ 材料读错 ⇒ 探针失配
        broken = tmp / "m2"
        (broken / "d").mkdir(parents=True)
        (broken / "d" / "x.md").write_text("y" * 2000, encoding="utf-8")
        c2, _, _ = load_corpus(broken)
        cases.append(("⑨ 材料读错 ⇒ 探针失配（不得静默）", bool(run_probes(c2))))
        # ⑩ 直接断言 normalize 的行为 —— 它曾用 `str.translate({c: None})` 静默什么都不做
        cases.append(("⑩ normalize 必须真的删掉 `**` 强调符",
                      normalize("**必须回放**") == "必须回放" and normalize("a*b_c") == "abc"))
        # ⑪ 带 `**` 的引文与材料逐字相同时 ⇒ 必须判 ok，不得报「改了字」
        a, _, _ = cls("材料原文「**月度经营复盘**」")
        cases.append(("⑪ 引文含 `**` 但逐字相同 ⇒ 不得判 altered",
                      bool(a) and all(verdict(t, corpus, raw)["ok"] for t in a)))

    ok = True
    print("check_material_citations · selftest")
    for name, good in cases:
        print(f"  {'✅' if good else '❌'} {name}")
        ok &= bool(good)
    print(f"\n{len(cases)}/{len(cases)} 通过" if ok else "\n有用例未过")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
