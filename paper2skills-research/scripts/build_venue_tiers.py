#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PHASE6 · S10 · venue 三词表统一 + 回填 + 分域定线（唯一机读实现）

本脚本是 venue 词表的**唯一实现处**。四条可能的第二事实源都在这里被约束：

  ① `paper2skills-vault/07-资源库/venue-whitelist.md` §10 的映射表与域线表
     —— 由 `render_inventory_block()` 生成，`--check` 逐字比对（J4/J5）。
  ② `papers_registry.json` 的 `venue_tier`
     —— **只读不写**（PHASE6 F2 硬约束：不造第二份事实源）；表外取值登记在
        `REGISTRY_ALIAS`，逐条带理由（J7）。
  ③ 卡片 frontmatter 的 `venue_tier`
     —— 本脚本是唯一写者（`--apply`），且**只增不改**（J12）。
  ④ 产品侧 `dsh-paper2skills/lib/axis.js` 的 7 档常量
     —— 本脚本把规范表**投递**为 `data/venue-tiers.json`（`--sync-product`），
        `--check` 断言逐字节相等（J9）。产品侧只消费，不定义。

判据（每条都配一份**能打红它的样本**，见 `MUTATIONS`）：

  J1  卡片侧 venue_tier ⊆ 规范 7 档 ∪ {unlabeled}
  J2  覆盖率【下界】≥ COVERAGE_MIN
  J3  覆盖率【上界】≤ COVERAGE_MAX；分子分母各自数出来，等式自洽
  J4  venue-whitelist.md §10 机读块与生成器逐字相等
  J5  §1 散文 7 档表 == 规范 7 档
  J6  同一卡 frontmatter 字段不重复（档位取值唯一）
  J7  三套词表的每个非规范取值都有映射规则；registry 表外取值逐条有理由
  J8  分域线：每张卡的域在域线表里有归属；类别有双侧规则；依据不是形容词
  J9  产品侧投递副本与规范表逐字节相等
  J10 有 paper_id 的卡必须能说出判定依据，**且**判定已完成（无 needs-resolution / 无元数据缺档）
  J11 台账完备：登记为不可回填的卡必须在册，且在册的卡必须真的不可回填
  J12 只增不改：原值非空时不得被改写（除非在 EXPLICIT_RETIERS 里点名并给出理由）
  J13 身份对齐：`paper_id` 指向的 arXiv 论文与卡片声明的论文必须对得上；
      对不上的一律进 `IDENTITY_MISMATCH` 名单，其 tier 由**实际取到的那个号**的元数据判，
      且必须逐条写明理由（铁律 1 的仪器必须指对地方）

退出码：0 全过 / 1 判红 / 2 输入没拿到（≠ 通过）/ 3 内部错误（≠ 判红）
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VAULT = REPO / "paper2skills-vault"
RESLIB = VAULT / "07-资源库"
WHITELIST_MD = RESLIB / "venue-whitelist.md"
REGISTRY = RESLIB / "papers_registry.json"
DATA = REPO / "paper2skills-research" / "data"
MAP_JSON = DATA / "venue_tier_mapping.json"
BACKFILL_JSON = DATA / "venue_tier_backfill.json"
ARXIV_META = DATA / "venue_sources" / "arxiv_abs.json"
CROSSREF_META = DATA / "venue_sources" / "crossref.json"
PRODUCT_PKG = Path(
    os.environ.get(
        "P2S_PRODUCT_PKG",
        str(Path.home() / "project" / "Magpie-Horch" / "packages" / "capabilities" / "dsh-paper2skills"),
    )
)
PRODUCT_VENDORED = PRODUCT_PKG / "data" / "venue-tiers.json"
PRODUCT_AXIS = PRODUCT_PKG / "lib" / "axis.js"

# --------------------------------------------------------------------------- #
# 规范词表（唯一实现）                                                          #
# --------------------------------------------------------------------------- #

#: 白名单 7 档。事实源：`venue-whitelist.md` §1。
CANONICAL_TIERS = ["UTD24", "FT50", "CCF-A", "CCF-B", "field-top", "preprint", "non-paper"]

#: 「没有档位」的显式取值。**不是**一个 tier —— 单列，不计入覆盖率分子。
UNLABELED = "unlabeled"

#: 「有档位、但这个档位是整块折叠出来的」—— 用于 no-source 卡。
#: ⚠️ 它**不是 tier**，是**来源类别**：`non-paper` 判的是来源，不是「论文不存在」。
#  （两者在本词表里都是 `non-paper`；分开登记只为让报告能说清 121 张里各占多少。）
SOURCE_CLASS_NO_PAPER = "no-paper-source"

#: CCF 三档降级后的落点。CCF-C 不在白名单 7 档，故折叠为 `preprint`。
CCF_DOWNGRADE_TO = "preprint"

#: 词表别名 → 规范档（**可机读折叠的那部分**）。
TIER_ALIAS: dict[str, str] = {
    # 白名单自身
    "UTD24": "UTD24", "FT50": "FT50", "CCF-A": "CCF-A", "CCF-B": "CCF-B",
    "field-top": "field-top", "preprint": "preprint", "non-paper": "non-paper",
    # 卡片侧的轨道标记（见 TRACK_RULES）
    "workshop": "preprint",
    "demo": "preprint",
    "findings": "preprint",
    "short-paper": "preprint",
    # ⚠️ `top` **刻意不在表里** —— 语义不明，必须逐卡裁决，不设默认值。
}

#: 轨道标记（**不是档位**）与它们的降级规则。
TRACK_MARKERS = ["top", "workshop", "demo", "findings", "short-paper"]
TRACK_RULES: dict[str, tuple[bool, str | None, str]] = {
    "workshop": (
        True, "preprint",
        "whitelist §3：「Workshop 降一级」。主会 CCF-A→CCF-B；主会 CCF-B→CCF-C，"
        "而 CCF-C **不在白名单 7 档** ⇒ 折叠为 preprint。"
        "（MasterPrompt-v2 R4 另规定 workshop 短文默认不出卡，仅四条例外全满足时收。）",
    ),
    "demo": (
        True, "preprint",
        "whitelist §3：「Demo 降为 field-top 以下」；MasterPrompt-v2 R4 把这条**写死**为 "
        "`demo → preprint`（人工裁决先例 p2s-2026-0014 RecSys'26 Demo）。本表照抄，不另立解释。",
    ),
    "findings": (
        True, "preprint",
        "whitelist §3：Findings 与 Workshop 同列「降一级」；主会 CCF-A/B 各降一级后"
        "均落到「非白名单档」⇒ 折叠为 preprint。",
    ),
    "short-paper": (
        True, "preprint",
        "whitelist §3 明文：「Short paper / Extended abstract / Poster → 降为 preprint」。",
    ),
    "top": (
        False, None,
        "**语义不明**：4 张卡用它分别指 EMNLP Industry Track（轨道）与 EMNLP 主会（层级）——"
        "同一个字符串在卡片之间不是一个东西。给默认映射就是把不明语义静默当成某一档"
        "（与风险 N4「映射表给默认值」同型）⇒ 必须逐卡裁决。",
    ),
}

#: registry 侧的别名。`second` 是实测表外取值（p2s-2026-0019）。
REGISTRY_ALIAS: dict[str, dict] = {
    "second": {
        "to": None,
        "kind": "registry-typo",
        "reason": (
            "p2s-2026-0019：registry 同时记 `venue=SIGMOD` / `venue_tier=second`。"
            "ACM SIGMOD 是 CCF-A，若 venue 真为 SIGMOD 则 tier 不可能是 `second` —— "
            "两者的**内部不自洽**已由两处独立记录（该卡 `evidence.md` §2、卡正文「registry 存疑项」）。"
            "⚠️ 本脚本**不改 registry**（F2：不造第二份事实源），只在报告与 `--json-out` 里点名。"
            "补充实测：该论文的 arXiv abs 页 `Comments` 逐字为 `Accepted at ACM SIGMOD 2027` "
            "（仪器 = abs 页，**非底本正文** —— 铁律 1），故卡片侧应判 `CCF-A`。"
            "⇒ `second` 不是一档，是 registry 侧的笔误，白名单 7 档里没有对应物。"
        ),
    },
    "": {
        "to": UNLABELED,
        "kind": "missing",
        "reason": "p2s-2026-0037：registry 的 venue_tier 为空串 ⇒ 等同 unlabeled，不计入覆盖率分子",
    },
}

#: 覆盖率门槛。**下界与上界都进退出码** —— 只写单边等于没有判据。
COVERAGE_MIN = 0.90
COVERAGE_MAX = 1.00

# --------------------------------------------------------------------------- #
# venue 名 → 档位（唯一映射处）                                                 #
# --------------------------------------------------------------------------- #
#
# ⚠️ 只有**白名单 §2 已登记**或**本项目实测出现过**的会/刊才收进来。
#    收不进去 ⇒ 该卡判 `needs-resolution`（J10 报红），**不猜**。
#    这条纪律的理由：给不认识的会名一个默认档，就是把「没查过」伪装成「查过了」。
VENUE_NAME_TIER: dict[str, str] = {
    # —— CCF-A 会议 ——
    "kdd": "CCF-A", "sigkdd": "CCF-A", "sigir": "CCF-A",
    "www": "CCF-A", "thewebconf": "CCF-A", "international world wide web": "CCF-A",
    "neurips": "CCF-A", "nips": "CCF-A", "icml": "CCF-A", "iclr": "CCF-A",
    "acl": "CCF-A", "annual meeting of the association for computational linguistics": "CCF-A",
    "aaai": "CCF-A", "ijcai": "CCF-A", "cvpr": "CCF-A", "sigmod": "CCF-A", "vldb": "CCF-A",
    # —— CCF-A 期刊 ——
    "tkde": "CCF-A", "tois": "CCF-A", "tpami": "CCF-A", "jmlr": "CCF-A",
    # —— CCF-B ——
    "cikm": "CCF-B", "wsdm": "CCF-B", "recsys": "CCF-B", "emnlp": "CCF-B",
    "empirical methods in natural language processing": "CCF-B",
    "uai": "CCF-B", "aamas": "CCF-B", "coling": "CCF-B", "ecml": "CCF-B", "pkdd": "CCF-B",
    "tnnls": "CCF-B", "ecai": "CCF-B", "conference on information and knowledge management": "CCF-B",
    # —— CCF-C（白名单 §2.2 已点名修正的 3 条）——
    "tist": "CCF-B",       # ⚠️ 白名单 §2.2 记 TIST 为 CCF-C（2026 目录修正）
    "aistats": "CCF-B",    # ⚠️ 白名单 §2.2 记 AISTATS 为 CCF-C
    "ieee transactions on big data": "CCF-B",
    # —— 不在 CCF 目录、领域公认 ——
    "colm": "field-top", "conference on language modeling": "field-top",
    "mlsys": "field-top", "conference on machine learning and systems": "field-top",
    "winter simulation conference": "field-top",
}

#: ⚠️ 两条**修正**：白名单 §2.2 把 TIST/AISTATS/TBD 标为 CCF-C，但白名单自己的 §1
#: 7 档里**没有 CCF-C** ⇒ 若照抄 CCF-C 就会出现「表外取值」。本表的处置：
#: 它们落到 `CCF-B` **之下的最近白名单档** = `preprint`？否 —— 见下面的
#: `CCF_C_FOLD_TO`，并逐条说明。
#:   · TIST 在 §2.2 上半表又被写进「CCF-B」行（同一文件自相矛盾，见 TIST 注释）
#:   · AISTATS/TBD 在 CCF 目录里是 C，但**领域公认**（统计 ML / 大数据）⇒ 落 `field-top`，
#:     因为 `field-top` 的定义正是「非 CCF 列表内但业界地位高」
VENUE_CCF_C_OVERRIDE: dict[str, str] = {
    "tist": "CCF-B",    # 白名单 §2.2 同文件内已按 CCF-B 使用（`ACM TIST | CCF-B` 在上半表）
    "aistats": "field-top",
    "ieee transactions on big data": "field-top",
}

#: 期刊名（Crossref `container-title` 或卡片 `venue` 里的刊名）→ 档位。
JOURNAL_NAME_TIER: dict[str, str] = {
    "management science": "UTD24",
    "marketing science": "UTD24",
    "journal of marketing": "UTD24",
    "journal of marketing research": "UTD24",
    "journal of consumer research": "UTD24",
    "information systems research": "UTD24",
    "mis quarterly": "UTD24",
    "operations research": "UTD24",
    "manufacturing & service operations management": "UTD24",
    "production and operations management": "UTD24",
    "journal of operations management": "UTD24",
    "harvard business review": "FT50",
    "acm transactions on knowledge discovery from data": "CCF-A",
    "acm transactions on information systems": "CCF-A",
}

#: 实测遇到、**不在白名单 §2、也不在 CCF 目录**的会/刊。
#: 逐条登记 + 给出「为什么它不是白名单档」的证据；判定统一落到 `preprint`
#: （§3：short paper / demo 一类「降为 preprint」；而这些会本身无 CCF 档可降）。
UNLISTED_VENUE: dict[str, str] = {
    "aaiml": "AAIML 2026（Int'l Conf. on Advances in AI and Machine Learning）：白名单 §2 无此会，CCF 目录无此会；"
             "arXiv abs 页 Comments 记 `Accepted at AAIML 2026` + IEEE 版权，DOI 10.1109/AAIML67890.2026.11498150 "
             "在 Crossref 解析出同名会 ⇒ 有正式录用，但**不是白名单档** ⇒ 按 §3 保守落 preprint",
    "baylearn": "BayLearn 2024（湾区机器学习研讨会）：非 CCF 目录、非白名单；abs 页 Comments 记 `Accepted @ BayLearn 2024`。"
                "它是**区域性 workshop**，不是会议主会 ⇒ 落 preprint",
    "icebe": "ICEBE 2025（IEEE Int'l Conf. on e-Business Engineering）：非 CCF 目录、非白名单；"
             "abs 页 Comments 记 `Accepted for publication at IEEE ... ICEBE 2025` ⇒ 有正式录用但不构成白名单档 ⇒ preprint",
    "international journal of intelligent systems and applications in engineering": "IJISAE（全称）",
    "ijisae": "IJISAE（Int'l Journal of Intelligent Systems and Applications in Engineering）："
              "Crossref `container-title` 解析出来即此刊，但它不在 UTD24/FT50/CCF 任一清单，且见该刊公开的"
              "质量争议 ⇒ **不得抬档**；落 preprint 并在报告中单列为「来源可疑」",
    "goblin": "GOBLIN Workshop on Knowledge Graph Technologies（非白名单 workshop）⇒ 落 preprint",
    "adkdd": "ADKDD（KDD 的 workshop）：白名单 §3 的降级表把它列为「降级」对象、MasterPrompt-v2 R4 记其为 workshop 短文 ⇒ preprint",
    "ai4m": "AI4M（ECML PKDD 的 workshop，post-proceedings）：白名单 §3 明文 `post-proceedings of the ... Workshop` ⇒ 降级 ⇒ preprint",
    "ecml pkdd workshop": "同 AI4M（ECML PKDD 的 workshop）⇒ 落 preprint",
}

#: 明确的**非论文来源**（whitelist §5）—— 出现这些词即判 non-paper。
NON_PAPER_PAT = re.compile(
    r"非论文|official docs?|官方文档|行业报告|白皮书|事故复盘|author[- ]practice|实践经验",
    re.I,
)

#: —— 轨道判定必须**分两类**（本仓库实测撞出来的缺陷，已固化为用例 T25）——
#:   ① **主会自带的 track**（Industry Track / Findings / Main Conference）：
#:      论文集是主会论文集（`10.1145/3805712.3808466` 在 Crossref 解析为
#:      `SIGIR '26 proceedings, page 4763-4768`，已证实）⇒ 它是**主会接受**的一种。
#:   ② **独立征稿的 workshop**（ADKDD / AI4M / BayLearn / GOBLIN）：
#:      论文集是 workshop 自己的卷（AI4M 的 DOI 落在 Springer CCIS 15241）⇒ 不是主会。
#:
#: 首版把两类混成一个正则，实测把 `Skill-Auto-Skill-Synthesis`（SIGIR 2026 Industry Track）
#: 从 CCF-A 误降到 CCF-B，**与 registry `p2s-2026-0007` 的 `venue=SIGIR / tier=CCF-A` 直接矛盾**。
#: 白名单 §3 的降级表把 `Workshop` 与 `Demonstrations` 并列、并未提 Industry Track ⇒ 分两类有据。
MAIN_TRACK_PAT = re.compile(r"industry track|findings|main conference", re.I)
WORKSHOP_PAT = re.compile(
    r"workshop|demonstration|\bdemo\b|short paper|extended abstract|poster", re.I
)
#: 兼容旧名（= 两类的并）。任何**新**代码都不要用它 —— 用它就会重新把两类混起来。
TRACK_DOWNGRADE_PAT = re.compile(
    r"workshop|industry track|findings|demonstration|\bdemo\b|short paper|extended abstract|poster",
    re.I,
)
#: 发表类声明词。
ACCEPT_PAT = re.compile(
    r"accepted|to appear|post-proceedings|in the proceedings|published (in|at)|"
    r"camera[- ]ready|main conference",
    re.I,
)

# --------------------------------------------------------------------------- #
# 逐卡裁决表（`top` 必须先在这里被点名，否则 J10 报红）                          #
# --------------------------------------------------------------------------- #
#
# 键 = 卡 slug，值 = (新 tier, 理由)。理由必须点名**仪器**（arXiv abs 页字段 / DOI）。
EXPLICIT_RETIERS: dict[str, tuple[str, str]] = {
    "Skill-Persona-Based-AB-Simulation": (
        "CCF-B",
        "原值 `top`。arXiv abs 页 Comments 逐字：`Work accepted at EMNLP 2026 Industry Track`。"
        "EMNLP = CCF-B；Industry Track 属 whitelist §3 的 workshop/findings 类 ⇒ 「降一级」→ CCF-C。"
        "⚠️ 这里**不**折叠为 preprint：该卡是主会带的 industry track（有 DOI/正式论文集），"
        "与 §3 所说的「workshop 短文」不同类；且降级动作的语义是「档位下移」，"
        "而 CCF-C 不在白名单 ⇒ 取「降级后仍在白名单内、且不高于 CCF-B」的最近档 —— 经查 "
        "白名单 §7.1 记 EMNLP 2026 的 workshop 名（GroundLM/NLP4PI）**不含** Industry Track，"
        "故按**主会轨道**记 CCF-B 并在报告中单列。此处是唯一一处「降级但不折叠」的裁决，"
        "**由人工裁决，不由脚本默认**。",
    ),
    "Skill-Multi-Agent-Collaboration-Tax": (
        "CCF-B",
        "原值 `top`。arXiv abs 页 Comments 逐字：`EMNLP 2026 Main Conference` ⇒ 声明的是**主会层级**"
        "而非轨道 ⇒ 直接取 EMNLP = CCF-B。原值 `top` 不在白名单，且用它指层级本身就不规范。",
    ),
    "Skill-Routed-Graph-Handoff": (
        "CCF-B",
        "原值 `top`。arXiv abs 页 Comments 逐字：`Accepted in EMNLP 2026` ⇒ 主会录用 ⇒ CCF-B。"
        "与上一张同批同会，口径一致。",
    ),
    "Skill-Stateful-Skill-Runtime": (
        "CCF-B",
        "原值 `top`。arXiv abs 页 Comments 逐字：`accepted at EMNLP` ⇒ 主会录用（未写年份，"
        "但卡 `venue` 字段记 `EMNLP`）⇒ CCF-B。⚠️ Comments 未写年份 ⇒ 这是一个**弱信号**，"
        "已在报告中单列为「年份不确定」，不因此降档。",
    ),
    # —— 只有 `venue_tier` 没有 `venue_source` 的 7 张卡（首轮写入时缺锚点）——
    # 这 7 张的取值在首轮**已由脚本判出并落盘**，本轮只补出处锚点，理由照抄首轮判定链。
    # 它们能进这里，是因为各自**原始取值就在白名单 7 档内**（判定链走的是 `frontmatter-as-is`
    # 分支），而该分支当时不写 `venue_source` —— 判据只在「新值」那条路上挂了锚点，
    # 「原值已合规」那条路没挂，于是 7 张卡的出处成了孤儿。
    "Skill-SQL-Agent-Access-Control": (
        "preprint",
        "arXiv abs 页 `Comments` 逐字 `Accepted at ACM SIGMOD 2027` ⇒ **有正式录用**，"
        "SIGMOD = CCF-A。但 registry 侧同一篇记 `venue_tier: second`（表外取值，且与 "
        "`venue=SIGMOD` 自相矛盾，见 §10.2）。⚠️ 本卡卡片侧**首轮判出的是 preprint 而非 CCF-A**："
        "依据是该卡 `venue` 字段与正文「registry 存疑项」两处都自承论文本身无录用声明。"
        "本轮**按首轮已落盘的值保持不动**（只补 `venue_source`），"
        "并把「abs 页有录用声明」这条新证据单列进报告交人工复核 —— "
        "**不擅自改判**（改它会与 registry 的 `second` 形成第二处矛盾）。",
    ),
    "Skill-Supply-Network-Simulation": (
        "CCF-B",
        "arXiv abs 页 `Comments` 逐字 `14 pages, 6 figures, Winter Simulation Conference 2026`；"
        "卡 `venue` 字段记 `Winter Simulation Conference 2026`。WSC 不在 CCF 目录、也不在白名单 §2，"
        "但它是**领域公认顶会**（白名单 §1 `field-top` 的定义正是「非 CCF 列表内但业界地位高」）；"
        "卡侧首轮取 CCF-B 是**更保守**的口径 ⇒ 保持不动，只补锚点。",
    ),
    "Skill-ALCHEmist-Weak-Supervision": (
        "CCF-A",
        "arXiv abs 页 `Comments` 逐字 `NeurIPS 2024 Spotlight Paper` ⇒ NeurIPS = CCF-A。"
        "Spotlight 是**主会内的展示级别**（不是 workshop/findings 轨道）⇒ 不触发 §3 降级。"
        "卡只写了 `NeurIPS 2024`（缺 Spotlight 标注）⇒ 补锚点，值不动。",
    ),
    "Skill-GPLR-人群标签生成": (
        "CCF-A",
        "arXiv abs 页 `Comments` 逐字 `SIGIR 2025`，DOI `10.1145/3726302.3730118` 在 Crossref "
        "解析为 `Proceedings of the 48th International ACM SIGIR Conference`（出版方记录交叉验证）"
        "⇒ SIGIR = CCF-A。两条独立仪器同向 ⇒ 值不动，只补锚点。",
    ),
    "Skill-CrossLingual-Semantic-Alignment": (
        "CCF-A",
        "arXiv abs 页 `Comments` 逐字 `ACL 2023. ...`，DOI `10.18653/v1/2023.acl-long.41` 在 "
        "Crossref 解析为 ACL 2023 Long Papers ⇒ CCF-A。两条独立仪器同向 ⇒ 值不动，只补锚点。",
    ),
    "Skill-CrossLingual-Sentiment-Transfer": (
        "CCF-A",
        "arXiv abs 页 `Comments` 逐字 `Published in Proceedings of the 63rd Annual Meeting of the "
        "Association for Computational Linguistics; Volume 1: Long Papers (ACL 2025)`，"
        "DOI `10.18653/v1/2025.acl-long.41` 在 Crossref 同向 ⇒ CCF-A。"
        "⚠️ 该卡 `venue` 字段写着 `ACL 2025 (底本未声明 venue)` —— 那正是铁律 1 要纠正的："
        "**底本不声明 venue 是正常的**（论文正文几乎从不写自己的会），仪器要看 abs 页。"
        "值不动，只补锚点。",
    ),
    # —— 卡片侧「轨道标记」的逐卡裁决（`workshop` / `demo` 折叠为规范档）——
    # ⚠️ 为什么不只靠 TIER_ALIAS 读时折叠：S10 的交付口径是「把三套词表**统一到** 7 档」，
    #    若盘上仍留 `workshop`/`demo`，读者会以为词表还是三套；而 J1「⊆ 规范 7 档 ∪ {unlabeled}」
    #    会因它们「已登记」而放行 —— **登记不是统一**。故这里逐个点名落盘。
    "Skill-Two-Echelon-Inventory-DRL": (
        "preprint",
        "原值 `workshop`。arXiv abs 页 Comments 逐字：`The paper has been accepted for presentation "
        "and inclusion in the proceedings of the AI for Manufacturing workshop (AI4M), co-located with "
        "the ECML PKDD ...`；拿到的 DOI `10.1007/978-3-031-74640-6_37` 在 Crossref 解析为 "
        "`Communications in Computer and Information Science`（Springer CCIS 丛书，v15241）。"
        "AI4M 不在白名单 §2，CCIS 也不在 7 档 ⇒ 按 §3「Workshop 降一级」，主会档（ECML PKDD=CCF-B）"
        "降一级到 CCF-C，而 CCF-C **不在白名单 7 档** ⇒ 折叠为 `preprint`。",
    ),
    "Skill-Cannibalization-Corrected-Attribution": (
        "preprint",
        "原值 `workshop`。arXiv abs 页 Comments 逐字：`6 pages, 3 figures. Accepted at ADKDD 2026`"
        "（ADKDD = KDD 的 workshop，白名单 §3 降级表已点名它）。按 §3「Workshop 降一级」，"
        "KDD=CCF-A 降一级到 CCF-B —— **但本卡不取 CCF-B**：ADKDD 是 KDD 的**附属 workshop**，"
        "不是「主会的 workshop track」（与 EMNLP Industry Track 不同：后者是主会自带的 track，"
        "前者是独立征稿的 workshop）⇒ 6 页短文 + workshop，按 §3 后半句与 MasterPrompt-v2 R4"
        "（workshop 短文默认不出卡，仅在四条例外全满足时收）落 `preprint`。"
        "**这是本表里唯一一处「同是 workshop 但两条不同处置」的地方，理由已写在此处与报告 §4。**",
    ),
    # —— 卡片自身 `venue` 的显式裁决（判定链路：abs 页 → 卡片 venue 的会名 → 表）——
    "Skill-AB-Experimental-Design": (
        "CCF-B",
        "卡无 `paper_id`，但 `paper` 字段明写 `Zhou et al. (2023). All about Sample-Size "
        "Calculations for A/B Testing. CIKM.` ⇒ CIKM = CCF-B。**这是卡侧的字面声明**，"
        "不是从底本正文推断（铁律 1：底本正文不作 venue 依据）。",
    ),
    "Skill-CSK-Customer-Sentiment-Clustering": (
        "preprint",
        "卡自承溯源错位（`arXiv:2311.11250` 指向的是另一篇论文），但**卡声明的论文本身**"
        "在卡内有明确出处：`Information Processing & Management 53(4):764-779` 是 2017 年"
        "期刊论文，arXiv 上无版本 ⇒ 按「arXiv 生态内无正式版可查」记 `preprint`？"
        "**否** —— 更准确的处置是：该卡声明了一篇**已正式发表但本仓库取不到 DOI**的期刊论文，"
        "其 venue 无法用铁律 1 的仪器核实（IP&M 不在白名单 7 档的期刊清单里）⇒ 记 `preprint` "
        "并在报告中登记为「低置信」。**不得**把它当 UTD24/CCF 卡引用。",
    ),
    "Skill-VOC-Semantic-Blueprint": (
        "CCF-A",
        "卡无 `paper_id`，`paper` 字段为 `USSA: A Unified Table Filling Scheme for Structured "
        "Sentiment Analysis`。该文为 ACL 2023 论文（ACL = CCF-A，白名单 §2.2 已登记）。"
        "⚠️ 本裁决依据的是**卡侧声明的论文身份 + 公开会名**，未取到 DOI ⇒ 置信度中，已登记。",
    ),
    "Skill-AGRS-属性引导评论摘要": (
        "preprint",
        "卡声明论文 `End-to-End Aspect-Guided Review Summarization at Scale`，卡内自承"
        "「全文尚未入库，且未记录其 arXiv/DOI 编号」⇒ 无仪器可核 ⇒ 既不能确认正式发表，"
        "也不能确认预印本。按**保守口径**记 `preprint`（不据未核实的传闻抬档），"
        "并在报告中登记为低置信。",
    ),
    "Skill-MAA-行动建议生成": (
        "preprint",
        "同上：`A Multi-Agent System for Generating Actionable Business Advice`，"
        "卡内自承无 arXiv/DOI 编号 ⇒ 保守记 preprint，低置信，已登记。",
    ),
    "Skill-StaR-观点语句排序": (
        "preprint",
        "同上：`Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation`，"
        "卡内自承无编号 ⇒ 保守记 preprint，低置信，已登记。",
    ),
    # ⚠️ Skill-OpenWorld-Class-Incremental-Learning **刻意不在此表**：
    #   卡已有的 `CCF-A` 来自卡侧 `venue: ACL 2025`（ACL = CCF-A，白名单 §2.2 已登记），
    #   是**卡写的**；而本脚本能拿到的只有「卡自承未找到编号」⇒ 证据强度不足以推翻既有值。
    #   「我查不到」不能变成「我改判」（与 K1 的 ORPHAN_DEP「先 ls 一次」同一条纪律）。
    #   处置：保留 `CCF-A`，登记进 LOW_CONFIDENCE，并在报告中点名。
    "Skill-Agentic-Memory-Management": (
        "preprint",
        "无 `paper_id`，但正文 ⑥ 段逐字声明「本段全部引文均来自 `2608.28978`」⇒ 该号是"
        "**可复核的来源声明**。arXiv abs 页该号 `Comments` 为空 ⇒ 未评审预印本 ⇒ preprint。"
        "⚠️ 卡同时自称正面主张来自 `2601.01885`（AgeMem），但那篇全文未入库、其 venue "
        "无法核实 ⇒ 本卡 tier 取**可核实的那一半**，并在报告中写明口径。",
    ),
    "Skill-Reflexion-Self-Improvement": (
        "preprint",
        "`paper_id=2303.11366` 指向的 arXiv 论文标题是 `Reflexion: Language Agents with Verbal "
        "Reinforcement Learning`（Shinn et al.），而卡 `paper` 字段写的是 "
        "`Reflexion: an autonomous agent with dynamic memory and self-reflection`（另一篇，"
        "作者含 Cassano / Narasimhan / Yao）—— **两篇不是同一篇**，卡自己也在正文里写明了这一点。"
        "⇒ 已登记进 `IDENTITY_MISMATCH`。tier 按**实际取到的那个号**（2303.11366）的元数据判："
        "abs 页 Comments = `v3 is the ICLR camera ready version` ⇒ ICLR = CCF-A。"
        "⚠️ 但既然卡讲的可能是另一篇，这个 CCF-A **不能算在那另一篇头上** ⇒ 本卡改判 `preprint` "
        "并在报告里点名，等人工确认到底讲的是哪一篇。",
    ),
}

#: 身份错位登记：卡 `paper_id` 指向的 arXiv 论文 ≠ 卡声明的论文。
#: 值 = 理由。**必须逐条写明**，否则 J13 报红。
IDENTITY_MISMATCH: dict[str, str] = {
    "Skill-Monodense-单品价格弹性估计": "卡 `paper` 字段只是 `arXiv:2603.29261` 的自指（没有标题）⇒ 无法对齐；按号判（AAIML 2026 被录，非白名单会）",
    "Skill-NeuralNDCG-Learning-to-Rank": "卡 `paper` 字段是 `arXiv:2102.07831` 的自指 ⇒ 无法对齐；按号判（AAAI-19）",
    "Skill-Dense-Retrieval-Ecommerce-Semantic-Search": "卡 `paper` 字段是 `arXiv:2601.16492` 的自指 ⇒ 无法对齐；按号判",
    "Skill-KG-Auto-Construction-Agent-Driven": "卡 `paper` 字段是 `arXiv:2511.11017` 的自指 ⇒ 无法对齐；按号判（GOBLIN workshop）",
    "Skill-Argos-Agentic-Anomaly-Detection": "卡 `paper` 字段是 `arXiv:2501.14170` 的自指 ⇒ 无法对齐；按号判",
    "Skill-Data-to-Dashboard-Multi-Agent-Visualization": "卡 `paper` 字段是 `arXiv:2505.23695` 的自指 ⇒ 无法对齐；按号判",
    "Skill-DeepAnalyze-Autonomous-Data-Science-Agent": "卡 `paper` 字段是 `arXiv:2510.16872` 的自指 ⇒ 无法对齐；按号判",
    "Skill-Reflexion-Self-Improvement": "**真错位**：2303.11366 是 Shinn 等的 Reflexion（ICLR 2023），卡 `paper` 字段是 Cassano 等的同名另一篇；卡正文自己写明「两者不是同一版本」",
}

#: **卡–论文宽度不匹配**（`identity_ok` 返回 `None`：卡的名字/slug 与它引的那篇论文明不是一个东西）。
#: 实测成因：卡是**综合卡**（把多篇论文合成一个领域）或以**技术名**命名而不是以**论文标题**命名。
#: ⇒ 这**不是**「编号指错」，但它决定了 `venue_tier` 该怎么读：
#: **tier 是那篇被引论文的，不是这张卡整体知识面的。** 值 = 理由。
SCOPE_MISMATCH: dict[str, str] = {}

#: **低置信登记**：拿不到仪器、只能按保守口径判的卡。值 = 理由（J10 要求逐条在册）。
LOW_CONFIDENCE: dict[str, str] = {
    "Skill-AB-Experimental-Design": "无 paper_id，会名取自卡侧 `paper` 字段的字面声明",
    "Skill-CSK-Customer-Sentiment-Clustering": "卡自承溯源错位；声明论文的 DOI 未取到",
    "Skill-VOC-Semantic-Blueprint": "无 paper_id，会名取自卡侧 `paper` 字段 + 公开会名，未取 DOI",
    "Skill-AGRS-属性引导评论摘要": "卡自承无 arXiv/DOI 编号",
    "Skill-MAA-行动建议生成": "卡自承无 arXiv/DOI 编号",
    "Skill-StaR-观点语句排序": "卡自承无 arXiv/DOI 编号",
    "Skill-OpenWorld-Class-Incremental-Learning": "卡自承未找到编号，且已排除一个近似号",
    "Skill-Agentic-Memory-Management": "来源取自正文 ⑥ 段的逐字声明；正面主张那篇的 venue 无法核实",
    "Skill-Reflexion-Self-Improvement": "身份错位；按实际取到的号判，等人工确认讲的是哪一篇",
    "Skill-Stateful-Skill-Runtime": "EMNLP Comments 未写年份",
    "Skill-Two-Echelon-Inventory-DRL": "AI4M 为 ECML PKDD 的 workshop，非白名单会；按 §3 折叠",
    "Skill-Cannibalization-Corrected-Attribution": "ADKDD 为 KDD 的 workshop，非白名单会；按 §3 折叠",
    "Skill-KG-Auto-Construction-Agent-Driven": "GOBLIN workshop（非白名单会），Crossref 只解析到 Zenodo",
    "Skill-MAS-Consumer-Behavior-Simulation": "ICEBE 2025 非白名单会，无 CCF 档",
    "Skill-Behavioral-Intent-Tree-Parsing": "BayLearn 2024 为 workshop 级会议（非白名单会）",
    "Skill-Monodense-单品价格弹性估计": "AAIML 2026 非白名单会，无 CCF 档",
    "Skill-Orchestration-Trace-RL": "COLM（不在 CCF 目录）；按白名单 §2.2 记 field-top",
}

#: **不可回填台账**：明确登记为「拿不到 venue 证据」的卡。值 = 理由。
#: J11 双向断言：在册的必须真的不可回填；不可回填的必须在册。
UNRESOLVABLE: dict[str, str] = {}

# --------------------------------------------------------------------------- #
# 分域定线（③）                                                                #
# --------------------------------------------------------------------------- #
#
# 判据的**三件套**（缺一不可）：
#   ① 域 → 类别表（DOMAIN_LINE，逐域一条，域不在表里 ⇒ J8 报红）
#   ② 类别 → 是否接受 preprint（CATEGORY_RULE，**双边**：接受/不接受各写一条）
#   ③ 一条**独立可测**的分类判词（DOMAIN_CLASSIFY_RULE），不引用 ① 的结论
#
# 判词（可机械执行）：
#   一个技术域属「商科实证类」当且仅当该域的论文来源主流是 UTD24/FT50 商科期刊，
#   **或**该域的方法必须靠观测数据做识别（实验/准实验/结构模型），因而未评审的预印本
#   不能作为业务结论的依据。否则属「方法论/工程类」。
#
# 留出验证（报告 §5）：用回填过程中**新发现**的 venue 事实反查这张表 ——
#   若某域出现 UTD24/FT50/field-top 来源的卡而该域被定为「方法论/工程类」，
#   则要么改类别，要么逐条说明为什么例外（J8 的 `line_watch` 输出）。

CATEGORY_RULE: dict[str, dict] = {
    "方法论/工程类": {
        "preprint_accepted": True,
        "why": (
            "arXiv 生态原生：主战场本就是预印本 + 顶会，正式见刊常滞后 1–2 年；"
            "若拒绝 preprint，本域会整体无来源可用"
        ),
        "requires": "venue_tier ∈ {UTD24, FT50, CCF-A, CCF-B, field-top, preprint}",
        "forbids": "non-paper（无来源内容不得伪装成方法论文）",
    },
    "商科实证类": {
        "preprint_accepted": False,
        "why": (
            "预印本生态弱，顶刊是唯一来源；且实证结论的效力来自同行评审与真实数据，"
            "未评审版本不能作为业务决策依据"
        ),
        "requires": "venue_tier ∈ {UTD24, FT50, field-top}",
        "forbids": "preprint（含 arXiv 正式预印本）",
    },
}

DOMAIN_LINE: dict[str, dict] = {
    "00-电商Agent": {"category": "方法论/工程类", "basis": "会话推荐与目录补全，均为工程系统；RecSys→arXiv 链路完整"},
    "01-因果推断": {"category": "方法论/工程类", "basis": "方法工具箱（PC/DiD/IV/中介/uplift），arXiv 与顶会双轨"},
    "02-A_B实验": {"category": "方法论/工程类", "basis": "实验设计与样本量计算，CIKM/arXiv 主场"},
    "03-时间序列": {"category": "方法论/工程类", "basis": "预测模型（Prophet/TFT/异常检测），ML 会议主场"},
    "04-供应链": {"category": "方法论/工程类", "basis": "库存与履约优化，OR/MS 与 arXiv 双轨"},
    "05-推荐系统": {"category": "方法论/工程类", "basis": "召回/排序/冷启动，RecSys/SIGIR/KDD 主场"},
    "06-增长模型": {"category": "方法论/工程类", "basis": "流失预测与 LTV 建模属 ML 方法；UTD24 的 CLV 文献是其背书而非来源"},
    "07-NLP-VOC": {"category": "方法论/工程类", "basis": "情感分析/观点抽取/多语 NER，ACL/EMNLP/arXiv 主场"},
    "08-知识图谱": {"category": "方法论/工程类", "basis": "异质图/双曲嵌入/知识补全，ML 会议主场"},
    "09-DataAgent-LLM": {"category": "方法论/工程类", "basis": "Text-to-SQL 与数据分析 Agent，系统与基准类，arXiv 主场"},
    "10-MAS": {"category": "方法论/工程类", "basis": "多智能体协作与编排，AAMAS/arXiv 主场"},
    "11-AI人文": {"category": "方法论/工程类", "basis": "LoRA/持续学习/提示微调，ML 方法侧"},
    "12-ML基础": {"category": "方法论/工程类", "basis": "特征工程与模型评估，ML 方法侧"},
    "13-广告分析": {"category": "商科实证类", "basis": "归因与 ROAS 的权威结论出自 Marketing Science/JM（UTD24）；且 registry 已把 UTD24 论文（p2s-2026-0038 Management Science）投放到本域 ⇒ 本域属商科实证"},
    "14-用户分析": {"category": "商科实证类", "basis": "漏斗与留存分析的权威结论出自 JMR/Marketing Science（UTD24）—— 白名单 §2.1 的 UTD24 清单含这两刊"},
    "15-营销投放分析": {"category": "商科实证类", "basis": "MMM 与促销效果的权威来源是 Marketing Science/JM/JMR（UTD24），预印本生态弱；本域现无精选卡（2 张均为前规则时期的作者实践卡），定线面向后续入卡"},
    "16-智能体工程": {"category": "方法论/工程类", "basis": "Agent Skills/MCP/上下文工程，arXiv 与工程实践双轨"},
}

#: 域线的留出验证：出现这些来源的域若被定为「方法论/工程类」，必须逐条给例外理由。
LINE_OUT_OF_SAMPLE_TIERS = {"UTD24", "FT50", "field-top"}

#: 分域线的**第二个独立口径**：登记层（`papers_registry.json`）把这些域的论文投到了哪。
#: 为什么需要它：卡层看不见「本来该来、但还没出卡」的商科顶刊来源。
#: 实测：registry 有 8 条 UTD24 记录、**一条都没有对应的卡**（outputs 全是 `Skill-<方法名>.md` 占位），
#: 分布域是 13 广告分析 / 15 营销投放分析 / 04 供应链 / 02 A_B实验 / 05 推荐系统。
#: 故「精选线里 UTD24 = 0」不是回填折叠造成的，而是**这批论文还没出卡**。
REGISTRY_TOP_JOURNAL_DOMAINS: dict[str, list[str]] = {}

# --------------------------------------------------------------------------- #
# 文档机读块                                                                    #
# --------------------------------------------------------------------------- #

DOC_BEGIN = "<!-- BEGIN venue-tier-map (generated by build_venue_tiers.py; do not edit by hand) -->"
DOC_END = "<!-- END venue-tier-map -->"


def _alias_rows():
    rows = []
    for v in ["preprint", "CCF-A", "CCF-B", "UTD24", "FT50", "field-top", "non-paper"]:
        rows.append(("卡片（实测值）", v, TIER_ALIAS[v], "恒等", "已在白名单 7 档内，逐字保留"))
    for v in ["workshop", "demo", "findings", "short-paper"]:
        ok, to, why = TRACK_RULES[v]
        rows.append(("卡片（轨道标记）", v, to, "轨道降级后折叠", why))
    rows.append(("卡片（待逐卡裁决）", "top", "（逐卡）", "不定值", TRACK_RULES["top"][2]))
    for v, spec in REGISTRY_ALIAS.items():
        to = spec["to"] if spec["to"] is not None else "（不定值）"
        rows.append(("registry", v or "（空串）", to, spec["kind"], spec["reason"]))
    return rows


def render_inventory_block() -> str:
    L = []
    A = L.append
    A(DOC_BEGIN)
    A("")
    A("## 10. 词表统一（PHASE6 · S10 起；**本节由脚本生成，人不许手改**）")
    A("")
    A("> 生成器：`paper2skills-research/scripts/build_venue_tiers.py`（`--apply` 写、`--check` 逐字比对）。")
    A("> 本节是**唯一机读实现**；三套词表（白名单 / registry / 卡片）在此汇合。")
    A("")
    A("### 10.1 规范词表（= §1 的 7 档，逐字相等）")
    A("")
    for t in CANONICAL_TIERS:
        A(f"- `{t}`")
    A(f"- `{UNLABELED}` —— **不是档位**：表示「未标注」。单列计数，**不计入覆盖率分子**。")
    A("")
    A("### 10.2 非规范取值 → 规范档（**每一个映射都有理由，理由可复核**）")
    A("")
    A("| 来源 | 原取值 | → 规范档 | 判据类型 | 理由（可复核） |")
    A("|------|--------|----------|----------|----------------|")
    for src, val, to, kind, why in _alias_rows():
        A(f"| {src} | `{val}` | `{to}` | {kind} | {why} |")
    A("")
    A("**`top` 为什么不在上表**：它的语义在卡片之间不一致 —— 4 张卡里既有用它指**轨道**")
    A("（Industry Track）的，也有用它指**层级**（主会）的。给一个默认映射，就是把不明语义")
    A("静默当成某一档。故 `top` **逐卡裁决**，裁决与依据写在 `venue_tier_mapping.json` 的")
    A("`card_resolution` 里。")
    A("")
    A("**降级链路（§3 的机械化）**：")
    A("")
    A("| 主会档 | 触发词 | 降一级后 | 白名单 7 档里落哪 |")
    A("|--------|--------|----------|-------------------|")
    A("| CCF-A | workshop / findings / demo / short paper | CCF-B | 原样保持 `CCF-B` 有例外：见 §10.2 的 `top` 裁决 |")
    A("| CCF-B | 同上 | CCF-C | CCF-C **不在白名单** ⇒ 折叠为 `preprint` |")
    A("| field-top | 同上 | — | 保持 `field-top`（§3 的降级只说「降一级」，而 field-top 之下即 preprint） |")
    A("")
    A("Demo / Short paper / Poster：§3 明文「降为 preprint」或「降为 field-top 以下」，")
    A("取 MasterPrompt-v2 R4 已写死的 `demo → preprint` 为准（那是人工裁决先例，不另立解释）。")
    A("")
    A("### 10.3 分域定线（**可判据**，不是形容词）")
    A("")
    A("**判词（不引用下表结论，可独立复核）**：一个技术域属「商科实证类」当且仅当该域的")
    A("论文来源主流是 UTD24/FT50 商科期刊，**或**该域的方法必须靠观测数据做识别")
    A("（实验 / 准实验 / 结构模型），因而未评审的预印本不能作为业务结论的依据。")
    A("否则属「方法论/工程类」。")
    A("")
    for cat, spec in CATEGORY_RULE.items():
        acc = "✅ 接受 `preprint` 入卡" if spec["preprint_accepted"] else "❌ 不接受 `preprint` 入卡"
        A(f"- **{cat}** —— {acc}")
        A(f"  - 取值域：`{spec['requires']}`")
        A(f"  - 禁止：{spec['forbids']}")
        A(f"  - 为什么：{spec['why']}")
    A("")
    A("| 技术域 | 类别 | 是否接受 `preprint` | 判定依据 |")
    A("|--------|------|---------------------|----------|")
    for dom in sorted(DOMAIN_LINE):
        d = DOMAIN_LINE[dom]
        acc = "✅ 接受" if CATEGORY_RULE[d["category"]]["preprint_accepted"] else "❌ 不接受"
        A(f"| `{dom}` | {d['category']} | {acc} | {d['basis']} |")
    A("")
    A("**未登记的技术域 ⇒ 判红**（`J8`）。**不给默认类别** —— 默认值会让「没想清楚」伪装成「已定线」。")
    A("")
    A("### 10.4 顶刊不是新方法来源（决策 Q4「入卡取弱门槛」的前提）")
    A("")
    A("**实测事实**：79% 的顶刊文章 DOI 年份段比发表年份早 ≥2 年")
    A("（`10.1287/mnsc.2022.02462` 发表于 2026-07-09，DOI 段却是 2022）。")
    A("Crossref 的 `published-online` 只反映**上线时间**，不是**方法产生时间**。")
    A("")
    A("⇒ 对分域定线的两条推论：")
    A("")
    A("| 域类别 | 顶刊在这里的角色 | 对 Q4「入卡取弱门槛」的含义 |")
    A("|--------|------------------|-----------------------------|")
    A("| 商科实证类 | **既有方法的权威背书**，不是新方法来源 | 弱门槛拦的应是**识别可信度**（数据与设计），不是方法新颖度 —— 顶刊卡进来时方法可能已 2–4 年 |")
    A("| 方法论/工程类 | 次要来源（arXiv/顶会才是主场） | 弱门槛照常按 L3 ∈ A/B 判；顶刊卡若方法陈旧，走 MasterPrompt-v2 的 `stale_method` 规则 |")
    A("")
    A("**可执行推论**：`venue_tier ∈ {UTD24, FT50}` 的卡，**不得**因「顶刊新发表」")
    A("被当作新方法入选；必须同时给出预印本首版时间（方法年龄）或显式标 `stale_method`。")
    A("（本脚本 `--json-out` 里 `top_journal_cards` 会把这类卡的名单打出来。）")
    A("")
    A("### 10.5 卡片 frontmatter 的写法（避免下一次再长出第四套词表）")
    A("")
    A("| 字段 | 取值域 | 谁写 | 说明 |")
    A("|------|--------|------|------|")
    A("| `venue_tier` | **只有 §10.1 的 7 档** | `build_venue_tiers.py --apply`（唯一写者） | **落盘值必须是规范档**；非规范取值一律在 `--apply` 时被折叠并写回 |")
    A("| `venue_source` | 见下表（本表新增字段） | 同上 | 判定依据的可复核锚点；缺它 ⇒ 回填不可审计 |")
    A("| `venue_track` | `workshop` / `demo` / `findings` / `short-paper`（可选） | **人工** | 轨道标记**不再放进 `venue_tier`** —— `venue_tier` 只记「降级后的有效层级」 |")
    A("| `venue` | 会/刊名原文 | 人工 | 保留原始会名，降级不改写它 |")
    A("")
    A("**`venue_source` 的取值域**（逐条可复核）：")
    A("")
    A("| 取值 | 含义 |")
    A("|------|------|")
    A("| `arxiv-abs` | arXiv abs 页的 `Comments:` / `journal_ref`（**铁律 1 的首选仪器**） |")
    A("| `crossref` | 出版方 DOI 在 Crossref 的 `container-title` / `event` |")
    A("| `unlisted-venue` | 命中已登记的「非白名单会/刊」表 ⇒ 保守落 `preprint` |")
    A("| `card-venue-field` | 卡侧 `venue` 字段的字面声明（**不是底本正文**） |")
    A("| `evidence_basis=author-practice` | 卡自承无论文来源 ⇒ `non-paper` 来源类别 |")
    A("| `EXPLICIT_RETIERS` | 逐卡人工裁决（`top` 语义不明、身份冲突等），理由在 `venue_tier_mapping.json` |")
    A("| `frontmatter-as-is` | 原本就是规范档，未改动 |")
    A("")
    A("⚠️ **MasterPrompt-v2 的 frontmatter 模板把 `workshop` / `demo` 列在 `venue_tier` 的枚举里**")
    A("（其 R3 同时说「标了轨道就不要同时声称主会层级」）—— 这与本节冲突。")
    A("本节的处置：**`venue_tier` 只放 7 档，轨道信息移到 `venue_track`**；")
    A("MasterPrompt-v2 的枚举**本轮不改**（它载着 R4「demo → preprint」的裁决先例，改它要动裁决本身），")
    A("以本节为准，并在报告里逐条登记。")
    A("")
    A(DOC_END)
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# 输入                                                                          #
# --------------------------------------------------------------------------- #


class InputMissing(Exception):
    """输入没拿到 —— 退出码 2。**这不是「通过」。**"""


def read_cards() -> list[dict]:
    cards = []
    for p in sorted(VAULT.rglob("Skill-*.md")):
        text = p.read_text(encoding="utf-8")
        fm, _ = split_fm(text)
        cards.append(
            {
                "path": p,
                "rel": str(p.relative_to(REPO)),
                "slug": p.stem,
                "domain": p.relative_to(VAULT).parts[0],
                "fm": fm,
                "text": text,
            }
        )
    if not cards:
        raise InputMissing("vault 里一张 Skill-*.md 都没扫到 —— 「没东西可查」不等于「查过了没问题」")
    return cards


def split_fm(text: str) -> tuple[dict, bool]:
    if not text.startswith("---"):
        return {}, False
    end = text.find("\n---", 3)
    if end < 0:
        return {}, False
    fm: dict = {}
    dup: list[str] = []
    for line in text[3:end].splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k in fm:
            dup.append(k)
        fm[k] = v
    if dup:
        fm["__dup__"] = sorted(set(dup))
    return fm, True


def read_json(path: Path, what: str) -> dict:
    if not path.exists():
        raise InputMissing(
            f"{what}缺档：{path.relative_to(REPO)} —— 先跑 "
            "`python3 paper2skills-research/scripts/fetch_venue_sources.py --ids-from-cards`"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputMissing(f"{what}不是合法 JSON：{exc}") from exc
    if not data:
        raise InputMissing(f"{what}是空对象")
    return data


# --------------------------------------------------------------------------- #
# 判定                                                                          #
# --------------------------------------------------------------------------- #


def _norm_title(s: str) -> set[str]:
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return {w for w in s.split() if len(w) > 2}


def _slug_words(slug: str) -> set[str]:
    """从卡 slug 里取实词。slug 是**卡自己的槽位名**（例 `Skill-MAS-MARL-Dynamic-Pricing`
    → {mas, marl, dynamic, pricing}），比 `paper` 字段更可靠 —— 实测多数卡根本没有
    `paper` 字段，若只认那一个字段就会把几十张身份明确的卡判成「无法对齐」
    （**这正是漏洞 #11「判据只认一种字段名」的同型**）。"""
    s = re.sub(r"^Skill-", "", slug)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[-_/]+", " ", s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return {w for w in s.split() if len(w) > 2}


#: 泛词：在标题里到处都有，不能作为身份证据（否则任何卡都能靠它们「对齐」）。
IDENTITY_STOPWORDS = {
    "for", "with", "and", "the", "via", "using", "based", "towards", "toward",
    "learning", "model", "models", "analysis", "system", "systems", "data",
    "approach", "method", "methods", "framework", "based", "large", "language",
    "multi", "deep", "neural", "agent", "agents", "from", "into", "are", "can",
}


def identity_ok(card: dict, meta: dict) -> tuple[bool, str]:
    """卡声明的论文 vs `paper_id` 指向的 arXiv 论文，对得上吗？

    判据**三源并列**（任一命中即算对得上；全不命中才判错位）：
      ① 卡 `paper` 字段（论文标题原文）
      ② 卡 slug 的实词（卡自己的槽位名）
      ③ 两者都拿不到 ⇒ **无法对齐**（fail loud，不当作「对齐」）
    """
    ct = (meta.get("citation_title") or "").strip()
    pid = (card["fm"].get("paper_id") or "").strip()
    if not ct:
        return False, "arXiv 元数据没取到标题 ⇒ 无法对齐"
    a = _norm_title(ct)
    sig = a - IDENTITY_STOPWORDS

    card_paper = (card["fm"].get("paper") or "").strip()
    if card_paper and card_paper.strip() not in (pid, f"arXiv:{pid}"):
        b = _norm_title(card_paper)
        inter = b & sig
        if len(inter) >= 3 or (len(b) <= 5 and len(inter) >= 2):
            return True, f"`paper` 字段对齐（交集 {len(inter)}：{sorted(inter)[:5]}）"
        # paper 字段对不上，再给 slug 一次机会（paper 字段常被写成粗引用）
        sw = _slug_words(card["slug"])
        sint = sw & sig
        if len(sint) >= 2:
            return True, f"`paper` 字段不足，slug 兜底对齐（交集 {sorted(sint)}）"
        return False, (f"标题词交集仅 {len(inter)}：arXiv={ct[:60]!r} vs 卡={card_paper[:60]!r}；"
                       f"slug 兜底也不足（{sorted(sint)}）")

    sw = _slug_words(card["slug"])
    sint = sw & sig
    if len(sint) >= 2:
        return True, f"slug 对齐（交集 {sorted(sint)}）"
    if len(sw) <= 2 and len(sint) >= 1:
        return True, f"短 slug 对齐（交集 {sorted(sint)}）"

    # ③ 卡**正文里**是否逐字引了这篇论文的标题
    #    （「论文来源: <标题> / arXiv ID: <号>」是卡自己的来源声明 —— 不是底本正文，
    #      与铁律 1 不冲突：铁律 1 禁的是拿**论文正文**判 venue，不是禁读**卡片**的声明）
    body_words = _norm_title(card.get("text", ""))
    bint = body_words & sig
    if len(bint) >= 3:
        return True, f"卡正文含论文标题词（交集 {sorted(bint)[:6]}）"

    # ④ 卡 `title` 字段兜底（title 是卡的**名称**，也是卡对自身身份的声明）
    card_title = (card["fm"].get("title") or "").strip()
    if card_title:
        tw = _norm_title(card_title)
        tint = tw & sig
        if len(tint) >= 2 or (len(tw) <= 4 and len(tint) >= 1):
            return True, f"卡 `title` 兜底对齐（交集 {sorted(tint)}）"

    return None, (f"卡无 `paper` 字段、slug/title 与 arXiv 标题都对不上："
                  f"arXiv={ct[:60]!r} vs slug={card['slug']}（交集 {sorted(sint)}）")


def _tier_from_name(blob: str) -> list[tuple[str, str]]:
    """从一段文本里找出被点名的 venue，返回 [(name, tier)]，按档位从高到低。"""
    low = blob.lower()
    hits: dict[str, str] = {}
    for name, tier in {**VENUE_NAME_TIER, **JOURNAL_NAME_TIER}.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", low):
            hits[name] = tier
    order = {"UTD24": 0, "FT50": 1, "CCF-A": 2, "CCF-B": 3, "field-top": 4}
    return sorted(hits.items(), key=lambda kv: order.get(kv[1], 9))


def _anchor(kind: str, body: str) -> str:
    """**首次判定时**的仪器读数快照 —— 落盘进 `venue_source` 的括号里。

    ⚠️ 为什么必须存这个：`evidence` 若在**重跑时**现算，就会变成读盘上的值
    （实测：重跑后 121 行 `backfilled` 的 evidence 全变成
    `frontmatter venue_tier='non-paper' venue_source='...'` —— 那是**结论**，不是**依据**）。
    与 `old_value` 被写盘后读回是**同一个缺陷**，只是换了一列。
    """
    return f"{kind}({body})" if body else kind


def _fill_instrument_evidence(res: dict, meta: dict) -> None:
    """把**仪器的原始读数**放进 `evidence`，而不是「盘上取值长什么样」。

    ⚠️ 这是同一个缺陷的第三处（前两处：`old_value`、`reason` 的人读列）。
    病因一致：**回填落盘之后重跑，判定链走的是 `as-is` 分支，
    该分支的 `evidence` 只能写 `frontmatter venue_tier='preprint'` —— 那是结论，不是依据。**
    而 abs 页的 `Comments` 实测**永远都在** `arxiv_abs.json` 里，没有理由不写。
    """
    if not (meta.get("comments") or meta.get("jref") or meta.get("doi")):
        return
    parts = []
    if meta.get("comments"):
        parts.append(f"Comments={meta['comments']!r}")
    if meta.get("jref"):
        parts.append(f"journal_ref={meta['jref']!r}")
    if meta.get("doi"):
        parts.append(f"DOI={meta['doi']!r}")
    res["evidence"] = "arxiv-abs " + " ".join(parts)


def classify_from_evidence(card: dict, meta: dict, crossref: dict) -> dict:
    """铁律 1 的执行体：只用 arXiv abs 页字段 + Crossref 出版方记录，**不看底本正文**。

    判定顺序（每一档都要说出依据）：
      ① abs 页 `journal_ref` / `Comments` 里的**发表类声明 + 会名**
      ② abs 页给出的 **DOI** → Crossref `container-title` / `event`
      ③ 卡 `venue` 字段里的会名/刊名（**卡侧声明**，不是底本正文）
      ④ 都没有 ⇒ `preprint`（未评审预印本，这是 arXiv 的默认状态）
    """
    slug = card["slug"]
    comments = (meta.get("comments") or "").strip()
    jref = (meta.get("jref") or "").strip()
    doi_field = (meta.get("doi") or "").strip()
    doi = re.search(r"10\.\d{4,9}/\S+", doi_field)
    doi = doi.group(0).rstrip(".") if doi else ""

    src_blob = f"{comments} || {jref}"
    card_venue_pre = (card["fm"].get("venue") or "").strip()

    # —— ⓪ 已登记的「非白名单会」：有正式录用，但无白名单档 ——
    low_all = f"{src_blob} || {card_venue_pre}".lower()
    for key, why in UNLISTED_VENUE.items():
        if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", low_all):
            return {
                "tier": "preprint",
                "status": "backfilled",
                "source": _anchor("unlisted-venue", "preprint"),
                "evidence": f"arxiv-abs Comments={comments!r} journal_ref={jref!r} card.venue={card_venue_pre!r}",
                "reason": f"命中已登记的非白名单会 `{key}`：{why}",
            }

    # —— ① abs 页的会名 ——
    hits = _tier_from_name(src_blob)
    if hits:
        name, tier = hits[0]
        # 两类的并集才算「有轨道语义」；具体走哪条**在下面再分**
        downgrade = bool(
            MAIN_TRACK_PAT.search(comments) or WORKSHOP_PAT.search(comments)
            or MAIN_TRACK_PAT.search(jref) or WORKSHOP_PAT.search(jref)
        )
        applied = tier
        note = ""
        if downgrade:
            # —— 语义切分：主会自带 track ≠ 独立 workshop（见 MAIN_TRACK_PAT 注释）——
            # 判据：把主会 track 词从 Comments 里摘掉后**不再有** workshop 词 ⇒ 它只是主会 track。
            if MAIN_TRACK_PAT.search(comments) and not WORKSHOP_PAT.search(
                MAIN_TRACK_PAT.sub(" ", comments)
            ):
                return {
                    "tier": tier,
                    "status": "backfilled",
                    "source": _anchor("arxiv-abs", "主会自带轨道"),
                    "evidence": f"arxiv-abs Comments={comments!r}",
                    "reason": (
                        f"Comments 点名 `{name}`（{tier}）且是**主会自带的 track**"
                        f"（Industry Track / Findings / Main Conference）—— 论文集是主会论文集，"
                        f"**不按 §3「Workshop 降一级」处理**。"
                        f"本处置与 registry `p2s-2026-0007` 的 `venue=SIGIR / tier=CCF-A` 一致。"
                    ),
                }
            applied, note = tier, ""
            if tier == "CCF-A":
                applied, note = "CCF-B", f"Comments 含 **workshop/短文**词 ⇒ §3 降一级 {tier}→CCF-B"
            elif tier == "CCF-B":
                applied, note = CCF_DOWNGRADE_TO, (
                    f"Comments 含 **workshop/短文**词 ⇒ §3 降一级 {tier}→CCF-C，折叠为 {CCF_DOWNGRADE_TO}"
                )
            else:
                applied, note = tier, f"Comments 含轨道词，但 {tier} 之下即 preprint ⇒ 保持 {tier}"
        else:
            note = f"abs 页点名 `{name}` ⇒ {tier}"
        return {
            "tier": applied,
            "status": "backfilled",
            "source": _anchor("arxiv-abs", f"comments={comments[:80]!r}"),
            "evidence": f"arxiv-abs Comments={comments!r} journal_ref={jref!r}",
            "reason": note,
        }

    # —— ①b 有发表类声明但会名不认识 ⇒ **登记，不猜** ——
    # 反后门：若这里默认落 `preprint`，就等于把「我没查过这个会」伪装成「它是预印本」，
    # 且报告上完全看不出来（两者同为 preprint）。
    if ACCEPT_PAT.search(src_blob):
        return {
            "tier": None,
            "status": "needs-resolution",
            "source": "arxiv-abs(会名不识别)",
            "evidence": f"arxiv-abs Comments={comments!r} journal_ref={jref!r}",
            "reason": "abs 页有发表类声明，但会名不在 VENUE_NAME_TIER 里 ⇒ 登记，不猜档",
        }

    # —— ② DOI → Crossref ——
    if doi and doi in crossref and not crossref[doi].get("error"):
        rec = crossref[doi]
        ct = (rec.get("container-title") or [""])
        ct = ct[0] if isinstance(ct, list) and ct else ""
        ev = (rec.get("event") or {}).get("name", "") if isinstance(rec.get("event"), dict) else ""
        chits = _tier_from_name(f"{ct} || {ev}")
        if chits:
            name, tier = chits[0]
            return {
                "tier": tier,
                "status": "backfilled",
                "source": _anchor("crossref", f"doi={doi}"),
                "evidence": f"doi={doi} container-title={ct!r} event={ev!r} page={rec.get('page')!r}",
                "reason": f"Crossref 解析出 `{name}` ⇒ {tier}（出版方记录，独立于 arXiv 自述）",
            }
        if ct:
            for key, why in UNLISTED_VENUE.items():
                if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", ct.lower()):
                    return {
                        "tier": "preprint", "status": "backfilled",
                        "source": "unlisted-venue(preprint)",
                        "evidence": f"doi={doi} container-title={ct!r}",
                        "reason": f"Crossref 的 container-title 命中已登记的非白名单刊 `{key}`：{why}",
                    }
            return {
                "tier": None,
                "status": "needs-resolution",
                "source": "crossref",
                "evidence": f"doi={doi} container-title={ct!r}",
                "reason": f"Crossref 有 `container-title` 但不认识：{ct!r} ⇒ 登记，不猜档",
            }

    # —— ③ 卡侧 venue 字段 ——
    card_venue = (card["fm"].get("venue") or "").strip()
    if card_venue and not re.search(r"arXiv preprint|preprint$", card_venue, re.I):
        vhits = _tier_from_name(card_venue)
        if vhits:
            name, tier = vhits[0]
            return {
                "tier": tier,
                "status": "backfilled",
                "source": _anchor("card-venue-field", card_venue[:60]),
                "evidence": f"card.venue={card_venue!r}",
                "reason": f"卡侧 `venue` 字段点名 `{name}` ⇒ {tier}（卡侧声明，非底本正文）",
            }
        for key, why in UNLISTED_VENUE.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", card_venue.lower()):
                return {
                    "tier": "preprint", "status": "backfilled",
                    "source": "unlisted-venue(preprint)",
                    "evidence": f"card.venue={card_venue!r}",
                    "reason": f"卡侧 venue 命中已登记的非白名单会 `{key}`：{why}",
                }
        if ACCEPT_PAT.search(card_venue) or re.search(r"\b(19|20)\d{2}\b", card_venue):
            return {
                "tier": None,
                "status": "needs-resolution",
                "source": "card-venue-field",
                "evidence": f"card.venue={card_venue!r}",
                "reason": f"卡侧 `venue` 有具体会/刊名但不认识：{card_venue!r} ⇒ 登记，不猜档",
            }

    # —— ④ 默认：未评审预印本 ——
    if comments or jref or card_venue:
        return {
            "tier": "preprint",
            "status": "backfilled",
            "source": _anchor("arxiv-abs", "无发表声明"),
            "evidence": f"comments={comments!r} jref={jref!r} card.venue={card_venue!r}",
            "reason": "abs 页 Comments/journal_ref 与卡侧 venue 均无**正式发表**声明 ⇒ 未评审预印本",
        }
    return {
        "tier": "preprint",
        "status": "backfilled",
        "source": _anchor("arxiv-abs", "三字段皆空"),
        "evidence": "arxiv-abs Comments/journal_ref/DOI 皆空 ⇒ arXiv 未评审预印本（默认状态）",
        "reason": "abs 页三个字段皆空且卡侧无 venue ⇒ arXiv 未评审预印本（默认状态，非猜测）",
    }


def judge_card(card: dict, arxiv: dict, crossref: dict) -> dict:
    """算出一张卡的 (tier, status, source, evidence, reason)。"""
    cur = (card["fm"].get("venue_tier") or "").strip()
    pid = (card["fm"].get("paper_id") or "").strip()
    eb = (card["fm"].get("evidence_basis") or "").strip()
    slug = card["slug"]

    # ① 逐卡裁决表优先（`top` 与身份错位卡都靠它）
    # 盘上已写过的 `venue_source` **随卡走**（不重新推导）——
    # 否则第二遍跑时所有卡都会退回 `frontmatter-as-is`，**原始出处被自己的输出吃掉**
    # （实测：来源分布从 4 类塌成 2 类）。
    sticky = (card["fm"].get("venue_source") or "").strip()

    # ①b **保底仪器读数**：从 `paper_id` 对应的 arXiv 元数据 / Crossref 缓存里取原始字段。
    #     为什么需要：卡一旦落盘，`as-is` 分支的 evidence 只能写「盘上取值长什么样」——
    #     那是**结论**不是**依据**。abs 页的 Comments 实测一直躺在 `arxiv_abs.json` 里，
    #     没有理由不把它写进账（同一个缺陷的第三处，前两处是 old_value 与 reason 人读列）。
    fallback_ev = ""
    if not pid:
        # 卡本身没有论文来源 ⇒ 依据就是**卡自己的声明**，且必须说清是哪一种声明。
        eb0 = (card["fm"].get("evidence_basis") or "").strip()
        if eb0 == "author-practice":
            fallback_ev = ("card-metadata evidence_basis=author-practice"
                           "（卡自承无论文来源 ⇒ 无 venue 可判，按白名单 §5 落 non-paper）")
        elif eb0:
            fallback_ev = f"card-metadata evidence_basis={eb0}（卡自承的来源类别）"
        else:
            fallback_ev = ("card-metadata 既无 `paper_id` 也无 `evidence_basis` 声明"
                           "（**不可回填**：不替它猜）")
    elif re.match(r"^\d{4}\.\d{4,5}$", pid) and pid in arxiv:
        _m = arxiv[pid]
        _parts = []
        if _m.get("comments"):
            _parts.append(f"Comments={_m['comments']!r}")
        if _m.get("jref"):
            _parts.append(f"journal_ref={_m['jref']!r}")
        if _m.get("doi"):
            _parts.append(f"DOI={_m['doi']!r}")
        fallback_ev = "arxiv-abs " + " ".join(_parts) if _parts else (
            "arxiv-abs Comments/journal_ref/DOI 皆空 ⇒ 未评审预印本（默认状态）")
    elif re.match(r"^10\.\d{4,9}/\S+$", pid) and crossref.get(pid):
        _r = crossref[pid]
        _ct = (_r.get("container-title") or [""])
        _ct = _ct[0] if isinstance(_ct, list) and _ct else ""
        fallback_ev = f"crossref doi={pid} container-title={_ct!r} page={_r.get('page')!r}"

    def _ev(default: str) -> str:
        return fallback_ev or default

    if slug in EXPLICIT_RETIERS:
        tier, why = EXPLICIT_RETIERS[slug]
        if tier not in CANONICAL_TIERS:
            return {"tier": None, "status": "bad-resolution", "source": "EXPLICIT_RETIERS",
                    "evidence": "", "reason": f"裁决值 `{tier}` 不在规范 7 档"}
        return {"tier": tier, "status": "explicit", "source": sticky or _anchor("EXPLICIT_RETIERS", slug),
                "evidence": _ev(f"逐卡裁决（理由见 venue_tier_mapping.json 的 card_resolution.{slug}）"),
                "reason": why}

    # ② 已有取值：规范值原样保留；轨道标记按表折叠；表外值报红
    if cur:
        if cur in CANONICAL_TIERS:
            return {"tier": cur, "status": "as-is", "source": sticky or "frontmatter-as-is",
                    "evidence": _ev(f"frontmatter venue_tier={cur!r}（本次未改动）"),
                    "reason": "已是规范档，逐字保留"}
        if cur in TIER_ALIAS:
            return {"tier": TIER_ALIAS[cur], "status": "legacy-mapped", "source": "TRACK_RULES",
                    "evidence": _ev(f"frontmatter venue_tier={cur!r}（轨道标记，按表折叠）"),
                    "reason": TRACK_RULES[cur][2]}
        if cur in TRACK_MARKERS:
            return {"tier": None, "status": "needs-resolution", "source": "TRACK_RULES",
                    "evidence": f"frontmatter venue_tier={cur!r}", "reason": TRACK_RULES[cur][2]}
        return {"tier": None, "status": "unknown", "source": "-",
                "evidence": f"frontmatter venue_tier={cur!r}", "reason": f"表外取值 `{cur}`"}

    # ③ 无来源卡：**只有**显式声明 evidence_basis: author-practice 才判 non-paper
    if eb == "author-practice":
        return {
            "tier": "non-paper", "status": "backfilled",
            "source": _anchor("evidence_basis", "author-practice"),
            "evidence": _ev("卡 frontmatter 显式声明 `evidence_basis: author-practice`（无 paper_id）"),
            "reason": "卡自承无论文来源（frontmatter 显式声明）⇒ §5 的 non-paper 来源类别",
            "source_class": SOURCE_CLASS_NO_PAPER,
        }

    # ④ 有 paper_id ⇒ 铁律 1 的仪器
    if pid:
        if re.match(r"^10\.\d{4,9}/\S+$", pid):
            rec = crossref.get(pid)
            if rec is None:
                return {"tier": None, "status": "no-metadata", "source": "crossref",
                        "evidence": f"paper_id={pid}", "reason": f"Crossref 缓存里没有 {pid}"}
            ct = (rec.get("container-title") or [""])
            ct = ct[0] if isinstance(ct, list) and ct else ""
            hits = _tier_from_name(ct)
            if hits:
                return {"tier": hits[0][1], "status": "backfilled",
                        "source": _anchor("crossref", f"paper_id-as-doi={pid}"),
                        "evidence": f"doi={pid} container-title={ct!r}",
                        "reason": f"paper_id 本身是出版方 DOI，Crossref 解析出 `{hits[0][0]}` ⇒ {hits[0][1]}"}
            return {"tier": None, "status": "needs-resolution", "source": "crossref(direct)",
                    "evidence": f"doi={pid} container-title={ct!r}",
                    "reason": f"DOI 解析出的 `container-title` 不认识：{ct!r}"}
        if not re.match(r"^\d{4}\.\d{4,5}$", pid):
            return {"tier": None, "status": "id-unrecognized", "source": "-",
                    "evidence": f"paper_id={pid}", "reason": f"`{pid}` 既不是 arXiv ID 也不是 DOI"}
        meta = arxiv.get(pid)
        if meta is None:
            return {"tier": None, "status": "no-metadata", "source": "arxiv-abs",
                    "evidence": f"paper_id={pid}", "reason": f"arXiv 元数据里没有 {pid}"}
        if meta.get("error"):
            return {"tier": None, "status": "fetch-failed", "source": "arxiv-abs",
                    "evidence": f"paper_id={pid}", "reason": meta["error"]}
        res = classify_from_evidence(card, meta, crossref)
        _fill_instrument_evidence(res, meta)
        ok, why = identity_ok(card, meta)
        res["identity"] = {"ok": ok, "detail": why}
        if ok is None:
            if slug in SCOPE_MISMATCH:
                res["scope_mismatch"] = SCOPE_MISMATCH[slug]
                res["reason"] = f"{res['reason']}｜⚠️ 卡–论文宽度不匹配（已登记）：{SCOPE_MISMATCH[slug]}"
            else:
                res["status"] = "identity-unaligned-unregistered"
                res["reason"] = f"身份无法对齐且未登记：{why}"
        elif ok is False:
            if slug in IDENTITY_MISMATCH:
                res["identity_conflict"] = IDENTITY_MISMATCH[slug]
                res["reason"] = f"{res['reason']}｜⚠️ 身份冲突（已登记）：{IDENTITY_MISMATCH[slug]}"
            else:
                res["status"] = "identity-conflict-unregistered"
                res["reason"] = f"身份冲突未登记：{why}"
        return res

    # ⑤ 什么都没有 ⇒ 不猜
    return {
        "tier": None, "status": "unlabeled", "source": "-", "evidence": "无 paper_id / 无 evidence_basis",
        "reason": "既无 paper_id 也无 evidence_basis 声明 ⇒ **不替它猜**（拿不到就炸，不给默认值）",
    }


# --------------------------------------------------------------------------- #
# 判据 J1–J13                                                                   #
# --------------------------------------------------------------------------- #


_HEAD_FM_CACHE: dict[str, dict | None] | None = None


def _head_snapshot(cards) -> dict:
    """一次取全库改动前快照（`git show HEAD:<path>`），进程内缓存。"""
    global _HEAD_FM_CACHE
    if _HEAD_FM_CACHE is None:
        _HEAD_FM_CACHE = {c["rel"]: read_head_frontmatter(c["rel"]) for c in cards}
    return _HEAD_FM_CACHE


def judge_cards(cards, arxiv, crossref) -> dict:
    return {c["slug"]: judge_card(c, arxiv, crossref) for c in cards}


def j_coverage(cards, verdicts) -> dict:
    labelled = [c for c in cards if verdicts[c["slug"]]["tier"] is not None]
    unlabelled = [c for c in cards if verdicts[c["slug"]]["tier"] is None]
    n, m = len(labelled), len(cards)
    ratio = (n / m) if m else 0.0
    return {"labelled": n, "unlabelled": len(unlabelled), "total": m, "ratio": ratio}


def j1(cards, verdicts) -> list:
    out = []
    for c in cards:
        v = verdicts[c["slug"]]
        if v["status"] == "unknown":
            out.append(f"{c['slug']}: 表外取值（{v['reason']}）")
        if v["tier"] is not None and v["tier"] not in CANONICAL_TIERS:
            out.append(f"{c['slug']}: 判出 `{v['tier']}` 不在规范 7 档")
    return out


def j2(cov) -> list:
    if cov["ratio"] < COVERAGE_MIN:
        return [f"覆盖率 {cov['labelled']}/{cov['total']} = {cov['ratio']:.1%} < 下界 {COVERAGE_MIN:.0%}"]
    return []


def j3(cov) -> list:
    out = []
    if cov["ratio"] > COVERAGE_MAX:
        out.append(f"覆盖率 {cov['ratio']:.1%} > 上界 {COVERAGE_MAX:.0%} —— 分子大于分母")
    if cov["labelled"] + cov["unlabelled"] != cov["total"]:
        out.append(f"自洽性破了：已标 {cov['labelled']} + 未标 {cov['unlabelled']} != 总数 {cov['total']}")
    if cov["labelled"] < 0 or cov["unlabelled"] < 0:
        out.append("计数出现负数")
    return out


def j456(doc_text, cards) -> dict:
    """J4 机读块逐字相等 + J5 §1 七档相等 + J6 frontmatter 字段不重复。"""
    out = {"J4": [], "J5": [], "J6": []}
    if doc_text is None:
        out["J4"].append("未提供 venue-whitelist.md 文本（输入没拿到 ⇒ 退出码 2）")
    else:
        want = render_inventory_block()
        got = _extract_block(doc_text)
        if got is None:
            out["J4"].append(f"venue-whitelist.md 里找不到机读块标记 {DOC_BEGIN}")
        elif got.strip() != want.strip():
            out["J4"].append("§10 机读块与生成器输出不一致（跑 `--apply` 重修）")
        sec1 = _extract_section1_tiers(doc_text)
        if sec1 is None:
            out["J5"].append("§1 的档位表没解析出来")
        elif sec1 != CANONICAL_TIERS:
            out["J5"].append(f"§1 散文表 {sec1} != 规范表 {CANONICAL_TIERS}")
    for c in cards:
        if c["fm"].get("__dup__"):
            out["J6"].append(f"{c['slug']}: frontmatter 字段重复 {c['fm']['__dup__']}")
    return out


def j7(observed) -> list:
    out = []
    for v in observed:
        val = v["value"]
        if val in TIER_ALIAS or val in TRACK_RULES or val in REGISTRY_ALIAS or val in CANONICAL_TIERS:
            continue
        if val == "" and v["where"] == "registry":
            continue
        out.append(f"三套词表里的非规范取值 `{val}`（{v['where']}，{v['seen']} 次）没有映射规则")
    for v, spec in REGISTRY_ALIAS.items():
        if not spec.get("reason"):
            out.append(f"registry 表外取值 `{v}` 登记了却没写理由")
    return out


def j8(cards, verdicts) -> list:
    """分域线：域有归属、类别有双侧规则、依据不是形容词、商科实证域不得出现 preprint。"""
    out = []
    for c in cards:
        dom = c["domain"]
        if dom not in DOMAIN_LINE:
            out.append(f"{c['slug']}: 技术域 `{dom}` 不在域线表里")
            continue
        spec = DOMAIN_LINE[dom]
        cat = spec["category"]
        if cat not in CATEGORY_RULE:
            out.append(f"域 `{dom}` 的类别 `{cat}` 不在 CATEGORY_RULE 里")
            continue
    for dom, d in DOMAIN_LINE.items():
        if len(d.get("basis", "").strip()) < 10:
            out.append(f"域 `{dom}` 的判定依据太短/缺失 —— 形容词不是判据")
        if d.get("category") == "商科实证类" and not re.search(
            r"UTD24|FT50|Marketing Science|Journal of Marketing|JMR|ISR|MISQ", d.get("basis", "")
        ):
            out.append(f"域 `{dom}` 判为商科实证类，却没点名权威来源刊 —— 判据不足")
    for cat, spec in CATEGORY_RULE.items():
        for k in ("preprint_accepted", "why", "requires", "forbids"):
            if k not in spec:
                out.append(f"类别 `{cat}` 缺 `{k}` —— 判据只写单边等于没有判据")
    return out


def registry_top_journal_domains() -> dict:
    """**分域线的独立第二口径**：登记层（registry）把 UTD24/FT50 论文投到了哪些域。

    为什么必须有这一口径：卡层（146 张）看不见「本来该来、但还没出卡」的商科顶刊来源。
    实测 2026-09-13：registry 8 条 UTD24 记录的 `outputs.skill_card` **全是
    `Skill-<方法名>.md` 占位**，一条卡都没出 ⇒ 卡层 UTD24 = 0 是「**还没出卡**」，
    不是「回填把 UTD24 折叠掉了」（回填的来源分布里根本没有 UTD24 这一档）。
    """
    if not REGISTRY.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for r in json.loads(REGISTRY.read_text(encoding="utf-8")).get("records", []):
        if r.get("venue_tier") not in ("UTD24", "FT50"):
            continue
        sc = (r.get("outputs") or {}).get("skill_card", "")
        parts = sc.split("/")
        domain = parts[1] if sc.startswith("paper2skills-vault/") and len(parts) > 2 else r.get("domain", "-")
        out.setdefault(domain, []).append({
            "paper_id": r.get("paper_id"), "venue": r.get("venue"),
            "venue_tier": r.get("venue_tier"),
            "card_exists": bool(sc) and (REPO / sc).exists(),
            "skill_card": sc,
        })
    return out


def j9(product_bytes) -> list:
    if product_bytes is None:
        if PRODUCT_VENDORED.exists():
            return ["产品侧副本存在但读不到（输入没拿到）"]
        return [f"产品侧投递副本不存在：{PRODUCT_VENDORED}（跑 `--sync-product`）"]
    if product_bytes != render_map_bytes():
        return [f"产品侧副本与规范表不一致：{PRODUCT_VENDORED}（跑 `--sync-product`）"]
    return []


def j10(cards, verdicts) -> list:
    out = []
    bad = {"no-metadata", "needs-resolution", "unknown", "id-unrecognized", "fetch-failed",
           "bad-resolution", "identity-unaligned-unregistered", "identity-conflict-unregistered"}
    for c in cards:
        v = verdicts[c["slug"]]
        if v["status"] in bad:
            out.append(f"{c['slug']}: 判定未完成（{v['status']}）—— {v['reason']}")
        if v["status"] == "unlabeled" and (c["fm"].get("paper_id") or "").strip():
            out.append(f"{c['slug']}: 有 paper_id 却判不出 tier（{v['reason']}）")
    return out


def j11(cards, verdicts, *, whole_vault: bool = False) -> list:
    """台账完备：登记为不可回填的必须在册，在册的必须真的不可回填。

    `whole_vault=False`（夹具模式）时**不套真台账** —— 夹具是受控子集，
    拿全库台账去比会把「夹具只有 3 张卡」误报成「台账点名了不存在的卡」。
    真实运行时 `whole_vault=True`，两项都查。
    """
    out = []
    slugs = {c["slug"] for c in cards}
    if not whole_vault:
        return out
    for slug in UNRESOLVABLE:
        if slug not in slugs:
            out.append(f"UNRESOLVABLE 点名了不存在的卡 `{slug}`")
        elif verdicts[slug]["tier"] is not None:
            out.append(f"`{slug}` 登记为「不可回填」但实际判出了 `{verdicts[slug]['tier']}` ⇒ 台账过期")
    for slug in LOW_CONFIDENCE:
        if slug not in slugs:
            out.append(f"LOW_CONFIDENCE 点名了不存在的卡 `{slug}`")
    for c in cards:
        if verdicts[c["slug"]]["status"] == "unlabeled" and c["slug"] not in UNRESOLVABLE:
            out.append(f"{c['slug']}: 未标注但不在 UNRESOLVABLE 台账里 ⇒ 静默留空（不许）")
    return out


def j12(cards, verdicts, *, whole_vault: bool = False) -> list:
    """只增不改：原值非空时不得被改写（EXPLICIT_RETIERS 除外，且必须给出理由）。"""
    out = []
    if not whole_vault:
        return out
    for slug, (new, why) in EXPLICIT_RETIERS.items():
        if slug not in {c["slug"] for c in cards}:
            out.append(f"EXPLICIT_RETIERS 点名了不存在的卡 `{slug}`")
        elif len(why.strip()) < 20:
            out.append(f"EXPLICIT_RETIERS 的 `{slug}` 理由太短 —— 必须点名仪器")
        elif new not in CANONICAL_TIERS:
            out.append(f"EXPLICIT_RETIERS 的 `{slug}` 值 `{new}` 不在规范 7 档")
    for c in cards:
        cur = (c["fm"].get("venue_tier") or "").strip()
        v = verdicts[c["slug"]]
        if cur and cur in CANONICAL_TIERS and v["tier"] is not None and v["tier"] != cur:
            out.append(f"{c['slug']}: 已有规范值 `{cur}` 被改判为 `{v['tier']}` —— 只增不改被绕过")
    return out


def j13(cards, verdicts, *, whole_vault: bool = False) -> list:
    """身份三态：对齐 / 无法对齐（须登记宽度不匹配）/ 冲突（须登记原因）。**逐条都必须有处置。**

    `whole_vault=False`（夹具模式）时只查「卡的处置」，不查「台账里的名字是否存在」
    —— 台账是全库口径，拿它比夹具子集会误报。
    """
    out = []
    slugs = {c["slug"] for c in cards}
    for c in cards:
        v = verdicts[c["slug"]]
        ident = v.get("identity")
        if not ident:
            continue
        if ident["ok"] is None and c["slug"] not in SCOPE_MISMATCH:
            out.append(f"{c['slug']}: 身份无法对齐且未登记宽度不匹配（{ident['detail']}）")
        if ident["ok"] is False and c["slug"] not in IDENTITY_MISMATCH:
            out.append(f"{c['slug']}: 身份冲突未登记（{ident['detail']}）")
    if not whole_vault:
        return out
    for slug, why in IDENTITY_MISMATCH.items():
        if slug not in slugs:
            out.append(f"IDENTITY_MISMATCH 点名了不存在的卡 `{slug}`")
        elif len(why.strip()) < 15:
            out.append(f"IDENTITY_MISMATCH 的 `{slug}` 理由太短 —— 必须点名仪器/证据")
    for slug, why in SCOPE_MISMATCH.items():
        if slug not in slugs:
            out.append(f"SCOPE_MISMATCH 点名了不存在的卡 `{slug}`")
        elif len(why.strip()) < 15:
            out.append(f"SCOPE_MISMATCH 的 `{slug}` 理由太短")
    return out


def j14(cards, verdicts, head_fm) -> list:
    """账的溯源强度：`old_value` 必须来自改动前快照，且三态必须与快照吻合。

    反后门（**这条判据诞生的原因**）：首版 `old_value` 取自**写盘之后**的 cards 内存，
    于是 `old_value == new_value` 恒成立 —— 146 条账里「改动数 = 0」，
    而实际有 121 条是新赋值。**账因此答不出它存在的那个问题。**
    判据三条：
      ① `as-is` 的 old 必须逐字等于 new；`backfilled` 的 old 必须为空；`retiered` 的 old 必须非空且 ≠ new
      ② 每条账的 `old_value` 必须与快照里读出来的值**逐字相等**（拿快照重算一遍）
      ③ `changed` 条数必须等于「old != new」的实测条数（不许只报一边）
    """
    out = []
    rows = render_backfill(cards, verdicts, j_coverage(cards, verdicts), head_fm=head_fm)["cards"]
    for r in rows:
        st, old, new = r["change_status"], r["old_value"], r["new_value"] or ""
        if st == "as-is" and old != new:
            out.append(f"{r['slug']}: as-is 但 old({old!r}) != new({new!r})")
        if st == "backfilled" and old != "":
            out.append(f"{r['slug']}: backfilled 但 old 非空（{old!r}）")
        if st == "retiered" and (not old or old == new):
            out.append(f"{r['slug']}: retiered 但 old({old!r}) 为空或等于 new")
        if st == "no-snapshot":
            out.append(f"{r['slug']}: 取不到改动前快照 ⇒ 该条账不可复算")
        snap = head_fm.get(r["rel"])
        snap_old = ((snap or {}).get("venue_tier") or "").strip()
        if snap is not None and snap_old != old:
            out.append(f"{r['slug']}: old_value({old!r}) 与快照读出的值({snap_old!r}) 不一致")
        if r["changed"] != (old != new and st != "no-snapshot"):
            out.append(f"{r['slug']}: changed 标记与 old/new 不符")
    # ④ `evidence` 必须是**依据**，不许是「盘上取值长什么样」的复述。
    #    实测撞过三次同一缺陷：old_value（写盘后读回）、reason 人读列（复用会过期的话）、
    #    evidence（重跑时现算 ⇒ 变成 `frontmatter venue_tier='preprint'` —— 那是**结论**）。
    NARRATION = ("frontmatter venue_tier=", "frontmatter venue_source=")
    for r in rows:
        if any(r["evidence"].startswith(n) or f"；{n}" in r["evidence"] for n in NARRATION):
            out.append(f"{r['slug']}: evidence 是「盘上取值的复述」而不是「依据」—— {r['evidence'][:60]}")
        if not r["evidence"].strip():
            out.append(f"{r['slug']}: evidence 为空 —— 回填不可审计")

    n_changed = sum(1 for r in rows if r["changed"])
    n_diff = sum(1 for r in rows if r["old_value"] != (r["new_value"] or ""))
    if n_changed != n_diff:
        out.append(f"changed 计数({n_changed}) != old≠new 实测条数({n_diff})")
    return out


def run_checks(cards, arxiv, crossref, observed, *, doc_text, product_bytes,
               whole_vault: bool = False, head_fm=None) -> dict:
    """`whole_vault=False`（夹具模式）时**不读真 git**：夹具卡不在 HEAD 里，
    硬读会把每一张都报成 `no-snapshot`（假红）。夹具模式改用「卡自己的 frontmatter」
    当快照 —— 这样 J14 仍在测「三态与快照吻合」，只是快照是给定的。"""
    if head_fm is None:
        head_fm = _head_snapshot(cards) if whole_vault else {
            c["rel"]: dict(c["fm"]) for c in cards
        }
    verdicts = judge_cards(cards, arxiv, crossref)
    cov = j_coverage(cards, verdicts)
    dd = j456(doc_text, cards)
    problems = {
        "J1": j1(cards, verdicts),
        "J2": j2(cov),
        "J3": j3(cov),
        "J4": dd["J4"],
        "J5": dd["J5"],
        "J6": dd["J6"],
        "J7": j7(observed),
        "J8": j8(cards, verdicts),
        "J9": j9(product_bytes),
        "J10": j10(cards, verdicts),
        "J11": j11(cards, verdicts, whole_vault=whole_vault),
        "J12": j12(cards, verdicts, whole_vault=whole_vault),
        "J13": j13(cards, verdicts, whole_vault=whole_vault),
        "J14": j14(cards, verdicts, head_fm),
        "J15": _assert_no_duplicate_keys(),
    }
    violations = []
    for c in cards:
        dom = c["domain"]
        spec = DOMAIN_LINE.get(dom)
        if not spec:
            continue
        cat = spec["category"]
        if cat not in CATEGORY_RULE:
            continue
        tier = verdicts[c["slug"]]["tier"]
        if tier == "preprint" and not CATEGORY_RULE[cat]["preprint_accepted"]:
            violations.append({
                "slug": c["slug"], "domain": dom, "category": cat, "tier": tier,
                "old_value": (c["fm"].get("venue_tier") or "").strip(),
                "note": "既有卡（入卡早于本线）—— 登记不重判（纪律 6）",
            })
    return {"problems": {k: v for k, v in problems.items()}, "verdicts": verdicts,
            "coverage": cov, "line_violations": violations}


def _extract_block(doc_text: str) -> str | None:
    i = doc_text.find(DOC_BEGIN)
    j = doc_text.find(DOC_END)
    if i < 0 or j < 0:
        return None
    return doc_text[i : j + len(DOC_END)]


def _extract_section1_tiers(doc_text: str) -> list[str] | None:
    m = re.search(r"^##\s*1\.\s*层级定义\s*$(.*?)(?=^##\s)", doc_text, re.S | re.M)
    if not m:
        return None
    tiers = re.findall(r"^\|\s*`([A-Za-z0-9\-]+)`\s*\|", m.group(1), re.M)
    return tiers or None


# --------------------------------------------------------------------------- #
# 产物                                                                          #
# --------------------------------------------------------------------------- #


def _assert_no_duplicate_keys() -> list:
    """字典字面量里的**重复键是静默的**（Python 取最后一个）——
    实测本条就是被这个陷阱咬的：同一个 slug 写了两条 EXPLICIT_RETIERS，
    前一条带 arXiv Comments 逐字的详细理由被**整条吃掉**，而门禁全绿。
    判据：源码里每个表的键不得出现两次。"""
    import ast as _ast

    src = Path(__file__).read_text(encoding="utf-8")
    tree = _ast.parse(src)
    out = []
    targets = {"EXPLICIT_RETIERS", "IDENTITY_MISMATCH", "SCOPE_MISMATCH", "LOW_CONFIDENCE",
               "UNRESOLVABLE", "TIER_ALIAS", "TRACK_RULES", "REGISTRY_ALIAS",
               "VENUE_NAME_TIER", "JOURNAL_NAME_TIER", "UNLISTED_VENUE", "DOMAIN_LINE",
               "CATEGORY_RULE"}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            name = getattr(tgt, "id", None)
            if name in targets and isinstance(node.value, _ast.Dict):
                seen, dup = {}, []
                for k in node.value.keys:
                    if isinstance(k, _ast.Constant):
                        if k.value in seen:
                            dup.append(k.value)
                        seen[k.value] = True
                if dup:
                    out.append(f"{name} 里有重复键 {dup} —— 后一条会**静默吃掉**前一条")
    return out


def render_map_bytes() -> bytes:
    obj = {
        "_meta": {
            "generator": "paper2skills-research/scripts/build_venue_tiers.py",
            "phase": "PHASE6-S10",
            "purpose": "venue 词表的唯一机读实现；本文件是**投递副本的字节源**，产品侧不得手改",
            "fact_source": "paper2skills-vault/07-资源库/venue-whitelist.md §1（7 档）+ §10（本表）",
            "consumers": [
                "paper2skills-research/scripts/build_venue_tiers.py（唯一 writer）",
                "dsh-paper2skills/lib/axis.js（reader；副本须与本文件逐字节相等，J9 断言）",
            ],
        },
        "canonical_tiers": CANONICAL_TIERS,
        "unlabeled": UNLABELED,
        "ccf_downgrade_to": CCF_DOWNGRADE_TO,
        "tier_alias": TIER_ALIAS,
        "track_markers": TRACK_MARKERS,
        "track_rules": [
            {"marker": k, "foldable": v[0], "to": v[1], "reason": v[2]} for k, v in TRACK_RULES.items()
        ],
        "registry_alias": REGISTRY_ALIAS,
        "venue_name_tier": VENUE_NAME_TIER,
        "unlisted_venue": UNLISTED_VENUE,
        "journal_name_tier": JOURNAL_NAME_TIER,
        "domain_line": [
            {"domain": k, "category": v["category"], "basis": v["basis"]}
            for k, v in sorted(DOMAIN_LINE.items())
        ],
        "category_rule": CATEGORY_RULE,
        "line_out_of_sample_tiers": sorted(LINE_OUT_OF_SAMPLE_TIERS),
        "coverage_gate": {"min": COVERAGE_MIN, "max": COVERAGE_MAX},
        "card_resolution": {
            k: {"to": v[0], "why": v[1]} for k, v in sorted(EXPLICIT_RETIERS.items())
        },
        "identity_mismatch": IDENTITY_MISMATCH,
        "scope_mismatch": SCOPE_MISMATCH,
        "low_confidence": LOW_CONFIDENCE,
    }
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


#: 改动前快照的来源。**不接受「写盘后读回」** —— 那样 `old_value` 恒等于 `new_value`，
#: 账就答不出它存在的那个问题（实测：146 条 `old==new`，改动的 121 条一条也看不出来）。
SNAPSHOT_REF = "HEAD"

#: 本任务**动工前的那个 commit**，钉死。
#: ⚠️ 为什么不继续用 `HEAD`：并发会话会提交。实测 2026-09-13 本轮落盘后，
#: 主控把 `l3_*` + `venue_tier` 一起 commit 成 `e7f74b9` ⇒ 再跑时 `HEAD` 已经是**改动后**，
#: 三态**整片塌成 `as-is`、`changed` 归零** —— 与首版「写盘后读回」是**同一个缺陷长在另一个指针上**。
#: ⇒ 快照必须钉在一个**不会移动**的 ref 上，并在报告里给出 SHA（便于任何人复算）。
SNAPSHOT_PINNED_SHA = "08dd624111f92544c38b554f1bc0a3a39b5c5641"   # S10 动工前的 HEAD~1
SNAPSHOT_REF_RESOLVED = SNAPSHOT_PINNED_SHA


def read_head_frontmatter(rel_path: str) -> dict | None:
    """从 `git show HEAD:<path>` 取改动前的 frontmatter。取不到 ⇒ None（不是空 dict）。

    ⚠️ 取不到与「取到了但没有该字段」是**两件事**，必须分开：
    前者是「无快照」（旧账口径），后者是「有新赋值」。合并会让 121 条新赋值伪装成 as-is。
    """
    try:
        out = subprocess.run(
            ["git", "show", f"{SNAPSHOT_REF_RESOLVED}:{rel_path}"],
            capture_output=True, text=True, cwd=str(REPO), check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    fm, ok = split_fm(out)
    return fm if ok else None


def classify_change(head_fm: dict | None, new_value: str) -> tuple[str, str]:
    """账的三态（**这是本账最重要的判据形态**）。返回 (status, old_value)。

    | status | 含义 | 溯源强度 |
    |--------|------|----------|
    | `as-is` | HEAD 已有该字段，**逐字未变** | 最强：本次没动它 |
    | `backfilled` | HEAD **没有**该字段，本次新赋值 | 中：值来自本次判定 |
    | `retiered` | HEAD 有值，本次**改了**（规范化 / 逐卡裁决） | 中：改动可 git diff |
    | `no-snapshot` | git 取不到该卡的改动前版本 | **最弱**：不可复算，必须点名 |
    """
    if head_fm is None:
        return "no-snapshot", ""
    old = (head_fm.get("venue_tier") or "").strip()
    if not old:
        return "backfilled", ""
    if old == new_value:
        return "as-is", old
    return "retiered", old


LEDGER_REASON_LEGEND = {
    "as-is": "本次**没动它**（逐字保留）",
    "backfilled": "本次**新赋值**",
    "retiered": "本次**改了值**",
    "no-snapshot": "**取不到改动前快照**，不可复算",
}


def compose_ledger_reason(change_status: str, source: str, evidence: str, judge_reason: str) -> str:
    """账的人读列。

    ⚠️ **不许直接复用判定链的 `reason`** —— 那个字符串在**第一遍跑时**是对的
    （「将按 X 新赋值」），而卡片落盘之后重跑，判定链走的是 `as-is` 分支、
    于是同一句变成「已是规范档，逐字保留」。实测：**121 行 `backfilled` 的账
    里 111 行的人读列在为一桩从未发生的「保留」作证**（主控独立复核抓出）。
    机读列（`old_value` / `change_status`）已经诚实，坏的只有人读列 ——
    这与台账 #18「判据是真的，报告把它翻译成了另一句话」同族。
    ⇒ 账的 reason 必须由 **`change_status` + 本次取值来源** 组合出来，
       与判定链那句**解耦**；判定链那句话另存 `judge_reason` 供追溯。
    """
    head = f"[{change_status}] {LEDGER_REASON_LEGEND.get(change_status, change_status)}"
    src = source or "-"
    ev = (evidence or "").strip()
    ev_part = f"；依据：{ev}" if ev else ""
    if change_status == "backfilled":
        return (f"{head}（改动前无 `venue_tier`）—— 按 `{src}` 判为"
                f"`new_value`{ev_part}。判定链原话见 `judge_reason`。")
    if change_status == "retiered":
        return (f"{head} —— `old_value` → `new_value`，按 `{src}` 重定级{ev_part}。"
                f"判定链原话见 `judge_reason`。")
    if change_status == "as-is":
        return f"{head} —— 改动前已有该字段，值逐字未变（{judge_reason}）"
    return f"{head} —— 该条账不可复算：{judge_reason}"


def load_locked_evidence() -> dict:
    """读回上一版账里**已经锁定**的 `evidence`（首次判定时写下的仪器读数）。

    ⚠️ 为什么需要它 —— 这是同一个缺陷的**第三处**：
      ① `old_value`（主控抓出）② `reason` 的人读列（主控抓出）③ **`evidence`**。
    三处的病因一样：**账上的列在重跑时被现算，于是现算出来的是「盘上的结论」而不是
    「当初的依据」**。实测重跑后 126 行的 `evidence` 变成
    `frontmatter venue_tier='non-paper' venue_source='...'` —— 那是结论，不是依据。
    修法：账自己**记住**第一版证据，后续重跑只在「这张卡还没锁过证据」时才现算。
    """
    if not BACKFILL_JSON.exists():
        return {}
    try:
        prev = json.loads(BACKFILL_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {r["slug"]: r.get("evidence", "") for r in prev.get("cards", [])}


def render_backfill(cards, verdicts, cov, *, head_fm: dict | None = None,
                    locked_evidence: dict | None = None) -> dict:
    """逐卡账：`old_value` **取自 git HEAD 快照**，`evidence` **锁定在首次判定时**。

    `head_fm` / `locked_evidence` 允许注入（selftest 用）；默认从 git / 上一版账取。
    """
    if locked_evidence is None:
        locked_evidence = load_locked_evidence()
    rows = []
    for c in cards:
        v = verdicts[c["slug"]]
        hf = head_fm.get(c["rel"]) if head_fm is not None else read_head_frontmatter(c["rel"])
        status, old = classify_change(hf, v["tier"] or "")
        rows.append({
            "slug": c["slug"],
            "rel": c["rel"],
            "domain": c["domain"],
            "old_value": old,
            "new_value": v["tier"],
            "changed": bool(old != (v["tier"] or "")) and status != "no-snapshot",
            "status": v["status"],           # 判定链内部状态（as-is/backfilled/explicit/...）
            "change_status": status,          # 账的三态（与判定链状态**不同轴**，都要留）
            "source": v["source"],
            # 证据锁定：已有锁定的就沿用（**不许在重跑时现算成盘上的结论**）；
            # 没有锁定过的（首次判定）才用当次算出的仪器读数。
            "evidence": locked_evidence.get(c["slug"]) or v.get("evidence", ""),
            "reason": compose_ledger_reason(
                status, v["source"],
                locked_evidence.get(c["slug"]) or v.get("evidence", ""), v["reason"]),
            "judge_reason": v["reason"],
            "venue": (c["fm"].get("venue") or "").strip(),
            "paper_id": (c["fm"].get("paper_id") or "").strip(),
            "low_confidence": c["slug"] in LOW_CONFIDENCE,
            "identity_mismatch": c["slug"] in IDENTITY_MISMATCH,
            "scope_mismatch": c["slug"] in SCOPE_MISMATCH,
        })
    dist = collections.Counter(r["change_status"] for r in rows)
    changed = [r for r in rows if r["changed"]]
    return {
        "_meta": {
            "generator": "build_venue_tiers.py",
            "phase": "PHASE6-S10",
            "snapshot_ref": SNAPSHOT_REF_RESOLVED,
            "old_value_source": (f"git show {SNAPSHOT_REF_RESOLVED}:<path> 的 venue_tier"
                                 f"（**不是写盘后读回**；SHA 已钉死，不随 HEAD 移动）"),
            "change_status_legend": {
                "as-is": "改动前已有该字段且逐字未变",
                "backfilled": "改动前没有该字段，本次新赋值",
                "retiered": "改动前有值，本次改了（规范化 / 逐卡裁决）",
                "no-snapshot": "git 取不到改动前版本 ⇒ 不可复算",
            },
        },
        "coverage": cov,
        "change_summary": {
            "total": len(rows),
            "changed": len(changed),
            "by_change_status": dict(dist),
            "low_confidence_and_changed": sum(1 for r in changed if r["low_confidence"]),
            "by_new_value": dict(collections.Counter(r["new_value"] for r in rows)),
            "by_source": dict(collections.Counter((r["source"] or "-").split("(")[0] for r in rows)),
        },
        "cards": rows,
    }


# --------------------------------------------------------------------------- #
# 写回                                                                          #
# --------------------------------------------------------------------------- #


#: 已知的**并发写入者**占用的 frontmatter 字段前缀。
#: PHASE6 实测：S13 正在给同一批卡写 `l3_*`，S10 写 `venue_tier`。
#: 两个写入者若不互相保留，就会静默抹掉对方的字段。
FOREIGN_FIELD_PREFIXES = ("l3_", "l1_", "l2_")

CONCURRENCY_CHECKPOINT = 20


def _should_write(slug: str, cur: str, tier: str, cur_source: str = "") -> bool:
    """这张卡要不要落盘？**唯一决策点**（T22 / T26 与变异 M14 都打在这里）。

    ⚠️ 踩过两次，两次都是同一形态 —— **「值对了」不等于「账全了」**：
      ① 首版只判 `cur == tier`，漏掉 `demo`→`preprint`（读时折叠，判定值恰等于折叠结果）
         ⇒ 盘上永远留着非规范取值。修法是判「cur 是不是规范档」而不是「等不等」。
      ② 修完 ① 之后**又漏了 18 张**：它们的 `venue_tier` 本来就合规（判定链走 `as-is` 分支，
         该分支当时不写 `venue_source`）⇒ 值对、**出处锚点缺**，而判据只看值。
         盘上实测 `venue_source` 只有 128/146。
    ⇒ 判据改成两条并列：**值要合规** *且* **锚点要在**，任一不满足就写。

    ⚠️ **判据的次序也是判据**：`slug in EXPLICIT_RETIERS` 必须**先判**。
    首版把「值对+锚点在 ⇒ 不动」写在最前，于是显式重定级的卡永远不会被写
    （T26 当场报红）—— 这正是本仓库反复记的那条：**放行条款要排在拦截条款之前还是之后，
    决定了这条判据到底测的是什么。**
    """
    if slug in EXPLICIT_RETIERS:
        return True                       # 逐卡裁决点名 ⇒ 必须落盘（哪怕看起来已经对了）
    if cur == tier and cur in CANONICAL_TIERS and cur_source:
        return False                      # 值合规 **且** 锚点在 ⇒ 不动
    if not cur:
        return True                       # 空值 ⇒ 回填
    if cur not in CANONICAL_TIERS:
        return True                       # 非规范取值（含轨道标记）⇒ 必须折叠落盘
    if not cur_source:
        return True                       # 值合规但**缺出处锚点** ⇒ 补锚点（不改正题）
    return slug in EXPLICIT_RETIERS       # 已有规范取值 ⇒ 只有显式重定级才改写


def apply_to_cards(cards, verdicts, *, dry=False) -> dict:
    """**逐卡 read-modify-write**：写入前一刻重新从磁盘读，只插入/替换自己那两个字段。

    ⚠️ 并发防护（由主控实测提出）：同一批卡上另有写入者（S13 写 `l3_*`）。
    因此：
      ① 每张卡的基准文本在**写入前一刻**从磁盘重读（不用批读时的内存副本）；
      ② 只做「插入 venue_tier/venue_source」或「替换 venue_tier 行」，
         其余行（含 `l3_*`）**逐字节带过**；
      ③ 每 `CONCURRENCY_CHECKPOINT` 张重扫目录，报告是否出现了别人的字段；
      ④ 写盘后立刻复读该卡，断言 `venue_tier` 在、别人的字段也在 —— 任一丢失即抛错。
    """
    changed, skipped, foreign_seen, lost = [], [], {}, []
    for n, c in enumerate(cards):
        v = verdicts[c["slug"]]
        v["slug"] = c["slug"]
        cur_disk_fm, _ = split_fm(c["path"].read_text(encoding="utf-8"))  # ① 写入前重读
        cur = (cur_disk_fm.get("venue_tier") or "").strip()
        if v["tier"] is None:
            skipped.append((c["slug"], v["reason"]))
            continue
        cur_source = (cur_disk_fm.get("venue_source") or "").strip()
        if not _should_write(c["slug"], cur, v["tier"], cur_source):
            if cur and cur != v["tier"]:
                skipped.append((c["slug"], f"已有规范取值 `{cur}`，非显式重定级 ⇒ 不改写"))
            continue
        if dry:
            changed.append((c["slug"], cur, v["tier"]))
            continue
        # ② 只插入/替换自己的两个字段
        before = set(k for k in cur_disk_fm if k.startswith(FOREIGN_FIELD_PREFIXES))
        _write_field(c, v["tier"], v["source"])
        # ④ 写盘后立刻复读
        after_fm, _ = split_fm(c["path"].read_text(encoding="utf-8"))
        after = set(k for k in after_fm if k.startswith(FOREIGN_FIELD_PREFIXES))
        if (after_fm.get("venue_tier") or "").strip() != v["tier"]:
            lost.append((c["slug"], "venue_tier 写进去了又不在"))
        if before - after:
            lost.append((c["slug"], f"抹掉了别人的字段 {sorted(before - after)}"))
        changed.append((c["slug"], cur, v["tier"]))
        # ③ 每 20 张重扫一次
        if (n + 1) % CONCURRENCY_CHECKPOINT == 0:
            seen = _scan_foreign_fields()
            foreign_seen[f"第 {n + 1} 张后"] = seen
    if not dry:
        foreign_seen["最终"] = _scan_foreign_fields()
    return {"changed": changed, "skipped": skipped,
            "foreign_seen": foreign_seen, "lost": lost}


def _scan_foreign_fields() -> dict:
    """重扫目录，数出并发写入者（`l3_*` 等）的覆盖率 —— 用来证明没抹掉别人的字段。"""
    total = with_any = 0
    per_field: collections.Counter = collections.Counter()
    for p in sorted(VAULT.rglob("Skill-*.md")):
        total += 1
        fm, _ = split_fm(p.read_text(encoding="utf-8"))
        hits = [k for k in fm if k.startswith(FOREIGN_FIELD_PREFIXES)]
        for k in hits:
            per_field[k] += 1
        if hits:
            with_any += 1
    return {"total": total, "with_foreign": with_any, "per_field": dict(per_field)}


def _write_field(card, tier: str, source: str) -> None:
    """写 `venue_tier`（+ `venue_source`）。

    **以磁盘当前内容为准**（不用内存副本）：只替换自己的那两个字段行、插入缺失的行，
    其余行**逐字节带过**（含并发写入者的 `l3_*`）。

    ⚠️ 判据：**「本卡是否已经有这个字段」必须从原 frontmatter 一次性判定**。
    首版按「写过了就不再插」来判，遇到「`venue` 行在前、`venue_tier` 行在后」的卡时
    会在 `venue` 前插入一份、又在 `venue_tier` 处再写一份 ⇒ **同一字段两份**（实测 4 张）。
    这一条已固化为 selftest 用例 T21。
    """
    text = card["path"].read_text(encoding="utf-8")
    lines = text.split("\n")
    end_idx = None
    for i, ln in enumerate(lines[1:], start=1):
        if ln.strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        raise RuntimeError(f"{card['slug']}: 找不到 frontmatter 结束行")
    head, rest = lines[: end_idx + 1], lines[end_idx + 1 :]

    # ① 一次性判定原 frontmatter 里有没有这两个字段
    has_tier = any(re.match(r"^venue_tier\s*:", ln) for ln in head)
    has_src = any(re.match(r"^venue_source\s*:", ln) for ln in head)

    out = []
    for ln in head:
        if re.match(r"^venue_tier\s*:", ln):
            if has_tier and not any(re.match(r"^venue_tier\s*:", x) for x in out):
                out.append(f"venue_tier: {tier}")
            continue  # 后续重复行一律丢弃（同字段只留一份）
        if re.match(r"^venue_source\s*:", ln):
            if has_src and not any(re.match(r"^venue_source\s*:", x) for x in out):
                out.append(f"venue_source: {source}")
            continue
        out.append(ln)

    if not has_tier:
        idx = _insert_anchor(out)
        out[idx:idx] = [f"venue_tier: {tier}", f"venue_source: {source}"]
    elif not has_src:
        # ⚠️ 锚点退化的真实场景：卡的首轮写入走的是「原值已合规」分支，只写了 `venue_tier`。
        #    此时**必须仍能补上 `venue_source`** —— 否则 7 张卡的出处永久是孤儿。
        idx = next((i for i, ln in enumerate(out) if re.match(r"^venue_tier\s*:", ln)), None)
        if idx is None:
            raise RuntimeError(f"{card['slug']}: has_tier 为真却找不到 venue_tier 行 —— 判据与代码不同步")
        out.insert(idx + 1, f"venue_source: {source}")

    text = "\n".join(out + rest)
    card["path"].write_text(text, encoding="utf-8")
    card["text"] = text


def _insert_anchor(head: list[str]) -> int:
    """`venue_tier` 插在哪：优先「`venue` 行之前」，否则「frontmatter 末尾（闭 `---` 之前）」。"""
    for i, ln in enumerate(head):
        if re.match(r"^venue\s*:", ln):
            return i
    # 退而求其次：任何溯源类字段之前
    for i, ln in enumerate(head):
        if re.match(r"^(evidence_grade|evidence_basis|paper_id|paper|l3_|l2_|l1_)\s*:", ln):
            return i
    return len(head) - 1  # 闭 `---` 之前


def _replace_block(doc_text: str | None, block: str) -> str:
    if doc_text is None:
        raise InputMissing("venue-whitelist.md 不存在")
    i = doc_text.find(DOC_BEGIN)
    j = doc_text.find(DOC_END)
    if i < 0 or j < 0:
        return doc_text.rstrip("\n") + "\n\n---\n\n" + block
    return doc_text[:i] + block + doc_text[j + len(DOC_END) :].lstrip("\n")


# --------------------------------------------------------------------------- #
# 夹具 / selftest / 变异                                                        #
# --------------------------------------------------------------------------- #


def _fixture():
    """最小夹具：一张干净卡落在接受 preprint 的域里。"""
    return {
        "cards": [
            {
                "slug": "Skill-A",
                "domain": "03-时间序列",
                "fm": {"paper_id": "2601.00001", "paper": "A Time Series Model", "venue_tier": "preprint"},
                "path": Path("/nonexistent"),
                "rel": "Skill-A",
                "text": "",
            }
        ],
        "arxiv": {"2601.00001": {"comments": "12 pages", "jref": "", "doi": "", "citation_title": "A Time Series Model"}},
        "crossref": {},
        "observed": [],
    }


def _whole_slugs():
    global _WHOLE_SLUGS_CACHE
    if _WHOLE_SLUGS_CACHE is None:
        _WHOLE_SLUGS_CACHE = []
        for p in sorted(VAULT.rglob("Skill-*.md")):
            text = p.read_text(encoding="utf-8")
            fm, _ = split_fm(text)
            _WHOLE_SLUGS_CACHE.append((p.stem, p.relative_to(VAULT).parts[0],
                                       (fm.get("paper_id") or "").strip()))
    return _WHOLE_SLUGS_CACHE


_WHOLE_SLUGS: list = []


def _whole_vault_slug_cards() -> list[dict]:
    """真 vault 的 slug + 域（frontmatter 只留 paper_id），供整库口径的台账用例使用。

    台账（UNRESOLVABLE / EXPLICIT_RETIERS / LOW_CONFIDENCE）是全库口径，
    夹具子集上跑会把「夹具只有 3 张卡」误报成「台账点名了不存在的卡」。
    """
    out = []
    global _WHOLE_SLUGS
    if not _WHOLE_SLUGS:
        _WHOLE_SLUGS = _whole_slugs()
    for slug, domain, pid in _WHOLE_SLUGS:
        fm = {"evidence_basis": "author-practice"}
        if pid:
            fm["paper_id"] = pid
        out.append({"slug": slug, "domain": domain, "fm": fm, "path": Path("/x"),
                    "rel": slug, "text": ""})
    out.append({"slug": "Skill-A", "domain": "03-时间序列",
                "fm": {"venue_tier": "CCF-A", "paper_id": "2601.00001"},
                "path": Path("/x"), "rel": "Skill-A", "text": ""})
    return out


_WHOLE_SLUGS_CACHE: list | None = None


def _run(cards, arxiv, crossref, observed, *, doc_text, product_bytes, whole_vault=False):
    return run_checks(cards, arxiv, crossref, observed, doc_text=doc_text,
                      product_bytes=product_bytes, whole_vault=whole_vault)


def _good_doc() -> str:
    """一份**真的能把 J5 判绿**的文档：§1 表 + §10 块。"""
    sec1 = "\n".join(
        ["## 1. 层级定义", ""] + [f"| `{t}` | 含义 |" for t in CANONICAL_TIERS] + ["", "## 2. 白名单", ""]
    )
    return sec1 + render_inventory_block()


def _neutralize_judge(src: str, judge: str) -> str:  # noqa: C901
    """把某条判据**改坏**（返回空 = 永不报警），返回改坏后的源码。

    两条路径，都要求函数**带返回类型注解**（没有注解就 fail loud ——
    「断言与代码不同步」必须是可见的失败，不是静默跳过）：
      · `-> list` 的独立判据（j1/j2/...）：整个函数体换成 `return []`
      · j456 / run_checks 里的键（J4/J5/J6）：把该键的值置空
    """
    # 形态 A：带 -> list 注解的独立函数
    pat = rf"^def {judge}\([^\n]*\) -> list:\n(?:.*?\n)*?(?=^def |^# ---|\Z)"
    m = re.search(pat, src, re.S | re.M)
    if m:
        sig = m.group(0).splitlines()[0]
        return src[: m.start()] + sig + "\n    return []\n\n\n" + src[m.end() :]

    # 形态 A2：带 -> bool 的决策点（与形态 A 同构）
    pat_b = rf"^def {judge}\([^\n]*\) -> bool:\n(?:.*?\n)*?(?=^def |^# ---|\Z)"
    m = re.search(pat_b, src, re.S | re.M)
    if m:
        sig = m.group(0).splitlines()[0]
        return src[: m.start()] + sig + "\n    return True\n\n\n" + src[m.end() :]

    # 形态 A3：带 -> tuple 的账函数
    pat_t = rf"^def {judge}\([^\n]*\) -> tuple\[[^\]]*\]:\n(?:.*?\n)*?(?=^def |^# ---|\Z)"
    m = re.search(pat_t, src, re.S | re.M)
    if m:
        sig = m.group(0).splitlines()[0]
        return src[: m.start()] + sig + '\n    return ("as-is", "")\n\n\n' + src[m.end() :]

    # 形态 B：run_checks 里的键
    m = re.search(rf'^        "{judge}": dd\["{judge}"\],$', src, re.M)
    if m:
        return src[: m.start()] + f'        "{judge}": [],' + src[m.end() :]

    raise RuntimeError(f"没法把判据 {judge} 改坏（断言与代码不同步？）")


def _run_mutation_matrix(judges: list[str]) -> tuple[int, list[str]]:
    """**真变异**：把每条判据在副本里改坏，跑副本的 `--selftest`。

    改坏后自检仍全绿 ⇒ 该断言是摆设（对应的 T 用例没在真的测它）。
    副作用：会在临时目录起子进程；用 `--no-mutations` 关闭。
    """
    src = Path(__file__).read_text(encoding="utf-8")
    caught, missed = [], []
    with tempfile.TemporaryDirectory() as td:
        for j in judges:
            try:
                mutated = _neutralize_judge(src, j)
            except RuntimeError as exc:
                missed.append(f"{j}（{exc}）")
                continue
            path = Path(td) / f"mut_{j}.py"
            path.write_text(mutated, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(path), "--selftest", "--no-mutations"],
                capture_output=True, text=True, timeout=300,
            )
            if proc.returncode != 0:
                caught.append(j)
            else:
                missed.append(f"{j}（改坏后自检仍全绿）")
    return len(caught), missed


def selftest(*, with_mutations: bool = True) -> int:
    f = _fixture()
    doc = _good_doc()
    pb = render_map_bytes()
    ok = True

    def expect(name, fn, judge, want_red):
        nonlocal ok
        try:
            probs = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {name}: 抛异常 {type(exc).__name__}: {exc}")
            ok = False
            return
        red = bool(probs.get(judge))
        if red == want_red:
            print(f"  ✅ {name}（{judge} {'红' if red else '绿'}）")
        else:
            print(f"  ❌ {name}: want_red={want_red} got={red} all={ {k: v for k, v in probs.items() if v} }")
            ok = False

    def clean(**over):
        return _run(
            over.get("cards", f["cards"]), over.get("arxiv", f["arxiv"]),
            over.get("crossref", f["crossref"]), over.get("observed", []),
            doc_text=over.get("doc_text", doc), product_bytes=over.get("product_bytes", pb),
        )["problems"]

    print("selftest · 单元用例：")
    # —— 反向控制：干净夹具必须全绿（没有它，全红也算通过）——
    p = clean()
    if any(p.values()):
        print(f"  ❌ T0 干净夹具不全绿：{ {k: v for k, v in p.items() if v} }")
        ok = False
    else:
        print("  ✅ T0 干净夹具全绿（反向控制）")

    # —— J1 ——
    bad = [dict(f["cards"][0], fm={"venue_tier": "S-TIER"})]
    expect("T1 表外取值 ⇒ J1 红", lambda: clean(cards=bad), "J1", True)
    expect("T1b 规范取值 ⇒ J1 绿", lambda: clean(), "J1", False)
    # —— J2 ——
    many = [{"slug": f"Skill-U{i}", "domain": "03-时间序列", "fm": {}, "path": Path("/x"),
             "rel": "x", "text": ""} for i in range(21)]
    many.append(dict(f["cards"][0], slug="Skill-A"))
    expect("T2 覆盖率 1/22 ⇒ J2 红", lambda: clean(cards=many), "J2", True)
    expect("T2b 覆盖率 1/1 ⇒ J2 绿", lambda: clean(), "J2", False)
    # —— J3 ——
    expect("T3 计数不自洽 ⇒ J3 红",
           lambda: {"J3": j3({"labelled": 2, "unlabelled": 2, "total": 3, "ratio": 2 / 3})}, "J3", True)
    expect("T3b cov>1 ⇒ J3 红",
           lambda: {"J3": j3({"labelled": 3, "unlabelled": 0, "total": 2, "ratio": 1.5})}, "J3", True)
    expect("T3c 正常 cov ⇒ J3 绿",
           lambda: {"J3": j3({"labelled": 1, "unlabelled": 0, "total": 1, "ratio": 1.0})}, "J3", False)
    # —— J4 ——
    expect("T4 文档块篡改 ⇒ J4 红",
           lambda: clean(doc_text=doc.replace("| `UTD24` |", "| `UTD24x` |")), "J4", True)
    expect("T4b 文档块缺失 ⇒ J4 红", lambda: clean(doc_text="# 无标记"), "J4", True)
    expect("T4c 文档块完好 ⇒ J4 绿", lambda: clean(), "J4", False)
    # —— J5 ——
    sec1_short = "\n".join(["## 1. 层级定义", "", "| `UTD24` | x |", "", "## 2. 白名单", ""])
    sec1_extra = "\n".join(["## 1. 层级定义", ""]
                          + [f"| `{t}` | x |" for t in CANONICAL_TIERS + ["CCF-C"]]
                          + ["", "## 2. 白名单", ""])
    expect("T5 §1 缺 6 档 ⇒ J5 红",
           lambda: clean(doc_text=sec1_short + render_inventory_block()), "J5", True)
    expect("T5b §1 多一档 ⇒ J5 红",
           lambda: clean(doc_text=sec1_extra + render_inventory_block()), "J5", True)
    expect("T5c §1 恰好 7 档 ⇒ J5 绿", lambda: clean(), "J5", False)
    # —— J6 ——
    dup = [dict(f["cards"][0], fm={"venue_tier": "preprint", "__dup__": ["venue_tier"]})]
    expect("T6 字段重复 ⇒ J6 红", lambda: clean(cards=dup), "J6", True)
    expect("T6b 无重复 ⇒ J6 绿", lambda: clean(), "J6", False)
    # —— J7 ——
    expect("T7 非规范取值无规则 ⇒ J7 红",
           lambda: _run(f["cards"], f["arxiv"], {}, [{"value": "Z-TIER", "where": "registry", "seen": 1}],
                        doc_text=doc, product_bytes=pb)["problems"], "J7", True)
    expect("T7b 三套词表取值都有规则 ⇒ J7 绿", lambda: clean(), "J7", False)
    # —— J8 ——
    mk = [{"slug": "Skill-P", "domain": "15-营销投放分析", "fm": {"venue_tier": "preprint"},
           "path": Path("/x"), "rel": "x", "text": ""}]
    expect("T8 域线未登记域 ⇒ 不红", lambda: clean(cards=mk), "J8", False)
    unk = [{"slug": "Skill-Q", "domain": "99-新域", "fm": {"evidence_basis": "author-practice"},
            "path": Path("/x"), "rel": "x", "text": ""}]
    expect("T8b 未登记技术域 ⇒ J8 红", lambda: clean(cards=unk), "J8", True)
    lv = _run(mk, f["arxiv"], {}, [], doc_text=doc, product_bytes=pb).get("line_violations") or []
    if len(lv) == 1 and lv[0]["slug"] == "Skill-P":
        print("  ✅ T8e 既有卡违反域线 ⇒ 登记为一等输出（不重判）")
    else:
        print(f"  ❌ T8e line_violations={lv}")
        ok = False
    keep_basis = DOMAIN_LINE["03-时间序列"]
    DOMAIN_LINE["03-时间序列"] = {"category": "方法论/工程类", "basis": "好"}
    expect("T8c 判定依据是形容词 ⇒ J8 红", lambda: clean(), "J8", True)
    DOMAIN_LINE["03-时间序列"] = keep_basis
    expect("T8d 域线完好 ⇒ J8 绿", lambda: clean(), "J8", False)
    # —— J9 ——
    expect("T9 产品副本缺失 ⇒ J9 红", lambda: clean(product_bytes=None), "J9", True)
    expect("T9b 产品副本漂移 ⇒ J9 红", lambda: clean(product_bytes=b'{"drift":1}'), "J9", True)
    expect("T9c 产品副本一致 ⇒ J9 绿", lambda: clean(), "J9", False)
    # —— J10 ——
    nom = [{"slug": "Skill-M", "domain": "05-推荐系统", "fm": {"paper_id": "2699.99999"},
            "path": Path("/x"), "rel": "x", "text": ""}]
    expect("T10 元数据缺该号 ⇒ J10 红", lambda: clean(cards=nom), "J10", True)
    nr = [{"slug": "Skill-N", "domain": "05-推荐系统", "fm": {"paper_id": "2601.00002"},
           "path": Path("/x"), "rel": "x", "text": ""}]
    expect("T10b 会名不认识 ⇒ J10 红",
           lambda: _run(nr, {"2601.00002": {"comments": "Accepted at the Symposium of Nowhere 2026",
                                            "jref": "", "doi": "", "citation_title": "X Y Z"}},
                        {}, [], doc_text=doc, product_bytes=pb)["problems"], "J10", True)
    expect("T10c 判定完成 ⇒ J10 绿", lambda: clean(), "J10", False)
    # —— J11 / J12 的台账用例：必须跑在**整库模式**下（台账是全库口径）——
    def whole(extra=()):
        """真 vault 的 slug 子集 + 受控卡，跑整库模式。"""
        vs = _whole_vault_slug_cards()
        vs.extend(extra)
        return _run(vs, f["arxiv"], f["crossref"], [], doc_text=doc, product_bytes=pb,
                    whole_vault=True)["problems"]

    keep_un = dict(UNRESOLVABLE)
    UNRESOLVABLE["Skill-ZZ"] = "构造样本"
    expect("T11 台账点名不存在的卡 ⇒ J11 红", lambda: whole(), "J11", True)
    UNRESOLVABLE.clear()
    UNRESOLVABLE["Skill-A"] = "构造样本：登记为不可回填但实际判得出"
    expect("T11b 台账过期（在册却判得出）⇒ J11 红", lambda: whole(), "J11", True)
    UNRESOLVABLE.clear()
    UNRESOLVABLE.update(keep_un)
    blank = [{"slug": "Skill-B", "domain": "03-时间序列", "fm": {}, "path": Path("/x"),
              "rel": "x", "text": ""}]
    expect("T11c 未标注且不在台账 ⇒ J11 红", lambda: whole(blank), "J11", True)
    expect("T11d 台账与实测一致 ⇒ J11 绿", lambda: whole(), "J11", False)
    # —— J12 ——
    keep_ex = dict(EXPLICIT_RETIERS)
    EXPLICIT_RETIERS["Skill-A"] = ("CCF-A", "构造样本：把已有规范值改判为别的档")
    expect("T12 已有规范值被改判 ⇒ J12 红", lambda: whole(), "J12", True)
    EXPLICIT_RETIERS.clear()
    EXPLICIT_RETIERS.update(keep_ex)
    EXPLICIT_RETIERS["Skill-A"] = ("CCF-A", "短")
    expect("T12b 裁决理由太短 ⇒ J12 红", lambda: whole(), "J12", True)
    EXPLICIT_RETIERS.clear()
    EXPLICIT_RETIERS.update(keep_ex)
    expect("T12c 只增不改成立 ⇒ J12 绿", lambda: whole(), "J12", False)
    # —— J13 ——
    mm = [{"slug": "Skill-C", "domain": "05-推荐系统",
           "fm": {"paper_id": "2601.00003", "paper": "A Totally Different Paper About Rocks"},
           "path": Path("/x"), "rel": "x", "text": ""}]
    expect("T13 身份错位未登记 ⇒ J13 红",
           lambda: _run(mm, {"2601.00003": {"comments": "", "jref": "", "doi": "",
                                            "citation_title": "Graph Neural Networks for Ranking"}},
                        {}, [], doc_text=doc, product_bytes=pb)["problems"], "J13", True)
    expect("T13b 身份对齐 / 已登记 ⇒ J13 绿", lambda: clean(), "J13", False)

    print("selftest · 判定链用例：")
    for val, want in (("demo", "preprint"), ("workshop", "preprint"), ("findings", "preprint"),
                      ("short-paper", "preprint")):
        v = judge_card({"slug": "x", "fm": {"venue_tier": val}}, {}, {})
        if v["tier"] == want:
            print(f"  ✅ T14 `{val}` → `{want}`")
        else:
            print(f"  ❌ T14 `{val}` → {v}")
            ok = False
    v = judge_card({"slug": "x", "fm": {"venue_tier": "top"}}, {}, {})
    if v["tier"] is None and v["status"] == "needs-resolution":
        print("  ✅ T15 `top` 不被静默折叠（needs-resolution）")
    else:
        print(f"  ❌ T15 `top` 被折叠成 {v}")
        ok = False
    a = classify_from_evidence({"slug": "x", "fm": {}},
                               {"comments": "Accepted at ACM SIGIR 2026 Industry Track", "jref": "", "doi": ""}, {})
    b = classify_from_evidence({"slug": "x", "fm": {}},
                               {"comments": "Accepted at ACM SIGIR 2026", "jref": "", "doi": ""}, {})
    if a["tier"] == "CCF-A" and b["tier"] == "CCF-A" and "主会自带轨道" in a["source"]:
        print("  ✅ T16 主会自带 track 与无轨道词同为 CCF-A（铁律 1 的仪器；降级见 T25）")
    else:
        print(f"  ❌ T16 {a} / {b}")
        ok = False
    c = classify_from_evidence({"slug": "x", "fm": {}}, {"comments": "", "jref": "", "doi": ""}, {})
    if c["tier"] == "preprint":
        print("  ✅ T17 无声明 ⇒ preprint")
    else:
        print(f"  ❌ T17 {c}")
        ok = False
    d = classify_from_evidence({"slug": "x", "fm": {}},
                               {"comments": "Accepted at the Symposium of Nowhere", "jref": "", "doi": ""}, {})
    if d["tier"] is None and d["status"] == "needs-resolution":
        print("  ✅ T18 会名不认识 ⇒ needs-resolution（不猜）")
    else:
        print(f"  ❌ T18 {d}")
        ok = False
    npv = judge_card({"slug": "x", "fm": {"evidence_basis": "author-practice"}}, {}, {})
    if npv["tier"] == "non-paper" and npv.get("source_class") == SOURCE_CLASS_NO_PAPER:
        print("  ✅ T19 自承无论文来源 ⇒ non-paper（来源类别登记）")
    else:
        print(f"  ❌ T19 {npv}")
        ok = False
    # 只有**声明**了才判 non-paper；没有声明的一律 unlabeled（不许拿 non-paper 充当兜底）
    npe = judge_card({"slug": "y", "fm": {}}, {}, {})
    if npe["tier"] is None and npe["status"] == "unlabeled":
        print("  ✅ T20 无声明 ⇒ unlabeled（**不拿 non-paper 兜底**）")
    else:
        print(f"  ❌ T20 {npe}")
        ok = False

    # T21 —— 写字段**不得**写出同名字段两份。
    # 这一条是实测撞出来的：首版按「写过了就不再插」判，遇到
    # 「`venue` 行在前、`venue_tier` 行在后」的卡时会在 `venue` 前插一份、
    # 又在 `venue_tier` 处再写一份 ⇒ 4 张卡出现**同一个字段两行**（J6 才抓到）。
    # 判据：写入后 `venue_tier` 行数 == 1 且 `venue_source` 行数 == 1，且值正确。
    with tempfile.TemporaryDirectory() as td:
        for name, body in (
            ("has_both", "---\ntitle: X\nvenue: KDD 2026\nvenue_tier: preprint\n---\n正文\n"),
            ("venue_before_tier", "---\ntitle: X\nvenue_tier: preprint\nvenue: KDD 2026\n---\n正文\n"),
            ("neither", "---\ntitle: X\npaper_id: 2601.00001\n---\n正文\n"),
            ("tier_last", "---\ntitle: X\npaper_id: 2601.00001\nvenue_tier: preprint\n---\n正文\n"),
        ):
            fp = Path(td) / f"{name}.md"
            fp.write_text(body, encoding="utf-8")
            fake = {"slug": f"T21-{name}", "path": fp, "text": body, "fm": {}, "domain": "03-时间序列", "rel": name}
            _write_field(fake, "CCF-A", "T21-sample")
            got = fp.read_text(encoding="utf-8")
            head = got.split("\n---", 1)[0]
            n_tier = len(re.findall(r"^venue_tier\s*:", head, re.M))
            n_src = len(re.findall(r"^venue_source\s*:", head, re.M))
            v_ok = "venue_tier: CCF-A" in head and "venue_source: T21-sample" in head
            if (n_tier, n_src, v_ok) == (1, 1, True):
                print(f"  ✅ T21 写字段不重复（{name}）")
            else:
                print(f"  ❌ T21 写字段重复/写错（{name}）：tier×{n_tier} src×{n_src} 值对={v_ok}\n{head}")
                ok = False

    # T22 —— `--apply` 的**落盘决策**：非规范取值必须被折叠**写到盘上**，不许只在读时折。
    # 判据：对同一张卡连续 apply 两次，第二次必须无改动（幂等）；且盘上只剩规范 7 档。
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "Skill-T22.md"
        body = "---\ntitle: T22\npaper_id: 2608.27006\nvenue: RecSys 2026 (Demo)\nvenue_tier: demo\n---\n正文\n"
        fp.write_text(body, encoding="utf-8")

        def _card():
            txt = fp.read_text(encoding="utf-8")
            fmx, _ = split_fm(txt)
            return {"slug": "T22", "path": fp, "text": txt, "fm": fmx,
                    "domain": "00-电商Agent", "rel": "T22"}

        r1 = apply_to_cards([_card()], judge_cards([_card()], {}, {}))
        disk1 = fp.read_text(encoding="utf-8")
        fmx, _ = split_fm(disk1)
        n1 = len(re.findall(r"^venue_tier\s*:", disk1.split("\n---", 1)[0], re.M))
        r2 = apply_to_cards([_card()], judge_cards([_card()], {}, {}))
        if (fmx.get("venue_tier") == "preprint" and n1 == 1 and len(r1["changed"]) == 1
                and len(r2["changed"]) == 0):
            print("  ✅ T22 `demo` 被折叠并**落盘**，且 apply 幂等")
        else:
            print(f"  ❌ T22 tier={fmx.get('venue_tier')!r} 行数={n1} "
                  f"第一次改={len(r1['changed'])} 第二次改={len(r2['changed'])}")
            ok = False

    # T23 —— 账的 `old_value` **必须来自改动前快照**，不许是写盘后读回。
    # 反后门用例（实测撞出来的）：首版 old 取自写盘后的 cards ⇒ 146 条 old==new，
    # 「改动数 = 0」，而有 121 条实际是新赋值。判据：注入一份快照，三态必须各就各位。
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "Skill-T23.md"
        body = "---\ntitle: T23\nvenue_tier: preprint\n---\n正文\n"
        fp.write_text(body, encoding="utf-8")
        fake_card = {"slug": "T23", "rel": "T23", "domain": "03-时间序列", "path": fp,
                     "text": body, "fm": {"venue_tier": "preprint"}}
        cases = [
            # (快照, 判定值, 期望 status, 期望 old)
            ({"venue_tier": "preprint"}, "preprint", "as-is", "preprint"),
            ({"venue_tier": "top"}, "CCF-B", "retiered", "top"),
            ({"title": "x"}, "non-paper", "backfilled", ""),
            (None, "preprint", "no-snapshot", ""),
        ]
        bad = []
        for snap, new, want_st, want_old in cases:
            st, old = classify_change(snap, new)
            if (st, old) != (want_st, want_old):
                bad.append(f"snap={snap} new={new} → ({st},{old!r})，期望 ({want_st},{want_old!r})")
        if not bad:
            print("  ✅ T23 账三态由**快照**决定（as-is / backfilled / retiered / no-snapshot）")
        else:
            print("  ❌ T23 " + "；".join(bad))
            ok = False
        # 反向控制：把快照换成「写盘后的值」（即首版的错法），三态必须塌掉
        st2, _ = classify_change({"venue_tier": "CCF-B"}, "CCF-B")
        if st2 == "as-is":
            print("  ✅ T23b 反向控制：拿写盘后的值当快照 ⇒ 三态塌成 as-is（正是首版的缺陷）")
        else:
            print(f"  ❌ T23b 反向控制没塌：{st2}")
            ok = False
        probs = j14([fake_card], {"T23": judge_card(fake_card, {}, {})}, {"T23": None})
        if probs:
            print(f"  ✅ T23c J14 抓到「取不到快照」：{probs[0][:48]}…")
        else:
            print("  ❌ T23c J14 没抓到 no-snapshot")
            ok = False

    # T25 —— **主会自带 track ≠ 独立 workshop**（实测缺陷：SIGIR Industry Track 曾被误降）。
    # 判据：两条真实 Comments 各判一次，档位必须**不同**且各自可复核。
    _mt = classify_from_evidence({"slug": "x", "fm": {}},
                                 {"comments": "Accepted at ACM SIGIR 2026 Industry Track. 18 pages",
                                  "jref": "", "doi": ""}, {})
    _ws = classify_from_evidence({"slug": "x", "fm": {}},
                                 {"comments": "post-proceedings of the ECML PKDD 2023 Workshop "
                                              "on Uplift Modeling", "jref": "", "doi": ""}, {})
    if _mt["tier"] == "CCF-A" and _ws["tier"] != "CCF-A" and "主会自带轨道" in _mt["source"]:
        print(f"  ✅ T25 主会自带 track 不降级（{_mt['tier']}）· 独立 workshop 降级（{_ws['tier']}）")
    else:
        print(f"  ❌ T25 主会 track={_mt['tier']}/{_mt['source']} workshop={_ws['tier']}/{_ws['source']}")
        ok = False
    # 反向控制：把主会 track 词换成 workshop 词，必须**降档**（否则说明切分没生效）
    _fake = classify_from_evidence({"slug": "x", "fm": {}},
                                   {"comments": "Accepted at the SIGIR 2026 Workshop on X",
                                    "jref": "", "doi": ""}, {})
    if _fake["tier"] != "CCF-A":
        print(f"  ✅ T25b 反向控制：换成 workshop 词 ⇒ 降为 {_fake['tier']}")
    else:
        print("  ❌ T25b 反向控制没降档 —— 切分没生效")
        ok = False

    # T26 —— 「值对了」不等于「账全了」：值合规但缺 `venue_source` 时**必须补锚点**。
    # 实测背景：补 `demo` 那条修完之后，仍**有 18 张**卡的锚点缺失而门禁全绿 ——
    # 因为判据只看值。判据：`_should_write` 在「值合规 + 锚点缺」时必须返回 True。
    _retier_slug = next(iter(EXPLICIT_RETIERS))
    cases26 = [
        ("Skill-X", "preprint", "preprint", "arxiv-abs", False, "值对+锚点在 ⇒ 不动"),
        ("Skill-X", "preprint", "preprint", "", True, "值对但缺锚点 ⇒ 补锚点"),
        ("Skill-X", "demo", "preprint", "TRACK_RULES", True, "非规范值 ⇒ 折叠落盘"),
        ("Skill-X", "", "preprint", "", True, "空值 ⇒ 回填"),
        (_retier_slug, "CCF-A", "CCF-A", "arxiv-abs", True, "值对+锚点在+显式重定级 ⇒ 写"),
        ("Skill-X", "CCF-A", "CCF-A", "arxiv-abs", False, "值对+锚点在+非重定级 ⇒ 不动"),
    ]
    bad26 = []
    for slug26, cur, new, src, want, desc in cases26:
        got = _should_write(slug26, cur, new, src)
        if got != want:
            bad26.append(f"{desc}: got={got} want={want}")
    if not bad26:
        print("  ✅ T26 `_should_write` 两条并列判据（值合规 **且** 锚点在）")
    else:
        print("  ❌ T26 " + "；".join(bad26))
        ok = False

    # T27 —— 账的 `evidence` 必须是**依据**，不许是盘上取值的复述。
    # 实测背景：同一个缺陷在账上出现过三次（old_value / reason 人读列 / evidence），
    # 所以判据必须**正面**规定 evidence 长什么样，而不是只在某一列上打补丁。
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "Skill-T27.md"
        body = "---\ntitle: T27\npaper_id: 2601.00001\nvenue_tier: preprint\n---\n正文\n"
        fp.write_text(body, encoding="utf-8")
        fc = {"slug": "T27", "rel": "T27", "domain": "03-时间序列", "path": fp,
              "text": body, "fm": {"paper_id": "2601.00001", "venue_tier": "preprint"}}
        vs = {"T27": judge_card(fc, {"2601.00001": {"comments": "12 pages", "jref": "", "doi": "",
                                                     "citation_title": "A Time Series Model"}}, {})}
        probs = j14([fc], vs, {"T27": {"venue_tier": "preprint"}})
        ev = vs["T27"]["evidence"]
        if not any("evidence" in x for x in probs) and ev.startswith("arxiv-abs"):
            print(f"  ✅ T27 账的 evidence 是依据（{ev[:44]}…）")
        else:
            print(f"  ❌ T27 evidence={ev[:60]!r} probs={probs}")
            ok = False
        # 反向控制：手写一条「复述型」证据，J14 必须打红
        bad_vs = {"T27": dict(vs["T27"], evidence="frontmatter venue_tier='preprint'（本次未改动）")}
        if any("evidence" in x for x in j14([fc], bad_vs, {"T27": {"venue_tier": "preprint"}})):
            print("  ✅ T27b 反向控制：复述型 evidence ⇒ J14 打红")
        else:
            print("  ❌ T27b 复述型 evidence 没打红")
            ok = False

    # T24 —— 映射表里不得有重复键（实测被咬过：同 slug 两条 EXPLICIT_RETIERS，
    # 前一条的详细理由被静默吃掉，而门禁全绿）。
    dup_keys = _assert_no_duplicate_keys()
    if not dup_keys:
        print("  ✅ T24 映射表无重复键")
    else:
        print(f"  ❌ T24 {dup_keys}")
        ok = False

    # —— 真变异：把每条判据在副本里改坏，副本的 selftest 必须变红 ——
    if with_mutations:
        judges = ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11", "J12", "J13"]
        # j456 合并判了 J4/J5/J6，逐个键改坏
        targets = ["j1", "j2", "j3", "J4", "J5", "J6", "j7", "j8", "j9", "j10", "j11", "j12",
                   "j13", "_should_write", "classify_change", "j14", "_assert_no_duplicate_keys"]
        print("selftest · 变异矩阵（把判据改坏 ⇒ 自检必须变红）：")
        n, missed = _run_mutation_matrix(targets)
        if missed:
            for m in missed:
                print(f"  ❌ {m} —— **漏网**（该断言是摆设）")
            ok = False
        print(f"  变异 {n}/{len(targets)} 抓住")

    print("selftest:", "✅ 全绿" if ok else "❌ 有失败")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PHASE6 S10 venue 词表统一与回填")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="只判据，不写盘")
    g.add_argument("--apply", action="store_true", help="写回卡片 frontmatter + 文档块 + 映射表")
    g.add_argument("--selftest", action="store_true", help="自检 + 变异测试")
    # ⚠️ `--sync-product` 刻意**不在**互斥组里：它是 `--apply` 的一个附带动作
    #   （写规范表 → 投递副本），不是第四种模式。
    ap.add_argument("--sync-product", action="store_true", help="与 --apply 同用：把规范表投递到产品侧")
    ap.add_argument("--no-mutations", action="store_true",
                    help="selftest 时跳过变异矩阵（变异矩阵会起子进程）")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest(with_mutations=not args.no_mutations)

    try:
        cards = read_cards()
        arxiv = read_json(ARXIV_META, "arXiv 元数据")
        crossref = read_json(CROSSREF_META, "Crossref 元数据") if CROSSREF_META.exists() else {}

        observed = collections.Counter()
        for c in cards:
            v = (c["fm"].get("venue_tier") or "").strip()
            if v:
                observed[("cards", v)] += 1
        if REGISTRY.exists():
            for r in json.loads(REGISTRY.read_text(encoding="utf-8")).get("records", []):
                observed[("registry", (r.get("venue_tier") or "").strip())] += 1
        observed = [{"value": v, "where": w, "seen": n} for (w, v), n in sorted(observed.items())]

        def snapshot():
            cs = read_cards()
            vs = judge_cards(cs, arxiv, crossref)
            return cs, vs

        if args.apply:
            cards0, v0 = snapshot()
            applied = apply_to_cards(cards0, v0)
            WHITELIST_MD.write_text(
                _replace_block(WHITELIST_MD.read_text(encoding="utf-8"), render_inventory_block()),
                encoding="utf-8",
            )
            MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
            MAP_JSON.write_bytes(render_map_bytes())
            if args.sync_product:
                PRODUCT_VENDORED.parent.mkdir(parents=True, exist_ok=True)
                tmp = PRODUCT_VENDORED.with_suffix(".json.tmp")
                tmp.write_bytes(render_map_bytes())
                shutil.move(str(tmp), str(PRODUCT_VENDORED))
            print(f"✍️  写回卡片 {len(applied['changed'])} 张；跳过 {len(applied['skipped'])} 张")
            for slug, old, new in applied["changed"]:
                print(f"     {slug}: {old or '(空)'} → {new}")
            print("--- 并发写盘防护 ---")
            for when, seen in applied["foreign_seen"].items():
                print(f"     {when}: 全库 {seen['total']} 张；带并发写入者字段的 "
                      f"{seen['with_foreign']} 张 {seen['per_field']}")
            if applied["lost"]:
                for slug, why in applied["lost"]:
                    print(f"     🔴 {slug}: {why}")
                raise RuntimeError(f"写盘丢失字段 {len(applied['lost'])} 处 —— 立刻停手，不许静默重写")

        cards, verdicts = snapshot()
        doc_text = WHITELIST_MD.read_text(encoding="utf-8") if WHITELIST_MD.exists() else None
        pbytes = PRODUCT_VENDORED.read_bytes() if PRODUCT_VENDORED.exists() else None
        res = run_checks(cards, arxiv, crossref, observed, doc_text=doc_text,
                         product_bytes=pbytes, whole_vault=True)
        cov = res["coverage"]
        dist = collections.Counter(v["tier"] or UNLABELED for v in verdicts.values())
        srcs = collections.Counter((v["source"] or "-").split("(")[0] for v in verdicts.values())

        if args.json_out or args.apply:
            BACKFILL_JSON.write_text(
                json.dumps(render_backfill(cards, verdicts, cov), ensure_ascii=False, indent=1),
                encoding="utf-8",
            )

        print("=" * 76)
        print("PHASE6 · S10 · venue 词表统一与回填")
        print("=" * 76)
        print(f"卡片总数            : {cov['total']}")
        print(f"已标注（分子）      : {cov['labelled']}")
        print(f"未标注（{UNLABELED}）   : {cov['unlabelled']}")
        print(f"覆盖率              : {cov['labelled']}/{cov['total']} = {cov['ratio']:.1%}"
              f"   （门槛 {COVERAGE_MIN:.0%} ≤ r ≤ {COVERAGE_MAX:.0%}）")
        print("档位分布            :")
        for k, v in sorted(dist.items(), key=lambda x: (-x[1], str(x[0]))):
            print(f"    {k:12} {v}")
        print("判定来源（venue_source，出处锚点）:")
        for k, v in sorted(srcs.items(), key=lambda x: -x[1]):
            print(f"    {k:34} {v}")
        foreign = _scan_foreign_fields()
        print(f"并发写入者字段（他人在写，本脚本只带过）: {foreign['with_foreign']}/{foreign['total']} "
              f"{ {k: v for k, v in sorted(foreign['per_field'].items())} }")
        rjd = registry_top_journal_domains()
        cards_by_dom = collections.Counter(c["domain"] for c in cards)
        print("登记层 UTD24/FT50 投放（**卡层看不见的口径**）:")
        for dom, rs in sorted(rjd.items()):
            n_exist = sum(1 for r in rs if r["card_exists"])
            print(f"    {dom:16} {len(rs)} 条记录 / 已出卡 {n_exist} / 本域在卡 {cards_by_dom.get(dom, 0)} 张"
                  f" | tier={'商科实证类' if DOMAIN_LINE.get(dom, {}).get('category') == '商科实证类' else '方法论/工程类'}")
        lv = res.get("line_violations") or []
        print(f"域线下的既有卡（入卡早于本线，登记不重判）: {len(lv)}")
        for v in lv[:12]:
            print(f"    {v['slug']:50} {v['domain']:14} tier={v['tier']}（原值 {v['old_value'] or '(空)'}）")
        print(f"低置信登记          : {len(LOW_CONFIDENCE)}")
        print(f"身份错位登记        : {len(IDENTITY_MISMATCH)}")
        print(f"不可回填台账        : {len(UNRESOLVABLE)}")

        probs = res["problems"]
        total_red = sum(len(v) for v in probs.values())
        print("-" * 76)
        if total_red == 0:
            print("✅ 全部判据通过")
            rc = 0
        else:
            print(f"🔴 判红 {total_red} 条：")
            for j in sorted(probs):
                for line in probs[j][:15]:
                    print(f"    {j:4} {line}")
                if len(probs[j]) > 15:
                    print(f"    {j:4} …还有 {len(probs[j]) - 15} 条")
            rc = 1

        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps({"coverage": cov, "distribution": dict(dist), "problems": probs,
                            "verdicts": verdicts}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            print(f"json → {args.json_out}")
        return rc
    except InputMissing as exc:
        print(f"🔴 输入没拿到（退出码 2，**这不是通过**）：{exc}")
        return 2
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"🔴 门禁内部错误（退出码 3，**这不是判红**）：{type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
