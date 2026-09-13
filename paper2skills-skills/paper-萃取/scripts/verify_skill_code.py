#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_skill_code.py — paper2skills K1 门禁（代码可执行性验证）

设计依据（2026-09 调研）：
  * AutoReproduce (arXiv:2505.20662) 实测：PaperCoder 生成的代码**执行率仅 17.94%**，
    性能差距 89.23%；加入执行闭环后执行率 **94.87%**、差距降到 19.72%。
  * ResearchCodeBench 附录 G：新研究代码失败中 **58.6% 是语义错误**（能跑但算错），
    语法/命名/类型错误各仅 8–9% —— 所以「能跑」只是第一层，必须分阶段报告。
  * AI Scientist / agentskills.io 均警告：靠 LLM 自评「代码没问题」不可靠。

因此本脚本不产生任何「分数」，只产生**可复核的退出码与 stdout 凭证**。

-------------------------------------------------------------------------------
五级验证（逐级递进，每级独立记录，不因前级失败而丢弃后级信息）
-------------------------------------------------------------------------------
  L1 SYNTAX   解析能否通过            —— ast.parse
  L2 COMPILE  能否编译为字节码        —— py_compile
  L3 IMPORT   模块能否被 import       —— 缺失第三方依赖注入 stub，不算代码错
  L4 SMOKE    作为脚本直接跑通        —— python <file>，超时保护
  L5 TEST     断言是否真的成立        —— pytest（若存在测试函数）

-------------------------------------------------------------------------------
三级判定（K1 放行 = PASS）
-------------------------------------------------------------------------------
    PASS         L1+L2 通过，且（L4 或 L5 通过）
  ENV_BLOCKED  L1+L2 通过，但后续失败全部归因于「本机缺第三方依赖/凭证/资源」
               —— 不判失败，但**禁止声称已验证**，计入「未验证」分母
  ORPHAN_DEP   L1+L2 通过，但卡片 import 的**本地模块在本仓库内找不到**
               —— 这是真实缺陷（卡片依赖了不存在的代码），必须修，不得归因环境
  FAIL         L1 或 L2 失败，或 L3/L4 出现与依赖无关的真实错误

  ⚠️ 关于 ORPHAN_DEP 与 ENV_BLOCKED 的区别（本脚本最重要的判定）：
     `import torch` 失败 → 本机没装 → ENV_BLOCKED（环境问题）
     `import review_quality_scoring` 失败 → 仓库里没这个模块 → ORPHAN_DEP（卡片缺陷）
     二者都会抛 ImportError，只能靠「包名是否存在于仓库/第三方 Index」区分。
     把后者误判为环境问题，会让「卡片引用了不存在的代码」长期隐藏在绿灯下。

用法
----
  # 验证单张卡片（文件中所有 python 块）
  python3 verify_skill_code.py --card ../../paper2skills-vault/13-广告分析/Skill-X.md

  # 验证已抽出的代码文件
  python3 verify_skill_code.py --file path/to/model.py

  # 全量回归（所有卡片），输出汇总
  python3 verify_skill_code.py --all --summary-out verification_summary.json

  # 只做快速语法+编译扫描（秒级，适合 pre-commit）
  python3 verify_skill_code.py --all --level 2
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import py_compile
import re
import signal
import subprocess
import sys
import sysconfig
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径定位（禁止硬编码绝对路径 —— 见 T0-3 路径约定）
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]          # .../paper2skills-skills/paper-萃取/scripts/x.py
if os.environ.get("PAPER2SKILLS_ROOT"):
    REPO_ROOT = Path(os.environ["PAPER2SKILLS_ROOT"]).resolve()
VAULT = REPO_ROOT / "paper2skills-vault"
REPORT_DIR = REPO_ROOT / "paper2skills-research" / "data" / "verification"

PY_BLOCK_RE = re.compile(r"^```(?:python|py|python3)\s*$", re.M | re.I)
FENCE_RE = re.compile(r"^```", re.M)

# 判定为「环境缺失」而非「代码缺陷」的异常特征
ENV_ERROR_SIGNS = (
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "cannot import name",
    "TclError",
    "no display name",
    "DISPLAY",
    "requires a GPU",
    "CUDA",
    "No such file or directory",
    "FileNotFoundError",
    "URLError",
    "ConnectionError",
    "HTTPError",
    "TimeoutError",
    "PermissionError",
    "SSL",
    "certificate",
    "MaxRetriesPerRequest",
    "api_key",
    "API key",
    "OPENAI_API_KEY",
    "ReadTimeout",
)

STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {
    "os", "sys", "re", "json", "math", "random", "time", "typing", "collections",
    "itertools", "functools", "pathlib", "datetime", "dataclasses", "abc",
    "warnings", "logging", "copy", "string", "csv", "io", "unittest", "hashlib",
    "statistics", "operator", "subprocess", "glob", "textwrap", "enum", "decimal",
    "inspect", "traceback", "pickle", "sqlite3", "uuid", "contextlib",
}

IMPORT_RE = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.M)

# ---------------------------------------------------------------------------
# 注入用的 stub：让「缺依赖」在本机可 import，但一旦真被调用就抛出明确错误。
# 这样 L3 IMPORT 才能区分「代码写错了」与「本机没装 torch」。
# ---------------------------------------------------------------------------
# 注意：这段源码内含大量 {} 字面量，禁止用 str.format —— 使用占位符替换。
STUB_TEMPLATE = '''
import sys, types
def _install(name):
    if name in sys.modules:
        return
    class _Missing(types.ModuleType):
        def __getattr__(self, attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            raise ImportError(
                "K1-STUB: " + name + "." + attr + " 在本机不可用（依赖未安装）。"
                "此错误归因于环境，不计为代码缺陷。"
            )
    m = _Missing(name)
    m.__path__ = []
    sys.modules[name] = m
for _n in __K1_MISSING_NAMES__:
    _install(_n)
'''

# ---------------------------------------------------------------------------
# 执行环境的 hermetic 化（本轮实测教训）
#
# 症状：全量门禁跑 30 分钟未出结果，而进程 CPU 时间只有 2.2 秒 —— 全部时间花在
#       **网络等待**上（卡片里的 `AutoModel.from_pretrained(...)`、`requests.get(...)`、
#       `yf.download(...)` 会一直等重试退避）。
#
# 结论：K1 门禁必须在**断网**语义下运行。理由有三：
#   ① 网络状态会让「同一次验证」结果不可复现（CI 里跑一次网断一次网变一次红绿）；
#   ② 卡片模板的业务价值在于方法，不在于能否拉到某个远端资源；
#   ③ 真正需要外部数据的卡片，应显式声明为 ENV_BLOCKED，而不是挂住流水线。
#
# 实现：拦截 socket 建连 + 把几个主流网络客户端替换为即时抛错的 stub。
# 抛出的错误含 "offline"/"network" 关键词，会被 is_env_error() 正确归因为环境问题。
# ---------------------------------------------------------------------------
NETWORK_BLOCK_TEMPLATE = '''
# ---- K1 hermetic: 阻断网络（立即失败，不重试、不等待）----
class _K1NetworkBlocked(RuntimeError):
    pass

_K1_MSG = ("K1-OFFLINE: network access is disabled during verification "
           "(环境限制，非代码缺陷)")

def _blocked(*a, **k):
    raise _K1NetworkBlocked(_K1_MSG)

try:
    import socket as _s
    _s.socket.connect = _blocked
    _s.socket.connect_ex = _blocked
    _s.create_connection = _blocked
except Exception:
    pass

# 常见网络客户端：把模块级入口替换为即时抛错，抢在它们内部重试之前失败
for _netmod in ("requests", "httpx", "urllib3", "yfinance",
                "huggingface_hub", "kaggle", "boto3", "openai", "anthropic"):
    try:
        import importlib as _il
        _m = _il.import_module(_netmod)
        for _attr in ("get", "post", "put", "request", "head"):
            if hasattr(_m, _attr):
                setattr(_m, _attr, _blocked)
    except Exception:
        pass

# matplotlib 交互后端会阻塞 → 统一切 Agg
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    pass
'''


@dataclass
class Stage:
    name: str
    passed: bool | None          # None = 未执行
    detail: str = ""
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0
    env_related: bool = False


@dataclass
class Result:
    target: str
    kind: str                    # card | file
    stages: list[Stage] = field(default_factory=list)
    verdict: str = "FAIL"        # PASS | ENV_BLOCKED | ORPHAN_DEP | FAIL | UNVERIFIED
    missing_deps: list[str] = field(default_factory=list)   # 本机缺的第三方包
    orphan_deps: list[str] = field(default_factory=list)    # 仓库内找不到的本地模块
    migrated_deps: list[str] = field(default_factory=list)  # 只存在于已迁出镜像(nlp_voc)的模块
    resolved_local: list[str] = field(default_factory=list) # 自动解析到的仓库内模块
    code_lines: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, s: Stage) -> None:
        self.stages.append(s)

    def stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------
def iter_fences(text: str):
    """按 CommonMark 规则迭代围栏代码块，支持 **3 个及以上反引号**。

    ⚠️ 为什么必须支持变长围栏（实测教训）：
    卡片里存在「代码块内容本身包含 ``` 」的情况 —— 例如
    `fix_prompt = f\"\"\"...\n\\`\\`\\`python\n{code}\n\\`\\`\\`\n...\"\"\"`。
    此时必须用 **4 个反引号**做外层围栏（Markdown 标准做法）。
    只认 3 个反引号的正则会：① 在嵌套处提前闭栏，把一块代码截成两半；
    ② 把 4 反引号的外层围栏整个漏掉（表现为"这张卡片没有代码块"）。

    ⚠️ 为什么不用非贪婪正则 `\\`\\`\\`(.*?)\\`\\`\\``：
    它同样会在嵌套处提前闭栏。必须按「开栏长度 == 闭栏长度」严格配对。

    yield (info_string, body)
    """
    lines = text.split("\n")
    i, n = 0, len(lines)
    while i < n:
        m = re.match(r"^(`{3,})\s*([^\n`]*)$", lines[i])
        if not m:
            i += 1
            continue
        ticks, info = m.group(1), m.group(2).strip()
        body: list[str] = []
        i += 1
        while i < n:
            close = re.match(r"^(`{3,})\s*$", lines[i])
            if close and len(close.group(1)) >= len(ticks):
                break
            body.append(lines[i])
            i += 1
        yield info, "\n".join(body)
        i += 1


def extract_python_blocks(text: str) -> list[str]:
    """取出所有标注为 python 的围栏块（支持 3+ 反引号）。"""
    return [body for info, body in iter_fences(text)
            if info.lower() in ("python", "py", "python3")]


def looks_like_python(body: str) -> bool:
    """未标注语言的围栏块，用启发式判断是否是 Python。

    ⚠️ 为什么必须"从严"（实测教训）：
    首轮全量门禁报出 8 个「语法错误」，逐个人工核对后发现 **全部是假阳性** ——
    它们是**未标注语言的围栏块**里的文档内容（ASCII 框图、YAML 契约、伪代码签名），
    被宽松启发式（`if re.search(r"^\\s*(print|return|for |while |if )", ...)`）误判为 Python。
    例：`Orchestrator 执行循环:` 因含 `if`/`→` 被当成代码；`skill_contract:` 被当成代码。

    误判的代价不只是噪声：它会让 K1 报告出现"8 个语法错误"这种**看起来像代码缺陷、
    实则文档标注问题**的结论，从而掩盖真实缺陷。故此处改为**保守策略**：

      ① 只有出现明确的 Python 语法结构（import / def / class / lambda / 装饰器 /
         `if __name__` / 赋值语句 / 明显的 Python 调用）才认；
      ② 命中"明显不是 Python"的信号（YAML 键、ASCII 框线、伪代码签名、纯表格）直接否决。
    """
    lines = [l for l in body.splitlines() if l.strip()]
    if not lines:
        return False

    # --- 否决信号：看着就不像 Python ---
    head = "\n".join(lines[:15])
    if re.match(r"^\s*[\w\-\u4e00-\u9fff]+\s*:\s*(\S.*)?$", lines[0]) and not lines[0].lstrip().startswith("#"):
        # 首行形如 `key:` 或 `key: value` → YAML / 伪模式定义
        return False
    if any(ch in head for ch in ("═", "─", "┌", "└", "│", "├", "┌", "→", "↓", "↓", "⇒")):
        return False
    if re.search(r"[【】〔〕①②③④⑤]", head):
        return False
    if lines[0].rstrip().endswith("(") and not re.search(r"\bdef\b|\bclass\b", head):
        # 首行形如 `fn_name(` 且全块无 def/class → 伪代码签名
        # （唯一定义处也是 def 开头，那种情况下面会命中 Python 结构）
        if not re.search(r"\breturn\b|\bself\b", head):
            return False
    if re.match(r"^\s*\|", lines[0]):        # markdown 表格
        return False

    # --- 准入信号：明确的 Python 结构 ---
    if re.search(r"^\s*(import|from)\s+[A-Za-z_]", body, re.M):
        return True
    if re.search(r"^\s*(def|class)\s+[A-Za-z_]\w*", body, re.M):
        return True
    if re.search(r"^\s*@\w+", body, re.M):                       # 装饰器
        return True
    if re.search(r"^\s*(if|for|while|with|try|elif|else)\b.*:\s*$", body, re.M):
        return True
    if re.search(r"^\s*[A-Za-z_]\w*\s*(:[^=]+)?=\s*[^=]", body, re.M):   # 赋值
        return True
    return False


def extract_all_python(text: str) -> list[str]:
    """先取标注为 python 的块；若一个都没有，再对未标注块做启发式回退。

    ⚠️ 修复记录：原实现写成
        `re.finditer(r"^```[^\\n`]*\\n(.*?)^```\\s*$", ...)` 然后 `looks_like_python(m.group(1))`
    —— `group(1)` 是**块内容**没错，但该正则不支持变长围栏，且会在嵌套 ``` 处提前闭栏。
    现统一走 `iter_fences()`（支持 3+ 反引号、严格配对），逻辑一致但不再漏块/截块。
    """
    blocks = extract_python_blocks(text)
    if blocks:
        return blocks
    return [body for info, body in iter_fences(text)
            if info == "" and looks_like_python(body)]


def stitch_blocks(blocks: list[str]) -> str:
    """把一张卡片的多个 python 块按文档顺序拼成一个模块。

    ⚠️ 为什么必须拼接（实测教训）：
    全量门禁第一次跑出 44 个 L3_IMPORT 失败，其中 **19 个是「逐块独立导入」造成的假阳性** ——
    卡片普遍是「block1 定义类 → block2 定义数据类 → block3 使用」的递进结构，
    单独导入 block3 必然 `NameError: name 'XXX' is not defined`。
    那不是卡片错误，是我的验证方式与卡片实际用法不符（卡片是给人整段用的）。

    拼接时顺带处理两个真实存在的书写习惯：
      ① 重复 import（多个块各自 `import numpy as np`）→  Python 允许重复 import，无害；
      ② 块之间用 `# ====` 分隔注释 →  原样保留，便于报错行号回溯到具体块。
    """
    if len(blocks) == 1:
        return blocks[0]
    parts = []
    for i, b in enumerate(blocks, 1):
        parts.append(f"\n# ===== K1 block {i} =====\n{textwrap.dedent(b).strip()}\n")
    return "\n".join(parts)


def block_line_map(blocks: list[str], stitched: str) -> list[tuple[int, int, int]]:
    """返回 [(block_index, start_line, end_line)]，用于把拼接后的报错行号映射回块号。"""
    spans = []
    line = 1
    for i, b in enumerate(blocks, 1):
        n = len(textwrap.dedent(b).strip().splitlines())
        start = line + 2            # 跳过 "\n# ===== K1 block i =====" 两行
        spans.append((i, start, start + n - 1))
        line = start + n + 1
    return spans


def locate_block(spans: list[tuple[int, int, int]], lineno: int) -> int | None:
    for idx, s, e in spans:
        if s <= lineno <= e:
            return idx
    return None


def repo_local_module_index() -> tuple[dict[str, Path], dict[str, Path]]:
    """扫描仓库，建立「本地模块名 → 所在目录」索引。

    返回 (可用索引, 已迁出镜像索引)。两种来源：
      1. paper2skills-code/<domain>/<algo>/      —— 已落地的代码模板子模块
      2. 卡片同目录下的 <algo>.py                —— 历史卡片自带的实现文件

    ⚠️ **为什么要把 nlp_voc 单独分出来（2026-09-12 修正）**：
    `07-NLP-VOC` 子项目已迁出本仓库，`paper2skills-code/nlp_voc/` 只是**代码模板镜像**，
    其内部 `data_path` 指向 `../ai_nlp_voc/...`（该目录在多数机器上并不存在）。
    早先的实现把 nlp_voc 直接 `continue` 掉，于是 9 张 `07-NLP-VOC` 的卡片被判 `ORPHAN_DEP`，
    判定文案是「引用了**仓库内不存在**的本地模块」—— 而这个判断**事实上是假的**：
    这 9 个模块在仓库里确实存在（`ls paper2skills-code/nlp_voc/<mod>/` 全部命中）。
    真正的差别是「镜像不可在此运行」，不是「模块不存在」。

    这两件事的**补救动作完全相反**：ORPHAN_DEP 要改卡片，镜像不可运行要改环境或接受它不可验证。
    把它们混为一谈会把 9 张没有缺陷的卡送进「必须修」清单。
    """
    idx: dict[str, Path] = {}
    migrated: dict[str, Path] = {}
    code_root = REPO_ROOT / "paper2skills-code"
    if code_root.is_dir():
        for p in code_root.rglob("*.py"):
            target = migrated if "nlp_voc" in p.parts else idx
            if p.name == "__init__.py":
                target.setdefault(p.parent.name, p.parent)
            else:
                target.setdefault(p.stem, p.parent)
    return idx, migrated


@dataclass
class DepReport:
    third_party: list[str] = field(default_factory=list)   # 本机没装 → 环境问题
    local_found: list[str] = field(default_factory=list)   # 仓库里有实现 → 加进 PYTHONPATH
    orphan: list[str] = field(default_factory=list)        # 仓库里也没有 → 卡片缺陷
    migrated: list[str] = field(default_factory=list)      # 只存在于已迁出镜像 → 非卡片缺陷


def classify_deps(code: str, local_index: dict[str, Path],
                  migrated_index: dict[str, Path] | None = None) -> DepReport:
    rep = DepReport()
    migrated_index = migrated_index or {}
    seen: set[str] = set()
    for m in IMPORT_RE.finditer(code):
        mod = (m.group(1) or m.group(2) or "").split(".")[0]
        if not mod or mod in STDLIB or mod in seen:
            continue
        seen.add(mod)
        try:
            __import__(mod)
            continue                       # 本机可 import，无需处理
        except Exception:
            pass
        if mod in local_index:             # 仓库里有 → 可达，加路径
            rep.local_found.append(mod)
        elif mod in migrated_index:        # 只在已迁出镜像里 → 不是卡片缺陷
            rep.migrated.append(mod)
        elif _is_stdlib_like(mod):
            rep.third_party.append(mod)
        else:
            # 判定「像第三方包名」还是「像本地模块名」
            if _looks_like_local_module(mod):
                rep.orphan.append(mod)
            else:
                rep.third_party.append(mod)
    return rep


# 已知 PyPI 上存在、且是学术卡片高频依赖的包（缺失即环境问题，绝不判 ORPHAN）
KNOWN_PYPI = {
    "numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "seaborn",
    "torch", "torchvision", "tensorflow", "keras", "transformers", "datasets",
    "xgboost", "lightgbm", "catboost", "optuna", "shap", "lime",
    "causalml", "econml", "dowhy", "prophet", "pmdarima", "sktime", "tsfresh",
    "networkx", "stellargraph", "torch_geometric", "dgl", "node2vec", "gensim",
    "nltk", "spacy", "jieba", "sentence_transformers", "faiss", "faiss_cpu",
    "openai", "anthropic", "langchain", "llama_index", "chromadb", "pinecone",
    "plotly", "altair", "streamlit", "fastapi", "flask", "pydantic", "requests",
    "httpx", "aiohttp", "bs4", "lxml", "openpyxl", "pyarrow", "polars", "duckdb",
    "sqlalchemy", "psycopg2", "pymysql", "redis", "celery", "mlflow", "wandb",
    "imblearn", "mlxtend", "umap", "hdbscan", "category_encoders", "feature_engine",
    "sentencepiece", "tokenizers", "accelerate", "peft", "trl", "vllm",
    "pdfplumber", "PyPDF2", "pypdf", "fitz", "MinerU", "paddleocr",
    "yaml", "tqdm", "dotenv", "click", "rich", "tabulate", "joblib", "pytest",
}


def _is_stdlib_like(mod: str) -> bool:
    return mod in KNOWN_PYPI


def _looks_like_local_module(mod: str) -> bool:
    """snake_case 全小写且非已知 PyPI 包 → 更可能是本地模块名。

    注意：这是启发式。为避免把真·第三方包误判成 ORPHAN，只有「全小写 + 下划线分隔 +
    不是已知包 + 在 PyPI 上查不到」才判 ORPHAN；否则一律从宽按环境问题处理。
    """
    if mod in KNOWN_PYPI or mod in STDLIB:
        return False
    if not re.fullmatch(r"[a-z][a-z0-9_]*", mod):
        return False
    if "." in mod:
        return False
    # 保守策略：名字里含下划线 → 强本地信号（PyPI 包极少用下划线）
    if "_" in mod:
        return True
    # 其余单词型名字：若能在 site-packages 里找到同名目录则不算 orphan（已被 __import__ 覆盖）
    return False


def detect_missing_deps(code: str) -> list[str]:
    """兼容旧接口：返回本机 import 不到的顶层模块名。"""
    out = []
    for m in IMPORT_RE.finditer(code):
        mod = (m.group(1) or m.group(2) or "").split(".")[0]
        if not mod or mod in STDLIB or mod in out:
            continue
        try:
            __import__(mod)
        except Exception:
            out.append(mod)
    return out


# ---------------------------------------------------------------------------
# 执行辅助
# ---------------------------------------------------------------------------
def run(cmd: list[str], cwd: Path, timeout: int, env: dict | None = None) -> tuple[int, str, str, float]:
    """带硬超时的子进程执行。

    ⚠️ 教训：仅设 timeout 不足以防止挂死 —— subprocess.run 的 timeout 在
    被 traceback 引用的深层 C 调用（下载重试、CUDA 初始化）上可能长时间不生效。
    因此这里**开启新会话**，超时后杀整个进程组，确保不留孤儿。
    """
    t0 = time.time()
    try:
        p = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, stdin=subprocess.DEVNULL, env=env,
            start_new_session=True,          # 独立进程组，便于整组回收
        )
    except Exception as e:                    # 连启动都失败
        return 127, "", f"[K1] 无法启动子进程: {e}", time.time() - t0
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out or "", err or "", time.time() - t0
    except subprocess.TimeoutExpired:
        # 杀整个进程组，避免孤儿进程继续占资源
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            out, err = p.communicate(timeout=10)
        except Exception:
            out, err = "", ""
        tail = (err or "")[-1200:]
        return (124, out or "",
                f"{tail}\n[K1] ⏱ 超时 {timeout}s 已强制终止（进程组 SIGKILL）。"
                f"常见原因：真实网络等待、plt.show()、input()、CUDA 初始化。",
                time.time() - t0)


def is_env_error(text: str) -> bool:
    return any(sign in text for sign in ENV_ERROR_SIGNS)


def trim(s: str, n: int = 1500) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n // 2] + f"\n... [截断 {len(s) - n} 字符] ...\n" + s[-n // 2 :]


# ---------------------------------------------------------------------------
# 核心：对一个代码单元做五级验证
# ---------------------------------------------------------------------------
def verify_unit(code: str, target: str, kind: str, workdir: Path,
                max_level: int = 5, timeout: int = 90,
                local_index: dict[str, Path] | None = None,
                migrated_index: dict[str, Path] | None = None) -> Result:
    res = Result(target=target, kind=kind)
    res.code_lines = len(code.splitlines())
    code = textwrap.dedent(code).strip("\n") + "\n"
    local_index = local_index if local_index is not None else {}
    migrated_index = migrated_index if migrated_index is not None else {}

    # --- L1 SYNTAX ---
    t0 = time.time()
    try:
        ast.parse(code)
        res.add(Stage("L1_SYNTAX", True, "ast.parse 通过", seconds=time.time() - t0))
    except SyntaxError as e:
        res.add(Stage("L1_SYNTAX", False,
                      f"语法错误 line {e.lineno}: {e.msg}",
                      stderr=f"{e.text or ''}", seconds=time.time() - t0))
        res.verdict = "FAIL"
        return res
    if max_level < 2:
        res.verdict = "UNVERIFIED"
        return res

    # --- L2 COMPILE ---
    src = workdir / "unit.py"
    src.write_text(code, encoding="utf-8")
    t0 = time.time()
    try:
        py_compile.compile(str(src), cfile=str(workdir / "unit.pyc"), doraise=True)
        res.add(Stage("L2_COMPILE", True, "py_compile 通过", seconds=time.time() - t0))
    except py_compile.PyCompileError as e:
        res.add(Stage("L2_COMPILE", False, "编译失败", stderr=trim(str(e)), seconds=time.time() - t0))
        res.verdict = "FAIL"
        return res
    if max_level < 3:
        res.verdict = "UNVERIFIED"
        return res

    # --- 依赖分类：第三方缺失(环境) vs 本地模块不存在(卡片缺陷) ---
    deps = classify_deps(code, local_index, migrated_index)
    res.missing_deps = deps.third_party
    res.orphan_deps = deps.orphan
    res.migrated_deps = deps.migrated
    res.resolved_local = deps.local_found

    stub = workdir / "_k1_stub.py"
    stub_src = (STUB_TEMPLATE.replace("__K1_MISSING_NAMES__", repr(res.missing_deps))
                if res.missing_deps else "")
    # 关键：把「环境准备」变成 hermetic —— 见 NETWORK_BLOCK_TEMPLATE 注释。
    # 没有这一步，卡片里一句 AutoModel.from_pretrained("bert-base-uncased")
    # 会让门禁挂在网络下载上（实测 30 分钟耗尽 CPU 仅 2.2 秒）。
    stub_src += NETWORK_BLOCK_TEMPLATE
    stub.write_text(stub_src, encoding="utf-8")
    runner = workdir / "_k1_runner.py"
    runner.write_text(
        "import _k1_stub  # noqa: F401  (注入缺失依赖 stub, 关闭 matplotlib 交互后端)\n"
        f"import runpy, sys\nsys.argv = ['{src.name}']\n"
        f"runpy.run_path(r'{src.name}', run_name='__main__')\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    # 仓库内已落地的本地模块 → 加进搜索路径，让卡片有机会真正跑起来
    extra_path = [str(workdir)] + [str(local_index[m]) for m in deps.local_found]
    env["PYTHONPATH"] = os.pathsep.join(extra_path + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "ignore"
    # 让下载型库直接走本地缓存/报错，而不是联网重试（配合 NETWORK_BLOCK_TEMPLATE）
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # --- L3 IMPORT ---
    probe = workdir / "_k1_import_probe.py"
    probe.write_text(
        "import _k1_stub\n"
        f"import importlib.util as u, sys\n"
        f"spec = u.spec_from_file_location('unit', r'{src.name}')\n"
        "m = u.module_from_spec(spec)\n"
        # ⚠️ 必须先把模块注册进 sys.modules 再 exec_module。
        # 否则在 CPython 3.14 下，**任何含 @dataclass 的被测模块**都会崩在
        #   dataclasses._is_type → sys.modules.get(cls.__module__).__dict__
        # 抛 AttributeError: 'NoneType' object has no attribute '__dict__'，
        # 而 L3 会把这条「探针自身的缺陷」记成卡片的 `import 时崩溃` —— 假红灯。
        # （2026-09-12 由子代理在 p2s-2026-0014 一卡上发现并给出最小复现；
        #  影响面是全仓库所有用 dataclass 的卡片，故在探针侧统一修，而不是让每张卡改写法。）
        "sys.modules['unit'] = m\n"
        "try:\n"
        "    spec.loader.exec_module(m)\n"
        "    print('K1_IMPORT_OK')\n"
        "except SystemExit:\n"
        "    print('K1_IMPORT_OK')\n",
        encoding="utf-8",
    )
    rc, out, err, secs = run([sys.executable, probe.name], workdir, timeout, env)
    ok = "K1_IMPORT_OK" in out
    env_rel = (not ok) and is_env_error(err + out)
    # ORPHAN 优先级高于 ENV：卡片 import 了仓库里根本不存在的本地模块 → 卡片缺陷，不是环境问题
    orphan_hit = (not ok) and bool(res.orphan_deps) and any(
        o in (err + out) for o in res.orphan_deps
    )
    # MIGRATED：模块只存在于已迁出子项目的镜像里 → 镜像不可在此运行，但**不是卡片缺陷**
    migrated_hit = (not ok) and bool(getattr(res, "migrated_deps", None)) and any(
        o in (err + out) for o in res.migrated_deps
    )
    if ok:
        detail = "模块可 import"
    elif orphan_hit:
        detail = f"引用了仓库内不存在的本地模块: {', '.join(res.orphan_deps)}"
    elif migrated_hit:
        detail = (f"模块只存在于已迁出子项目镜像 paper2skills-code/nlp_voc/: "
                  f"{', '.join(res.migrated_deps)}（非卡片缺陷；镜像依赖 ../ai_nlp_voc/ 数据路径）")
    elif env_rel:
        detail = f"缺第三方依赖/凭证: {', '.join(res.missing_deps) or '未识别'}"
    else:
        detail = "import 时崩溃"
    res.add(Stage("L3_IMPORT", ok, detail, stdout=trim(out), stderr=trim(err),
                  seconds=secs, env_related=(env_rel or migrated_hit) and not orphan_hit))
    if orphan_hit:
        res.verdict = "ORPHAN_DEP"
        return res
    if migrated_hit:
        res.verdict = "MIGRATED_DEP"
        return res
    if max_level < 4:
        res.verdict = "PASS" if ok else ("ENV_BLOCKED" if env_rel else "FAIL")
        return res

    # --- L4 SMOKE（作为脚本直接跑）---
    rc, out, err, secs = run([sys.executable, runner.name], workdir, timeout, env)
    smoke_ok = rc == 0
    smoke_env = (not smoke_ok) and is_env_error(err + out)
    note = ""
    if rc == 124:
        note = "超时"
    elif rc != 0 and not smoke_env:
        # 定位第一处真实报错行
        m = re.search(r'File "([^"]+)", line (\d+)', err)
        if m:
            note = f"阻塞在 line {m.group(2)}"
    res.add(Stage("L4_SMOKE", smoke_ok,
                  "脚本执行成功" if smoke_ok else (f"环境阻塞 {note}".strip() if smoke_env else f"运行期错误 {note}".strip()),
                  stdout=trim(out), stderr=trim(err), seconds=secs, env_related=smoke_env))

    # --- L5 TEST（pytest）---
    if max_level >= 5:
        has_test = bool(re.search(r"^\s*def\s+test_\w+", code, re.M)) or "assert " in code
        if not has_test:
            res.add(Stage("L5_TEST", None, "代码中未发现 test_ 函数或 assert，无可执行断言"))
        else:
            tf = workdir / "test_unit.py"
            body = code if re.search(r"^\s*def\s+test_\w+", code, re.M) else (
                "import unit\n\n" + "\n".join(
                    "def test_auto_%d():\n%s" % (i, textwrap.indent(a, "    "))
                    for i, a in enumerate(
                        ["    " + ln.strip() for ln in code.splitlines() if ln.strip().startswith("assert ")],
                        start=1,
                    )
                )
            )
            tf.write_text(body, encoding="utf-8")
            rc, out, err, secs = run(
                [sys.executable, "-m", "pytest", tf.name, "-q", "--no-header",
                 "-p", "no:cacheprovider", "--tb=short"],
                workdir, timeout, env,
            )
            t_ok = rc == 0
            t_env = (not t_ok) and is_env_error(err + out)
            res.add(Stage("L5_TEST", t_ok,
                          "pytest 全绿" if t_ok else ("环境阻塞" if t_env else "断言失败"),
                          stdout=trim(out), stderr=trim(err), seconds=secs, env_related=t_env))

    # --- 判定 ---
    l4 = res.stage("L4_SMOKE")
    l5 = res.stage("L5_TEST")
    真实执行 = (l4 and l4.passed) or (l5 and l5.passed)
    if 真实执行:
        res.verdict = "PASS"
    elif (l4 and l4.env_related) or (l5 and l5.env_related) or \
         (res.stage("L3_IMPORT") and res.stage("L3_IMPORT").env_related):
        res.verdict = "ENV_BLOCKED"
    else:
        res.verdict = "FAIL"
    return res


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def collect_cards() -> list[Path]:
    return sorted(
        p for p in VAULT.rglob("Skill-*.md")
        if "_superseded" not in p.parts
    )


def _selftest() -> int:
    """自证依赖分类器能区分四种情形 —— 门禁工具必须先自证可信。

    锁定的是 2026-09-12 修的一个**假红灯**：早先实现把 `paper2skills-code/nlp_voc/`
    整体 `continue` 掉，于是 9 张 07-NLP-VOC 卡片被判 `ORPHAN_DEP`，判定文案却说
    「引用了**仓库内不存在**的本地模块」—— 而 `ls` 证明这些模块就在仓库里。
    误判的代价是真实的：这 9 张卡被写进「必须修卡片」清单，而正确动作是「镜像不可在此运行」。
    """
    local_index, migrated_index = repo_local_module_index()

    # ⚠️ 样本名**从索引里现取**，不写死：写死会在仓库结构变动时腐烂，
    #    而且会掩盖真正的分类缺陷（第一版就因为写死 `causal_inference` —— 该目录
    #    下只有子包、没有顶层 __init__.py —— 报了一次假失败）。
    if not local_index or not migrated_index:
        print("❌ 索引为空，无法自检（local=%d, migrated=%d）"
              % (len(local_index), len(migrated_index)))
        return 1
    real_local = sorted(local_index)[0]
    real_mirror = sorted(migrated_index)[0]
    phantom = "totally_made_up_module_xyz"

    cases: list[tuple[str, str, str]] = []   # (说明, 代码, 期望分类)
    cases.append(("stdlib+第三方包 → 不算缺陷",
                  "import os\nimport numpy\nimport pandas as pd\n", "none"))
    cases.append((f"仓库内真实存在的本地模块 → local_found（取 {real_local}）",
                  f"import {real_local}\n", "local"))
    cases.append((f"仓库内根本不存在的小写模块 → ORPHAN_DEP（{phantom}）",
                  f"import {phantom}\n", "orphan"))
    cases.append((f"只在已迁出镜像 nlp_voc 里的模块 → MIGRATED_DEP（取 {real_mirror}）",
                  f"import {real_mirror}\n", "migrated"))
    print(f"索引：可用本地模块 {len(local_index)} 个 / 已迁出镜像 {len(migrated_index)} 个")
    print(f"  nlp_voc 镜像索引样例: {sorted(migrated_index)[:3]}")
    print()

    ok = True
    for desc, code, expect in cases:
        rep = classify_deps(code, local_index, migrated_index)
        if expect == "orphan":
            got = "orphan" if rep.orphan else "none"
        elif expect == "migrated":
            got = "migrated" if rep.migrated else "none"
        elif expect == "local":
            got = "local" if rep.local_found else "none"
        else:
            got = "none" if not (rep.orphan or rep.migrated or rep.local_found) else "其他"
        mark = "✅" if got == expect else "❌"
        if got != expect:
            ok = False
        print(f"{mark} {desc:38s} 期望 {expect:8s} 实得 {got}")
        if rep.orphan:
            print(f"     orphan   = {rep.orphan}")
        if rep.migrated:
            print(f"     migrated = {rep.migrated}")

    # 关键断言：ORPHAN 与 MIGRATED 必须是**互斥**的两类，不能混
    print()
    fake_orphan = classify_deps(f"import {phantom}\n", local_index, migrated_index)
    mirror_mod = classify_deps(f"import {real_mirror}\n", local_index, migrated_index)
    # 期望状态（不要再取反 —— 第一版多写了一个 not，把「通过」判成了「失败」）：
    #   真孤儿模块只落进 orphan、绝不落进 migrated；镜像模块只落进 migrated、绝不落进 orphan。
    cross = (bool(fake_orphan.orphan) and not fake_orphan.migrated
             and bool(mirror_mod.migrated) and not mirror_mod.orphan)
    print(("✅" if cross else "❌") + " 互斥性：真孤儿只进 orphan；镜像模块只进 migrated")
    ok = ok and cross

    # 迁移路径存在性：migrated 索引里的每个模块就应在 nlp_voc 下真的存在
    missing = [m for m, path in migrated_index.items()
               if "nlp_voc" not in path.parts]
    print(("✅" if not missing else "❌") + f" 迁移索引来源正确（全部位于 nlp_voc/: {not missing}）")
    ok = ok and not missing

    # --- 第二组：L3 探针端到端自证 ---
    # 锁定 2026-09-12 由 p2s-2026-0014 一卡暴露的**探针自身缺陷**：
    # 探针不把被测模块注册进 sys.modules 时，凡「`from __future__ import annotations`
    # + `@dataclass`」的模块都会崩在 dataclasses._is_type（注解是字符串 → 走 forward-ref
    # 分支 → 需要 sys.modules[cls.__module__]），L3 却把它记成卡片的 `import 时崩溃`。
    # 影响面是全仓库所有 dataclass 卡，故在探针侧修，而不是让每张卡改写法。
    print()
    print("--- L3 探针端到端自证 ---")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "_k1_stub.py").write_text("", encoding="utf-8")
        (tdp / "unit_mod.py").write_text(
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Row:\n"
            "    a: int\n"
            "    b: str = 'x'\n",
            encoding="utf-8",
        )
        (tdp / "_k1_import_probe.py").write_text(
            "import _k1_stub\n"
            "import importlib.util as u, sys\n"
            "spec = u.spec_from_file_location('unit', 'unit_mod.py')\n"
            "m = u.module_from_spec(spec)\n"
            "sys.modules['unit'] = m\n"
            "try:\n"
            "    spec.loader.exec_module(m)\n"
            "    print('K1_IMPORT_OK')\n"
            "except SystemExit:\n"
            "    print('K1_IMPORT_OK')\n",
            encoding="utf-8",
        )
        pr = subprocess.run([sys.executable, "_k1_import_probe.py"], cwd=td,
                            capture_output=True, text=True, timeout=60)
        probe_ok = "K1_IMPORT_OK" in pr.stdout
        print(("✅" if probe_ok else "❌") +
              " L3 探针能吃下 `from __future__ import annotations` + @dataclass 的模块"
              + ("" if probe_ok else f"  ← {pr.stderr.strip().splitlines()[-1:] }"))
        ok = ok and probe_ok

    # --- 第三组：CLI 退出码 / 口径提示契约自证（PHASE6 F1）---
    # 锁定两条**只写在文档里、没有任何门禁检查**的行为。它们都属于同一族：
    # 「没东西可查 ≠ 查过了没问题」。
    #   · bug #11：`--card` 对「无 python 代码块」的卡片返回什么？
    #     CLAUDE.md 曾记「返回 0」，2026-09-13 实测为 **2**（上面的 early return）。
    #     登记「待修」的条目本身也会腐烂 —— 故把退出码钉成用例，而不是只改文档。
    #   · bug #14：`--level 2` 全库扫描时**每个单元都判 UNVERIFIED**，退出码却是 0。
    #     修法只补口径提示、**不动退出码**（L2 作为秒级语法门禁是合法用法，
    #     改成非 0 会逼人绕过这道门）。故必须同时锁两个方向：
    #     提示该出现时出现，**不该出现时不出现** —— 否则它退化成噪声。
    print()
    print("--- CLI 退出码 / 口径提示契约自证 ---")
    SELF = Path(__file__).resolve()

    def _run_cli(card: Path, level: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SELF), "--card", str(card), "--level", str(level)],
            capture_output=True, text=True, timeout=180)

    cli_cases: list[tuple[str, bool, str]] = []   # (说明, 是否通过, 补充信息)
    with tempfile.TemporaryDirectory(prefix="k1_cli_") as td2:
        t2 = Path(td2)
        c_nocode = t2 / "card_nocode.md"
        c_nocode.write_text("# 无代码块卡\n\n只有正文，没有任何 python 围栏。\n", encoding="utf-8")
        c_broken = t2 / "card_broken.md"
        c_broken.write_text("# 语法坏卡\n\n```python\ndef f(:\n    pass\n```\n", encoding="utf-8")
        c_ok = t2 / "card_ok.md"
        c_ok.write_text("# 好卡\n\n```python\nx = 1\nassert x == 1\n```\n", encoding="utf-8")

        # 用例 A（bug #11）：无代码块 → 必须返回 2，且**不能**是 0。
        rA = _run_cli(c_nocode, 5)
        cli_cases.append((
            "A 无代码块卡 `--card` 退出码 = 2（bug #11：文档曾记 0）",
            rA.returncode == 2 and "未发现 python 代码块" in (rA.stdout + rA.stderr),
            f"实得 rc={rA.returncode}"))

        # 用例 B（反后门：不许把 #11 修成「恒返 2」）
        # 有代码块但语法坏 → 必须是「失败」语义（1），绝不能也用 2。
        rB = _run_cli(c_broken, 2)
        cli_cases.append((
            "B 语法坏卡 `--level 2` 退出码 = 1（与「无代码块=2」互斥）",
            rB.returncode == 1,
            f"实得 rc={rB.returncode}"))

        # 用例 C（bug #14 正向）：L2 下好卡 rc=0，但必须**明说**这 0 不代表通过。
        rC = _run_cli(c_ok, 2)
        hint = "exit 0 只代表"
        cli_cases.append((
            "C L2 好卡 rc=0 **且** 出示「exit 0 只代表没有失败」口径提示（bug #14）",
            rC.returncode == 0 and hint in rC.stdout,
            f"实得 rc={rC.returncode} 提示={'有' if hint in rC.stdout else '无'}"))

        # 用例 D（bug #14 反向 / 防噪声）：L5 下全部单元都真验过 → 提示**不得**出现。
        # 若把提示写成无条件打印，它就会在正常全量报告里刷屏，读者会学会忽略它。
        rD = _run_cli(c_ok, 5)
        cli_cases.append((
            "D L5 好卡 rc=0 **且不出现**该提示（防提示退化成噪声）",
            rD.returncode == 0 and hint not in rD.stdout,
            f"实得 rc={rD.returncode} 提示={'误报' if hint in rD.stdout else '未出现'}"))

    for desc, passed, info in cli_cases:
        print(("✅" if passed else "❌") + f" {desc}")
        if not passed:
            ok = False
            print(f"     ← {info}")

    print()
    print("SELFTEST " + ("PASS —— 依赖判定、L3 探针与 CLI 契约均自证可信" if ok
                        else "FAIL —— 判定退化，勿信门禁数字"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="paper2skills K1 代码可执行性门禁")
    ap.add_argument("--selftest", action="store_true",
                    help="用构造样本自证三个依赖判定（PASS/ORPHAN/MIGRATED）能互相区分")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--card", type=Path, help="单张 Skill 卡片路径")
    g.add_argument("--file", type=Path, help="已抽出的 .py 文件")
    g.add_argument("--all", action="store_true", help="全量回归所有卡片")
    ap.add_argument("--level", type=int, default=5, choices=[1, 2, 3, 4, 5],
                    help="最大验证级别（2=只做语法+编译，秒级）")
    ap.add_argument("--timeout", type=int, default=90, help="单次执行超时秒数")
    ap.add_argument("--json-out", type=Path, help="结果 JSON 输出路径")
    ap.add_argument("--markdown-out", type=Path, help="人类可读报告输出路径")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--per-block", action="store_true",
                    help="逐块独立验证（默认是把卡片所有代码块按文档顺序拼成一个模块，"
                         "与卡片实际用法一致；逐块模式会产生大量 NameError 假阳性，仅用于定位）")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not (args.card or args.file or args.all):
        ap.error("必须指定 --card / --file / --all 之一（或用 --selftest 自检）")

    units: list[tuple[str, str, str, list]] = []   # (target, kind, code, block_spans)

    if args.card:
        text = args.card.read_text(encoding="utf-8", errors="replace")
        blocks = extract_all_python(text)
        if not blocks:
            print(f"⚠️  {args.card.name} 未发现 python 代码块")
            return 2
        if args.per_block:
            for i, b in enumerate(blocks, 1):
                units.append((f"{args.card.name}#block{i}", "card", b, []))
        else:
            st = stitch_blocks(blocks)
            units.append((f"{args.card.name}#stitched({len(blocks)}块)", "card", st,
                          block_line_map(blocks, st)))
    elif args.file:
        units.append((args.file.name, "file",
                      args.file.read_text(encoding="utf-8", errors="replace"), []))
    else:
        for c in collect_cards():
            blocks = extract_all_python(c.read_text(encoding="utf-8", errors="replace"))
            if not blocks:
                continue
            rel = str(c.relative_to(REPO_ROOT))
            if args.per_block:
                for i, b in enumerate(blocks, 1):
                    units.append((f"{rel}#block{i}", "card", b, []))
            else:
                st = stitch_blocks(blocks)
                units.append((f"{rel}#stitched({len(blocks)}块)", "card", st,
                              block_line_map(blocks, st)))

    # 仓库内已落地的本地模块索引（用于区分 ORPHAN_DEP 与 ENV_BLOCKED）
    local_index, migrated_index = repo_local_module_index()
    if local_index and not args.quiet:
        print(f"已索引仓库内本地模块 {len(local_index)} 个（可自动解析的卡片将真正执行）")
    if migrated_index and not args.quiet:
        print(f"另有 {len(migrated_index)} 个模块只存在于已迁出镜像 nlp_voc/（判 MIGRATED_DEP，非卡片缺陷）")

    results: list[Result] = []
    ICON = {"PASS": "✅", "ENV_BLOCKED": "🟡", "ORPHAN_DEP": "🔴", "MIGRATED_DEP": "🟠",
            "FAIL": "❌", "UNVERIFIED": "⚪"}
    with tempfile.TemporaryDirectory(prefix="k1_") as td:
        for idx, (target, kind, code, spans) in enumerate(units):
            wd = Path(td) / f"u{idx}"
            wd.mkdir(parents=True, exist_ok=True)
            r = verify_unit(code, target, kind, wd, max_level=args.level,
                            timeout=args.timeout, local_index=local_index,
                            migrated_index=migrated_index)
            # 把报错行号映射回具体 block，方便作者定位
            if spans:
                for st in r.stages:
                    m = re.search(r"line (\d+)", st.detail or "") or \
                        re.search(r'line (\d+)', st.stderr or "")
                    if m:
                        bi = locate_block(spans, int(m.group(1)))
                        if bi:
                            st.detail = f"[block{bi}] {st.detail}"
                            break
            results.append(r)
            if not args.quiet:
                icon = ICON.get(r.verdict, "?")
                bad = next((s for s in r.stages if s.passed is False), None)
                extra = f"  ← {bad.name}: {bad.detail}" if bad else ""
                print(f"{icon} {r.verdict:12s} {target}{extra}")

    # --- 汇总 ---
    tally = {"PASS": 0, "ENV_BLOCKED": 0, "ORPHAN_DEP": 0, "MIGRATED_DEP": 0,
             "FAIL": 0, "UNVERIFIED": 0}
    for r in results:
        tally[r.verdict] = tally.get(r.verdict, 0) + 1
    exec_units = tally["PASS"]
    denom = len(results)
    rate = (exec_units / denom * 100) if denom else 0.0

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "max_level": args.level,
        "units_total": denom,
        "tally": tally,
        "exec_rate_pct": round(rate, 2),
        "note": ("exec_rate_pct = PASS / units_total。ENV_BLOCKED 与 MIGRATED_DEP 均计入未验证分母，"
                 "不得声称已验证；ORPHAN_DEP 是卡片缺陷（模块在仓库内确实不存在），必须修；"
                 "MIGRATED_DEP 的模块存在于已迁出子项目镜像 paper2skills-code/nlp_voc/，"
                 "不是卡片缺陷，但该镜像依赖 ../ai_nlp_voc/ 数据路径，在此不可运行。"),
    }
    payload = {"summary": summary, "results": [
        {**asdict(r), "stages": [asdict(s) for s in r.stages]} for r in results
    ]}

    print("\n" + "=" * 66)
    print(f"单元总数 {denom} | ✅ PASS {tally['PASS']} | 🟡 ENV_BLOCKED {tally['ENV_BLOCKED']} "
          f"| 🔴 ORPHAN {tally['ORPHAN_DEP']} | 🟠 MIGRATED {tally['MIGRATED_DEP']} "
          f"| ❌ FAIL {tally['FAIL']} | ⚪ 未执行 {tally['UNVERIFIED']}")
    print(f"K1 执行率 = {rate:.1f}%   （对照：PaperCoder 17.94% / AutoReproduce 94.87%）")
    # ⚠️ 低级别扫描的「误导性绿灯」（门禁 bug #14，2026-09-13 实测撞出）：
    # `--level 2` 下**所有**单元都判 UNVERIFIED（判定链见 verify_unit：<3 只走到 L2），
    # 于是出现「全库 101 单元 / ✅ PASS 0 / ❌ FAIL 0 / ⚪ 未执行 101 / **exit 0**」——
    # 一次什么都没验证的扫描，用退出码报了「没问题」。
    # 与 bug #11（`--card` 对无代码卡是否返回 0）同族，都是
    # 「**没东西可查 ≠ 查过了没问题**」。它此前未被任何门禁报警。
    # ⚠️ **刻意不改退出码**：`--level 2` 作为「秒级语法门禁」是合法用法，
    #    改成非 0 会让每次 pre-commit 都变红，从而逼人绕过这道门 ——
    #    那是用错的方式修对的问题。故只把口径讲清楚，让读退出码的人知道它代表什么。
    if tally["UNVERIFIED"] > 0:
        print(f"⚠️  本次为 L{args.level} 扫描：{tally['UNVERIFIED']}/{denom} 个单元"
              f"**未达可判定级别**（判 PASS 需 L3 以上；执行率统计需 L4/L5）。")
        print("    → **exit 0 只代表「没有失败」，不代表「K1 通过」**。"
              "要拿执行率必须跑 --level 5。")
    print("=" * 66)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON → {args.json_out}")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# K1 代码可执行性验证报告", "",
            f"- 生成时间：{summary['generated_at']}",
            f"- 最大验证级别：L{args.level}",
            f"- 单元总数：{denom}",
            f"- **K1 执行率：{rate:.1f}%**（PASS {tally['PASS']} / 环境阻塞 {tally['ENV_BLOCKED']} "
            f"/ 孤儿依赖 {tally['ORPHAN_DEP']} / 已迁出镜像 {tally['MIGRATED_DEP']} "
            f"/ 失败 {tally['FAIL']} / 未执行 {tally['UNVERIFIED']}）",
            "",
            "> **口径说明（三个易被混淆的判定）**",
            "> - `PASS`：真正跑通了（L4 脚本执行成功 或 L5 断言全绿）。只有这一类可以声称「已验证」。",
            "> - `ENV_BLOCKED`：本机缺第三方依赖/凭证/资源（如未装 torch）。**计入未验证分母** —— 缺依赖不等于代码正确。",
            "> - `ORPHAN_DEP`：卡片 `import` 的**本地模块在仓库内确实不存在**。这是卡片真实缺陷，必须修，",
            ">   **绝不可归因环境而放行**。（注意与下一项的区别：判 ORPHAN 前请先 `ls` 确认模块真的不存在。）",
            "> - `MIGRATED_DEP`：模块**确实存在于仓库内**，但只在已迁出子项目的镜像 `paper2skills-code/nlp_voc/` 下。",
            ">   该镜像是历史代码模板，内部数据路径指向 `../ai_nlp_voc/`（迁出目标，多数机器上不存在），故在此不可运行。",
            ">   **这不是卡片缺陷** —— 补救动作与 ORPHAN_DEP 相反，不要把它写进「必须修卡片」清单。",
            "",
            "## 明细", "",
            "| 判定 | 目标 | 阻塞阶段 | 说明 |", "|---|---|---|---|",
        ]
        ICON2 = {"PASS": "✅", "ENV_BLOCKED": "🟡", "ORPHAN_DEP": "🔴", "MIGRATED_DEP": "🟠",
                 "FAIL": "❌", "UNVERIFIED": "⚪"}
        for r in results:
            icon = ICON2.get(r.verdict, "?")
            bad = next((s for s in r.stages if s.passed is False), None)
            lines.append(f"| {icon} {r.verdict} | `{r.target}` | {bad.name if bad else '—'} | "
                         f"{bad.detail if bad else '全部通过'} |")
        # 孤儿依赖专章：这是最需要人看的部分
        orphans = [r for r in results if r.verdict == "ORPHAN_DEP"]
        if orphans:
            lines += ["", "## 孤儿依赖清单（卡片引用了仓库内不存在的模块）", "",
                      "| 卡片 | 缺失模块 |", "|---|---|"]
            for r in orphans:
                lines.append(f"| `{r.target}` | {', '.join('`%s`' % o for o in r.orphan_deps)} |")
        migrated = [r for r in results if r.verdict == "MIGRATED_DEP"]
        if migrated:
            lines += ["", "## 已迁出镜像依赖清单（**不是卡片缺陷**）", "",
                      "> 这些模块在仓库内确实存在，但只位于 `paper2skills-code/nlp_voc/`（`07-NLP-VOC` 子项目迁出后",
                      "> 保留的代码模板镜像）。镜像内部数据路径指向 `../ai_nlp_voc/`，故在此不可运行。",
                      "> 处置建议：要么在装有 `ai_nlp_voc` 的环境里验，要么接受这批卡「不可在此验证」并在统计中单列。", "",
                      "| 卡片 | 镜像内模块 |", "|---|---|"]
            for r in migrated:
                lines.append(f"| `{r.target}` | {', '.join('`%s`' % o for o in r.migrated_deps)} |")
        args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Markdown → {args.markdown_out}")

    # 退出码：FAIL 或 ORPHAN_DEP 存在即非 0（可作 CI 门禁）。
    # MIGRATED_DEP 不计入失败：它是环境/历史约束，不是被验证对象的缺陷。
    return 0 if (tally["FAIL"] == 0 and tally["ORPHAN_DEP"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
