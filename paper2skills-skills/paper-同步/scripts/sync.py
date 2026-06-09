#!/usr/bin/env python3
"""
paper2skills 同步脚本
同步 Skill 卡片和代码模板到各个平台
支持状态追踪
"""

import os
import shutil
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import requests

# 配置路径：动态推导项目根目录（脚本位于 paper2skills-skills/paper-同步/scripts/sync.py，
# 因此 parents[3] 即为 paper_to_skills 项目根）。
# 可通过环境变量 PAPER2SKILLS_ROOT 覆盖。
BASE_DIR = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.domains import DomainInferenceError, infer_domain_or_raise, load_domain_registry

VAULT_DIR = BASE_DIR / "paper2skills-vault"
CODE_DIR = BASE_DIR / "paper2skills-code"
SKILLS_DIR = BASE_DIR / "paper2skills-skills"
STATUS_FILE = VAULT_DIR / "07-资源库" / "sync_status.json"

DOMAIN_REGISTRY = load_domain_registry(BASE_DIR)
DOMAINS = {entry.key: entry.vault_dir for entry in DOMAIN_REGISTRY.entries}


# 从 Skill 卡片文件名提取领域
def skill_stem(skill_name):
    return skill_name[:-3] if skill_name.endswith(".md") else skill_name


def status_key(skill_name):
    return f"{skill_stem(skill_name)}.md"


def extract_domain(skill_name):
    """从现有 vault 文件系统推断领域。无法可靠推断时抛错。"""
    return infer_domain_or_raise(BASE_DIR, skill_name)


def load_status():
    """加载同步状态"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_status(status):
    """保存同步状态"""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)


# 内存中累积的状态变更，由 main() 在结尾批量 flush 一次（P2-1 优化）
_PENDING_STATUS = None


def _ensure_pending_status():
    global _PENDING_STATUS
    if _PENDING_STATUS is None:
        _PENDING_STATUS = load_status()
    return _PENDING_STATUS


def update_status(skill_name, target, success=True, error=None):
    """更新同步状态（内存缓存，由 main() 批量写盘）"""
    status = _ensure_pending_status()
    skill_name = status_key(skill_name)

    if skill_name not in status:
        status[skill_name] = {}

    status[skill_name][target] = {
        "synced": success,
        "timestamp": datetime.now().isoformat(),
        "error": error
    }

    return status


def flush_status():
    """将累积的状态变更一次性写入磁盘"""
    global _PENDING_STATUS
    if _PENDING_STATUS is not None:
        save_status(_PENDING_STATUS)
        _PENDING_STATUS = None


def sync_vault(skill_name, domain=None):
    """同步到 Obsidian Vault"""
    stem = skill_stem(skill_name)
    if domain is None:
        domain = extract_domain(stem)
    else:
        domain = infer_domain_or_raise(BASE_DIR, stem, domain)

    # 源文件可能是 skill 输出目录或 vault 已有文件
    possible_sources = [
        SKILLS_DIR / "output" / f"{stem}.md",
        VAULT_DIR / "output" / f"{stem}.md",
    ]

    source = None
    for s in possible_sources:
        if s.exists():
            source = s
            break

    if source is None:
        print(f"源文件不存在")
        update_status(stem, "vault", False, "source file not found")
        return False

    target_dir = VAULT_DIR / DOMAIN_REGISTRY.vault_dir_for(domain)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stem}.md"

    shutil.copy2(source, target)
    print(f"已同步到 Vault: {target}")
    update_status(stem, "vault", True)
    return True


def sync_code(skill_name, domain=None):
    """同步到 GitHub 代码仓库"""
    stem = skill_stem(skill_name)
    if domain is None:
        domain = extract_domain(stem)
    else:
        domain = infer_domain_or_raise(BASE_DIR, stem, domain)

    # 模块名从 skill 名转换
    module_name = stem.lower().replace("skill-", "").replace("-", "_")

    possible_source_dirs = [
        SKILLS_DIR / "output" / module_name,
        CODE_DIR / domain / module_name,
    ]

    source_dir = None
    for s in possible_source_dirs:
        if s.exists() and s.is_dir():
            source_dir = s
            break

    if source_dir is None:
        print(f"源目录不存在")
        update_status(stem, "github", False, "source directory not found")
        return False

    target_dir = CODE_DIR / domain / module_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件
    files_copied = 0
    for file in ["__init__.py", "model.py", "example.py"]:
        source = source_dir / file
        if source.exists():
            target = target_dir / file
            shutil.copy2(source, target)
            print(f"已同步到代码仓库: {target}")
            files_copied += 1

    if files_copied == 0:
        update_status(stem, "github", False, "no files to copy")
        return False

    update_status(stem, "github", True)
    return True


def sync_feishu(skill_name):
    """同步到飞书（webhook 优先读环境变量 PAPER2SKILLS_FEISHU_WEBHOOK）"""
    stem = skill_stem(skill_name)
    # P2-3 安全修复：环境变量优先，文件 fallback
    webhook_url = os.environ.get("PAPER2SKILLS_FEISHU_WEBHOOK", "").strip()

    if not webhook_url:
        webhook_path = Path.home() / ".paper2skills" / "feishu_webhook"
        if webhook_path.exists():
            with open(webhook_path, 'r') as f:
                webhook_url = f.read().strip()

    if not webhook_url:
        update_status(stem, "feishu", False, "not configured")
        print("飞书 webhook 未配置（未设置环境变量 PAPER2SKILLS_FEISHU_WEBHOOK 也无配置文件），跳过")
        return False

    # 读取 skill 内容
    skill_content = ""
    for domain_dir in VAULT_DIR.iterdir():
        if domain_dir.is_dir():
            skill_file = domain_dir / f"{stem}.md"
            if skill_file.exists():
                skill_content = skill_file.read_text()
                break

    if not skill_content:
        update_status(stem, "feishu", False, "skill content not found")
        return False

    try:
        payload = {
            "msg_type": "text",
            "content": {"text": f"Skill 同步: {skill_name}\n\n{skill_content[:500]}..."}
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            update_status(stem, "feishu", True)
            print(f"已同步到飞书")
            return True
        else:
            update_status(stem, "feishu", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        update_status(stem, "feishu", False, str(e))
        return False


def show_status(skill_name):
    """显示同步状态"""
    status = load_status()
    skill_name = status_key(skill_name)

    if skill_name not in status:
        print(f"未找到 {skill_name} 的同步记录")
        return

    print(f"=== {skill_name} 同步状态 ===")
    for target, info in status[skill_name].items():
        if target.startswith("_"):  # 跳过 _README 等元字段
            continue
        status_str = "✅" if info["synced"] else "❌"
        print(f"{status_str} {target}: {info.get('timestamp', 'n/a')}")
        if info.get("error"):
            print(f"   错误: {info['error']}")


def main():
    parser = argparse.ArgumentParser(description="paper2skills 同步脚本")
    parser.add_argument("--skill", help="技能名称")
    parser.add_argument("--domain", help="领域名称（可选，自动检测）")
    parser.add_argument("--target", default="vault,github", help="目标：vault,github,feishu,all")
    parser.add_argument("--status", action="store_true", help="查看同步状态")

    args = parser.parse_args()

    if args.status:
        if args.skill:
            show_status(args.skill)
        else:
            status = load_status()
            print("=== 所有同步状态 ===")
            for skill_name, targets in status.items():
                if skill_name.startswith("_"):  # 跳过 _README 元字段
                    continue
                print(f"\n{skill_name}:")
                for target, info in targets.items():
                    if target.startswith("_"):
                        continue
                    status_str = "✅" if info["synced"] else "❌"
                    print(f"  {status_str} {target}")
        return 0

    if not args.skill:
        print("请指定 --skill 参数")
        return 1

    # P0-4: 实现 --target all 派发逻辑
    raw_targets = args.target.split(",") if args.target else ["vault", "github"]
    if "all" in [t.strip().lower() for t in raw_targets]:
        targets = ["vault", "github", "feishu"]
    else:
        targets = [t.strip() for t in raw_targets if t.strip()]

    print(f"同步 {args.skill} 到 {targets}...")

    success = True

    try:
        try:
            if "vault" in targets:
                success &= sync_vault(args.skill, args.domain)

            if "github" in targets:
                success &= sync_code(args.skill, args.domain)

            if "feishu" in targets:
                success &= sync_feishu(args.skill)
        except DomainInferenceError as e:
            print(f"领域识别失败: {e}")
            return 2
    finally:
        # P2-1: 批量写盘，无论成败都 flush 一次
        flush_status()

    if success:
        print("同步完成!")
    else:
        print("部分同步失败，请检查状态")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
