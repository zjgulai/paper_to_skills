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
from pathlib import Path
from datetime import datetime

# 配置路径
BASE_DIR = Path("/Users/pray/project/paper_to_skills")
VAULT_DIR = BASE_DIR / "paper2skills-vault"
CODE_DIR = BASE_DIR / "paper2skills-code"
SKILLS_DIR = BASE_DIR / "paper2skills-skills"
STATUS_FILE = VAULT_DIR / "07-资源库" / "sync_status.json"

# 领域映射
DOMAINS = {
    "causal_inference": "01-因果推断",
    "ab_testing": "02-A_B实验",
    "time_series": "03-时间序列",
    "supply_chain": "04-供应链",
    "recommendation": "05-推荐系统",
    "growth_model": "06-增长模型",
    "nlp_voc": "07-NLP-VOC"
}

# 从 Skill 卡片文件名提取领域
def extract_domain(skill_name):
    """从 Skill 名称推断领域"""
    name_lower = skill_name.lower()

    domain_keywords = {
        "uplift": "causal_inference",
        "causal": "causal_inference",
        "treatment": "causal_inference",
        "bandit": "ab_testing",
        "experiment": "ab_testing",
        "forecasting": "time_series",
        "demand": "time_series",
        "time series": "time_series",
        "inventory": "supply_chain",
        "supply chain": "supply_chain",
        "recommendation": "recommendation",
        "collaborative": "recommendation",
        "churn": "growth_model",
        "ltv": "growth_model",
        "sentiment": "nlp_voc",
        "voc": "nlp_voc",
        "opinion": "nlp_voc"
    }

    for keyword, domain in domain_keywords.items():
        if keyword in name_lower:
            return domain

    return "causal_inference"  # 默认


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


def update_status(skill_name, target, success=True, error=None):
    """更新同步状态"""
    status = load_status()

    if skill_name not in status:
        status[skill_name] = {}

    status[skill_name][target] = {
        "synced": success,
        "timestamp": datetime.now().isoformat(),
        "error": error
    }

    save_status(status)
    return status


def sync_vault(skill_name, domain=None):
    """同步到 Obsidian Vault"""
    if domain is None:
        domain = extract_domain(skill_name)

    # 源文件可能是 skill 输出目录或 vault 已有文件
    possible_sources = [
        SKILLS_DIR / "output" / f"{skill_name}.md",
        VAULT_DIR / "output" / f"{skill_name}.md",
    ]

    source = None
    for s in possible_sources:
        if s.exists():
            source = s
            break

    if source is None:
        print(f"源文件不存在")
        update_status(skill_name, "vault", False, "source file not found")
        return False

    target_dir = VAULT_DIR / DOMAINS.get(domain, domain)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{skill_name}.md"

    shutil.copy2(source, target)
    print(f"已同步到 Vault: {target}")
    update_status(skill_name, "vault", True)
    return True


def sync_code(skill_name, domain=None):
    """同步到 GitHub 代码仓库"""
    if domain is None:
        domain = extract_domain(skill_name)

    # 模块名从 skill 名转换
    module_name = skill_name.lower().replace("skill-", "").replace("-", "_")

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
        update_status(skill_name, "github", False, "source directory not found")
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
        update_status(skill_name, "github", False, "no files to copy")
        return False

    update_status(skill_name, "github", True)
    return True


def sync_feishu(skill_name):
    """同步到飞书（需要配置 webhook）"""
    webhook_path = Path.home() / ".paper2skills" / "feishu_webhook"

    if not webhook_path.exists():
        update_status(skill_name, "feishu", False, "not configured")
        print("飞书 webhook 未配置，跳过")
        return False

    with open(webhook_path, 'r') as f:
        webhook_url = f.read().strip()

    if not webhook_url:
        update_status(skill_name, "feishu", False, "webhook empty")
        return False

    # 读取 skill 内容
    skill_content = ""
    for domain_dir in VAULT_DIR.iterdir():
        if domain_dir.is_dir():
            skill_file = domain_dir / f"{skill_name}.md"
            if skill_file.exists():
                skill_content = skill_file.read_text()
                break

    if not skill_content:
        update_status(skill_name, "feishu", False, "skill content not found")
        return False

    # 发送飞书消息（简化版）
    import requests
    try:
        payload = {
            "msg_type": "text",
            "content": {"text": f"Skill 同步: {skill_name}\n\n{skill_content[:500]}..."}
        }
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            update_status(skill_name, "feishu", True)
            print(f"已同步到飞书")
            return True
        else:
            update_status(skill_name, "feishu", False, f"HTTP {response.status_code}")
            return False
    except Exception as e:
        update_status(skill_name, "feishu", False, str(e))
        return False


def show_status(skill_name):
    """显示同步状态"""
    status = load_status()

    if skill_name not in status:
        print(f"未找到 {skill_name} 的同步记录")
        return

    print(f"=== {skill_name} 同步状态 ===")
    for target, info in status[skill_name].items():
        status_str = "✅" if info["synced"] else "❌"
        print(f"{status_str} {target}: {info['timestamp']}")
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
                print(f"\n{skill_name}:")
                for target, info in targets.items():
                    status_str = "✅" if info["synced"] else "❌"
                    print(f"  {status_str} {target}")
        return 0

    if not args.skill:
        print("请指定 --skill 参数")
        return 1

    targets = args.target.split(",") if args.target else ["vault", "github"]

    print(f"同步 {args.skill} 到 {targets}...")

    success = True

    if "vault" in targets:
        success &= sync_vault(args.skill, args.domain)

    if "github" in targets:
        success &= sync_code(args.skill, args.domain)

    if "feishu" in targets:
        success &= sync_feishu(args.skill)

    if success:
        print("同步完成!")
    else:
        print("部分同步失败，请检查状态")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())