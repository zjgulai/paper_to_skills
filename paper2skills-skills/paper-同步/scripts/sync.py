#!/usr/bin/env python3
"""
paper2skills 同步脚本
同步 Skill 卡片和代码模板到各个平台
"""

import os
import shutil
import argparse
from pathlib import Path

# 配置路径
BASE_DIR = Path(__file__).parent.parent.parent

VAULT_DIR = BASE_DIR / "paper2skills-vault"
CODE_DIR = BASE_DIR / "paper2skills-code"
SKILLS_DIR = BASE_DIR / "paper2skills-skills"

# 领域映射
DOMAINS = {
    "causal_inference": "01-因果推断",
    "ab_testing": "02-A_B实验",
    "time_series": "03-时间序列",
    "supply_chain": "04-供应链",
    "recommendation": "05-推荐系统",
    "growth_model": "06-增长模型"
}


def sync_vault(domain, skill_name):
    """同步到 Obsidian Vault"""
    source = SKILLS_DIR / "output" / f"Skill-{skill_name}.md"
    target_dir = VAULT_DIR / DOMAINS.get(domain, domain)

    if not source.exists():
        print(f"源文件不存在: {source}")
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"Skill-{skill_name}.md"

    shutil.copy2(source, target)
    print(f"已同步到 Vault: {target}")
    return True


def sync_code(domain, module_name):
    """同步到 GitHub 代码仓库"""
    source_dir = SKILLS_DIR / "output" / module_name
    target_dir = CODE_DIR / domain / module_name

    if not source_dir.exists():
        print(f"源目录不存在: {source_dir}")
        return False

    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件
    for file in ["__init__.py", "model.py", "example.py"]:
        source = source_dir / file
        if source.exists():
            target = target_dir / file
            shutil.copy2(source, target)
            print(f"已同步到代码仓库: {target}")

    return True


def main():
    parser = argparse.ArgumentParser(description="paper2skills 同步脚本")
    parser.add_argument("--domain", required=True, help="领域名称")
    parser.add_argument("--skill", required=True, help="技能名称")
    parser.add_argument("--module", help="模块名称（默认与skill相同）")

    args = parser.parse_args()

    module_name = args.module or args.skill.lower().replace("-", "_")

    print(f"同步 {args.skill} 到 {args.domain}...")

    success = True
    success &= sync_vault(args.domain, args.skill)
    success &= sync_code(args.domain, module_name)

    if success:
        print("同步完成!")
    else:
        print("同步失败!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
