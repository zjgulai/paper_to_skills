# AGENTS.md

This file is the entry point for **Codex / Codex.ai/code** when working in this repository.

## 单一信息源

项目结构、工作流、领域映射、Skill 卡片格式、代码规范、质量标准等内容,
**统一以 [`CLAUDE.md`](./CLAUDE.md) 为唯一信息源**。Codex 与 Claude Code 共用同一份描述,
仅在以下"代理特定差异"处保留独立说明,以避免双份文档同步漂移。

> 实操约定:任何对项目结构/领域/工作流的修改都直接编辑 `CLAUDE.md`,
> AGENTS.md 只承载 Codex 自身的差异性指引。

## Codex 特定差异

- **触发语**: Codex 通过自然语言触发 paper2skills 工作流时,
  与 Claude Code 完全等价(见 `CLAUDE.md → Workflow Commands`)。
- **运行入口**: Codex 优先使用 CLI/SDK 入口,
  与 Claude Code 的 Skill 触发互不冲突。
- **协作礼仪**: 与 Claude Code 共享同一仓库时,
  双方都遵守 `CLAUDE.md → Working with Skills` 中的 evolve / 版本演进规则。

## 历史脏化标记

- ~~原 AGENTS.md 中的 Project Structure / Domain Mapping 已迁出~~,
  请改读 `CLAUDE.md` 对应章节。如发现 AGENTS.md 与 CLAUDE.md 描述出现不一致,
  以 CLAUDE.md 为准并同步修正 AGENTS.md。
