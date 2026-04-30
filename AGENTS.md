# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

paper2skills is a knowledge pipeline (not a web service). It converts academic papers into business skill cards via Claude Code skills (markdown prompts). There is no web server, database, or Docker infrastructure. See `CLAUDE.md` for full project structure and workflow documentation.

### Services & components

| Component | Description | How to run |
|-----------|-------------|------------|
| Python code templates | 34 standalone `model.py` scripts across 7 domains | `python3 paper2skills-code/<domain>/<module>/model.py` |
| Sync script | Syncs skills to vault/GitHub/feishu | `python3 paper2skills-skills/paper-同步/scripts/sync.py --status` |
| Skills graph analyzer | Analyzes knowledge gaps | `python3 paper2skills-skills/paper-skills-graph/scripts/skills_graph_analyzer.py --vault /workspace/paper2skills-vault --gaps` |

### Non-obvious caveats

- **Hardcoded paths**: `sync.py` and `skills_graph_analyzer.py` default to `/Users/pray/project/paper_to_skills/`. In Cloud Agent environments, pass `--vault /workspace/paper2skills-vault` to the graph analyzer explicitly. The sync script reads `STATUS_FILE` from the hardcoded path; it will show empty status unless the path constant is updated.
- **No formal test suite**: There is no pytest config. Each `model.py` has a `if __name__ == "__main__"` block as a smoke test. Run all models with: `find paper2skills-code -name model.py -exec python3 {} \;`
- **Pre-existing code bugs**: 2 of 34 models have compatibility issues with modern numpy/pandas (numpy.trapz removed in numpy 2.x, pandas Timedelta type error). These are `growth_model/churn_prediction/model.py` and `growth_model/uplift_churn_prediction/model.py`.
- **No linter configured**: The project has no pylint/flake8/ruff configuration.
- **PyTorch CPU only**: Install torch with `--index-url https://download.pytorch.org/whl/cpu` to avoid downloading CUDA libraries (~2GB). Only 4 models need torch.
- **sentence-transformers**: 2 NLP models (`spiral_of_silence`, `revision_intent_mining`) require the `sentence-transformers` package.
