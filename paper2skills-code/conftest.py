"""Code-template pytest preflight.

Code templates intentionally depend on heavier analytical packages. When those
packages are not installed, skip template tests with a clear environment message
instead of failing during test collection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


REQUIRED_MODULES = ("matplotlib", "numpy", "openai", "scipy", "sklearn")


def missing_code_template_dependencies() -> list[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    missing = missing_code_template_dependencies()
    if not missing:
        return False
    path = Path(str(collection_path))
    if path.name == "test_environment.py":
        return False
    return path.name.startswith("test_") and path.suffix == ".py"


def pytest_report_header(config) -> str | None:
    missing = missing_code_template_dependencies()
    if not missing:
        return None
    return (
        "paper2skills-code dependencies missing: "
        f"{', '.join(missing)}. Run `python3 -m paper2skills_common.doctor --check deps --json` "
        "or install `paper2skills-code/requirements-lock.txt`."
    )

