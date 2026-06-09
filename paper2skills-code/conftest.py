"""Code-template pytest preflight.

Code templates intentionally depend on heavier analytical packages. When those
packages are not installed, skip template tests with a clear environment message
instead of failing during test collection.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REQUIRED_MODULES = ("matplotlib", "numpy", "openai", "scipy", "sklearn")
OPTIONAL_MODULES = ("econml", "imblearn", "prophet", "tensorflow", "torch")


def missing_code_template_dependencies() -> list[str]:
    return [module for module in REQUIRED_MODULES if importlib.util.find_spec(module) is None]


def missing_optional_code_template_dependencies() -> list[str]:
    return [module for module in OPTIONAL_MODULES if importlib.util.find_spec(module) is None]


def _imported_top_level_modules(source_path: Path) -> set[str]:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def _relative_import_source_paths(test_path: Path) -> list[Path]:
    try:
        tree = ast.parse(test_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    sources: list[Path] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0 or not node.module:
            continue
        first_segment = node.module.split(".", 1)[0]
        source = test_path.parent / f"{first_segment}.py"
        package_init = test_path.parent / first_segment / "__init__.py"
        if source.exists():
            sources.append(source)
        elif package_init.exists():
            sources.append(package_init)
    return sources


def missing_optional_dependencies_for_test_path(
    test_path: Path,
    optional_modules: tuple[str, ...] = OPTIONAL_MODULES,
) -> list[str]:
    source_paths = [test_path, *_relative_import_source_paths(test_path)]
    imported_modules: set[str] = set()
    for source_path in source_paths:
        imported_modules.update(_imported_top_level_modules(source_path))

    return [
        module
        for module in optional_modules
        if module in imported_modules and importlib.util.find_spec(module) is None
    ]


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    missing = missing_code_template_dependencies()
    path = Path(str(collection_path))
    if path.name == "test_environment.py":
        return False
    if not path.name.startswith("test_") or path.suffix != ".py":
        return False
    if missing:
        return True
    return bool(missing_optional_dependencies_for_test_path(path))


def pytest_report_header(config) -> str | None:
    missing = missing_code_template_dependencies()
    optional_missing = missing_optional_code_template_dependencies()
    messages: list[str] = []
    if missing:
        messages.append(
            "paper2skills-code dependencies missing: "
            f"{', '.join(missing)}. Run `python3 -m paper2skills_common.doctor --check deps --json` "
            "or install `paper2skills-code/requirements-lock.txt`."
        )
    if optional_missing:
        messages.append(
            "paper2skills-code optional dependencies missing: "
            f"{', '.join(optional_missing)}. Tests importing these modules are skipped during collection."
        )
    if not messages:
        return None
    return "\n".join(messages)
