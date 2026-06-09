from __future__ import annotations

import pytest

from conftest import missing_code_template_dependencies


def test_code_template_dependencies_available():
    missing = missing_code_template_dependencies()
    if missing:
        pytest.skip(
            "Missing code-template dependencies: "
            + ", ".join(missing)
            + ". Install paper2skills-code/requirements-lock.txt before running code-template tests."
        )
