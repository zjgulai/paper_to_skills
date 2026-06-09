#!/usr/bin/env python3
"""Run the incremental paper2skills workflow driver."""

from __future__ import annotations

import os
import sys
from pathlib import Path


BASE_DIR = Path(os.environ.get("PAPER2SKILLS_ROOT") or Path(__file__).resolve().parents[3])
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from paper2skills_common.workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
