"""Checkpointing stub: SQLite-based WorkflowContext 持久化.

生产可替换为 PostgresSaver (LangGraph 内置).
本实现使用 stdlib sqlite3,无需额外依赖.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_checkpoints (
    workflow_id TEXT PRIMARY KEY,
    workflow_type TEXT NOT NULL,
    operator_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wf_type ON workflow_checkpoints(workflow_type);
CREATE INDEX IF NOT EXISTS idx_updated ON workflow_checkpoints(updated_at);
"""


class SQLiteCheckpointer:
    def __init__(self, db_path: str = ".mas/checkpoints.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def save(self, state: Dict[str, Any]) -> None:
        workflow_id = state["workflow_id"]
        workflow_type = state.get("workflow_type", "unknown")
        operator_id = state.get("operator_id", "unknown")
        state_json = json.dumps(state, ensure_ascii=False, default=str)
        now = datetime.utcnow().isoformat() + "Z"
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO workflow_checkpoints (workflow_id, workflow_type, operator_id, state_json, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(workflow_id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (workflow_id, workflow_type, operator_id, state_json, now),
            )
            conn.commit()

    def load(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute("SELECT state_json FROM workflow_checkpoints WHERE workflow_id=?", (workflow_id,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else None

    def list_pending(self, workflow_type: Optional[str] = None) -> list[Dict[str, Any]]:
        sql = "SELECT workflow_id, workflow_type, updated_at, state_json FROM workflow_checkpoints"
        params: tuple = ()
        if workflow_type:
            sql += " WHERE workflow_type=?"
            params = (workflow_type,)
        sql += " ORDER BY updated_at DESC LIMIT 100"
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(sql, params)
            return [
                {
                    "workflow_id": r[0],
                    "workflow_type": r[1],
                    "updated_at": r[2],
                    "approved": json.loads(r[3]).get("approved"),
                }
                for r in cur.fetchall()
            ]
