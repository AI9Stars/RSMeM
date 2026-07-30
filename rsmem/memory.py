"""Failure-Aware Memory (FAM): online error memory that stores, per question,
the reflections distilled from failed tool-use traces (load/save JSON)."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class FailureAwareMemory:
    """Failure-Aware Memory (FAM): holds reflection history per question id, with
    optional JSON persistence."""

    def __init__(self, file_path: str | Path | None = None) -> None:
        self.file_path = Path(file_path) if file_path is not None else None
        self._data: dict[str, Any] = {
            "by_qid": {},
            "metadata": {"last_updated": "", "total_reflections": 0},
        }
        if self.file_path is not None and self.file_path.exists():
            self._load()

    @property
    def by_qid(self) -> dict[str, list[dict[str, Any]]]:
        return self._data["by_qid"]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._data["metadata"]

    def _load(self) -> None:
        """Load JSON from disk; malformed or missing pieces get safe defaults."""
        assert self.file_path is not None
        try:
            with open(self.file_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                self._reset()
                return
            if "by_qid" not in data:
                data["by_qid"] = {}
            if "metadata" not in data:
                data["metadata"] = {"last_updated": "", "total_reflections": 0}
            self._data = data
        except OSError:
            self._reset()
        except json.JSONDecodeError:
            self._reset()

    def _reset(self) -> None:
        self._data = {
            "by_qid": {},
            "metadata": {"last_updated": "", "total_reflections": 0},
        }

    def save(self) -> None:
        """Write memory to ``file_path`` via a temp file and atomic replace."""
        if self.file_path is None:
            raise ValueError("file_path is required to save online memory")

        memory_path = self.file_path
        temp_path = memory_path.with_suffix(".tmp")

        self._data["metadata"]["last_updated"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self._data["metadata"]["total_reflections"] = sum(
            len(v) for v in self._data["by_qid"].values()
        )

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

        os.replace(temp_path, memory_path)

    def add_reflection(
        self, question_id: str | int, reflective_text: str, attempt: int
    ) -> FailureAwareMemory:
        qid = str(question_id)
        if qid not in self._data["by_qid"]:
            self._data["by_qid"][qid] = []

        entry = {
            "question_id": qid,
            "attempt": attempt,
            "reflective_text": reflective_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._data["by_qid"][qid].append(entry)
        return self

    def get_latest_reflection(self, question_id: str | int) -> str:
        qid = str(question_id)
        entries = self._data["by_qid"].get(qid) or []
        if not entries:
            return ""
        return entries[-1]["reflective_text"]
