#!/usr/bin/env python3
"""Public question-path variant generator.

Produces ``benchmark/question{N}.json`` from ``benchmark/question.json`` by
rewriting every ``benchmark/data/`` path prefix to ``benchmark/data{N}/`` so a
run can point at an isolated data copy (``benchmark/data{N}``). This avoids the
small-model contamination problem where an agent may write intermediate files
into the shared dataset.

Only ``benchmark/data/`` is rewritten; ``benchmark/supported_data/`` is left
untouched (its substring is not ``benchmark/data/``).

Usage:
    python benchmark/make_question_variant.py --version 1
    # -> benchmark/question1.json  (benchmark/data/ -> benchmark/data1/)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple


def rewrite_data_paths(obj: Any, version: Any) -> Tuple[Any, int]:
    """Recursively rewrite ``benchmark/data/`` -> ``benchmark/data{version}/``.

    Pure function: returns ``(new_obj, n_replacements)`` and never mutates *obj*.
    ``benchmark/supported_data/`` is left untouched. Non-string scalars and the
    overall structure (dict keys, list lengths) are preserved verbatim.
    """
    old = "benchmark/data/"
    new = f"benchmark/data{version}/"

    if isinstance(obj, dict):
        out: dict = {}
        total = 0
        for k, v in obj.items():
            nv, n = rewrite_data_paths(v, version)
            out[k] = nv
            total += n
        return out, total

    if isinstance(obj, list):
        out_list: list = []
        total = 0
        for item in obj:
            ni, n = rewrite_data_paths(item, version)
            out_list.append(ni)
            total += n
        return out_list, total

    if isinstance(obj, str):
        n = obj.count(old)
        if n:
            return obj.replace(old, new), n
        return obj, 0

    return obj, 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite benchmark/data/ -> benchmark/data{version}/ in question.json."
    )
    parser.add_argument("--version", required=True, help="Variant version, e.g. 1 -> question1.json/data1")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "question.json"),
        help="Source question.json (default: benchmark/question.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: benchmark/question{version}.json)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.parent / f"question{args.version}.json"
    )

    data = json.loads(in_path.read_text(encoding="utf-8"))
    n_items = len(data) if isinstance(data, (list, dict)) else 0

    updated, n_repl = rewrite_data_paths(data, args.version)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=4), encoding="utf-8"
    )

    print(f"[OK] wrote {out_path}")
    print(f"[stats] items={n_items}  benchmark/data/ -> benchmark/data{args.version}/ replacements={n_repl}")


if __name__ == "__main__":
    main()
