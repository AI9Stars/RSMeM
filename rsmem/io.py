"""I/O helpers for benchmark questions, run results, and chat logging."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union

PathLike = Union[str, Path]


def load_questions(json_path: PathLike) -> List[MutableMapping[str, Any]]:
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        test_json: Mapping[str, Any] = json.load(f)

    out: List[MutableMapping[str, Any]] = []
    for question_idx, question_info in test_json.items():
        evaluation = question_info.get("evaluation")
        if not evaluation:
            continue

        ap_index = _autonomous_planning_index(evaluation)
        auto_ev = evaluation[ap_index]
        instruct_ev = evaluation[1 - ap_index]

        data = auto_ev.get("data") or instruct_ev.get("data")
        if data is None:
            continue

        auto = auto_ev["question"]
        instruct = instruct_ev["question"]

        choices = question_info.get("choices")
        if choices is None:
            choices = _choices_from_evaluation(evaluation)

        out.append(
            {
                "question_id": question_idx,
                "auto": auto,
                "instruct": instruct,
                "data": data,
                "choices": choices,
            }
        )

    return out


def _first_tool_content(dialogs: Sequence[Mapping[str, Any]]) -> Any | None:
    for d in dialogs:
        if str(d.get("role", "")).lower() != "tool":
            continue
        if "content" in d:
            return d["content"]
    return None


def _autonomous_planning_index(evaluation: Sequence[Mapping[str, Any]]) -> int:
    if not evaluation:
        return 0
    t0 = str(evaluation[0].get("type", "") or "").lower()
    return 0 if t0 == "autonomous planning" else 1


def _choices_from_evaluation(evaluation: Iterable[Mapping[str, Any]]) -> Any | None:
    for entry in evaluation:
        c = entry.get("choices")
        if isinstance(c, list):
            return c
    return None


def save_chat_message(
    chat_log_path: PathLike,
    role: str,
    content: Any,
    question_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path = Path(chat_log_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if role == "user":
        md = {**(metadata or {}), "question_id": question_id}
        name = "user"
    elif role == "assistant":
        md = None
        name = question_id
    else:
        md = None
        name = "system"

    record = {
        "__module__": "langchain.schema.messages",
        "__name__": "ChatMessage",
        "id": uuid.uuid4().hex,
        "name": name,
        "role": role,
        "content": content,
        "metadata": md,
        "timestamp": ts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _JsonFormatter(logging.Formatter):
    """Format a log record using the schema supported by earlier RSMeM runs."""

    def format(self, record: logging.LogRecord) -> str:
        args = record.args if record.args else ()
        log_record = {
            "question_index": args[0] if len(args) > 0 else "unknown",
            "timestamp": self.formatTime(record, self.datefmt),
            "conversations": args[1] if len(args) > 1 else [],
            "final_answer": args[2] if len(args) > 2 else None,
        }
        return json.dumps(log_record, ensure_ascii=False, indent=4)


def create_experiment_logger(
    output_dir: PathLike,
    model_name: str,
    *,
    autoplanning: bool = True,
) -> logging.Logger:
    """Create a per-experiment file logger that writes JSON records.

    Produces ``{model_name}_{AP|IF}_langchain.log`` inside *output_dir*, one
    JSON object per line, compatible with earlier RSMeM releases.
    """
    mode_tag = "AP" if autoplanning else "IF"
    log_path = Path(output_dir) / f"{model_name}_{mode_tag}_langchain.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"rsmem.experiment.{log_path.stem}.{id(log_path)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = RotatingFileHandler(log_path, encoding="utf-8")
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)

    return logger


def _filter_chat_through_attempt(chat_path: Path, n: int) -> list[str]:
    """Return messages from attempts 1..n of a ``.chat`` file.

    Attempt boundaries are inferred by counting how many times each
    ``question_id`` appears as a *user* message. The k-th user message
    for a given qid starts attempt-k; messages are kept while the count
    for the current qid is <= n.
    """
    counts: dict[str, int] = {}
    capturing = True
    out: list[str] = []

    with open(chat_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            role = msg.get("role", "")
            md = msg.get("metadata") or {}
            qid = str(md.get("question_id", ""))

            if role == "user" and qid:
                counts[qid] = counts.get(qid, 0) + 1
                capturing = counts[qid] <= n
                if not capturing:
                    continue

            if capturing:
                out.append(raw)

    return out


def _filter_chat_attempt1(chat_path: Path) -> list[str]:
    """Return only the attempt-1 slice of a ``.chat`` file."""
    return _filter_chat_through_attempt(chat_path, 1)


def _iter_json_log_blocks(log_path: Path) -> Iterable[Tuple[str, Any]]:
    """Yield ``(block_text, parsed_record_or_None)`` for each JSON value in a
    ``.log`` file.

    Earlier formatters wrote each record with ``indent=4`` and
    handlers append them back-to-back separated by newlines, so a ``.log``
    file is a stream of pretty-printed JSON values (not JSONL).

    Splitting by raw ``{``/``}`` counts is unsafe because tool-output /
    assistant-content strings frequently contain unbalanced braces which
    silently glue multiple records into one huge (unparseable) block —
    downstream that causes ``seed_resume_files`` to keep only the first
    couple of records of a 200+ record run.

    Instead, delegate the split to ``json.JSONDecoder.raw_decode``, which
    consumes exactly one complete JSON value from a string and returns the
    index where it stopped. That correctly handles ``{``/``}`` inside string
    literals because it *is* the same parser used everywhere else.

    ``block_text`` is the exact substring of the source file that produced
    the JSON value (leading whitespace stripped). ``parsed_record`` is the
    decoded object, or ``None`` if the fragment could not be decoded (kept
    for symmetry with the previous behavior of preserving unparseable
    trailers).
    """
    text = log_path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    n = len(text)
    i = 0
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            obj, end = dec.raw_decode(text, idx=i)
        except json.JSONDecodeError:
            # Preserve whatever junk remains as one final unparseable block so
            # callers can still inspect it (matching the earlier append behavior for
            # blocks that fail ``json.loads``). Then stop — we can't safely
            # skip past malformed JSON.
            tail = text[i:].rstrip()
            if tail:
                yield tail, None
            return
        block_text = text[i:end]
        yield block_text, obj
        i = end


def _filter_log_through_attempt(log_path: Path, n: int) -> list[str]:
    """Return the first *n* occurrences of each ``question_index`` in a ``.log``.

    The ``.log`` is a sequence of JSON records (possibly multi-line with
    ``indent=4``).  Each record has a ``question_index`` key.
    """
    counts: dict[str, int] = {}
    out: list[str] = []

    for block, record in _iter_json_log_blocks(log_path):
        if not isinstance(record, dict):
            out.append(block)
            continue
        qidx = str(record.get("question_index", ""))
        if not qidx:
            out.append(block)
            continue
        counts[qidx] = counts.get(qidx, 0) + 1
        if counts[qidx] <= n:
            out.append(block)

    return out


def _filter_log_attempt1(log_path: Path) -> list[str]:
    """Return only the first occurrence of each ``question_index`` in a ``.log``."""
    return _filter_log_through_attempt(log_path, 1)


def seed_resume_files(
    resume_dir: PathLike, experiment_dir: PathLike, through_attempt: int = 1
) -> None:
    """Copy artefacts for attempts 1..*through_attempt* into *experiment_dir*.

    Makes the new experiment directory self-contained so that
    ``evaluation/run_pipeline.py`` can process it without manual file copying.

    Seeds:
      - ``results_attempt_{1..n}.jsonl`` — verbatim copies
      - ``results_summary_ROUND{1..n}.jsonl`` — verbatim copies (when present)
      - ``results_summary_polished_ROUND{1..n}.jsonl`` — verbatim copies (when present)
      - ``*.chat`` → ``run.chat``        — filtered to attempts 1..n only
      - ``*.log``  → same name           — filtered to first n occurrences per question

    For a resume-through-attempt run, rounds 1..n are byte-identical to the
    source, so carrying over the source's already-computed eval summaries and
    polished answers lets ``evaluation/run_pipeline.py`` skip re-polishing them
    (saving LLM calls). Sources predating this feature may lack these files;
    missing ones are simply skipped.

    Idempotent: skips files that already exist in *experiment_dir*.
    """
    import shutil

    src = Path(resume_dir)
    dst = Path(experiment_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for k in range(1, through_attempt + 1):
        jsonl = src / f"results_attempt_{k}.jsonl"
        dst_jsonl = dst / f"results_attempt_{k}.jsonl"
        if jsonl.exists() and not dst_jsonl.exists():
            shutil.copy2(jsonl, dst_jsonl)

    # Carry over already-computed eval artefacts for rounds 1..n (verbatim).
    # Older sources may not have these — skip whatever is missing.
    for k in range(1, through_attempt + 1):
        for name in (
            f"results_summary_ROUND{k}.jsonl",
            f"results_summary_polished_ROUND{k}.jsonl",
        ):
            src_file = src / name
            dst_file = dst / name
            if src_file.exists() and not dst_file.exists():
                shutil.copy2(src_file, dst_file)

    chat_files = sorted(src.glob("*.chat"))
    dst_chat = dst / "run.chat"
    if chat_files and not dst_chat.exists():
        filtered = _filter_chat_through_attempt(chat_files[0], through_attempt)
        dst_chat.write_text("\n".join(filtered) + "\n" if filtered else "", encoding="utf-8")

    for log_file in sorted(src.glob("*_langchain.log")):
        dst_log = dst / log_file.name
        if not dst_log.exists():
            filtered = _filter_log_through_attempt(log_file, through_attempt)
            dst_log.write_text("\n".join(filtered) + "\n" if filtered else "", encoding="utf-8")


def cleanup_chat_after_continue(
    chat_path: PathLike, error_qids: set[str] | None = None
) -> None:
    """Remove error-qid entries from ``.chat`` before retrying.

    When *error_qids* is given, only segments belonging to those qids are
    removed entirely.  When *error_qids* is None, falls back to deduplication
    (keeps last occurrence per qid).

    Called **before** the retry runs, so the retry appends clean results.
    """
    path = Path(chat_path)
    if not path.exists():
        return

    raw_lines: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                raw_lines.append(line)

    # Identify segments: each segment starts with a user msg that has a qid
    segments: list[tuple[str, list[str]]] = []  # (qid, [lines])
    current_qid = ""
    current_lines: list[str] = []

    for raw in raw_lines:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            if current_lines:
                current_lines.append(raw)
            continue

        role = msg.get("role", "")
        md = msg.get("metadata") or {}
        qid = str(md.get("question_id", ""))

        if role == "user" and qid:
            if current_lines:
                segments.append((current_qid, current_lines))
            current_qid = qid
            current_lines = [raw]
        else:
            current_lines.append(raw)

    if current_lines:
        segments.append((current_qid, current_lines))

    if error_qids is not None:
        # Remove ALL segments for error qids (they'll be re-run fresh)
        kept: list[str] = []
        for qid, lines in segments:
            if qid not in error_qids:
                kept.extend(lines)
    else:
        # Fallback: keep only the LAST segment per qid
        last_index: dict[str, int] = {}
        for i, (qid, _) in enumerate(segments):
            if qid:
                last_index[qid] = i
        kept = []
        for i, (qid, lines) in enumerate(segments):
            if not qid or last_index.get(qid) == i:
                kept.extend(lines)

    path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")


def cleanup_log_after_continue(
    log_path: PathLike, error_qids: set[str] | None = None
) -> None:
    """Remove error-qid entries from ``.log`` before retrying.

    When *error_qids* is given, all records for those qids are removed.
    When None, falls back to deduplication (keeps last per qid).
    """
    path = Path(log_path)
    if not path.exists():
        return

    blocks: list[tuple[str, str]] = []  # (qid, block_text)
    for block, record in _iter_json_log_blocks(path):
        if isinstance(record, dict):
            qidx = str(record.get("question_index", ""))
            blocks.append((qidx, block))
        else:
            blocks.append(("", block))

    if error_qids is not None:
        kept: list[str] = [block for qid, block in blocks if qid not in error_qids]
    else:
        last_index: dict[str, int] = {}
        for i, (qid, _) in enumerate(blocks):
            if qid:
                last_index[qid] = i
        kept = []
        for i, (qid, block) in enumerate(blocks):
            if not qid or last_index.get(qid) == i:
                kept.append(block)

    path.write_text("\n".join(kept) + "\n" if kept else "", encoding="utf-8")


def _normalise_result_record(r: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Ensure result records from earlier RSMeM or Earth-Agent runs carry the
    ``agent_response`` and ``heuristic_success`` keys that downstream code expects."""
    if "agent_response" not in r:
        r["agent_response"] = r.get("answer", "")
    if "heuristic_success" not in r:
        r["heuristic_success"] = not r.get("heuristic_issue", True)
    return r


def _is_error_result(r: MutableMapping[str, Any]) -> bool:
    """Return True if *r* represents an HTTP 401 API error (e.g. quota
    exhausted) rather than a genuine model answer.

    Heuristics:
    - answer starts with ``"Error: Error code: 401 - "`` (auth/quota failure)
    - ``token_usage`` is empty/missing (API never returned a response)
    """
    answer = str(r.get("answer", ""))
    if answer.startswith("Error: Error code: 401 - "):
        return True
    tu = r.get("token_usage")
    if not tu or (isinstance(tu, dict) and not tu):
        return True
    return False


def load_completed_results(
    jsonl_path: PathLike,
) -> Tuple[List[MutableMapping[str, Any]], List[str]]:
    """Load a partial ``results_attempt_N.jsonl`` and separate completed
    results from error results.

    Returns ``(completed_results, error_question_ids)`` where
    *completed_results* preserves the original order and *error_question_ids*
    lists question IDs whose answers are API/runtime errors.
    """
    path = Path(jsonl_path)
    completed: List[MutableMapping[str, Any]] = []
    error_ids: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _normalise_result_record(json.loads(line))
            if _is_error_result(r):
                error_ids.append(str(r["question_id"]))
            else:
                completed.append(r)

    return completed, error_ids


def load_attempt_results(
    resume_dir: PathLike,
) -> Tuple[List[MutableMapping[str, Any]], List[str], Path]:
    base = Path(resume_dir)
    attempt1_file = base / "results_attempt_1.jsonl"
    if not attempt1_file.exists():
        raise FileNotFoundError(f"Attempt 1 results not found: {attempt1_file}")

    results: List[MutableMapping[str, Any]] = []
    with open(attempt1_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(_normalise_result_record(json.loads(line)))

    failed_ids: List[str] = [
        str(r["question_id"]) for r in results if r.get("heuristic_issue", False)
    ]

    chat_files = sorted(base.glob("*.chat"))
    if not chat_files:
        raise FileNotFoundError(f"Chat log file not found in {base}")
    chat_log_path = chat_files[0]

    return results, failed_ids, chat_log_path


def load_attempts_through(
    resume_dir: PathLike, n: int
) -> Tuple[List[dict], List[str], Path]:
    """Load attempts 1..*n* from *resume_dir*.

    Returns ``(loaded_attempts, failed_ids_of_attempt_n, source_chat_path)``
    where each element of *loaded_attempts* is a dict with keys
    ``attempt``, ``results``, ``failed_ids``, ``token_usage``.

    The returned *failed_ids* are from **attempt n** (not attempt 1), which
    drives the next attempt.
    """
    base = Path(resume_dir)
    loaded_attempts: List[dict] = []

    for k in range(1, n + 1):
        attempt_file = base / f"results_attempt_{k}.jsonl"
        if not attempt_file.exists():
            raise FileNotFoundError(
                f"results_attempt_{k}.jsonl not found in {base}"
            )

        results: List[MutableMapping[str, Any]] = []
        with open(attempt_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(_normalise_result_record(json.loads(line)))

        failed_ids_k: List[str] = [
            str(r["question_id"]) for r in results if r.get("heuristic_issue", False)
        ]

        loaded_attempts.append({
            "attempt": k,
            "results": results,
            "failed_ids": failed_ids_k,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })

        # Defensive: attempt k+1's questions should be a subset of attempt k's failures
        if k >= 2:
            prev_failed = set(loaded_attempts[k - 2]["failed_ids"])
            current_qids = {str(r["question_id"]) for r in results}
            unexpected = current_qids - prev_failed
            if unexpected:
                print(
                    f"  Warning: attempt {k} has {len(unexpected)} question(s) "
                    f"not in attempt {k-1} failed set: {sorted(unexpected)[:5]}"
                )

    final_failed_ids = loaded_attempts[-1]["failed_ids"]

    chat_files = sorted(base.glob("*.chat"))
    if not chat_files:
        raise FileNotFoundError(f"Chat log file not found in {base}")

    return loaded_attempts, final_failed_ids, chat_files[0]
