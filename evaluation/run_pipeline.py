#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Bingxian Wu
# One-click evaluation pipeline with checkpoint resume support.
# Step scripts live under evaluation/steps/; this entrypoint orchestrates them.
"""
Usage:
python evaluation/run_pipeline.py \
    --base-dir <RESULTS_DIR> \
    --model-name <MODEL_NAME> \
    --rounds 1,2,3 \
    --result-dir-name <TAG>
"""
import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Step:
    name: str
    cmd: List[str]
    outputs: List[Path]


def _exists_all(paths: List[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _filter_existing_rounds_by_attempt_files(base_dir: Path, rounds: List[int]) -> List[int]:
    """Keep rounds whose results_attempt_{r}.jsonl exists and is non-empty.

    ROUND1 requires attempt_1; ROUND r requires attempt_r.
    """
    kept: List[int] = []
    for r in rounds:
        attempt_path = base_dir / f"results_attempt_{r}.jsonl"
        if attempt_path.exists() and attempt_path.stat().st_size > 0:
            kept.append(r)
    return kept


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _polish_config_from_yaml(config_path):
    """Read (api_key, base_url, model) for the polish/judge LLM from a configs/*.yaml,
    so that one file is the single source of truth for the judge endpoint + model.
    Returns empty strings when no config is given or a field is absent; the polish step
    then falls back to env vars (RSMeM_API_KEY / RSMeM_BASE_URL / RSMeM_POLISH_MODEL)."""
    if not config_path:
        return "", "", ""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return (
        str(cfg.get("api_key", "") or ""),
        str(cfg.get("base_url", "") or ""),
        str(cfg.get("model", "") or ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-click evaluation pipeline with resume support.\n"
        "Steps: complete_results -> polish -> extract_tools -> merge -> step2step -> end2end"
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Model run directory (contains results_attempt_*.jsonl, log, experiment_summary.json)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Used to infer log filename: {model_name}_AP_langchain.log. "
             "If omitted, read from base-dir/experiment_config.yaml",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a configs/*.yaml (e.g. evaluation/configs/polish.yaml); its "
             "api_key / base_url / model are forwarded to the polish step as the "
             "judge endpoint + judge model.",
    )
    parser.add_argument(
        "--result-dir-name",
        required=True,
        help="Folder name under results/ to store evaluation outputs",
    )
    parser.add_argument(
        "--rounds",
        default="1,2,3",
        help="Comma-separated round numbers, e.g. 1,2,3",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps whose outputs already exist (default behavior).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if outputs exist.",
    )
    parser.add_argument(
        "--start-from",
        choices=["complete", "polish", "extract", "merge", "step2step", "end2end"],
        default=None,
        help="Start from a specific step (previous steps will be skipped).",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to a JSON state file for resume info. Default: {results_dir}/pipeline_state.json",
    )

    parser.add_argument("--max-workers", type=int, default=3, help="Max concurrent rounds for polishing")
    parser.add_argument(
        "--strict-rounds",
        action="store_true",
        help="If set, fail when any requested round is missing required input files.",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.model_name is None:
        exp_cfg = base_dir / "experiment_config.yaml"
        if exp_cfg.exists():
            import yaml
            cfg_data = yaml.safe_load(exp_cfg.read_text(encoding="utf-8")) or {}
            args.model_name = cfg_data.get("model_name", "deepseek-chat")
            print(f"  model-name inferred from experiment_config.yaml: {args.model_name}")
        else:
            args.model_name = "deepseek-chat"
            print(f"  Warning: no experiment_config.yaml found, defaulting model-name to {args.model_name}")

    here = Path(__file__).resolve().parent
    results_dir = here / "results" / args.result_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    requested_rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]
    requested_rounds = sorted(set(r for r in requested_rounds if r >= 1))
    if not requested_rounds:
        print("Error: no valid rounds provided")
        return 2

    # Filter rounds based on existence of results_attempt_{r}.jsonl.
    # This prevents failures when users pass 1..5 but only 1..3 exist in base_dir.
    effective_rounds = _filter_existing_rounds_by_attempt_files(base_dir, requested_rounds)
    missing_rounds = [r for r in requested_rounds if r not in effective_rounds]
    if missing_rounds:
        msg = (
            "Warning: missing or empty attempt files for rounds: "
            + ",".join(str(r) for r in missing_rounds)
            + ". Will skip these rounds."
        )
        if args.strict_rounds:
            print(msg.replace("Will skip these rounds", "Strict mode enabled; failing"))
            return 3
        print(msg)

    rounds = effective_rounds
    if not rounds:
        print("Error: no effective rounds to run (all requested rounds missing attempt files)")
        return 3

    state_file = Path(args.state_file) if args.state_file else (results_dir / "pipeline_state.json")
    state = _load_state(state_file)

    steps_dir = here / "steps"
    complete_script = steps_dir / "prepare_complete_results.py"
    polish_script = steps_dir / "prepare_polish_answers.py"
    extract_script = steps_dir / "prepare_extract_tool_calls.py"
    merge_script = steps_dir / "prepare_merge_tool_calls.py"
    step2step_script = steps_dir / "metric_step_by_step.py"
    end2end_script = steps_dir / "metric_end_to_end.py"

    # Build expected outputs
    summary_outputs = [base_dir / f"results_summary_ROUND{r}.jsonl" for r in rounds]
    polished_outputs = [base_dir / f"results_summary_polished_ROUND{r}.jsonl" for r in rounds]
    extracted_outputs = [base_dir / f"extracted_tool_calls_round{r}.json" for r in rounds]
    merged_outputs = [base_dir / f"merged_tool_calls_round{r}.json" for r in rounds]

    step2step_outputs = [results_dir / f"step_by_step_evaluation_results_round{r}.json" for r in rounds]
    end2end_outputs = [results_dir / f"end_to_end_evaluation_results_round{r}.json" for r in rounds]

    polish_api_key, polish_base_url, polish_model = _polish_config_from_yaml(args.config)

    # Steps
    py = sys.executable

    polish_cmd = [
        py,
        str(polish_script),
        "--base-dir",
        str(base_dir),
        "--model-name",
        args.model_name,
        "--rounds",
        ",".join(str(r) for r in rounds),
        "--max-workers",
        str(args.max_workers),
        "--api-key",
        polish_api_key,
        "--base-url",
        polish_base_url,
    ]
    # NB two different "models": `--model-name` above is the AGENT model being evaluated
    # (only used to locate the run's {model}_AP_langchain.log for question text); `--model`
    # here is the POLISH/JUDGE LLM that extracts the final <Answer>X<Answer>. Forward the
    # config yaml's `model:` when present; when absent the polish step fails loudly (it has
    # no hardcoded default) rather than silently using an unreachable model.
    if polish_model:
        polish_cmd += ["--model", polish_model]

    steps: List[Step] = [
        Step(
            name="complete",
            cmd=[
                py,
                str(complete_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
            ],
            outputs=summary_outputs,
        ),
        Step(
            name="polish",
            cmd=polish_cmd,
            outputs=polished_outputs,
        ),
        Step(
            name="extract",
            cmd=[
                py,
                str(extract_script),
                "--base-dir",
                str(base_dir),
                "--model-name",
                args.model_name,
            ],
            outputs=extracted_outputs,
        ),
        Step(
            name="merge",
            cmd=[
                py,
                str(merge_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
            ],
            outputs=merged_outputs,
        ),
        Step(
            name="step2step",
            cmd=[
                py,
                str(step2step_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
                "--result-dir-name",
                args.result_dir_name,
            ],
            outputs=step2step_outputs,
        ),
        Step(
            name="end2end",
            cmd=[
                py,
                str(end2end_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
                "--result-dir-name",
                args.result_dir_name,
            ],
            outputs=end2end_outputs,
        ),
    ]

    # Apply start-from
    if args.start_from:
        start_index = next((i for i, s in enumerate(steps) if s.name == args.start_from), 0)
        steps = steps[start_index:]

    for step in steps:
        already_done = _exists_all(step.outputs)
        if already_done and (args.resume or not args.force):
            print(f"[SKIP] {step.name}: outputs already exist")
            state[step.name] = {"status": "skipped", "outputs": [str(p) for p in step.outputs]}
            _save_state(state_file, state)
            continue

        print(f"\n===== STEP: {step.name} =====")
        try:
            _run(step.cmd)
        except subprocess.CalledProcessError as e:
            state[step.name] = {"status": "failed", "returncode": e.returncode, "cmd": step.cmd}
            _save_state(state_file, state)
            print(f"[FAIL] {step.name}: returncode={e.returncode}")
            return e.returncode

        if not _exists_all(step.outputs):
            state[step.name] = {"status": "incomplete", "outputs": [str(p) for p in step.outputs]}
            _save_state(state_file, state)
            print(f"[FAIL] {step.name}: outputs missing or empty")
            return 3

        state[step.name] = {"status": "done", "outputs": [str(p) for p in step.outputs]}
        _save_state(state_file, state)
        print(f"[OK] {step.name}")

    print("\nAll steps finished.")
    print(f"State file: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
