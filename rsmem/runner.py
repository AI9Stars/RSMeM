"""Multi-attempt evaluation orchestration."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

from rsmem.agent import _chat_openai_class, create_agent, handle_question, load_config
from rsmem.config import ExperimentConfig
from rsmem.evaluation import (
    aggregate_results,
    compute_statistics,
    heuristic_critic_has_issue,
)
from rsmem.io import (
    cleanup_chat_after_continue,
    cleanup_log_after_continue,
    create_experiment_logger,
    load_attempt_results,
    load_attempts_through,
    load_completed_results,
    load_questions,
    seed_resume_files,
)
from rsmem.kb import load_knowledge_base
from rsmem.memory import FailureAwareMemory
from rsmem.retrieval import EmbeddingModel, precompute_kb_embeddings
from rsmem.strategies import generate_reflections_for_failed_questions


def _build_reflection_kwargs(
    config: ExperimentConfig,
    *,
    compressor_llm: Any | None = None,
    artifact_dir: Path | None = None,
) -> dict:
    """Build kwargs for the reflection callable (base RSMeM uses the default strategy)."""
    kwargs: dict = {}
    if config.reflection_experience_prompt:
        kwargs["reflection_template"] = config.reflection_experience_prompt
    if config.reflection_strategy == "summary_ftro":
        if compressor_llm is None or artifact_dir is None:
            raise ValueError("summary_ftro requires compressor_llm and artifact_dir")
        kwargs["compressor_llm"] = compressor_llm
        kwargs["artifact_dir"] = artifact_dir
    return kwargs


def _build_trajectory_kwargs(config: ExperimentConfig) -> dict:
    if config.trajectory_strategy != "summary_ftro_source":
        return {}
    return {
        "max_step_chars": config.summary_max_step_chars,
        "max_trace_chars": config.summary_max_trace_chars,
    }



async def _maybe_await(fn_result: Any) -> Any:
    if inspect.isawaitable(fn_result):
        return await fn_result
    return fn_result


def _flush_results(path: Path, results: list[Any]) -> None:
    """Atomically overwrite *path* with current results."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


async def run_single_attempt(
    agent,
    questions: list,
    attempt_num: int,
    chat_log_path: Path,
    config: ExperimentConfig,
    memory: FailureAwareMemory | None = None,
    kb_embeds: dict | None = None,
    emb_model=None,
    output_dir: Path | None = None,
    logger=None,
    completed_results: list[Any] | None = None,
) -> tuple[list[Any], list[str], dict[str, int]]:
    results: list[Any] = []
    failed_ids: list[str] = []
    total_tokens: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    completed_map: dict[str, Any] = {}
    if completed_results:
        for r in completed_results:
            completed_map[str(r["question_id"])] = r

    save_dir = output_dir if output_dir is not None else Path.cwd()
    save_dir.mkdir(parents=True, exist_ok=True)
    attempt_results_path = save_dir / f"results_attempt_{attempt_num}.jsonl"
    dirty = False

    for question in questions:
        qid = str(question["question_id"])

        cached = completed_map.get(qid)
        if cached is not None:
            results.append(cached)
            if cached.get("heuristic_issue"):
                failed_ids.append(qid)
            tu = cached.get("token_usage") or {}
            if tu:
                total_tokens["prompt_tokens"] += int(tu.get("prompt_tokens", 0))
                total_tokens["completion_tokens"] += int(tu.get("completion_tokens", 0))
                total_tokens["total_tokens"] += int(tu.get("total_tokens", 0))
            continue

        answer, token_usage = await handle_question(
            agent,
            question,
            chat_log_path,
            config,
            memory=memory,
            kb_embeds=kb_embeds,
            emb_model=emb_model,
            logger=logger,
        )
        has_issue, reasons = heuristic_critic_has_issue(answer)

        if token_usage:
            total_tokens["prompt_tokens"] += int(token_usage.get("prompt_tokens", 0))
            total_tokens["completion_tokens"] += int(token_usage.get("completion_tokens", 0))
            total_tokens["total_tokens"] += int(token_usage.get("total_tokens", 0))

        if has_issue:
            failed_ids.append(qid)

        results.append(
            {
                "question_id": question["question_id"],
                "attempt": attempt_num,
                "answer": answer,
                "agent_response": answer,
                "heuristic_issue": has_issue,
                "heuristic_reasons": reasons,
                "heuristic_success": not has_issue,
                "token_usage": token_usage if token_usage else {},
            }
        )
        dirty = True

        if completed_map:
            _flush_results(attempt_results_path, results)
            dirty = False

        await asyncio.sleep(0)

    if dirty or not attempt_results_path.exists():
        _flush_results(attempt_results_path, results)

    return results, failed_ids, total_tokens


async def run_experiment(
    config: ExperimentConfig,
    resume_from_dir: str | Path | None = None,
    continue_from_dir: str | Path | None = None,
    resume_through_attempt: int = 1,
) -> None:
    if continue_from_dir:
        experiment_dir = Path(continue_from_dir)
    else:
        base = Path(config.output_base_dir) if config.output_base_dir else Path.cwd()
        experiment_dir = base / config.make_output_dir()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = experiment_dir / "experiment_config.yaml"
    config.to_yaml(cfg_path)

    print(f"\n{'='*60}")
    print(f"  Method: {config.method_label} | Model: {config.model_name}")
    print(f"  HKG: {'ON' if config.enable_hkg else 'OFF'} | "
          f"FAR: {'ON' if config.enable_far else 'OFF'}")
    print(f"  Output: {experiment_dir}")
    print(f"{'='*60}\n")

    # Seed resume files BEFORE logger creation (logger creates .log file on init)
    if resume_from_dir:
        seed_resume_files(
            Path(resume_from_dir), experiment_dir, through_attempt=resume_through_attempt
        )

    llm, mcp_servers = load_config(config, str(experiment_dir))
    if mcp_servers is None:
        mcp_servers = {}

    if config.reflection_model_name:
        co = _chat_openai_class()
        reflection_llm = co(
            model=config.reflection_model_name,
            api_key=config.reflection_api_key or config.resolve_api_key(),
            base_url=config.reflection_base_url or config.base_url,
            temperature=config.reflection_temperature,
            request_timeout=config.reflection_request_timeout,
        )
    else:
        reflection_llm = llm

    compressor_llm = None
    if config.reflection_strategy == "summary_ftro":
        co = _chat_openai_class()
        compressor_llm = co(
            model=config.compressor_model_name or config.reflection_model_name or config.model_name,
            api_key=(
                config.compressor_api_key
                or config.reflection_api_key
                or config.resolve_api_key()
            ),
            base_url=(
                config.compressor_base_url
                or config.reflection_base_url
                or config.base_url
            ),
            temperature=config.compressor_temperature,
            request_timeout=config.compressor_request_timeout,
        )

    created = create_agent(llm, mcp_servers)
    agent, client = await _maybe_await(created)

    memory_path = experiment_dir / "failure_aware_memory.json"
    memory = FailureAwareMemory(file_path=memory_path)

    chat_log_path = experiment_dir / "run.chat"

    logger = create_experiment_logger(
        output_dir=experiment_dir,
        model_name=config.model_name,
        autoplanning=config.autoplanning,
    )

    kb_embeds: dict | None = None
    emb_model = None
    if config.enable_hkg:
        kb_data = load_knowledge_base(config.kb_file_path)
        emb_model = EmbeddingModel(config.model_path)
        kb_embeds = precompute_kb_embeddings(kb_data, emb_model)
        print(f"  HKG: loaded {len(kb_data)} domains, {sum(len(v.get('sub_tasks', [])) for v in kb_embeds.values())} subtasks")

    all_questions = load_questions(config.questions_json)[: config.sample]
    attempt_results_history: list[dict[str, Any]] = []
    failed_ids: list[str] = []
    start_attempt = 1
    continue_completed: list[Any] | None = None

    try:
        if continue_from_dir:
            attempt1_jsonl = experiment_dir / "results_attempt_1.jsonl"
            if attempt1_jsonl.exists():
                completed, error_ids = load_completed_results(attempt1_jsonl)
                if not error_ids:
                    print(f"  All {len(completed)} questions already completed. Nothing to do.")
                    return
                print(f"  Continuing: {len(completed)} completed, {len(error_ids)} to re-run")
                cleanup_chat_after_continue(chat_log_path, set(error_ids))
                for log_f in experiment_dir.glob("*_langchain.log"):
                    cleanup_log_after_continue(log_f, set(error_ids))
                continue_completed = completed

        if resume_from_dir:
            resume_path = Path(resume_from_dir)
            n = resume_through_attempt

            loaded_attempts, failed_ids, old_chat_log = load_attempts_through(resume_path, n)
            attempt_results_history.extend(loaded_attempts)

            if not failed_ids:
                return

            if config.enable_far:
                latest = loaded_attempts[-1]
                failed_results = [r for r in latest["results"] if r.get("heuristic_issue")]
                if n == 1:
                    refl_chat, first_only = old_chat_log, True
                else:
                    refl_chat, first_only = chat_log_path, False
                reflections, reflection_tokens = await generate_reflections_for_failed_questions(
                    llm=reflection_llm,
                    chat_log_path=refl_chat,
                    failed_results=failed_results,
                    attempt_num=n,
                    first_attempt_only=first_only,
                    truncation_strategy=config.truncation_strategy,
                    trajectory_strategy=config.trajectory_strategy,
                    reflection_strategy=config.reflection_strategy,
                    reflection_kwargs=_build_reflection_kwargs(
                        config,
                        compressor_llm=compressor_llm,
                        artifact_dir=experiment_dir / "feedback" / f"attempt_{n}",
                    ),
                    trajectory_kwargs=_build_trajectory_kwargs(config),
                    reflection_failure_policy=config.reflection_failure_policy,
                )
                attempt_results_history[-1]["reflection_token_usage"] = reflection_tokens
                for reflection in reflections:
                    memory.add_reflection(
                        reflection["question_id"],
                        reflection["reflective_text"],
                        n,
                    )
                memory.save()
            start_attempt = n + 1

        for attempt_num in range(start_attempt, config.max_attempts + 1):
            if attempt_num == 1:
                questions_to_eval = all_questions
            else:
                if not failed_ids:
                    break
                failed_set = set(failed_ids)
                questions_to_eval = [
                    q for q in all_questions if str(q["question_id"]) in failed_set
                ]

            attempt_results, failed_ids, attempt_tokens = await run_single_attempt(
                agent,
                questions_to_eval,
                attempt_num,
                chat_log_path,
                config,
                memory=memory,
                kb_embeds=kb_embeds,
                emb_model=emb_model,
                output_dir=experiment_dir,
                logger=logger,
                completed_results=continue_completed if attempt_num == 1 else None,
            )
            if attempt_num == 1:
                continue_completed = None

            attempt_results_history.append(
                {
                    "attempt": attempt_num,
                    "results": attempt_results,
                    "failed_ids": failed_ids,
                    "token_usage": attempt_tokens,
                }
            )

            if failed_ids and attempt_num < config.max_attempts and config.enable_far:
                failed_results = [r for r in attempt_results if r.get("heuristic_issue")]
                reflections, reflection_tokens = await generate_reflections_for_failed_questions(
                    llm=reflection_llm,
                    chat_log_path=chat_log_path,
                    failed_results=failed_results,
                    attempt_num=attempt_num,
                    truncation_strategy=config.truncation_strategy,
                    trajectory_strategy=config.trajectory_strategy,
                    reflection_strategy=config.reflection_strategy,
                    reflection_kwargs=_build_reflection_kwargs(
                        config,
                        compressor_llm=compressor_llm,
                        artifact_dir=experiment_dir / "feedback" / f"attempt_{attempt_num}",
                    ),
                    trajectory_kwargs=_build_trajectory_kwargs(config),
                    reflection_failure_policy=config.reflection_failure_policy,
                )
                attempt_results_history[-1]["reflection_token_usage"] = reflection_tokens
                for reflection in reflections:
                    memory.add_reflection(
                        reflection["question_id"],
                        reflection["reflective_text"],
                        attempt_num,
                    )
                memory.save()

        attempt_results_list: list[list[Any]] = []
        for a in attempt_results_history:
            if isinstance(a, dict) and "results" in a:
                attempt_results_list.append(a["results"])
            elif isinstance(a, list):
                attempt_results_list.append(a)
            else:
                attempt_results_list.append([])

        aggregated_results = aggregate_results(attempt_results_list)
        aggregated_path = experiment_dir / "results_final_aggregated.json"
        with open(aggregated_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_results, f, ensure_ascii=False, indent=2)

        statistics = compute_statistics(
            aggregated_results, attempt_results_history
        )

        summary = {
            "total_questions": len(all_questions),
            "max_attempts": config.max_attempts,
            "attempts_executed": len(attempt_results_history),
            "statistics": statistics,
            "attempt_summaries": [
                {
                    "attempt": a["attempt"],
                    "questions_evaluated": len(a["results"]),
                    "failed_count": len(a["failed_ids"]),
                    "failed_ids": a["failed_ids"],
                }
                for a in attempt_results_history
            ],
        }
        summary_path = experiment_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    finally:
        if client is not None and hasattr(client, "close"):
            close_result = client.close()
            await _maybe_await(close_result)
