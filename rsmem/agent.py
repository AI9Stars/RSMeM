"""LangChain MCP agent bootstrap and single-question runner (extracted)."""

from __future__ import annotations

import json
import sys

from pathlib import Path
from typing import Any

from rsmem.config import ExperimentConfig
from rsmem.evaluation import extract_answer_from_response
from rsmem.io import save_chat_message
from rsmem.memory import FailureAwareMemory
from rsmem.retrieval import hybrid_retrieval


_LAZY = object()

ChatOpenAI: Any = _LAZY
MultiServerMCPClient: Any = _LAZY
create_react_agent: Any = _LAZY


def _chat_openai_class() -> Any:
    cls = globals()["ChatOpenAI"]
    if cls is not _LAZY:
        return cls
    from langchain_openai import ChatOpenAI as Impl

    globals()["ChatOpenAI"] = Impl
    return Impl


def _mcp_and_react_factories() -> tuple[Any, Any]:
    mcc = globals()["MultiServerMCPClient"]
    if mcc is _LAZY:
        from langchain_mcp_adapters.client import MultiServerMCPClient as MCPClient

        globals()["MultiServerMCPClient"] = MCPClient
        mcc = MCPClient
    cra = globals()["create_react_agent"]
    if cra is _LAZY:
        from langgraph.prebuilt import create_react_agent as react_agent

        globals()["create_react_agent"] = react_agent
        cra = react_agent
    return mcc, cra


def load_config(config: ExperimentConfig, output_dir: Path | str) -> tuple[Any, dict]:
    """Build ``(llm, mcp_servers)`` from an :class:`ExperimentConfig`.

    The model is configured entirely from the YAML config: ``model_name``,
    ``base_url`` and ``generate_args`` (forwarded as ``extra_body``). The API key
    is resolved by :meth:`ExperimentConfig.resolve_api_key` (from the YAML
    ``api_key``, falling back to the ``"EMPTY"`` sentinel).

    The agent/MCP JSON at ``config.agent_config_path`` only contributes its
    ``mcpServers`` block (tool servers); it does not configure the model. When no
    such file is given, the agent runs tool-free, which keeps the framework
    importable and runnable without the full perception/analysis tool stack.
    """

    out_root = Path(output_dir)

    model_name = config.model_name
    base_url = config.base_url
    generate_args = config.generate_args or None
    raw_mcp_servers: dict = {}

    agent_cfg_path = (config.agent_config_path or "").strip()
    if agent_cfg_path and Path(agent_cfg_path).exists():
        with open(agent_cfg_path, encoding="utf-8") as f:
            agent_cfg = json.load(f)
        raw_mcp_servers = agent_cfg.get("mcpServers", {}) or {}

    api_key = config.resolve_api_key()
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "temperature": 0.1,
        "request_timeout": config.request_timeout,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    if generate_args:
        llm_kwargs["extra_body"] = generate_args

    co = _chat_openai_class()
    llm = co(**llm_kwargs)

    mcp_servers: dict = {}
    for server_name, server_config in raw_mcp_servers.items():
        updated_args = []
        for arg in server_config.get("args", []):
            if isinstance(arg, str) and "tmp/tmp/out" in arg:
                updated_args.append(str(out_root / "out"))
            else:
                updated_args.append(arg)

        command = server_config["command"]
        if command in ("python", "python3"):
            command = sys.executable

        import os
        env = {k: v for k, v in os.environ.items()}
        if server_config.get("env"):
            env.update(server_config["env"])

        mcp_servers[server_name] = {
            "command": command,
            "args": updated_args,
            "transport": "stdio",
            "env": env,
        }

    return llm, mcp_servers


async def create_agent(llm, mcp_servers) -> tuple[Any, Any]:
    """Create ReAct agent with MCP tools. Returns ``(agent, client)``."""

    MCPClient, react_agent_factory = _mcp_and_react_factories()
    client = MCPClient(mcp_servers)

    try:
        tools = await client.get_tools()
        agent = react_agent_factory(llm, tools)
        return agent, client
    except Exception:
        if hasattr(client, "close"):
            close_result = client.close()
            if hasattr(close_result, "__await__"):
                await close_result
        raise


def _log_info(logger: Any | None, *args: Any) -> None:
    if logger is None:
        return
    info_fn = getattr(logger, "info", None)
    if callable(info_fn):
        info_fn(*args)


def _normalize_token_usage(t: dict[str, int]) -> dict[str, int]:
    if t["total_tokens"] == 0 and (
        t["prompt_tokens"] > 0 or t["completion_tokens"] > 0
    ):
        t["total_tokens"] = t["prompt_tokens"] + t["completion_tokens"]
    return t


def extract_token_usage_from_response(response: Any) -> dict[str, int]:
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    if hasattr(response, "usage_metadata"):
        um = getattr(response, "usage_metadata")
        token_usage = {
            "prompt_tokens": getattr(um, "prompt_tokens", 0),
            "completion_tokens": getattr(um, "completion_tokens", 0),
            "total_tokens": getattr(um, "total_tokens", 0),
        }
    elif isinstance(response, dict) and "usage_metadata" in response:
        metadata = response["usage_metadata"]
        if isinstance(metadata, dict):
            token_usage = {
                "prompt_tokens": metadata.get("prompt_tokens", 0),
                "completion_tokens": metadata.get("completion_tokens", 0),
                "total_tokens": metadata.get("total_tokens", 0),
            }
    else:
        msgs = response.get("messages", []) if isinstance(response, dict) else []
        for message in msgs:
            if hasattr(message, "type") and getattr(message, "type", None) == "ai":
                rm = getattr(message, "response_metadata", None)
                if isinstance(rm, dict) and (
                    "token_usage" in rm or "usage_metadata" in rm
                ):
                    if "token_usage" in rm:
                        tu = rm["token_usage"]
                        token_usage["prompt_tokens"] += int(
                            tu.get("prompt_tokens", 0)
                        )
                        token_usage["completion_tokens"] += int(
                            tu.get("completion_tokens", 0)
                        )
                        token_usage["total_tokens"] += int(tu.get("total_tokens", 0))
                    else:
                        um = rm["usage_metadata"]
                        token_usage["prompt_tokens"] += int(
                            um.get("input_tokens", um.get("prompt_tokens", 0))
                        )
                        token_usage["completion_tokens"] += int(
                            um.get("output_tokens", um.get("completion_tokens", 0))
                        )
                        token_usage["total_tokens"] += int(um.get("total_tokens", 0))
                elif hasattr(message, "usage_metadata"):
                    um = message.usage_metadata
                    if hasattr(um, "input_tokens"):
                        token_usage["prompt_tokens"] += int(um.input_tokens)
                        token_usage["completion_tokens"] += int(um.output_tokens)
                        token_usage["total_tokens"] += (
                            int(um.input_tokens) + int(um.output_tokens)
                        )
                    elif hasattr(um, "prompt_tokens"):
                        token_usage["prompt_tokens"] += int(um.prompt_tokens)
                        token_usage["completion_tokens"] += int(
                            um.completion_tokens
                        )
                        token_usage["total_tokens"] += int(um.total_tokens)

    return _normalize_token_usage(token_usage)


async def handle_question(
    agent,
    question: dict,
    chat_log_path: Path,
    config: ExperimentConfig,
    memory: FailureAwareMemory | None = None,
    kb_embeds: dict | None = None,
    emb_model=None,
    logger=None,
) -> tuple[str, dict]:
    """Handle a single question. Returns ``(answer, token_usage)``."""

    qid_str = str(question["question_id"])

    try:
        data_str = (
            question["data"]
            if isinstance(question["data"], str)
            else json.dumps(question["data"], ensure_ascii=False)
        )
        base_query = (
            question["auto"] + data_str
            if config.autoplanning
            else question["instruct"] + data_str
        )

        retrieved_context = ""
        if config.enable_hkg and kb_embeds and emb_model:
            retrieval_results = hybrid_retrieval(
                query=base_query,
                kb_embeds=kb_embeds,
                emb_model=emb_model,
                top_k=1,
            )
            if retrieval_results:
                result = retrieval_results[0]
                summary = result.get("summary", "")[:1000].strip()
                retrieved_context = (
                    f"**Domain**: {result.get('domain', 'Unknown')}\n"
                    f"**Sub-task**: {result.get('sub_title', 'General')}\n"
                    f"**Core Methodology**: {summary}\n"
                    f"**Execution Rule**: Prioritize the above logic while dynamically adjusting "
                    f"file paths and parameters according to the current environment."
                )

        reflection_block = ""
        if config.enable_far and memory is not None:
            reflection_text = memory.get_latest_reflection(question["question_id"])
            if reflection_text:
                reflection_block = f"""### Previous Attempt Reflection\n{reflection_text}\n"""

        base_question_text = (
            question["auto"] if config.autoplanning else question["instruct"]
        )
        data_path = question["data"]
        question_body = f"### Question\n{base_question_text}\n\n### Data Path\n{data_path}"

        choices = question.get("choices")
        if choices and isinstance(choices, list):
            choices_text = "\n".join(
                f"{chr(ord('A') + i)}. {choice}"
                for i, choice in enumerate(choices)
            )
            question_body += f"\n\n### Options\n{choices_text}"

        full_query = f"{config.resolve_sys_prompt()}\n\n{question_body}"
        if retrieved_context:
            full_query += f"\n\n### Reference Knowledge (Priority Execution)\n{retrieved_context}"
        if reflection_block:
            full_query += f"\n\n{reflection_block}"

        save_chat_message(
            chat_log_path,
            "user",
            full_query,
            qid_str,
            metadata={"retrieved_context": retrieved_context},
        )

        from langchain_core.messages import HumanMessage

        response = await agent.ainvoke(
            {"messages": [HumanMessage(content=full_query)]},
            config={
                "recursion_limit": 50,
                "max_execution_time": 300,
            },
        )

        final_answer = extract_answer_from_response(response)

        conversation_log = []
        for message in response.get("messages", []):
            if not hasattr(message, "type"):
                continue
            mtype = message.type
            if mtype == "human":
                conversation_log.append({"role": "user", "content": message.content})
            elif mtype == "ai":
                assistant_content: list[Any] = []
                if message.content and str(message.content).strip():
                    assistant_content.append(
                        {"type": "text", "content": message.content}
                    )
                aks = getattr(message, "additional_kwargs", None) or {}
                tool_calls = aks.get("tool_calls") if isinstance(aks, dict) else None
                if tool_calls:
                    for tool_call in tool_calls:
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except Exception:
                            arguments = tool_call["function"]["arguments"]
                        assistant_content.append(
                            {
                                "name": tool_call["function"]["name"],
                                "input": arguments,
                            }
                        )
                if assistant_content:
                    conversation_log.append(
                        {"role": "assistant", "content": assistant_content}
                    )
            elif mtype == "tool":
                conversation_log.append(
                    {
                        "role": "tool",
                        "name": message.name,
                        "content": [{"output": [{"text": str(message.content)}]}],
                    }
                )

        for message in response.get("messages", []):
            if not hasattr(message, "type"):
                continue
            mtype = message.type
            if mtype == "human":
                continue
            if mtype == "ai":
                assistant_chat_content: list[Any] = []
                if message.content and str(message.content).strip():
                    assistant_chat_content.append(
                        {"type": "text", "text": message.content}
                    )
                aks = getattr(message, "additional_kwargs", None) or {}
                tool_calls = aks.get("tool_calls") if isinstance(aks, dict) else None
                if tool_calls:
                    for tool_call in tool_calls:
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except Exception:
                            arguments = tool_call["function"]["arguments"]
                        assistant_chat_content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call["id"],
                                "name": tool_call["function"]["name"],
                                "input": arguments,
                            }
                        )
                if assistant_chat_content:
                    save_chat_message(
                        chat_log_path,
                        "assistant",
                        assistant_chat_content,
                        qid_str,
                        metadata=None,
                    )
            elif mtype == "tool":
                save_chat_message(
                    chat_log_path,
                    "system",
                    [
                        {
                            "type": "tool_result",
                            "id": getattr(message, "tool_call_id", "unknown"),
                            "output": [
                                {
                                    "type": "text",
                                    "text": str(message.content),
                                    "annotations": None,
                                    "meta": None,
                                }
                            ],
                            "name": message.name,
                        }
                    ],
                    qid_str,
                    metadata=None,
                )

        _log_info(
            logger, "Chat Content", question["question_id"], conversation_log, final_answer
        )

        token_usage = extract_token_usage_from_response(response)

        return final_answer, token_usage

    except Exception as e:
        error_msg = f"Error processing question {qid_str}: {e}"
        save_chat_message(
            chat_log_path,
            "system",
            [{"type": "text", "content": error_msg}],
            qid_str,
            metadata={"error": True},
        )

        _log_info(logger, "Error", question["question_id"], [], error_msg)
        return f"Error: {e}", {}
