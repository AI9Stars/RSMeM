from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Default resource paths are relative to the project root so the package stays
# portable. Override any of them through the YAML config file.
_DEFAULT_KB = "knowledge_base/knowledge_copyfile.json"
_DEFAULT_MODEL_EMB = "models/all-mpnet-base-v2"
_DEFAULT_QUESTIONS = "benchmark/question.json"

_DEFAULT_OUTPUT_DIR_TEMPLATE = (
    "{timestamp}_{method_label}_{model_name}_n{sample}_r{repeat}"
)


_SYS_PROMPT_RS_MEM = """### Core Role & Instructions
You are a professional geoscientist specializing in Earth observation data analysis. Follow these rules STRICTLY:
1. Tool Usage: Use designated tools to solve the problem step by step; retry ONLY ONCE if a tool returns an error.
2. Path Rule: When a tool returns "Result saved at /path/to/file", you must use the FULL path "/path/to/file" in all subsequent tool calls.
3. Answer Format: Output ONLY the correct choice in this format - <Answer>Letter<Answer> (e.g., <Answer>A<Answer>).
   - Do NOT add explanations, reasoning, or extra text outside the <Answer> tags.
   - Do NOT change the answer format under any circumstances.
"""

_SYS_PROMPT_BASELINE = """
You are a geoscientist, and you need to use tools to answer multiple-choice questions about Earth observation data analysis. Note that if a tool returns an error, you can only try again once. Ultimately, you only need to explicitly tell me the correct choice.
ATTENTION:
1. When a tool returns "Result saved at /path/to/file", you must use the full returned path "/path/to/file" in all subsequent tool calls.
2. For each question, you must provide the choice you think is most appropriate.Don't gibe me another format. Your final answer format must be:
<Answer>Your choice<Answer>
"""

_REFLECTION_PROMPT = """You are an advanced reasoning agent that improves from self-reflection (Reflexion-style).
You will be given:
1) The original question (includes data path and multiple-choice options),
2) Retrieved knowledge context (may be empty),
3) A compressed trajectory from the previous run (tool_use + tool_result + key assistant text),


Your task:
- In 3-5 sentences (100-300 words), diagnose the most likely reason for failure and devise a new, concise, high-level plan that aims to avoid the same failure. Focus on actionable fixes specific to this question.
- Output ONLY a valid JSON object with the required keys. No markdown, no extra text.

Required JSON schema:
{{
  "question_id": "{question_id}",
  "reflective_text": "A concise reflection to prepend/append in the next attempt (a few sentences)."
}}

Inputs:
- question_id: {question_id}
- attempt: {attempt}

Previous trial:
Question:
{question_text}

Retrieved context:
{retrieved_context}

Compressed trajectory (chronological):
{trajectory_json}

Now output the JSON only:
"""


@dataclass
class ExperimentConfig:
    enable_hkg: bool = True  # Hierarchical Knowledge Grounding (HKG)
    enable_far: bool = True
    model_name: str = "deepseek-chat"
    autoplanning: bool = True
    sample: int = 248
    repeat: int = 1
    kb_file_path: str = _DEFAULT_KB
    model_path: str = _DEFAULT_MODEL_EMB
    questions_json: str = _DEFAULT_QUESTIONS
    sys_prompt: str = ""  # empty → resolved from method by resolve_sys_prompt()
    reflection_experience_prompt: str = _REFLECTION_PROMPT
    debug_token_extraction: bool = True
    output_dir_template: str = _DEFAULT_OUTPUT_DIR_TEMPLATE
    truncation_strategy: str = "default"
    trajectory_strategy: str = "full_tool_result_only"
    reflection_strategy: str = "default"
    base_url: str = ""
    api_key: str = ""
    request_timeout: float = 120.0
    generate_args: dict = field(default_factory=dict)  # extra OpenAI body, e.g. {"enable_thinking": False}
    agent_config_path: str = ""
    output_base_dir: str = ""
    max_attempts: int = 3
    reflection_model_name: str = ""
    reflection_base_url: str = ""
    reflection_api_key: str = ""
    reflection_request_timeout: float = 120.0
    reflection_temperature: float = 0.1
    compressor_model_name: str = ""
    compressor_base_url: str = ""
    compressor_api_key: str = ""
    compressor_request_timeout: float = 180.0
    compressor_temperature: float = 0.0
    summary_max_step_chars: int = 6000
    summary_max_trace_chars: int = 120000
    reflection_failure_policy: str = "fallback"

    def __post_init__(self) -> None:
        if self.model_name is None or (
            isinstance(self.model_name, str) and not self.model_name.strip()
        ):
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.model_name, str):
            raise ValueError("model_name must be a non-empty string")
        self.model_name = self.model_name.strip()
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if self.reflection_request_timeout <= 0:
            raise ValueError("reflection_request_timeout must be positive")
        if self.compressor_request_timeout <= 0:
            raise ValueError("compressor_request_timeout must be positive")
        if self.summary_max_step_chars <= 0 or self.summary_max_trace_chars <= 0:
            raise ValueError("summary trajectory limits must be positive")
        if self.reflection_failure_policy not in {"fallback", "error"}:
            raise ValueError("reflection_failure_policy must be 'fallback' or 'error'")
        if self.reflection_strategy == "summary_ftro":
            if self.trajectory_strategy != "summary_ftro_source":
                raise ValueError(
                    "reflection_strategy='summary_ftro' requires "
                    "trajectory_strategy='summary_ftro_source'"
                )
            if self.reflection_failure_policy != "error":
                raise ValueError(
                    "summary_ftro requires reflection_failure_policy='error'"
                )

    def resolve_api_key(self) -> str:
        """Resolve the LLM API key.

        The key comes from the YAML ``api_key`` field; when empty, the ``"EMPTY"``
        sentinel is used (accepted by OpenAI-compatible servers that ignore the key).
        """
        return self.api_key or "EMPTY"

    def resolve_sys_prompt(self) -> str:
        """Resolve the system prompt for this run.

        Precedence: an explicit ``sys_prompt`` from the YAML config wins; otherwise
        a plain ``baseline`` run uses the baseline prompt, and every other variant
        (HKG / FAR / RSMeM) uses the RSMeM prompt.
        """
        if self.sys_prompt and self.sys_prompt.strip():
            return self.sys_prompt
        if self.method_label == "baseline":
            return _SYS_PROMPT_BASELINE
        return _SYS_PROMPT_RS_MEM

    @property
    def method_label(self) -> str:
        if self.enable_hkg and self.enable_far:
            base = "RSMeM"
        elif not self.enable_hkg and not self.enable_far:
            base = "baseline"
        elif self.enable_hkg:
            base = "HKG"
        else:
            base = "FAR"
        return base

    def make_output_dir(self, timestamp: str | None = None) -> str:
        ts = timestamp
        if ts is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir_template.format(
            timestamp=ts,
            method_label=self.method_label,
            model_name=self.model_name,
            sample=self.sample,
            repeat=self.repeat,
        )

    def to_yaml(self, path: str | Path) -> None:
        _sensitive = {
            "api_key", "base_url", "reflection_api_key", "reflection_base_url",
            "compressor_api_key", "compressor_base_url",
        }
        p = Path(path)
        data = {k: v for k, v in asdict(self).items() if k not in _sensitive}
        p.write_text(
            yaml.safe_dump(
                data, sort_keys=False, allow_unicode=True, default_flow_style=False
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw if isinstance(raw, dict) else {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        field_names = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - field_names)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Unknown configuration field(s): {names}")
        base = asdict(cls())
        updates = {k: v for k, v in data.items() if k in field_names}
        merged = {**base, **updates}
        return cls(**merged)
