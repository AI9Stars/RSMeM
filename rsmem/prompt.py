"""Prompt template construction for RSMeM agents."""

DEFAULT_SYS_PROMPT = """### Core Role & Instructions
You are a professional geoscientist specializing in Earth observation data analysis. Follow these rules STRICTLY:
1. Tool Usage: Use designated tools to solve the problem step by step; retry ONLY ONCE if a tool returns an error.
2. Path Rule: When a tool returns "Result saved at /path/to/file", you must use the FULL path "/path/to/file" in all subsequent tool calls.
3. Answer Format: Output ONLY the correct choice in this format - <Answer>Letter<Answer> (e.g., <Answer>A<Answer>).
   - Do NOT add explanations, reasoning, or extra text outside the <Answer> tags.
   - Do NOT change the answer format under any circumstances.
"""

DEFAULT_REFLECTION_TEMPLATE = """You are an advanced reasoning agent that improves from self-reflection (Reflexion-style).
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

def build_agent_query(
    question_text: str,
    retrieved_context: str = "",
    reflection_block: str = "",
    data_info: str = "",
) -> str:
    """Build the full query sent to the agent.

    Combines question text with optional retrieved context, reflection, and data info.
    """
    body = f"### Question\n{question_text}"
    if data_info.strip():
        body += f"\n\n### Data Path\n{data_info}"

    parts: list[str] = [body]

    ctx = retrieved_context.strip() if retrieved_context else ""
    if ctx:
        parts.append(f"### Reference Knowledge (Priority Execution)\n{retrieved_context}")

    refl = reflection_block.strip() if reflection_block else ""
    if refl:
        parts.append(reflection_block)

    return "\n\n".join(parts)


def build_reflection_prompt(
    question_id: str,
    attempt: int,
    question_text: str,
    retrieved_context: str,
    trajectory_json: str,
    template: str = DEFAULT_REFLECTION_TEMPLATE,
) -> str:
    """Build the reflection generation prompt."""
    kwargs = dict(
        question_id=question_id,
        attempt=attempt,
        question_text=question_text,
        retrieved_context=retrieved_context,
        trajectory_json=trajectory_json,
    )
    return template.format(**kwargs)
