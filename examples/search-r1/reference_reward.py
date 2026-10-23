"""Boxed-answer exact match, format rewards and asymmetric tool-use shaping for Search-R1."""
import re
import string


TOOL_CALL_REWARD = 0.1
MAX_TOOL_CALL_REWARD = 0.3
FORMAT_REWARD = 0.1
CORRECT_WITH_TOOL_SCORE = 2.0


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def extract_answer(solution_str):
    # Exclude complete and unfinished tool blocks before extracting answers.
    text = solution_str
    for tag in ("tool_response", "tool_call"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.S)
        text = text.split(f"<{tag}>", 1)[0]
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for end in range(start, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                answer = text[start:end].strip()
                return answer or None
    return None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source is not None and not data_source.startswith("searchR1_"):
        raise ValueError(f"Search reward received unexpected source: {data_source}")
    answers = ground_truth.get("target") if isinstance(ground_truth, dict) else ground_truth
    if isinstance(answers, str):
        answers = [answers]
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    if not isinstance(answers, (list, tuple)) or not answers or not all(isinstance(a, str) for a in answers):
        raise ValueError("ground_truth must contain a nonempty list of answer strings")
    answer = extract_answer(solution_str)
    acc = answer is not None and any(normalize_answer(answer) == normalize_answer(a) for a in answers)
    answer_score = 1.0 if acc else -1.0
    format_reward = FORMAT_REWARD if answer is not None else 0.0
    return {
        "score": answer_score + format_reward,
        "answer_score": answer_score,
        "format_reward": format_reward,
        "acc": bool(acc), "em": float(acc),
        "pred": answer if answer is not None else "[INVALID]",
    }


def compute_score_grpo(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Apply tool-use shaping, with num_turns = 2 + 2 * executed tool calls."""
    result = compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )

    num_turns = 2
    if isinstance(extra_info, dict):
        num_turns = extra_info.get("num_turns", 2)
    try:
        num_turns = int(num_turns)
    except (TypeError, ValueError):
        num_turns = 2

    num_tool_calls = max(0, (num_turns - 2) // 2)
    tool_call_reward = min(MAX_TOOL_CALL_REWARD, num_tool_calls * TOOL_CALL_REWARD)
    if result["score"] > 0:
        if num_tool_calls > 0:
            result["score"] = CORRECT_WITH_TOOL_SCORE
    else:
        result["score"] = min(0.0, result["score"] + tool_call_reward)
    result["tool_call_reward"] = tool_call_reward
    result["num_tool_calls"] = num_tool_calls
    return result
