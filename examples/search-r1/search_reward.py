"""Slime reward callback for Search-R1 trajectories."""

import os

from reference_reward import compute_score, compute_score_grpo


def _ground_truth(label):
    # The prepared datasets may select the whole reward_model column as label.
    while isinstance(label, dict) and "target" not in label:
        if "reward_model" in label:
            label = label["reward_model"]
        elif "ground_truth" in label:
            label = label["ground_truth"]
        else:
            break
    return label


async def reward_func(args, sample, **kwargs):
    """Score assistant answer text, counting only tool calls with appended responses."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    search_metadata = metadata.get("search_r1", {})
    if not isinstance(search_metadata, dict):
        search_metadata = {}
    calls = search_metadata.get("num_tool_calls", 0)
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
        raise ValueError("search_r1.num_tool_calls must be a nonnegative integer")
    mode = os.environ.get("SEARCH_REWARD_MODE", "grpo").strip().lower()
    if mode not in ("grpo", "base"):
        raise ValueError("SEARCH_REWARD_MODE must be grpo or base")
    source = metadata.get("data_source", search_metadata.get("data_source"))
    protocol = search_metadata.get("protocol", os.environ.get("SEARCH_PROTOCOL", "qwen"))
    if protocol not in ("qwen", "llama", "llama3"):
        raise ValueError("Unknown Search protocol for reward extraction")
    if "reward_response" in search_metadata:
        solution = search_metadata["reward_response"]
        if not isinstance(solution, str):
            raise ValueError("search_r1.reward_response must be assistant text")
    elif protocol in ("llama", "llama3"):
        # Flattened Llama text cannot reliably separate answers from evidence.
        raise ValueError("Llama Search reward requires assistant-only reward_response from the rollout")
    else:
        # Qwen responses retain XML tool boundaries that the extractor removes.
        solution = sample.response
    score_fn = compute_score_grpo if mode == "grpo" else compute_score
    result = score_fn(
        data_source=source,
        solution_str=solution,
        ground_truth=_ground_truth(sample.label),
        extra_info={"num_turns": 2 + 2 * calls},
    )
    if mode == "base":
        result["tool_call_reward"] = 0.0
        result["num_tool_calls"] = calls
    return result
