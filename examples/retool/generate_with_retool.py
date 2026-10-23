# Modified for COPC.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/volcengine/verl/blob/cb809d66e46dfd3342d008628891a14a054fa424/recipe/retool/retool.py
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

try:
    from slime.rollout.rm_hub.math_dapo_utils import (
        compute_score as math_dapo_compute_score,
        last_boxed_only_string,
        normalize_final_answer,
        remove_boxed,
    )
except ImportError as e:
    raise ImportError("MathDapo is not installed") from e


def _math_verify_score(solution_str: str, ground_truth: str) -> tuple[float, str]:
    """Check full-text mathematical equivalence, falling back to normalized boxed matching."""

    gt_boxed = "\\boxed{" + str(ground_truth) + "}"
    try:
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        verify = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        with contextlib.suppress(Exception):
            score, pred = verify([gt_boxed], [solution_str])
            return float(score), (pred or "")
    except ImportError:
        pass

    boxed = last_boxed_only_string(solution_str)
    pred = remove_boxed(boxed) if boxed else None
    if pred is None:
        return 0.0, ""
    pred_norm = normalize_final_answer(pred)
    try:
        gt_norm = str(int(float(ground_truth)))
    except (ValueError, TypeError):
        gt_norm = normalize_final_answer(str(ground_truth))
    return (1.0 if pred_norm == gt_norm else 0.0), (pred or "")

from tool_sandbox import SEMAPHORE, TOOL_CONFIGS, tool_registry

from slime.utils.retool_protocol import (
    PROTOCOL_VERSION,
    parse_assistant_action,
    render_initial_prompt,
    render_tool_response_suffix,
)


RUNTIME_COMPAT_VERSION = "legacy-tolerant-v1"
INVALID_ACTION_FEEDBACK = (
    "The previous assistant action could not be parsed. To execute Python, "
    'return <tool_call>{"name":"code_interpreter","arguments":{"code":"YOUR_PYTHON_CODE"}}</tool_call> '
    "with complete JSON, or use a complete <code>...</code> or Python code block. "
    "To give the final answer, put it in a LaTeX boxed expression. Please try again."
)


def _reset_generation(sample):
    """Clear response-side state before restarting an aborted trajectory."""
    sample.status = Sample.Status.PENDING
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    sample.teacher_log_probs = None
    sample.weight_versions = []
    sample.reward = None
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.tool_call_count = 0
    if hasattr(sample, "SpecInfo"):
        sample.spec_info = sample.SpecInfo()
    if hasattr(sample, "PrefixCacheInfo"):
        sample.prefix_cache_info = sample.PrefixCacheInfo()
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.pop("retool_error", None)
    sample.metadata.pop("retool_async_checkpoint_state", None)
    sample.metadata["tool_call_count"] = 0
    sample.metadata["retool_protocol"] = PROTOCOL_VERSION
    sample.metadata["retool_runtime_compat"] = RUNTIME_COMPAT_VERSION
    sample.metadata["retool_invalid_action_count"] = 0
    sample.metadata.pop("retool_last_invalid_action", None)
    sample.metadata["retool_inserted_end_count"] = 0


def _fail(sample, message, *, aborted=False):
    sample.status = Sample.Status.ABORTED if aborted else Sample.Status.FAILED
    sample.metadata["retool_error"] = message


ASYNC_CURSOR_KEY = "retool_async_checkpoint_state"
ASYNC_CURSOR_VERSION = 1


def _checkpoint_request_seed(args, sample, turn):
    """Stable request seeds; independent of Python hash and coroutine order."""
    if getattr(sample, "index", None) is None or getattr(sample, "group_index", None) is None:
        raise ValueError("Checkpointed ReTool requires stable sample.index and sample.group_index.")
    identity = ["retool-request-v1", int(getattr(args, "rollout_seed", 42)),
                int(sample.group_index), int(sample.index), int(turn)]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _validate_checkpoint_weight_version(meta):
    """Require the actual engine version without rewriting its representation."""
    version = meta.get("weight_version")
    numeric = type(version) is int and version >= 0
    if isinstance(version, str):
        numeric = bool(version) and version.isascii() and version.isdigit()
    if not numeric:
        raise ValueError(
            "Checkpointed ReTool requires a non-negative numeric meta_info.weight_version "
            "on every generation response; missing or invalid engine versions cannot be inferred."
        )


def _sync_checkpoint_sample(sample, cursor, tokenizer):
    """Publish a complete, internally aligned sample before a safe point."""
    prompt_length = cursor["prompt_length"]
    response_ids = sample.tokens[prompt_length:]
    sample.response = tokenizer.decode(response_ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
    sample.response_length = len(response_ids)
    sample.tool_call_count = cursor["tool_call_count"]
    sample.metadata["tool_call_count"] = cursor["tool_call_count"]
    sample.payload_text = cursor["prompt"] + sample.response
    sample.payload_has_system = "<|im_start|>system" in cursor["prompt"]
    sample.payload_has_tools = "# Tools" in cursor["prompt"]
    sample._validate_response_metadata_lengths()


async def _generate_checkpointed(args, sample, sampling_params):
    """Generate resumable trajectories, parking only after completed model/tool requests commit."""
    from slime.rollout.checkpoint_barrier import checkpoint_safe_point

    # Register before validation so even untouched samples can park.
    await checkpoint_safe_point(sample)
    if args.partial_rollout:
        raise ValueError("Partial rollout is not supported by the ReTool generator.")
    cursor = (sample.metadata or {}).get(ASYNC_CURSOR_KEY)
    if cursor is None and sample.status in {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}:
        return sample
    state = GenerateState(args)
    tokenizer = state.tokenizer
    if cursor is not None:
        if cursor.get("schema_version") != ASYNC_CURSOR_VERSION:
            raise ValueError("Unsupported ReTool async checkpoint cursor version.")
        if cursor["stage"] not in {"generate", "tools", "observe", "done"}:
            raise ValueError("Invalid ReTool async checkpoint stage.")
        if len(sample.tokens) - cursor["prompt_length"] != sample.response_length:
            raise ValueError("ReTool checkpoint cursor and response length disagree.")
        sample._validate_response_metadata_lengths()
        if cursor["stage"] == "done":
            return sample
        if sample.status == Sample.Status.ABORTED:
            sample.status = Sample.Status.PENDING
            sample.metadata.pop("retool_error", None)
    else:
        _reset_generation(sample)
        raw_prompt = getattr(sample, "raw_prompt", None)
        if raw_prompt is None:
            raw_prompt = sample.prompt
        template_kwargs = getattr(sample, "apply_chat_template_kwargs", None) or {}
        prompt = render_initial_prompt(tokenizer, raw_prompt, tools=tool_registry.get_tool_specs(),
                                       chat_template_kwargs=template_kwargs)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        sample.tokens = list(prompt_ids)
        if not isinstance(sample.prompt, str):
            sample.prompt = prompt
        response_limit = sampling_params.get("max_new_tokens")
        if response_limit is None:
            response_limit = args.rollout_max_response_len
        if response_limit <= 0:
            raise ValueError("The ReTool response token budget must be positive.")
        max_length = len(prompt_ids) + response_limit
        if args.rollout_max_context_len is not None:
            if len(prompt_ids) >= args.rollout_max_context_len:
                raise ValueError("The formatted ReTool prompt, including its tool schema, must be "
                                 "shorter than rollout_max_context_len.")
            max_length = min(max_length, args.rollout_max_context_len)
        max_turns, max_tool_calls = TOOL_CONFIGS["max_turns"], TOOL_CONFIGS["max_tool_calls"]
        if max_turns <= 0:
            raise ValueError("The ReTool turn limit must be positive.")
        if max_tool_calls < 0:
            raise ValueError("The ReTool tool-call limit must be non-negative.")
        assistant_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(assistant_end_id, int) or tokenizer.decode(
            [assistant_end_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        ) != "<|im_end|>":
            raise ValueError("The ReTool tokenizer must provide the native <|im_end|> token.")
        cursor = {
            "schema_version": ASYNC_CURSOR_VERSION, "stage": "generate", "turn": 0,
            "prompt": prompt, "prompt_length": len(prompt_ids), "max_length": max_length,
            "max_turns": max_turns, "max_tool_calls": max_tool_calls, "tool_call_count": 0,
            "reasoning_prefilled": prompt.rstrip().endswith("<think>"),
            "assistant_end_id": assistant_end_id, "template_kwargs": dict(template_kwargs),
            "sampling_params": dict(sampling_params), "calls": [], "call_index": 0,
            "results": [], "call_budget_exhausted": False,
            "sampling_seeds": {},
        }
        json.dumps(cursor)
        sample.metadata[ASYNC_CURSOR_KEY] = cursor

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    try:
        while cursor["stage"] != "done":
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                break
            if cursor["stage"] == "generate":
                remaining = cursor["max_length"] - len(sample.tokens)
                if cursor["turn"] >= cursor["max_turns"] or remaining <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    cursor["stage"] = "done"
                    break
                seed_key = str(cursor["turn"])
                if seed_key not in cursor["sampling_seeds"]:
                    cursor["sampling_seeds"][seed_key] = _checkpoint_request_seed(args, sample, cursor["turn"])
                params = dict(cursor["sampling_params"])
                params.update(max_new_tokens=remaining, no_stop_trim=True,
                              sampling_seed=cursor["sampling_seeds"][seed_key])
                _sync_checkpoint_sample(sample, cursor, tokenizer)
                await checkpoint_safe_point(sample)
                if state.aborted:
                    sample.status = Sample.Status.ABORTED
                    break
                try:
                    output = await post(url, {"input_ids": list(sample.tokens),
                                              "sampling_params": params, "return_logprob": True})
                except Exception as exc:
                    _fail(sample, "Generation request failed: " + type(exc).__name__, aborted=True)
                    break
                meta = output.get("meta_info") if isinstance(output, dict) else None
                finish_info = meta.get("finish_reason") if isinstance(meta, dict) else None
                if not isinstance(finish_info, dict):
                    _fail(sample, "SGLang did not return a finish reason.", aborted=True)
                    break
                finish = finish_info.get("type")
                if state.aborted or finish == "abort":
                    sample.status = Sample.Status.ABORTED
                    break
                _validate_checkpoint_weight_version(meta)
                token_logprobs = meta.get("output_token_logprobs")
                if token_logprobs is None:
                    _fail(sample, "SGLang did not return output token log probabilities.", aborted=True)
                    break
                if not token_logprobs:
                    _fail(sample, "SGLang returned an empty assistant turn.", aborted=True)
                    break
                try:
                    turn_ids = [item[1] for item in token_logprobs]
                    turn_logprobs = [item[0] for item in token_logprobs]
                    if any(not isinstance(token, int) for token in turn_ids):
                        raise ValueError("Invalid token ID")
                    if any(value is None or not math.isfinite(value) for value in turn_logprobs):
                        raise ValueError("Invalid token log probability")
                except (IndexError, TypeError, ValueError):
                    _fail(sample, "SGLang returned invalid token/log-probability pairs.", aborted=True)
                    break
                if len(turn_ids) > remaining:
                    _fail(sample, "SGLang exceeded the requested token budget.", aborted=True)
                    break
                sample.append_response_tokens(args, tokens=turn_ids, log_probs=turn_logprobs,
                                              trainable=True, meta_info=meta)
                sample.status = Sample.Status.PENDING
                cursor["turn"] += 1
                if finish == "length":
                    sample.status = Sample.Status.TRUNCATED
                    cursor["stage"] = "done"
                    continue
                if finish != "stop":
                    _fail(sample, "Unsupported SGLang finish reason: " + str(finish))
                    cursor["stage"] = "done"
                    continue
                turn_text = tokenizer.decode(turn_ids, skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
                action = parse_assistant_action(turn_text, reasoning_prefilled=cursor["reasoning_prefilled"],
                                                runtime_compat=True)
                if action.kind == "answer":
                    sample.status = Sample.Status.COMPLETED
                    cursor["stage"] = "done"
                    continue
                if len(sample.tokens) >= cursor["max_length"]:
                    sample.status = Sample.Status.TRUNCATED
                    cursor["stage"] = "done"
                    continue
                cursor.update(stage="tools", calls=[{"name": call.name, "arguments": call.arguments}
                                                     for call in action.calls], call_index=0,
                              results=[], call_budget_exhausted=False, turn_end_id=turn_ids[-1])
                if action.kind != "tool":
                    sample.metadata["retool_invalid_action_count"] += 1
                    sample.metadata["retool_last_invalid_action"] = action.error
                    cursor["results"].append(INVALID_ACTION_FEEDBACK)
                continue

            if cursor["stage"] == "tools":
                if cursor["call_index"] >= len(cursor["calls"]):
                    cursor["stage"] = "observe"
                    continue
                if cursor["tool_call_count"] >= cursor["max_tool_calls"]:
                    cursor.update(stage="observe", call_budget_exhausted=True)
                    continue
                _sync_checkpoint_sample(sample, cursor, tokenizer)
                await checkpoint_safe_point(sample)
                if state.aborted:
                    sample.status = Sample.Status.ABORTED
                    break
                call = cursor["calls"][cursor["call_index"]]
                async with SEMAPHORE:
                    if state.aborted:
                        sample.status = Sample.Status.ABORTED
                        break
                    try:
                        result = await tool_registry.execute_tool(call["name"], call["arguments"])
                    except Exception as exc:
                        result = "Error: tool execution failed (" + type(exc).__name__ + ")."
                # Commit before awaiting to avoid replaying a completed tool call.
                cursor["tool_call_count"] += 1
                cursor["results"].append(str(result))
                cursor["call_index"] += 1
                continue

            if cursor["stage"] == "observe":
                if cursor["results"]:
                    try:
                        suffix = render_tool_response_suffix(tokenizer, cursor["results"],
                                                             chat_template_kwargs=cursor["template_kwargs"])
                        if cursor["turn_end_id"] != cursor["assistant_end_id"]:
                            suffix = "<|im_end|>" + suffix
                            sample.metadata["retool_inserted_end_count"] += 1
                    except ValueError as exc:
                        _fail(sample, str(exc))
                        cursor["stage"] = "done"
                        continue
                    cursor["reasoning_prefilled"] = suffix.rstrip().endswith("<think>")
                    observation_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                    remaining = cursor["max_length"] - len(sample.tokens)
                    overflow = len(observation_ids) > remaining
                    sample.append_response_tokens(args, tokens=observation_ids[:max(0, remaining)], trainable=False)
                    if overflow:
                        sample.status = Sample.Status.TRUNCATED
                        cursor["stage"] = "done"
                        continue
                if cursor["call_budget_exhausted"]:
                    sample.status = Sample.Status.TRUNCATED
                    cursor["stage"] = "done"
                    continue
                cursor.update(stage="generate", calls=[], results=[], call_index=0)
    except asyncio.CancelledError:
        sample.status = Sample.Status.ABORTED
        raise
    finally:
        _sync_checkpoint_sample(sample, cursor, tokenizer)
    return sample


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Generate Qwen tool turns while preserving sampled tokens, logprobs and masks."""
    if getattr(args, "async_checkpoint", False):
        return await _generate_checkpointed(args, sample, sampling_params)
    if args.partial_rollout:
        raise ValueError("Partial rollout is not supported by the ReTool generator.")
    _reset_generation(sample)
    state = GenerateState(args)
    tokenizer = state.tokenizer
    raw_prompt = getattr(sample, "raw_prompt", None)
    if raw_prompt is None:
        raw_prompt = sample.prompt
    template_kwargs = getattr(sample, "apply_chat_template_kwargs", None) or {}
    tool_specs = tool_registry.get_tool_specs()
    prompt = render_initial_prompt(
        tokenizer, raw_prompt, tools=tool_specs, chat_template_kwargs=template_kwargs
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    sample.tokens = list(prompt_ids)
    if not isinstance(sample.prompt, str):
        sample.prompt = prompt
    response_ids = []
    reasoning_prefilled = prompt.rstrip().endswith("<think>")
    tool_call_count = 0
    response_limit = sampling_params.get("max_new_tokens")
    if response_limit is None:
        response_limit = args.rollout_max_response_len
    if response_limit <= 0:
        raise ValueError("The ReTool response token budget must be positive.")
    max_length = len(prompt_ids) + response_limit
    if args.rollout_max_context_len is not None:
        if len(prompt_ids) >= args.rollout_max_context_len:
            raise ValueError(
                "The formatted ReTool prompt, including its tool schema, must be "
                "shorter than rollout_max_context_len."
            )
        max_length = min(max_length, args.rollout_max_context_len)
    max_turns = TOOL_CONFIGS["max_turns"]
    max_tool_calls = TOOL_CONFIGS["max_tool_calls"]
    if max_turns <= 0:
        raise ValueError("The ReTool turn limit must be positive.")
    if max_tool_calls < 0:
        raise ValueError("The ReTool tool-call limit must be non-negative.")
    assistant_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(assistant_end_id, int) or tokenizer.decode(
        [assistant_end_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    ) != "<|im_end|>":
        raise ValueError("The ReTool tokenizer must provide the native <|im_end|> token.")
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    try:
        for turn in range(max_turns):
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                break
            remaining = max_length - len(prompt_ids) - len(response_ids)
            if remaining <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            params = dict(sampling_params)
            params["max_new_tokens"] = remaining
            # Text trimming must not change sampled IDs or their logprobs.
            params["no_stop_trim"] = True
            payload = {
                "input_ids": prompt_ids + response_ids,
                "sampling_params": params,
                "return_logprob": True,
            }
            try:
                output = await post(url, payload)
            except Exception as exc:
                _fail(sample, "Generation request failed: " + type(exc).__name__, aborted=True)
                break
            meta = output.get("meta_info") if isinstance(output, dict) else None
            finish_info = meta.get("finish_reason") if isinstance(meta, dict) else None
            if not isinstance(finish_info, dict):
                _fail(sample, "SGLang did not return a finish reason.", aborted=True)
                break
            finish = finish_info.get("type")
            if state.aborted or finish == "abort":
                sample.status = Sample.Status.ABORTED
                break
            token_logprobs = meta.get("output_token_logprobs")
            if token_logprobs is None:
                _fail(sample, "SGLang did not return output token log probabilities.", aborted=True)
                break
            if not token_logprobs:
                _fail(sample, "SGLang returned an empty assistant turn.", aborted=True)
                break
            try:
                turn_ids = [item[1] for item in token_logprobs]
                turn_logprobs = [item[0] for item in token_logprobs]
                if any(not isinstance(token, int) for token in turn_ids):
                    raise ValueError("Invalid token ID")
                if any(value is None or not math.isfinite(value) for value in turn_logprobs):
                    raise ValueError("Invalid token log probability")
            except (IndexError, TypeError, ValueError):
                _fail(sample, "SGLang returned invalid token/log-probability pairs.", aborted=True)
                break
            if len(turn_ids) > remaining:
                _fail(sample, "SGLang exceeded the requested token budget.", aborted=True)
                break
            response_ids.extend(turn_ids)
            sample.append_response_tokens(
                args, tokens=turn_ids, log_probs=turn_logprobs,
                trainable=True, meta_info=meta,
            )
            # Request status is not trajectory status.
            sample.status = Sample.Status.PENDING
            if finish == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if finish != "stop":
                _fail(sample, "Unsupported SGLang finish reason: " + str(finish))
                break
            turn_text = tokenizer.decode(turn_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            action = parse_assistant_action(
                turn_text, reasoning_prefilled=reasoning_prefilled, runtime_compat=True
            )
            if action.kind == "answer":
                sample.status = Sample.Status.COMPLETED
                break
            if len(prompt_ids) + len(response_ids) >= max_length:
                sample.status = Sample.Status.TRUNCATED
                break
            results = []
            call_budget_exhausted = False
            if action.kind != "tool":
                sample.metadata["retool_invalid_action_count"] += 1
                sample.metadata["retool_last_invalid_action"] = action.error
                results.append(INVALID_ACTION_FEEDBACK)
            for call in action.calls:
                if state.aborted:
                    sample.status = Sample.Status.ABORTED
                    break
                if tool_call_count >= max_tool_calls:
                    call_budget_exhausted = True
                    break
                async with SEMAPHORE:
                    # Abort may occur while awaiting a slot.
                    if state.aborted:
                        sample.status = Sample.Status.ABORTED
                        break
                    tool_call_count += 1
                    try:
                        result = await tool_registry.execute_tool(call.name, call.arguments)
                    except Exception as exc:
                        result = "Error: tool execution failed (" + type(exc).__name__ + ")."
                results.append(str(result))
            if sample.status is Sample.Status.ABORTED or state.aborted:
                sample.status = Sample.Status.ABORTED
                break
            if results:
                try:
                    suffix = render_tool_response_suffix(
                        tokenizer, results, chat_template_kwargs=template_kwargs
                    )
                    if turn_ids[-1] != assistant_end_id:
                        # Mask the inserted turn end; preserve sampled tokens and logprobs.
                        suffix = "<|im_end|>" + suffix
                        sample.metadata["retool_inserted_end_count"] += 1
                except ValueError as exc:
                    _fail(sample, str(exc))
                    break
                reasoning_prefilled = suffix.rstrip().endswith("<think>")
                observation_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                remaining = max_length - len(prompt_ids) - len(response_ids)
                overflow = len(observation_ids) > remaining
                observation_ids = observation_ids[:max(0, remaining)]
                response_ids.extend(observation_ids)
                sample.append_response_tokens(args, tokens=observation_ids, trainable=False)
                if overflow:
                    sample.status = Sample.Status.TRUNCATED
                    break
            if call_budget_exhausted:
                sample.status = Sample.Status.TRUNCATED
                break
        else:
            sample.status = Sample.Status.TRUNCATED
    except asyncio.CancelledError:
        sample.status = Sample.Status.ABORTED
        raise
    finally:
        sample.response = tokenizer.decode(response_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.tool_call_count = tool_call_count
        sample.metadata["tool_call_count"] = tool_call_count
        sample.payload_text = prompt + sample.response
        sample.payload_has_system = "<|im_start|>system" in prompt
        sample.payload_has_tools = "# Tools" in prompt
        sample._validate_response_metadata_lengths()
    return sample


async def reward_func(args, sample, **kwargs):
    """Score prompt plus response with math_verify; strict mode adjusts negative scores by tool usage."""
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    solution_str = sample.prompt + sample.response

    ground_truth = sample.label if sample.label is not None else ""

    num_turns = getattr(sample, "tool_call_count", 0)

    if os.environ.get("RETOOL_REWARD_MODE", "math_verify") == "strict":
        result = math_dapo_compute_score(solution_str, ground_truth, strict_box_verify=True)
        if result["score"] < 0:
            tool_call_reward = (num_turns - 2) / 2 * 0.1
            result["score"] = min(-0.6, result["score"] + tool_call_reward)
        if result["pred"] is None:
            result["pred"] = ""
        return result

    r_ans, pred = _math_verify_score(solution_str, ground_truth)
    reward = float(r_ans)

    return {"score": float(reward), "pred": pred or ""}
