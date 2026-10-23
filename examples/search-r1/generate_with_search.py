# Modified for COPC.
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/ceee7b89655ed52f205b9beb98e1190c3eedcfb0/search_r1/llm_agent/generation.py
# Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
"""Generate Search-SFT trajectories without reconstructing sampled tokens from text."""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache

from retrieval_client import RetrievalError, SearchClient
from search_protocol import (
    parse_tool_call, protocol_turn_end, render_search_results,
    render_tool_response, truncate_assistant_turn,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchSettings:
    url: str
    max_turns: int = 8
    topk: int = 3
    max_doc_length: int = 200
    timeout: float = 65.0
    max_retries: int = 3
    concurrency: int = 120
    protocol: str = "qwen"
    retrieval_failure_mode: str = "abort"

    def __post_init__(self):
        from urllib.parse import urlsplit

        protocol_turn_end(self.protocol)
        if self.retrieval_failure_mode not in ("abort", "observation"):
            raise ValueError("SEARCH_RETRIEVAL_FAILURE_MODE must be abort or observation")
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("RETRIEVAL_SERVICE_URL must be a concrete HTTP(S) /retrieve URL without credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("RETRIEVAL_SERVICE_URL must not contain a query string or fragment")
        for name in ("max_turns", "topk", "max_doc_length", "max_retries", "concurrency"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Search {name} must be positive")
        if self.topk > 10 or self.max_retries > 10:
            raise ValueError("Search topk and max_retries must be <= 10")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Search timeout must be finite and positive")

    @classmethod
    def from_env(cls):
        protocol = os.environ.get("SEARCH_PROTOCOL", "llama3")
        return cls(
            url=os.environ.get("RETRIEVAL_SERVICE_URL", ""),
            max_turns=int(os.environ.get("SEARCH_MAX_TURNS", "8")),
            topk=int(os.environ.get("SEARCH_TOPK", "3")),
            max_doc_length=int(os.environ.get("SEARCH_MAX_DOC_LENGTH", "200")),
            timeout=float(os.environ.get("SEARCH_TOOL_TIMEOUT", "65")),
            max_retries=int(os.environ.get("SEARCH_TOOL_MAX_RETRIES", "3")),
            concurrency=int(os.environ.get("SEARCH_TOOL_RATE_LIMIT", "120")),
            protocol=protocol,
            retrieval_failure_mode=os.environ.get(
                "SEARCH_RETRIEVAL_FAILURE_MODE", "observation" if protocol in ("llama", "llama3") else "abort",
            ),
        )


@lru_cache(maxsize=8)
def _get_client(settings):
    return SearchClient(
        settings.url, topk=settings.topk, timeout=settings.timeout,
        max_retries=settings.max_retries, concurrency=settings.concurrency,
    )


def _decode(tokenizer, tokens):
    return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _prefix_token_count(tokenizer, tokens, text, end):
    """Keep the original token prefix covering a text boundary, including any overlapping final token."""
    if end <= 0:
        return 0
    if end >= len(text):
        return len(tokens)
    prefix = text[:end]
    low, high = 1, len(tokens)
    while low < high:
        middle = (low + high) // 2
        if _decode(tokenizer, tokens[:middle]).startswith(prefix):
            high = middle
        else:
            low = middle + 1
    if not _decode(tokenizer, tokens[:low]).startswith(prefix):
        raise ValueError("Tokenizer cannot map assistant text boundary to original token IDs")
    return low


def _slice_meta(meta, count, original_count):
    """Keep optional top-p replay metadata aligned when trimming a token suffix."""
    if count == original_count:
        return meta
    meta = dict(meta)
    # MoE routing metadata cannot be sliced as response-only tokens.
    if "routed_experts" in meta:
        raise ValueError("Search-R1 does not support trimming MoE routing replay")
    id_keys = ("top_p_token_ids", "top_p_kept_token_ids")
    offset_keys = ("top_p_token_offsets", "top_p_kept_token_offsets")
    if any(key in meta for key in id_keys + offset_keys):
        from slime.utils.misc import decode_int32_meta_array

        ids = decode_int32_meta_array(meta, id_keys)
        offsets = decode_int32_meta_array(meta, offset_keys)
        if ids is None or offsets is None or len(offsets) != original_count + 1:
            raise ValueError("Incomplete top-p replay metadata on trimmed Search trajectory")
        for key in id_keys + offset_keys:
            meta.pop(key, None)
        meta["top_p_token_ids"] = ids[:int(offsets[count])].tolist()
        meta["top_p_token_offsets"] = offsets[:count + 1].tolist()
    return meta


def _reset_sample(sample, prompt_ids):
    sample.tokens = list(prompt_ids)
    sample.response = ""
    sample.response_length = 0
    sample.loss_mask = []
    sample.rollout_log_probs = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    sample.weight_versions = []
    sample.reward = None
    sample.remove_sample = False
    sample.status = sample.Status.PENDING
    sample.non_generation_time = 0.0
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["search_r1"] = {"num_tool_calls": 0, "assistant_turns": 0}
    sample.metadata["tool_call_count"] = 0
    # Exclude discarded attempts from replayed metrics.
    if hasattr(sample, "spec_info"):
        sample.spec_info = type(sample.spec_info)()
    if hasattr(sample, "prefix_cache_info"):
        sample.prefix_cache_info = type(sample.prefix_cache_info)()


def _abort(sample, reason):
    # Retry from scratch after abort; checkpoint parking bypasses this reset.
    sample.metadata.pop("search_async_checkpoint_state", None)
    sample.status = sample.Status.ABORTED
    sample.metadata["search_r1"]["abort_reason"] = reason
    return sample


ASYNC_CURSOR_KEY = "search_async_checkpoint_state"


def _checkpoint_settings_signature(settings):
    # Keep the external endpoint out of saved cursors and exception messages.
    import hashlib
    import json
    from dataclasses import asdict

    return hashlib.sha256(json.dumps(asdict(settings), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


async def _generate_checkpointed_trajectory(args, sample, sampling_params, *, tokenizer, post_fn, client, settings, is_aborted):
    """Persist each committed assistant/tool result before another external action."""
    from slime.rollout.checkpoint_barrier import checkpoint_safe_point
    from slime.rollout.checkpoint_protocol import request_seed, validate_checkpoint_sample, validate_weight_version

    await checkpoint_safe_point(sample)
    validate_checkpoint_sample(args, sample)
    if getattr(args, "partial_rollout", False):
        raise ValueError("Search-R1 retries full trajectories; --partial-rollout is unsupported")
    if getattr(args, "use_rollout_routing_replay", False):
        raise ValueError("Search-R1 currently supports dense models without MoE routing replay")
    cursor = (sample.metadata or {}).get(ASYNC_CURSOR_KEY)
    if cursor is None:
        turn_end = protocol_turn_end(settings.protocol)
        prompt = sample.prompt
        if not isinstance(prompt, str):
            template_kwargs = dict(getattr(sample, "apply_chat_template_kwargs", {}) or {})
            if settings.protocol in ("llama", "llama3"):
                tools = (sample.metadata or {}).get("tools")
                if not isinstance(tools, list) or len(tools) != 1:
                    raise ValueError("Llama Search message prompts require one search schema in sample.metadata.tools")
                template_kwargs["tools"] = tools
            else:
                # Legacy Qwen datasets already contain a rendered tool system.
                template_kwargs.pop("tools", None)
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, **template_kwargs,
            )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        _reset_sample(sample, prompt_ids)
        sample.prompt = prompt
        stats = sample.metadata["search_r1"]
        stats.update(protocol=settings.protocol, reward_response="", retrieval_failures=0,
                     retrieval_failure_mode=settings.retrieval_failure_mode)
        response_budget = int(sampling_params["max_new_tokens"])
        if response_budget < 0:
            raise ValueError("max_new_tokens must be non-negative")
        max_length = len(prompt_ids) + response_budget
        context_cap = getattr(args, "rollout_max_context_len", None)
        if context_cap is not None:
            max_length = min(max_length, int(context_cap))
        params = copy.deepcopy(sampling_params)
        stops = params.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        params.update(stop=list(dict.fromkeys([*stops, turn_end])), no_stop_trim=True, skip_special_tokens=False)
        cursor = {"version": 1, "stage": "generate", "turn": 0,
                  "settings_sha256": _checkpoint_settings_signature(settings),
                  "prompt_length": len(prompt_ids), "max_length": max_length,
                  "params": params, "turn_end": turn_end, "sampling_seeds": {}}
        sample.metadata[ASYNC_CURSOR_KEY] = cursor
    else:
        if cursor.get("version") != 1 or cursor.get("stage") not in {"generate", "retrieve", "done"}:
            raise ValueError("Unsupported Search async checkpoint cursor")
        if cursor["settings_sha256"] != _checkpoint_settings_signature(settings):
            raise ValueError("Search tool configuration changed since checkpoint")
        if len(sample.tokens) - cursor["prompt_length"] != sample.response_length:
            raise ValueError("Search checkpoint cursor and response length disagree")
        sample._validate_response_metadata_lengths()
        if cursor["stage"] == "done":
            return sample
        sample.status = sample.Status.PENDING
        params, max_length, turn_end = cursor["params"], cursor["max_length"], cursor["turn_end"]
    stats = sample.metadata["search_r1"]
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    headers = None
    if getattr(args, "router_policy", None) == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    for turn in range(cursor["turn"], settings.max_turns):
        if is_aborted():
            return _abort(sample, "weight_update")
        remaining = max_length - len(sample.tokens)
        if remaining <= 0 and cursor["stage"] == "generate":
            sample.status = sample.Status.TRUNCATED
            break
        if cursor["stage"] == "generate":
            turn_params = copy.deepcopy(params)
            turn_params["max_new_tokens"] = remaining
            seed_key = str(turn)
            if seed_key not in cursor["sampling_seeds"]:
                cursor["sampling_seeds"][seed_key] = request_seed(args, sample, turn, namespace="search-request-v1")
            turn_params["sampling_seed"] = cursor["sampling_seeds"][seed_key]
            await checkpoint_safe_point(sample)
            payload = {"input_ids": list(sample.tokens), "sampling_params": turn_params, "return_logprob": True}
            try:
                output = await post_fn(url, payload, headers=headers)
            except Exception as exc:
                # Propagate cancellation; fully-async collection retries aborted groups.
                logger.warning("SGLang request failed for sample %s: %s", sample.index, exc)
                return _abort(sample, "generation_failure")
            meta = output.get("meta_info", {})
            finish = meta.get("finish_reason", {}).get("type")
            if finish == "abort" or is_aborted():
                return _abort(sample, "weight_update")
            pairs = meta.get("output_token_logprobs")
            if not pairs or finish not in ("stop", "length"):
                return _abort(sample, "missing_generation_metadata")
            validate_weight_version(meta)
            tokens = [int(item[1]) for item in pairs]
            log_probs = [float(item[0]) for item in pairs]
            if not all(math.isfinite(p) for p in log_probs):
                return _abort(sample, "nonfinite_behavior_logprob")
            text = _decode(tokenizer, tokens)
            terminal_prefix, boundary = truncate_assistant_turn(text, protocol=settings.protocol)
            terminal_end = len(terminal_prefix) if boundary in ("boxed", "turn_end") else None
            call, invalid_call = None, None
            if terminal_end is None:
                try:
                    call = parse_tool_call(text, protocol=settings.protocol)
                except ValueError as exc:
                    invalid_call = str(exc)
            # Slice sampled IDs and logprobs together at the protocol boundary.
            end = terminal_end or len(text)
            if invalid_call is not None:
                # Exclude boxes inside malformed tool arguments.
                end = len(terminal_prefix) if boundary == "invalid_tool_call" else 0
            if call is not None:
                if boundary != "tool_call":
                    raise ValueError("Parsed Search call has no matching assistant token boundary")
                end = len(terminal_prefix)
                if text[end:].strip() == turn_end:
                    end = len(text)
            intended_count = _prefix_token_count(tokenizer, tokens, text, end)
            count = min(intended_count, remaining)
            clipped_by_budget = count < intended_count
            kept_text = _decode(tokenizer, tokens[:count])
            turn_meta = _slice_meta(meta, count, len(tokens))
            sample.append_response_tokens(
                args, tokens=tokens[:count], log_probs=log_probs[:count], trainable=True,
                meta_info=turn_meta, text=kept_text,
            )
            stats["assistant_turns"] = turn + 1
            # Score only assistant answers outside tool arguments and evidence.
            stats["reward_response"] = (
                kept_text[:end]
                if call is None and (
                    settings.protocol == "qwen" or (boundary == "boxed" and not clipped_by_budget)
                ) else ""
            )
            if clipped_by_budget:
                sample.status = sample.Status.TRUNCATED
                break
            if terminal_end is not None:
                sample.status = sample.Status.COMPLETED
                break
            if invalid_call is not None:
                stats["invalid_tool_call"] = invalid_call
                if sample.response_length == 0:
                    # A masked terminator keeps an empty turn valid for packing.
                    closing = tokenizer.encode(turn_end, add_special_tokens=False)[:remaining]
                    sample.append_response_tokens(args, tokens=closing, trainable=False,
                                                  text=_decode(tokenizer, closing))
                    sample.remove_sample = True
                sample.status = sample.Status.COMPLETED
                break
            if call is None:
                sample.status = sample.Status.TRUNCATED if finish == "length" else sample.Status.COMPLETED
                break
            cursor.update(stage="retrieve", turn=turn, query_list=list(call["arguments"]["query_list"]),
                          kept_text=kept_text)
        # Restoring this cursor must not repeat the completed generation.
        sample.status = sample.Status.PENDING
        sample._validate_response_metadata_lengths()
        await checkpoint_safe_point(sample)
        if is_aborted():
            return _abort(sample, "weight_update")
        start = time.monotonic()
        retrieval_failed = False
        try:
            documents = await client.retrieve(cursor["query_list"])
        except RetrievalError as exc:
            logger.warning("Search retrieval failed for sample %s: %s", sample.index, exc)
            stats["retrieval_failures"] += 1
            if settings.retrieval_failure_mode == "abort":
                return _abort(sample, "retrieval_failure")
            # Failed retrieval remains an observation and counts as a tool call.
            documents = "Search error: " + str(exc)
            retrieval_failed = True
        finally:
            sample.non_generation_time += time.monotonic() - start
        if is_aborted():
            return _abort(sample, "weight_update")
        observation, metrics = render_search_results(
            documents, tokenizer, topk=settings.topk, max_doc_length=settings.max_doc_length,
        )
        stats["num_tool_calls"] += 1
        stats["last_retrieval"] = dict(metrics, status="error" if retrieval_failed else "success")
        sample.metadata["tool_call_count"] = stats["num_tool_calls"]
        # Permit a tool call on the final allowed assistant turn.
        suffix = "" if cursor["kept_text"].rstrip().endswith(turn_end) else turn_end
        suffix += render_tool_response(observation, protocol=settings.protocol)
        observation_ids = tokenizer.encode(suffix, add_special_tokens=False)
        available = max(0, max_length - len(sample.tokens))
        sample.append_response_tokens(
            args, tokens=observation_ids[:available], trainable=False,
            text=_decode(tokenizer, observation_ids[:available]),
        )
        if len(observation_ids) >= available:
            sample.status = sample.Status.TRUNCATED
            break
        # Commit retrieval before awaiting so restore cannot repeat it.
        sample.status = sample.Status.PENDING
        cursor.update(stage="generate", turn=turn + 1)
        cursor.pop("query_list", None)
        cursor.pop("kept_text", None)
    else:
        sample.status = sample.Status.TRUNCATED
        stats["turn_limit_reached"] = True
    cursor["stage"] = "done"
    sample._validate_response_metadata_lengths()
    return sample


async def generate_trajectory(args, sample, sampling_params, *, tokenizer, post_fn, client, settings, is_aborted):
    if getattr(args, "async_checkpoint", False):
        return await _generate_checkpointed_trajectory(
            args, sample, sampling_params, tokenizer=tokenizer, post_fn=post_fn,
            client=client, settings=settings, is_aborted=is_aborted,
        )
    if getattr(args, "partial_rollout", False):
        raise ValueError("Search-R1 retries full trajectories; --partial-rollout is unsupported")
    if getattr(args, "use_rollout_routing_replay", False):
        raise ValueError("Search-R1 currently supports dense models without MoE routing replay")
    turn_end = protocol_turn_end(settings.protocol)
    prompt = sample.prompt
    if not isinstance(prompt, str):
        template_kwargs = dict(getattr(sample, "apply_chat_template_kwargs", {}) or {})
        if settings.protocol in ("llama", "llama3"):
            tools = (sample.metadata or {}).get("tools")
            if not isinstance(tools, list) or len(tools) != 1:
                raise ValueError("Llama Search message prompts require one search schema in sample.metadata.tools")
            template_kwargs["tools"] = tools
        else:
            # Legacy Qwen datasets already contain a rendered tool system.
            template_kwargs.pop("tools", None)
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True, **template_kwargs,
        )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    _reset_sample(sample, prompt_ids)
    sample.prompt = prompt
    stats = sample.metadata["search_r1"]
    stats.update(protocol=settings.protocol, reward_response="", retrieval_failures=0,
                 retrieval_failure_mode=settings.retrieval_failure_mode)
    response_budget = int(sampling_params["max_new_tokens"])
    if response_budget < 0:
        raise ValueError("max_new_tokens must be non-negative")
    max_length = len(prompt_ids) + response_budget
    context_cap = getattr(args, "rollout_max_context_len", None)
    if context_cap is not None:
        max_length = min(max_length, int(context_cap))
    params = copy.deepcopy(sampling_params)
    stops = params.get("stop") or []
    if isinstance(stops, str):
        stops = [stops]
    params.update(stop=list(dict.fromkeys([*stops, turn_end])), no_stop_trim=True, skip_special_tokens=False)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    headers = None
    if getattr(args, "router_policy", None) == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    for turn in range(settings.max_turns):
        if is_aborted():
            return _abort(sample, "weight_update")
        remaining = max_length - len(sample.tokens)
        if remaining <= 0:
            sample.status = sample.Status.TRUNCATED
            break
        turn_params = copy.deepcopy(params)
        turn_params["max_new_tokens"] = remaining
        payload = {"input_ids": list(sample.tokens), "sampling_params": turn_params, "return_logprob": True}
        try:
            output = await post_fn(url, payload, headers=headers)
        except Exception as exc:
            # Propagate cancellation; fully-async collection retries aborted groups.
            logger.warning("SGLang request failed for sample %s: %s", sample.index, exc)
            return _abort(sample, "generation_failure")
        meta = output.get("meta_info", {})
        finish = meta.get("finish_reason", {}).get("type")
        if finish == "abort" or is_aborted():
            return _abort(sample, "weight_update")
        pairs = meta.get("output_token_logprobs")
        if not pairs or finish not in ("stop", "length"):
            return _abort(sample, "missing_generation_metadata")
        tokens = [int(item[1]) for item in pairs]
        log_probs = [float(item[0]) for item in pairs]
        if not all(math.isfinite(p) for p in log_probs):
            return _abort(sample, "nonfinite_behavior_logprob")
        text = _decode(tokenizer, tokens)
        terminal_prefix, boundary = truncate_assistant_turn(text, protocol=settings.protocol)
        terminal_end = len(terminal_prefix) if boundary in ("boxed", "turn_end") else None
        call, invalid_call = None, None
        if terminal_end is None:
            try:
                call = parse_tool_call(text, protocol=settings.protocol)
            except ValueError as exc:
                invalid_call = str(exc)
        # Slice sampled IDs and logprobs together at the protocol boundary.
        end = terminal_end or len(text)
        if invalid_call is not None:
            # Exclude boxes inside malformed tool arguments.
            end = len(terminal_prefix) if boundary == "invalid_tool_call" else 0
        if call is not None:
            if boundary != "tool_call":
                raise ValueError("Parsed Search call has no matching assistant token boundary")
            end = len(terminal_prefix)
            if text[end:].strip() == turn_end:
                end = len(text)
        intended_count = _prefix_token_count(tokenizer, tokens, text, end)
        count = min(intended_count, remaining)
        clipped_by_budget = count < intended_count
        kept_text = _decode(tokenizer, tokens[:count])
        turn_meta = _slice_meta(meta, count, len(tokens))
        sample.append_response_tokens(
            args, tokens=tokens[:count], log_probs=log_probs[:count], trainable=True,
            meta_info=turn_meta, text=kept_text,
        )
        stats["assistant_turns"] = turn + 1
        # Score only assistant answers outside tool arguments and evidence.
        stats["reward_response"] = (
            kept_text[:end]
            if call is None and (
                settings.protocol == "qwen" or (boundary == "boxed" and not clipped_by_budget)
            ) else ""
        )
        if clipped_by_budget:
            sample.status = sample.Status.TRUNCATED
            break
        if terminal_end is not None:
            sample.status = sample.Status.COMPLETED
            break
        if invalid_call is not None:
            stats["invalid_tool_call"] = invalid_call
            if sample.response_length == 0:
                # A masked terminator keeps an empty turn valid for packing.
                closing = tokenizer.encode(turn_end, add_special_tokens=False)[:remaining]
                sample.append_response_tokens(args, tokens=closing, trainable=False,
                                              text=_decode(tokenizer, closing))
                sample.remove_sample = True
            sample.status = sample.Status.COMPLETED
            break
        if call is None:
            sample.status = sample.Status.TRUNCATED if finish == "length" else sample.Status.COMPLETED
            break
        if is_aborted():
            return _abort(sample, "weight_update")
        start = time.monotonic()
        retrieval_failed = False
        try:
            documents = await client.retrieve(call["arguments"]["query_list"])
        except RetrievalError as exc:
            logger.warning("Search retrieval failed for sample %s: %s", sample.index, exc)
            stats["retrieval_failures"] += 1
            if settings.retrieval_failure_mode == "abort":
                return _abort(sample, "retrieval_failure")
            # Failed retrieval remains an observation and counts as a tool call.
            documents = "Search error: " + str(exc)
            retrieval_failed = True
        finally:
            sample.non_generation_time += time.monotonic() - start
        if is_aborted():
            return _abort(sample, "weight_update")
        observation, metrics = render_search_results(
            documents, tokenizer, topk=settings.topk, max_doc_length=settings.max_doc_length,
        )
        stats["num_tool_calls"] += 1
        stats["last_retrieval"] = dict(metrics, status="error" if retrieval_failed else "success")
        sample.metadata["tool_call_count"] = stats["num_tool_calls"]
        # Permit a tool call on the final allowed assistant turn.
        suffix = "" if kept_text.rstrip().endswith(turn_end) else turn_end
        suffix += render_tool_response(observation, protocol=settings.protocol)
        observation_ids = tokenizer.encode(suffix, add_special_tokens=False)
        available = max(0, max_length - len(sample.tokens))
        sample.append_response_tokens(
            args, tokens=observation_ids[:available], trainable=False,
            text=_decode(tokenizer, observation_ids[:available]),
        )
        if len(observation_ids) >= available:
            sample.status = sample.Status.TRUNCATED
            break
        sample.status = sample.Status.PENDING
    else:
        sample.status = sample.Status.TRUNCATED
        stats["turn_limit_reached"] = True
    sample._validate_response_metadata_lengths()
    return sample


async def generate(args, sample, sampling_params, evaluation=False):
    """slime --custom-generate-function-path entry point (train and evaluation)."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post

    state = GenerateState(args)
    settings = SearchSettings.from_env()
    result = await generate_trajectory(
        args, sample, sampling_params, tokenizer=state.tokenizer,
        post_fn=post, client=_get_client(settings), settings=settings,
        is_aborted=lambda: state.aborted,
    )
    # Only fully-async collection can retry an aborted group.
    fully_async = getattr(args, "rollout_function_path", "") == (
        "slime.rollout.fully_async_rollout.generate_rollout_fully_async"
    )
    if result.status == result.Status.ABORTED:
        reason = result.metadata["search_r1"].get("abort_reason", "unknown")
        # Synchronous collection aborts surplus samples after reaching its target.
        planned_stop = state.aborted and reason == "weight_update"
        if evaluation or (not fully_async and not planned_stop):
            phase = "evaluation" if evaluation else "synchronous rollout"
            raise RuntimeError(f"Search-R1 trajectory aborted during {phase}: {reason}")
    return result
