"""Align all ReTool records with the RL template and masks without executing recorded code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.retool_protocol import (
    PROTOCOL_VERSION, RETOOL_SYSTEM_PROMPT, canonicalize_tools, get_tool_specs,
    normalize_tool_observation, render_initial_prompt, render_tool_response_suffix,
)

PREPARATION_VERSION = "retool-sft-alignment-v1"
ANSWER_SUFFIX = "\n\nRemember to place the final answer in the last part using the format: \n\n\\boxed{{'The final answer goes here.'}}"
_THINK_TAG = re.compile(r"</?think>")
_ASSISTANT_FRAME = re.compile(r"<\|im_start\|>assistant\n([\s\S]*?)<\|im_end\|>\n")



def load_local_tokenizer(model_dir):
    """Load the checkpoint template without training or multimodal dependencies."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    template_file = Path(model_dir) / "chat_template.jinja"
    if not tokenizer.chat_template and template_file.is_file():
        tokenizer.chat_template = template_file.read_text(encoding="utf-8")
    return tokenizer


def json_hash(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_record(row, index):
    """Preserve structured traces while aligning wrappers and assistant padding."""
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        raise ValueError("Each record must contain a messages array.")
    if canonicalize_tools(row.get("tools") or []) != get_tool_specs():
        raise ValueError("The source tools must use the shared code_interpreter(code: string) schema.")
    source = copy.deepcopy(row["messages"])
    if source and source[0].get("role") == "system":
        source.pop(0)
    if not source or source[0].get("role") != "user":
        raise ValueError("A conversation must begin with exactly one user problem.")
    question = source[0].get("content")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The original user problem must be nonempty text.")
    changes = []
    if question.endswith(ANSWER_SUFFIX):
        question = question[:-len(ANSWER_SUFFIX)]
        changes.append("removed_exact_legacy_answer_instruction")
    if not question.strip():
        raise ValueError("The problem is empty after removing the known wrapper.")
    messages = [{"role": "system", "content": RETOOL_SYSTEM_PROMPT}, {"role": "user", "content": question}]
    pending_calls, call_count = [], 0
    for source_index, original in enumerate(source[1:], start=1):
        role, content = original.get("role"), original.get("content")
        if not isinstance(content, str):
            raise ValueError("Every assistant/tool message must contain text.")
        if role == "assistant":
            if pending_calls:
                raise ValueError("An assistant turn precedes a pending recorded tool result.")
            # Strip legacy markers, preserving reasoning text.
            normalized = _THINK_TAG.sub("", content).strip()
            if _THINK_TAG.search(content):
                changes.append("removed_legacy_thinking_tags")
            if normalized != _THINK_TAG.sub("", content):
                changes.append("trimmed_assistant_boundary_whitespace")
            message = {"role": "assistant", "content": normalized, "step_loss_mask": 1}
            if original.get("reasoning_content"):
                raise ValueError("Separate reasoning_content requires an explicit non-thinking conversion.")
            calls = original.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be a list when provided.")
            converted_calls = []
            for call in calls:
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    raise ValueError("Only function calls are supported.")
                function = call.get("function")
                if not isinstance(function, dict) or function.get("name") != "code_interpreter":
                    raise ValueError("Every call must target code_interpreter.")
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict) or set(arguments) != {"code"} or not isinstance(arguments["code"], str):
                    raise ValueError("A code_interpreter call must contain exactly one string argument: code.")
                call_id = f"call_{call_count}"
                converted_calls.append({"id": call_id, "type": "function", "function": {"name": "code_interpreter", "arguments": copy.deepcopy(arguments)}})
                pending_calls.append(call_id)
                call_count += 1
            if converted_calls:
                message["tool_calls"] = converted_calls
            elif not normalized:
                raise ValueError("An assistant turn has neither text nor a tool call.")
            messages.append(message)
        elif role == "tool":
            if not pending_calls:
                raise ValueError("A recorded tool result has no matching call.")
            observation = normalize_tool_observation(content)
            if observation != content:
                changes.append("trimmed_observation_boundary_whitespace")
            messages.append({"role": "tool", "name": "code_interpreter", "tool_call_id": pending_calls.pop(0), "content": observation})
        else:
            raise ValueError(f"Unsupported source role at message {source_index}: {role!r}")
    if pending_calls or messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls"):
        raise ValueError("A complete trajectory must end in a final assistant response.")
    tools = get_tool_specs()
    metadata = {
        "tools": copy.deepcopy(tools), "retool_protocol": PROTOCOL_VERSION,
        "preparation_version": PREPARATION_VERSION, "source_row": index,
        "source_record_sha256": json_hash(row), "source_sha256": json_hash(row["messages"]),
        "tool_call_count": call_count, "observation_format": "plain-text-v1",
        "preparation_changes": changes,
    }
    return {"messages": messages, "tools": tools, "metadata": metadata}


def validate_replay(converted, generator, max_length):
    """Replay RL prompt/environment boundaries and compare every token and mask."""
    tokenizer = generator.tokenizer
    messages, tools = converted["messages"], converted["tools"]
    ids, mask = generator.get_loss_mask(messages, tools, structured_tools=True)
    rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False)
    initial = render_initial_prompt(tokenizer, messages[1]["content"])
    replay_text = initial
    replay_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    replay_mask = [0] * len(replay_ids)
    assistant_frames = list(_ASSISTANT_FRAME.finditer(rendered))
    assistant_index, cursor = 0, 2
    while cursor < len(messages):
        message = messages[cursor]
        if message["role"] != "assistant" or assistant_index >= len(assistant_frames):
            raise ValueError("The template and structured assistant turns disagree.")
        frame = assistant_frames[assistant_index]
        if replay_text != rendered[:frame.start(1)] or replay_ids != ids[:len(replay_ids)]:
            raise ValueError(f"SFT and RL prompt tokens differ before assistant turn {assistant_index}.")
        generated = frame.group(1) + "<|im_end|>"
        generated_ids = tokenizer(generated, add_special_tokens=False)["input_ids"]
        replay_text += generated
        replay_ids.extend(generated_ids)
        replay_mask.extend([1] * len(generated_ids))
        assistant_index += 1
        cursor += 1
        observations = []
        while cursor < len(messages) and messages[cursor]["role"] == "tool":
            # Add the display heading that the runtime renderer strips.
            observations.append("Output:\n" + messages[cursor]["content"])
            cursor += 1
        if observations:
            suffix = render_tool_response_suffix(tokenizer, observations)
        elif cursor == len(messages):
            suffix = "\n"
        else:
            raise ValueError("Consecutive assistant messages have no matching RL environment transition.")
        suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        replay_text += suffix
        replay_ids.extend(suffix_ids)
        replay_mask.extend([0] * len(suffix_ids))
    if assistant_index != len(assistant_frames) or replay_text != rendered or replay_ids != ids or replay_mask != mask:
        raise ValueError("Full SFT tokens/masks differ from the RL prompt, assistant, and tool replay.")
    if len(ids) > max_length:
        raise ValueError(f"Full conversation has {len(ids)} tokens; limit is {max_length}. No truncation was applied.")
    if not any(mask):
        raise ValueError("The conversation has no supervised assistant tokens.")
    return len(ids), sum(mask), assistant_index


def prepare_records(rows, tokenizer, max_length):
    generator = MultiTurnLossMaskGenerator(tokenizer, "qwen")
    prepared, decisions = [], []
    for index, row in enumerate(rows):
        decision = {"source_row": index, "source_record_sha256": json_hash(row)}
        try:
            converted = normalize_record(row, index)
            tokens, supervised, turns = validate_replay(converted, generator, max_length)
        except (ValueError, TypeError, KeyError) as error:
            decision.update(status="failed_alignment", detail=str(error))
        else:
            converted["metadata"].update(token_count=tokens, supervised_token_count=supervised)
            prepared.append(converted)
            decision.update(status="retained", token_count=tokens, supervised_token_count=supervised,
                            assistant_turns=turns, tool_call_count=converted["metadata"]["tool_call_count"],
                            normalizations=converted["metadata"]["preparation_changes"])
        decisions.append(decision)
    return prepared, decisions


def _publish_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args(argv)
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if len({path.resolve() for path in (args.input, args.output, args.report)}) != 3:
        parser.error("Input, output, and report must use different paths")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    import pyarrow as pa
    import pyarrow.parquet as pq
    tokenizer = load_local_tokenizer(args.model_dir)
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise ValueError("The local model must provide its own string chat template.")
    rows = pq.read_table(args.input).to_pylist()
    prepared, decisions = prepare_records(rows, tokenizer, args.max_length)
    source_sha256 = file_hash(args.input)
    for row in prepared:
        row["metadata"]["source_file_sha256"] = source_sha256
    failed = len(rows) - len(prepared)
    report = {
        "preparation_version": PREPARATION_VERSION, "protocol": PROTOCOL_VERSION,
        "source_file": args.input.name, "source_sha256": source_sha256,
        "tokenizer_dir": str(args.model_dir.resolve()),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "max_length": args.max_length, "total_rows": len(rows), "accepted_rows": len(prepared),
        "rejected_rows": failed, "filtered_rows": 0,
        "verified_assistant_turns": sum(row.get("assistant_turns", 0) for row in decisions),
        "max_sequence_length": max((row["token_count"] for row in decisions if row["status"] == "retained"), default=0),
        "normalization_counts": dict(Counter(change for row in decisions for change in row.get("normalizations", []))),
        "data_written": bool(rows and not failed),
        "policy": {
            "retention": "All source records are retained; no data quality filtering is applied.",
            "question": "Original problem text; only the exact known answer-instruction suffix is removed.",
            "supervision": "Assistant reasoning, native tool-call JSON, final answers, and EOS. System/user/tool tokens are masked.",
            "replay": "Full rendered token IDs and loss masks must match the shared RL prompt/tool-response replay.",
            "code": "Code strings and recorded observations are not executed, repaired, or re-scored.",
            "limitations": "Template alignment does not validate code execution or recorded tool outputs.",
        },
        "alignment_failures": [row for row in decisions if row["status"] != "retained"],
    }
    if not rows or failed:
        _publish_report(report, args.report)
        raise ValueError(f"Refusing partial publication: {failed} alignment failures across {len(rows)} source rows. See {args.report}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    published = False
    try:
        pq.write_table(pa.Table.from_pylist(prepared), temporary)
        # Atomic creation without overwriting a concurrent result.
        os.link(str(temporary), str(args.output))
        published = True
        _publish_report(report, args.report)
    except BaseException:
        if published:
            args.output.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({key: report[key] for key in ("total_rows", "accepted_rows", "filtered_rows", "max_sequence_length")}, sort_keys=True))


if __name__ == "__main__":
    main()
