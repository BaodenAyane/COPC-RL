# Modified for COPC.
"""Convert recorded ReTool traces to Qwen tool messages without executing tool code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from slime.utils.retool_protocol import (
    PROTOCOL_VERSION, RETOOL_SYSTEM_PROMPT, get_tool_specs, normalize_tool_observation,
)

_ANSWER_INSTRUCTION = "<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
_USER_PREFIX = (
    "Solve the following problem step by step. You now have the ability to selectively write executable "
    "Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, "
    "and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning "
    "and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \n"
    "Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\n"
    "The last part of your response should be in the following format:\n"
    + _ANSWER_INSTRUCTION + "\n\n*user question:*\n"
)
_USER_SUFFIX = (
    "\n\nRemember to place the final answer in the last part using the format: \n" + _ANSWER_INSTRUCTION
)
_TRACE_TAG = re.compile(r"</?(?:code|interpreter)>")
_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def extract_question(content: str) -> str:
    """Remove only the recognized wrapper, without rewriting the problem."""
    if not content.startswith(_USER_PREFIX) or not content.endswith(_USER_SUFFIX):
        raise ValueError("Unrecognized ReTool user prompt wrapper; the original question cannot be extracted safely.")
    question = content[len(_USER_PREFIX) : -len(_USER_SUFFIX)]
    if not question.strip() or "*user question:*" in question:
        raise ValueError("The source question is empty or has ambiguous boundaries.")
    return question


def _python_code(content: str) -> str:
    fenced = re.fullmatch(r"\s*```python[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t]*\s*", content)
    if fenced:
        code = fenced.group(1)
    elif "```" in content:
        raise ValueError("Malformed or unsupported Python code fence.")
    else:
        code = content.strip("\r\n")
    if not code.strip() or _TRACE_TAG.search(code):
        raise ValueError("Empty code or ambiguous ReTool delimiters inside code.")
    return code


def _final_content(content: str) -> str:
    """Unwrap a final answer while preserving all reasoning and answer text."""
    if "<answer>" not in content and "</answer>" not in content:
        return content
    if content.count("<answer>") != 1 or content.count("</answer>") != 1:
        raise ValueError("Ambiguous final answer wrappers.")
    match = re.fullmatch(r"([\s\S]*?)<answer>([\s\S]*?)</answer>(\s*)", content)
    if not match:
        raise ValueError("The answer wrapper is malformed or is not at the end of the trace.")
    return "".join(match.groups())


def convert(sample: dict, index: int | None = None) -> dict:
    """Convert a user/assistant trace, rejecting ambiguous input and preserving recorded observations."""
    if not isinstance(sample, dict):
        raise ValueError("Each source example must be a dictionary.")
    source = sample.get("messages")
    if not isinstance(source, list) or any(not isinstance(message, dict) for message in source):
        raise ValueError("Source messages must be a list of message dictionaries.")
    if [message.get("role") for message in source] != ["user", "assistant"]:
        raise ValueError("Expected exactly one source user message and one assistant trace.")
    if any(not isinstance(message.get("content"), str) for message in source):
        raise ValueError("Source message content must be text.")
    if any(token in message["content"] for message in source for token in _CONTROL_TOKENS):
        raise ValueError("Source data contains reserved Qwen conversation control tokens.")
    if any(message.get("tool_calls") or message.get("reasoning_content") for message in source):
        raise ValueError("The input is already structured or uses an unsupported reasoning field.")
    question = extract_question(source[0]["content"])
    trace = source[1]["content"]
    if "<think>" in trace or "</think>" in trace:
        raise ValueError("Explicit thinking delimiters need a separate, template-aware conversion.")
    if "<tool_call>" in trace or "<tool_response>" in trace:
        raise ValueError("Mixed native and legacy tool protocols are not supported.")
    step_loss_mask = source[1].get("step_loss_mask")
    if step_loss_mask is None:
        step_loss_mask = 1
    if step_loss_mask not in (0, 1):
        raise ValueError("step_loss_mask must be 0 or 1 when provided.")
    messages = [
        {"role": "system", "content": RETOOL_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    cursor = 0
    call_index = 0
    while True:
        opening = _TRACE_TAG.search(trace, cursor)
        if opening is None:
            break
        if opening.group() != "<code>":
            raise ValueError("An observation or closing delimiter has no preceding code call.")
        closing = _TRACE_TAG.search(trace, opening.end())
        if closing is None or closing.group() != "</code>":
            raise ValueError("Unclosed or nested code block.")
        code = _python_code(trace[opening.end() : closing.start()])
        observation = _TRACE_TAG.search(trace, closing.end())
        if observation is None or observation.group() != "<interpreter>":
            raise ValueError("Every code call must have exactly one recorded interpreter result.")
        if trace[closing.end() : observation.start()].strip():
            raise ValueError("Unexpected text between a code call and its recorded result.")
        observation_end = _TRACE_TAG.search(trace, observation.end())
        if observation_end is None or observation_end.group() != "</interpreter>":
            raise ValueError("Unclosed or nested interpreter result.")
        call_id = f"call_{call_index}"
        assistant = {
            "role": "assistant", "content": trace[cursor : opening.start()],
            "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "code_interpreter", "arguments": {"code": code}},
            }],
        }
        assistant["step_loss_mask"] = step_loss_mask
        messages.append(assistant)
        messages.append({
            "role": "tool", "name": "code_interpreter", "tool_call_id": call_id,
            "content": normalize_tool_observation(trace[observation.end() : observation_end.start()]),
        })
        cursor = observation_end.end()
        call_index += 1
    final = _final_content(trace[cursor:])
    if not final.strip():
        raise ValueError("The trace has no final assistant response after its last tool result.")
    assistant = {"role": "assistant", "content": final}
    assistant["step_loss_mask"] = step_loss_mask
    messages.append(assistant)
    tools = get_tool_specs()
    source_json = json.dumps(source, ensure_ascii=False, sort_keys=True)
    metadata = {
        "tools": copy.deepcopy(tools), "retool_protocol": PROTOCOL_VERSION,
        "source_sha256": hashlib.sha256(source_json.encode("utf-8")).hexdigest(),
        "tool_call_count": call_index, "observation_format": "plain-text-v1",
    }
    if index is not None:
        metadata["source_row"] = index
    return {"messages": messages, "tools": tools, "metadata": metadata}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local JSON/JSONL or Parquet source; otherwise load --dataset.")
    parser.add_argument("--dataset", default="JoeYing/ReTool-SFT")
    parser.add_argument("--revision", help="Dataset revision to pin when loading from Hugging Face.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/retool/ReTool-SFT.parquet"))
    args = parser.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.input is not None:
        if args.input.suffix == ".parquet":
            source = pq.read_table(str(args.input)).to_pylist()
        elif args.input.suffix == ".jsonl":
            with args.input.open(encoding="utf-8") as handle:
                source = [json.loads(line) for line in handle if line.strip()]
        elif args.input.suffix == ".json":
            source = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(source, list):
                raise ValueError("A JSON source must contain an array of examples.")
        else:
            raise ValueError("Local input must be a .json, .jsonl or .parquet file.")
        source_name = args.input.name
    else:
        from datasets import load_dataset
        source = load_dataset(args.dataset, revision=args.revision, split=args.split)
        source_name = args.dataset
    converted = []
    for index, row in enumerate(source):
        try:
            result = convert(row, index)
        except ValueError as error:
            raise ValueError(f"Refusing to publish a partial conversion: source row {index}: {error}") from error
        result["metadata"]["source_dataset"] = source_name
        result["metadata"]["source_revision"] = args.revision or "unspecified"
        converted.append(result)
    if not converted:
        raise ValueError("The source dataset is empty.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), str(args.output))
    print(f"Converted {len(converted)} examples to {args.output} using {PROTOCOL_VERSION}.")


if __name__ == "__main__":
    main()
