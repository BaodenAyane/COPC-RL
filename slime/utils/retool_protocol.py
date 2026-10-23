"""Shared Qwen ReTool protocol; never re-render sampled tokens through the chat template."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "qwen-tools-v1"
RETOOL_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
RETOOL_TOOLS = [{
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "A tool for executing code.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The code to execute."}},
            "required": ["code"],
        },
    },
}]


def get_tool_specs() -> list[dict[str, Any]]:
    """Return an independent copy of the shared schema."""
    return canonicalize_tools(RETOOL_TOOLS)


def canonicalize_tools(tools):
    """Keep schema serialization stable across JSON and Parquet round trips."""
    return json.loads(json.dumps(tools, sort_keys=True, ensure_ascii=False))


def normalize_messages(prompt, *, system_prompt=RETOOL_SYSTEM_PROMPT):
    """Normalize raw input without attempting to undo an existing template."""
    if isinstance(prompt, str):
        if "<|im_start|>" in prompt or "<|im_end|>" in prompt:
            raise ValueError(
                "ReTool received a formatted prompt without raw messages. Reload "
                "the dataset with this version to populate Sample.raw_prompt."
            )
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and prompt:
        messages = copy.deepcopy(prompt)
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system", "user", "assistant", "tool"
            }:
                raise ValueError("ReTool expects text chat messages with explicit roles.")
            if not isinstance(message.get("content", ""), str):
                raise ValueError("ReTool supports text message content only.")
            content = message.get("content", "")
            if "<|im_start|>" in content or "<|im_end|>" in content:
                raise ValueError("Raw ReTool messages must not contain ChatML control tokens.")
            message.setdefault("content", "")
    else:
        raise ValueError("ReTool requires a raw string or a nonempty list of messages.")
    if any(message["role"] == "system" for message in messages[1:]):
        raise ValueError("ReTool supports a single leading system message.")
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def _template_kwargs(kwargs):
    kwargs = dict(kwargs or {})
    reserved = {"tools", "tokenize", "add_generation_prompt", "return_dict", "messages", "conversation"}
    conflicts = reserved.intersection(kwargs)
    if conflicts:
        raise ValueError("ReTool template options cannot override: " + ", ".join(sorted(conflicts)))
    return kwargs


def render_initial_prompt(tokenizer, prompt, *, tools=None, chat_template_kwargs=None):
    """Apply the actual checkpoint template once to the raw conversation."""
    rendered = tokenizer.apply_chat_template(
        normalize_messages(prompt),
        tools=get_tool_specs() if tools is None else canonicalize_tools(tools),
        tokenize=False,
        add_generation_prompt=True,
        **_template_kwargs(chat_template_kwargs),
    )
    if not isinstance(rendered, str) or not rendered.startswith("<|im_start|>system\n"):
        raise ValueError("ReTool qwen-tools-v1 requires a Qwen ChatML tool template.")
    if "<tools>" not in rendered or "code_interpreter" not in rendered:
        raise ValueError("The checkpoint chat template did not render the tool schema.")
    return rendered


def normalize_tool_observation(text, *, from_executor=False):
    """Remove the executor's display-only stdout heading, preserving errors and stderr."""
    if not isinstance(text, str):
        raise ValueError("Tool observations must be strings.")
    if from_executor and text.startswith("Output:\n"):
        text = text[len("Output:\n"):]
    return text.strip()


def render_tool_response_suffix(tokenizer, results, *, chat_template_kwargs=None):
    """Render appended environment tokens only; the entire suffix has loss mask zero."""
    if not results or any(not isinstance(result, str) for result in results):
        raise ValueError("Tool results must be a nonempty list of strings.")
    if any("<|im_start|>" in result or "<|im_end|>" in result for result in results):
        raise ValueError("Tool output contains unsupported ChatML control tokens.")
    rendered = tokenizer.apply_chat_template(
        [{"role": "tool", "content": normalize_tool_observation(result, from_executor=True)} for result in results],
        tokenize=False,
        add_generation_prompt=True,
        **_template_kwargs(chat_template_kwargs),
    )
    if not isinstance(rendered, str) or not rendered.startswith("<|im_start|>user\n<tool_response>\n"):
        raise ValueError("The checkpoint does not provide the expected Qwen tool-response template.")
    if rendered.count("<|im_start|>assistant\n") != 1:
        raise ValueError("The tool suffix must open exactly one assistant turn.")
    return "\n" + rendered


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantAction:
    kind: str
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    error: str = ""


def _invalid(message):
    return AssistantAction("invalid", error=message)


def strip_python_fence(code):
    """Unwrap one complete outer Python fence, preserving the program."""
    code = code.strip()
    if code.startswith("```"):
        match = re.fullmatch(r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n?```", code, re.DOTALL)
        if match is None:
            raise ValueError("Incomplete or unsupported Python code fence.")
        code = match.group(1).strip()
    if not code:
        raise ValueError("The Python code is empty.")
    return code


def _last_boxed_answer(text):
    answer = None
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, start, end = 1, match.end(), match.end()
        while end < len(text) and depth:
            char = text[end]
            backslashes, cursor = 0, end - 1
            while cursor >= 0 and text[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
            end += 1
        if depth == 0:
            answer = text[start:end - 1].strip()
    return answer


def _invalid_outside_action(text):
    markers = ("<|im_start|>", "<|im_end|>", "<|endoftext|>",
               "<tool_response>", "</tool_response>", "<interpreter>",
               "</interpreter>", "</tool_call>")
    return any(marker in text for marker in markers)


def parse_assistant_action(text, *, allow_legacy=True, reasoning_prefilled=False, runtime_compat=False):
    """Parse tool-first assistant actions without rewriting sampled text.

    Legacy code syntax is optional; runtime_compat relaxes JSON and extra arguments only."""
    if not isinstance(text, str):
        return _invalid("Assistant output must be text.")
    body = text.strip()
    for terminal in ("<|im_end|>", "<|endoftext|>"):
        if body.endswith(terminal):
            body = body[:-len(terminal)].rstrip()
            break
    if body.startswith("<think>") or reasoning_prefilled:
        end = body.find("</think>")
        if end < 0:
            return _invalid("The assistant reasoning block is incomplete.")
        body = body[end + len("</think>"):].lstrip()
    elif body.startswith("</think>"):
        body = body[len("</think>"):].lstrip()

    calls, cursor, outside = [], 0, []
    decoder = json.JSONDecoder(strict=not runtime_compat)
    while True:
        start = body.find("<tool_call>", cursor)
        if start < 0:
            break
        outside.append(body[cursor:start])
        json_start = start + len("<tool_call>")
        while json_start < len(body) and body[json_start].isspace():
            json_start += 1
        try:
            value, end = decoder.raw_decode(body, json_start)
        except (json.JSONDecodeError, ValueError):
            return _invalid("Malformed JSON in tool_call.")
        while end < len(body) and body[end].isspace():
            end += 1
        if not body.startswith("</tool_call>", end):
            return _invalid("Missing tool_call closing tag.")
        if not isinstance(value, dict) or value.get("name") != "code_interpreter":
            return _invalid("Unsupported tool name; expected code_interpreter.")
        arguments = value.get("arguments")
        if not isinstance(arguments, dict) or "code" not in arguments:
            return _invalid("code_interpreter requires a code argument.")
        if not runtime_compat and set(arguments) != {"code"}:
            return _invalid("code_interpreter requires exactly one code argument.")
        if not isinstance(arguments["code"], str) or not arguments["code"].strip():
            return _invalid("The code argument must be a nonempty string.")
        calls.append(ToolCall("code_interpreter", {"code": arguments["code"]}))
        cursor = end + len("</tool_call>")
    if calls:
        outside.append(body[cursor:])
        if _invalid_outside_action("".join(outside)):
            return _invalid("Unexpected role, observation, or tool delimiter outside a tool call.")
        return AssistantAction("tool", calls=calls)
    if "</tool_call>" in body:
        return _invalid("Unexpected tool_call closing tag.")

    if allow_legacy:
        code_start = body.find("<code>")
        if code_start >= 0:
            code_end = body.find("</code>", code_start + len("<code>"))
            if code_end < 0 or "<code>" in body[code_end + len("</code>"):]:
                return _invalid("Incomplete or ambiguous legacy code block.")
            try:
                code = strip_python_fence(body[code_start + len("<code>"):code_end])
            except ValueError as exc:
                return _invalid(str(exc))
            if _invalid_outside_action(body[:code_start] + body[code_end + len("</code>"):]):
                return _invalid("Unexpected role or observation outside legacy code.")
            return AssistantAction("tool", calls=[ToolCall("code_interpreter", {"code": code})])
        fence = re.search(r"```(?:python|py)[ \t]*\r?\n.*?```", body, re.DOTALL)
        if fence:
            if re.search(r"```(?:python|py)[ \t]*\r?\n", body[fence.end():]):
                return _invalid("Multiple unstructured Python blocks are ambiguous.")
            try:
                code = strip_python_fence(fence.group())
            except ValueError as exc:
                return _invalid(str(exc))
            if _invalid_outside_action(body[:fence.start()] + body[fence.end():]):
                return _invalid("Unexpected role or observation outside a Python fence.")
            return AssistantAction("tool", calls=[ToolCall("code_interpreter", {"code": code})])
    if _invalid_outside_action(body):
        return _invalid("Unexpected role or observation in assistant output.")
    answer = _last_boxed_answer(body)
    if answer is not None:
        return AssistantAction("answer", answer=answer)
    return _invalid("No complete tool call or boxed final answer in assistant output.")
