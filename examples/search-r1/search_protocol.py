"""Qwen/Llama Search-SFT tool protocols, following the reference SearchSFTTool adapter."""

import json
import re


MAX_QUERIES = 32
MAX_QUERY_LENGTH = 2048


def normalize_protocol(protocol):
    if protocol == "llama":
        return "llama3"
    if protocol not in ("qwen", "llama3"):
        raise ValueError("search protocol must be qwen or llama3")
    return protocol


def protocol_turn_end(protocol):
    return "<|eot_id|>" if normalize_protocol(protocol) == "llama3" else "<|im_end|>"


def _validate_search_call(call, argument_key):
    if not isinstance(call, dict) or call.get("name") != "search":
        raise ValueError("only the search tool is supported")
    arguments = call.get(argument_key)
    if not isinstance(arguments, dict):
        raise ValueError(f"search {argument_key} must be a JSON object")
    queries = arguments.get("query_list")
    if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
        raise ValueError(f"query_list must contain 1 to {MAX_QUERIES} queries")
    if not all(isinstance(q, str) and q.strip() and len(q) <= MAX_QUERY_LENGTH for q in queries):
        raise ValueError(f"each query must be a nonempty string of at most {MAX_QUERY_LENGTH} characters")
    return {"name": "search", "arguments": {"query_list": queries}}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON keys in tool call")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"non-JSON constant in tool call: {value}")


_LLAMA_JSON = json.JSONDecoder(object_pairs_hook=_unique_json_object, parse_constant=_invalid_json_constant)


def _json_container_end_and_keys(text, start):
    """Scan an outer JSON container and its keys, ignoring quoted braces and nested objects."""
    stack = []
    keys = set()
    pos = start
    while pos < len(text):
        char = text[pos]
        if char == '"':
            try:
                value, end = json.JSONDecoder().raw_decode(text, pos)
            except ValueError:
                return None, keys
            if len(stack) == 1 and stack[0] == "{":
                after = end
                while after < len(text) and text[after].isspace():
                    after += 1
                if after < len(text) and text[after] == ":":
                    keys.add(value)
            pos = end
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != ("{" if char == "}" else "["):
                return pos + 1, keys
            stack.pop()
            if not stack:
                return pos + 1, keys
        pos += 1
    return None, keys


def _complete_box_end(text, start):
    depth = 1
    for end in range(start + len(r"\boxed{"), len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return end + 1
    return None


def _llama_events(text):
    """Yield outer JSON calls, boxed answers and turn boundaries, skipping quoted or nested content."""
    pos = 0
    while pos < len(text):
        if text.startswith(r"\boxed{", pos):
            end = _complete_box_end(text, pos)
            if end is not None:
                yield "boxed", pos, end, None
            return
        if text.startswith("<|eot_id|>", pos):
            yield "turn_end", pos, pos + len("<|eot_id|>"), None
            return
        if text[pos] in "{[":
            end, keys = _json_container_end_and_keys(text, pos)
            try:
                value, parsed_end = _LLAMA_JSON.raw_decode(text, pos)
            except (ValueError, RecursionError):
                if text[pos] == "{" and keys.intersection(("name", "parameters", "arguments")):
                    yield "invalid_tool_call", pos, end, None
                    return
                # Do not execute a nested object rescued out of invalid JSON.
                if end is None:
                    return
                pos = end
                continue
            if isinstance(value, dict) and (
                value.get("name") == "search"
                or "parameters" in value
                or ("name" in value and "arguments" in value)
            ):
                yield "tool_call", pos, parsed_end, value
            pos = parsed_end
            continue
        if text[pos] == '"':
            try:
                _, end = _LLAMA_JSON.raw_decode(text, pos)
            except ValueError:
                pass
            else:
                pos = end
                continue
        pos += 1


def parse_tool_call(text, *, protocol="qwen", multiple="first"):
    """Parse the first tool call; return None for prose and raise on malformed requests."""
    protocol = normalize_protocol(protocol)
    if multiple not in ("first", "error"):
        raise ValueError("multiple must be first or error")
    if not isinstance(text, str):
        raise ValueError("assistant response must be a string")
    if protocol == "llama3":
        calls = []
        for boundary, _start, _end, value in _llama_events(text):
            if boundary in ("boxed", "turn_end"):
                break
            if boundary == "invalid_tool_call":
                raise ValueError("expected a complete JSON tool call")
            calls.append(_validate_search_call(value, "parameters"))
            if multiple == "first":
                return calls[0]
            if len(calls) > 1:
                raise ValueError("expected exactly one complete tool call per assistant round")
        return calls[0] if calls else None
    if "<tool_call" not in text and "</tool_call" not in text:
        return None
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S))
    if not matches or matches[0].start() != text.find("<tool_call"):
        raise ValueError("expected a complete tool_call")
    if multiple == "error" and (len(matches) != 1 or text.count("<tool_call") != 1 or text.count("</tool_call") != 1):
        raise ValueError("expected exactly one complete tool_call per assistant round")
    try:
        call = json.loads(matches[0].group(1))
    except (ValueError, TypeError) as exc:
        raise ValueError("tool_call must contain a JSON object") from exc
    return _validate_search_call(call, "arguments")


def truncate_assistant_turn(text, *, protocol="qwen"):
    """Trim at a final box or tool boundary; the caller must align the original sampled token IDs."""
    protocol = normalize_protocol(protocol)
    if not isinstance(text, str):
        raise ValueError("assistant response must be a string")
    if protocol == "llama3":
        for boundary, start, end, _value in _llama_events(text):
            if boundary == "invalid_tool_call":
                return text[:start].rstrip(), boundary
            return text[:end], boundary
        return text, None
    tool_start = text.find("<tool_call")
    assistant_text = text if tool_start < 0 else text[:tool_start]
    for match in re.finditer(r"\\boxed\{", assistant_text):
        end = _complete_box_end(assistant_text, match.start())
        if end is not None:
            return text[:end], "boxed"
    if tool_start >= 0:
        match = re.search(r"<tool_call>.*?</tool_call>", text[tool_start:], re.S)
        if match is not None and match.start() == 0:
            return text[:tool_start + match.end()], "tool_call"
        return text[:tool_start].rstrip(), "invalid_tool_call"
    return text, None


def unwrap_search_response(response):
    """Unwrap tool JSON and normalize retrieved-document headings."""
    try:
        parsed = json.loads(response)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
        response = parsed["result"]
    return re.sub(r"(?m)^Doc (\d+) \(Title: ([^\n]*)\)\n", r"Doc \1(Title: \2) ", response).strip()


def split_search_documents(response):
    """Split the adapter's formatted response into queries and ranked documents."""
    groups = []
    for group in re.split(r"\n\s*---\s*\n", response):
        documents = [part.strip() for part in re.split(r"(?m)(?=^Doc \d+\(Title: )", group) if part.strip()]
        if not documents or any(not re.match(r"^Doc \d+\(Title: ", doc) for doc in documents):
            return None
        groups.append(documents)
    return groups or None


def _truncate_token_prefix(text, tokenizer, max_tokens):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, False
    truncated_ids = token_ids[:max_tokens]
    text = tokenizer.decode(truncated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    while len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
        truncated_ids = truncated_ids[:-1]
        text = tokenizer.decode(truncated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text, True


def truncate_search_response(response, tokenizer, max_doc_length=200):
    """Apply the reference token cap to each entire document, including title."""
    if not isinstance(max_doc_length, int) or isinstance(max_doc_length, bool) or max_doc_length <= 0:
        raise ValueError("max_doc_length must be a positive integer")
    original_length = len(tokenizer.encode(response, add_special_tokens=False))
    groups = split_search_documents(response)
    if groups is None:
        response, truncated = _truncate_token_prefix(response, tokenizer, max_doc_length)
        return response, original_length, truncated
    output_groups = []
    any_truncated = False
    for documents in groups:
        output_documents = []
        for document in documents:
            document, truncated = _truncate_token_prefix(document, tokenizer, max_doc_length)
            output_documents.append(document)
            any_truncated = any_truncated or truncated
        output_groups.append("\n\n".join(output_documents))
    if not any_truncated:
        return response, original_length, False
    return "\n\n---\n\n".join(output_groups), original_length, True


def render_search_results(payload, tokenizer, *, topk=3, max_doc_length=200):
    """Render grouped contents/document rows or a result wrapper, taking titles from the corpus text."""
    if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
        raise ValueError("topk must be a positive integer")
    if isinstance(payload, dict):
        if "result" not in payload:
            raise ValueError("retrieval response is missing result")
        payload = payload["result"]
    if isinstance(payload, str):
        response = unwrap_search_response(payload)
    elif isinstance(payload, list):
        groups = []
        for group in payload:
            if not isinstance(group, list):
                raise ValueError("each retrieval query result must be a list")
            documents = []
            for rank, item in enumerate(group[:topk], 1):
                if not isinstance(item, dict):
                    raise ValueError("retrieval document must be an object")
                document = item.get("document", item)
                if not isinstance(document, dict) or not isinstance(document.get("contents"), str):
                    raise ValueError("retrieval document must contain string contents")
                contents = document["contents"]
                title, _, body = contents.partition("\n")
                # Preserve the reference formatter's empty/title-only spacing.
                documents.append(f"Doc {rank} (Title: {title})\n{body}\n\n")
            groups.append("".join(documents).strip())
        response = unwrap_search_response("\n---\n".join(groups) if payload else "No search results found.")
    else:
        raise ValueError("retrieval result must be a list of query groups or a string")
    response, original_tokens, truncated = truncate_search_response(response, tokenizer, max_doc_length)
    groups = split_search_documents(response)
    document_tokens = [len(tokenizer.encode(doc, add_special_tokens=False)) for group in (groups or []) for doc in group]
    return response, {
        "tool_response_original_tokens": original_tokens,
        "tool_response_tokens": len(tokenizer.encode(response, add_special_tokens=False)),
        "tool_response_truncated": truncated,
        "tool_response_document_count": len(document_tokens),
        "tool_response_max_document_tokens": max(document_tokens, default=0),
        "tool_response_max_tokens_per_document": max_doc_length,
        "tool_response_truncation_strategy": "per_document_prefix",
    }


def render_tool_response(response, *, protocol="qwen"):
    """Render only appended environment text, with zero loss mask and logprob.

    The caller supplies any missing turn end; Llama observations are JSON strings."""
    protocol = normalize_protocol(protocol)
    if not isinstance(response, str):
        raise ValueError("tool response must be a string")
    if protocol == "llama3":
        return (
            "<|start_header_id|>ipython<|end_header_id|>\n\n"
            + json.dumps(response, ensure_ascii=False)
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    return (
        "\n<|im_start|>user\n<tool_response>\n"
        + response
        + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
