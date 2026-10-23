# Modified for COPC.
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.system_message_length = 0
        self.gen_token_length = 0
        if self.tokenizer_type in ("qwen", "qwen3"):
            self.system_message_length, self.gen_token_length = self.get_system_message_length()

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages,
                add_special_tokens=False,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_structured_tool_loss_mask(
        self, messages: list[dict], tools: list[dict], chat_template_kwargs: dict | None = None
    ) -> tuple[list[int], list[int]]:
        """Mask the full Qwen conversation, rejecting data that spoofs ChatML role delimiters."""
        if not isinstance(messages, list) or not messages or not tools:
            raise ValueError("Structured tool SFT requires nonempty messages and tool definitions.")
        from slime.utils.retool_protocol import canonicalize_tools
        tools = canonicalize_tools(tools)

        def check_reserved(value):
            if isinstance(value, str):
                if any(token in value for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                    raise ValueError("Structured tool SFT data contains reserved Qwen conversation control tokens.")
            elif isinstance(value, dict):
                for key, item in value.items():
                    check_reserved(key)
                    check_reserved(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check_reserved(item)

        check_reserved(messages)
        check_reserved(tools)
        if any(not isinstance(message, dict) or message.get("role") not in
               ("system", "user", "assistant", "tool") for message in messages):
            raise ValueError("Structured tool SFT supports only system, user, assistant and tool messages.")
        assistants = [message for message in messages if message["role"] == "assistant"]
        if not assistants:
            raise ValueError("Structured tool SFT requires an assistant target.")
        kwargs = dict(chat_template_kwargs or {})
        if any(key in kwargs for key in ("tokenize", "return_dict", "tools", "add_generation_prompt",
                                        "continue_final_message", "return_assistant_tokens_mask")):
            raise ValueError("SFT template kwargs must not override conversation rendering controls.")
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=False, **kwargs
        )
        # Input delimiters are rejected, so these frames belong to the template.
        frame = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n([\s\S]*?)<\|im_end\|>(?:\n)?")
        spans = []
        cursor = 0
        assistant_index = 0
        for match in frame.finditer(rendered):
            if match.start() != cursor:
                raise ValueError("Unsupported Qwen template: unrecognized text outside ChatML message frames.")
            cursor = match.end()
            if match.group(1) == "assistant":
                if assistant_index >= len(assistants):
                    raise ValueError("The template emitted more assistant frames than the input conversation.")
                if assistants[assistant_index].get("step_loss_mask") in (None, 1):
                    # Include assistant EOS; exclude the environment newline.
                    spans.append((match.start(2), match.end(2) + len("<|im_end|>")))
                assistant_index += 1
        if cursor != len(rendered) or assistant_index != len(assistants):
            raise ValueError("Unsupported Qwen template or mismatched assistant message frames.")
        try:
            encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        except (NotImplementedError, TypeError) as error:
            raise ValueError("Structured tool SFT requires a fast tokenizer with offset mappings.") from error
        token_ids = encoded["input_ids"]
        offsets = encoded.get("offset_mapping")
        if offsets is None or len(offsets) != len(token_ids):
            raise ValueError("Structured tool SFT requires one character-offset mapping per token.")
        expected = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=False, return_dict=False, **kwargs
        )
        if token_ids != expected:
            raise ValueError("Full-conversation tokenization differs from the native chat template token IDs.")
        loss_mask = []
        span_index = 0
        for start, end in offsets:
            if not (0 <= start <= end <= len(rendered)):
                raise ValueError("Invalid tokenizer offset mapping for the rendered conversation.")
            while span_index < len(spans) and spans[span_index][1] <= start:
                span_index += 1
            active = (end > start and span_index < len(spans)
                      and start < spans[span_index][1] and end > spans[span_index][0])
            loss_mask.append(int(active))
        return token_ids, loss_mask

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, tools=tools, return_dict=False
                )
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True, return_dict=False)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message],
                    tokenize=True,
                    tools=tools,
                    return_dict=False,
                )
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template(
                    [prefix_message, message],
                    tokenize=True,
                    return_dict=False,
                )
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3_5(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        rendered_text = self.tokenizer.apply_chat_template(messages, tokenize=False, tools=tools, return_dict=False)
        tokenized = self.tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = tokenized["input_ids"]
        offset_mapping = tokenized.get("offset_mapping")

        if offset_mapping is None:
            raise ValueError(
                "Qwen3.5 loss mask generation requires a fast tokenizer " "with `return_offsets_mapping` support."
            )

        expected_token_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, tools=tools, return_dict=False
        )
        if token_ids != expected_token_ids:
            raise ValueError(
                "Qwen3.5 rendered text tokenization does not match "
                "`apply_chat_template(..., tokenize=True)` output."
            )

        assistant_header = "<|im_start|>assistant\n"
        think_prefix = "<think>\n"
        end_marker = "<|im_end|>"

        char_mask = [0] * len(rendered_text)
        cursor = 0

        for message in messages:
            if message["role"] != "assistant":
                continue

            header_pos = rendered_text.find(assistant_header, cursor)
            if header_pos < 0:
                raise ValueError("Failed to locate assistant message in rendered Qwen3.5 chat template output.")

            content_start = header_pos + len(assistant_header)
            end_pos = rendered_text.find(end_marker, content_start)
            if end_pos < 0:
                raise ValueError("Failed to locate <|im_end|> for assistant message in rendered text.")

            span_end = end_pos + len(end_marker)
            if span_end < len(rendered_text) and rendered_text[span_end] == "\n":
                span_end += 1
            cursor = span_end

            if message.get("step_loss_mask", 1) != 1:
                continue

            if rendered_text[content_start : content_start + len(think_prefix)] == think_prefix:
                mask_start = content_start + len(think_prefix)
            else:
                mask_start = content_start

            for pos in range(mask_start, span_end):
                char_mask[pos] = 1

        char_mask_prefix_sum = [0]
        for value in char_mask:
            char_mask_prefix_sum.append(char_mask_prefix_sum[-1] + value)

        loss_mask = []
        for start, end in offset_mapping:
            if end <= start:
                loss_mask.append(0)
            else:
                loss_mask.append(1 if char_mask_prefix_sum[end] - char_mask_prefix_sum[start] > 0 else 0)

        return token_ids, loss_mask

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask

    def get_loss_mask(
        self, messages: list[dict], tools: list[dict] = None, *,
        structured_tools: bool = False, chat_template_kwargs: dict | None = None,
    ) -> tuple[list[int], list[int]]:
        if structured_tools:
            return self.gen_structured_tool_loss_mask(messages, tools, chat_template_kwargs)
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "qwen3_5":
            return self.gen_multi_turn_loss_mask_qwen3_5(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        loss_mask = [0] * diff + loss_mask_text

        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
