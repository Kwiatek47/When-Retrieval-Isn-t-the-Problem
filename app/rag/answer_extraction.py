from __future__ import annotations

import re


_ANSWER_BLOCK_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_ANSWER_OPEN_PATTERN = re.compile(r"<answer>\s*(.*)", re.IGNORECASE | re.DOTALL)
_THINKING_BLOCK_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.IGNORECASE | re.DOTALL)


def extract_answer_content(content: str) -> str:
    """Return the user-facing answer from model markup when present."""

    answer_blocks = [match.strip() for match in _ANSWER_BLOCK_PATTERN.findall(content) if match.strip()]
    if answer_blocks:
        return _normalize(answer_blocks[-1])

    open_answer = _ANSWER_OPEN_PATTERN.search(content)
    if open_answer:
        return _normalize(open_answer.group(1))

    without_thinking = _THINKING_BLOCK_PATTERN.sub("", content)
    return _normalize(without_thinking)


def _normalize(content: str) -> str:
    return content.strip()
