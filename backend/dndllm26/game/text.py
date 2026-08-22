from __future__ import annotations

import re
from typing import Any

MAX_DM_CHARS = 3_000
MAX_DM_WORDS = 260

_PLACEHOLDER_CHOICES = (
    "a concise action",
    "another concise action",
    "specific playable action",
    "the player can take",
    "option 1",
    "option 2",
    "option 3",
    "insert action",
    "player action here",
)
_PLACEHOLDER_TEXT = (
    "short current location",
    "current immediate objective",
    "one sentence scene summary",
    "specific scene summary",
    "location here",
    "objective here",
)


def strip_dm_meta_output(text: str) -> str:
    clean = text.strip()
    quoted = re.search(
        r"""(?:here(?:'s| is)|this is|a possible|possible first|first dm message).*?["“](?P<body>.+?)["”]""",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if quoted and len(quoted.group("body").split()) > 20:
        clean = quoted.group("body").strip()
    clean = re.sub(
        r"^\s*(?:here(?:'s| is)|this is|a possible|possible)\b[^:\n]*:\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.split(
        r"\n?\s*(?:this message establishes|the message establishes|it establishes)\b",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return clean.strip()


def trim_dm_text(text: str) -> str:
    clean = "\n".join(line.strip() for line in text.replace("\r\n", "\n").splitlines())
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    clean = strip_dm_meta_output(clean)
    clean = re.split(
        r"\n\s*(?:(?:you have )?the following options|choices|options|what do you do)(?:\s+are)?\s*:.*",
        clean,
        maxsplit=1,
        flags=re.IGNORECASE | re.DOTALL,
    )[0].strip()
    words = clean.split()
    if len(words) > MAX_DM_WORDS:
        clean = " ".join(words[:MAX_DM_WORDS]).rstrip()
    if len(clean) > MAX_DM_CHARS:
        candidate = clean[:MAX_DM_CHARS]
        sentence = max(candidate.rfind(". "), candidate.rfind("! "), candidate.rfind("? "))
        clean = (
            candidate[: sentence + 1]
            if sentence > MAX_DM_CHARS // 2
            else candidate.rsplit(" ", 1)[0]
        ).rstrip()
    # Small local models often stop halfway through a numbered list or sentence.
    # Keep the last complete sentence when enough useful prose precedes it.
    if clean and clean[-1] not in '.!?"”':
        endings = [clean.rfind(mark) for mark in (". ", "! ", "? ", '."', '!"', '?"')]
        boundary = max(endings)
        if boundary >= max(80, len(clean) // 2):
            clean = clean[: boundary + 1].rstrip()
    return clean


def clean_choice_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("action", "choice", "text", "label", "description"):
            if key in value:
                return clean_choice_value(value[key])
        return ""
    if not isinstance(value, (str, int, float)):
        return ""
    text = str(value).strip()
    object_match = re.search(
        r"""["']?(?:action|choice|text|label)["']?\s*:\s*["'](?P<value>.+?)["']\s*[},]?$""",
        text,
        flags=re.IGNORECASE,
    )
    if object_match:
        text = object_match.group("value")
    text = re.sub(r"^\s*(?:\d+[\).:]|-|\*)\s+", "", text).strip()
    text = re.sub(r"^\{+|\}+$", "", text).strip()
    text = re.sub(r"^['\"]+|['\"]+$", "", text).strip()
    text = re.sub(r"\*\*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_placeholder_choice(choice: str) -> bool:
    normalized = choice.casefold()
    return any(fragment in normalized for fragment in _PLACEHOLDER_CHOICES)


def clean_choice_list(value: Any, *, limit: int = 4) -> list[str]:
    if isinstance(value, dict):
        value = value.get("choices") or value.get("actions") or value.get("options") or []
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        choice = clean_choice_value(item)[:140]
        normalized = choice.casefold()
        if 4 <= len(choice) <= 140 and not is_placeholder_choice(choice) and normalized not in seen:
            seen.add(normalized)
            result.append(choice)
        if len(result) >= limit:
            break
    return result


def extract_choices(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n")
    section = re.search(
        r"(?:^|\n)\s*(?:choices|what do you do)[?:]?\s*\n(?P<body>.*)$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if section:
        cleaned = section.group("body")
    choices: list[str] = []
    for line in cleaned.splitlines():
        match = re.match(
            r"^\s*(?:(?:\d+[\).:])|(?:option\s+\d+[\).:])|-|\*)\s+(.*?)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            choices.append(clean_choice_value(match.group(1)))
    return clean_choice_list(choices)


def is_placeholder_text(value: str) -> bool:
    normalized = value.casefold().strip()
    return any(fragment in normalized for fragment in _PLACEHOLDER_TEXT)


def clean_extracted_text(value: Any, fallback: str, *, limit: int, default: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text or is_placeholder_text(text):
        text = fallback.strip()
    if not text or is_placeholder_text(text):
        text = default
    return text[:limit]


def format_summary_source(text: str) -> str:
    return re.sub(r"(\*\*|#{1,6}\s|---+)", "", text).strip()


def fallback_summary(text: str) -> str:
    return " ".join(format_summary_source(text).split()[:42])[:260]


def clean_roll_text(value: Any, fallback: str) -> str:
    text = clean_choice_value(value)
    blocked = (
        "you have three options",
        "you have 3 options",
        "option 1",
        "option 2",
        "option 3",
        "this message establishes",
    )
    if not text or any(fragment in text.casefold() for fragment in blocked):
        text = fallback
    return text[:180]
