"""Secret redaction applied before anything is stored (ARCH §12).

Deliberately over-eager and irreversible: a prompt discussing credentials may
come back partly masked. `redactions` counts how often it fired.
"""
from __future__ import annotations

import re
from typing import Tuple

MASK = "[REDACTED]"

_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
               re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bASIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
    # Authorization headers carry the credential in the value, not after an `=`.
    re.compile(r"(?i)\b(?:bearer|token)\s+([A-Za-z0-9._\-+/=]{12,})"),
    re.compile(r"(?i)\bbasic\s+([A-Za-z0-9+/=]{12,})"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key|"
               r"authorization|auth[_-]?token|private[_-]?key|client[_-]?secret)\b"
               r"\s*[:=]\s*[\"']?(?!\[REDACTED\])([^\s\"',;]{8,})[\"']?"),
]


def redact(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    count = 0
    out = text
    for pattern in _PATTERNS:
        out, n = pattern.subn(
            lambda m: m.group(0).replace(m.group(1), MASK) if m.groups() else MASK, out)
        count += n
    return out, count
