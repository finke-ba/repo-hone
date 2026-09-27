"""Minimization of host tool inputs before storage (ARCH §5, §12).

Tool inputs carry whole file contents and shell commands, so this is an
allowlist, not a denylist: a tool that gains a new secret-bearing field stays
unrecorded until someone adds it here deliberately.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from . import redact

# Enough to answer "what did the agent consult before acting?"; nothing more.
KEEP = ("file_path", "notebook_path", "path", "pattern", "glob", "command",
        "query", "url", "subagent_type")

PATHLIKE = ("file_path", "notebook_path", "path")
MAX_VALUE_CHARS = 300
OUTSIDE = "<outside-repo>"
REPO_MARKER = "<repo>"
DATA_MARKER = "<data>"


def _relative(value: str, repo_root) -> str:
    """Absolute paths identify the machine and its other projects; a path outside
    the repository is recorded as a fact, not a location."""
    try:
        resolved = Path(value).resolve()
        return str(resolved.relative_to(Path(repo_root).resolve()))
    except (ValueError, OSError):
        return OUTSIDE if os.path.isabs(value) else value


def minimize(tool_input: Optional[Dict[str, Any]], repo_root,
             content_policy: str) -> Optional[Dict[str, Any]]:
    if not isinstance(tool_input, dict):
        return None
    out: Dict[str, Any] = {}
    for key in KEEP:
        if key not in tool_input:
            continue
        value = tool_input[key]
        if not isinstance(value, str):
            continue
        if key in PATHLIKE:
            out[key] = _relative(value, repo_root)
            continue
        if content_policy == "hash_only":
            out[key] = None
            continue
        cleaned, _ = redact.redact(value)
        out[key] = scrub_paths(cleaned, repo_root)[:MAX_VALUE_CHARS]
    return out or None


def _variants(path: str):
    """macOS reports /var and /private/var for one directory; a command string may
    use either spelling. Relative roots are skipped: prefixing them produces
    strings that match unrelated text."""
    if not os.path.isabs(path):
        return set()
    forms = {path.rstrip("/")}
    try:
        forms.add(str(Path(path).resolve()))
    except OSError:
        pass
    for form in list(forms):
        if form.startswith("/private/"):
            forms.add(form[len("/private"):])
        else:
            forms.add("/private" + form)
    return {f for f in forms if f and f != "/"}


# Characters that continue a file name; anything else after the prefix ends it.
_NAME_CHARS = "-_.~+@%"


def _replace_prefix(text: str, prefix: str, marker: str) -> str:
    """Only at a path boundary: /tmp/a must not rewrite /tmp/abc, and
    `cd /tmp/a && ls` must."""
    out, start = [], 0
    while True:
        at = text.find(prefix, start)
        if at == -1:
            out.append(text[start:])
            return "".join(out)
        end = at + len(prefix)
        nxt = text[end:end + 1]
        continues = bool(nxt) and (nxt.isalnum() or nxt in _NAME_CHARS)
        out.append(text[start:at])
        out.append(prefix if continues else marker)
        start = end


def scrub_paths(text: str, repo_root, data=None) -> str:
    """Free-form values (shell commands) carry absolute paths that identify the
    machine and defeat comparison across checkouts."""
    replacements = {}
    for form in _variants(str(data)) if data else ():
        replacements[form] = DATA_MARKER
    for form in _variants(str(repo_root)):
        replacements.setdefault(form, REPO_MARKER)
    home = os.path.expanduser("~")
    if home and home != "/":
        for form in _variants(home):
            replacements.setdefault(form, "~")
    for form in sorted(replacements, key=len, reverse=True):
        text = _replace_prefix(text, form, replacements[form])
    return text


# Outcome fields worth keeping. Never stdout/stderr bodies: a command's output
# carries the same file contents and secrets the inputs do.
RESPONSE_SCALARS = ("status", "interrupted", "is_error", "isError", "success",
                    "returnCode", "exit_code", "exitCode", "agentType")
RESPONSE_SIZED = ("stdout", "stderr", "content", "error", "message")


def summarize_response(tool_response, duration_ms) -> Optional[Dict[str, Any]]:
    """What happened, never what was returned. `keys` records the response shape
    so real sessions can teach us fields worth promoting, without storing values.
    """
    out: Dict[str, Any] = {}
    if isinstance(duration_ms, (int, float)):
        out["duration_ms"] = int(duration_ms)
    if isinstance(tool_response, dict):
        out["keys"] = sorted(tool_response)[:20]
        for key in RESPONSE_SCALARS:
            value = tool_response.get(key)
            if isinstance(value, (bool, int, float)) or (
                    isinstance(value, str) and len(value) <= 40):
                out[key] = value
        for key in RESPONSE_SIZED:
            value = tool_response.get(key)
            if isinstance(value, str):
                out[f"{key}_chars"] = len(value)
            elif isinstance(value, list):
                out[f"{key}_items"] = len(value)
    elif isinstance(tool_response, str):
        out["response_chars"] = len(tool_response)
    return out or None
