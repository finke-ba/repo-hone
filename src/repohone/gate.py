"""The approval gate (ARCH §9): which RepoHone commands wait for the developer's
approval in the host's prompt, and how they are recognised in a shell command.

The CLI applies `decide` to its own argv, so the gate and the command it guards
can never disagree about what needs approval."""
from __future__ import annotations

import json
import os
import re
import shlex
from typing import List, Optional, Tuple

ASK, DENY = "ask", "deny"

MINTS = ("This command runs code RepoHone cannot read that names its hook or its "
         "approvals, which can record an approval. Approve only if you asked for it.")

_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_REDIRECT = re.compile(r"\d*(?:>>|>&|<&|&>|>\||<<<|>|<)(.*)")
_WRAPPERS = {"env", "command", "exec", "nohup", "time", "builtin", "nice", "sudo"}
_UV_VALUED = {"--with", "--from", "--python", "-p", "--directory", "--project",
              "--env-file", "--extra", "--group", "--index", "--with-requirements"}
_PYTHON = re.compile(r"python[0-9.]*")
_MODULES = {"repohone": None, "repohone.cli": None, "repohone.hook": "hook"}
# Forms the parser cannot follow. No approval is recorded for what they run, so a
# gated command inside one refuses by itself; only a way to record one is asked.
_OPAQUE = re.compile(r"\$\(|`|<<(?!<)|\beval\b|\bxargs\b|\bsource\b|(?:^|[\s;&|(])\.\s"
                     r"|\b(?:ba|z|da|k|c|tc|fi)?sh\b\s+(?:-\w+\s+)*-\w*c|"
                     r"\bpython[0-9.]*\b\s+(?:-\w+\s+)*-c\b")

_RECORDS_APPROVAL = re.compile(r"hook\b|approvals", re.IGNORECASE)


def _safe(value: Optional[str]) -> str:
    """Ids go into a prompt that is also egress: nothing but id characters."""
    return re.sub(r"[^A-Za-z0-9_.-]", "", value or "")[:48] or "?"


def _value(argv: List[str], option: str) -> Optional[str]:
    for i, word in enumerate(argv):
        if word == option and i + 1 < len(argv):
            return argv[i + 1]
        if word.startswith(option + "="):
            return word.split("=", 1)[1]
    return None


def _positional(argv: List[str], valued=()) -> Optional[str]:
    """The first argument after the subcommand that is not an option or its value."""
    skip = False
    for word in argv[1:]:
        if skip:
            skip = False
            continue
        if word.startswith("-"):
            skip = word in valued
            continue
        return word
    return None


def decide(argv: List[str]) -> Optional[Tuple[str, str]]:
    """(ASK or DENY, the reason the prompt shows), or None when nothing needs approval.
    `argv` is what follows the program: `["apply", "p-9", "--yes"]`."""
    sub = next((w for w in argv if not w.startswith("-")), None)
    if sub is None:
        return None
    argv = argv[argv.index(sub):]
    flags = {w.split("=", 1)[0] for w in argv if w.startswith("--")}
    yes = "--yes" in flags
    tail = " Approve only if you asked for it."
    if sub == "hook":
        return DENY, ("`repohone hook` takes events from Claude Code's hooks only, "
                      "never from a shell.")
    if sub == "diagnose" and yes:
        candidate = _value(argv, "--candidate")
        valued = ("--rule", "--turn", "--model", "--timeout", "--path")
        what = (f"rule {_safe(candidate)}" if candidate else
                f"session {_safe(_positional(argv, valued))}")
        return ASK, (f"RepoHone will send selected evidence about {what} to the analysis "
                     f"model for a diagnosis.{tail}")
    if sub == "propose" and yes:
        target = _safe(_positional(argv, ("--model", "--timeout", "--validate-timeout", "--path")))
        return ASK, (f"RepoHone will send diagnosis {target} and this repository's check "
                     f"inventory to the analysis model, then run project commands in a "
                     f"scratch checkout to validate the change.{tail}")
    if sub == "apply" and yes:
        return ASK, (f"RepoHone will change files in your working tree to apply proposal "
                     f"{_safe(_positional(argv, ('--path',)))}; rollback undoes it.{tail}")
    if sub == "rollback" and yes:
        return ASK, (f"RepoHone will restore the files proposal "
                     f"{_safe(_positional(argv, ('--path',)))} changed.{tail}")
    if sub == "init":
        return ASK, ("RepoHone will record your acceptance of this project's profile and "
                     f"start capturing your sessions here.{tail}")
    if sub == "deinit" and yes and "--dry-run" not in flags:
        purge = (" and permanently delete this checkout's local evidence"
                 if "--purge" in flags else "")
        return ASK, f"RepoHone will stop capturing your sessions here{purge}.{tail}"
    if sub == "mechanisms" and "--probe" in flags and (yes or "--only" in flags):
        return ASK, ("RepoHone will run this project's own commands in a scratch checkout, "
                     f"with no network access.{tail}")
    if sub == "bootstrap" and "--include-remote" in flags:
        return ASK, ("RepoHone will call GitHub through gh to read pull-request review "
                     f"comments and CI failures.{tail}")
    if sub == "uninstall" and yes and "--dry-run" not in flags:
        purge = " and permanently delete all its local evidence" if "--purge" in flags else ""
        return ASK, f"RepoHone will remove its plugin from this machine{purge}.{tail}"
    if sub == "install":
        return ASK, f"RepoHone will rewrite its plugin on this machine.{tail}"
    return None


def _simple_commands(command: str) -> List[str]:
    """Split at shell operators outside quotes."""
    parts, current, quote, i = [], [], None, 0
    while i < len(command):
        ch = command[i]
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"' and i + 1 < len(command):
                current.append(command[i + 1])
                i += 1
        elif ch in "'\"":
            quote = ch
            current.append(ch)
        elif ch == "\\" and i + 1 < len(command):
            current.extend(command[i:i + 2])
            i += 1
        elif ch in ";|&\n(){}":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    parts.append("".join(current))
    return [p for p in parts if p.strip()]


def _without_redirections(words: List[str]) -> List[str]:
    out, skip = [], False
    for word in words:
        if skip:
            skip = False
            continue
        match = _REDIRECT.fullmatch(word)
        if match:
            skip = not match.group(1)
            continue
        out.append(word)
    return out


def invocation(words: List[str]) -> Optional[List[str]]:
    """The argv RepoHone would receive from one simple command, or None."""
    i = 0
    while i < len(words):
        word = words[i]
        base = os.path.basename(word)
        if _ASSIGNMENT.match(word):
            i += 1
        elif base in _WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or _ASSIGNMENT.match(words[i])):
                i += 1
        elif base == "uvx" or (base == "uv" and words[i + 1:i + 2] in (["run"], ["tool"])):
            i += 1 if base == "uvx" else 2
            if words[i:i + 1] == ["run"]:
                i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in _UV_VALUED else 1
        elif _PYTHON.fullmatch(base):
            j = i + 1
            while j < len(words) and words[j].startswith("-") and words[j] not in ("-m", "-c"):
                j += 1
            if words[j:j + 1] == ["-m"] and j + 1 < len(words) and words[j + 1] in _MODULES:
                forced = _MODULES[words[j + 1]]
                return [forced] if forced else words[j + 2:]
            return None
        elif base == "repohone":
            return words[i + 1:]
        elif base in ("repohone-hook", "repohone_hook.py"):
            return ["hook"]
        else:
            return None
    return None


def for_command(command: str) -> Tuple[Optional[Tuple[str, str]], List[List[str]]]:
    """The gate's decision for one Bash command, and the argv of each RepoHone
    invocation in it that needs approval."""
    if "repohone" not in (command or "").lower():
        return None, []
    asked, decisions = [], []
    opaque = bool(_OPAQUE.search(command))
    for part in _simple_commands(command):
        try:
            words = _without_redirections(shlex.split(part))
        except ValueError:
            opaque = True
            continue
        argv = invocation(words)
        if argv is None:
            continue
        decision = decide(argv)
        if decision:
            decisions.append(decision)
            if decision[0] == ASK:
                asked.append(argv)
    denied = next((d for d in decisions if d[0] == DENY), None)
    if denied:
        return denied, []
    reasons = [d[1] for d in decisions]
    if opaque and _RECORDS_APPROVAL.search(command):
        reasons.append(MINTS)
    if reasons:
        return (ASK, " ".join(reasons)), asked
    return None, []


def hook_output(decision: Tuple[str, str]) -> str:
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": decision[0],
        "permissionDecisionReason": decision[1]}})
