"""Retrospective reasoning engine and its failure contract (ARCH §11, §12.1, §12.3).

The engine is probabilistic infrastructure, so every job ends in exactly one
terminal state and a failure never weakens enforcement, blocks work or changes
policy. Reasoning runs in an isolated context: it must not consume the
developer's active coding session.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import paths, processes
from .profile import MODEL_NAME

SUCCESS = "SUCCESS"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NO_ACTIONABLE = "NO_ACTIONABLE_PROJECT_IMPROVEMENT"
MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
INTERRUPTED = "INTERRUPTED"
INVALID_OUTPUT = "INVALID_OUTPUT"
VALIDATION_REJECTED = "VALIDATION_REJECTED"

DISABLED, INTERACTIVE = "disabled", "interactive"
DEFAULT_MODEL = "opus"
_RESTRICTIVENESS = {DISABLED: 0, INTERACTIVE: 1}

# Every provider call, including Phase 5 selection and probabilistic judging,
# uses the same ceiling as diagnosis.  Keeping the final guard at the engine
# boundary means a future call site cannot silently bypass the egress budget.
# Bytes, not tokens: a token is at least one byte, so 8,000 bytes is always
# under ARCH §12's 8,000-token maximum, usually by about four times.
MAX_INPUT_BYTES = 8000
MAX_INPUT_ITEMS = 8

# Fail-closed CLI controls, rather than an open-ended denylist. Safe mode turns
# off settings/customizations (including hooks, skills and MCP servers), strict
# MCP mode prevents ambient server configuration, and the empty tool set covers
# future built-ins without RepoHone having to know their names.
ISOLATION_ARGS = (
    "--safe-mode", "--restricted", "--strict-mcp-config",
    "--tools", "", "--permission-prompts", "none",
    "--no-session-persistence", "--disable-slash-commands", "--no-chrome",
)
MIN_ISOLATED_CLI_VERSION = (2, 1, 267)


class Unavailable(RuntimeError):
    """The reasoning surface could not be reached. Evidence is preserved."""


def measure_bytes(text: str) -> int:
    """What the payload weighs, exactly, with no tokenizer to be wrong about.

    Counting characters/4 badly undercounts code, random strings and non-ASCII
    text, and the provider's tokenizer is not available locally. Bytes cost
    capacity — this is roughly a quarter of the contracted budget — and buy a
    ceiling that cannot be exceeded by surprise.
    """
    return max(1, len(text.encode("utf-8")))


class Interrupted(RuntimeError):
    """Ended before a valid result. Partial output is never a proposal."""


def effective_policy(project: str, local: Optional[str]) -> str:
    """ARCH §11 — a local setting may only make the project policy stricter."""
    if local is None:
        return project
    return min((project, local), key=lambda p: _RESTRICTIVENESS.get(p, 0))


def local_policy() -> Tuple[Optional[str], Optional[str]]:
    """``(policy, problem)``. ``None`` means the developer set nothing.

    A present-but-unreadable restriction is not absence: it was written down to
    be obeyed, so a damaged one disables reasoning instead of quietly reverting
    to whatever the project allows.
    """
    path = paths.data_dir() / "settings.json"
    try:
        text = paths.read_regular(path)
        if text is None:
            return None, None
        document = json.loads(text)
        if not isinstance(document, dict):
            raise ValueError("settings.json is not a JSON object")
        value = document.get("reasoning_egress")
    except (ValueError, OSError) as exc:
        return DISABLED, (f"{path} is unreadable ({exc}); reasoning is disabled "
                          f"until it is repaired or removed")
    if value is None:
        return None, None
    if value not in _RESTRICTIVENESS:
        allowed = ", ".join(sorted(_RESTRICTIVENESS))
        return DISABLED, (f"{path} sets reasoning_egress to {value!r}, which is not "
                          f"one of {allowed}; reasoning is disabled until it is fixed")
    return value, None


def local_model() -> Tuple[Optional[str], Optional[str]]:
    """``(model, problem)`` from the developer's own settings; ``(None, None)``
    when they set none. A damaged setting stops analysis rather than quietly
    using a model they did not choose."""
    path = paths.data_dir() / "settings.json"
    try:
        text = paths.read_regular(path)
        document = json.loads(text) if text is not None else {}
        if not isinstance(document, dict):
            raise ValueError("settings.json is not a JSON object")
    except (ValueError, OSError) as exc:
        return None, f"{path} is unreadable ({exc}); fix or remove it"
    value = document.get("reasoning_model")
    if value is None:
        return None, None
    if not (isinstance(value, str) and MODEL_NAME.fullmatch(value)):
        return None, f"{path} sets reasoning_model to {value!r}, which is not a model name"
    return value, None


def choose_model(requested: Optional[str], local: Optional[str],
                 project: Optional[str]) -> Tuple[str, str]:
    """``(model, where it came from)``: this command, then the developer, then the
    project, then the default."""
    for value, source in ((requested, "this command"), (local, "your settings"),
                          (project, "the project profile")):
        if value:
            return value, source
    return DEFAULT_MODEL, "the default"


@dataclass
class Reply:
    text: str
    model: Optional[str]
    isolated: bool
    denials: int = 0


class ClaudeCliEngine:
    """The developer's already-configured Claude, run headless with every tool
    disabled and a working directory outside the repository, so the only input
    it can reach is the evidence payload (ARCH §11, §12)."""

    name = "claude-cli"

    def __init__(self, model: Optional[str] = None, timeout: float = 120.0):
        self.model = model
        self.timeout = timeout

    def _isolated_cwd(self) -> Path:
        directory = paths.data_dir() / "reasoning"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _check_isolation_capability(self) -> None:
        """Refuse CLIs older than the version whose isolation flags we verified."""
        try:
            checked = subprocess.run(
                ["claude", "--version"], cwd=str(self._isolated_cwd()),
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise Unavailable(f"cannot verify claude CLI isolation support: {exc}") from exc
        match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", checked.stdout)
        if checked.returncode != 0 or match is None:
            raise Unavailable("cannot verify claude CLI isolation support from --version")
        version = tuple(int(part) for part in match.groups())
        if version < MIN_ISOLATED_CLI_VERSION:
            required = ".".join(map(str, MIN_ISOLATED_CLI_VERSION))
            found = ".".join(map(str, version))
            raise Unavailable(
                f"claude CLI {found} predates verified isolation support {required}")

    def run(self, prompt: str) -> Reply:
        """The prompt goes in on stdin, never on argv: a command line is visible
        to every user on the machine through the process table, which would
        undo the whole minimization pipeline (§12)."""
        size = measure_bytes(prompt)
        if size > MAX_INPUT_BYTES:
            raise Unavailable(
                f"model input is {size} bytes; the local limit is "
                f"{MAX_INPUT_BYTES}, so nothing was sent")
        self._check_isolation_capability()
        args = ["claude", "-p", "--output-format", "json", *ISOLATION_ARGS]
        if self.model:
            args += ["--model", self.model]
        try:
            run_env, token = processes.tracked_env(dict(os.environ))
        except processes.TrackingUnavailable as exc:
            raise Unavailable(str(exc)) from exc
        try:
            # Its own process group: the CLI spawns a child of its own, and
            # killing only the CLI leaves that child still talking to the
            # provider — egress continuing after we reported it stopped.
            proc = subprocess.Popen(
                args, cwd=str(self._isolated_cwd()), env=run_env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True)
        except FileNotFoundError as exc:
            raise Unavailable(f"claude CLI not found: {exc}") from exc
        try:
            stdout, stderr = proc.communicate(prompt, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                processes.terminate(proc, token)
            except processes.TrackingUnavailable as cleanup:
                raise Interrupted(str(cleanup)) from cleanup
            raise Interrupted(f"reasoning exceeded {self.timeout}s") from exc
        except BaseException:
            try:
                processes.terminate(proc, token)
            except processes.TrackingUnavailable as cleanup:
                raise Interrupted(str(cleanup)) from cleanup
            raise
        try:
            # A CLI may start an independently managed helper. The inherited
            # marker lets cleanup find it after the direct CLI has exited.
            processes.terminate(proc, token)
        except processes.TrackingUnavailable as exc:
            raise Interrupted(str(exc)) from exc

        if proc.returncode != 0:
            raise Unavailable(f"claude exited {proc.returncode}: {stderr.strip()[:300]}")
        try:
            envelope = json.loads(stdout)
        except ValueError as exc:
            raise Unavailable(f"unparseable engine envelope: {exc}") from exc
        if envelope.get("is_error"):
            raise Unavailable(f"engine reported an error: {envelope.get('subtype')}")

        denials = len(envelope.get("permission_denials") or [])
        if denials:
            raise Unavailable(
                f"reasoning attempted {denials} disabled capability request(s)")
        return Reply(text=envelope.get("result") or "",
                     model=answering_model(envelope.get("modelUsage")), isolated=True,
                     denials=0)


def answering_model(usage) -> Optional[str]:
    """`modelUsage` also lists background calls, and one of them comes first; the
    model that wrote the answer is the one that wrote the most."""
    if not isinstance(usage, dict) or not usage:
        return None

    def written(item):
        tokens = item[1].get("outputTokens") if isinstance(item[1], dict) else None
        return tokens if isinstance(tokens, (int, float)) else -1

    return max(usage.items(), key=written)[0]


def extract_json(text: str) -> Optional[dict]:
    """Models fence their JSON. Nothing is inferred: if no object parses, the
    job is INVALID_OUTPUT rather than a guess at the missing fields."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1]
        if candidate.lstrip().lower().startswith("json"):
            candidate = candidate.lstrip()[4:]
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
