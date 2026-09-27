"""Profile parsing and repository state (ARCH §5, §6, §7).

Core fails closed: anything it cannot fully understand makes the repository
INVALID, which suspends capture. An unreadable profile is not permission.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import CORE_VERSION, PROFILE_VERSION, gitcmd, paths

PROFILE_PATH = Path(".repohone") / "profile.yaml"
# In the clone's Git metadata: one developer's consent, never committed.
ACCEPTED = Path("repohone") / "accepted-profiles"

UNINITIALIZED = "UNINITIALIZED"
ACTIVE = "ACTIVE"
INVALID = "INVALID"

KNOWN_KEYS = {"profile_version", "requires_core", "reasoning_egress",
              "content_policy", "tool_capture", "reasoning_model"}
# Which model analyses, not what is captured or sent, so accepting a profile
# does not cover it and changing it pauses no one's capture.
NOT_CONSENTED = {"reasoning_model"}
MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\[\]-]{0,63}")
# `automatic` is not in the beta contract (ARCH §11): reasoning runs only on
# explicit user intent, so a "reason unasked" policy has nothing to describe.
EGRESS = ("disabled", "interactive")
CONTENT = ("redacted", "hash_only")


@dataclass
class Profile:
    state: str
    path: Optional[Path] = None
    values: Dict[str, Any] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def capture_enabled(self) -> bool:
        return self.state == ACTIVE

    @property
    def reasoning_egress(self) -> str:
        return self.values.get("reasoning_egress", "interactive")

    @property
    def content_policy(self) -> str:
        return self.values.get("content_policy", "redacted")

    @property
    def tool_capture(self) -> bool:
        """Which files the agent consulted. Local only; never an egress surface."""
        return bool(self.values.get("tool_capture", True))

    @property
    def reasoning_model(self) -> Optional[str]:
        return self.values.get("reasoning_model")

    @property
    def digest(self) -> str:
        """What acceptance binds to: the configuration, not its comments."""
        consented = {k: v for k, v in self.values.items() if k not in NOT_CONSENTED}
        return hashlib.sha256(json.dumps(consented, sort_keys=True).encode()).hexdigest()


def _parse_yaml(text: str) -> Dict[str, Any]:
    """Parse the one flat mapping RepoHone supports.

    One implementation is deliberate: optional PyYAML previously made consent
    depend on which packages happened to be installed, and both paths silently
    overwrote duplicate privacy keys.
    """
    out: Dict[str, Any] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _without_comment(raw).rstrip()
        if not line.strip():
            continue
        if line[:1].isspace():
            raise ValueError(f"line {lineno}: nested structures are not supported")
        if ":" not in line:
            raise ValueError(f"line {lineno}: not a key: value pair")
        key, _, value = line.partition(":")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            raise ValueError(f"line {lineno}: invalid key {key!r}")
        if key in out:
            raise ValueError(f"line {lineno}: duplicate configuration key {key!r}")
        out[key] = _scalar(value.strip(), lineno)
    return out


def _without_comment(raw: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in ("'", '"'):
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            return raw[:index]
    if quote is not None:
        raise ValueError("unterminated quoted value")
    return raw


def _scalar(value: str, lineno: int) -> Any:
    if not value:
        raise ValueError(f"line {lineno}: value is required")
    if value[:1] in ("'", '"'):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"line {lineno}: invalid quoted value") from exc
        if not isinstance(parsed, str):
            raise ValueError(f"line {lineno}: quoted value must be text")
        return parsed
    if value in ("true", "false"):
        return value == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if value[:1] in "[{&*!|>" or value in ("null", "~"):
        raise ValueError(f"line {lineno}: structured YAML values are not supported")
    return value


def _version_tuple(v: str):
    parts = re.split(r"[.\-+]", v.strip())
    nums = []
    for p in parts:
        if p.isdigit():
            nums.append(int(p))
        else:
            break
    return tuple(nums) or (0,)


def core_satisfies(spec: str, core: str = CORE_VERSION) -> bool:
    """Comma-separated clauses, all of which must hold: '>=0.4,<0.5'."""
    have = _version_tuple(core)
    for clause in spec.split(","):
        clause = clause.strip()
        m = re.fullmatch(r"(>=|<=|==|!=|>|<)?\s*([0-9][0-9A-Za-z.\-+]*)", clause)
        if not m:
            raise ValueError(f"unparseable version clause {clause!r}")
        op, want_s = m.group(1) or "==", m.group(2)
        want = _version_tuple(want_s)
        n = max(len(have), len(want))
        a = have + (0,) * (n - len(have))
        b = want + (0,) * (n - len(want))
        if op == ">=" and not a >= b: return False
        if op == "<=" and not a <= b: return False
        if op == ">" and not a > b: return False
        if op == "<" and not a < b: return False
        if op == "==" and a[:len(want)] != want: return False
        if op == "!=" and a[:len(want)] == want: return False
    return True


def load(repo_root) -> Profile:
    path = Path(repo_root) / PROFILE_PATH
    try:
        text = paths.read_regular(path)
        if text is None:
            return Profile(state=UNINITIALIZED)
        values = _parse_yaml(text)
    except Exception as exc:
        # Not "uninitialized": something is there that says what to capture.
        return Profile(state=INVALID, path=path, problems=[f"unreadable profile: {exc}"])

    problems: List[str] = []
    unknown = sorted(set(values) - KNOWN_KEYS)
    if unknown:
        problems.append(f"unknown configuration: {', '.join(unknown)}")

    version = values.get("profile_version")
    if version is None:
        problems.append("profile_version is required")
    elif isinstance(version, bool) or not isinstance(version, int):
        problems.append(f"profile_version must be an integer; got {version!r}")
    elif version > PROFILE_VERSION:
        problems.append(f"profile_version {version} is newer than this Core understands "
                        f"({PROFILE_VERSION})")
    elif version < PROFILE_VERSION:
        problems.append(f"profile_version {version} needs an explicit migration")

    spec = values.get("requires_core")
    if spec is not None:
        try:
            if not core_satisfies(str(spec)):
                problems.append(f"Core {CORE_VERSION} does not satisfy requires_core {spec!r}")
        except ValueError as exc:
            problems.append(str(exc))

    egress = values.get("reasoning_egress")
    if egress is not None and egress not in EGRESS:
        problems.append(f"reasoning_egress must be one of {', '.join(EGRESS)}; got {egress!r}")

    policy = values.get("content_policy")
    if policy is not None and policy not in CONTENT:
        problems.append(f"content_policy must be one of {', '.join(CONTENT)}; got {policy!r}")

    tools = values.get("tool_capture")
    if tools is not None and not isinstance(tools, bool):
        problems.append(f"tool_capture must be true or false; got {tools!r}")

    model = values.get("reasoning_model")
    if model is not None and not (isinstance(model, str) and MODEL_NAME.fullmatch(model)):
        problems.append(f"reasoning_model must be a model name such as opus or sonnet; "
                        f"got {model!r}")

    if problems:
        return Profile(state=INVALID, path=path, values=values, problems=problems)
    return Profile(state=ACTIVE, path=path, values=values)


def accepted(repo_root, prof: Profile) -> Optional[bool]:
    """True when this developer accepted this configuration in this clone, False
    when only another version of it, None when never (§5)."""
    common = gitcmd.common_dir(repo_root)
    try:
        text = paths.read_regular(common / ACCEPTED) if common else None
    except OSError:
        return None
    if not text:
        return None
    return prof.digest in text.split()


def accept(repo_root, prof: Profile) -> None:
    common = gitcmd.common_dir(repo_root)
    if common is None:
        raise OSError("no Git metadata to record acceptance in")
    target = common / ACCEPTED
    known = (paths.read_regular(target) or "").split()
    if prof.digest in known:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(dir=str(target.parent), prefix=".rh-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("".join(f"{d}\n" for d in known + [prof.digest]))
        os.replace(staging, target)
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


def withdraw(repo_root) -> bool:
    common = gitcmd.common_dir(repo_root)
    if common is None or not paths.present(common / ACCEPTED):
        return False
    (common / ACCEPTED).unlink()
    return True


def template() -> str:
    return (f"profile_version: {PROFILE_VERSION}\n"
            f"requires_core: \">=0.4,<0.5\"\n"
            f"reasoning_egress: interactive\n"
            f"reasoning_model: opus\n"
            f"content_policy: redacted\n"
            f"tool_capture: true\n")
