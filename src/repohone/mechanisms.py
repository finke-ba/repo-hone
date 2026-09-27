"""Native mechanism discovery (Phase 4; ARCH §3, §24, §25).

Answers *what does this repository already provide that could carry a required
property?* — never *what should we add*. Selection is Phase 5.

Two rules govern everything here:

**Observation, not inference.** A mechanism is reported only when its own
configuration is present. Matching a substring is not observing a mechanism: a
comment saying a tool was removed, or a dependency whose name merely contains
another tool's name, must not become evidence that the project uses it.
Precision beats recall — a missed mechanism costs a weaker proposal, an invented
one costs credibility.

**Never a shell string.** Filenames and config values are repository-controlled
content. Every invocation is an argv list so nothing from the repository can be
concatenated into a command line.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from . import CONTRACT_VERSION, artifacts, paths, redact, toolinput

SCHEMA_ID = "repohone.mechanisms/v1"
SCHEMA_VERSION = 1

TEST, LINT, FORMAT, TYPES = "test", "lint", "format", "types"
ARCHITECTURE, BUILD, CI, SCRIPT = "architecture", "build", "ci", "script"
VCS_HOOK, AGENT_HOOK, AGENT_INSTRUCTION, AGENT_SKILL = (
    "vcs-hook", "agent-hook", "agent-instruction", "agent-skill")

CARRIES = {
    TEST: ["deterministic-test"],
    LINT: ["static-rule"],
    FORMAT: ["formatting"],
    TYPES: ["type-constraint"],
    ARCHITECTURE: ["architecture-constraint", "static-rule"],
    BUILD: ["workflow-constraint"],
    CI: ["workflow-constraint"],
    SCRIPT: ["workflow-constraint"],
    VCS_HOOK: ["workflow-constraint"],
    AGENT_HOOK: ["workflow-constraint"],
    AGENT_INSTRUCTION: ["agent-instruction", "context-discovery"],
    AGENT_SKILL: ["agent-instruction", "context-discovery"],
}

NON_DETERMINISTIC = {AGENT_INSTRUCTION, AGENT_SKILL, AGENT_HOOK}

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CONFIG_EXCERPT_CHARS = 1200


@dataclass
class Mechanism:
    kind: str
    name: str
    tier: str
    argv: Optional[Sequence[str]]
    evidence: List[str]
    config_path: Optional[str] = None
    config_excerpt: Optional[str] = None
    enforced_in: List[str] = field(default_factory=list)
    duration_ms: Optional[int] = None
    exit_code: Optional[int] = None
    probed: bool = False
    note: Optional[str] = None

    def __post_init__(self):
        self.argv = list(self.argv) if self.argv else None
        self.enforced_in = list(self.enforced_in or [])

    @property
    def invocation(self) -> Optional[str]:
        return shlex.join(self.argv) if self.argv else None

    @property
    def id(self) -> str:
        key = f"{self.kind}\x00{self.name}\x00{self.invocation or ''}"
        return "mech_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    def as_record(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "name": self.name, "tier": self.tier,
            "argv": self.argv, "invocation": self.invocation,
            "config_path": self.config_path, "config_excerpt": self.config_excerpt,
            "deterministic": self.kind not in NON_DETERMINISTIC,
            "evidence": self.evidence[:6],
            "carries": CARRIES.get(self.kind, []),
            "enforced_in": sorted(set(self.enforced_in)),
            "cost": {"probed": self.probed, "duration_ms": self.duration_ms,
                     "exit_code": self.exit_code, "note": self.note},
        }


def _read(root, name: str, limit: int = 40000) -> Optional[str]:
    path = Path(root) / name
    try:
        return path.read_text(encoding="utf-8")[:limit] if path.is_file() else None
    except OSError:
        return None


def _excerpt(text: Optional[str], marker: Optional[str] = None,
             root=None) -> Optional[str]:
    """Config excerpts are stored AND sent to a model, so they pass the same
    minimization boundary as any other model input (ARCH §12)."""
    if not text:
        return None
    if marker and marker in text:
        start = text.index(marker)
        chunk = text[start:start + CONFIG_EXCERPT_CHARS]
    else:
        chunk = text[:CONFIG_EXCERPT_CHARS]
    cleaned, _ = redact.redact(chunk)
    return toolinput.scrub_paths(cleaned, root) if root is not None else cleaned


# --- python ---------------------------------------------------------------

PY_TOOLS = {
    "pytest": (TEST, ["pytest", "-q"]),
    "ruff": (LINT, ["ruff", "check", "."]),
    "flake8": (LINT, ["flake8"]),
    "pylint": (LINT, ["pylint", "."]),
    "mypy": (TYPES, ["mypy", "."]),
    "black": (FORMAT, ["black", "--check", "."]),
    "importlinter": (ARCHITECTURE, ["lint-imports"]),
}
PY_DEP_ALIASES = {"import-linter": "importlinter", "importlinter": "importlinter"}


def _declared_python_deps(data: dict) -> set:
    """Exact distribution names only: `pytest-benchmark` is not `pytest`."""
    names = set()
    project = data.get("project") or {}
    groups = [project.get("dependencies") or []]
    groups += list((project.get("optional-dependencies") or {}).values())
    groups.append((data.get("dependency-groups") or {}).get("dev") or [])
    for group in groups:
        for entry in group:
            if not isinstance(entry, str):
                continue
            name = re.split(r"[\s\[<>=!~;]", entry.strip(), maxsplit=1)[0].lower()
            if name:
                names.add(PY_DEP_ALIASES.get(name, name))
    return names


def _python(root) -> List[Mechanism]:
    found: List[Mechanism] = []
    raw = _read(root, "pyproject.toml")
    configured, declared = set(), set()

    if raw:
        try:
            data = tomllib.loads(raw)
        except Exception:
            data = {}
        configured = set((data.get("tool") or {}).keys())
        declared = _declared_python_deps(data)

    for tool, (kind, argv) in PY_TOOLS.items():
        if tool in configured:
            found.append(Mechanism(kind, tool, "existing-project", argv,
                                   [f"pyproject.toml configures [tool.{tool}]"],
                                   "pyproject.toml",
                                   _excerpt(raw, f"[tool.{tool}]", root)))
        elif tool in declared:
            found.append(Mechanism(kind, tool, "existing-ecosystem", argv,
                                   [f"pyproject.toml declares {tool} as a dependency, "
                                    f"but configures no [tool.{tool}]"],
                                   "pyproject.toml"))

    cfg = _read(root, "setup.cfg")
    if cfg:
        sections = set(re.findall(r"^\[([A-Za-z0-9_.:-]+)\]", cfg, re.M))
        for section, tool in (("tool:pytest", "pytest"), ("flake8", "flake8"),
                              ("mypy", "mypy")):
            if section in sections and not any(m.name == tool for m in found):
                kind, argv = PY_TOOLS[tool]
                found.append(Mechanism(kind, tool, "existing-project", argv,
                                       [f"setup.cfg defines [{section}]"], "setup.cfg",
                                       _excerpt(cfg, f"[{section}]", root)))

    if not any(m.name == "pytest" for m in found):
        found += _pytest_elsewhere(root)
    if not any(m.kind == TEST for m in found):
        found += _unittest(root)
    return found


def _pytest_elsewhere(root) -> List[Mechanism]:
    """pytest's own configuration files. A `tests/` directory is not one."""
    ini = _read(root, "pytest.ini")
    if ini is not None:
        return [Mechanism(TEST, "pytest", "existing-project", PY_TOOLS["pytest"][1],
                          ["pytest.ini exists"], "pytest.ini", _excerpt(ini, None, root))]
    tox = _read(root, "tox.ini")
    if tox and re.search(r"^\[pytest\]", tox, re.M):
        return [Mechanism(TEST, "pytest", "existing-project", PY_TOOLS["pytest"][1],
                          ["tox.ini defines [pytest]"], "tox.ini",
                          _excerpt(tox, "[pytest]", root))]
    for name in ("conftest.py", "tests/conftest.py"):
        if _read(root, name) is not None:
            return [Mechanism(TEST, "pytest", "existing-project", PY_TOOLS["pytest"][1],
                              [f"{name} exists"], name)]
    return []


def _unittest(root) -> List[Mechanism]:
    """Only when a test module imports unittest: a directory named `tests` says
    nothing about which language or runner lives in it."""
    base = Path(root) / "tests"
    if not base.is_dir():
        return []
    for path in sorted(base.glob("test*.py"))[:20]:
        head = _read(root, f"tests/{path.name}", limit=4000) or ""
        if re.search(r"^\s*(import unittest|from unittest\b)", head, re.M):
            return [Mechanism(TEST, "python unittest", "existing-project",
                              ["python3", "-m", "unittest", "discover", "-s", "tests"],
                              [f"tests/{path.name} imports unittest"], "tests")]
    return []


# --- node -----------------------------------------------------------------

def _node(root) -> List[Mechanism]:
    raw = _read(root, "package.json")
    if not raw:
        return []
    try:
        package = json.loads(raw)
    except ValueError:
        return []
    scripts = package.get("scripts") or {}
    declared = {n.lower() for n in
                set(package.get("devDependencies") or {}) | set(package.get("dependencies") or {})}
    found: List[Mechanism] = []

    for script, kind in (("test", TEST), ("lint", LINT), ("typecheck", TYPES),
                         ("format", FORMAT), ("build", BUILD)):
        if script in scripts:
            found.append(Mechanism(kind, f"npm run {script}", "existing-project",
                                   ["npm", "run", script],
                                   [f"package.json scripts.{script} = "
                                    f"{str(scripts[script])[:80]}"],
                                   "package.json", _excerpt(str(scripts[script]), None, root)))
    for tool, kind in (("eslint", LINT), ("prettier", FORMAT), ("typescript", TYPES),
                       ("jest", TEST), ("vitest", TEST)):
        if tool in declared and not any(m.kind == kind for m in found):
            found.append(Mechanism(kind, tool, "existing-ecosystem", None,
                                   [f"package.json declares {tool} but no matching "
                                    f"script"], "package.json"))
    return found


# --- make, rust, go -------------------------------------------------------

def _make(root) -> List[Mechanism]:
    raw = _read(root, "Makefile")
    if not raw:
        return []
    targets = set(re.findall(r"^([A-Za-z][A-Za-z0-9_.-]*)\s*:(?!=)", raw, re.M))
    interesting = {"test": TEST, "lint": LINT, "check": SCRIPT, "verify": SCRIPT,
                   "typecheck": TYPES, "fmt": FORMAT, "format": FORMAT, "build": BUILD}
    return [Mechanism(kind, f"make {target}", "existing-project", ["make", target],
                      [f"Makefile defines target `{target}`"], "Makefile",
                      _excerpt(raw, f"{target}:", root))
            for target, kind in interesting.items() if target in targets]


def _rust_go(root) -> List[Mechanism]:
    found = []
    if _read(root, "Cargo.toml"):
        found += [Mechanism(TEST, "cargo test", "existing-ecosystem", ["cargo", "test"],
                            ["Cargo.toml exists"], "Cargo.toml"),
                  Mechanism(LINT, "clippy", "existing-ecosystem", ["cargo", "clippy"],
                            ["Cargo.toml exists"], "Cargo.toml")]
    if _read(root, "go.mod"):
        found += [Mechanism(TEST, "go test", "existing-ecosystem",
                            ["go", "test", "./..."], ["go.mod exists"], "go.mod"),
                  Mechanism(LINT, "go vet", "existing-ecosystem",
                            ["go", "vet", "./..."], ["go.mod exists"], "go.mod")]
    return found


# --- ci, hooks, agent config ----------------------------------------------

def _ci(root) -> List[Mechanism]:
    found = []
    workflows = Path(root) / ".github" / "workflows"
    if workflows.is_dir():
        files = sorted(p.name for p in workflows.glob("*.y*ml"))
        if files:
            found.append(Mechanism(
                CI, "GitHub Actions", "existing-project", None,
                [f".github/workflows/ contains {', '.join(files[:4])}"],
                ".github/workflows"))
    for name, label in ((".gitlab-ci.yml", "GitLab CI"),
                        (".circleci/config.yml", "CircleCI"),
                        ("azure-pipelines.yml", "Azure Pipelines")):
        if (Path(root) / name).is_file():
            found.append(Mechanism(CI, label, "existing-project", None,
                                   [f"{name} exists"], name))
    return found


def _vcs_hooks(root) -> List[Mechanism]:
    found = []
    raw = _read(root, ".pre-commit-config.yaml")
    if raw:
        found.append(Mechanism(VCS_HOOK, "pre-commit", "existing-project",
                               ["pre-commit", "run", "--all-files"],
                               [".pre-commit-config.yaml exists"],
                               ".pre-commit-config.yaml", _excerpt(raw, None, root)))
    if (Path(root) / ".husky").is_dir():
        found.append(Mechanism(VCS_HOOK, "husky", "existing-project", None,
                               [".husky/ directory exists"], ".husky"))
    return found


def _agent(root) -> List[Mechanism]:
    """The coding agent's own configuration is a first-class project mechanism:
    §3 prefers a native agent instruction over RepoHone-specific runtime."""
    found = []
    for name in ("CLAUDE.md", "AGENTS.md", ".cursorrules"):
        raw = _read(root, name)
        if raw:
            found.append(Mechanism(AGENT_INSTRUCTION, name, "existing-project", None,
                                   [f"{name} exists"], name, _excerpt(raw, None, root)))
    settings = _read(root, ".claude/settings.json")
    if settings:
        try:
            hooks = sorted((json.loads(settings).get("hooks") or {}).keys())
        except ValueError:
            hooks = []
        if hooks:
            found.append(Mechanism(AGENT_HOOK, "Claude Code hooks", "existing-project",
                                   None, [f".claude/settings.json registers "
                                          f"{', '.join(hooks[:6])}"],
                                   ".claude/settings.json"))
    skills = Path(root) / ".claude" / "skills"
    if skills.is_dir():
        names = sorted(p.name for p in skills.iterdir() if p.is_dir())
        if names:
            found.append(Mechanism(AGENT_SKILL, "Claude skills", "existing-project",
                                   None, [f".claude/skills/ contains "
                                          f"{', '.join(names[:5])}"],
                                   ".claude/skills"))
    return found


def _looks_executable(path: Path) -> bool:
    """A data file named `validate_notes.md` is not a project check."""
    if path.suffix.lower() in (".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".lock"):
        return False
    try:
        if path.stat().st_mode & 0o111:
            return True
        with path.open("rb") as handle:
            return handle.read(2) == b"#!"
    except OSError:
        return False


def _scripts(root) -> List[Mechanism]:
    found = []
    for directory in ("scripts", "bin", "tools"):
        base = Path(root) / directory
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir())[:40]:
            if not path.is_file() or not _looks_executable(path):
                continue
            if not SAFE_NAME.match(path.name):
                # Repository-controlled and not safe to name in a command.
                continue
            if not re.search(r"(check|verify|lint|test|validate)", path.name, re.I):
                continue
            relative = f"{directory}/{path.name}"
            found.append(Mechanism(
                SCRIPT, relative, "project-owned-custom", [f"./{relative}"],
                [f"{relative} is executable and named like a project check"], relative))
    return found


DETECTORS = (_python, _node, _make, _rust_go, _ci, _vcs_hooks, _agent, _scripts)


# --- enforcement linkage --------------------------------------------------

def _without_yaml_comment(line: str) -> str:
    """Remove YAML comments without treating a quoted # as a comment."""
    quote = None
    escaped = False
    for position, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None and (
                position == 0 or line[position - 1].isspace()):
            return line[:position]
    return line


def _yaml_values(text: str, keys, allowed=None) -> List[str]:
    """Extract selected YAML values while retaining their structural parents.

    This deliberately supports only ordinary mapping keys and sequence items.
    Unsupported YAML becomes unknown enforcement. A value is returned only when
    ``allowed(key, ancestors)`` confirms that its location is executable.
    """
    wanted = set(keys)
    pattern = re.compile(
        r"^(?P<key>[A-Za-z0-9_.-]+)[ \t]*:[ \t]*(?P<value>.*)$")
    lines = text.splitlines()
    values: List[str] = []
    stack: List[tuple] = []
    position = 0
    while position < len(lines):
        raw = lines[position]
        visible = _without_yaml_comment(raw)
        if not visible.strip():
            position += 1
            continue
        expanded = visible.expandtabs(8)
        indent = len(expanded) - len(expanded.lstrip(" "))
        while stack and indent <= stack[-1][0]:
            stack.pop()
        content = expanded.lstrip(" ")
        key_indent = indent
        if content.startswith("- "):
            stack.append((indent, None))
            content = content[2:].lstrip(" ")
            key_indent = indent + 2
        match = pattern.match(content)
        if not match:
            position += 1
            continue
        key = match.group("key")
        value = match.group("value").strip()
        ancestors = tuple(name for _level, name in stack if name is not None)
        if key not in wanted:
            if not value:
                stack.append((key_indent, key))
            position += 1
            continue
        permitted = allowed is None or allowed(key, ancestors)
        if value and value not in ("|", ">", "|-", ">-", "|+", ">+"):
            if permitted:
                values.append(value.strip("'\""))
            position += 1
            continue

        base_indent = indent
        block_lines = []
        position += 1
        while position < len(lines):
            child_raw = lines[position]
            child_visible = _without_yaml_comment(child_raw)
            expanded = child_raw.expandtabs(8)
            child_indent = len(expanded) - len(expanded.lstrip(" "))
            if child_visible.strip() and child_indent <= base_indent:
                break
            if child_visible.strip():
                block_lines.append(child_visible)
            position += 1
        if not permitted or not block_lines:
            continue
        if value in ("|", "|-", "|+"):
            commands = [line.strip() for line in block_lines if line.strip()]
        elif value in (">", ">-", ">+"):
            # Folded YAML joins ordinary lines with spaces. Treating them as
            # separate shell commands turns `echo` followed by a tool name into
            # a false execution claim.
            commands = [" ".join(line.strip() for line in block_lines
                                 if line.strip())]
        elif key == "script":
            commands = [line.strip()[2:].strip()
                        if line.strip().startswith("- ") else line.strip()
                        for line in block_lines if line.strip()]
        elif key == "run":
            nested = "\n".join(block_lines)
            commands = _yaml_values(nested, ("command",))
        else:
            commands = []
        if commands:
            values.append("\n".join(commands))
    return values


def _under_steps(key: str, ancestors) -> bool:
    """A workflow command must be a direct property of one steps item."""
    if "steps" not in ancestors:
        return False
    position = len(ancestors) - 1 - tuple(reversed(ancestors)).index("steps")
    return not ancestors[position + 1:]


_GITLAB_TOP_LEVEL = {
    "default", "include", "stages", "variables", "workflow", "image",
    "services", "cache", "before_script", "after_script", "pages",
}


def _workflow_values(path: Path, text: str) -> List[str]:
    portable = path.as_posix()
    if "/.github/workflows/" in portable or portable.startswith(".github/workflows/"):
        return _yaml_values(
            text, ("run",), lambda key, ancestors: key == "run"
            and len(ancestors) >= 3 and ancestors[0] == "jobs"
            and _under_steps(key, ancestors))
    if portable.endswith(".circleci/config.yml"):
        return _yaml_values(
            text, ("run",), lambda key, ancestors: key == "run"
            and len(ancestors) >= 3 and ancestors[0] == "jobs"
            and _under_steps(key, ancestors))
    if portable.endswith("azure-pipelines.yml"):
        return _yaml_values(
            text, ("script",), lambda key, ancestors: key == "script"
            and _under_steps(key, ancestors))
    if portable.endswith(".gitlab-ci.yml"):
        return _yaml_values(
            text, ("script",),
            lambda key, ancestors: key == "script" and len(ancestors) == 1
            and ancestors[0] not in _GITLAB_TOP_LEVEL
            and not ancestors[0].startswith("."))
    return []


def _pre_commit_values(text: str) -> List[str]:
    return _yaml_values(
        text, ("id", "entry"),
        lambda _key, ancestors: bool(ancestors) and ancestors[0] == "repos"
        and "hooks" in ancestors
        and not ancestors[ancestors.index("hooks") + 1:])


def _run_steps(root) -> List[str]:
    """Command lines CI actually executes; comments are never enforcement."""
    steps: List[str] = []
    workflows = Path(root) / ".github" / "workflows"
    files = list(workflows.glob("*.y*ml")) if workflows.is_dir() else []
    for name in (".gitlab-ci.yml", ".circleci/config.yml", "azure-pipelines.yml"):
        candidate = Path(root) / name
        if candidate.is_file():
            files.append(candidate)
    for path in files[:20]:
        try:
            steps += _workflow_values(path, path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return steps


_SHELL_CONTROL = re.compile(r"(^|[;&|]\s*)(if|then|elif|else|fi|case|esac|for|while|until)\b")
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _executed_commands(text: str) -> List[List[str]]:
    """Conservatively identify direct shell command positions.

    Tool names printed as arguments are not execution. Complex conditional
    shell programs become unknown rather than an enforcement claim.
    """
    lines = text.splitlines()
    # Multi-line shell constructs need a real shell parser to distinguish code
    # from data.  In particular, a heredoc body or a continued `echo` argument
    # can begin with the exact spelling of a tool command without executing it.
    # Treat the whole step as unknown when any such construct is present.
    if any(_SHELL_CONTROL.search(line.strip())
           or any(char in line for char in "`(){}")
           or "<<" in line
           or line.rstrip().endswith("\\")
           for line in lines):
        return []
    commands: List[List[str]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = "#"
            tokens = list(lexer)
        except ValueError:
            continue
        segment: List[str] = []
        for token in tokens + [";"]:
            if token and all(char in ";&|" for char in token):
                if segment:
                    while segment and _ASSIGNMENT.match(segment[0]):
                        segment.pop(0)
                    if segment and segment[0] == "env":
                        segment.pop(0)
                        while segment and (segment[0].startswith("-")
                                           or _ASSIGNMENT.match(segment[0])):
                            segment.pop(0)
                    if segment and segment[0] in ("command", "exec"):
                        segment.pop(0)
                    if segment:
                        commands.append(segment)
                segment = []
            else:
                segment.append(token)
    return commands


def _invokes(commands: List[List[str]], mechanism: Mechanism,
             strict: bool = True) -> bool:
    argv = list(mechanism.argv or [])
    if not argv:
        return False
    expected = Path(argv[0]).name
    for command in commands:
        actual = Path(command[0]).name
        if actual != expected:
            continue
        if not strict:
            return True
        signature_length = {"make": 2, "npm": 3, "cargo": 2,
                            "go": 2, "ruff": 2}.get(expected)
        if signature_length is not None and len(argv) >= signature_length:
            if (len(command) >= signature_length
                    and command[:signature_length] == argv[:signature_length]):
                return True
            continue
        if len(argv) > 2 and argv[1] == "-m":
            if len(command) > 2 and command[1:3] == argv[1:3]:
                return True
            continue
        required_flags = [flag for flag in argv[1:]
                          if flag in ("--check", "--check-only", "--no-emit")]
        if all(flag in command[1:] for flag in required_flags):
            return True
    return False


def _recipe(root, target: str) -> str:
    """The commands a Makefile target actually runs."""
    raw = _read(root, "Makefile") or ""
    match = re.search(rf"^{re.escape(target)}\s*:(?!=).*$((?:\n\t.*)*)", raw, re.M)
    return match.group(1) if match else ""


def annotate_enforcement(root, found: List[Mechanism]) -> None:
    """Configured is not enforced. Phase 5 has to answer why an existing
    mechanism did not already help (LEARNING_PLAN §11 pass 2), and it cannot if
    a linter nobody runs looks identical to one CI gates on."""
    steps = [command for step in _run_steps(root)
             for command in _executed_commands(step)]
    pre_commit = [command for value in _pre_commit_values(
        _read(root, ".pre-commit-config.yaml") or "")
                  for command in _executed_commands(value)]
    for mechanism in found:
        if _invokes(steps, mechanism):
            mechanism.enforced_in.append("ci")
        if _invokes(pre_commit, mechanism, strict=False):
            mechanism.enforced_in.append("vcs-hook")

    # One level of indirection: CI commonly runs `make verify`, which runs the
    # tools. Without this a gated linter reads as unenforced, which is exactly
    # the wrong answer to "why didn't the existing mechanism help?".
    for wrapper in [m for m in found if m.enforced_in and m.argv
                    and m.argv[0] == "make"]:
        target_name = wrapper.argv[-1] if wrapper.argv else ""
        body = _recipe(root, target_name)
        # A target may delegate to sibling targets as well as to tools.
        delegated = r"\$\(MAKE\)\s+([A-Za-z0-9_.-]+)|make\s+([A-Za-z0-9_.-]+)"
        for referenced in re.findall(delegated, body):
            body += "\n" + _recipe(root, next(filter(None, referenced), ""))
        for prerequisite in re.findall(rf"^{re.escape(target_name)}\s*:(?!=)\s*(.*)$",
                                       _read(root, "Makefile") or "", re.M):
            for target in prerequisite.split():
                body += "\n" + _recipe(root, target)
        for mechanism in found:
            if mechanism is wrapper or not mechanism.argv:
                continue
            if _invokes(_executed_commands(body), mechanism):
                for place in wrapper.enforced_in:
                    mechanism.enforced_in.append(place)


def discover(root):
    """Returns (mechanisms, errors). A detector that fails is reported, never
    silently read as 'this repository has no such mechanism'."""
    found: List[Mechanism] = []
    errors: List[str] = []
    seen = set()
    for detector in DETECTORS:
        try:
            produced = detector(root)
        except Exception as exc:
            errors.append(f"{detector.__name__}: {type(exc).__name__}: {exc}"[:300])
            continue
        for mechanism in produced:
            if mechanism.id in seen:
                continue
            seen.add(mechanism.id)
            found.append(mechanism)
    try:
        annotate_enforcement(root, found)
    except Exception as exc:
        errors.append(f"annotate_enforcement: {type(exc).__name__}: {exc}"[:300])
    return found, errors


TIER_ORDER = {"existing-project": 0, "existing-ecosystem": 1, "project-owned-custom": 2}


def ranked(found: List[Mechanism]) -> List[Mechanism]:
    """ARCH §3: existing project mechanism, then ecosystem, then project-owned,
    and deterministic before probabilistic within a tier."""
    return sorted(found, key=lambda m: (TIER_ORDER.get(m.tier, 9),
                                        m.kind in NON_DETERMINISTIC, m.kind, m.name))


def build_record(checkout_id: str, repository_id: Optional[str], found: List[Mechanism],
                 discovered_at: str, probed: bool = False,
                 working_tree_unchanged: Optional[bool] = None,
                 snapshot_ref: Optional[str] = None,
                 errors: Optional[List[str]] = None) -> dict:
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "checkout_id": checkout_id,
        "repository_id": repository_id,
        "discovered_at": discovered_at,
        "probe": {"ran": probed, "working_tree_unchanged": working_tree_unchanged,
                  "snapshot_ref": snapshot_ref},
        "discovery_errors": list(errors or []),
        "mechanisms": [m.as_record() for m in ranked(found)],
    }


def path_for(checkout_id: str) -> Path:
    return paths.checkout_dir(checkout_id) / "mechanisms.json"


def save(checkout_id: str, record: dict) -> Path:
    paths.ensure(paths.checkout_dir(checkout_id))
    target = path_for(checkout_id)
    return artifacts.save(target, record, "mechanisms",
                          expected_checkout_id=checkout_id)


def load(checkout_id: str) -> Optional[dict]:
    target = path_for(checkout_id)
    return artifacts.load(target, "mechanisms", expected_checkout_id=checkout_id)
