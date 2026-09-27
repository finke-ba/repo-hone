"""Claude Code adapter (ARCH §32). Verified against 2.1.267.

Per-adapter work is an event-name map, a payload field map and an install
target. Claude fires no event for an interrupt, so `turn.interrupted` is never
produced here; capture derives it at the next snapshot.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import paths
from ..capture import Event

NAME = "claude-code"

EVENTS = {
    "SessionStart": "session.start",
    "UserPromptSubmit": "turn.prompt",
    "Stop": "turn.stop",
    "StopFailure": "turn.failed",
    "SessionEnd": "session.end",
    "PostModelSwitch": "model.switch",
    "PreToolUse": "turn.tool",
    "PostToolUse": "turn.tool_result",
    "SubagentStart": "subagent.start",
    "SubagentStop": "subagent.stop",
}

HOOK_TIMEOUTS = {"SessionStart": 15, "UserPromptSubmit": 30, "Stop": 30,
                 "StopFailure": 30, "PostModelSwitch": 10, "SessionEnd": 30,
                 "PreToolUse": 5, "PostToolUse": 5,   # per tool call; must stay cheap
                 "SubagentStart": 5, "SubagentStop": 10}


# Verified on 2.1.267: PostToolUse fires only when a tool call succeeds. A
# failed, blocked or interrupted call produces no result event at all, so an
# absent outcome on a closed turn means the call did not succeed.
POST_TOOL_USE_ONLY_ON_SUCCESS = True


def _effort(payload: Dict[str, Any]) -> Optional[str]:
    """2.1.267 reports {"level": "high"}; older builds reported a bare string."""
    value = payload.get("effort")
    if isinstance(value, dict):
        value = value.get("level")
    return value if isinstance(value, str) else None


def _error_class(payload: Dict[str, Any]) -> Optional[str]:
    err = payload.get("error")
    if isinstance(err, dict):
        return err.get("type") or err.get("class") or err.get("message")
    return err


def to_event(payload: Dict[str, Any]) -> Optional[Event]:
    hook = payload.get("hook_event_name")
    kind = EVENTS.get(hook) if isinstance(hook, str) else None
    if kind is None:
        return None
    return Event(
        kind=kind,
        host=NAME,
        host_session_id=str(payload.get("session_id") or ""),
        cwd=payload.get("cwd") or ".",
        logical_turn_id=payload.get("prompt_id"),
        text=payload.get("prompt"),
        error=_error_class(payload),
        effort=_effort(payload),
        model=payload.get("model") or payload.get("to_model"),
        permission_mode=payload.get("permission_mode"),
        host_continuation=payload.get("stop_hook_active"),
        agent_version=payload.get("version"),
        end_reason=payload.get("reason"),
        source=payload.get("source"),
        transcript_path=payload.get("transcript_path"),
        last_assistant_message=payload.get("last_assistant_message"),
        tool_name=payload.get("tool_name"),
        tool_use_id=payload.get("tool_use_id"),
        tool_input=payload.get("tool_input"),
        tool_response=payload.get("tool_response"),
        duration_ms=payload.get("duration_ms"),
        agent_id=payload.get("agent_id"),
        agent_type=payload.get("agent_type"),
    )


ENTRY_NAMES = ("repohone_hook.py", "repohone.hook", "repohone-hook")

def invoking_session() -> Optional[str]:
    """The session a command runs in. Claude Code sets this for Bash, PowerShell
    and hooks, matching the hooks' session_id, and updates it on /clear."""
    return (os.environ.get("CLAUDE_CODE_SESSION_ID") or "").strip() or None


def owns_hook(entry: Dict[str, Any]) -> bool:
    """Matched on the entry, not the interpreter: uninstalling must still
    recognise its own hooks after the Python that installed them has moved."""
    if not isinstance(entry, dict):
        return False
    fields = [entry.get("command")] + list(entry.get("args") or [])
    return any(isinstance(f, str) and f.endswith(ENTRY_NAMES) for f in fields)


# Python exits 2 when it cannot open a script, and Claude Code treats exit 2 as a
# blocking hook: a moved source tree would erase every prompt and deny every tool
# call. The hook imports RepoHone itself and exits 0 when it cannot.
HOOK_BOOTSTRAP = (
    "import sys\n"
    "sys.path[:0] = [p for p in sys.argv[2:3] if p]\n"
    "try:\n"
    "    from repohone.cli import main\n"
    "except Exception:\n"
    "    sys.exit(0)\n"
    "sys.exit(main(['hook']))\n")

# Per developer and never committed.
SETTINGS_FILE = ".claude/settings.local.json"
SHARED_SETTINGS_FILE = ".claude/settings.json"

# A plugin under the user's skills directory loads everywhere unless a settings
# file turns it off, so install turns it off at user scope and init turns it on
# for one checkout (verified on 2.1.267).
PLUGIN = "repohone"
PLUGIN_ID = PLUGIN + "@skills-dir"
SKILL = "repohone-rule"
SKILLS_SOURCE = Path(__file__).resolve().parent.parent / "skills"
# The only skills the agent may start; every other one waits for the developer's
# slash command and costs nothing in the model's context.
AGENT_STARTABLE = (SKILL, "ask")

CLI_BOOTSTRAP = (
    "import sys\n"
    "sys.path[:0] = [p for p in sys.argv[1:2] if p]\n"
    "from repohone.cli import main\n"
    "sys.exit(main(sys.argv[2:]))\n")


def hook_args(source: Optional[str]) -> list:
    """`repohone.hook` is the argument `owns_hook` recognises; `source` is a
    Core checkout to import from, or empty for an installed package."""
    return ["-c", HOOK_BOOTSTRAP, "repohone.hook", source or "", "hook"]


def settings(interpreter: str, source: Optional[str] = None) -> Dict[str, Any]:
    """Exec form: no shell, nothing to quote, no $CLAUDE_PROJECT_DIR to misread."""
    hooks = {}
    for hook, timeout in HOOK_TIMEOUTS.items():
        hooks[hook] = [{"hooks": [{"type": "command", "command": interpreter,
                                   "args": hook_args(source), "timeout": timeout}]}]
    return {"hooks": hooks}


def config_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude").expanduser()


def plugin_dir() -> Path:
    return config_dir() / "skills" / PLUGIN


def skill_sources() -> Dict[str, str]:
    """Each skill the Core ships, by name, before the launcher is filled in."""
    return {d.name: (d / "SKILL.md").read_text(encoding="utf-8")
            for d in sorted(SKILLS_SOURCE.iterdir()) if (d / "SKILL.md").is_file()}


def frontmatter(text: str) -> Dict[str, str]:
    head = text.split("---", 2)[1] if text.startswith("---") else ""
    return {k.strip(): v.strip() for k, v in
            (line.split(":", 1) for line in head.splitlines() if ":" in line)}


def plugin_files(interpreter: str, source: Optional[str], version: str) -> Dict[str, str]:
    """Every file of the plugin, by path. The skills run the same Core as the
    hooks, through the launcher, so the two can never disagree on a version."""
    launcher = shlex.quote(str(plugin_dir() / "bin" / "repohone"))
    files = {
        ".claude-plugin/plugin.json": json.dumps(
            {"name": PLUGIN, "version": version,
             "description": "RepoHone: capture hooks, the approval gate and its chat commands"},
            indent=2) + "\n",
        "hooks/hooks.json": json.dumps(settings(interpreter, source), indent=2) + "\n",
        "bin/repohone": (f"#!/bin/sh\nexec {shlex.quote(interpreter)} -c "
                         f"{shlex.quote(CLI_BOOTSTRAP)} {shlex.quote(source or '')} \"$@\"\n"),
    }
    for name, text in skill_sources().items():
        files[f"skills/{name}/SKILL.md"] = text.replace("{repohone}", launcher)
    return files


def installed_skills() -> Optional[Dict[str, str]]:
    """The plugin's skills as installed, or None when it has no skills directory."""
    listed = paths.entries(plugin_dir() / "skills")
    if listed is None:
        return None
    found = {}
    for directory in listed:
        text = paths.read_regular(directory / "SKILL.md")
        if text is not None:
            found[directory.name] = text
    return found


def owns_plugin(directory: Path) -> bool:
    try:
        manifest_path = directory / ".claude-plugin" / "plugin.json"
        manifest = json.loads(paths.read_regular(manifest_path) or "{}")
    except (ValueError, OSError):
        return False
    return isinstance(manifest, dict) and manifest.get("name") == PLUGIN


def write_plugin(interpreter: str, source: Optional[str], version: str,
                 core: Optional[Path] = None) -> Path:
    """`core`, a package directory, is copied into the plugin and imported from
    there: a source tree is also a working tree, and hooks must not follow its
    edits between turns (§4)."""
    root = plugin_dir()
    if paths.present(root) and not owns_plugin(root):
        raise ValueError(f"{root} exists and is not RepoHone's plugin; left alone")
    if core is not None:
        source = str(_vendor(core, root / "core"))
    # A skill an older Core shipped must not outlive it.
    shutil.rmtree(root / "skills", ignore_errors=True)
    for relative, text in plugin_files(interpreter, source, version).items():
        _write(root / relative, text, executable=relative.startswith("bin/"))
    return root


def _vendor(package: Path, target: Path) -> Path:
    staging, previous = target.with_name(".core-staging"), target.with_name(".core-previous")
    for leftover in (staging, previous):
        shutil.rmtree(leftover, ignore_errors=True)
    shutil.copytree(package, staging / package.name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if paths.present(target):
        os.replace(target, previous)
    os.replace(staging, target)
    shutil.rmtree(previous, ignore_errors=True)
    return target


def remove_plugin() -> bool:
    root = plugin_dir()
    if not paths.present(root):
        return False
    if not owns_plugin(root):
        raise ValueError(f"{root} is not RepoHone's plugin; left alone")
    shutil.rmtree(root)
    return True


def plugin_hooks() -> Optional[List[Dict[str, Any]]]:
    """The installed plugin's hook entries, or None when it is absent or unreadable."""
    try:
        doc = json.loads(paths.read_regular(plugin_dir() / "hooks" / "hooks.json") or "null")
    except (ValueError, OSError):
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("hooks"), dict):
        return None
    return [dict(entry, event=event) for event, groups in doc["hooks"].items()
            for group in groups or [] for entry in (group.get("hooks") or [])]


def read_settings(path: Path) -> dict:
    """{} when absent. Raises ValueError for anything but a JSON object."""
    text = paths.read_regular(path)
    doc = json.loads(text) if text and text.strip() else {}
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a JSON object")
    return doc


def set_enabled(path: Path, value: Optional[bool]) -> bool:
    """RepoHone's `enabledPlugins` entry in one settings file, every other key
    kept; None removes it. Returns whether the file changed."""
    doc = read_settings(path)
    plugins = doc.get("enabledPlugins", {})
    if not isinstance(plugins, dict):
        raise ValueError(f"{path}: enabledPlugins is not an object")
    if plugins.get(PLUGIN_ID) == value and (value is not None or PLUGIN_ID not in plugins):
        return False
    if value is None:
        plugins.pop(PLUGIN_ID, None)
    else:
        plugins[PLUGIN_ID] = value
    if plugins:
        doc["enabledPlugins"] = plugins
    else:
        doc.pop("enabledPlugins", None)
    _write(path, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return True


def enabled_here(root) -> Tuple[bool, str]:
    """Whether the plugin loads in this checkout, and which scope decided it:
    local, then project, then user, and on by default."""
    for scope, path in (("local", Path(root) / SETTINGS_FILE),
                        ("project", Path(root) / SHARED_SETTINGS_FILE),
                        ("user", config_dir() / "settings.json")):
        try:
            value = (read_settings(path).get("enabledPlugins") or {}).get(PLUGIN_ID)
        except (ValueError, OSError, AttributeError):
            continue
        if isinstance(value, bool):
            return value, scope
    return True, "default"


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(dir=str(path.parent), prefix=".rh-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(staging, 0o755 if executable else 0o644)
        os.replace(staging, path)
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


def legacy_install(root) -> List[str]:
    """What the per-repository installer of 1.22 and earlier left in this tree."""
    found = []
    for name in (SETTINGS_FILE, SHARED_SETTINGS_FILE):
        count = _owned_hooks(Path(root) / name)
        if count:
            found.append(f"{count} RepoHone hook(s) in {name}")
    try:
        servers = read_settings(Path(root) / ".mcp.json").get("mcpServers") or {}
    except (ValueError, OSError):
        servers = {}
    if isinstance(servers, dict) and PLUGIN in servers:
        found.append("the RepoHone server in .mcp.json")
    if paths.present(Path(root) / ".claude" / "skills" / SKILL):
        found.append(f"the skill copy in .claude/skills/{SKILL}")
    return found


def _owned_hooks(path: Path) -> int:
    try:
        hooks = read_settings(path).get("hooks")
    except (ValueError, OSError):
        return 0
    if not isinstance(hooks, dict):
        return 0
    return sum(1 for groups in hooks.values() if isinstance(groups, list)
               for group in groups if isinstance(group, dict)
               for entry in (group.get("hooks") or []) if owns_hook(entry))
