"""RepoHone CLI and hook entry point.

The hook path is bound by one rule (ARCH §9.2): it exits 0 and writes nothing to
stdout unless it has an announcer payload or a gate decision. A telemetry failure
must never look like a project-code failure, so every error is swallowed into
errors.log.

Exit codes: 0 done, 1 failed, 2 not applicable here, 3 needs approval, 4 conflict.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

from . import (
    CONTRACT_VERSION,
    CORE_VERSION,
    announcer,
    approvals,
    artifacts,
    bootstrap,
    capture,
    deadline,
    diagnosis,
    doctor,
    fingerprint,
    gate,
    gitcmd,
    identity,
    mechanisms,
    paths,
    probe,
    profile,
    proposal,
    reasoning,
    record,
    redact,
    rule,
    selection,
    snapshot,
    state,
    targets,
    toolinput,
)
from .adapters import claude

ERROR_LOG_MAX_BYTES = 1 << 20


def _log_error(message: str) -> None:
    """Rotates at 1 MiB: a hook that fails every turn must not fill the disk."""
    try:
        path = paths.error_log()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > ERROR_LOG_MAX_BYTES:
            path.replace(path.with_suffix(".log.1"))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{record.now()} {message}\n")
    except OSError:
        pass


def cmd_hook(args) -> int:
    """Total: any failure is logged and exits 0 with no stdout."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except (ValueError, OSError) as exc:
        _log_error(f"hook: unreadable input: {exc}")
        return 0

    hook = payload.get("hook_event_name") if isinstance(payload, dict) else None
    try:
        timeout = claude.HOOK_TIMEOUTS.get(hook or "", 30)
        with deadline.budget(timeout):
            if hook == "SessionStart":
                _announce(payload)
            if hook in ("UserPromptSubmit", "SessionEnd"):
                _void_approvals(payload)
            event = claude.to_event(payload)
            if event is not None:
                capture.handle(event)
    except Exception:
        _log_error("hook:\n" + traceback.format_exc())
    if hook == "PreToolUse":
        _gate(payload)
    return 0


def _gate(payload) -> None:
    """Runs after capture, in every consent state: `init` is approved exactly when
    capture is off. The approval is recorded only after the decision is written, so
    a hook killed on the way leaves an ask with nothing to consume, never the reverse."""
    try:
        tool_input = payload.get("tool_input")
        if payload.get("tool_name") != "Bash" or not isinstance(tool_input, dict):
            return
        decision, asked = gate.for_command(str(tool_input.get("command") or ""))
        if decision is None:
            return
        sys.stdout.write(gate.hook_output(decision))
        sys.stdout.flush()
        session = str(payload.get("session_id") or "")
        if session and decision[0] == gate.ASK:
            for argv in asked:
                approvals.mint(session, argv)
    except Exception:
        _log_error("gate:\n" + traceback.format_exc())


def _void_approvals(payload) -> None:
    """A new prompt means an unused approval belongs to a request that is over."""
    session = str(payload.get("session_id") or "")
    if session:
        approvals.void(session)


def _announce(payload) -> None:
    try:
        text = announcer.message(payload.get("cwd") or ".")
    except Exception:
        _log_error("announcer:\n" + traceback.format_exc())
        return
    if not text:
        return
    sys.stdout.write(json.dumps({"hookSpecificOutput": {
        "hookEventName": "SessionStart", "additionalContext": text}}))


def cmd_init(args) -> int:
    """Per checkout (§5): the project's profile, this developer's acceptance of
    it, and the plugin turned on here. A profile already committed is accepted,
    not overwritten."""
    root = gitcmd.toplevel(args.path or ".")
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return 2
    target = root / profile.PROFILE_PATH
    created = args.force or not target.exists()
    if created:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(profile.template(), encoding="utf-8")
    prof = profile.load(root)
    if prof.state != profile.ACTIVE:
        print(f"{target} is {prof.state}: {'; '.join(prof.problems)}; nothing was "
              f"accepted", file=sys.stderr)
        return 1
    local = root / claude.SETTINGS_FILE
    try:
        profile.accept(root, prof)
        claude.set_enabled(local, True)
    except (ValueError, OSError) as exc:
        print(f"{exc}; RepoHone is not on here", file=sys.stderr)
        return 1
    checkout = identity.checkout_id(root)
    if checkout is None:
        print("no checkout id could be assigned; RepoHone is not on here", file=sys.stderr)
        return 1
    state.initialize(checkout)
    migrated = claude.legacy_install(root)
    _remove_owned_hooks(root)
    _remove_owned_integration(root)

    print(f"{'initialized' if created else 'accepted'} {target}")
    print(f"  profile   content {prof.content_policy} · tool capture "
          f"{'on' if prof.tool_capture else 'off'} · reasoning {prof.reasoning_egress}")
    print(f"  consent   yours, for this clone (profile {prof.digest[:12]})")
    print(f"  plugin    on here in {claude.SETTINGS_FILE} ({_keep_out_of_git(root, local)})")
    print(f"  checkout  {checkout}")
    print(f"  telemetry {paths.checkout_dir(checkout)}")
    for item in migrated:
        print(f"  migrated  removed {item}, left by an older per-repository install")
    if not claude.owns_plugin(claude.plugin_dir()):
        print("  note      RepoHone is not installed on this machine yet: run "
              "`repohone install`", file=sys.stderr)
    if created:
        print("\nCommit .repohone/profile.yaml so the team shares the same configuration.")
    return 0


def cmd_deinit(args) -> int:
    """Turns RepoHone off for this developer in this checkout (§29). The profile
    is the project's, so it stays."""
    root = gitcmd.toplevel(args.path or ".")
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return 2
    checkout = identity.peek_checkout_id(root)
    purge = bool(getattr(args, "purge", False))
    dry_run = bool(getattr(args, "dry_run", False))
    yes = bool(getattr(args, "yes", False))

    elsewhere = _captured_elsewhere(root, checkout) if purge and checkout else []
    if elsewhere:
        print(f"refusing to purge: this telemetry also holds sessions captured in "
              f"{', '.join(elsewhere[:3])}, which still exist{'s' if len(elsewhere) == 1 else ''}"
              f" (a copy of this checkout). Purging here would delete that evidence; "
              f"nothing was changed.", file=sys.stderr)
        return 1
    plan = _deinit_plan(root, checkout, purge)
    print("deinitialization plan:")
    for action in plan:
        print(f"  {action}")
    if dry_run or not yes:
        print("no changes made")
        if not dry_run:
            print("re-run with --yes to apply this plan")
        return 0 if dry_run else 3

    local = root / claude.SETTINGS_FILE
    try:
        claude.set_enabled(local, False)
    except (ValueError, OSError) as exc:
        print(f"{exc}; the plugin setting was left alone", file=sys.stderr)
    else:
        print(f"turned the plugin off here in {claude.SETTINGS_FILE} "
              f"({_keep_out_of_git(root, local)})")
    if profile.withdraw(root):
        print("withdrew your acceptance of the profile")
    purge_failed = False
    if purge and checkout:
        d = paths.checkout_dir(checkout)
        historical = _historical_snapshot_refs(checkout)
        count, failures = snapshot.purge_refs(root, checkout, historical)
        print(f"removed {count} snapshot refs for this checkout, including "
              f"current and historical submodules (other worktrees are untouched)")
        for failure in failures:
            print(f"  could not remove {failure}", file=sys.stderr)
        if failures:
            purge_failed = True
            print(f"kept telemetry {d} so the incomplete purge can be retried",
                  file=sys.stderr)
        elif d.exists():
            try:
                shutil.rmtree(d)
            except OSError as exc:
                purge_failed = True
                print(f"could not remove telemetry {d}: {exc}", file=sys.stderr)
            else:
                print(f"removed telemetry {d}")
    removed = _remove_owned_hooks(root)
    if removed:
        print(f"removed {removed} RepoHone hook(s) from .claude/ settings "
              f"(other hooks kept)")
    for line in _remove_owned_integration(root):
        print(line)
    if not purge and checkout:
        print(f"local telemetry kept at {paths.checkout_dir(checkout)} (use --purge to remove)")
    return 1 if purge_failed else 0


def _captured_elsewhere(root: Path, checkout: str) -> List[str]:
    """Other existing working trees whose sessions are in this checkout's store."""
    here = Path(root).resolve()
    found = set()
    for session_id in record.list_sessions(checkout):
        rec = record.load(checkout, session_id, strict=False) or {}
        where = (rec.get("repository") or {}).get("root")
        if not isinstance(where, str) or not where:
            continue
        other = Path(where)
        if other.resolve() != here and other.is_dir():
            found.add(str(other))
    return sorted(found)


def _deinit_plan(root: Path, checkout: Optional[str], purge: bool) -> list:
    """Describe every owned surface before deinit mutates any of them."""
    actions = [f"turn the plugin off for you here ({claude.SETTINGS_FILE})"]
    prof = profile.load(root)
    if prof.state == profile.ACTIVE and profile.accepted(root, prof) is not None:
        actions.append("withdraw your acceptance of the profile")
    actions += [f"remove {item}" for item in claude.legacy_install(root)]
    target = root / profile.PROFILE_PATH
    if target.exists():
        actions.append(f"keep {target}: it is the project's; delete it with Git "
                       f"if the project drops RepoHone")
    if checkout and purge:
        count = sum(len(refs) for refs in snapshot.refs_by_store(root, checkout).values())
        actions.append(f"remove {count} checkout-owned snapshot ref(s)")
        actions.append(f"remove telemetry {paths.checkout_dir(checkout)}")
    elif checkout:
        actions.append(f"keep telemetry {paths.checkout_dir(checkout)}")
    return actions


def _historical_snapshot_refs(checkout_id: str) -> list:
    """Exact child refs retained before telemetry is removed.

    Older RepoHone versions could follow an invalid .gitmodules path. Purging an
    exact recorded ref only when it still points to the recorded commit cleans
    that evidence without treating an untrusted path as authority to delete a
    whole namespace elsewhere.
    """
    found = []

    def collect(snap):
        if not isinstance(snap, dict):
            return
        if snap.get("path"):
            found.append({key: snap.get(key) for key in ("path", "ref", "commit")})
        for child in snap.get("submodules") or []:
            collect(child)

    for session_id in record.list_sessions(checkout_id):
        rec = record.load(checkout_id, session_id, strict=False)
        if not rec:
            continue
        for turn in rec.get("turns") or []:
            for event in ((turn.get("prompt_events") or [])
                          + (turn.get("stop_events") or [])):
                collect(event.get("snapshot"))
        for snap in rec.get("session_end_snapshots") or []:
            collect(snap)
    return found


def _remove_owned_integration(root):
    """The MCP entry and the skill copy, and only those: another server in the
    project's .mcp.json is not ours to remove."""
    done = []
    path = root / ".mcp.json"
    if path.exists():
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            doc = None
        servers = doc.get("mcpServers") if isinstance(doc, dict) else None
        if isinstance(servers, dict) and servers.pop("repohone", None):
            if not servers:
                doc.pop("mcpServers")
            if doc:
                path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
            else:
                path.unlink()
            done.append("removed the repohone MCP server (other servers kept)")
    skill = root / ".claude" / "skills" / "repohone-rule"
    if skill.is_dir():
        shutil.rmtree(skill, ignore_errors=True)
        done.append("removed the repohone-rule skill")
    return done


def _strip_owned(hooks: dict) -> int:
    """Removes only what install added: a project's own hooks are not ours to touch."""
    removed = 0
    for event in list(hooks):
        original = hooks.get(event)
        if not isinstance(original, list):
            continue
        groups = []
        for group in original:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                groups.append(group)
                continue
            entries = group["hooks"]
            kept = [e for e in entries
                    if not (isinstance(e, dict) and claude.owns_hook(e))]
            removed += len(entries) - len(kept)
            if kept:
                groups.append({**group, "hooks": kept})
        if groups:
            hooks[event] = groups
        else:
            del hooks[event]
    return removed


def _remove_owned_hooks_in(path: Path) -> int:
    """An unreadable settings file is left exactly as it is."""
    if not path.exists():
        return 0
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        print(f"{path} is not readable JSON; hooks left in place", file=sys.stderr)
        return 0
    hooks = doc.get("hooks") if isinstance(doc, dict) else None
    if not isinstance(hooks, dict):
        return 0
    removed = _strip_owned(hooks)
    if removed:
        if not hooks:
            doc.pop("hooks", None)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return removed


def _remove_owned_hooks(root) -> int:
    """Both places install has written hooks: the per-developer file, and the
    shared one older installs used."""
    return sum(_remove_owned_hooks_in(Path(root) / name)
               for name in (claude.SETTINGS_FILE, claude.SHARED_SETTINGS_FILE))


def cmd_status(args) -> int:
    root = gitcmd.toplevel(args.path or ".")
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return 2
    prof = profile.load(root)
    checkout = identity.peek_checkout_id(root)
    consent = profile.accepted(root, prof) if prof.capture_enabled else None
    capture_state = ("disabled" if not prof.capture_enabled else "on" if consent
                     else "paused" if consent is False else "off")
    sessions = len(record.list_sessions(checkout)) if checkout else 0
    if getattr(args, "json", False):
        return _json({"core_version": CORE_VERSION, "contract_version": CONTRACT_VERSION,
                      "repository": str(root), "state": prof.state, "capture": capture_state,
                      "checkout_id": checkout, "sessions": sessions,
                      "problems": list(prof.problems)})
    print(f"core       {CORE_VERSION} (contract {CONTRACT_VERSION})")
    print(f"repository {root}")
    print(f"state      {prof.state}")
    print("capture    " + {"disabled": "disabled", "on": "on for you",
                           "paused": "paused: the profile changed since you accepted it",
                           "off": "off: you have not accepted this profile (`repohone init`)"}
          [capture_state])
    print(f"checkout   {checkout or '-'}")
    if checkout:
        print(f"sessions   {sessions}")
    for p in prof.problems:
        print(f"  problem  {p}")
    return 0


def _json(value) -> int:
    json.dump(value, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def cmd_doctor(args) -> int:
    rows = doctor.run(args.path or ".")
    if getattr(args, "json", False):
        _json({"worst": doctor.worst(rows),
               "checks": [{"check": n, "status": s, "detail": d} for n, s, d in rows]})
    else:
        print(doctor.format_report(rows))
    return {doctor.OK: 0, doctor.WARN: 0, doctor.FAIL: 1}[doctor.worst(rows)]


def cmd_list(args) -> int:
    root = gitcmd.toplevel(args.path or ".")
    checkout = identity.peek_checkout_id(root) if root else None
    rows = []
    for sid in record.list_sessions(checkout) if checkout else []:
        rec = (record.load(checkout, sid, strict=False) if checkout else None) or {}
        life = rec.get("lifecycle", {})
        try:
            record.check_versions(rec)
            problem = None
        except record.UnsupportedVersion as exc:
            problem = str(exc)
        rows.append({"session_id": sid, "turns": len(rec.get("turns", [])),
                     "state": life.get("state"), "started_at": life.get("started_at"),
                     "unreadable": problem})
    if getattr(args, "json", False):
        return _json({"sessions": rows})
    if not checkout:
        print("no telemetry for this checkout")
    for row in rows:
        note = f"  [unreadable: {row['unreadable']}]" if row["unreadable"] else ""
        print(f"{row['session_id']}  turns={row['turns']}  state={row['state']}  "
              f"started={row['started_at']}{note}")
    return 0


def cmd_show(args) -> int:
    if in_agent_shell():
        print("`repohone show` prints a whole session record, prompts included, which "
              "would reach the model provider from here; run it in a terminal",
              file=sys.stderr)
        return 2
    root = gitcmd.toplevel(args.path or ".")
    checkout = identity.peek_checkout_id(root) if root else None
    rec = record.load(checkout, args.session_id, strict=False) if checkout else None
    if rec is None:
        print(f"no record for {args.session_id}", file=sys.stderr)
        return 1
    json.dump(rec, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def cmd_install(args) -> int:
    """Once per machine (§4): the Claude Code plugin, off in every repository
    until `repohone init` turns it on in one. The interpreter is resolved now so
    hooks never silently pick a different Core between turns."""
    interpreter = sys.executable or "python3"
    source, durable = _hook_source(interpreter)
    user = claude.config_dir() / "settings.json"
    try:
        # Off everywhere before the plugin exists, so it never loads unasked.
        claude.set_enabled(user, False)
        where = claude.write_plugin(interpreter, source, CORE_VERSION,
                                    core=None if durable else Path(__file__).resolve().parent)
    except (ValueError, OSError) as exc:
        print(f"{exc}; the plugin was not installed", file=sys.stderr)
        return 1
    print(f"installed {where}")
    print(f"  interpreter {interpreter}")
    print(f"  core        {'installed package' if durable else where / 'core'} ({CORE_VERSION})")
    print(f"  scope       off in every repository ({user}); "
          f"`repohone init` turns it on in one")
    if not durable:
        print("  note        RepoHone is not installed into this interpreter, so the "
              "plugin runs a copy\n              of this source tree; re-run "
              "`repohone install` to pick up changes.")
    return 0


def cmd_uninstall(args) -> int:
    """Once per machine (§30). Visits no repository: a checkout that still turns
    the plugin on loads nothing once it is gone."""
    plugin = claude.plugin_dir()
    user = claude.config_dir() / "settings.json"
    purge = bool(getattr(args, "purge", False))
    plan = []
    if paths.present(plugin):
        plan.append(f"remove the plugin {plugin}")
    try:
        if claude.PLUGIN_ID in (claude.read_settings(user).get("enabledPlugins") or {}):
            plan.append(f"remove RepoHone's entry from {user}")
    except (ValueError, OSError):
        pass
    if purge:
        plan.append(f"remove all local telemetry in {paths.data_dir()}")
    print("uninstall plan:")
    for action in plan or ["nothing installed was found"]:
        print(f"  {action}")
    if args.dry_run or not args.yes:
        print("no changes made")
        if not args.dry_run:
            print("re-run with --yes to apply this plan")
        return 0 if args.dry_run else 3
    try:
        claude.remove_plugin()
        claude.set_enabled(user, None)
    except (ValueError, OSError) as exc:
        print(f"{exc}; uninstall stopped", file=sys.stderr)
        return 1
    if purge and paths.present(paths.data_dir()):
        shutil.rmtree(paths.data_dir())
        print("snapshot refs stay in repositories RepoHone never visits; "
              "`git for-each-ref refs/repohone/` lists them")
    print("uninstalled; the Python package itself is left to the tool that installed it")
    return 0


def _keep_out_of_git(root: Path, path: Path) -> str:
    """The hook names this machine's interpreter; committed, it breaks for everyone else."""
    relative = path.relative_to(root).as_posix()
    if gitcmd.ok(["check-ignore", "-q", relative], root):
        return f"{relative} is ignored"
    common = gitcmd.run(["rev-parse", "--path-format=absolute", "--git-common-dir"],
                        root, check=False)
    if not common:
        return f"{relative} is NOT ignored; keep it out of commits"
    exclude = Path(common) / "info" / "exclude"
    try:
        exclude.parent.mkdir(parents=True, exist_ok=True)
        current = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
        separator = "" if not current or current.endswith("\n") else "\n"
        with open(exclude, "a", encoding="utf-8") as handle:
            handle.write(f"{separator}{relative}\n")
    except OSError as exc:
        return f"{relative} is NOT ignored ({exc}); keep it out of commits"
    return f"{relative} added to .git/info/exclude"


def _hook_source(interpreter: str):
    """(source, durable). None when the interpreter imports an installed Core; a
    source checkout otherwise, which stops working the moment it moves."""
    probe = subprocess.run([interpreter, "-c", "import repohone"],
                           cwd=tempfile.gettempdir(), capture_output=True,
                           env={k: v for k, v in os.environ.items()
                                if k != "PYTHONPATH"})
    if probe.returncode == 0:
        return None, True
    return str(Path(__file__).resolve().parent.parent), False


def cmd_rule(args) -> int:
    """Phase 2. Local only: no model call, and no authority to change anything."""
    context = _analysis_context(args)
    if context is None:
        return 2
    root, _, checkout = context
    try:
        if args.session:
            candidate = rule.mark(root, checkout, args.session, args.statement, args.turn)
        else:
            host_session = claude.invoking_session()
            if host_session is None:
                print("run this in the agent session the rule was stated in, or name "
                      "that session with --session (`repohone list` shows them); "
                      "nothing was recorded", file=sys.stderr)
                return 2
            candidate = rule.mark_from_host(root, checkout, claude.NAME, host_session,
                                            args.statement, args.turn)
    except rule.UnknownSession as exc:
        print(f"{exc}; nothing was recorded", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{candidate['candidate_id']}  marked against session "
          f"{candidate['session_id']}")
    print(f"  statement  {candidate['statement'][:72]}")
    print(f"  turn       {candidate['corrective_turn'] or 'not identified'}")
    print(f"  evidence   {len(candidate['evidence_refs'])} refs resolved locally")
    print("  consent    analysis of this candidate only; applying still needs "
          "a separate approval")
    print(f"\nAnalyse it with `repohone diagnose --candidate "
          f"{candidate['candidate_id']}`.")
    return 0


def cmd_rules(args) -> int:
    context = _analysis_context(args)
    if context is None:
        return 2
    _, _, checkout = context
    found = rule.load_all(checkout)
    if getattr(args, "json", False):
        return _json({"rules": [{key: c.get(key) for key in (
            "candidate_id", "state", "session_id", "corrective_turn", "statement",
            "diagnosis_id", "created_at")} for c in found]})
    if not found:
        print("no rule candidates marked in this checkout")
        return 0
    for candidate in found:
        print(f"{candidate['candidate_id']}  {candidate['state']:<9}"
              f"{candidate['session_id']}  {candidate['statement'][:56]}")
    return 0


def cmd_diagnose(args) -> int:
    """ARCH §11 — an explicit model-provider egress surface, policy-gated."""
    root = gitcmd.toplevel(args.path or ".")
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return 2
    prof = profile.load(root)
    if not prof.capture_enabled:
        print(f"repository is {prof.state}; nothing to diagnose", file=sys.stderr)
        return 2

    local, local_problem = reasoning.local_policy()
    if local_problem:
        print(f"local reasoning policy: {local_problem}", file=sys.stderr)
    policy = reasoning.effective_policy(prof.reasoning_egress, local)
    if policy == reasoning.DISABLED and not getattr(args, "show", False):
        print("retrospective reasoning is disabled by policy; no evidence was sent.",
              file=sys.stderr)
        return 3
    if policy == reasoning.INTERACTIVE and not (args.yes or getattr(args, "show", False)):
        print(f"policy is '{policy}': this sends selected evidence to the model "
              f"provider.\nRe-run with --show to print exactly what would be sent, or "
              f"--yes to approve sending it.", file=sys.stderr)
        return 3

    checkout = identity.peek_checkout_id(root)
    if checkout is None:
        print("no telemetry for this checkout", file=sys.stderr)
        return 2
    chosen = _analysis_model(args, prof)
    if chosen is None:
        return 2

    candidate = None
    session_id, statement, boundary = args.session_id, args.rule, args.turn
    if not session_id and not args.candidate:
        if args.yes and not getattr(args, "show", False):
            return _unnamed("diagnose", "a session or --candidate")
        pick = targets.for_diagnosis(checkout, claude.NAME, claude.invoking_session())
        if not _announce_pick(pick):
            return 2
        if pick.candidate:
            args.candidate = pick.chosen
        else:
            session_id = pick.chosen
    if args.candidate:
        try:
            candidate = rule.load(checkout, args.candidate)
        except artifacts.ArtifactError as exc:
            print(f"cannot diagnose: {exc}", file=sys.stderr)
            return 1
        if candidate is None:
            print(f"no rule candidate {args.candidate}", file=sys.stderr)
            return 1
        session_id = candidate["session_id"]
        statement = candidate["statement"]
        boundary = candidate["corrective_turn"]
    if not session_id:
        print("name a session, or a rule candidate with --candidate", file=sys.stderr)
        return 2

    if getattr(args, "show", False):
        try:
            prepared = diagnosis.prepare(root, checkout, session_id, statement, boundary)
        except (FileNotFoundError, ValueError, record.ArtifactError) as exc:
            print(f"cannot diagnose: {exc}", file=sys.stderr)
            return 1
        verdict = ("would NOT be sent: reasoning_egress is disabled" if policy == reasoning.DISABLED
                   else f"would be sent to {chosen[0]} ({chosen[1]})" if prepared.sendable else
                   "would NOT be sent: it lacks developer intent or agent behaviour")
        if in_agent_shell():
            kinds: dict = {}
            for item in prepared.selection.items:
                kinds[item.kind] = kinds.get(item.kind, 0) + 1
            about = (f"rule {candidate['candidate_id']} in session {session_id}" if candidate
                     else f"session {session_id}")
            print(f"{len(prepared.selection.items)} evidence items about {about} "
                  f"({', '.join(f'{n} {k}' for k, n in kinds.items()) or 'none'}), "
                  f"{prepared.sent_bytes} bytes; {verdict}. Nothing was sent.")
            print("The exact text is not shown here, because the chat would send it too: "
                  "run the same command with --show in a terminal to read it.")
            return 0
        sys.stdout.write(prepared.prompt)
        print(f"\n[{prepared.sent_bytes} bytes, {len(prepared.selection.items)} items; "
              f"{verdict}. Nothing was sent.]", file=sys.stderr)
        return 0

    engine = reasoning.ClaudeCliEngine(model=chosen[0], timeout=args.timeout)
    try:
        result = diagnosis.run(root, checkout, session_id, engine,
                               rule_statement=statement,
                               corrective_turn=boundary)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except record.ArtifactError as exc:
        print(f"cannot diagnose: {exc}", file=sys.stderr)
        return 1

    if candidate is not None:
        rule.link_diagnosis(checkout, candidate["candidate_id"],
                            result["diagnosis_id"],
                            result["outcome"] == reasoning.SUCCESS)
    _print_diagnosis(result)
    return 0


def _unnamed(command: str, what: str) -> int:
    print(f"--yes needs its target named ({what}), so what was shown is what is "
          f"approved; `repohone {command}` without --yes picks one and says which",
          file=sys.stderr)
    return 2


def _announce_pick(pick) -> bool:
    """Says which target a command chose and why, or why it chose none."""
    if pick.chosen is None:
        print(pick.why, file=sys.stderr)
        for choice in pick.choices:
            print(f"  {choice}", file=sys.stderr)
        return False
    print(f"target: {pick.chosen} — {pick.why}", file=sys.stderr)
    if pick.choices:
        print(f"  (others: {', '.join(pick.choices[:5])})", file=sys.stderr)
    return True


def _analysis_model(args, prof):
    """``(model, where it came from)``, or None after saying why not."""
    local, problem = reasoning.local_model()
    if problem:
        print(f"cannot choose the analysis model: {problem}", file=sys.stderr)
        return None
    return reasoning.choose_model(getattr(args, "model", None), local, prof.reasoning_model)


def _print_diagnosis(result: dict) -> None:
    budget = result["budget"]
    print(f"{result['diagnosis_id']}  {result['outcome']}")
    print(f"  evidence   {budget['items_sent']}/{budget['max_items']} items, "
          f"{budget['estimated_tokens']}/{budget['max_tokens']} bytes")
    print(f"  engine     {result['engine']['name']} "
          f"{result['engine']['model'] or '-'} (isolated={result['engine']['isolated']})")
    recurrence = result["recurrence"]
    if result["outcome"] == reasoning.SUCCESS:
        mark = result["fingerprint"]
        seen = ("first time seen" if recurrence["sessions"] <= 1
                else f"seen in {recurrence['sessions']} sessions")
        if recurrence.get("complete") is False:
            seen = (f"seen in at least {recurrence['sessions']} session(s): part of the "
                    f"local history could not be read (`repohone doctor`)")
        print(f"  cause      {result['root_cause']['class']} "
              f"({result['confidence']} confidence)")
        print(f"  pattern    {mark['area']} / {mark['behavior']} — {seen}"
              f"{' (new)' if mark['created'] else ' (existing)'}")
        print(f"             {result['root_cause']['summary']}")
        print(f"  property   {result['required_property']}")
        for link in result.get("history") or []:
            print(f"  history    same problem as {link['commit']} ({link['source']}): "
                  f"{link['statement'][:70]}")
        for label, key in (("for", "evidence_for"), ("against", "evidence_against"),
                           ("risks", "risks")):
            for line in result[key]:
                print(f"  {label:<10} {line}")
    elif result["failure"]:
        print(f"  reason     {result['failure']['reason']}")
    if result["outcome"] == reasoning.SUCCESS:
        if recurrence["sessions"] <= 1 and not recurrence.get("historical"):
            print("\n  One observation is weak grounds for a project-level claim; "
                  "recurrence will strengthen or refute it.")
    print("\nA diagnosis is a hypothesis, not a fact. It names no mechanism: "
          "`repohone propose` selects one.")


def cmd_diagnoses(args) -> int:
    root = gitcmd.toplevel(args.path or ".")
    checkout = identity.peek_checkout_id(root) if root else None
    if getattr(args, "json", False):
        return _json({"diagnoses": _diagnosis_rows(checkout) if checkout else []})
    if not checkout:
        print("no telemetry for this checkout")
        return 0
    for rec in diagnosis.load_all(checkout):
        prop = rec.get("required_property") or "-"
        mark = rec.get("fingerprint") or {}
        seen = (rec.get("recurrence") or {}).get("sessions", 0)
        label = f"{mark.get('area','-')}/{mark.get('behavior','-')}"
        print(f"{rec['diagnosis_id']}  {rec['outcome']:<32} {label[:38]:<40}"
              f"x{seen}  {prop[:40]}")
    return 0


def _diagnosis_rows(checkout: str) -> list:
    """With the links a skill follows: the rule it analysed, the proposals made from it."""
    rules = {c.get("diagnosis_id"): c["candidate_id"] for c in rule.load_all(checkout)
             if c.get("diagnosis_id")}
    made: dict = {}
    for rec in proposal.load_all(checkout):
        made.setdefault(rec.get("diagnosis_id"), []).append(rec["proposal_id"])
    rows = []
    for rec in sorted(diagnosis.load_all(checkout), key=lambda r: r.get("created_at") or "",
                      reverse=True):
        mark = rec.get("fingerprint") or {}
        rows.append({"diagnosis_id": rec["diagnosis_id"], "outcome": rec["outcome"],
                     "session_id": rec.get("session_id"), "created_at": rec.get("created_at"),
                     "candidate_id": rules.get(rec["diagnosis_id"]),
                     "area": mark.get("area"), "behavior": mark.get("behavior"),
                     "sessions": (rec.get("recurrence") or {}).get("sessions", 0),
                     "required_property": rec.get("required_property"),
                     "proposals": made.get(rec["diagnosis_id"], [])})
    return rows


def cmd_patterns(args) -> int:
    """Recurrence is what turns a single correction into a project-level claim."""
    root = gitcmd.toplevel(args.path or ".")
    checkout = identity.peek_checkout_id(root) if root else None
    if not checkout:
        print("no telemetry for this checkout")
        return 0
    repository_id = identity.repository_id(identity.origin_url(root))
    marks = fingerprint.known(checkout, repository_id)
    if marks.problems:
        print("recurrence history is incomplete: " + "; ".join(marks.problems[:5]),
              file=sys.stderr)
        return 2
    if not marks:
        print("no failure patterns recorded yet")
        return 0
    for mark in marks:
        seen = "1 session" if mark.sessions == 1 else f"{mark.sessions} sessions"
        earlier = (f"+ {mark.historical} earlier commit{'s' if mark.historical != 1 else ''}"
                   if mark.historical else "")
        print(f"{mark.fingerprint_id}  {mark.cls:<28} {mark.label():<46} "
              f"{seen:<12} {earlier}".rstrip())
    return 0


def cmd_mechanisms(args) -> int:
    """Phase 4 — what the repository already provides. Runs only when asked
    (ARCHITECTURE §1.1); probing executes project tooling, so it is opt-in."""
    root = gitcmd.toplevel(args.path or ".")
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return 2
    prof = profile.load(root)
    if not prof.capture_enabled:
        print(f"repository is {prof.state}; nothing to discover", file=sys.stderr)
        return 2
    checkout = identity.checkout_id(root)
    if checkout is None:
        print("no checkout id could be assigned", file=sys.stderr)
        return 2
    repository_id = identity.repository_id(identity.origin_url(root))

    if args.cached:
        record_ = mechanisms.load(checkout)
        if record_ is None:
            print("no cached inventory; run without --cached")
            return 1
    else:
        found, errors = mechanisms.discover(root)
        unchanged, snapshot_ref = None, None
        if args.probe:
            targets = _probe_targets(found, args)
            if targets is None:
                return 3
            before = probe.tree_fingerprint(root, deep=True)
            snapshot_ref = "HEAD"
            for mechanism in targets:
                try:
                    result = probe.run(root, checkout, "HEAD", mechanism.argv,
                                       timeout=args.timeout)
                except probe.WorkingTreeTouched as exc:
                    print(f"aborting probe: {exc}", file=sys.stderr)
                    unchanged = False
                    break
                mechanism.probed = result.ran
                mechanism.duration_ms = result.duration_ms
                mechanism.exit_code = result.exit_code
                mechanism.note = result.note
            else:
                unchanged = probe.tree_fingerprint(root, deep=True) == before
        record_ = mechanisms.build_record(checkout, repository_id, found, record.now(),
                                          probed=bool(args.probe),
                                          working_tree_unchanged=unchanged,
                                          snapshot_ref=snapshot_ref, errors=errors)
        mechanisms.save(checkout, record_)
        for problem in errors:
            print(f"discovery error: {problem}", file=sys.stderr)

    _print_mechanisms(record_)
    return 0


def _probe_targets(found, args):
    """Probing runs the project's own code, so the selection is explicit.
    Returns None when the caller has to confirm first."""
    runnable = [m for m in found if m.argv]
    if args.only:
        wanted = set(args.only)
        chosen = [m for m in runnable if m.name in wanted or m.kind in wanted]
        if not chosen:
            print(f"nothing matches --only {', '.join(args.only)}", file=sys.stderr)
            return None
        return chosen
    if not args.yes:
        worst = len(runnable) * args.timeout / 60
        print(f"--probe would execute {len(runnable)} project commands "
              f"(worst case ~{worst:.0f} min, network denied):", file=sys.stderr)
        for mechanism in runnable:
            print(f"    {mechanism.invocation}", file=sys.stderr)
        print("Re-run with --yes to run them all, or --only NAME to pick.",
              file=sys.stderr)
        return None
    return runnable


def _print_mechanisms(record_: dict) -> None:
    items = record_["mechanisms"]
    if not items:
        print("no native mechanisms discovered")
        return
    width = max(len(m["name"]) for m in items)
    current = None
    for mechanism in items:
        if mechanism["tier"] != current:
            current = mechanism["tier"]
            print(f"\n{current}")
        cost = mechanism["cost"]
        timing = ""
        if cost["probed"]:
            timing = (f"  [{cost['duration_ms']}ms exit={cost['exit_code']}]"
                      if cost["duration_ms"] is not None else f"  [{cost['note']}]")
        enforced = mechanism.get("enforced_in") or []
        where = f"  enforced in {'+'.join(enforced)}" if enforced else "  not enforced"
        print(f"  {mechanism['kind']:<18}{mechanism['name'].ljust(width)}  "
              f"{mechanism['invocation'] or '-'}{timing}{where}")
        print(f"  {'':<18}{'':<{width}}  {mechanism['evidence'][0]}")
    probe_state = record_["probe"]
    if probe_state["ran"]:
        print(f"\nprobed against {probe_state['snapshot_ref']} in scratch; "
              f"working tree unchanged: {probe_state['working_tree_unchanged']}")
    else:
        print("\nnot probed — costs are unmeasured. Use --probe to run them in a "
              "detached checkout.")


def _analysis_context(args):
    """Shared setup for the phases that reason. Returns (root, profile, checkout)
    or None after printing why not."""
    root = gitcmd.toplevel(args.path or ".")
    if root is None:
        print("not inside a git repository", file=sys.stderr)
        return None
    prof = profile.load(root)
    if not prof.capture_enabled:
        print(f"repository is {prof.state}; nothing to do", file=sys.stderr)
        return None
    # The profile is ACTIVE, so assigning the checkout identity is authorized
    # (§5); a repository with no sessions yet can still be surveyed or mined.
    checkout = identity.checkout_id(root)
    if checkout is None:
        print("no checkout id could be assigned", file=sys.stderr)
        return None
    return root, prof, checkout


def cmd_propose(args) -> int:
    """Phase 5 — select a mechanism for a diagnosis and validate it against the
    states this project actually produced."""
    context = _analysis_context(args)
    if context is None:
        return 2
    root, prof, checkout = context
    if not args.diagnosis_id:
        if args.yes and not getattr(args, "show", False):
            return _unnamed("propose", "a diagnosis id")
        pick = targets.for_proposal(checkout)
        if not _announce_pick(pick):
            return 2
        args.diagnosis_id = pick.chosen

    local, local_problem = reasoning.local_policy()
    if local_problem:
        print(f"local reasoning policy: {local_problem}", file=sys.stderr)
    policy = reasoning.effective_policy(prof.reasoning_egress, local)
    if policy == reasoning.DISABLED and not getattr(args, "show", False):
        print("retrospective reasoning is disabled by policy; nothing was sent.",
              file=sys.stderr)
        return 3
    if policy == reasoning.INTERACTIVE and not (args.yes or getattr(args, "show", False)):
        print(f"policy is '{policy}': selection sends the diagnosis and up to "
              f"{reasoning.MAX_INPUT_ITEMS} mechanism inventory items to the model "
              f"provider. If the selected mechanism is advice an agent reads, a "
              f"second call sends up to {reasoning.MAX_INPUT_ITEMS} minimized, "
              f"labelled cross-session summaries (request, response, correction) "
              f"for judging.\nRe-run with --show to print the selection request, or "
              f"--yes to approve both possible calls.",
              file=sys.stderr)
        return 3

    try:
        found = [d for d in diagnosis.load_all(checkout)
                 if d["diagnosis_id"] == args.diagnosis_id]
        inventory = mechanisms.load(checkout)
    except artifacts.ArtifactError as exc:
        print(f"cannot propose: {exc}", file=sys.stderr)
        return 1
    if not found:
        print(f"no diagnosis {args.diagnosis_id}", file=sys.stderr)
        return 1
    dx = found[0]
    chosen = _analysis_model(args, prof)
    if chosen is None:
        return 2

    if inventory is None:
        discovered, errors = mechanisms.discover(root)
        inventory = mechanisms.build_record(
            checkout, identity.repository_id(identity.origin_url(root)),
            discovered, record.now(), errors=errors)
        mechanisms.save(checkout, inventory)

    if getattr(args, "show", False) and in_agent_shell():
        request = selection.build_prompt(dx, inventory)
        print(f"the selection request for diagnosis {dx['diagnosis_id']}: its cause and "
              f"property, and up to {reasoning.MAX_INPUT_ITEMS} of the "
              f"{len(inventory.get('mechanisms') or [])} checks this repository has, "
              f"{len(request.encode('utf-8'))} bytes, for {chosen[0]} ({chosen[1]}). If it "
              f"picks advice an agent reads, a judging request follows. Nothing was sent.")
        print("The exact text is not shown here, because the chat would send it too: "
              "run the same command with --show in a terminal to read it.")
        return 0
    if getattr(args, "show", False):
        sys.stdout.write(selection.build_prompt(dx, inventory))
        print(f"\n[the selection request, for {chosen[0]} ({chosen[1]}); if it picks advice "
              f"an agent reads, a judging request built from its answer follows. Nothing "
              f"was sent.]", file=sys.stderr)
        return 0

    try:
        session = record.load(checkout, dx["session_id"])
    except record.ArtifactError as exc:
        print(f"cannot propose: {exc}", file=sys.stderr)
        return 1

    engine = reasoning.ClaudeCliEngine(model=chosen[0], timeout=args.timeout)
    result = proposal.propose(root, checkout, dx, inventory, session, engine,
                              timeout=args.validate_timeout)
    _print_proposal(result, dx)
    if result["outcome"] == reasoning.SUCCESS:
        print("\n  the exact change you would be approving:")
        _print_patch(root, checkout, result)
    return 0


PATCH_LINES = 200


def _print_patch(root, checkout: str, rec: dict) -> None:
    """The exact text being approved. A list of filenames and byte counts is not
    something a maintainer can consent to."""
    staged = proposal.staged(checkout, rec["proposal_id"])
    if staged is None:
        print("  unavailable: the staged proposal contents are missing or malformed; "
              "run `repohone doctor` and re-run propose")
        return
    shown = 0
    for entry in rec["change"]["files"]:
        target = Path(root) / entry["path"]
        before = (target.read_text(encoding="utf-8").splitlines(keepends=True)
                  if target.is_file() else [])
        after = (staged.get(entry["path"]) or "").splitlines(keepends=True)
        diff = list(difflib.unified_diff(before, after,
                                         fromfile=f"a/{entry['path']}",
                                         tofile=f"b/{entry['path']}"))
        if not diff:
            print(f"  (no textual change in {entry['path']})")
            continue
        removed = sum(1 for line in diff[2:] if line.startswith("-"))
        if removed:
            print(f"  REMOVES {removed} existing line(s) from {entry['path']}")
        for line in diff:
            if shown >= PATCH_LINES:
                print(f"  ... patch truncated at {PATCH_LINES} lines; the full text "
                      f"is staged under {proposal.staged_path(checkout, rec['proposal_id'])}")
                return
            print("  " + line.rstrip("\n"))
            shown += 1


def _print_estimate(estimate: dict) -> None:
    """Said plainly: this was not run. A number that looks like a measurement
    but is a model's opinion is the worst thing to put in front of an approval."""
    print(f"  ESTIMATED  not executed — {estimate['basis']}")
    if not estimate.get("performed"):
        print(f"             {estimate.get('note') or 'no estimate was produced'}")
        return
    print(f"             judged by  {estimate.get('provider')} "
          f"{estimate.get('model') or ''}".rstrip())
    print(f"             would have caught {estimate['caught']} of "
          f"{estimate['of_occurrences']} past occurrence(s)")
    if estimate["of_clean_sessions"]:
        print(f"             misfired on {estimate['misfired']} of "
              f"{estimate['of_clean_sessions']} session(s) diagnosed with a "
              f"different problem\n             — a proxy for clean work, not "
              f"a measured false-positive rate")
    else:
        print("             no clean session was available to judge against, so "
              "how often\n             it would fire on correct work is unknown")
    if estimate.get("note"):
        print(f"             {estimate['note']}")


def _print_story(rec: dict, dx: Optional[dict]) -> None:
    """What happened and why RepoHone thinks so, above what it proposes."""
    cause = (dx or {}).get("root_cause") or {}
    mark = rec.get("fingerprint") or {}
    if rec.get("problem"):
        print(f"  problem    {rec['problem']}")
    if cause.get("class"):
        print(f"  cause      {cause['class']} ({(dx or {}).get('confidence')} confidence)")
    if rec.get("required_property"):
        print(f"  property   {rec['required_property']}")
    if mark:
        seen = mark.get("sessions", 0)
        print(f"  pattern    {mark['area']} / {mark['behavior']} — "
              + ("first time seen" if seen <= 1 else f"seen in {seen} sessions"))
    for label, key in (("for", "evidence_for"), ("against", "evidence_against")):
        for line in (dx or {}).get(key) or []:
            print(f"  {label:<10} {line}")


def _print_proposal(rec: dict, dx: Optional[dict] = None) -> None:
    print(f"{rec['proposal_id']}  {rec['outcome']}")
    _print_story(rec, dx)
    if rec["outcome"] != reasoning.SUCCESS:
        if rec["failure"]:
            print(f"  reason     {rec['failure']['reason']}")
        for note in rec["validation"]["notes"]:
            print(f"  validation {note}")
        return
    mech = rec["mechanism"]
    print(f"  mechanism  {mech['name']} ({mech['kind']}, {mech['tier']})")
    print(f"  why        {mech['why_this']}")
    print(f"  gap        {mech['why_existing_did_not_help']}")
    for alternative in mech["alternatives_considered"]:
        print(f"  considered {alternative}")
    print(f"  change     {rec['change']['summary']}")
    for entry in rec["change"]["files"]:
        print(f"             {entry['action']:<7}{entry['path']} ({entry['bytes']}b)")
    if not mech["deterministic"]:
        print(f"  JUDGEMENT  {mech['name']} judges rather than decides: it can miss "
              f"the mistake\n             and flag correct work. Treat its verdict "
              f"as advice, not a gate.")
    cost = mech.get("cost") or {}
    if cost.get("probed"):
        print(f"  cost       {cost.get('duration_ms')}ms when probed "
              f"(exit {cost.get('exit_code')})")
    elif mech["tier"] != "existing-project":
        note = cost.get("note") or "the mechanism was not run here"
        print(f"  cost       not measured: {note}")
    print(f"  effect     {rec['expected_effect']}")
    for friction in rec["friction"]:
        print(f"  friction   {friction}")
    validation_ = rec["validation"]
    estimate = validation_.get("estimate")
    if estimate:
        _print_estimate(estimate)
    else:
        print(f"  validated  rejects the state the agent produced: "
              f"{validation_['rejects_problematic']}")
        print(f"             accepts the state you accepted:       "
              f"{validation_['accepts_repaired']}")
    for note in validation_["notes"]:
        print(f"             {note}")
    if rec["privacy"]["adds_model_egress"]:
        print(f"  PRIVACY    {rec['privacy']['note']}")
        for field, value in (rec["privacy"].get("egress") or {}).items():
            print(f"             {field.replace('_', ' '):<12}{value}")
    print(f"\n  Nothing has been changed. `repohone apply {rec['proposal_id']}` "
          f"applies it; `repohone rollback {rec['proposal_id']}` undoes it.")


def cmd_proposals(args) -> int:
    root = gitcmd.toplevel(args.path or ".")
    checkout = identity.peek_checkout_id(root) if root else None
    if getattr(args, "json", False):
        found = proposal.load_all(checkout) if checkout else []
        return _json({"proposals": [{
            "proposal_id": rec["proposal_id"], "diagnosis_id": rec.get("diagnosis_id"),
            "state": rec.get("state"), "outcome": rec.get("outcome"),
            "mechanism": (rec.get("mechanism") or {}).get("name"),
            "required_property": rec.get("required_property"),
            "files": [f["path"] for f in (rec.get("change") or {}).get("files") or []],
            "created_at": rec.get("created_at"), "applied_at": rec.get("applied_at")}
            for rec in sorted(found, key=lambda r: r.get("created_at") or "", reverse=True)]})
    if not checkout:
        print("no telemetry for this checkout")
        return 0
    for rec in proposal.load_all(checkout):
        mech = (rec.get("mechanism") or {}).get("name", "-")
        print(f"{rec['proposal_id']}  {rec['state']:<10}{rec['outcome']:<22}"
              f"{mech:<24}{(rec.get('required_property') or '-')[:40]}")
    return 0


def cmd_apply(args) -> int:
    """Approval. Analysis consent is not application consent (ARCHITECTURE §1.2)."""
    context = _analysis_context(args)
    if context is None:
        return 2
    root, _, checkout = context
    if not args.proposal_id:
        if args.yes:
            return _unnamed("apply", "a proposal id")
        pick = targets.for_apply(checkout)
        if not _announce_pick(pick):
            return 2
        args.proposal_id = pick.chosen
    try:
        rec = proposal.load(checkout, args.proposal_id)
    except artifacts.ArtifactError as exc:
        print(f"cannot apply proposal: {exc}", file=sys.stderr)
        return 1
    if rec is None:
        print(f"no proposal {args.proposal_id}", file=sys.stderr)
        return 1
    if (rec.get("outcome") != reasoning.SUCCESS or not rec.get("change")
            or rec.get("state") not in ("candidate", "approved")):
        print(f"proposal is {rec.get('state')}/{rec.get('outcome')}; nothing can be "
              f"applied", file=sys.stderr)
        return 1
    if not args.yes:
        dx = next((d for d in diagnosis.load_all(checkout)
                   if d["diagnosis_id"] == rec["diagnosis_id"]), None)
        _print_story(rec, dx)
        print(f"this will change {len(rec['change']['files'])} file(s) in your "
              f"working tree:")
        mech = rec.get("mechanism") or {}
        if not mech.get("deterministic", True):
            print(f"  JUDGEMENT  {mech.get('name')} judges rather than decides: it "
                  f"can miss the mistake and flag correct work.")
        if rec["validation"].get("estimate"):
            _print_estimate(rec["validation"]["estimate"])
        _print_patch(root, checkout, rec)
        print(f"\nRe-run with --yes to apply: `repohone apply {rec['proposal_id']} --yes`.")
        return 3
    contents = _stored_contents(checkout, rec)
    if contents is None:
        print("the proposed contents are no longer available; re-run propose",
              file=sys.stderr)
        return 1
    try:
        applied = proposal.apply(root, checkout, args.proposal_id, contents)
    except (proposal.NotApplicable, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except proposal.Conflict as exc:
        print(str(exc), file=sys.stderr)
        return 4
    print(f"applied {applied['proposal_id']}; {len(applied['rollback']['files'])} "
          f"file(s) backed up. Undo with `repohone rollback {applied['proposal_id']}`.")
    return 0


def _stored_contents(checkout, rec):
    return proposal.staged(checkout, rec["proposal_id"])


def cmd_rollback(args) -> int:
    """Previews unless --yes, like apply: undoing is also a change to the tree."""
    context = _analysis_context(args)
    if context is None:
        return 2
    root, _, checkout = context
    if not args.proposal_id:
        if args.yes:
            return _unnamed("rollback", "a proposal id")
        pick = targets.for_rollback(checkout)
        if not _announce_pick(pick):
            return 2
        args.proposal_id = pick.chosen
    if not args.yes:
        try:
            plan = proposal.rollback_plan(root, checkout, args.proposal_id)
        except (artifacts.ArtifactError, proposal.NotApplicable, FileNotFoundError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"rolling back {args.proposal_id} would:")
        for line in plan:
            print(f"  {line}")
        print(f"\nRe-run with --yes to roll back: "
              f"`repohone rollback {args.proposal_id} --yes`.")
        return 3
    try:
        rec = proposal.rollback(root, checkout, args.proposal_id)
    except (artifacts.ArtifactError, proposal.NotApplicable, proposal.Conflict,
            FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if rec["state"] == proposal.UNRESOLVED:
        print(f"{rec['proposal_id']} is NOT fully undone.", file=sys.stderr)
        for path in rec.get("unresolved_paths") or []:
            print(f"  still changed: {path}", file=sys.stderr)
        print(rec["rollback"]["instructions"], file=sys.stderr)
        return 4
    print(f"rolled back {rec['proposal_id']}; the repository is as it was")
    return 0


def cmd_bootstrap(args) -> int:
    """Phase 6 — what the project has already written down and already corrected.
    Candidates only; nothing is adopted."""
    context = _analysis_context(args)
    if context is None:
        return 2
    root, _, checkout = context
    candidates, errors, unavailable = bootstrap.collect(
        root, mechanisms.load(checkout), include_remote=args.include_remote)
    rec = bootstrap.build_record(checkout,
                                 identity.repository_id(identity.origin_url(root)),
                                 candidates, record.now(), errors, unavailable)
    bootstrap.save(checkout, rec)

    if not rec["candidates"]:
        print("no historical candidates found")
        _print_missing_sources(unavailable, errors)
        return 0

    by_id = {c["id"]: c for c in rec["candidates"]}
    linked = _linked_history(checkout)
    print(f"for review ({len(rec['review_set'])} of {len(rec['candidates'])}) — "
          f"{rec['ranking_basis']}\n")
    for position, cid in enumerate(rec["review_set"], 1):
        candidate = by_id[cid]
        seen = f" (x{candidate['occurrences']})" if candidate["occurrences"] > 1 else ""
        print(f"  {position}. [{candidate['source']}] "
              f"{candidate['statement'][:88]}{seen}")
        print(f"     {', '.join(candidate['evidence'][:3])}")
        for dx in linked.get(cid, [])[:3]:
            print(f"     seen again in a session: {dx}")
    rest = len(rec["candidates"]) - len(rec["review_set"])
    if rest:
        print(f"\n  {rest} more in the saved record; "
              f"`repohone bootstrap --all` lists them.")
    if args.all:
        current = None
        for candidate in rec["candidates"]:
            if candidate["source"] != current:
                current = candidate["source"]
                print(f"\n{current}")
            seen = f" (x{candidate['occurrences']})" if candidate["occurrences"] > 1 else ""
            print(f"  {candidate['statement'][:96]}{seen}")
    print("\nCandidates only — nothing here is adopted.")
    _print_missing_sources(unavailable, errors)
    return 0


def _linked_history(checkout: str) -> dict:
    """Candidate id -> the diagnoses that judged a captured session the same problem."""
    linked: dict = {}
    try:
        found = diagnosis.load_all(checkout)
    except artifacts.ArtifactError:
        return linked
    for rec in found:
        for link in rec.get("history") or []:
            linked.setdefault(link["candidate_id"], []).append(rec["diagnosis_id"])
    return linked


def _print_missing_sources(unavailable, errors) -> None:
    """An absent source is a stated limit, never a silent gap (plan, phase 6)."""
    if unavailable:
        print("sources not read:")
        for source in unavailable:
            print(f"  {source}")
    for problem in errors:
        print(f"collection error: {problem}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    # No abbreviations: the gate reads `--yes` from the command line as written.
    p = argparse.ArgumentParser(
        prog="repohone",
        description="Turn the corrections you give a coding agent "
                    "into lasting project improvements.",
        allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"repohone {CORE_VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text, json_output=False):
        s = sub.add_parser(name, help=help_text, allow_abbrev=False)
        s.add_argument("--path", default=None, help="repository path (default: cwd)")
        if json_output:
            s.add_argument("--json", action="store_true", help="machine-readable output")
        s.set_defaults(func=fn)
        return s

    add("hook", cmd_hook, "process one host hook event from stdin")
    init = add("init", cmd_init, "turn RepoHone on for you in this checkout (the privacy boundary)")
    init.add_argument("--force", action="store_true")
    de = add("deinit", cmd_deinit, "turn RepoHone off for you in this checkout")
    de.add_argument("--purge", action="store_true", help="also delete local telemetry")
    de.add_argument("--dry-run", action="store_true",
                    help="print the plan without changing anything")
    de.add_argument("--yes", action="store_true", help="confirm and apply the displayed plan")
    add("status", cmd_status, "show repository state", json_output=True)
    add("doctor", cmd_doctor, "diagnose the installation", json_output=True)
    add("install", cmd_install, "install the Claude Code plugin, off in every repository")
    un = add("uninstall", cmd_uninstall, "remove the plugin from this machine")
    un.add_argument("--purge", action="store_true", help="also delete all local telemetry")
    un.add_argument("--dry-run", action="store_true",
                    help="print the plan without changing anything")
    un.add_argument("--yes", action="store_true", help="confirm and apply the displayed plan")
    add("list", cmd_list, "list captured sessions", json_output=True)
    show = add("show", cmd_show, "print one session record")
    show.add_argument("session_id")
    mark = add("rule", cmd_rule,
               "mark a project rule the developer stated (local only, no model call)")
    mark.add_argument("statement")
    mark.add_argument("--session", default=None,
                      help="the session it was stated in (default: the agent session "
                           "running this command)")
    mark.add_argument("--turn", type=int, default=None,
                      help="which turn holds the correction (default: the turn in progress)")
    add("rules", cmd_rules, "list rule candidates marked in this checkout", json_output=True)

    diag = add("diagnose", cmd_diagnose, "diagnose one session: its cause and the missing property")
    diag.add_argument("session_id", nargs="?", default=None)
    diag.add_argument("--candidate", default=None,
                      help="analyse a rule candidate marked with `repohone rule`")
    diag.add_argument("--rule", default=None,
                      help="explicit project-rule statement from the developer")
    diag.add_argument("--turn", type=int, default=None,
                      help="which turn holds the correction, if you know it")
    diag.add_argument("--yes", action="store_true",
                      help="approve egress when policy is 'interactive'")
    diag.add_argument("--show", action="store_true",
                      help="print exactly what would be sent, and send nothing")
    diag.add_argument("--model", default=None,
                      help="model for this run (default: your settings, then the "
                           "profile's reasoning_model, then opus)")
    diag.add_argument("--timeout", type=float, default=120.0)
    add("diagnoses", cmd_diagnoses, "list diagnoses for this checkout", json_output=True)
    add("patterns", cmd_patterns, "list recurring failure patterns and how often they recur")
    mech = add("mechanisms", cmd_mechanisms,
               "discover the checks and tools the repository already has")
    mech.add_argument("--probe", action="store_true",
                      help="run each mechanism in a detached checkout to measure cost")
    mech.add_argument("--cached", action="store_true", help="show the last inventory")
    mech.add_argument("--only", action="append", default=[],
                      help="probe only this mechanism name or kind (repeatable)")
    mech.add_argument("--yes", action="store_true",
                      help="approve probing every discovered mechanism")
    prop = add("propose", cmd_propose, "select and validate an improvement")
    prop.add_argument("diagnosis_id", nargs="?", default=None,
                      help="default: the only successful diagnosis without a proposal")
    prop.add_argument("--yes", action="store_true", help="approve model egress")
    prop.add_argument("--show", action="store_true",
                      help="print the selection request, and send nothing")
    prop.add_argument("--model", default=None,
                      help="model for this run (default: your settings, then the "
                           "profile's reasoning_model, then opus)")
    prop.add_argument("--timeout", type=float, default=120.0)
    prop.add_argument("--validate-timeout", type=float, default=120.0)
    add("proposals", cmd_proposals, "list improvement proposals", json_output=True)
    ap = add("apply", cmd_apply, "apply an approved proposal to the working tree")
    ap.add_argument("proposal_id", nargs="?", default=None,
                    help="default: the only validated proposal not yet applied")
    ap.add_argument("--yes", action="store_true", help="confirm the change")
    rb = add("rollback", cmd_rollback, "undo an applied proposal")
    rb.add_argument("proposal_id", nargs="?", default=None,
                    help="default: the most recently applied proposal")
    rb.add_argument("--yes", action="store_true", help="confirm the rollback")
    boot = add("bootstrap", cmd_bootstrap,
               "mine existing rules, reverts and fix-ups for candidates")
    boot.add_argument("--all", action="store_true",
                      help="list every candidate, not the review set")
    boot.add_argument("--include-remote", action="store_true",
                      help="also read PR review comments and CI failures via `gh`")
    mech.add_argument("--timeout", type=float, default=probe.DEFAULT_TIMEOUT_S)
    return p


# The chat command for each CLI command, where the names differ.
SLASH = {"list": "sessions"}


def in_agent_shell() -> bool:
    """The agent's Bash tool, a skill's `!` line or Claude Code's `!` mode — all
    of which reach the model — and never the developer's own terminal."""
    return claude.invoking_session() is not None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if args.command == "hook" or not in_agent_shell():
        return _run(args)
    root = gitcmd.toplevel(getattr(args, "path", None) or ".")
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _ChatStream(out, root), _ChatStream(err, root)
    try:
        decision = gate.decide(argv)
        session = claude.invoking_session()
        if (decision and decision[0] == gate.ASK
                and not (session and approvals.consume(session, argv))):
            print(f"RepoHone did not see this approved, so nothing was sent or changed. "
                  f"In the chat use /repohone:{SLASH.get(args.command, args.command)}, "
                  f"which asks for your approval; or run the command in a terminal.",
                  file=sys.stderr)
            return 3
        return _run(args)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = out, err


def _run(args) -> int:
    try:
        return args.func(args)
    except artifacts.ArtifactError as exc:
        print(f"cannot read RepoHone artifact: {exc}", file=sys.stderr)
        return 1


class _ChatStream:
    """What a command prints in the agent's shell reaches the model provider (§10)."""

    def __init__(self, stream, root):
        self._stream, self._root = stream, root

    def write(self, text):
        clean = toolinput.scrub_paths(text, self._root, data=paths.data_dir())
        return self._stream.write(redact.redact(clean)[0])

    def __getattr__(self, name):
        return getattr(self._stream, name)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
