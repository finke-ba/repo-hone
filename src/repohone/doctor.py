"""Diagnostics (ARCH §9.2). The primary surface for capture problems."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

from . import (
    CONTRACT_VERSION,
    CORE_VERSION,
    SCHEMA_VERSION,
    artifacts,
    capture,
    fingerprint,
    gitcmd,
    identity,
    paths,
    profile,
    proposal,
    reasoning,
    record,
    snapshot,
    state,
)
from .adapters import claude

OK, WARN, FAIL = "ok", "warn", "fail"


def _check(name: str, status: str, detail: str) -> Tuple[str, str, str]:
    return (name, status, detail)


def run(cwd) -> List[Tuple[str, str, str]]:
    out = [_check("core", OK, f"{CORE_VERSION} · contract {CONTRACT_VERSION} · "
                              f"schema v{SCHEMA_VERSION}"),
           _check("python", OK, sys.version.split()[0]),
           _check("git", OK if shutil.which("git") else FAIL,
                  gitcmd.run(["--version"], ".", check=False) or "not found")]

    root = gitcmd.toplevel(cwd)
    if root is None:
        out.append(_check("repository", FAIL, "not inside a git repository"))
        return out
    out.append(_check("repository", OK, str(root)))

    prof = profile.load(root)
    detail = prof.state if not prof.problems else f"{prof.state}: {'; '.join(prof.problems)}"
    out.append(_check("profile", {profile.ACTIVE: OK, profile.UNINITIALIZED: WARN,
                                  profile.INVALID: FAIL}[prof.state], detail))

    if prof.state == profile.ACTIVE:
        out.append(_check("tool capture", OK if prof.tool_capture else WARN,
                          "enabled — tool names and targets, never file contents"
                          if prof.tool_capture else "disabled by profile"))

    if prof.state == profile.ACTIVE:
        local, local_problem = reasoning.local_policy()
        policy = reasoning.effective_policy(prof.reasoning_egress, local)
        if local_problem:
            out.append(_check("reasoning egress", FAIL, local_problem))
        else:
            note = f"{policy}" + ("" if policy == prof.reasoning_egress
                                  else f" (project {prof.reasoning_egress}, tightened locally)")
            out.append(_check("reasoning egress", WARN if policy != "disabled" else OK,
                              note + " — sends selected evidence to the model provider"
                              if policy != "disabled" else note + " — nothing is sent"))
        model, model_problem = reasoning.local_model()
        if model_problem:
            out.append(_check("analysis model", FAIL, model_problem))
        else:
            chosen, source = reasoning.choose_model(None, model, prof.reasoning_model)
            out.append(_check("analysis model", OK, f"{chosen} ({source}); `--model` "
                                                    f"overrides it for one run"))

    checkout = identity.peek_checkout_id(root)
    original = identity.copy_of(root)
    out.append(_check("checkout", OK if checkout else WARN,
                      checkout or (f"copied from {original}; this checkout gets its own "
                                   f"identity on its first captured event" if original
                                   else "not yet assigned (assigned on first capture)")))

    remote = identity.origin_url(root)
    repo_id = identity.repository_id(remote)
    out.append(_check("repository id", OK if repo_id else WARN,
                      repo_id or "no origin remote; records cannot join across checkouts"))

    if checkout:
        db = paths.state_db(checkout)
        try:
            db_present = paths.present(db)
        except OSError as exc:
            db_present = None
            out.append(_check("state db", FAIL, f"cannot examine {db}: {exc}"))
        if db_present:
            try:
                with state.connect(db) as conn:
                    state.validate_state(conn, checkout)
                    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                    sessions = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
                out.append(_check("state db", OK if mode.lower() == "wal" else WARN,
                                  f"{db} · journal={mode} · {sessions} sessions"))
            except sqlite3.Error as exc:
                out.append(_check("state db", FAIL, str(exc)))
        elif db_present is False:
            out.append(_check("state db", WARN, "not created yet"))
        unsupported = _unsupported_records(checkout)
        if unsupported:
            out.append(_check(
                "records", FAIL,
                f"{len(unsupported)} record(s) are invalid or use unsupported versions "
                f"(e.g. {unsupported[0]}); semantic commands will refuse them"))
        else:
            try:
                count = len(record.list_sessions(checkout))
            except OSError as exc:
                out.append(_check("records", FAIL, f"cannot list records: {exc}"))
            else:
                out.append(_check("records", OK,
                                  f"{count} in {paths.records_dir(checkout)}"))
        leftovers = _interrupted_writes(checkout)
        if leftovers:
            out.append(_check(
                "interrupted writes", WARN,
                f"{len(leftovers)} staging file(s) left by writes that never finished "
                f"(e.g. {leftovers[0]}); nothing reads them and they are safe to delete"))
        counted, unreadable = fingerprint.diagnosis_status(checkout)
        if unreadable:
            out.append(_check(
                "diagnoses", FAIL,
                f"{unreadable[0]}; recurrence is incomplete, so proposals refuse until "
                "the file is repaired or deleted"))
        else:
            out.append(_check("diagnoses", OK,
                              f"{counted} successful diagnosis(es) count toward recurrence"))

    writable, detail = _storage_writable(root, checkout)
    out.append(_check("storage writable", OK if writable else FAIL, detail))

    if checkout:
        live_by_path = snapshot.refs_by_store(root, checkout)
    else:
        refs = gitcmd.run(
            ["for-each-ref", "--format=%(refname)", "refs/repohone/snapshots/"],
            root, check=False)
        live_by_path = {"": {r.strip() for r in refs.splitlines() if r.strip()}}
    count = sum(len(refs) for refs in live_by_path.values())

    # A record naming a ref that no longer resolves means evidence was destroyed,
    # which must not read as a healthy "0 refs".
    try:
        missing = _dangling_refs(checkout, live_by_path, root) if checkout else []
    except OSError as exc:
        out.append(_check("snapshot refs", FAIL,
                          f"cannot be checked: records cannot be listed ({exc})"))
        return out
    if missing:
        out.append(_check("snapshot refs", FAIL,
                          f"{len(missing)} snapshot(s) referenced by records no longer "
                          f"resolve (e.g. {missing[0]}); evidence was deleted or the "
                          f"refs belong to another checkout"))
        return out

    allocated = state.ordinal_count(checkout) if checkout else None
    if count and not allocated:
        out.append(_check("snapshot refs", FAIL,
                          f"{count} refs but no ordinal state: ordinals would restart at 1 and "
                          f"collide. Run `repohone deinit --purge` then re-init."))
    else:
        stores = sum(bool(refs) for refs in live_by_path.values())
        out.append(_check("snapshot refs", OK,
                          f"{count} under refs/repohone/ in {stores} Git store(s)"))

    out.extend(_integration(root, prof))
    legacy = _legacy_install(root)
    if legacy:
        out.append(legacy)
    if checkout:
        out.append(_capture_errors(checkout))
    out.append(_check("file locking", OK, "flock" if record.fcntl is not None
                      else "lock files (no flock on this host)"))
    return out


def _unsupported_records(checkout_id: str) -> List[str]:
    """Inspect every RepoHone JSON artifact, not only session records."""
    base = paths.checkout_dir(checkout_id)
    unsupported: List[str] = []
    proposals = {}
    staged_payloads = {}
    # `rglob` skips a directory it cannot read without a word, which is the one
    # thing an audit must not do.
    unreadable: List[OSError] = []
    files: list = []
    try:
        if not paths.present(base):
            return []
        for directory, _, names in os.walk(base, onerror=unreadable.append):
            files.extend(Path(directory) / name for name in names)
    except OSError as exc:
        return [f"cannot enumerate {base}: {exc}"]
    unsupported.extend(f"{exc.filename}: cannot be read ({exc.strerror})"
                       for exc in unreadable)
    candidates = sorted(p for p in files if p.name.endswith(".json")
                        and not p.name.startswith(artifacts.STAGING_PREFIX))
    quarantined = sorted(p for p in files if ".corrupt-" in p.name)
    unsupported.extend(
        f"{candidate.relative_to(base)}: quarantined historical evidence"
        for candidate in quarantined)
    for candidate in candidates:
        relative = candidate.relative_to(base)
        parts = relative.parts
        # Proposal backups are byte-for-byte project files, not RepoHone JSON
        # artifacts. A backed-up package.json must not make Doctor report a
        # corrupt telemetry store, even when its historical bytes are not JSON.
        if (len(parts) >= 4 and parts[0] == "proposals"
                and parts[2] == "backup"):
            continue
        try:
            value = json.loads(candidate.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            unsupported.append(f"{candidate.name}: unreadable JSON ({exc})")
            continue
        if (len(parts) == 3 and parts[0] == "proposals"
                and parts[2] == "staged.json"):
            if not isinstance(value, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                unsupported.append(f"{relative}: malformed staged proposal contents")
            else:
                staged_payloads[parts[1]] = value
            continue
        kind = _artifact_kind(relative)
        if kind is None:
            unsupported.append(f"{relative}: unexpected JSON artifact")
            continue
        try:
            checked = artifacts.validate(value, kind)
            artifacts.validate_context(
                checked, kind, expected_checkout_id=checkout_id,
                expected_artifact_id=(candidate.stem
                                      if kind in artifacts.ID_FIELDS else None))
            if kind == "session":
                record.validate_context(checked, checkout_id, candidate.stem)
            elif kind == "proposal":
                proposals[candidate.stem] = checked
                if (checked.get("state") in ("candidate", "approved")
                        and (blocker := proposal.application_blocker(checked))):
                    unsupported.append(f"{relative}: {blocker}")
        except artifacts.ArtifactError as exc:
            unsupported.append(f"{relative}: {exc}")
    for proposal_id, rec in proposals.items():
        if rec.get("outcome") != reasoning.SUCCESS \
                or rec.get("state") not in ("candidate", "approved"):
            continue
        payload = staged_payloads.get(proposal_id)
        if payload is None:
            unsupported.append(
                f"proposals/{proposal_id}.json: successful candidate has no staged contents")
            continue
        expected = {entry["path"]: entry["sha256"]
                    for entry in (rec.get("change") or {}).get("files") or []}
        actual = {name: proposal._sha256(contents) for name, contents in payload.items()}
        if actual != expected:
            unsupported.append(
                f"proposals/{proposal_id}/staged.json: contents do not match proposal hashes")
    for proposal_id in sorted(set(staged_payloads) - set(proposals)):
        unsupported.append(
            f"proposals/{proposal_id}/staged.json: staged contents have no proposal record")
    return unsupported[:20]


def _interrupted_writes(checkout_id: str, settled_s: float = 60.0) -> List[str]:
    """Staging files old enough that no write in progress still owns them."""
    base = paths.checkout_dir(checkout_id)
    cutoff = time.time() - settled_s
    found = []
    for directory, _, names in os.walk(base):
        for name in names:
            path = Path(directory) / name
            try:
                if name.startswith(artifacts.STAGING_PREFIX) and path.stat().st_mtime < cutoff:
                    found.append(str(path.relative_to(base)))
            except OSError:
                continue
    return sorted(found)


def _artifact_kind(relative: Path) -> Optional[str]:
    """The checkout store has a finite set of JSON artifacts."""
    parts = relative.parts
    if parts and parts[0] == "records" and len(parts) == 2:
        return "session"
    if parts and parts[0] == "rules" and len(parts) == 2:
        return "rule"
    if parts and parts[0] == "diagnoses" and len(parts) == 2:
        return "diagnosis"
    if parts and parts[0] == "proposals" and len(parts) == 2:
        return "proposal"
    if relative == Path("mechanisms.json"):
        return "mechanisms"
    if relative == Path("bootstrap.json"):
        return "bootstrap"
    return None


def _probe_write(target: Path) -> Optional[str]:
    """Create and remove one empty sentinel in the nearest existing directory."""
    directory = Path(target)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    if not directory.is_dir():
        return f"{target}: no existing directory can hold it"
    name = None
    try:
        fd, name = tempfile.mkstemp(prefix=".repohone-doctor-", dir=str(directory))
        os.close(fd)
        os.unlink(name)
        return None
    except OSError as exc:
        if name:
            try:
                os.unlink(name)
            except OSError:
                pass
        return f"{target}: {exc}"


def _storage_writable(root: Path, checkout_id: Optional[str]) -> Tuple[bool, str]:
    """Capture needs both local records and Git object/ref storage to be writable."""
    data_target = (paths.checkout_dir(checkout_id) if checkout_id
                   else paths.data_dir() / "checkouts")
    common_text = gitcmd.run(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"], root,
        check=False)
    if not common_text:
        return False, "cannot locate Git storage for snapshot objects and refs"
    common = Path(common_text.strip())
    targets = (data_target, common / "objects", common / "refs" / "repohone")
    failures = [problem for target in targets if (problem := _probe_write(target))]
    if failures:
        return False, "; ".join(failures[:3])
    return True, "record storage and Git object/ref storage accept writes"


def _dangling_refs(checkout_id: str, live_by_path, root) -> List[str]:
    missing = []
    resolvable = {path: set(refs) for path, refs in live_by_path.items()}

    def check(snap, where=""):
        """Submodules keep their own ref store, so a snapshot pinned inside one
        is only resolvable there — checking it against the root's refs reported
        deleted submodule evidence as healthy."""
        if not snap:
            return
        path = snap.get("path") or where
        if path not in resolvable:
            listed = gitcmd.run(["for-each-ref", "--format=%(refname)",
                                 "refs/repohone/"], Path(root) / path, check=False)
            resolvable[path] = {r.strip() for r in listed.splitlines() if r.strip()}
        if snap.get("ref") and snap["ref"] not in resolvable[path]:
            missing.append(f"{path + '/' if path else ''}{snap['ref']}")
        for child in snap.get("submodules") or []:
            check(child, child.get("path") or path)

    for session_id in record.list_sessions(checkout_id):
        try:
            # Invalid records are already reported by _unsupported_records.
            # Never walk their untrusted nested values while diagnosing refs.
            rec = record.load(checkout_id, session_id, strict=True)
        except (artifacts.ArtifactError, OSError):
            continue
        if not rec:
            continue
        for turn in rec.get("turns") or []:
            for event in (turn.get("prompt_events") or []) + (turn.get("stop_events") or []):
                check(event.get("snapshot"))
        for snap in rec.get("session_end_snapshots") or []:
            check(snap)
    return missing[:20]


def _integration(root, prof) -> List[tuple]:
    """Each of these looks healthy from the profile alone, and each alone stops
    all capture."""
    if prof.state != profile.ACTIVE:
        return []
    out = []
    entries = claude.plugin_hooks()
    ours = [entry for entry in entries or [] if claude.owns_hook(entry)]
    missing = sorted(set(claude.HOOK_TIMEOUTS) - {entry["event"] for entry in ours})
    broken = _unrunnable(ours)
    if entries is None:
        out.append(_check("plugin", FAIL, "RepoHone is not installed on this machine, "
                                          "so nothing is captured; run `repohone install`"))
    elif missing:
        more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
        out.append(_check("plugin", FAIL,
                          f"{len(missing)} of {len(claude.HOOK_TIMEOUTS)} events are not "
                          f"hooked ({', '.join(missing[:3])}{more}), so that evidence is "
                          f"never captured; re-run `repohone install`"))
    elif broken:
        out.append(_check("plugin", FAIL, f"{'; '.join(broken[:2])} — nothing is being "
                                          f"captured"))
    else:
        out.append(_check("plugin", OK, f"{len(ours)} hooks in {claude.plugin_dir()}"))

    out.append(_chat_commands())

    enabled, scope = claude.enabled_here(root)
    if not enabled:
        out.append(_check("enabled here", FAIL, f"off at {scope} scope, so nothing is "
                                                f"captured; `repohone init` turns it on"))
    elif scope in ("user", "default"):
        out.append(_check("enabled here", WARN,
                          f"on at {scope} scope, which is every repository on this "
                          f"machine; `repohone install` turns that off"))
    else:
        out.append(_check("enabled here", OK, f"on at {scope} scope"))

    consent = profile.accepted(root, prof)
    out.append(_check("consent", OK if consent else FAIL, {
        True: "you accepted this profile",
        False: "the profile changed since you accepted it, so capture is paused; "
               "`repohone init` shows and accepts it",
        None: "you have not accepted this profile, so nothing of yours is captured; "
              "run `repohone init`"}[consent]))
    return out


def _chat_commands() -> tuple:
    """The skills are how RepoHone is used from the chat; an older install lacks them."""
    try:
        installed = claude.installed_skills()
    except OSError as exc:
        return _check("chat commands", FAIL, f"the plugin's skills cannot be read ({exc})")
    shipped = claude.skill_sources()
    missing = sorted(set(shipped) - set(installed or {}))
    if installed is None or missing:
        return _check("chat commands", WARN,
                      f"{len(missing)} of {len(shipped)} are not installed "
                      f"({', '.join(missing[:3])}{'…' if len(missing) > 3 else ''}); "
                      f"re-run `repohone install`")
    startable = [n for n, text in installed.items()
                 if claude.frontmatter(text).get("disable-model-invocation") != "true"]
    spent = sum(len(claude.frontmatter(installed[n]).get("description", "").encode("utf-8"))
                for n in startable)
    return _check("chat commands", OK,
                  f"{len(installed) - len(startable)} slash commands and "
                  f"{len(startable)} skills the agent may start "
                  f"({spent} bytes of their descriptions in every session)")


def _legacy_install(root) -> Optional[tuple]:
    found = claude.legacy_install(root)
    if not found:
        return None
    return _check("older install", WARN,
                  f"{'; '.join(found)} left by a per-repository install from before "
                  f"1.23, which still runs beside the plugin; `repohone init` removes it")


def _unrunnable(entries) -> List[str]:
    """Hooks that cannot start: the settings look healthy and the hook exits 0."""
    problems, probed = [], set()
    for entry in entries:
        command = entry.get("command")
        resolved = shutil.which(command) if command else None
        if command and not resolved:
            if not Path(command).exists():
                problems.append(f"its interpreter {command} does not exist")
                continue
            if not os.access(command, os.X_OK):
                problems.append(f"its interpreter {command} is not executable")
                continue
        args = list(entry.get("args") or [])
        missing = next((a for a in args if a.endswith(".py") and not Path(a).exists()), None)
        if missing:
            problems.append(f"its entry {missing} does not exist")
            continue
        if args[:1] == ["-c"] and len(args) >= 4 and (command, args[3]) not in probed:
            probed.add((command, args[3]))
            if not _imports_core(command, args[3]):
                problems.append(f"{command} cannot import RepoHone"
                                + (f" from {args[3]}" if args[3] else ""))
    return problems


def _imports_core(interpreter: str, source: str) -> bool:
    try:
        return subprocess.run(
            [interpreter, "-c", "import sys; sys.path[:0] = [p for p in sys.argv[1:2] "
             "if p]; import repohone.cli", source],
            cwd=tempfile.gettempdir(), capture_output=True, timeout=30,
            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _capture_errors(checkout_id: str) -> Tuple[str, str, str]:
    """Events capture refused or degraded. They are recorded per session; nothing
    surfaced them, so a record missing a turn looked complete."""
    dropped = state.refusals(checkout_id)
    seen = len(dropped)
    # A tool call lost from a captured turn is a gap. A turn lost is the turn.
    lost = sum(1 for row in dropped if row["kind"] in capture.TURN_SCOPED)
    kept = 0
    example = f"{dropped[0]['kind']}: {dropped[0]['reason']}" if dropped else None
    try:
        listed = record.list_sessions(checkout_id)
    except OSError as exc:
        return _check("capture errors", FAIL, f"records cannot be listed ({exc})")
    for session_id in listed:
        try:
            rec = record.load(checkout_id, session_id, strict=True)
        except (artifacts.ArtifactError, OSError) as exc:
            seen += 1
            example = example or f"record {session_id}: {exc}"
            continue
        for turn in (rec or {}).get("turns") or []:
            kept += len(turn.get("prompt_events") or []) + len(turn.get("stop_events") or [])
        for problem in ((rec or {}).get("capture") or {}).get("errors") or []:
            seen += 1
            lost += problem.get("event") in capture.TURN_SCOPED
            example = example or f"{problem.get('event')}: {problem.get('error')}"
    for error_path in (paths.error_log(), paths.error_log().with_suffix(".log.1")):
        try:
            lines = [line for line in error_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as exc:
            seen += 1
            example = example or f"{error_path.name}: unreadable ({exc})"
            continue
        if lines:
            seen += len(lines)
            example = example or f"{error_path.name}: {lines[-1][:100]}"
    if not seen:
        return _check("capture errors", OK, "none recorded")
    # A checkout that captured nothing looks exactly like a checkout where
    # nobody worked, so the count alone cannot tell them apart.
    detail = (f"{seen} event(s) could not be captured or attributed, against "
              f"{kept} recorded (e.g. {str(example)[:120]})")
    if lost and lost > kept:
        return _check("capture errors", FAIL, detail + "; more turns were lost "
                      "than recorded, so this checkout is not capturing")
    return _check("capture errors", WARN, detail)





def format_report(rows) -> str:
    width = max(len(r[0]) for r in rows)
    mark = {OK: "ok  ", WARN: "warn", FAIL: "FAIL"}
    return "\n".join(f"[{mark[s]}] {n.ljust(width)}  {d}" for n, s, d in rows)


def worst(rows) -> str:
    statuses = {s for _, s, _ in rows}
    return FAIL if FAIL in statuses else (WARN if WARN in statuses else OK)
