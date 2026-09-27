"""Phase 5: proposal generation, validation, application and rollback.

Every accepted improvement is an experiment (ARCH §26), so a proposal records
what changed, why, the target failure, the expected effect, how to measure it
and **how to undo it** (§27). Applying is a separate, explicit act: analysis
consent is not application consent (§1.2).
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from . import (
    CONTRACT_VERSION,
    artifacts,
    evidence,
    fingerprint,
    paths,
    probe,
    reasoning,
    record,
    selection,
    validation,
)

_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

SCHEMA_ID = "repohone.proposal/v1"
SCHEMA_VERSION = 1


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_id(diagnosis_id: str, created_at: str) -> str:
    nonce = os.urandom(8).hex()
    digest = hashlib.sha256(f"{diagnosis_id}\x00{created_at}\x00{nonce}"
                            .encode()).hexdigest()
    return "prop_" + digest[:16]


def directory(checkout_id: str) -> Path:
    return paths.checkout_dir(checkout_id) / "proposals"


def path_for(checkout_id: str, proposal_id: str) -> Path:
    artifacts.require_artifact_id("proposal", proposal_id)
    return directory(checkout_id) / f"{proposal_id}.json"


def backup_dir(checkout_id: str, proposal_id: str) -> Path:
    artifacts.require_artifact_id("proposal", proposal_id)
    return directory(checkout_id) / proposal_id / "backup"


def staged_path(checkout_id: str, proposal_id: str) -> Path:
    """Proposed contents live beside the record, not inside it: the record keeps
    hashes so it stays reviewable, the staging area keeps the bytes to apply."""
    artifacts.require_artifact_id("proposal", proposal_id)
    return directory(checkout_id) / proposal_id / "staged.json"


def stage(checkout_id: str, proposal_id: str, contents: Dict[str, str]) -> Path:
    if not isinstance(contents, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in contents.items()):
        raise artifacts.MalformedArtifact(
            "staged proposal contents must map file names to text")
    target = staged_path(checkout_id, proposal_id)
    return artifacts.atomic_write(target, contents)


def staged(checkout_id: str, proposal_id: str) -> Optional[Dict[str, str]]:
    target = staged_path(checkout_id, proposal_id)
    try:
        text = paths.read_regular(target)
        value = json.loads(text) if text is not None else None
    except (ValueError, OSError):
        return None
    if not isinstance(value, dict) or not all(
            isinstance(name, str) and isinstance(contents, str)
            for name, contents in value.items()):
        return None
    return value


def save(checkout_id: str, rec: dict) -> Path:
    """Atomic: a proposal record read after a crash is a whole record, not a
    truncated one. It is the only account of what was changed."""
    target = path_for(checkout_id, rec["proposal_id"])
    return artifacts.save(target, rec, "proposal",
                          expected_checkout_id=checkout_id,
                          expected_artifact_id=rec["proposal_id"])


def load(checkout_id: str, proposal_id: str) -> Optional[dict]:
    target = path_for(checkout_id, proposal_id)
    return artifacts.load(target, "proposal", expected_checkout_id=checkout_id,
                          expected_artifact_id=proposal_id)


def load_all(checkout_id: str) -> List[dict]:
    listed = artifacts.members(directory(checkout_id))
    return artifacts.load_all([p for p in listed or [] if p.name.startswith("prop_")],
                              "proposal", expected_checkout_id=checkout_id)


def build_record(diagnosis: dict, checkout_id: str, outcome: str, created_at: str,
                 proposal_id: str, body: Optional[dict] = None,
                 mechanism: Optional[dict] = None, files: Optional[List[dict]] = None,
                 result: Optional[validation.Result] = None,
                 pre=None, post=None, failure: Optional[str] = None) -> dict:
    succeeded = outcome == reasoning.SUCCESS
    body = body or {}
    mechanism = mechanism or {}
    files = files or []
    mark = diagnosis.get("fingerprint") or {}
    recurrence = diagnosis.get("recurrence") or {}
    # §12.3 keeps a rejected candidate as failed-analysis evidence, and evidence
    # that cannot say what was tested does not stop anyone testing it again.
    # Hashes only: the staged contents of a rejected candidate are not kept.
    tested = bool(mechanism and files)
    change = None
    if tested:
        change = {"summary": str((body.get("change") or {}).get("summary", ""))[:600],
                  "files": [{"path": f["path"], "action": f["action"],
                             "sha256": _sha256(f["contents"]),
                             "baseline_sha256": f.get("baseline_sha256"),
                             "baseline_mode": f.get("baseline_mode"),
                             "bytes": len(f["contents"].encode("utf-8"))}
                            for f in files]}
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "proposal_id": proposal_id,
        "diagnosis_id": diagnosis["diagnosis_id"],
        "session_id": diagnosis["session_id"],
        "checkout_id": checkout_id,
        "created_at": created_at,
        "state": "candidate" if succeeded else "rejected",
        "outcome": outcome,
        "problem": (diagnosis.get("root_cause") or {}).get("summary"),
        "required_property": (diagnosis.get("required_property")
                              if succeeded or tested else None),
        "fingerprint": ({"fingerprint_id": mark["fingerprint_id"], "area": mark["area"],
                         "behavior": mark["behavior"],
                         "sessions": recurrence.get("sessions", 0)} if mark else None),
        "mechanism": ({"id": mechanism["id"], "kind": mechanism["kind"],
                       "name": mechanism["name"], "tier": mechanism["tier"],
                       "why_this": str(body.get("why_this", ""))[:600],
                       "why_existing_did_not_help":
                           str(body.get("why_existing_did_not_help", ""))[:600],
                       "alternatives_considered":
                           [str(a)[:300] for a in
                            (body.get("alternatives_considered") or [])][:8],
                       "deterministic": bool(mechanism.get("deterministic", True)),
                       "cost": mechanism.get("cost")}
                      if tested else None),
        "change": change,
        "expected_effect": str(body.get("expected_effect", ""))[:600] or None,
        "friction": [str(f)[:400] for f in (body.get("friction") or [])][:8],
        "privacy": {"adds_model_egress": bool(body.get("adds_model_egress")),
                    "note": ("this change sends code to an external model provider"
                             if body.get("adds_model_egress") else None),
                    # §13: a boolean is not a disclosure. The maintainer approves
                    # what leaves, to whom, when, how often and at what cost.
                    "egress": ({k: str(v)[:400] for k, v in
                                (body.get("egress") or {}).items()
                                if k in selection.EGRESS_FIELDS}
                               if body.get("adds_model_egress") else None)},
        "validation": {
            "performed": bool(result and result.performed),
            "rejects_problematic": result.rejects_problematic if result else None,
            "accepts_repaired": result.accepts_repaired if result else None,
            "attributable": result.attributable if result else None,
            "occurrences": [o.as_record() for o in result.occurrences] if result else [],
            "pre_correction": pre.as_record() if pre else None,
            "post_correction": post.as_record() if post else None,
            "working_tree_unchanged": result.working_tree_unchanged if result else None,
            "estimate": (result.estimate if result else None),
            "notes": [str(note)[:400] for note in (result.notes if result else [])][:8],
        },
        "measurement": {
            "target_fingerprint": mark.get("fingerprint_id"),
            "baseline_sessions": recurrence.get("sessions", 0),
            "signal": "recurrence of this fingerprint in sessions after the change",
        },
        "rollback": {"files": [], "instructions":
                     "not applied; nothing to undo" if not succeeded else
                     "run `repohone rollback <proposal-id>` to restore the "
                     "previous contents of every file this changed"},
        "applied_at": None,
        "repohone_runtime_required": False,
        "failure": {"reason": failure[:1000]} if failure else None,
    }


class _Existing:
    """Answers "does this path already exist?" without walking the repository."""

    def __init__(self, root):
        self.root = Path(root)

    def __contains__(self, relative) -> bool:
        try:
            return paths.present(self.root / relative)
        except OSError:
            return True        # cannot tell: never create over it

    def text(self, relative) -> Optional[str]:
        """What edits apply to. None when it cannot be read as text."""
        try:
            return paths.read_regular(self.root / relative)
        except (OSError, ValueError):
            return None

    def mode(self, relative) -> Optional[int]:
        """Part of the baseline: a check approved as executable that lands
        without that bit does not do what validation showed."""
        try:
            info = (self.root / relative).lstat()
        except OSError:
            return None
        return None if stat.S_ISLNK(info.st_mode) else stat.S_IMODE(info.st_mode)

    def sha256(self, relative) -> Optional[str]:
        current = self.text(relative)
        return None if current is None else _sha256(current)


def _existing_paths(root):
    return _Existing(root)


def _corrective_turn_for(checkout_id: str, session_id: str) -> Optional[int]:
    from . import diagnosis as diagnosis_module
    for record_ in diagnosis_module.load_all(checkout_id):
        if record_.get("session_id") == session_id and record_.get("corrective_turn"):
            return record_["corrective_turn"]
    return None


def _creates_its_runner(mechanism: dict, files, root) -> bool:
    """True when the command only exists because of this change — a `make verify`
    target the change adds. Its before-state run cannot succeed, and treating
    that as a failed baseline blocked the plan's third rung entirely."""
    if mechanism.get("tier") != "project-owned-custom":
        return False
    def created(relative):
        try:
            return not paths.present(Path(root) / relative)
        except OSError:
            return False          # cannot tell: it must still pass a baseline
    return any(Path(f["path"]).name in selection.RUNNER_FILES and created(f["path"])
               for f in files)


def _estimate(root, checkout_id: str, diagnosis: dict, files, engine):
    """A rule an agent reads cannot be run, so §14's executable ladder does not
    apply to it. It is graded against labelled sessions instead, and the result
    is labelled an estimate so nobody reads it as a demonstration."""
    rule_text = evidence.clean_text(
        "\n\n".join(f"{f['path']}:\n{f['contents']}" for f in files), root)[:2000]
    positives = _labelled(checkout_id, diagnosis, same=True, root=root)
    negatives = _labelled(checkout_id, diagnosis, same=False, root=root)
    estimate = validation.estimate_probabilistic(rule_text, positives, negatives, engine)
    # The caveat belongs to the estimate block, which prints it; repeating it
    # here would show the same sentence twice at the approval moment.
    notes = ["this mechanism is advice an agent reads, not a check that runs, so it "
             "was graded against past sessions rather than executed"]
    if not estimate.get("performed"):
        if estimate.get("note"):
            notes.append(estimate["note"])
        return validation.Result(False, notes=notes, estimate=estimate)
    caught = estimate["caught"]
    return validation.Result(
        True,
        rejects_problematic=(caught > 0),
        accepts_repaired=None,
        attributable=None,
        occurrences=[],
        notes=notes,
        estimate=estimate)


def _labelled(checkout_id: str, diagnosis: dict, same: bool, root=None) -> List[dict]:
    """Positive examples share this fingerprint. "Negatives" are sessions
    diagnosed with a *different* one — which means they had some other problem,
    not that they were free of this one. They are a proxy, and the estimate says
    so rather than reporting a false-positive rate they cannot support."""
    mark = (diagnosis.get("fingerprint") or {}).get("fingerprint_id")
    if not mark:
        return []
    occurrences = fingerprint.occurrences_for(checkout_id, mark).require_complete()
    wanted = set(occurrences)
    out = []
    from . import diagnosis as diagnosis_module
    owners = sorted({owner for owner, _ in wanted} | {checkout_id}) if same else [checkout_id]
    for owner in owners:
        for other in diagnosis_module.load_all(owner):
            session_id = other.get("session_id")
            matches = (owner, session_id) in wanted
            if matches != same or not session_id:
                continue
            try:
                rec = record.load(owner, session_id)
            except record.UnsupportedVersion:
                continue
            if rec is None:
                continue
            example = _summarise_session(
                session_id, rec, _corrective_turn_for(owner, session_id),
                root=root, owner=owner)
            if example:
                out.append(example)
    return out


def _summarise_session(session_id: str, rec: dict,
                       corrective_turn: Optional[int] = None, root=None,
                       owner: Optional[str] = None) -> Optional[dict]:
    """The turn that was actually corrected, not turns 1 and 2.

    A five-turn session whose correction is at turn 5 was being shown to the
    judge as if turn 2 corrected turn 1 — two unrelated requests presented as a
    mistake and its repair."""
    turns = rec.get("turns") or []
    if not turns:
        return None
    position = None
    if corrective_turn is not None:
        position = next((i for i, t in enumerate(turns)
                         if t.get("index") == corrective_turn), None)
    if position is None or position == 0:
        return None
    mistake, correction = turns[position - 1], turns[position]
    asked = ((mistake.get("prompt_events") or [{}])[0].get("content") or {}).get("text")
    produced = ((mistake.get("stop_events") or [{}])[0]
                .get("last_assistant_message") or {}).get("text")
    corrected = ((correction.get("prompt_events") or [{}])[0]
                 .get("content") or {}).get("text")
    if not asked:
        return None
    def clean(value):
        value = str(value or "")
        return evidence.clean_text(value, root)[:400] if root is not None else value[:400]

    # Session ids are checkout-local. Qualifying cross-checkout examples avoids
    # duplicate verdict ids being collapsed by the judge parser.
    example_id = f"{owner}/{session_id}" if owner else session_id
    return {"id": example_id, "asked": clean(asked),
            "produced": clean(produced),
            "corrected": clean(corrected) if corrected else None}


def _other_occurrences(checkout_id: str, diagnosis: dict, exclude: str):
    """The other sessions this fingerprint was seen in, with their own authentic
    pairs. A mechanism validated on one correction may miss the others."""
    mark = diagnosis.get("fingerprint") or {}
    if not mark.get("fingerprint_id"):
        return []
    out: list = []
    occurrences = fingerprint.occurrences_for(
        checkout_id, mark["fingerprint_id"]).require_complete()
    for owner, session_id in occurrences:
        if owner == checkout_id and session_id == exclude:
            continue
        try:
            other = record.load(owner, session_id)
        except record.UnsupportedVersion:
            other = None
        if other is None:
            out.append((owner, session_id, None, None))
            continue
        # Each occurrence needs its own identified boundary; without one it is
        # reported unevaluable rather than guessed at.
        boundary = _corrective_turn_for(owner, session_id)
        pre, post, _notes = validation.authentic_pair(other, boundary)
        out.append((owner, session_id, pre, post))
    return out


def _declined(body: Optional[dict]) -> Optional[str]:
    """"Nothing here fits" is a valued answer, so it has to arrive with its
    reasons; otherwise the record shows a refusal and no way to weigh it."""
    if not isinstance(body, dict):
        return None
    why = str(body.get("why_existing_did_not_help") or "").strip()
    alternatives = [str(a).strip() for a in (body.get("alternatives_considered") or [])
                    if str(a).strip()]
    if not why and not alternatives:
        return None
    parts = [why] if why else []
    parts += [f"considered: {a}" for a in alternatives[:4]]
    return " | ".join(parts)[:1000]


def propose(root, checkout_id: str, diagnosis: dict, mechanisms_record: Optional[dict],
            session_record: Optional[dict], engine,
            timeout: Optional[float] = None) -> dict:
    """One selection job, one terminal state. The record is written whatever
    happens: a rejected proposal is evidence, never something to apply."""
    created_at = record.now()
    proposal_id = new_id(diagnosis["diagnosis_id"], created_at)

    def finish(outcome, **kw):
        built = build_record(diagnosis, checkout_id, outcome, created_at,
                             proposal_id, **kw)
        save(checkout_id, built)
        return built

    if diagnosis.get("outcome") != reasoning.SUCCESS:
        return finish(reasoning.INSUFFICIENT_EVIDENCE,
                      failure=f"diagnosis {diagnosis['diagnosis_id']} ended "
                              f"{diagnosis.get('outcome')}; there is nothing to carry")
    mark_id = (diagnosis.get("fingerprint") or {}).get("fingerprint_id")
    if mark_id:
        try:
            fingerprint.occurrences_for(checkout_id, mark_id).require_complete()
        except fingerprint.IncompleteHistory as exc:
            return finish(reasoning.INSUFFICIENT_EVIDENCE, failure=str(exc))
    try:
        reply = engine.run(selection.build_prompt(diagnosis, mechanisms_record))
    except reasoning.Unavailable as exc:
        return finish(reasoning.MODEL_UNAVAILABLE, failure=str(exc))
    except reasoning.Interrupted as exc:
        return finish(reasoning.INTERRUPTED, failure=str(exc))

    verdict = selection.interpret(reasoning.extract_json(reply.text),
                                 mechanisms_record, _existing_paths(root))
    if verdict["outcome"] != reasoning.SUCCESS:
        return finish(verdict["outcome"], body=verdict.get("body"),
                      failure=verdict.get("failure")
                      or _declined(verdict.get("body")))

    mechanism, files, body = verdict["mechanism"], verdict["files"], verdict["body"]
    pre, post, notes = validation.authentic_pair(session_record or {},
                                                 diagnosis.get("corrective_turn"))
    overlay = {f["path"]: f["contents"] for f in files}
    try:
        if mechanism.get("deterministic", True):
            others = _other_occurrences(checkout_id, diagnosis,
                                        diagnosis["session_id"])
            result = validation.validate(
                root, checkout_id, mechanism.get("argv"), overlay,
                pre, post, timeout=timeout or 120.0, others=others,
                creates_runner=_creates_its_runner(mechanism, files, root))
        else:
            result = _estimate(root, checkout_id, diagnosis, files, engine)
    except fingerprint.IncompleteHistory as exc:
        return finish(reasoning.INSUFFICIENT_EVIDENCE, body=body,
                      mechanism=mechanism, files=files, pre=pre, post=post,
                      failure=str(exc))
    result.notes = notes + result.notes

    if not result.passed:
        return finish(reasoning.VALIDATION_REJECTED, body=body, mechanism=mechanism,
                      files=files, result=result, pre=pre, post=post,
                      failure="the candidate was not shown to catch the mistake this "
                              "project actually made")
    contents = {f["path"]: f["contents"] for f in files}
    # The record is the public candidate boundary.  Publish its reviewed bytes
    # first so a visible SUCCESS proposal is always applicable.  A crash before
    # the record leaves only an unreferenced private payload, which Doctor
    # reports; the reverse ordering exposed an incomplete candidate.
    staged_file = stage(checkout_id, proposal_id, contents)
    try:
        return finish(reasoning.SUCCESS, body=body, mechanism=mechanism, files=files,
                      result=result, pre=pre, post=post)
    except BaseException:
        try:
            staged_file.unlink()
        except OSError:
            pass
        raise


# --- application ----------------------------------------------------------

class NotApplicable(RuntimeError):
    """The proposal is not in a state where applying it is meaningful."""


class Conflict(RuntimeError):
    """A file changed since the proposal was made; applying would clobber it."""


class Escaped(Conflict):
    """A write may have landed outside the repository because the directory
    holding it was moved. Nothing can undo that, so it is reported as unresolved
    rather than folded into a tidy 'nothing changed'."""


def application_blocker(rec: dict) -> Optional[str]:
    """Why a readable historical proposal must be regenerated before apply."""
    contract = tuple(int(part) for part in rec["contract_version"].split("."))
    if contract < (1, 13) and any(
            entry.get("action") == "modify"
            for entry in (rec.get("change") or {}).get("files") or []):
        return ("this historical proposal predates mode-aware approval; re-run "
                "propose before applying it")
    privacy = rec.get("privacy") or {}
    if contract < (1, 15) and privacy.get("adds_model_egress"):
        return ("this historical proposal predates complete egress disclosure; "
                "re-run propose before applying it")
    return None


@contextmanager
def transition(checkout_id: str, proposal_id: str):
    """One writer per proposal, across the whole read-preflight-journal-write-save
    sequence. Two processes applying the same proposal otherwise both journal,
    and the loser's stale record overwrites the winner's — leaving a changed
    file with no rollback entry."""
    artifacts.require_artifact_id("proposal", proposal_id)
    paths.ensure(directory(checkout_id) / proposal_id)
    with record._locked(directory(checkout_id) / proposal_id / "transition",
                        suffix=".lock"):
        yield


def apply(root, checkout_id: str, proposal_id: str,
          contents: Dict[str, str]) -> dict:
    try:
        with transition(checkout_id, proposal_id):
            return _apply_locked(root, checkout_id, proposal_id, contents)
    except record.LockUnavailable as exc:
        raise Conflict(f"{exc}; nothing was changed") from exc


def _apply_locked(root, checkout_id: str, proposal_id: str,
                  contents: Dict[str, str]) -> dict:
    """Applying is an explicit, separate act. Previous contents are backed up
    first, so §27's 'how to undo it' is a fact rather than a promise."""
    rec = load(checkout_id, proposal_id)
    if rec is None:
        raise FileNotFoundError(f"no proposal {proposal_id}")
    if rec["outcome"] != reasoning.SUCCESS or rec["state"] not in ("candidate", "approved"):
        raise NotApplicable(f"proposal is {rec['state']}/{rec['outcome']}")
    blocker = application_blocker(rec)
    if blocker:
        raise NotApplicable(blocker)

    # Every staged file is checked before anything is written: verifying each one
    # as it lands leaves a half-applied tree that the rollback record does not describe.
    staged = {}
    for entry in rec["change"]["files"]:
        proposed = contents.get(entry["path"])
        if proposed is None or _sha256(proposed) != entry["sha256"]:
            raise Conflict(f"staged contents for {entry['path']} do not match the "
                           f"reviewed proposal; nothing was changed")
        staged[entry["path"]] = proposed

    backup = paths.ensure(backup_dir(checkout_id, proposal_id))
    rollback_files = []
    for entry in rec["change"]["files"]:
        safe_target(root, entry["path"])
        previous = _read_inside(root, entry["path"])
        existed = previous is not None
        current = _sha256(previous) if previous is not None else None
        if current != entry.get("baseline_sha256"):
            raise Conflict(
                f"{entry['path']} changed since this proposal was validated; "
                f"applying it would overwrite that edit and the validated result "
                f"no longer describes what would land. Re-run propose.")
        now_mode = current_mode(root, entry["path"])
        if (entry.get("baseline_mode") is not None and now_mode is not None
                and now_mode != entry["baseline_mode"]):
            raise Conflict(
                f"{entry['path']} was {oct(entry['baseline_mode'])} when this "
                f"proposal was validated and is {oct(now_mode or 0)} now; the "
                f"change was approved for a file that behaved differently. "
                f"Re-run propose.")
        if previous is not None:
            _save_backup(backup, entry["path"], previous)
        rollback_files.append({"path": entry["path"], "existed": existed,
                               "previous_sha256": current,
                               "previous_mode": current_mode(root, entry["path"])})

    keep_mode = {f["path"]: f.get("previous_mode") for f in rollback_files}
    quiet_before = probe.tree_fingerprint(root, deep=True)
    # Durable before the first write: a process killed mid-apply leaves files
    # changed, and without this the record would still say nothing happened.
    was_state = rec["state"]
    stage_token = rec.get("stage_token") or os.urandom(6).hex()
    rec["stage_token"] = stage_token
    rec["rollback"]["files"] = rollback_files
    rec["state"] = "applying"
    save(checkout_id, rec)
    written = []
    try:
        for entry in rec["change"]["files"]:
            # Re-checked here, not only in the preflight above: a file can be
            # edited, or replaced by a link out of the tree, while the earlier
            # files in this same proposal are being written.
            safe_target(root, entry["path"])
            # The parent existed during preflight. Creating it again by pathname
            # would follow a parent swapped to a link before the descriptor walk.
            # A missing parent is therefore a conflict; V1 never creates project
            # directories during apply.
            _replace_inside(root, entry["path"], staged[entry["path"]],
                            expect_sha=entry.get("baseline_sha256"),
                            mode=keep_mode.get(entry["path"]),
                            stage_token=stage_token)
            written.append(entry["path"])
    except (OSError, Conflict) as exc:
        cause = str(exc) if isinstance(exc, Conflict) \
            else f"writing {entry['path']} failed ({exc})"
        if isinstance(exc, Escaped):
            # Undoing cannot reach what left the repository, and claiming the
            # tree is as it was would be false.
            rec["rollback"]["files"] = [f for f in rollback_files
                                        if f["path"] in written]
            rec["state"] = UNRESOLVED
            rec["unresolved_paths"] = sorted(
                set(rec.get("unresolved_paths") or []) | {entry["path"]})
            rec["rollback"]["instructions"] = (
                f"{cause}. {entry['path']} is not in the repository and RepoHone "
                f"cannot undo a write outside it; resolve it by hand.")
            save(checkout_id, rec)
            raise
        recovered, failed = _undo(root, backup, rollback_files, written,
                                  rec["change"]["files"])
        if failed:
            # Only what is still changed. Listing a file recovery already put
            # back made a later rollback reject its own restored content.
            rec["rollback"]["files"] = [f for f in rollback_files
                                        if f["path"] in failed]
            rec["state"] = "partially_applied"
            save(checkout_id, rec)
            raise Conflict(
                f"{cause}; {', '.join(failed)} could not be restored, so the "
                f"proposal records what did land. Undo with "
                f"`repohone rollback {proposal_id}`.") from exc
        # Everything came back, so the journal has nothing left to describe.
        rec["rollback"]["files"] = []
        rec["state"] = was_state
        save(checkout_id, rec)
        raise Conflict(f"{cause}; {len(recovered)} file(s) restored, "
                       f"the repository is as it was") from exc

    disturbed = _disturbed(root, quiet_before,
                           {f["path"] for f in rec["change"]["files"]})
    if disturbed:
        # §27.1: never a clean apply when the tree moved underneath it.
        rec["state"] = "partially_applied"
        rec["applied_at"] = record.now()
        save(checkout_id, rec)
        raise Conflict(
            f"the working tree changed while this proposal was being applied "
            f"({', '.join(sorted(disturbed)[:3])}); what landed is recorded and "
            f"`repohone rollback {proposal_id}` undoes it, but this is not the "
            f"state that was validated.")
    # ARCH §26: an applied improvement is an experiment under observation. Nothing
    # measures it yet — that is Phase 9 — and the record says so rather than
    # implying an evaluation that is not happening.
    rec["state"] = "observing"
    rec["measurement"]["signal"] = (
        rec["measurement"]["signal"] + " (not yet measured)")
    rec["applied_at"] = record.now()
    save(checkout_id, rec)
    return rec


def _save_backup(backup: Path, relative: str, previous: str) -> None:
    """Write rollback data outside the project mutation boundary."""
    copy = backup / relative
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(previous, encoding="utf-8")


def safe_target(root, relative: str) -> Path:
    """Where `relative` lands, for messages and bookkeeping only.

    Checking a path and then opening it by name is a check of something else: the
    file can become a link to anywhere in between. Reads and writes go through
    the `_open_*` helpers below, which never follow one."""
    base = Path(root).resolve()
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise Conflict(f"{relative} escapes the repository")
    return base / relative


@contextmanager
def _parent_fd(root, relative: str):
    """Walks to the file's directory one component at a time, refusing to follow a
    link at any of them, and yields a descriptor for it. Opening the file relative
    to that descriptor is what makes the final step unswappable."""
    parts = Path(relative).parts
    if not parts:
        raise Conflict(f"{relative} names no file")
    fd = os.open(str(Path(root).resolve()), os.O_RDONLY)
    try:
        for name in parts[:-1]:
            try:
                nxt = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | _O_DIRECTORY,
                              dir_fd=fd)
            except NotADirectoryError:
                # O_NOFOLLOW on a link to a directory reports ENOTDIR, not ELOOP.
                raise Conflict(
                    f"{relative}: {name} is a symbolic link or not a directory; "
                    f"an approved change is confined to the repository") from None
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.EMLINK):
                    raise Conflict(
                        f"{relative} sits under the symbolic link {name}; an "
                        f"approved change is confined to the repository") from None
                raise
            os.close(fd)
            fd = nxt
        yield fd, parts[-1]
    finally:
        os.close(fd)


def _read_at(fd, name: str, relative: str) -> Optional[str]:
    """None only when the file does not exist.

    Content that is not text is a conflict, never an absence: treating it as
    absent made a binary file created after validation look like the `create`
    this proposal was validated against."""
    try:
        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise Conflict(
                f"{relative} is a symbolic link; an approved change is confined "
                f"to files inside the repository") from None
        raise
    try:
        with os.fdopen(handle, "r", encoding="utf-8") as fh:
            return fh.read()
    except UnicodeDecodeError:
        raise Conflict(f"{relative} holds bytes that are not text; this proposal "
                       f"was not validated against them") from None


def _read_inside(root, relative: str) -> Optional[str]:
    with _parent_fd(root, relative) as (fd, name):
        return _read_at(fd, name, relative)


_UNSET = object()


def _identity(fd, name: str):
    """What the directory entry is right now. Compared again at the commit, so an
    editor that rewrote it in place or renamed over it is noticed (§27.1)."""
    try:
        info = os.lstat(name, dir_fd=fd)
    except FileNotFoundError:
        return None
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size, info.st_mode)


def _disturbed(root, before, ours) -> set:
    """Anything that changed which this proposal did not name. Cheap, and it is
    what turns "someone edited alongside us" from invisible into reported."""
    after = probe.tree_fingerprint(root, deep=True)
    if before[0] != after[0]:
        return {"HEAD moved"}
    was_listed, now_listed = _status_paths(before[1]), _status_paths(after[1])
    # Both directions: a file that disappeared is as much a change as a new one.
    appeared = {path for path in now_listed ^ was_listed if path not in ours}
    # Content, not just which paths are dirty: a file that was already modified
    # stays on the same status line when it is modified again.
    rewritten = {path for path, digest in after[2].items()
                 if path in before[2] and before[2][path] != digest
                 and path not in ours}
    return appeared | rewritten


def _status_paths(porcelain: bytes) -> set:
    return probe.status_paths(porcelain)


def _still_inside(root, relative: str, fd: int) -> bool:
    """Is the directory we are holding open still the one this path names?

    A descriptor survives its directory being moved out of the tree, and the
    rename would then land outside. Re-walking from the root and comparing the
    inode is what notices that."""
    try:
        held = os.fstat(fd)
        with _parent_fd(root, relative) as (fresh, _name):
            now = os.fstat(fresh)
    except (OSError, Conflict):
        return False
    return (held.st_dev, held.st_ino) == (now.st_dev, now.st_ino)


def stage_name(token: str, name: str) -> str:
    """The staging file this proposal owns. Sweeping by a loose `.rh-*` pattern
    deleted a developer's own `.rh-my-notes-a.txt` sitting beside the target."""
    return f".rh-stage-{token}-{name}"


def _replace_inside(root, relative: str, text: str, expect_sha=_UNSET,
                    mode: Optional[int] = None, stage_token: str = "orphan") -> None:
    """Stages beside the target and renames over it.

    Truncating in place would rewrite the inode and every other name for it,
    including a hard link outside the repository. §27.1: the entry's identity is
    re-checked immediately before the rename and the result is re-read through a
    fresh walk, so a concurrent change is reported rather than silently kept."""
    with _parent_fd(root, relative) as (fd, name):
        before = _identity(fd, name)
        if before is not None and stat.S_ISLNK(before[4]):
            raise Conflict(f"{relative} is a symbolic link; an approved change "
                           f"is confined to files inside the repository")
        if expect_sha is not _UNSET:
            current = _read_at(fd, name, relative)
            if (_sha256(current) if current is not None else None) != expect_sha:
                raise Conflict(f"{relative} changed while this proposal was being "
                               f"applied; nothing of it was kept")
        keep = mode if mode is not None else (
            stat.S_IMODE(before[4]) if before is not None else 0o644)
        staged = stage_name(stage_token, name)
        try:
            handle = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW, 0o600, dir_fd=fd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise Conflict(f"{relative}: cannot stage a replacement") from None
            raise
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
                os.fchmod(fh.fileno(), keep)
            if _identity(fd, name) != before:
                raise Conflict(
                    f"{relative} was changed by something else while this "
                    f"proposal was being applied; it was left as it was found")
            if not _still_inside(root, relative, fd):
                raise Conflict(
                    f"the directory holding {relative} was moved out of the "
                    f"repository while this proposal was being applied")
            os.rename(staged, name, src_dir_fd=fd, dst_dir_fd=fd)
        except BaseException:
            try:
                os.unlink(staged, dir_fd=fd)
            except OSError:
                pass
            raise
    # A fresh walk from the repository root: the directory we held open could
    # have been moved out of the tree between opening it and the rename.
    try:
        landed = _read_inside(root, relative)
    except (OSError, Conflict):
        # Whatever went wrong, the file cannot be read back from the root, so
        # where the write landed is unknown — which is `Escaped`, not a tidy
        # conflict the caller can undo.
        landed = None
    if landed != text:
        raise Escaped(f"{relative} is not where it was written; the directory "
                      f"holding it moved while this proposal was being applied, "
                      f"so the content may have landed outside the repository")


def current_mode(root, relative: str) -> Optional[int]:
    """None for a symlink: its own bits say nothing about the file that was
    validated, and the link itself is refused before anything is written."""
    with _parent_fd(root, relative) as (fd, name):
        info = _identity(fd, name)
    if info is None or stat.S_ISLNK(info[4]):
        return None
    return stat.S_IMODE(info[4])


def _write_inside(root, relative: str, text: str) -> None:
    _replace_inside(root, relative, text)


def _unlink_inside(root, relative: str, expect_sha=_UNSET) -> None:
    with _parent_fd(root, relative) as (fd, name):
        if expect_sha is not _UNSET:
            current = _read_at(fd, name, relative)
            if current is not None and _sha256(current) != expect_sha:
                raise Conflict(f"{relative} changed after it was written; "
                               f"removing it would discard that edit")
        try:
            os.unlink(name, dir_fd=fd)
        except FileNotFoundError:
            pass


def _undo(root, backup, rollback_files, written, changed=None):
    """Best effort: returns how many came back and which did not, so a failure
    here is reported rather than leaving the caller to assume a clean tree."""
    by_path = {f["path"]: f for f in rollback_files}
    applied = {f["path"]: f.get("sha256") for f in changed or []}
    recovered, failed = [], []
    for path in reversed(written):
        try:
            # Only over content we put there. Someone editing a file RepoHone
            # had already written would otherwise lose it to the recovery.
            if by_path[path]["existed"]:
                _replace_inside(root, path,
                                (backup / path).read_text(encoding="utf-8"),
                                expect_sha=applied.get(path, _UNSET),
                                mode=by_path[path].get("previous_mode"))
            else:
                _unlink_inside(root, path, expect_sha=applied.get(path, _UNSET))
            recovered.append(path)
        except (OSError, Conflict):
            failed.append(path)
    return recovered, failed


UNRESOLVED = "unresolved"


def reconcile(root, checkout_id: str, proposal_id: str) -> dict:
    """What is actually on disk for an interrupted operation.

    A run killed between writes leaves the record saying `applying` and nothing
    else. Reading each file back and comparing it to the two hashes we know —
    what it was and what we meant it to become — is what turns that into a state
    that can be acted on."""
    rec = load(checkout_id, proposal_id)
    if rec is None:
        raise FileNotFoundError(f"no proposal {proposal_id}")
    if rec["state"] not in ("applying", "rolling_back"):
        return rec
    was_undoing = rec["state"] == "rolling_back"
    orphans = _sweep_stages(root, rec)
    proposed = {f["path"]: f.get("sha256") for f in rec["change"]["files"]}
    landed, original, unknown = [], [], []
    for entry in rec["rollback"]["files"]:
        path = entry["path"]
        try:
            current = _read_inside(root, path)
        except (OSError, Conflict):
            unknown.append(path)
            continue
        digest = _sha256(current) if current is not None else None
        if digest == proposed.get(path):
            landed.append(entry)
        elif digest == entry.get("previous_sha256") or (
                current is None and not entry.get("existed")):
            original.append(path)
        else:
            unknown.append(path)
    rec["rollback"]["files"] = landed
    if unknown:
        rec["state"] = UNRESOLVED
        rec["unresolved_paths"] = sorted(
            set(rec.get("unresolved_paths") or []) | set(unknown))
        rec["rollback"]["instructions"] = (
            f"interrupted while {'undoing' if was_undoing else 'applying'}; "
            f"{', '.join(unknown)} is neither what it was nor what was proposed "
            f"and must be resolved by hand. {len(landed)} other file(s) can be "
            f"undone with `repohone rollback {proposal_id}`.")
    elif landed:
        rec["state"] = "partially_applied"
        rec["rollback"]["instructions"] = (
            f"interrupted; {len(landed)} file(s) changed and can be undone with "
            f"`repohone rollback {proposal_id}`.")
    elif was_undoing:
        rec["state"] = "rejected"
        rec["rollback"]["instructions"] = (
            "interrupted while undoing, but every file was already restored")
    else:
        rec["state"] = "approved"
        rec["rollback"]["instructions"] = (
            "interrupted before anything changed"
            + (f"; removed {orphans} unfinished staging file(s)" if orphans else ""))
    save(checkout_id, rec)
    return rec


def _sweep_stages(root, rec) -> int:
    """A process killed between staging and the rename leaves this proposal's
    staging file beside the target: an unapproved file in the project, which the
    record would otherwise call `interrupted before anything changed`.

    Only the exact name this proposal wrote is removed. A pattern match deleted
    a developer's own file that happened to start with `.rh-`."""
    token = rec.get("stage_token")
    if not token:
        return 0
    removed = 0
    for entry in rec["change"]["files"]:
        relative = entry["path"]
        try:
            with _parent_fd(root, relative) as (fd, name):
                try:
                    os.unlink(stage_name(token, name), dir_fd=fd)
                    removed += 1
                except FileNotFoundError:
                    pass
        except (OSError, Conflict):
            continue
    return removed


def rollback_plan(root, checkout_id: str, proposal_id: str) -> List[str]:
    """What `rollback` would do, read only: an interrupted operation is reconciled
    by the rollback itself, not by its preview."""
    rec = load(checkout_id, proposal_id)
    if rec is None:
        raise FileNotFoundError(f"no proposal {proposal_id}")
    if rec["state"] in ("applying", "rolling_back"):
        return [f"first settle an interrupted operation ({rec['state']}), "
                f"then undo whatever it finds changed"]
    if rec["state"] not in ("applied", "observing", "ineffective", "harmful",
                            "partially_applied", UNRESOLVED):
        raise NotApplicable(f"proposal is {rec['state']}; nothing was applied")
    applied = {f["path"]: f for f in rec["change"]["files"]}
    lines = []
    for entry in rec["rollback"]["files"]:
        try:
            current = _read_inside(root, entry["path"])
        except (OSError, Conflict):
            current = None
        verb = "restore" if entry["existed"] else "remove"
        if current is not None and _sha256(current) != applied[entry["path"]]["sha256"]:
            lines.append(f"{verb} {entry['path']} — REFUSED: it changed after it was "
                         f"applied, and rolling back would discard that edit")
        else:
            lines.append(f"{verb} {entry['path']}")
    for path in rec.get("unresolved_paths") or []:
        lines.append(f"leave {path}: RepoHone cannot undo it (see the proposal's notes)")
    return lines or ["nothing: no file of this proposal is still changed"]


def rollback(root, checkout_id: str, proposal_id: str) -> dict:
    try:
        with transition(checkout_id, proposal_id):
            return _rollback_locked(root, checkout_id, proposal_id)
    except record.LockUnavailable as exc:
        raise Conflict(f"{exc}; nothing was undone") from exc


def _rollback_locked(root, checkout_id: str, proposal_id: str) -> dict:
    """Refuses when a file changed after application: undoing must not silently
    discard someone's later edit."""
    rec = reconcile(root, checkout_id, proposal_id)
    if rec["state"] not in ("applied", "observing", "ineffective", "harmful",
                           "partially_applied", UNRESOLVED):
        raise NotApplicable(f"proposal is {rec['state']}; nothing was applied")
    if rec["state"] == UNRESOLVED and not rec["rollback"]["files"]:
        raise Conflict(rec["rollback"]["instructions"])

    backup = backup_dir(checkout_id, proposal_id)
    applied = {f["path"]: f for f in rec["change"]["files"]}
    for entry in rec["rollback"]["files"]:
        current = _read_inside(root, entry["path"])
        if current is not None and _sha256(current) != applied[entry["path"]]["sha256"]:
            raise Conflict(f"{entry['path']} changed after it was applied; "
                           f"rolling back would discard that edit")
    unresolved = list(rec.get("unresolved_paths") or [])
    rec["state"] = "rolling_back"
    save(checkout_id, rec)
    undone: list = []
    for entry in rec["rollback"]["files"]:
        expect = applied[entry["path"]]["sha256"]
        try:
            if entry["existed"]:
                _replace_inside(root, entry["path"],
                                (backup / entry["path"]).read_text(encoding="utf-8"),
                                expect_sha=expect,
                                mode=entry.get("previous_mode"))
            else:
                _unlink_inside(root, entry["path"], expect_sha=expect)
        except (OSError, Conflict) as exc:
            # What was already undone stays undone, and the record says so
            # rather than claiming the repository is as it was. `partially_applied`
            # so a retry addresses only what is left.
            rec["rollback"]["files"] = [f for f in rec["rollback"]["files"]
                                        if f["path"] not in undone]
            rec["change"]["files"] = [f for f in rec["change"]["files"]
                                      if f["path"] not in undone]
            rec["state"] = "partially_applied"
            save(checkout_id, rec)
            raise Conflict(
                f"{entry['path']} could not be restored ({exc}); "
                f"{len(undone)} file(s) were. Re-run "
                f"`repohone rollback {proposal_id}` for the rest.") from exc
        undone.append(entry["path"])
    if unresolved:
        # Still unresolved, and its backups stay: what left the repository has
        # not come back, and its original content is the only copy left.
        rec["rollback"]["files"] = []
        rec["state"] = UNRESOLVED
        rec["rollback"]["instructions"] = (
            f"{len(undone)} file(s) undone. {', '.join(unresolved)} is still not "
            f"what it was and RepoHone cannot undo it; the original content is "
            f"kept under {backup}. Resolve it, then this proposal can be closed.")
        save(checkout_id, rec)
        return rec
    shutil.rmtree(directory(checkout_id) / proposal_id, ignore_errors=True)
    rec["state"] = "rejected"
    rec["rollback"]["instructions"] = "rolled back; the repository is as it was"
    save(checkout_id, rec)
    return rec
