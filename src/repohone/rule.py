"""Phase 2 — the explicit "this is a project rule" path.

Entirely local: resolve the session, resolve the evidence, persist a candidate.
No model call. Marking is consent to analyse *this* candidate (option B in the
plan) and never consent to change the project (ARCHITECTURE §1.2).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List, Optional

from . import CONTRACT_VERSION, SCHEMA_VERSION, artifacts, evidence, identity, paths, record

SCHEMA_ID = "repohone.rule_candidate/v1"
MARKED, ANALYSED, DISCARDED = "marked", "analysed", "discarded"
MAX_STATEMENT = 2000


class UnknownSession(LookupError):
    """The session is not in this checkout, so nothing can be attached to it."""


def candidates_dir(checkout_id: str) -> Path:
    return paths.checkout_dir(checkout_id) / "rules"


def new_id(checkout_id: str, session_id: str, created_at: str,
           nonce: Optional[str] = None) -> str:
    nonce = nonce if nonce is not None else os.urandom(8).hex()
    digest = hashlib.sha256(
        f"{checkout_id}\x00{session_id}\x00{created_at}\x00{nonce}".encode()
    ).hexdigest()
    return "rule_" + digest[:16]


def require_session(session_id: Optional[str]) -> str:
    """ARCHITECTURE §23: `session_id` is mandatory.

    Never inferred, not even from a sole live session. A caller whose own
    session was never captured would be handed somebody else's, and the rule
    would be filed against evidence it has nothing to do with."""
    session_id = (session_id or "").strip()
    if not session_id:
        raise UnknownSession(
            "session_id is required and is never inferred; pass the id of the "
            "session this statement was made in (`repohone list` shows them)")
    return session_id


def open_turn(rec: dict) -> Optional[int]:
    """The turn in progress, or None. A new prompt closes any earlier unfinished
    turn, and ending a session closes its own, so a resumed session is open again
    once it has a new prompt."""
    turns = rec.get("turns") or []
    pending = [t for t in turns if t.get("completion") == "pending"]
    if len(pending) != 1 or pending[0] is not turns[-1]:
        return None
    return pending[0]["index"]


def _host_session(checkout_id: str, host: str, host_session_id: str):
    session_id = identity.session_id(host, host_session_id)
    rec = None
    if session_id is not None:
        rec = record.load(checkout_id, session_id, expected_host=host,
                          expected_host_session_id=host_session_id)
    return session_id, rec


def mark_from_host(root, checkout_id: str, host: str, host_session_id: str,
                   statement: str, corrective_turn: Optional[int] = None) -> dict:
    """The host names the session and capture names the turn in progress; any
    doubt refuses rather than falling back to the most recent of either."""
    session_id, rec = _host_session(checkout_id, host, host_session_id)
    if rec is None:
        raise UnknownSession("this session has not been captured in this checkout")
    current = open_turn(rec)
    if current is None:
        raise UnknownSession("this session has no turn in progress to attach the rule to")
    return mark(root, checkout_id, session_id, statement, corrective_turn or current)


def mark(root, checkout_id: str, session_id: str, statement: str,
         corrective_turn: Optional[int] = None) -> dict:
    """Both ids must be supplied exactly; neither is inferred. A session id that
    belongs to another checkout simply does not resolve here, which is what keeps
    two concurrent sessions from being attributed to one another."""
    if not (checkout_id or "").strip():
        raise ValueError("a rule candidate needs the checkout it was marked in")
    if not (session_id or "").strip():
        raise ValueError("a rule candidate needs the session it came from")
    if not (statement or "").strip():
        raise ValueError("a rule candidate needs the developer's statement")

    rec = record.load(checkout_id, session_id)
    if rec is None:
        raise UnknownSession(f"no session {session_id} in this checkout")
    turn = evidence.validated_turn(rec, corrective_turn)

    created_at = record.now()
    selection = evidence.select(rec, root, rule_statement=statement,
                                corrective_turn=turn)
    candidate = {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "candidate_id": new_id(checkout_id, session_id, created_at),
        "checkout_id": checkout_id,
        "session_id": session_id,
        "created_at": created_at,
        "state": MARKED,
        "statement": statement.strip()[:MAX_STATEMENT],
        "corrective_turn": turn,
        "evidence_refs": selection.refs(),
        "consent": {
            "analysis": True,
            "scope": "this candidate only",
            "authorizes_apply": False,
        },
        "diagnosis_id": None,
    }
    save(checkout_id, candidate)
    return candidate


def save(checkout_id: str, candidate: dict) -> Path:
    artifacts.require_artifact_id("rule", candidate.get("candidate_id"))
    target = paths.ensure(candidates_dir(checkout_id)) / f"{candidate['candidate_id']}.json"
    return artifacts.save(target, candidate, "rule",
                          expected_checkout_id=checkout_id,
                          expected_artifact_id=candidate["candidate_id"])


def load(checkout_id: str, candidate_id: str) -> Optional[dict]:
    artifacts.require_artifact_id("rule", candidate_id)
    target = candidates_dir(checkout_id) / f"{candidate_id}.json"
    return artifacts.load(target, "rule", expected_checkout_id=checkout_id,
                          expected_artifact_id=candidate_id)


def load_all(checkout_id: str) -> List[dict]:
    listed = artifacts.members(candidates_dir(checkout_id))
    found = artifacts.load_all([p for p in listed or [] if p.name.startswith("rule_")],
                               "rule",
                               expected_checkout_id=checkout_id)
    return sorted(found, key=lambda c: c.get("created_at") or "", reverse=True)


def link_diagnosis(checkout_id: str, candidate_id: str, diagnosis_id: str,
                   succeeded: bool) -> Optional[dict]:
    """A failed diagnosis leaves the candidate `marked`: an unreachable model is
    a reason to try again, not a reason to call the question answered."""
    candidate = load(checkout_id, candidate_id)
    if candidate is None:
        return None
    candidate["diagnosis_id"] = diagnosis_id
    if succeeded:
        candidate["state"] = ANALYSED
    save(checkout_id, candidate)
    return candidate
