"""What "diagnose it", "apply it" or "undo that" refers to when no id is given.

A preview picks the default and says why; the gated form always names its
target, so what was shown, the host's prompt and the approval agree."""
from __future__ import annotations

from typing import List, NamedTuple, Optional

from . import diagnosis, identity, proposal, reasoning, record, rule

ROLLBACK_STATES = ("applied", "observing", "ineffective", "harmful",
                   "partially_applied", proposal.UNRESOLVED)


class Pick(NamedTuple):
    chosen: Optional[str]
    why: str
    choices: List[str]
    candidate: bool = False


def _one(ids: List[str], what: str, none: str) -> Pick:
    if len(ids) == 1:
        return Pick(ids[0], f"the only {what}", [])
    if ids:
        return Pick(None, f"{len(ids)} {what}s fit; name one", ids)
    return Pick(None, none, [])


def for_diagnosis(checkout: str, host: str, host_session: Optional[str]) -> Pick:
    """A rule marked and not yet analysed — in this session when an agent asks —
    else the agent's own session. A rule whose turn is still in progress waits:
    its corrected state does not exist yet."""
    session_id, rec = None, None
    if host_session:
        session_id = identity.session_id(host, host_session)
        rec = record.load(checkout, session_id, strict=False) if session_id else None
    open_turn = rule.open_turn(rec) if rec else None
    marked = [c for c in rule.load_all(checkout) if c.get("state") == rule.MARKED
              and (session_id is None or c.get("session_id") == session_id)]
    ready = [c for c in marked if not (c.get("session_id") == session_id
                                       and c.get("corrective_turn") == open_turn)]
    if ready:
        pick = _one([c["candidate_id"] for c in ready], "rule marked and not yet analysed",
                    "")
        return pick._replace(candidate=True)
    if marked:
        return Pick(None, "the rule was marked in the turn still in progress; ask again "
                          "once this reply has finished", [])
    if rec is not None:
        return Pick(session_id, "this session: no rule is marked in it", [])
    return Pick(None, "no rule is waiting for analysis; name a session or a rule "
                      "(`repohone rules`, `repohone list`)", [])


def for_proposal(checkout: str) -> Pick:
    proposed = {p.get("diagnosis_id") for p in proposal.load_all(checkout)}
    ready = [d["diagnosis_id"] for d in _newest(diagnosis.load_all(checkout))
             if d.get("outcome") == reasoning.SUCCESS and d["diagnosis_id"] not in proposed]
    return _one(ready, "successful diagnosis without a proposal",
                "no successful diagnosis is waiting for a proposal; name one "
                "(`repohone diagnoses`)")


def applicable(rec: dict) -> bool:
    return (rec.get("outcome") == reasoning.SUCCESS and bool(rec.get("change"))
            and rec.get("state") in ("candidate", "approved"))


def for_apply(checkout: str) -> Pick:
    ready = [p["proposal_id"] for p in _newest(proposal.load_all(checkout)) if applicable(p)]
    return _one(ready, "validated proposal not yet applied",
                "no validated proposal is waiting to be applied (`repohone proposals`)")


def for_rollback(checkout: str) -> Pick:
    """The newest applied one: "undo that" means the last change."""
    applied = [p for p in proposal.load_all(checkout) if p.get("state") in ROLLBACK_STATES]
    applied.sort(key=lambda p: p.get("applied_at") or p.get("created_at") or "", reverse=True)
    if not applied:
        return Pick(None, "no applied proposal to roll back (`repohone proposals`)", [])
    others = [p["proposal_id"] for p in applied[1:]]
    return Pick(applied[0]["proposal_id"], "the most recently applied proposal", others)


def _newest(records: List[dict]) -> List[dict]:
    return sorted(records, key=lambda r: r.get("created_at") or "", reverse=True)
