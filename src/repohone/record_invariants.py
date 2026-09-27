"""Cross-field invariants for repohone.session/v1 that JSON Schema cannot express.

The JSON Schema governs shape. These are the writer-side rules a record must also
satisfy — chiefly that a lifecycle event and the snapshot it points at agree on
identity. Phase 1 capture calls `check_record` in its own tests; the contract
tests here prove the checks catch real violations.

Returns a list of human-readable violations; empty means conformant.
"""
from __future__ import annotations

from typing import Any, Dict, List

CLASS_FOR_STOP_KIND = {"stop": "stop", "stop_failure": "stop_failure"}
REF_SEGMENT = {"prompt": "prompt", "stop": "stop",
               "stop_failure": "stop-failure", "session_end": "session-end"}


def check_record(rec: Dict[str, Any]) -> List[str]:
    bad: List[str] = []
    for turn in rec.get("turns") or []:
        where = f"turn {turn.get('index')}"
        for event in turn.get("prompt_events") or []:
            bad += _check_event(event, "prompt", f"{where} prompt/{event.get('ordinal')}")
        for event in turn.get("stop_events") or []:
            expected = CLASS_FOR_STOP_KIND.get(event.get("kind"))
            bad += _check_event(event, expected, f"{where} stop/{event.get('ordinal')}")
        bad += _check_ordinals(turn.get("prompt_events"), f"{where} prompt_events")
        bad += _check_ordinals(turn.get("stop_events"), f"{where} stop_events")
        bad += _check_completion(turn, where)
    bad += _check_turn_identity(rec)
    bad += _check_model_attribution(rec)
    ends = rec.get("session_end_snapshots") or []
    for i, snap in enumerate(ends):
        if snap.get("class") != "session_end":
            bad.append(f"session_end_snapshots[{i}]: class is {snap.get('class')!r}, "
                       "expected 'session_end'")
        bad += _check_ref(snap, f"session_end_snapshots[{i}]")
    bad += _check_ordinals(ends, "session_end_snapshots")
    bad += _check_lifecycle(rec)
    return bad


def _check_event(event, expected_class, where) -> List[str]:
    snap = event.get("snapshot")
    if snap is None:
        return []
    bad = []
    if expected_class and snap.get("class") != expected_class:
        bad.append(f"{where}: snapshot class {snap.get('class')!r}, expected {expected_class!r}")
    if snap.get("ordinal") != event.get("ordinal"):
        bad.append(f"{where}: snapshot ordinal {snap.get('ordinal')} "
                   f"!= event ordinal {event.get('ordinal')}")
    bad += _check_ref(snap, where)
    return bad


def _check_ref(snap, where) -> List[str]:
    """The ref is what a reader actually resolves, so it must encode the class and
    ordinal the record claims; the schema only checks its prefix.

    A missing ref is the schema's required-field violation, not a disagreement.
    """
    ref = snap.get("ref")
    segment = REF_SEGMENT.get(snap.get("class"))
    if not ref or segment is None:
        return []
    expected = f"/{segment}/{snap.get('ordinal')}"
    if not ref.endswith(expected):
        return [f"{where}: ref {ref!r} does not end with {expected!r}"]
    return []


def _check_turn_identity(rec) -> List[str]:
    """One logical turn, one entry: type-ahead extends a turn rather than opening
    a second one. Fragments in other records may repeat the id; this one may not."""
    seen: Dict[Any, Any] = {}
    bad = []
    for turn in rec.get("turns") or []:
        ltid = turn.get("logical_turn_id")
        if not isinstance(ltid, str) or not ltid.strip():
            bad.append(f"turn {turn.get('index')}: logical_turn_id is {ltid!r}; "
                       f"a turn with no host identity cannot be joined or completed")
        if ltid in seen:
            bad.append(f"turn {turn.get('index')}: logical_turn_id {ltid!r} "
                       f"already used by turn {seen[ltid]}")
        seen[ltid] = turn.get("index")
    return bad


def _check_model_attribution(rec) -> List[str]:
    """A turn's model must be one the record saw the host report. Otherwise a
    diagnosis could attribute behaviour to a model that never appears in the
    evidence, and nothing would contradict it."""
    observed = {entry.get("model") for entry in rec.get("models") or []}
    bad = []
    for turn in rec.get("turns") or []:
        model = turn.get("model")
        if model and model not in observed:
            bad.append(f"turn {turn.get('index')}: model {model!r} is not in "
                       f"models[]; nothing observed the host reporting it")
    return bad


def _check_lifecycle(rec) -> List[str]:
    """The phase marker and the reconciliation payload must agree (ARCH 19).
    Under 'observed' the outcome is still pending and any agent-final is provisional.
    """
    life = rec.get("lifecycle") or {}
    recon = rec.get("reconciliation") or {}
    state, outcome = life.get("state"), recon.get("outcome")
    final = recon.get("agent_final_snapshot") or {}
    bad = []

    if state == "observed":
        if outcome != "pending":
            bad.append(f"lifecycle: 'observed' with outcome {outcome!r}; "
                       "reconciliation has not run")
        if life.get("reconciled_at") is not None:
            bad.append("lifecycle: 'observed' with a reconciled_at")
        if recon.get("accepted_snapshot") is not None:
            bad.append("lifecycle: 'observed' with an accepted_snapshot")
        if final.get("resolution") == "reconciled":
            bad.append("lifecycle: 'observed' with a reconciled agent_final_snapshot")
    elif state == "reconciled":
        if outcome == "pending":
            bad.append("lifecycle: 'reconciled' with outcome 'pending'")
        if life.get("reconciled_at") is None:
            bad.append("lifecycle: 'reconciled' with no reconciled_at")
    return bad


def _check_ordinals(events, where) -> List[str]:
    """Ordinals are allocated, append-only and never reused, so within a record
    they must be strictly increasing — gaps are legal, repeats are not."""
    if not events:
        return []
    ordinals = [e.get("ordinal") for e in events]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        return [f"{where}: ordinals {ordinals} are not strictly increasing"]
    return []


def _check_completion(turn, where) -> List[str]:
    """Stop and StopFailure completions require the corresponding captured event.

    Interruption is deliberately NOT constrained to one source here: some hosts
    report it directly (observed), others expose it only as an absence and it is
    closed at the next snapshot (derived). Which applies is adapter knowledge and
    belongs in adapter tests, not in the host-neutral data contract.
    """
    completion = turn.get("completion")
    source = turn.get("completion_source")
    stops = turn.get("stop_events") or []
    bad: List[str] = []

    if completion == "pending":
        if source is not None:
            bad.append(f"{where}: 'pending' has no completion source; got {source!r}")
        return bad

    if completion in ("stop", "stop_failure"):
        if not stops:
            bad.append(f"{where}: completion {completion!r} with no stop_events")
        elif stops[-1].get("kind") != completion:
            bad.append(f"{where}: completion {completion!r} but last stop event is "
                       f"{stops[-1].get('kind')!r}")
        if source != "observed":
            bad.append(f"{where}: completion {completion!r} must have completion_source 'observed'")

    if completion == "interrupted" and source not in ("observed", "derived"):
        bad.append(f"{where}: 'interrupted' must be observed or derived; got {source!r}")
    return bad
