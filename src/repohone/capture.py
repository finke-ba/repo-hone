"""Adapter-neutral capture (ARCH §9, §16, §18, §20.1).

Capture observes; it never enforces. Every entry point is total: a failure
degrades the event to a null snapshot plus a capture error rather than losing
the event, and never reaches the host.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from . import CORE_VERSION, gitcmd, identity, profile, record, redact, snapshot, state, toolinput

PROMPT, STOP, STOP_FAILURE, SESSION_END = "prompt", "stop", "stop_failure", "session_end"


class Event:
    """One canonical lifecycle event, already mapped from a host payload."""

    def __init__(self, kind: str, host: str, host_session_id: str, cwd: str,
                 logical_turn_id: Optional[str] = None, text: Optional[str] = None,
                 error: Optional[str] = None, effort: Optional[str] = None,
                 model: Optional[str] = None, permission_mode: Optional[str] = None,
                 host_continuation: Optional[bool] = None, agent_version: Optional[str] = None,
                 end_reason: Optional[str] = None, source: Optional[str] = None,
                 transcript_path: Optional[str] = None,
                 last_assistant_message: Optional[str] = None,
                 tool_name: Optional[str] = None,
                 tool_use_id: Optional[str] = None,
                 tool_input: Optional[dict] = None,
                 tool_response: Any = None,
                 duration_ms: Optional[int] = None,
                 agent_id: Optional[str] = None,
                 agent_type: Optional[str] = None):
        self.kind = kind
        self.host = host
        self.host_session_id = host_session_id
        self.cwd = cwd
        self.logical_turn_id = logical_turn_id
        self.text = text
        self.error = error
        self.effort = effort
        self.model = model
        self.permission_mode = permission_mode
        self.host_continuation = host_continuation
        self.agent_version = agent_version
        self.end_reason = end_reason
        self.source = source
        self.transcript_path = transcript_path
        self.last_assistant_message = last_assistant_message
        self.tool_name = tool_name
        self.tool_use_id = tool_use_id
        self.tool_input = tool_input
        self.tool_response = tool_response
        self.duration_ms = duration_ms
        self.agent_id = agent_id
        self.agent_type = agent_type


def _content(text: Optional[str], policy: str) -> Optional[dict]:
    if text is None:
        return None
    cleaned, hits = redact.redact(text)
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return {"text": None if policy == "hash_only" else cleaned,
            "sha256": digest, "chars": len(text), "redactions": hits}


def _repository(root, checkout: str) -> dict:
    return {"root": str(root), "id": identity.repository_id(identity.origin_url(root)),
            "checkout_id": checkout, "start_head": gitcmd.head(root),
            "start_branch": gitcmd.branch(root),
            "branches_observed": [b for b in [gitcmd.branch(root)] if b]}


def _last_stop_tree(turn: dict) -> Optional[str]:
    for event in reversed(turn.get("stop_events") or []):
        snap = event.get("snapshot")
        if snap:
            return snap.get("tree")
    return None


def _close_unfinished(rec: dict) -> None:
    """Claude fires no event for an interrupt, so a still-pending turn that the
    session has moved past is closed here as derived (§20.1)."""
    for turn in rec["turns"]:
        if turn["completion"] == "pending" and not turn["stop_events"]:
            turn["completion"] = "interrupted"
            turn["completion_source"] = "derived"
            turn["completed_at"] = record.now()


SNAPSHOT_CLASS = {"turn.prompt": PROMPT, "turn.stop": STOP,
                  "turn.failed": STOP_FAILURE, "session.end": SESSION_END}

FOLD_ON = ("turn.stop", "turn.failed", "session.end")
# Events that only mean something as part of a turn. The schema requires a
# non-empty logical_turn_id, so without one there is no honest place to put them.
TURN_SCOPED = ("turn.prompt", "turn.stop", "turn.failed", "turn.interrupted")
TOOL_KINDS = ("turn.tool", "turn.tool_result", "subagent.start", "subagent.stop")
MAX_TOOL_EVENTS_PER_TURN = 250
TOOL_EXTENSION = "tool_events"
UNATTRIBUTED = "<unattributed>"
SUBAGENT_START, SUBAGENT_STOP = "<subagent-start>", "<subagent-stop>"


def _refusal(checkout: str, event: Event, reason: str,
             session_id: Optional[str] = None) -> Dict[str, Any]:
    """Return a refusal only after making the evidence loss durable."""
    try:
        recorded = state.record_refusal(checkout, event.kind, reason, record.now())
    except Exception:
        recorded = False
    status = {"captured": False, "reason": reason,
              "refusal_recorded": bool(recorded)}
    if session_id is not None:
        status["session_id"] = session_id
    return status


class Plan:
    """Ordinal and snapshot, obtained before the record lock is taken.

    Git must not run while the record is locked: a snapshot can take seconds,
    and a second hook on the same event would block for all of them (§9.1, §21).
    """

    def __init__(self, cls, ordinal, snapshot_record=None, errors=None):
        self.cls = cls
        self.ordinal = ordinal
        self.snapshot = snapshot_record
        self.errors = errors or []


def _plan(event: Event, root, checkout: str, session_id: str) -> Optional[Plan]:
    cls = SNAPSHOT_CLASS.get(event.kind)
    if cls is None:
        return None
    turn_key = None if cls == SESSION_END else event.logical_turn_id
    ordinal = state.allocate_ordinal(checkout, session_id, turn_key or "", cls)
    try:
        snap = snapshot.capture(root, checkout, session_id, cls, ordinal, turn_key,
                                record.now())
        return Plan(cls, ordinal, snap.as_record(), snap.errors)
    except Exception as exc:
        return Plan(cls, ordinal, None, [f"{type(exc).__name__}: {exc}"])


def handle(event: Event) -> Dict[str, Any]:
    """Returns a small status dict for diagnostics. Never raises."""
    root = gitcmd.toplevel(event.cwd)
    if root is None:
        return {"captured": False, "reason": "not a git repository"}

    prof = profile.load(root)
    if not prof.capture_enabled:
        return {"captured": False, "reason": f"profile {prof.state}"}
    # A committed profile is the project's; capturing a developer takes their own.
    if profile.accepted(root, prof) is not True:
        return {"captured": False, "reason": "not accepted by this developer"}

    checkout = identity.checkout_id(root)
    if checkout is None:
        return {"captured": False, "reason": "no checkout id"}

    session_id = identity.session_id(event.host, event.host_session_id)
    if session_id is None:
        reason = "host supplied no session id"
        return _refusal(checkout, event, reason)
    try:
        state.ensure_session(checkout, session_id, event.host, event.host_session_id,
                             record.now())
    except state.StateCompatibilityError as exc:
        return _refusal(checkout, event, str(exc), session_id)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return _refusal(checkout, event, reason, session_id)

    if event.kind in TURN_SCOPED and not (event.logical_turn_id or "").strip():
        return _reject(checkout, session_id, event, prof, root,
                       "host supplied no logical turn id")

    if event.kind in TOOL_KINDS:
        return _on_tool(event, root, checkout, session_id, prof)

    handler = {
        "session.start": _on_start,
        "turn.prompt": _on_prompt,
        "turn.stop": _on_stop,
        "turn.failed": _on_stop,
        "turn.interrupted": _on_interrupted,
        "model.switch": _on_model_switch,
        "session.end": _on_end,
    }.get(event.kind)

    factory = _factory(session_id, event, prof, root, checkout)

    # Allocation, snapshot and append are one ordered step per session (§18):
    # a later event must not overtake an earlier one and become its baseline.
    try:
        with record.session_lock(checkout, session_id):
            # Refuse an incompatible existing record before allocating an
            # ordinal or snapshot. A future record is evidence, not a current
            # record that capture may append to or replace.
            record.load(checkout, session_id, expected_host=event.host,
                        expected_host_session_id=event.host_session_id)
            plan, plan_error = None, None
            if handler is not None:
                try:
                    plan = _plan(event, root, checkout, session_id)
                except Exception as exc:
                    plan_error = f"{type(exc).__name__}: {exc}"
            return _commit(checkout, session_id, event, handler, plan, plan_error,
                           prof, factory)
    except Exception as exc:
        # §9.2: the event is lost rather than written unordered, and the loss is
        # recorded so doctor can say evidence is missing.
        reason = (str(exc) if isinstance(exc, (record.LockUnavailable,
                                               record.ArtifactError))
                  else f"{type(exc).__name__}: {exc}")
        return _refusal(checkout, event, reason, session_id)


def _factory(session_id, event, prof, root, checkout):
    def build():
        return record.new_record(
            session_id, event.host, event.host_session_id, event.host, CORE_VERSION,
            prof.content_policy, _repository(root, checkout),
            identity.developer_id(root), event.host, event.agent_version,
            record.now(), str(event.cwd))
    return build


def _reject(checkout, session_id, event, prof, root, reason: str) -> Dict[str, Any]:
    """Records that an event arrived and could not be placed. Dropping it silently
    would make the record look complete when a turn is missing from it."""
    try:
        with record.session_lock(checkout, session_id), record.open_record(
                checkout, session_id, _factory(session_id, event, prof, root, checkout),
                expected_host=event.host,
                expected_host_session_id=event.host_session_id) as rec:
            record.add_error(rec, event.kind, reason)
    except Exception as exc:
        return _refusal(checkout, event, f"{type(exc).__name__}: {exc}", session_id)
    return {"captured": False, "reason": reason, "session_id": session_id}


def _commit(checkout, session_id, event, handler, plan, plan_error, prof, factory):
    with record.open_record(checkout, session_id, factory,
                            expected_host=event.host,
                            expected_host_session_id=event.host_session_id) as rec:
        if handler is None:
            record.add_error(rec, event.kind, "unknown event kind")
            return {"captured": False, "reason": "unknown event"}
        for message in ([plan_error] if plan_error else []) + (plan.errors if plan else []):
            record.add_error(rec, event.kind, message)
        try:
            refused = handler(rec, event, plan, prof)
            if event.kind in FOLD_ON and prof.tool_capture:
                _fold_tool_events(rec, checkout, session_id)
                if event.kind == "session.end":
                    state.clear_tool_events(checkout, session_id)
        except Exception as exc:
            record.add_error(rec, event.kind, f"{type(exc).__name__}: {exc}")
            return {"captured": False, "reason": str(exc), "session_id": session_id}
        if refused:
            return {"captured": False, "reason": refused, "session_id": session_id}
    return {"captured": True, "session_id": session_id}


def _on_tool(event, root, checkout: str, session_id: str, prof) -> Dict[str, Any]:
    """Appended to local state only. Which files the agent consulted before acting
    is what separates a missing-context failure from a reasoning one."""
    if not prof.tool_capture:
        return {"captured": False, "reason": "tool capture disabled"}
    turn_key = event.logical_turn_id or UNATTRIBUTED
    try:
        if event.kind == "turn.tool_result":
            return _on_tool_result(event, checkout, session_id)
        tool, detail = _tool_row(event, root, prof)
        seq = state.append_tool_event(checkout, session_id, turn_key, record.now(),
                                      tool, detail, event.tool_use_id,
                                      MAX_TOOL_EVENTS_PER_TURN)
    except Exception as exc:
        return _refusal(checkout, event, f"{type(exc).__name__}: {exc}", session_id)
    if seq is None:
        return {"captured": False, "reason": "per-turn tool cap reached"}
    return {"captured": True, "session_id": session_id, "seq": seq}


def _on_tool_result(event, checkout: str, session_id: str) -> Dict[str, Any]:
    """PostToolUse: the outcome the pre-event could not know. A recorded call is
    otherwise indistinguishable from one the host blocked."""
    if not event.tool_use_id:
        return _refusal(checkout, event, "tool result without a tool_use_id", session_id)
    summary = toolinput.summarize_response(event.tool_response, event.duration_ms)
    matched = state.set_tool_outcome(checkout, session_id, event.tool_use_id,
                                     json.dumps(summary) if summary else json.dumps({}))
    if not matched:
        return _refusal(checkout, event,
                        f"no staged tool call for tool_use_id {event.tool_use_id!r}",
                        session_id)
    return {"captured": True, "session_id": session_id}


def _tool_row(event, root, prof):
    """Subagent boundaries ride the same stream so folding can bracket them."""
    if event.kind == "subagent.start":
        return SUBAGENT_START, json.dumps(
            {"agent_id": event.agent_id, "agent_type": event.agent_type})
    if event.kind == "subagent.stop":
        return SUBAGENT_STOP, json.dumps(
            {"agent_id": event.agent_id, "agent_type": event.agent_type})
    minimized = toolinput.minimize(event.tool_input, root, prof.content_policy)
    return (event.tool_name or "unknown"), (json.dumps(minimized) if minimized else None)


def _fold_tool_events(rec: dict, checkout: str, session_id: str) -> None:
    """Rebuilt wholesale so folding twice is harmless. Experimental, so it lives
    in `extensions` until it earns a place in a v2 record shape (§22.1)."""
    grouped = state.tool_events(checkout, session_id)
    if not grouped:
        return
    # Merge, never replace: staged rows are cleared at session end, so a resumed
    # session would otherwise rebuild from what is left and lose earlier turns.
    existing = (rec["extensions"].get(TOOL_EXTENSION) or {}).get("turns") or {}
    turns = dict(existing)
    for turn_key, events in grouped.items():
        attributed = _attribute(events)
        turns[turn_key] = {
            "count": len(attributed),          # markers are boundaries, not tool calls
            "at_cap": len(events) >= MAX_TOOL_EVENTS_PER_TURN,
            "events": attributed,
        }
    rec["extensions"][TOOL_EXTENSION] = {
        "schema": "repohone.tool_events/experimental",
        "turns": turns,
    }


def _attribute(events):
    """Adds `outcome_observed`, and attributes subagent work.

    Whether an absent outcome means failure is host-specific and belongs to the
    adapter: on Claude PostToolUse fires only on success, so absence on a closed
    turn means the call did not succeed. The generic fold only says whether a
    result was observed.

    A subagent's own tool calls arrive under the parent's turn with no marker
    of their own, so they are bracketed by the subagent start/stop events. With
    two subagents running at once the bracket cannot separate them, and the
    events say `ambiguous` rather than naming the wrong one."""
    active = []
    out = []
    for event in events:
        detail = event.get("detail") or {}
        if event["tool"] == SUBAGENT_START:
            active.append(detail.get("agent_id"))
            continue
        if event["tool"] == SUBAGENT_STOP:
            if detail.get("agent_id") in active:
                active.remove(detail.get("agent_id"))
            continue
        entry = dict(event)
        entry["outcome_observed"] = event.get("outcome") is not None
        if len(active) == 1:
            entry["agent_id"] = active[0]
        elif len(active) > 1:
            entry["agent_id"] = "ambiguous"
        out.append(entry)
    return out


def _note_cwd(rec: dict, cwd: str) -> None:
    if cwd and cwd not in rec["capture"]["cwds"]:
        rec["capture"]["cwds"].append(cwd)


def _on_start(rec, event, plan, prof):
    _note_cwd(rec, str(event.cwd))
    starts = rec["lifecycle"]["starts"]
    if starts and starts[-1]["source"] is None:
        starts[-1]["source"] = event.source
    else:
        starts.append({"at": record.now(), "source": event.source, "cwd": str(event.cwd)})
    if event.agent_version and not rec["agent"]["version"]:
        rec["agent"]["version"] = event.agent_version
    if event.model:
        _observe_model(rec, event.model, "session_start", len(rec["turns"]) or 1)


def _on_model_switch(rec, event, plan, prof):
    """A switch applies to what the host generates next, not to the turn in flight."""
    if event.model:
        _observe_model(rec, event.model, "model_switch", len(rec["turns"]) + 1)


def _observe_model(rec, model: str, source: str, from_turn: int) -> None:
    rec["models"].append({"model": model, "from_turn": max(1, from_turn),
                          "observed_at": record.now(), "source": source})


def _on_prompt(rec, event, plan, prof):
    _note_cwd(rec, str(event.cwd))
    turn_key = event.logical_turn_id
    turn = next((t for t in rec["turns"] if t["logical_turn_id"] == turn_key), None)

    if turn is None:
        _close_unfinished(rec)
        previous = rec["turns"][-1] if rec["turns"] else None
        turn = record.new_turn(len(rec["turns"]) + 1, turn_key, str(event.cwd),
                               None, event.permission_mode)
        rec["turns"].append(turn)
    else:
        previous = None

    turn["prompt_events"].append({
        "ordinal": plan.ordinal, "at": record.now(),
        "content": _content(event.text, prof.content_policy), "snapshot": plan.snapshot})

    if previous is not None and len(turn["prompt_events"]) == 1:
        turn["workspace_changed_before_turn"] = _changed_since(previous, plan.snapshot)


def _changed_since(previous: dict, snap: Optional[dict]) -> Optional[bool]:
    """Null unless both boundaries exist: an interrupted turn has no moment at
    which the agent stopped, so nothing is attributable."""
    if snap is None or previous["completion"] != "stop":
        return None
    before = _last_stop_tree(previous)
    return None if before is None else before != snap.get("tree")


def _find_turn(rec, turn_key):
    """No positional fallback: attributing a stop to the latest turn because its
    id matched nothing records a completion the host never reported."""
    return next((t for t in rec["turns"] if t["logical_turn_id"] == turn_key), None)


def _on_stop(rec, event, plan, prof):
    turn = _find_turn(rec, event.logical_turn_id)
    if turn is None:
        reason = (f"stop for unknown turn {event.logical_turn_id!r}; "
                  f"not attributed")
        record.add_error(rec, event.kind, reason)
        return reason

    failed = event.kind == "turn.failed"
    turn["stop_events"].append({
        "ordinal": plan.ordinal, "at": record.now(), "kind": plan.cls,
        "error": event.error if failed else None, "snapshot": plan.snapshot,
        "last_assistant_message": _content(event.last_assistant_message,
                                           prof.content_policy),
        "effort": event.effort, "model": event.model,
        "host_continuation": event.host_continuation})

    turn["completion"] = plan.cls
    turn["completion_source"] = "observed"
    turn["completed_at"] = record.now()
    if event.model:
        turn["model"] = event.model
        _observe_model(rec, event.model, "hook_input", turn["index"])


def _on_interrupted(rec, event, plan, prof):
    """Hosts that report an interrupt directly (Codex). Claude never reaches here."""
    turn = _find_turn(rec, event.logical_turn_id)
    if turn is None:
        reason = (f"interrupt for unknown turn {event.logical_turn_id!r}; "
                  f"not attributed")
        record.add_error(rec, event.kind, reason)
        return reason
    turn["completion"] = "interrupted"
    turn["completion_source"] = "observed"
    turn["completed_at"] = record.now()


def _on_end(rec, event, plan, prof):
    _close_unfinished(rec)
    if plan.snapshot is not None:
        rec["session_end_snapshots"].append(plan.snapshot)
    rec["lifecycle"]["ended_at"] = record.now()
    rec["lifecycle"]["end_reason"] = event.end_reason
    if event.transcript_path:
        rec["transcript"]["source_path"] = event.transcript_path
