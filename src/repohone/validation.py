"""Validation against authentic captured states (LEARNING_PLAN §14, ARCH §12.2).

A candidate mechanism is checked against the states this project actually
produced — the source as the agent left it, and the source after the developer
corrected it. That is strictly stronger than a synthetic fixture: it proves the
mechanism would have caught **the mistake this project actually made**.

Everything runs against detached checkouts in scratch. A candidate that fails
terminates as VALIDATION_REJECTED and is never presented as ready to apply.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import probe, reasoning

# LEARNING_PLAN §14, in preference order. The fallbacks are not decoration: a
# type-ahead correction arrives at prompt/2 while the agent is still working and
# never stopped, and an interrupted turn fires no Stop at all.
PRE_LADDER = ("stop", "prompt", "stop_failure")
POST_LADDER = ("stop", "session_end")


@dataclass
class State:
    source: str
    tree: str
    ref: Optional[str]

    def as_record(self) -> dict:
        return {"source": self.source, "tree": self.tree, "ref": self.ref}


@dataclass
class Occurrence:
    session_id: str
    checkout_id: Optional[str] = None
    caught: Optional[bool] = None
    false_positive: Optional[bool] = None
    note: Optional[str] = None

    def as_record(self) -> dict:
        return {"session_id": self.session_id, "checkout_id": self.checkout_id,
                "caught": self.caught,
                "false_positive": self.false_positive, "note": self.note}


@dataclass
class Result:
    performed: bool
    rejects_problematic: Optional[bool] = None
    accepts_repaired: Optional[bool] = None
    attributable: Optional[bool] = None
    working_tree_unchanged: Optional[bool] = None
    occurrences: List[Occurrence] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    estimate: Optional[dict] = None       # probabilistic mechanisms only

    def __post_init__(self):
        self.notes = list(self.notes or [])
        self.occurrences = list(self.occurrences or [])

    @property
    def coverage(self):
        evaluated = [o for o in self.occurrences if o.caught is not None]
        return sum(1 for o in evaluated if o.caught), len(evaluated)

    @property
    def false_positives(self) -> int:
        return sum(1 for o in self.occurrences if o.false_positive)

    @property
    def passed(self) -> bool:
        """Three separate claims, all required: the change is what catches the
        mistake, it catches the diagnosed one, and it fires on nothing that was
        already correct.

        A mechanism that cannot be executed is judged on its estimate instead,
        and the proposal says so rather than borrowing this word."""
        if self.estimate is not None:
            # Same standard as the executable path: it must fire on nothing that
            # was already correct. An estimate is weaker evidence, not a lower bar.
            return (bool(self.estimate.get("performed"))
                    and self.estimate.get("caught", 0) > 0
                    and self.estimate.get("misfired", 0) == 0)
        # §12.2: a run that changed the developer's checkout has already broken
        # the boundary it was supposed to respect, whatever its exit codes said.
        # Unevaluated occurrences are reported, never failed.
        return (bool(self.attributable) and bool(self.rejects_problematic)
                and bool(self.accepts_repaired) and self.false_positives == 0
                and self.working_tree_unchanged is True)


def _snapshot_state(snapshot, source: str) -> Optional[State]:
    if not snapshot or not snapshot.get("tree"):
        return None
    return State(source, snapshot["tree"], snapshot.get("ref"))


def _last_stop_state(turn) -> Optional[State]:
    for event in reversed(turn.get("stop_events") or []):
        state = _snapshot_state(event.get("snapshot"),
                                "stop_failure" if event.get("kind") == "stop_failure"
                                else "stop")
        if state:
            return state
    return None


def _position_of(rec: dict, turn_index: Optional[int]) -> Optional[int]:
    """Records number turns from 1; this is their position in the list."""
    if turn_index is None:
        return None
    for position, turn in enumerate(rec.get("turns") or []):
        if turn.get("index") == turn_index:
            return position
    return None


def authentic_pair(rec: dict, corrective_turn: Optional[int] = None
                   ) -> Tuple[Optional[State], Optional[State], List[str]]:
    """Returns (pre-correction, post-correction, notes).

    The corrective turn must be **identified**, never inferred from adjacency.
    Two unrelated turns — a feature followed by a docs request — look exactly
    like a correction and its repair, and calling them one would certify a
    mechanism against a change nobody made.
    """
    turns = rec.get("turns") or []
    notes: List[str] = []
    index = _position_of(rec, corrective_turn)
    if index is None:
        return None, None, ["no identified correction boundary; authentic "
                            "validation is unavailable for this session"]
    if index <= 0 or index >= len(turns):
        return None, None, [f"turn {corrective_turn} has no predecessor to "
                            f"compare against"]

    turn, previous = turns[index], turns[index - 1]

    pre = _last_stop_state(previous)
    if pre is None:
        pre = _snapshot_state((turn.get("prompt_events") or [{}])[0].get("snapshot"),
                              "prompt")
        if pre:
            notes.append("no stop before the correction; used the corrective "
                         "prompt's own snapshot")
    if pre is None:
        notes.append("no pre-correction state could be reconstructed")

    post = _last_stop_state(turn)
    if post is None:
        ends = rec.get("session_end_snapshots") or []
        post = _snapshot_state(ends[-1] if ends else None, "session_end")
        if post:
            notes.append("the corrective turn never stopped; used the session-end "
                         "snapshot")
    if post is None:
        notes.append("no post-correction state could be reconstructed")

    if pre and post and pre.tree == post.tree:
        notes.append("pre- and post-correction trees are identical; the correction "
                     "changed nothing, so it cannot discriminate a mechanism")
    return pre, post, notes


JUDGE_PROMPT = """You are grading a proposed project rule against real sessions
from this repository. You are not writing the rule and not improving it.

THE PROPOSED RULE
{rule}

For each session below, answer whether an agent following that rule, before it
acted, would have avoided the mistake the developer had to correct — or, for a
session where no such mistake happened, whether the rule would have wrongly
blocked or complained about correct work.

SESSIONS
{examples}

Reply with ONE JSON object and nothing else:

{{"verdicts": [{{"id": "<session id>", "would_have_caught": true|false,
                "would_have_misfired": true|false, "why": "<one short sentence>"}}]}}
"""

MAX_JUDGED = reasoning.MAX_INPUT_ITEMS


def _example(label: str, entry: dict) -> str:
    short = lambda value: str(value or "")[:80]
    return (f"--- {label} [{entry['id']}] ---\n"
            f"asked: {short(entry['asked'])}\n"
            f"agent produced: {short(entry['produced'])}\n"
            f"developer then said: {short(entry['corrected']) or '(no correction)'}")


def _balanced_examples(positives: List[dict], negatives: List[dict]) -> list:
    """At most eight evidence items total, with both labels represented."""
    queues = [list(positives), list(negatives)]
    chosen: list = []
    while len(chosen) < MAX_JUDGED and any(queues):
        for label, queue in zip(("THE MISTAKE HAPPENED HERE",
                                 "A DIFFERENT PROBLEM WAS DIAGNOSED HERE"), queues, strict=True):
            if queue and len(chosen) < MAX_JUDGED:
                chosen.append((label, queue.pop(0)))
    return chosen


def _invalid_estimate(note: str) -> dict:
    return {"performed": False, "basis": "model-judged", "note": note}


def estimate_probabilistic(rule_text: str, positives: List[dict],
                           negatives: List[dict], engine) -> dict:
    """A rule an agent reads cannot be executed, so it cannot be validated the
    way a check is. Judged against labelled sessions instead, and the result says
    so: this is an estimate from a model, never a demonstration."""
    judged = _balanced_examples(positives, negatives)
    if not judged:
        return _invalid_estimate("no labelled sessions were available to judge against")
    expected = [str(entry.get("id")) for _, entry in judged]
    missing = any(not value or value == "None" for value in expected)
    if missing or len(set(expected)) != len(expected):
        return _invalid_estimate("labelled sessions did not have unique occurrence ids")
    prompt = JUDGE_PROMPT.format(
        rule=rule_text[:600],
        examples="\n\n".join(_example(label, e) for label, e in judged))
    if reasoning.measure_bytes(prompt) > reasoning.MAX_INPUT_BYTES:
        return _invalid_estimate("the bounded judge request exceeded the local input budget")
    try:
        reply = engine.run(prompt)
    except (reasoning.Unavailable, reasoning.Interrupted) as exc:
        return _invalid_estimate(f"the judge was unavailable: {exc}")
    parsed = reasoning.extract_json(reply.text) or {}
    raw_verdicts = parsed.get("verdicts")
    if not isinstance(raw_verdicts, list):
        return _invalid_estimate("the judge returned no complete verdict list")
    ids = [str(v.get("id")) for v in raw_verdicts if isinstance(v, dict)]
    complete = (len(ids) == len(raw_verdicts) == len(expected)
                and len(set(ids)) == len(ids) and set(ids) == set(expected)
                and all(isinstance(v.get("would_have_caught"), bool)
                        and isinstance(v.get("would_have_misfired"), bool)
                        and isinstance(v.get("why"), str)
                        for v in raw_verdicts if isinstance(v, dict)))
    if not complete:
        return _invalid_estimate(
            "the judge response was partial, duplicated, unknown, or malformed")
    verdicts = {str(v["id"]): v for v in raw_verdicts}
    positive_ids = {str(e["id"]) for label, e in judged
                    if label == "THE MISTAKE HAPPENED HERE"}
    negative_ids = set(expected) - positive_ids
    caught = sum(verdicts[value]["would_have_caught"] for value in positive_ids)
    misfired = sum(verdicts[value]["would_have_misfired"] for value in negative_ids)
    n_pos, n_neg = len(positive_ids), len(negative_ids)
    return {
        "performed": bool(verdicts),
        "basis": ("model-judged against captured sessions, not executed; the "
                  "controls are sessions with a different diagnosis, which is a "
                  "proxy for clean work, not a verified one"),
        "provider": getattr(engine, "name", type(engine).__name__),
        "model": reply.model,
        "caught": caught, "of_occurrences": n_pos,
        "misfired": misfired, "of_clean_sessions": n_neg,
        # Deliberately absent: the controls are not verified clean sessions, so
        # a rate computed over them would read as a measurement it is not.
        "false_positive_rate": None,
        "note": ("an estimate over %d labelled session(s); too few to be a rate "
                 "you should trust" % (n_pos + n_neg)) if n_pos + n_neg < 10 else None,
    }


def _not_started(result) -> str:
    return (f"the command could not run in the detached checkout (exit "
            f"{result.exit_code}: {result.detail or 'no output'}); a snapshot has no "
            f"ignored files, so dependencies installed in node_modules or a "
            f"virtualenv are absent and this mechanism cannot be validated here")


def validate(repo, checkout_id: str, argv, overlay: Dict[str, str],
             pre: Optional[State], post: Optional[State],
             timeout: float = probe.DEFAULT_TIMEOUT_S,
             others: Optional[List[Tuple[str, State, State]]] = None,
             creates_runner: bool = False) -> Result:
    """The candidate must reject the state the agent produced and accept the
    state the developer accepted — **because of the change**, not despite it."""
    if not argv:
        return Result(False, notes=["the selected mechanism has no runnable command"])
    if pre is None or post is None:
        return Result(False, notes=["no authentic pre/post pair was available"])
    if pre.tree == post.tree:
        return Result(False, notes=["pre- and post-correction trees are identical"])

    try:
        before = probe.tree_fingerprint(repo, deep=True)
    except probe.FingerprintUnavailable as exc:
        return Result(False, working_tree_unchanged=False,
                      notes=[f"cannot establish the live-tree baseline: {exc}"])
    notes: List[str] = []
    touched = False

    def untouched() -> bool:
        """Content, not just which paths are dirty: a project command can write
        an absolute path into a file that was already modified."""
        if not touched:
            try:
                if probe.tree_fingerprint(repo, deep=True) == before:
                    return True
            except probe.FingerprintUnavailable as exc:
                notes.append(f"cannot verify the live tree after validation: {exc}")
        notes.append("running this mechanism changed the working tree; validation "
                     "must not touch the checkout it is reasoning about")
        return False

    def run(state, with_change):
        nonlocal touched
        try:
            return probe.run(repo, checkout_id, state.tree, argv, timeout=timeout,
                             overlay=overlay if with_change else {})
        except (probe.WorkingTreeTouched, probe.FingerprintUnavailable) as exc:
            touched = True
            notes.append(str(exc))
            return probe.ProbeResult(False, None, None, str(exc))

    # Attribution first: if the mechanism already flags the problematic state
    # without the change, the change is not what catches the mistake and the
    # proposal would be taking credit for a detection it did not add.
    if creates_runner:
        # The command exists only because of this change. Running it beforehand
        # either fails to start or reports a missing target, and reading either
        # as "it already catches the mistake" blocked the plan's third rung for
        # exactly the projects it is meant for.
        baseline = None
        notes.append("the command does not exist until this change adds it, so "
                     "there is no before-state to compare against; attribution "
                     "rests on the change alone")
    else:
        baseline = run(pre, False)
    if baseline is not None and (not baseline.ran or baseline.exit_code is None):
        notes.append(f"the mechanism could not be baselined: {baseline.note}")
        return Result(False, notes=notes,
                      working_tree_unchanged=untouched())
    if baseline is not None and baseline.could_not_run:
        notes.append(_not_started(baseline))
        return Result(False, notes=notes, working_tree_unchanged=untouched())
    if baseline is not None and baseline.exit_code != 0:
        notes.append(f"the command already fails on the problematic state without this "
                     f"change (exit {baseline.exit_code}: {baseline.detail or 'no output'}), "
                     f"so it cannot show that the change is what catches the mistake")
        return Result(True, None, None, False,
                      untouched(), [], notes)

    problematic = run(pre, True)
    repaired = run(post, True)
    if not problematic.ran or not repaired.ran:
        notes.append(f"the mechanism could not be executed: "
                     f"{problematic.note or repaired.note}")
        return Result(False, notes=notes,
                      working_tree_unchanged=untouched())
    if problematic.could_not_run or repaired.could_not_run:
        notes.append(_not_started(problematic if problematic.could_not_run else repaired))
        return Result(False, notes=notes, working_tree_unchanged=untouched())
    if problematic.exit_code is None or repaired.exit_code is None:
        notes.append("the mechanism timed out; cost makes it unsuitable as a gate")
        return Result(True, None, None, None,
                      untouched(), [], notes)

    rejects = problematic.exit_code != 0
    accepts = repaired.exit_code == 0
    notes.append(f"baseline (no change) on the problematic state "
                 f"{'exit=0' if baseline is not None else 'not runnable yet'}, "
                 f"with the change exit={problematic.exit_code} "
                 f"({problematic.duration_ms}ms); repaired state "
                 f"exit={repaired.exit_code} ({repaired.duration_ms}ms)")
    if not rejects:
        notes.append("the mechanism does not flag the state the agent actually "
                     "produced, so it would not have caught this mistake")
    if not accepts:
        notes.append("the mechanism also flags the state the developer accepted, "
                     "so it would fire on correct work")

    # Recurrence is what justified acting, so every occurrence is checked, not
    # only the one that produced the diagnosis.
    occurrences: List[Occurrence] = []
    for item in (others or []):
        if len(item) == 4:
            owner, session_id, other_pre, other_post = item
        else:
            session_id, other_pre, other_post = item
            owner = None
        occurrence = Occurrence(session_id, owner)
        if other_pre is None or other_post is None or other_pre.tree == other_post.tree:
            occurrence.note = "no usable pre/post pair"
            occurrences.append(occurrence)
            continue
        other_bad = run(other_pre, True)
        other_good = run(other_post, True)
        if other_bad.exit_code is not None and not other_bad.could_not_run:
            occurrence.caught = other_bad.exit_code != 0
        if other_good.exit_code is not None and not other_good.could_not_run:
            occurrence.false_positive = other_good.exit_code != 0
        occurrences.append(occurrence)

    result = Result(True, rejects, accepts, True,
                    untouched(), occurrences, notes)
    caught, evaluated = result.coverage
    if evaluated:
        result.notes.append(f"catches {caught} of {evaluated} other captured "
                            f"occurrence(s) of this pattern")
    unevaluated = sum(1 for o in occurrences
                      if o.caught is None or o.false_positive is None)
    if unevaluated:
        result.notes.append(f"{unevaluated} other occurrence(s) could not be evaluated "
                            f"(no identified correction, or a run that did not finish)")
    if result.false_positives:
        result.notes.append(f"fires on {result.false_positives} state(s) the "
                            f"developer had already accepted")
    return result
