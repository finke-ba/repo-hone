"""Evidence selection for retrospective reasoning (ARCH §12).

Two limits, whichever is reached first: 8,000 bytes of model input and 8 items.
When the selected evidence does not justify a diagnosis the answer is
INSUFFICIENT_EVIDENCE — never automatic context expansion.

Selection is **diverse before it is deep**. Taking candidates in kind order lets
a long session fill every slot with prompts and starve the evidence of what the
agent actually did, which reads as a clean INSUFFICIENT_EVIDENCE rather than the
ranking bug it is. Kinds are therefore filled round-robin in priority order.

Selection does not classify. Whether a developer change is a correction is the
model's question (ARCH §20); this layer only ranks what is worth sending.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from . import artifacts, bootstrap, fingerprint, gitcmd, reasoning, redact, toolinput

MAX_BYTES = reasoning.MAX_INPUT_BYTES
MAX_ITEMS = reasoning.MAX_INPUT_ITEMS
DIFF_CHARS = 1000
INSTRUCTION_CHARS = 800
TEXT_CHARS = 800
CONSULTED_TURN_CHARS = 320
CONSULTED_CALL_CHARS = 90
CONSULTED_TURNS = 3

# `consulted` is what separates a rule the agent never saw (MISSING_CONTEXT)
# from one it read and ignored (AGENT_REASONING_ERROR): identical trees,
# opposite improvements.
KIND_PRIORITY = ("rule_statement", "correction", "consulted", "tool_failure", "diff",
                 "history", "turn_prompt", "project_instruction", "assistant_message")
TARGET_FIELDS = ("file_path", "notebook_path", "path", "command", "pattern", "url")

INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md", "CONVENTIONS.md",
                     "docs/CONVENTIONS.md", ".repohone/rules.md")

# Bootstrap candidates that are something the project already undid, as opposed
# to rules it already wrote down (those arrive as project_instruction).
HISTORY_SOURCES = ("revert", "fixup")
HISTORY_ITEMS = 2
HISTORY_MIN_SHARED = 2


def measure_bytes(text: str) -> int:
    return reasoning.measure_bytes(text)


@dataclass
class Item:
    kind: str
    ref: str
    text: str
    meta: Optional[dict] = None

    @property
    def bytes(self) -> int:
        return measure_bytes(self.text)


@dataclass
class Candidate:
    """Text is produced only if the candidate is actually chosen: building a diff
    costs a git subprocess, and most candidates never fit the budget."""
    kind: str
    ref: str
    load: Callable[..., Optional[str]]
    meta: Optional[dict] = None


@dataclass
class Selection:
    items: List[Item] = field(default_factory=list)
    skipped: int = 0

    @property
    def bytes(self) -> int:
        return sum(i.bytes for i in self.items)

    def refs(self) -> List[dict]:
        # `estimated_tokens` is the v1 field name; the value is bytes.
        return [{"kind": i.kind, "ref": i.ref, "estimated_tokens": i.bytes}
                for i in self.items]


def clean_text(text: Optional[str], root) -> str:
    """The one minimization boundary for any captured text sent to a model."""
    if not text:
        return ""
    cleaned, _ = redact.redact(text)
    return toolinput.scrub_paths(cleaned, root)


# Kept private alias for the existing evidence loaders.
_clean = clean_text


def _turn_diff(root, prompt_snap, stop_snap) -> Optional[str]:
    """A bounded diff, never a whole snapshot (§12)."""
    if not prompt_snap or not stop_snap:
        return None
    out = gitcmd.run(["diff", "--unified=3", "--no-color",
                      prompt_snap["tree"], stop_snap["tree"]], root, check=False)
    return out[:DIFF_CHARS] if out.strip() else None


def _tool_events(rec: dict, turn: dict) -> List[dict]:
    tools = ((rec.get("extensions") or {}).get("tool_events") or {}).get("turns") or {}
    return (tools.get(turn.get("logical_turn_id")) or {}).get("events") or []


def _failure(event: dict) -> Optional[str]:
    """How a call failed, or None. Claude reports only successes, so a call with
    no result is the failure (§32)."""
    return None if event.get("outcome_observed") else "no result"


def _consulted(rec: dict, turns: List[dict], root) -> Optional[str]:
    """Per turn, oldest first: how many calls and how many failed, then the calls,
    failures first. Each call is cleaned before it is cut: half a path no longer
    matches the scrubber."""
    lines = []
    for turn in sorted(turns, key=lambda t: t.get("index", 0)):
        events = _tool_events(rec, turn)
        if not events:
            continue
        failed, other, seen = [], [], set()
        for event in events:
            detail = event.get("detail") or {}
            target = next((detail[f] for f in TARGET_FIELDS if detail.get(f)), None)
            label = (f"{event.get('tool')} {_clean(str(target), root)}" if target
                     else str(event.get("tool")))
            if len(label) > CONSULTED_CALL_CHARS:
                label = label[:CONSULTED_CALL_CHARS - 1] + "…"
            why = _failure(event)
            if why:
                failed.append(f"{label} ({why})")
            elif label not in seen:
                seen.add(label)
                other.append(label)
        by_tool = ", ".join(f"{name} {count}" for name, count in
                            Counter(event.get("tool") for event in events).most_common())
        head = (f"turn {turn['index']}: {len(events)} call{'s' if len(events) != 1 else ''} "
                f"({by_tool}), {len(failed)} failed")
        line, shown = head, 0
        for label in failed + other:
            if len(line) + len(label) + 2 > CONSULTED_TURN_CHARS:
                break
            line += ("; " if shown else ": ") + label
            shown += 1
        if shown < len(failed) + len(other):
            line += f" … +{len(failed) + len(other) - shown} more"
        lines.append(line)
    return "\n".join(lines) or None


def _failed_tool_calls(rec: dict, turn: dict) -> List[tuple]:
    """A check the agent ran that failed is the strongest available signal of a
    verification gap."""
    out = []
    for event in _tool_events(rec, turn):
        why = _failure(event)
        if not why:
            continue
        detail = event.get("detail") or {}
        target = detail.get("command") or detail.get("file_path") or ""
        if target:
            said = "produced no result" if why == "no result" else why
            out.append((event["seq"], f"{event.get('tool')}: {target} — {said}"))
    return out


def session_tree(rec: dict) -> Optional[str]:
    """The project as it stood when the session began. Instruction files must be
    read from here, never from the working tree: a rule written *after* the
    session would otherwise be presented as context the agent had, turning a
    MISSING_CONTEXT failure into an apparent AGENT_REASONING_ERROR."""
    for turn in rec.get("turns") or []:
        for event in turn.get("prompt_events") or []:
            snapshot = event.get("snapshot")
            if snapshot and snapshot.get("tree"):
                return snapshot["tree"]
    return None


_NOT_TERMS = {"must", "never", "always", "should", "please", "use", "not", "don",
              "no", "yes", "instead", "again", "make", "sure", "just", "only"}


def _terms(text: Optional[str]) -> List[str]:
    return [t for t in fingerprint._tokens(text or "") if t not in _NOT_TERMS]


def _historical_instruction(root, tree: str, name: str,
                            terms: Sequence[str] = ()) -> Optional[str]:
    """The lines that bear on the correction, not merely the first ones: a rule
    past the head would otherwise read as a rule the project never had."""
    content = gitcmd.run(["show", f"{tree}:{name}"], root, check=False)
    if not content.strip():
        return None
    if len(content) <= INSTRUCTION_CHARS:
        return content

    def score(line):
        return sum(1 for word in set(fingerprint._tokens(line))
                   if any(fingerprint._equivalent(word, term) for term in terms))

    lines = content.splitlines()
    ranked = sorted(((score(line), i) for i, line in enumerate(lines)),
                    key=lambda item: (-item[0], item[1]))
    header = (f"(excerpt: 000 of {len(lines)} lines, chosen for the words they "
              f"share with the correction)")
    chosen, used = [], len(header)
    for points, index in ranked:
        if points == 0:
            break
        if used + len(lines[index]) + 1 > INSTRUCTION_CHARS:
            continue
        chosen.append(index)
        used += len(lines[index]) + 1
    if not chosen:
        return content[:INSTRUCTION_CHARS]
    header = header.replace("000", str(len(chosen)), 1)
    return "\n".join([header] + [lines[i] for i in sorted(chosen)])


def _history(rec: dict, root, terms: List[str]) -> List[Candidate]:
    """Reverts and fix-ups mined by `repohone bootstrap` that share words with the
    correction. Only commits the session started from count: a revert made after
    it may be this very correction, counted twice."""
    repository = rec.get("repository") or {}
    head, checkout = repository.get("start_head"), repository.get("checkout_id")
    if not head or not checkout or not terms:
        return []
    try:
        mined = bootstrap.load(checkout) or {}
    except artifacts.ArtifactError:
        return []
    scored = []
    for position, found in enumerate(mined.get("candidates") or []):
        if found.get("source") not in HISTORY_SOURCES:
            continue
        words = set(fingerprint._tokens(found.get("statement") or ""))
        shared = sum(1 for word in words
                     if any(fingerprint._equivalent(word, term) for term in terms))
        if shared >= HISTORY_MIN_SHARED:
            scored.append((-shared, position, found))
    out = []
    for _, _, found in sorted(scored, key=lambda item: item[:2]):
        commit = next((sha for sha in found.get("evidence") or []
                       if _started_from(root, sha, head)), None)
        if commit is None:
            continue
        meta = {"candidate_id": found["id"], "source": found["source"], "commit": commit,
                "statement": _clean(found.get("statement"), root)[:TEXT_CHARS]}
        out.append(Candidate("history", found["id"],
                             lambda m=meta: f"{m['commit']} ({m['source']}): {m['statement']}",
                             meta))
        if len(out) >= HISTORY_ITEMS:
            break
    return out


def _started_from(root, sha: str, head: str) -> bool:
    return (isinstance(sha, str) and all(c in "0123456789abcdef" for c in sha)
            and 4 <= len(sha) <= 40
            and gitcmd.ok(["merge-base", "--is-ancestor", sha, head], root))


def validated_turn(rec: dict, corrective_turn: Optional[int]) -> Optional[int]:
    """A boundary naming no turn in this session would silently steer nothing."""
    if corrective_turn is None:
        return None
    if isinstance(corrective_turn, bool) or not isinstance(corrective_turn, int):
        raise ValueError(f"corrective turn must be a turn number; got {corrective_turn!r}")
    if not any(t.get("index") == corrective_turn for t in rec.get("turns") or []):
        raise ValueError(f"this session has no turn {corrective_turn}")
    return corrective_turn


def _rank(turns: List[dict], corrective_turn: Optional[int]) -> List[dict]:
    """Most-relevant turn first.

    The budget runs out long before the turns do, so this order decides what the
    model never sees; without a named turn it can only guess, and guesses recent.
    """
    # The correction, then the turn that produced what was corrected.
    named = {} if corrective_turn is None else {corrective_turn: 2,
                                                corrective_turn - 1: 1}

    def key(turn):
        return (named.get(turn.get("index"), 0),
                bool(turn.get("workspace_changed_before_turn")),
                turn.get("index", 0))

    return sorted(turns, key=key, reverse=True)


def _buckets(rec: dict, root, rule_statement: Optional[str],
             corrective_turn: Optional[int] = None) -> Dict[str, List[Candidate]]:
    buckets: Dict[str, List[Candidate]] = {k: [] for k in KIND_PRIORITY}
    turns = rec.get("turns") or []
    label = (rec.get("labels") or {}).get("developer_label")

    if rule_statement:
        text = _clean(rule_statement, root)[:TEXT_CHARS]
        buckets["rule_statement"].append(
            Candidate("rule_statement", "developer", lambda t=text: t))

    ranked = _rank(turns, corrective_turn)
    last_index = turns[-1]["index"] if turns else None

    consulted = _consulted(rec, ranked[:CONSULTED_TURNS], root)
    if consulted:
        text = _clean(consulted, root)
        buckets["consulted"].append(Candidate(
            "consulted", "tools/turn/" + ",".join(
                str(t["index"]) for t in sorted(ranked[:CONSULTED_TURNS],
                                                key=lambda t: t["index"])),
            lambda t=text: t))

    focus = [t for t in turns if t["index"] == corrective_turn] or ranked[:1]
    terms = _terms(rule_statement) + [
        term for turn in focus for event in turn.get("prompt_events") or []
        for term in _terms((event.get("content") or {}).get("text"))]

    for turn in ranked:
        for event in turn.get("prompt_events") or []:
            raw = ((event.get("content") or {}).get("text") or "")
            if not raw.strip():
                continue
            # A session-level CORRECTION label describes the developer's latest
            # statement, not every prompt in the session (LEARNING_PLAN §7).
            # Nothing labels turns before Phase 7, so a turn the developer named
            # is the only correction signal this phase has.
            kind = ("correction"
                    if turn["index"] == corrective_turn
                    or (label == "CORRECTION" and turn["index"] == last_index)
                    else "turn_prompt")
            text = _clean(raw, root)[:TEXT_CHARS]
            buckets[kind].append(
                Candidate(kind, f"turn/{turn['index']}/prompt/{event['ordinal']}",
                          lambda t=text: t))

        for seq, description in _failed_tool_calls(rec, turn):
            buckets["tool_failure"].append(
                Candidate("tool_failure", f"turn/{turn['index']}/tool/{seq}",
                          lambda d=description: d))

        prompts, stops = turn.get("prompt_events") or [], turn.get("stop_events") or []
        if prompts and stops:
            buckets["diff"].append(Candidate(
                "diff", f"turn/{turn['index']}/diff",
                lambda p=prompts[0], s=stops[-1]: _clean(
                    _turn_diff(root, p.get("snapshot"), s.get("snapshot")), root)))

        for event in stops:
            message = ((event.get("last_assistant_message") or {}).get("text") or "")
            if message.strip():
                text = _clean(message, root)[:TEXT_CHARS]
                buckets["assistant_message"].append(
                    Candidate("assistant_message",
                              f"turn/{turn['index']}/stop/{event['ordinal']}",
                              lambda t=text: t))

    buckets["history"].extend(_history(rec, root, terms))

    tree = session_tree(rec)
    for name in INSTRUCTION_FILES:
        if tree is None:
            # No captured state means no historical instruction evidence. The
            # working tree is not a substitute for what the session actually saw.
            break
        buckets["project_instruction"].append(
            Candidate("project_instruction", f"{name}@{tree[:12]}",
                      lambda n=name, t=tree: _clean(
                          _historical_instruction(root, t, n, terms), root)))
    return buckets


def select(rec: dict, root, rule_statement: Optional[str] = None,
           max_items: int = MAX_ITEMS, max_bytes: int = MAX_BYTES,
           corrective_turn: Optional[int] = None) -> Selection:
    """Round-robin across kinds in priority order, so every kind present is
    represented before any kind takes a second slot."""
    buckets = _buckets(rec, root, rule_statement, corrective_turn)
    selection = Selection()

    while len(selection.items) < max_items:
        progressed = False
        for kind in KIND_PRIORITY:
            if len(selection.items) >= max_items:
                break
            bucket = buckets[kind]
            while bucket:
                candidate = bucket.pop(0)
                text = candidate.load()
                if not text or not text.strip():
                    continue
                item = Item(candidate.kind, candidate.ref, text, candidate.meta)
                if selection.bytes + item.bytes > max_bytes:
                    selection.skipped += 1
                    continue
                selection.items.append(item)
                progressed = True
                break
        if not progressed:
            break

    selection.skipped += sum(len(b) for b in buckets.values())
    return selection


def is_sufficient(selection: Selection) -> bool:
    """A diagnosis needs the developer's intent and something the agent did.
    Project instructions alone describe the project, not a failure."""
    kinds = {i.kind for i in selection.items}
    has_intent = bool(kinds & {"rule_statement", "correction", "turn_prompt"})
    has_behaviour = bool(kinds & {"diff", "tool_failure", "assistant_message"})
    return has_intent and has_behaviour
