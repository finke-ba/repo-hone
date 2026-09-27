"""Failure fingerprints and recurrence (LEARNING_PLAN §9).

The taxonomy answers *what kind of problem*; a fingerprint answers *which
recurring repository-specific problem*. Recurrence is what lets a diagnosis
answer "is this genuinely a project-level pattern?" — without it every
hypothesis rests on a single session.

Matching has to fail in both directions at once. A forked series undercounts and
merely looks unconvincing; an **over-merged series invents a recurring pattern
that does not exist**, and recurrence is what a proposal leans on. So area and
behavior must each clear their own bar, and the behavior bar is the strict one
because behavior is what distinguishes two problems in the same area.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from . import artifacts, paths, record

AREA_THRESHOLD = 0.5
BEHAVIOR_THRESHOLD = 0.8
_MIN_PREFIX = 5
SUMMARY_MAX_BYTES = 1200
SUMMARY_FIELD_CHARS = 200

_WORD = re.compile(r"[a-z0-9]+")
# "not" is deliberately absent: dropping it makes a rule and its negation the
# same fingerprint.
_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "are",
         "must", "with", "by", "that", "this", "its", "it"}


@dataclass
class Fingerprint:
    fingerprint_id: str
    cls: str
    area: str
    behavior: str
    sessions: int = 0
    checkouts: int = 1
    developers: int = 0
    complete: bool = True
    historical: int = 0
    _session_keys: set = field(default_factory=set, repr=False, compare=False)
    _checkout_keys: set = field(default_factory=set, repr=False, compare=False)
    _developer_keys: set = field(default_factory=set, repr=False, compare=False)
    _history_keys: set = field(default_factory=set, repr=False, compare=False)

    def label(self) -> str:
        return f"{self.area} / {self.behavior}"


class IncompleteHistory(RuntimeError):
    """Cross-checkout recurrence could not be established completely."""


class Scan(list):
    """List-compatible result that carries any unavailable sibling databases."""

    def __init__(self, values=(), problems=()):
        super().__init__(values)
        self.problems = tuple(problems)

    @property
    def complete(self) -> bool:
        return not self.problems

    def require_complete(self):
        if self.problems:
            raise IncompleteHistory(
                "recurrence history is incomplete: " + "; ".join(self.problems[:5]))
        return self


def normalize(text: Optional[str]) -> str:
    """Labels arrive as prose, kebab-case or snake_case; recurrence must not
    fork because one session said 'handler-direct-db' and the next said
    'handler direct database'."""
    if not text:
        return ""
    return " ".join(w for w in _WORD.findall(str(text).lower()) if w not in _STOP)


def _tokens(text: str) -> List[str]:
    return normalize(text).split()


# Abbreviations are not prefixes ("db" is not the start of "database"), so the
# common ones are listed rather than guessed at.
_SYNONYMS = {"db": "database", "cfg": "config", "configuration": "config",
             "auth": "authentication", "repo": "repository", "env": "environment",
             "impl": "implementation", "fn": "function", "func": "function",
             "val": "value", "idx": "index", "arg": "argument", "param": "parameter",
             "req": "request", "res": "response", "resp": "response", "err": "error"}


def _canon(token: str) -> str:
    return _SYNONYMS.get(token, token)


def _shared_prefix(a: str, b: str) -> int:
    count = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        count += 1
    return count


def _equivalent(a: str, b: str) -> bool:
    """`db` and `database` are one token, and so are `validate` and `validation`;
    `length` and `null` are not."""
    a, b = _canon(a), _canon(b)
    if a == b:
        return True
    return min(len(a), len(b)) >= _MIN_PREFIX and _shared_prefix(a, b) >= _MIN_PREFIX


def similarity(a: str, b: str) -> float:
    """Jaccard over tokens, where an abbreviation counts as its expansion."""
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    remaining = list(right)
    matched = 0
    for token in left:
        for index, candidate in enumerate(remaining):
            if _equivalent(token, candidate):
                matched += 1
                remaining.pop(index)
                break
    union = len(left) + len(right) - matched
    return matched / union if union else 0.0


def new_id(repository_id: Optional[str], cls: str, area: str, behavior: str) -> str:
    """Every field `match` distinguishes on. Omitting the class gave two
    explicitly different root causes one durable identity, and the second
    diagnosis then contradicted the first."""
    key = "\x00".join((repository_id or "-", normalize(cls),
                       normalize(area), normalize(behavior)))
    return "fp_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def match(existing: List[Fingerprint], cls: str, area: str,
          behavior: str) -> Optional[Fingerprint]:
    """Same class, and area and behavior each clear their own bar. Averaging the
    two lets a shared area carry a behavior that means something else entirely."""
    best, best_score = None, 0.0
    for candidate in existing:
        if candidate.cls != cls:
            continue
        area_score = similarity(candidate.area, area)
        behavior_score = similarity(candidate.behavior, behavior)
        if area_score < AREA_THRESHOLD or behavior_score < BEHAVIOR_THRESHOLD:
            continue
        score = area_score + behavior_score
        if score > best_score:
            best, best_score = candidate, score
    return best


def _checkout_inventory() -> List[str]:
    """Every checkout directory. A missing or damaged state.db holds no
    diagnoses, so it is never a reason to forget the checkout."""
    names = []
    for entry in paths.entries(paths.data_dir() / "checkouts") or []:
        try:
            if not stat.S_ISDIR(os.stat(entry).st_mode):
                continue          # a stray file cannot hold a checkout's evidence
        except FileNotFoundError:
            continue
        except OSError:
            pass                  # cannot tell: keep it, and let reading it say why
        names.append(entry.name)
    return names


def _is_clone_of(checkout_id: str, repository_id: str):
    """``(verdict, problems)``: True, False, or None for undecidable.

    True if any record names this repository; otherwise None if any record is
    unreadable or carries no repository id; False only when every record names
    another. Exclusion is the one answer an undercount never recovers from.
    """
    try:
        listed = artifacts.members(paths.records_dir(checkout_id))
    except OSError as exc:
        return None, [f"checkout {checkout_id} records: {type(exc).__name__}: {exc}"]
    unreadable, unidentified, elsewhere = [], False, False
    for candidate in listed or []:
        if not candidate.name.startswith("rh_"):
            continue
        try:
            seen = artifacts.load(candidate, "session",
                                  expected_checkout_id=checkout_id,
                                  expected_artifact_id=candidate.stem)
            if seen is None:
                raise artifacts.MalformedArtifact(f"{candidate.name}: vanished while read")
        except (artifacts.ArtifactError, OSError) as exc:
            unreadable.append(f"checkout {checkout_id} session {candidate.name}: "
                              f"{type(exc).__name__}: {exc}")
            continue
        found = (seen.get("repository") or {}).get("id")
        if found == repository_id:
            return True, []
        if found is None:
            unidentified = True
        else:
            elsewhere = True
    if unreadable:
        return None, unreadable
    if unidentified or not elsewhere:
        return None, []
    return False, []


def _related_checkouts(checkout_id: str, repository_id: Optional[str]):
    """The checkouts whose evidence can belong to this repository.

    One selector for every recurrence query: when `known` and `occurrences_for`
    scoped differently, unrelated corruption blocked one and not the other.
    """
    if repository_id is None:
        # ``None`` is the absence of a cross-checkout identity, not an id shared
        # by every local-only project on this machine.
        return [checkout_id], []
    try:
        inventory = _checkout_inventory()
    except OSError as exc:
        return [checkout_id], [f"checkout inventory: {type(exc).__name__}: {exc}"]
    if checkout_id not in inventory:
        inventory.insert(0, checkout_id)
    selected, problems = [], []
    for name in inventory:
        if name == checkout_id:
            selected.append(name)
            continue
        # Another project's checkout holds no recurrence evidence and no reason
        # to refuse this analysis — and naming it would leak its id into this record.
        verdict, trouble = _is_clone_of(name, repository_id)
        problems.extend(trouble)
        if verdict is not False:
            selected.append(name)
    return selected, problems


def _occurrence_of(checkout_id: str, path: Path) -> Optional[dict]:
    """One successful diagnosis as a recurrence occurrence; None if it did not
    succeed. Raises when the artifact cannot be read, or, for a diagnosis older
    than 1.24, when its session cannot."""
    diagnosis = artifacts.load(path, "diagnosis", expected_checkout_id=checkout_id,
                               expected_artifact_id=path.stem)
    if not diagnosis or diagnosis.get("outcome") != "SUCCESS":
        return None
    mark = diagnosis.get("fingerprint") or {}
    session_id = diagnosis["session_id"]
    if "repository_id" in diagnosis:
        repository_id, developer_id = diagnosis["repository_id"], diagnosis["developer_id"]
    else:
        session = record.load(checkout_id, session_id)
        if session is None:
            raise artifacts.MalformedArtifact(f"{path.name}: session {session_id!r} is missing")
        repository_id = (session.get("repository") or {}).get("id")
        developer_id = (session.get("developer") or {}).get("id")
    return {
        "fingerprint_id": mark["fingerprint_id"],
        "session_id": session_id,
        "repository_id": repository_id,
        "developer_id": developer_id,
        "class": (diagnosis.get("root_cause") or {})["class"],
        "area": mark["area"],
        "behavior": mark["behavior"],
        "at": diagnosis["created_at"],
        "history": [link["candidate_id"] for link in diagnosis.get("history") or []],
    }


def _canonical_diagnoses(checkout_id: str):
    """Successful occurrences in one checkout, keyed by diagnosis id. The files
    are the only record of recurrence: a deleted diagnosis stops counting, and
    one that cannot be read makes the history incomplete."""
    directory = paths.checkout_dir(checkout_id) / "diagnoses"
    found: dict = {}
    problems: list = []
    try:
        listed = artifacts.members(directory)
    except OSError as exc:
        return found, [f"checkout {checkout_id} diagnoses: {type(exc).__name__}: {exc}"]
    for candidate in listed or []:
        try:
            item = _occurrence_of(checkout_id, candidate)
        except (artifacts.ArtifactError, OSError, KeyError, TypeError) as exc:
            problems.append(
                f"checkout {checkout_id} diagnosis {candidate.name}: "
                f"{type(exc).__name__}: {exc}")
            continue
        if item is not None:
            found[candidate.stem] = item
    return found, problems


def diagnosis_status(checkout_id: str) -> Tuple[int, List[str]]:
    """How many of this checkout's diagnoses count, and which cannot be read."""
    found, problems = _canonical_diagnoses(checkout_id)
    return len(found), problems


def known(checkout_id: str, repository_id: Optional[str]) -> Scan:
    """Aggregated across every local checkout of the same repository.

    ARCH §15 keeps mutable telemetry checkout-scoped but allows analysis to group
    by repository identity, and one developer's clones and worktrees are all one
    project. **Other developers' machines are not readable**, so this is local
    recurrence, not team-wide recurrence.
    """
    merged: Dict[str, Fingerprint] = {}
    sessions: Dict[str, set] = {}
    developers: Dict[str, set] = {}
    checkouts: Dict[str, set] = {}
    history: Dict[str, set] = {}

    names, selection_problems = _related_checkouts(checkout_id, repository_id)
    problems = list(selection_problems)
    for name in names:
        canonical, canonical_problems = _canonical_diagnoses(name)
        problems.extend(canonical_problems)
        for item in canonical.values():
            if item["repository_id"] != repository_id:
                continue
            fid = item["fingerprint_id"]
            merged.setdefault(fid, Fingerprint(fid, item["class"], item["area"],
                                               item["behavior"]))
            checkouts.setdefault(fid, set()).add(name)
            sessions.setdefault(fid, set()).add((name, item["session_id"]))
            if item["developer_id"]:
                developers.setdefault(fid, set()).add(item["developer_id"])
            history.setdefault(fid, set()).update(item["history"])

    out = []
    for fid, mark in merged.items():
        mark.sessions = len(sessions.get(fid, ()))
        mark.checkouts = len(checkouts.get(fid, ()))
        mark.developers = len(developers.get(fid, ()))
        mark.historical = len(history.get(fid, ()))
        mark._session_keys = set(sessions.get(fid, ()))
        mark._checkout_keys = set(checkouts.get(fid, ()))
        mark._developer_keys = set(developers.get(fid, ()))
        mark._history_keys = set(history.get(fid, ()))
        out.append(mark)
    return Scan(sorted(out, key=lambda f: f.sessions, reverse=True), problems)


def sessions_for(checkout_id: str, fingerprint_id: str) -> List[str]:
    """Every session this pattern was seen in, across local checkouts. Recurrence
    justified acting, so validation gets to use all of it."""
    occurrences = occurrences_for(checkout_id, fingerprint_id).require_complete()
    return sorted({session_id for _, session_id in occurrences})


def occurrences_for(checkout_id: str, fingerprint_id: str) -> Scan:
    """Checkout and session form the identity of one captured occurrence."""
    canonical, problems = _canonical_diagnoses(checkout_id)
    problems = list(problems)
    own = next((item for item in canonical.values()
                if item["fingerprint_id"] == fingerprint_id), None)
    if own is None:
        # Unknown here means no occurrences — unless this checkout's own evidence
        # could not be read, which is exactly when a known pattern looks unknown.
        return Scan((), problems)
    repository_id = own["repository_id"]
    names, selection_problems = _related_checkouts(checkout_id, repository_id)
    problems.extend(selection_problems)
    found: set = set()
    for name in names:
        items, trouble = ((canonical, []) if name == checkout_id
                          else _canonical_diagnoses(name))
        problems.extend(trouble)
        found.update((name, item["session_id"]) for item in items.values()
                     if item["fingerprint_id"] == fingerprint_id)
    return Scan(sorted(found), list(dict.fromkeys(problems)))


def record_occurrence(checkout_id: str, repository_id: Optional[str], cls: str,
                      area: str, behavior: str, session_id: str, diagnosis_id: str,
                      at: str, developer_id: Optional[str] = None,
                      publish: Optional[Callable[[Fingerprint, bool, Fingerprint], None]] = None,
                      history: Tuple[str, ...] = ()) -> Tuple[Fingerprint, bool]:
    """Match against every local occurrence, then publish one diagnosis.

    The callback must atomically save and validate the diagnosis artifact; the
    published file is the occurrence, so nothing else is written.
    """
    if publish is None:
        raise TypeError("record_occurrence requires a canonical diagnosis publisher")
    # Repository-scoped: two clones diagnosing one pattern at once would
    # otherwise each start their own series.
    scope = repository_id or f"checkout:{checkout_id}"
    lock_id = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
    lock_path = paths.data_dir() / "locks" / "fingerprints" / lock_id
    with record._locked(lock_path):
        # Incomplete history can only undercount or fork a series, never invent
        # one, so the occurrence is recorded and the count says it is a floor.
        existing = known(checkout_id, repository_id)
        hit = match(existing, cls, area, behavior)
        created = hit is None
        if hit is None:
            hit = Fingerprint(new_id(repository_id, cls, area, behavior),
                              cls, area, behavior)

        session_keys = set(getattr(hit, "_session_keys", ()))
        checkout_keys = set(getattr(hit, "_checkout_keys", ()))
        developer_keys = set(getattr(hit, "_developer_keys", ()))
        history_keys = set(getattr(hit, "_history_keys", ())) | set(history)
        session_keys.add((checkout_id, session_id))
        checkout_keys.add(checkout_id)
        if developer_id:
            developer_keys.add(developer_id)
        projected = Fingerprint(hit.fingerprint_id, hit.cls, hit.area, hit.behavior,
                                sessions=len(session_keys),
                                checkouts=len(checkout_keys),
                                developers=len(developer_keys),
                                complete=existing.complete,
                                historical=len(history_keys))
        publish(hit, created, projected)

        published = paths.checkout_dir(checkout_id) / "diagnoses" / f"{diagnosis_id}.json"
        try:
            source = _occurrence_of(checkout_id, published)
        except (artifacts.ArtifactError, OSError, KeyError, TypeError) as exc:
            raise artifacts.MalformedArtifact(
                f"{published.name}: {type(exc).__name__}: {exc}") from exc
        if source is None or source["fingerprint_id"] != hit.fingerprint_id \
                or source["session_id"] != session_id \
                or source["repository_id"] != repository_id \
                or source["developer_id"] != developer_id \
                or source["class"] != cls or source["area"] != hit.area \
                or source["behavior"] != hit.behavior or source["at"] != at \
                or source["history"] != list(history):
            raise artifacts.MalformedArtifact(
                f"{published.name}: published artifact does not match occurrence")
        return projected, created


def summarise(existing: List[Fingerprint]) -> str:
    """What the model is shown so it reuses a label instead of inventing a
    near-duplicate."""
    if not existing:
        return "(none recorded for this repository yet)"
    lines: List[str] = []
    used = 0
    for fp in existing[:12]:
        seen = "1 session" if fp.sessions == 1 else f"{fp.sessions} sessions"
        who = "" if fp.developers <= 1 else f", {fp.developers} developers"
        line = (f"- {fp.cls} | area: {fp.area[:SUMMARY_FIELD_CHARS]} | behavior: "
                f"{fp.behavior[:SUMMARY_FIELD_CHARS]} ({seen}{who})")
        encoded = line.encode("utf-8")
        separator = 1 if lines else 0
        if used + separator + len(encoded) > SUMMARY_MAX_BYTES:
            break
        lines.append(line)
        used += separator + len(encoded)
    if not lines:
        return "(known fingerprints omitted because their labels exceed the input budget)"
    return "\n".join(lines)
