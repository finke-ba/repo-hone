"""Phase 6: historical bootstrap.

Mines what the project has already written down and already corrected —
instruction files, revert commits, fix-up commits, project check scripts — and
**produces candidates only**. A convention is never adopted merely because it is
frequent: frequency is evidence, not consent.

Deterministic and local by construction. No model runs here (ARCH §1.1); this is
indexing, and interpretation happens when the developer asks for it.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import CONTRACT_VERSION, artifacts, gitcmd, paths, redact, toolinput

SCHEMA_ID = "repohone.bootstrap/v1"
SCHEMA_VERSION = 1

INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md", "CONVENTIONS.md",
                     "docs/CONVENTIONS.md", ".cursorrules", ".repohone/rules.md")

# A normative sentence, not prose about the project.
NORMATIVE = re.compile(
    r"\b(must not|must|never|always|do not|don't|should not|shouldn't|required to|"
    r"forbidden|only ever|prefer)\b", re.I)

REVERT = re.compile(r"^Revert\s+\"?(.+?)\"?$", re.I)
FIXUP = re.compile(r"^(fixup!|squash!|oops|typo\b|fix typo|hotfix\b|correct\b)", re.I)

MAX_PER_SOURCE = 25
REVIEW_SET = 5
GH_TIMEOUT_S = 20.0
GH_PAGE = 100
GH_LIMIT = 200

# A correction the project kept making outranks a rule already written down:
# an instruction file is context the agent already had, so its presence there is
# evidence the rule is known, not evidence it is missing.
SOURCE_RANK = {"revert": 0, "ci-failure": 1, "pr-review": 2, "fixup": 3,
               "custom-check": 4, "instruction-file": 5}
RANKING_BASIS = ("one candidate from each source in turn — reverts, CI failures, "
                 "review comments, fix-ups, custom checks, then rules already written "
                 "down — the most repeated first within a source")
# A fix-up subject made only of these says nothing about what was wrong.
CONTENT_FREE = {"typo", "typos", "fix", "fixes", "fixed", "oops", "minor", "small",
                "cleanup", "wip", "again", "more", "stuff", "things", "tweak", "tweaks"}
MAX_STATEMENT = 300
HISTORY_LIMIT = 400


@dataclass
class Candidate:
    source: str
    statement: str
    evidence: List[str]
    occurrences: int = 1

    @property
    def id(self) -> str:
        key = f"{self.source}\x00{self.statement.lower()}"
        return "cand_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]

    def as_record(self) -> dict:
        return {"id": self.id, "source": self.source,
                "statement": self.statement[:MAX_STATEMENT],
                "evidence": [str(item)[:300] for item in self.evidence[:5]],
                "occurrences": self.occurrences}


def _clean(text: str, root) -> str:
    redacted, _ = redact.redact(text)
    return toolinput.scrub_paths(redacted, root).strip()


def _instruction_rules(root) -> List[Candidate]:
    """Lines the project already wrote as rules. Extraction is mechanical: a
    normative verb in a short line, not an interpretation of the document."""
    found: List[Candidate] = []
    for name in INSTRUCTION_FILES:
        path = Path(root) / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        fenced = False
        for number, raw in enumerate(lines, 1):
            if raw.strip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            line = raw.strip().lstrip("-*•").strip()
            if not line or line.startswith(("#", "|")) or len(line) < 12:
                continue
            if len(line) > MAX_STATEMENT or not NORMATIVE.search(line):
                continue
            found.append(Candidate("instruction-file", _clean(line, root),
                                   [f"{name}:{number}"]))
            if len(found) >= MAX_PER_SOURCE:
                return found
    return found


def _history(root, fmt: str) -> List[str]:
    out = gitcmd.run(["log", f"--max-count={HISTORY_LIMIT}", f"--format={fmt}"],
                     root, check=False)
    return [line for line in out.splitlines() if line.strip()]


def _reverts(root) -> List[Candidate]:
    """A revert is the project stating, in its own history, that something was
    wrong. It is evidence to review, never a rule to adopt."""
    found = []
    for line in _history(root, "%h\x1f%s"):
        parts = line.split("\x1f", 1)
        if len(parts) != 2:
            continue
        sha, subject = parts
        match = REVERT.match(subject.strip())
        if match:
            found.append(Candidate(
                "revert", _clean(f"reverted: {match.group(1)}", root), [sha]))
        if len(found) >= MAX_PER_SOURCE:
            break
    return found


def _fixups(root) -> List[Candidate]:
    found = []
    for line in _history(root, "%h\x1f%s"):
        parts = line.split("\x1f", 1)
        if len(parts) != 2:
            continue
        sha, subject = parts
        if FIXUP.match(subject.strip()) and _says_something(subject):
            found.append(Candidate("fixup", _clean(subject.strip(), root), [sha]))
        if len(found) >= MAX_PER_SOURCE:
            break
    return found


def _says_something(subject: str) -> bool:
    rest = FIXUP.sub("", subject.strip(), count=1)
    return any(word not in CONTENT_FREE for word in re.findall(r"[a-z]{3,}", rest.lower()))


def _gh(args: List[str], root, timeout: float = GH_TIMEOUT_S) -> Optional[str]:
    """None when the host cannot answer — no gh, not authenticated, no network.
    Every caller reports that as a named missing source, never as an empty one."""
    try:
        proc = subprocess.run(["gh"] + args, cwd=str(root), capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _pr_comments(root) -> List[Candidate]:
    """Review comments are the project correcting itself in prose, by people who
    had to read the change. Only normative lines are kept; the rest is chatter."""
    raw = _gh(["api", "repos/{owner}/{repo}/pulls/comments",
               "--paginate", "--slurp",
               "-X", "GET", "-F", f"per_page={GH_PAGE}", "-F", "sort=created",
               "-F", "direction=desc"], root)
    if raw is None:
        return []
    try:
        pages = json.loads(raw)
    except ValueError:
        return []
    comments = [c for page in pages for c in (page if isinstance(page, list) else [])]
    found: List[Candidate] = []
    for comment in comments[:GH_LIMIT]:
        where = str(comment.get("path") or "")
        for line in str(comment.get("body") or "").splitlines():
            line = line.strip().lstrip("-*> ").strip()
            if not (8 < len(line) <= MAX_STATEMENT) or not NORMATIVE.search(line):
                continue
            if line.startswith("#") or "```" in line:
                continue
            found.append(Candidate("pr-review", _clean(line, root),
                                   [f"PR review on {where}" if where else "PR review"]))
            if len(found) >= MAX_PER_SOURCE:
                return found
    return found


def _ci_failures(root) -> List[Candidate]:
    """A check that keeps failing is a rule the project already decided it wants;
    the candidate is the check's name, never an interpretation of the logs."""
    raw = _gh(["run", "list", "--status", "failure", "--limit", str(GH_LIMIT),
               "--json", "workflowName,displayTitle,conclusion"], root)
    if raw is None:
        return []
    try:
        runs = json.loads(raw)
    except ValueError:
        return []
    found: List[Candidate] = []
    for run in runs if isinstance(runs, list) else []:
        name = str(run.get("workflowName") or "").strip()
        if not name:
            continue
        found.append(Candidate(
            "ci-failure", _clean(f"the `{name}` check must pass before merge", root),
            [f"a failed `{name}` run"]))
        if len(found) >= MAX_PER_SOURCE:
            break
    return found


def _custom_checks(root, mechanisms_record: Optional[dict]) -> List[Candidate]:
    """A check the project wrote itself encodes a rule someone cared about."""
    found = []
    for mechanism in (mechanisms_record or {}).get("mechanisms") or []:
        if mechanism["tier"] != "project-owned-custom":
            continue
        enforced = mechanism.get("enforced_in") or []
        note = ("enforced in " + "+".join(enforced)) if enforced else "nothing runs it"
        found.append(Candidate(
            "custom-check",
            f"the project wrote its own check `{mechanism['name']}` ({note})",
            mechanism["evidence"][:2]))
    return found[:MAX_PER_SOURCE]


def _merge(candidates: List[Candidate]) -> List[Candidate]:
    merged: Dict[str, Candidate] = {}
    for candidate in candidates:
        existing = merged.get(candidate.id)
        if existing is None:
            merged[candidate.id] = candidate
        else:
            existing.occurrences += 1
            existing.evidence.extend(candidate.evidence)
    return sorted(merged.values(),
                  key=lambda c: (SOURCE_RANK.get(c.source, 9), c.source,
                                 -c.occurrences, c.statement))


def review_set(candidates: List[Candidate]) -> List[Candidate]:
    """Sources in turn, so one noisy source cannot fill the set by repetition."""
    queues: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        queues.setdefault(candidate.source, []).append(candidate)
    picked: List[Candidate] = []
    while len(picked) < REVIEW_SET and any(queues.values()):
        for queue in queues.values():
            if queue and len(picked) < REVIEW_SET:
                picked.append(queue.pop(0))
    return picked


def collect(root, mechanisms_record: Optional[dict] = None,
            include_remote: bool = False):
    """Returns (candidates, errors, unavailable). Nothing here is adopted.

    Remote sources are opt-in: opening a repository is consent to read it, not
    consent to call GitHub on the developer's behalf."""
    candidates: List[Candidate] = []
    errors: List[str] = []
    unavailable: List[str] = []
    for name, detector in (("instruction-file", _instruction_rules),
                           ("revert", _reverts), ("fixup", _fixups)):
        try:
            candidates += detector(root)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
    try:
        candidates += _custom_checks(root, mechanisms_record)
    except Exception as exc:
        errors.append(f"custom-check: {type(exc).__name__}: {exc}"[:300])

    remote = (("pull-request review comments", _pr_comments),
              ("CI failure history", _ci_failures))
    for name, detector in remote:
        if not include_remote:
            unavailable.append(f"{name} (not requested; use --include-remote)")
            continue
        try:
            found = detector(root)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
            unavailable.append(f"{name} (collection failed)")
            continue
        if found:
            candidates += found
        else:
            unavailable.append(f"{name} (gh unavailable, unauthenticated, or none found)")
    return _merge(candidates), errors, unavailable


def build_record(checkout_id: str, repository_id: Optional[str],
                 candidates: List[Candidate], collected_at: str,
                 errors: Optional[List[str]] = None,
                 unavailable: Optional[List[str]] = None) -> dict:
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "checkout_id": checkout_id,
        "repository_id": repository_id,
        "collected_at": collected_at,
        "adopted": False,
        "note": ("candidates only — a historical convention is never adopted "
                 "because it is frequent"),
        "sources_unavailable": list(unavailable or []),
        "collection_errors": list(errors or []),
        "review_set": [c.id for c in review_set(candidates)],
        "ranking_basis": RANKING_BASIS,
        "candidates": [c.as_record() for c in candidates],
    }


def path_for(checkout_id: str) -> Path:
    return paths.checkout_dir(checkout_id) / "bootstrap.json"


def save(checkout_id: str, rec: dict) -> Path:
    paths.ensure(paths.checkout_dir(checkout_id))
    target = path_for(checkout_id)
    return artifacts.save(target, rec, "bootstrap",
                          expected_checkout_id=checkout_id)


def load(checkout_id: str) -> Optional[dict]:
    target = path_for(checkout_id)
    return artifacts.load(target, "bootstrap", expected_checkout_id=checkout_id)
