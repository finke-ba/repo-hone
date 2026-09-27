"""Pass 2 of retrospective reasoning: mechanism selection (LEARNING_PLAN §11).

Diagnosis named a property without naming a mechanism. Discovery surveyed what
the repository already provides. Only now may a mechanism be chosen — and it
must come from that survey, because a mechanism the project does not already
have is exactly as foreign as RepoHone-specific enforcement.

Questions 3, 4 and 5 are load-bearing: a selection that cannot explain why what
already existed failed, cannot name the friction it adds, or cannot conclude
"do nothing" will manufacture improvements.
"""
from __future__ import annotations

import hashlib
import posixpath
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import reasoning

NO_MECHANISM = "NO_MECHANISM_AVAILABLE"
NEW_MECHANISM = "NEW_MECHANISM"
NEW_KINDS = {"lint", "test", "format", "types", "script"}
EGRESS_FIELDS = ("what_leaves", "provider", "when", "frequency", "cost")

# Files that are themselves a way of running something.
RUNNER_FILES = {"Makefile", "makefile", "GNUmakefile", "justfile", "Justfile",
                "package.json", "pyproject.toml", "tox.ini", "noxfile.py",
                ".pre-commit-config.yaml", "Cargo.toml", "go.mod"}
MAX_FILES = 6
MAX_FILE_CHARS = 8000
MAX_EDITS = 8

# A proposal must keep working if RepoHone is removed (ARCH §25).
RUNTIME_DEPENDENCE = re.compile(r"\brepohone\b", re.I)
WORD = re.compile(r"[a-z0-9]+")

PROMPT = """RepoHone mechanism selection. Choose from the discovered inventory
and write the concrete project-owned change.

DIAGNOSIS: cause={cause}; confidence={confidence}; property={required_property};
pattern={fingerprint}; sessions={sessions}{history}

INVENTORY
{inventory}

Answer which mechanism best carries that property, whether there is a simpler
option, why didn't an existing mechanism already help, what friction will this add,
and whether to make no proposal.

Rules:
- Prefer an inventory mechanism_id, especially one enforced in ci.
- If none fits, one new project-owned check is allowed. It must install nothing,
  be directly runnable, and be wired into an existing project runner. Explain
  why close existing options fail. If no small check fits, return
  NO_MECHANISM_AVAILABLE; that is valid.
- The result must work after RepoHone is removed. Paths are normalized and
  repository-relative. A new file gets complete contents. An existing file gets
  edits to its current text, never a restatement: each `find` must occur in it
  exactly once and becomes `replace`; `append` adds to the end.
- Disclose external model egress fully.

Return one JSON object only:
{{"outcome":"SUCCESS"|"NEW_MECHANISM"|"NO_MECHANISM_AVAILABLE",
"mechanism_id":"inventory id or null","new_mechanism":{{"name":"...",
"kind":"lint|test|format|types|script","invocation":"developer command",
"argv":["..."]}},"why_this":"...","why_existing_did_not_help":"required",
"alternatives_considered":["simpler option and why not","doing nothing: ..."],
"change":{{"summary":"...","files":[{{"path":"new/path","action":"create",
"contents":"complete contents"}},{{"path":"existing/path","action":"modify",
"edits":[{{"find":"exact existing text","replace":"new text"}},{{"append":"..."}}]}}]}},
"expected_effect":"...","friction":["..."],"adds_model_egress":false,
"egress":null}}

If adds_model_egress is true, egress requires what_leaves, provider, when,
frequency, and cost. For SUCCESS set new_mechanism null. For
NO_MECHANISM_AVAILABLE set mechanism_id, new_mechanism and change null but keep
why_existing_did_not_help, alternatives_considered and friction.
"""


def _short(value, limit: int) -> str:
    return str(value or "")[:limit].replace("\n", " ").strip()


def describe_inventory(mechanisms_record: Optional[dict],
                       max_items: int = reasoning.MAX_INPUT_ITEMS) -> str:
    items = ((mechanisms_record or {}).get("mechanisms") or [])[:max_items]
    if not items:
        return ("(nothing was discovered: this project has no test, lint, type or "
                "check mechanism. Only a new project-owned check can carry the "
                "property here, if a small one honestly would.)")
    lines = []
    for mechanism in items:
        enforced = [_short(value, 60) for value in
                    (mechanism.get("enforced_in") or [])[:4]]
        where = f"enforced in {'+'.join(enforced)}" if enforced else "NOT enforced anywhere"
        lines.append(
            f"- id={_short(mechanism.get('id'), 80)} "
            f"kind={_short(mechanism.get('kind'), 60)} "
            f"name={_short(mechanism.get('name'), 120)} "
            f"tier={_short(mechanism.get('tier'), 80)} "
            f"invocation={_short(mechanism.get('invocation') or '-', 240)} "
            f"({where})")
        excerpt = mechanism.get("config_excerpt")
        if excerpt:
            lines.append(f"    current config: {_short(excerpt, 240)}")
    return "\n".join(lines)


def _terms(value: Any) -> set:
    """Small deterministic vocabulary for bounded inventory relevance."""
    out = set()
    for token in WORD.findall(str(value or "").lower()):
        out.add(token)
        if len(token) > 3 and token.endswith("s"):
            out.add(token[:-1])
    return out


def _inventory_order(diagnosis: dict, mechanisms_record: Optional[dict]) -> List[dict]:
    """Put relevant entries first while retaining one representative per kind.

    Discovery's original order still breaks ties, preserving the native tier
    preference.  Relevance only decides which entries survive the hard bound.
    """
    items = [item for item in
             ((mechanisms_record or {}).get("mechanisms") or [])
             if isinstance(item, dict)]
    cause = diagnosis.get("root_cause") or {}
    mark = diagnosis.get("fingerprint") or {}
    query = _terms(" ".join(str(value or "") for value in (
        diagnosis.get("required_property"), cause.get("class"), cause.get("summary"),
        mark.get("area"), mark.get("behavior"))))

    scored = []
    for index, item in enumerate(items):
        carries = item.get("carries") or []
        description = " ".join(str(value or "") for value in (
            item.get("kind"), item.get("name"), *carries))
        overlap = len(query & _terms(description))
        scored.append((overlap, index, item))
    scored.sort(key=lambda row: (-row[0], row[1]))

    ordered, used, represented = [], set(), set()
    # First preserve capability diversity. Relevant kinds win when the number
    # of kinds itself exceeds the item budget.
    for _, index, item in scored:
        kind = str(item.get("kind") or "")
        if kind in represented:
            continue
        ordered.append(item)
        used.add(index)
        represented.add(kind)
    # Then use remaining capacity for the strongest additional mechanisms.
    ordered.extend(item for _, index, item in scored if index not in used)
    return ordered


def _history_note(diagnosis: dict) -> str:
    count = (diagnosis.get("recurrence") or {}).get("historical") or 0
    return f"; earlier_reverted_or_fixed_commits={count}" if count else ""


def build_prompt(diagnosis: dict, mechanisms_record: Optional[dict]) -> str:
    cause = (diagnosis.get("root_cause") or {})
    mark = diagnosis.get("fingerprint") or {}
    fields = {
        "cause": _short(f"{cause.get('class')}: {cause.get('summary')}", 500),
        "confidence": _short(diagnosis.get("confidence"), 30),
        "required_property": _short(diagnosis.get("required_property"), 500),
        "fingerprint": _short(f"{mark.get('area')}/{mark.get('behavior')}", 240),
        "sessions": (diagnosis.get("recurrence") or {}).get("sessions", 0),
        "history": _history_note(diagnosis),
    }
    all_items = _inventory_order(diagnosis, mechanisms_record)[
        :reasoning.MAX_INPUT_ITEMS]
    selected: list = []
    # Add ranked mechanisms only while the complete prompt fits. The output
    # instructions remain intact; arbitrary tail truncation could remove them.
    for item in all_items:
        candidate = selected + [item]
        prompt = PROMPT.format(
            **fields, inventory=describe_inventory({"mechanisms": candidate}))
        if reasoning.measure_bytes(prompt) <= reasoning.MAX_INPUT_BYTES:
            selected = candidate
    inventory = describe_inventory({"mechanisms": selected})
    if all_items and not selected:
        inventory = "(the bounded inventory could not fit in this request)"
    prompt = PROMPT.format(**fields, inventory=inventory)
    if reasoning.measure_bytes(prompt) > reasoning.MAX_INPUT_BYTES:
        # Static instructions plus tightly bounded fields are expected to fit.
        # Failing here prevents egress if that invariant changes later.
        raise reasoning.Unavailable("mechanism-selection prompt exceeds the local input budget")
    return prompt


def _existing_sha(known_paths, path) -> Optional[str]:
    """What the target looks like now, so applying can refuse a later edit."""
    reader = getattr(known_paths, "sha256", None)
    return reader(path) if reader else None


def _existing_mode(known_paths, path) -> Optional[int]:
    reader = getattr(known_paths, "mode", None)
    return reader(path) if reader else None


def _apply_edits(current: Optional[str], edits, path: str):
    """``(contents, added_text, problem)``. Text no edit names is kept byte for
    byte: the model never sees a whole file, so it must not be asked to restate one."""
    if current is None:
        return None, None, f"the current contents of {path} cannot be read"
    if not isinstance(edits, list) or not edits:
        return None, None, f"no edits proposed for {path}"
    if len(edits) > MAX_EDITS:
        return None, None, f"{path} has {len(edits)} edits; at most {MAX_EDITS}"
    text, added = current, []
    for edit in edits:
        if isinstance(edit, dict) and set(edit) == {"append"} \
                and isinstance(edit["append"], str) and edit["append"].strip():
            addition = edit["append"] if edit["append"].endswith("\n") \
                else edit["append"] + "\n"
            text = (text if not text or text.endswith("\n") else text + "\n") + addition
            added.append(addition)
            continue
        if not (isinstance(edit, dict) and set(edit) == {"find", "replace"}
                and isinstance(edit["find"], str) and edit["find"]
                and isinstance(edit["replace"], str)):
            return None, None, (f"an edit to {path} needs a non-empty `find` and a "
                                f"`replace`, or a non-empty `append`")
        found = text.count(edit["find"])
        if found != 1:
            return None, None, (f"`find` text occurs {found} times in {path}; it must "
                                f"occur exactly once")
        text = text.replace(edit["find"], edit["replace"], 1)
        added.append(edit["replace"])
    if text == current:
        return None, None, f"the edits to {path} change nothing"
    return text, "\n".join(added), None


def _clean_files(raw, known_paths) -> Tuple[Optional[List[dict]], Optional[str]]:
    if not isinstance(raw, list) or not raw:
        return None, "change.files is empty"
    if len(raw) > MAX_FILES:
        return None, f"change touches {len(raw)} files; at most {MAX_FILES}"
    cleaned = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, dict):
            return None, "change.files contains a non-object"
        path, action = entry.get("path"), entry.get("action")
        if not isinstance(path, str) or not path.strip():
            return None, "a change file has no path"
        path = path.strip()
        if path.startswith("/") or ".." in path.split("/"):
            return None, f"path escapes the repository: {path!r}"
        if "\\" in path:
            return None, f"path must use portable '/' separators: {path!r}"
        normalized = posixpath.normpath(path)
        if normalized != path or normalized in ("", "."):
            return None, f"path must be normalized: {path!r}"
        # APFS and Windows commonly fold case and Unicode. Conservatively reject
        # aliases everywhere rather than validate one overlay and apply another.
        key = unicodedata.normalize("NFC", normalized).casefold()
        if key in seen:
            return None, f"change.files contains duplicate path {path!r}"
        seen.add(key)
        exists = path in known_paths
        if action is None:
            action = "modify" if exists else "create"
        if action not in ("create", "modify"):
            return None, f"invalid action {action!r} for {path}"
        if action == "create" and exists:
            return None, f"{path} already exists and cannot be created"
        if action == "modify" and not exists:
            return None, f"{path} does not exist and cannot be modified"
        if action == "create":
            contents = added = entry.get("contents")
            if not isinstance(contents, str) or not contents.strip():
                return None, f"no contents proposed for {path}"
        else:
            if entry.get("contents") is not None:
                return None, (f"{path} already exists; give edits to its current text "
                              f"rather than restating it")
            reader = getattr(known_paths, "text", None)
            contents, added, problem = _apply_edits(
                reader(path) if reader else None, entry.get("edits"), path)
            if problem:
                return None, problem
        if len(contents) > MAX_FILE_CHARS:
            return None, f"{path} is {len(contents)} chars; at most {MAX_FILE_CHARS}"
        # Only what the change adds: an existing mention of RepoHone in a file
        # is not a dependency this change creates.
        if RUNTIME_DEPENDENCE.search(added or "") or RUNTIME_DEPENDENCE.search(path):
            return None, (f"{path} would make the project depend on RepoHone at "
                          f"runtime; an accepted improvement must survive its removal")
        cleaned.append({"path": path, "action": action, "contents": contents,
                        "baseline_sha256": _existing_sha(known_paths, path),
                        "baseline_mode": _existing_mode(known_paths, path)})
    return cleaned, None


def _unwired(mechanism: dict, files, mechanisms_record) -> Optional[str]:
    """A check nothing invokes is decoration: it can pass §14 on the historical
    pair and still never run again. It has to be reachable from something the
    project already runs."""
    known = [m for m in ((mechanisms_record or {}).get("mechanisms") or [])
             if m.get("id") != mechanism.get("id")]
    if mechanism.get("enforced_in"):
        return None       # something already runs it; that is the wiring
    if not mechanism.get("deterministic", True):
        return None       # advice an agent reads has no runner to be wired into
    # Sharing an interpreter is not wiring: `python3 scripts/check.py` beside an
    # existing `python3 -m pytest` is still a script nothing invokes. Only a
    # change to something that actually runs things counts.
    wired_into = {m.get("config_path") for m in known if m.get("config_path")}
    for m in known:
        for token in str(m.get("invocation") or "").split():
            if "/" in token or token in RUNNER_FILES:
                wired_into.add(token)
        name = str(m.get("name") or "")
        if "/" in name:
            wired_into.add(name)
    # A project with no mechanisms at all has no runner to hook into, so the
    # change has to leave one behind.
    wired_into |= RUNNER_FILES
    # The declared command must be the one a developer runs, not the new script
    # itself: §14 executes this argv, so declaring the script directly proves the
    # script works and says nothing about whether anything invokes it.
    argv = " ".join(mechanism.get("argv") or [])
    direct = [f["path"] for f in files if f["path"] in argv]
    if direct:
        return (f"the command `{argv}` runs {direct[0]} directly. Declare the "
                f"command a developer actually runs, so validating it proves the "
                f"project invokes this check rather than proving the check works "
                f"when called by hand.")
    runner_changed = [f for f in files
                      if f["path"] in wired_into
                      or Path(f["path"]).name in RUNNER_FILES
                      or f["path"].startswith(".github/workflows/")
                      or f["path"].startswith(".claude/")]
    # Touching a runner is not wiring, but the proof is §14 running `argv`
    # against the problematic tree: a runner that never calls the check exits
    # zero there and the proposal is rejected.
    if runner_changed:
        return None
    return (f"{mechanism['name']} would be run by `{' '.join(mechanism.get('argv') or [])}`, "
            f"which nothing in this project runs automatically, and the change "
            f"wires it into nothing that does. A check no workflow invokes cannot "
            f"carry the property for a future session, however well it does on "
            f"the historical pair.")


def _new_mechanism(raw) -> Tuple[Optional[dict], Optional[str]]:
    """Rung three of the plan's ladder. Still has to be runnable: an unvalidatable
    mechanism is worse than none, because §14 cannot show it catches anything."""
    if not isinstance(raw, dict):
        return None, "NEW_MECHANISM needs a new_mechanism object"
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "new_mechanism.name is empty"
    kind = raw.get("kind")
    if kind not in NEW_KINDS:
        return None, f"new_mechanism.kind must be one of {', '.join(sorted(NEW_KINDS))}"
    argv = raw.get("argv")
    if (not isinstance(argv, list) or not argv
            or not all(isinstance(a, str) and a.strip() for a in argv)):
        return None, ("new_mechanism.argv must be the command that runs it; a "
                      "mechanism that cannot be run cannot be shown to catch anything")
    invocation = raw.get("invocation")
    if not isinstance(invocation, str) or not invocation.strip():
        return None, "new_mechanism.invocation is empty"
    for field in (name, invocation, " ".join(argv)):
        if RUNTIME_DEPENDENCE.search(field):
            return None, ("the new mechanism names RepoHone; an accepted improvement "
                          "must survive its removal")
    return {"id": "mech_" + hashlib.sha256(
                " ".join(argv).encode("utf-8")).hexdigest()[:12],
            "kind": kind, "name": name.strip()[:120], "tier": "project-owned-custom",
            "invocation": invocation.strip()[:300], "argv": argv[:12],
            "deterministic": True, "enforced_in": [],
            "cost": {"probed": False, "duration_ms": None, "exit_code": None,
                     "note": "newly proposed; never run in this project before"}}, None


def interpret(parsed: Optional[dict], mechanisms_record: Optional[dict],
              known_paths=()) -> Dict[str, Any]:
    """Maps a reply to a terminal state. Nothing is inferred."""
    if parsed is None:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": "reply contained no JSON object"}

    outcome = parsed.get("outcome")
    for field in ("why_existing_did_not_help",):
        if not isinstance(parsed.get(field), str) or not parsed[field].strip():
            return {"outcome": reasoning.INVALID_OUTPUT,
                    "failure": f"{field} is required; a selection that cannot explain "
                               f"why what existed failed will manufacture improvements"}
    if not isinstance(parsed.get("alternatives_considered"), list):
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": "alternatives_considered must be a list"}

    if outcome == NO_MECHANISM:
        return {"outcome": NO_MECHANISM, "body": parsed}
    if outcome not in (reasoning.SUCCESS, NEW_MECHANISM):
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"unknown outcome {outcome!r}"}

    if outcome == NEW_MECHANISM:
        chosen, problem = _new_mechanism(parsed.get("new_mechanism"))
        if problem or chosen is None:
            return {"outcome": reasoning.INVALID_OUTPUT, "failure": problem}
    else:
        index = {m["id"]: m for m in (mechanisms_record or {}).get("mechanisms") or []}
        chosen = index.get(parsed.get("mechanism_id"))
        if chosen is None:
            return {"outcome": reasoning.INVALID_OUTPUT,
                    "failure": f"mechanism_id {parsed.get('mechanism_id')!r} is not "
                               f"one this repository provides"}

    if parsed.get("adds_model_egress"):
        if not isinstance(parsed.get("egress"), dict):
            return {"outcome": reasoning.INVALID_OUTPUT,
                    "failure": "this mechanism sends content to a model provider, "
                               "so it requires an egress disclosure object"}
        missing = [f for f in EGRESS_FIELDS
                   if not str(parsed["egress"].get(f, "")).strip()]
        if missing:
            return {"outcome": reasoning.INVALID_OUTPUT,
                    "failure": f"this mechanism sends content to a model provider, "
                               f"so it requires {', '.join(missing)} before it can "
                               f"be offered for approval"}
    change = parsed.get("change")
    if not isinstance(change, dict):
        return {"outcome": reasoning.INVALID_OUTPUT, "failure": "change is missing"}
    files, problem = _clean_files(change.get("files"), known_paths)
    if problem:
        return {"outcome": reasoning.INVALID_OUTPUT, "failure": problem}
    problem = _unwired(chosen, files, mechanisms_record) if (
        outcome == NEW_MECHANISM or not (chosen.get("enforced_in") or [])) else None
    if problem:
        return {"outcome": reasoning.INVALID_OUTPUT, "failure": problem}
    if not isinstance(change.get("summary"), str) or not change["summary"].strip():
        return {"outcome": reasoning.INVALID_OUTPUT, "failure": "change.summary is empty"}

    return {"outcome": reasoning.SUCCESS, "body": parsed,
            "mechanism": chosen, "files": files}
