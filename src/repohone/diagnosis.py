"""Pass 1 of retrospective reasoning: cause and required property (LEARNING_PLAN §11).

Diagnosis runs before the repository has been surveyed, so it must not name a
mechanism. Letting the model pick enforcement before discovery is how a project
ends up with a bespoke RepoHone rule beside a tool that already did the job.

A diagnosis is a hypothesis. Every job ends in exactly one terminal state and a
partial result is never promoted to one (ARCH §12.3).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import CONTRACT_VERSION, artifacts, evidence, fingerprint, paths, reasoning, record

SCHEMA_ID = "repohone.diagnosis/v1"
SCHEMA_VERSION = 1

TAXONOMY = ("MISSING_GUARDRAIL", "MISSING_CONTEXT", "MISSED_EXISTING_CAPABILITY",
            "WRONG_PROJECT_PATTERN", "VERIFICATION_GAP", "TOOLING_FRICTION",
            "BAD_GUARDRAIL", "INTENT_MISMATCH", "AGENT_REASONING_ERROR",
            "NEW_REQUIREMENT")

CONFIDENCE = ("low", "medium", "high")
FINGERPRINT_MAX_CHARS = 200

# A required property describes the project, not the thing that would enforce it.
# Naming a specific tool is always naming a mechanism, whatever the sentence
# around it says.
MECHANISM_TOOLS = (
    "eslint", "pylint", "ruff", "flake8", "mypy", "pyright", "black", "prettier",
    "pytest", "jest", "vitest", "playwright", "cypress", "husky", "pre-commit",
    "precommit", "import-linter", "semgrep", "sonarqube", "sonarcloud",
    "checkstyle", "rubocop", "golangci", "biome", "tsc", "clippy", "rustfmt",
    "phpstan", "psalm", "php-cs-fixer", "detekt", "ktlint", "swiftlint",
    "spotbugs", "bandit", "shellcheck", "hadolint", "tflint", "stylelint",
    "commitlint", "dependabot", "renovate", "codecov", "danger",
    "codeowners", "makefile", "editorconfig", "dockerfile",
    "claude.md", "agents.md", "cursor rule", "agent skill", "github action",
)

# These name a *kind* of mechanism. A property may say what the agent must know
# or do about one ("how to run the test suite must be discoverable"); it
# prescribes one when the kind is the thing to add.
MECHANISM_KINDS = (
    "lint rule", "linter", "ci check", "ci pipeline", "ci gate", "git hook",
    "unit test", "test suite", "integration test",
    "type checker", "static analysis", "code review", "review checklist",
    "instruction file", "pull request template",
)
_KIND = "(?:" + "|".join(re.escape(kind) for kind in MECHANISM_KINDS) + ")s?"
_ADDING = (r"add(?:s|ed|ing)?|creat(?:e|es|ed|ing)|introduc(?:e|es|ed|ing)|"
           r"install(?:s|ed|ing)?|configur(?:e|es|ed|ing)|adopt(?:s|ed|ing)?|"
           r"writ(?:e|es|ing|ten)|wrote|build(?:s|ing)?|built|enabl(?:e|es|ed|ing)|"
           r"set(?:s|ting)? up|wir(?:e|es|ed|ing) up")
# These prescribe only something new: "use a linter", not "use the test suite".
_WANTING = (r"requir(?:e|es|ed|ing)|us(?:e|es|ed|ing)|enforc(?:e|es|ed|ing)|"
            r"gat(?:e|es|ed|ing)|mandat(?:e|es|ed|ing)|need(?:s|ed|ing)?")
_PRESCRIBED = re.compile(
    rf"\b(?:{_ADDING})\b(?:\W+\w+){{0,3}}?\W+{_KIND}\b"
    rf"|\b(?:{_WANTING})\s+(?:\w+\s+)?(?:a|an|new|another)\s+(?:[\w-]+\s+){{0,2}}?{_KIND}\b")
MECHANISM_OPENERS = ("add ", "create ", "configure ", "install ", "introduce ",
                     "set up ", "enforce with ", "use a ", "adopt ")

_WORDS = re.compile(r"[a-z0-9.+#-]+")

PROMPT = """RepoHone diagnosis: use evidence from one repository session.

Answer: What objectively happened? What did the developer change or correct?
What caused the initial failure? Is it a project pattern or a one-off?
What property is the project missing? What evidence supports this? What contradicts it?
Is any project improvement warranted at all?
One observation is weak grounds for a project-level claim.

Rules:
- You must NOT name or suggest a mechanism: no tool names and nothing to add
  (test, linter, hook, CI step, instruction file). Existing ones may be named.
- Insufficient evidence and no improvement are valid. Do not speculate.
- `evidence_against` is required.
- A fingerprint identifies the specific recurring problem. Reuse a matching
  known `area` and `behavior` verbatim; do not create a near-duplicate.
- `history` items are earlier reverted or fixed commits; put the ref of any that
  is this same problem in `same_as_history`.

Return one JSON object only:
{{"outcome":"SUCCESS"|"INSUFFICIENT_EVIDENCE"|"NO_ACTIONABLE_PROJECT_IMPROVEMENT",
"root_cause":{{"class":"one of: {taxonomy}","summary":"one sentence"}},
"confidence":"low"|"medium"|"high",
"required_property":"project property, no mechanism",
"corrective_turn":2,
"fingerprint":{{"area":"kebab-case","behavior":"kebab-case"}},
"evidence_for":["..."],"evidence_against":["..."],"risks":["..."],"same_as_history":[]}}

For non-success outcomes set root_cause, confidence, required_property and
fingerprint to null. Evidence refs contain turn numbers. Set corrective_turn
to a 1-based integer only for the actual correction, or null; a later, unrelated
request is not a correction.

KNOWN FINGERPRINTS
{fingerprints}

EVIDENCE
{payload}
"""


def _payload(selection: evidence.Selection) -> str:
    blocks = []
    for index, item in enumerate(selection.items, 1):
        blocks.append(f"--- [{index}] {item.kind} ({item.ref}) ---\n{item.text}")
    return "\n\n".join(blocks)


_STATIC_BYTES = None


def static_prompt_bytes() -> int:
    """The fixed instructions are model input and belong inside the ceiling."""
    global _STATIC_BYTES
    if _STATIC_BYTES is None:
        _STATIC_BYTES = evidence.measure_bytes(
            PROMPT.format(taxonomy=", ".join(TAXONOMY), payload="", fingerprints=""))
    return _STATIC_BYTES


def build_prompt(selection, existing=None) -> str:
    return PROMPT.format(taxonomy=", ".join(TAXONOMY), payload=_payload(selection),
                         fingerprints=fingerprint.summarise(existing or []))


def fit_to_budget(selection, existing, ceiling: Optional[int] = None):
    """Drops the lowest-priority items until the built prompt fits, and returns
    what will actually be sent. Summing the parts understates it: the payload's
    own per-item framing is real tokens too."""
    ceiling = evidence.MAX_BYTES if ceiling is None else ceiling
    prompt = build_prompt(selection, existing)
    while evidence.measure_bytes(prompt) > ceiling and len(selection.items) > 1:
        selection.items.pop()
        selection.skipped += 1
        prompt = build_prompt(selection, existing)
    return prompt, evidence.measure_bytes(prompt)


def names_a_mechanism(text: Optional[str]) -> Optional[str]:
    """Returns the offending term, or None. Phase 3's one hard output rule.

    A blocklist cannot be complete, so it errs where the cost is lower: a missed
    tool name anchors Phase 4, while a wrongly flagged property is discarded
    whole.
    """
    if not text:
        return None
    lowered = text.lower()
    words = set(_WORDS.findall(lowered))
    for term in MECHANISM_TOOLS:
        if term in (lowered if " " in term else words):
            return term
    prescribed = _PRESCRIBED.search(lowered)
    if prescribed:
        return prescribed.group(0)
    for opener in MECHANISM_OPENERS:
        if lowered.startswith(opener):
            return opener.strip()
    return None


def interpret(parsed: Optional[dict], history_refs=()) -> Dict[str, Any]:
    """Maps a model reply to a terminal state. Missing fields are never inferred;
    an absent `same_as_history` claims no link."""
    if parsed is None:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": "reply contained no JSON object"}

    outcome = parsed.get("outcome")
    if outcome in (reasoning.INSUFFICIENT_EVIDENCE, reasoning.NO_ACTIONABLE):
        return {"outcome": outcome, "body": parsed}
    if outcome != reasoning.SUCCESS:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"unknown outcome {outcome!r}"}

    cause = parsed.get("root_cause")
    if not isinstance(cause, dict) or cause.get("class") not in TAXONOMY:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"root_cause.class is not in the taxonomy: {cause!r}"}
    if not isinstance(cause.get("summary"), str) or not cause["summary"].strip():
        return {"outcome": reasoning.INVALID_OUTPUT, "failure": "root_cause.summary is empty"}
    if parsed.get("confidence") not in CONFIDENCE:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"confidence is not low/medium/high: {parsed.get('confidence')!r}"}

    prop = parsed.get("required_property")
    if not isinstance(prop, str) or not prop.strip():
        return {"outcome": reasoning.INVALID_OUTPUT, "failure": "required_property is empty"}
    offending = names_a_mechanism(prop)
    if offending:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"required_property names a mechanism ({offending!r}); "
                           "a diagnosis names the property, and a proposal selects the mechanism"}
    mark = parsed.get("fingerprint")
    if not isinstance(mark, dict):
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": "fingerprint is required for a successful diagnosis"}
    for part in ("area", "behavior"):
        if not isinstance(mark.get(part), str) or not mark[part].strip():
            return {"outcome": reasoning.INVALID_OUTPUT,
                    "failure": f"fingerprint.{part} is empty"}
        if len(mark[part]) > FINGERPRINT_MAX_CHARS:
            return {"outcome": reasoning.INVALID_OUTPUT,
                    "failure": f"fingerprint.{part} exceeds "
                               f"{FINGERPRINT_MAX_CHARS} characters"}
    corrective = parsed.get("corrective_turn")
    if corrective is not None and not (isinstance(corrective, int)
                                       and not isinstance(corrective, bool)
                                       and corrective >= 1):
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"corrective_turn must be a turn number or null; "
                           f"got {corrective!r}"}
    for field in ("evidence_for", "evidence_against", "risks"):
        if not isinstance(parsed.get(field), list):
            return {"outcome": reasoning.INVALID_OUTPUT, "failure": f"{field} must be a list"}
    if not [v for v in parsed["evidence_for"] if str(v).strip()]:
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": "a successful diagnosis with no supporting evidence is an "
                           "assertion, not a finding"}
    linked = parsed.get("same_as_history")
    if linked is None:
        linked = []
    if not isinstance(linked, list) or any(ref not in history_refs for ref in linked):
        return {"outcome": reasoning.INVALID_OUTPUT,
                "failure": f"same_as_history must list refs of history items that were "
                           f"sent; got {linked!r}"}

    adjustment = None
    if parsed["confidence"] == "high" and not (parsed["evidence_against"]
                                               or parsed["risks"]):
        # Certainty has to be earned: high confidence that weighed nothing against
        # itself reads as decisive downstream, where it drives a real change.
        adjustment = {"from": "high", "to": "medium",
                      "reason": "high confidence was claimed without naming any "
                                "counter-evidence or risk"}
        parsed = dict(parsed, confidence="medium")
    return {"outcome": reasoning.SUCCESS, "body": parsed, "adjustment": adjustment,
            "history": list(dict.fromkeys(linked))}


def _strings(value) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v)[:600] for v in value if isinstance(v, (str, int, float))][:12]


def new_id(session_id: str, created_at: str, nonce: Optional[str] = None) -> str:
    """A timestamp alone collides when a session is diagnosed twice quickly, and
    a collision silently overwrites the earlier record."""
    nonce = nonce if nonce is not None else os.urandom(8).hex()
    digest = hashlib.sha256(
        f"{session_id}\x00{created_at}\x00{nonce}".encode()).hexdigest()
    return "dx_" + digest[:16]


def build_record(session_id: str, checkout_id: str, outcome: str, selection,
                 engine_name: str, model: Optional[str], isolated: bool,
                 body: Optional[dict] = None, failure: Optional[str] = None,
                 created_at: Optional[str] = None, diagnosis_id: Optional[str] = None,
                 mark: Optional[dict] = None, recurrence: Optional[dict] = None,
                 overhead_bytes: int = 0,
                 corrective_turn: Optional[int] = None,
                 adjustment: Optional[dict] = None,
                 repository_id: Optional[str] = None,
                 developer_id: Optional[str] = None,
                 history: Optional[List[dict]] = None) -> dict:
    created_at = created_at or record.now()
    # "do nothing" is the answer that prevents manufactured proposals, so its
    # reasoning is kept; only the diagnosis fields are cleared (§12.3).
    diagnosed = body if outcome == reasoning.SUCCESS else None
    cause = (diagnosed or {}).get("root_cause")
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "diagnosis_id": diagnosis_id or new_id(session_id, created_at),
        "session_id": session_id,
        "checkout_id": checkout_id,
        # Recurrence reads these here rather than loading the session record.
        "repository_id": repository_id,
        "developer_id": developer_id,
        "created_at": created_at,
        "pass": "diagnosis",
        "outcome": outcome,
        "fingerprint": mark if outcome == reasoning.SUCCESS else None,
        "recurrence": recurrence or {"sessions": 0, "checkouts": 0, "developers": 0,
                                     "known_fingerprints": 0, "scope": "local-checkouts"},
        "root_cause": ({"class": cause["class"], "summary": str(cause["summary"])[:600]}
                       if cause else None),
        "confidence": (diagnosed or {}).get("confidence"),
        "required_property": (str(diagnosed["required_property"])[:600]
                              if diagnosed else None),
        # A turn the developer named outranks one the model inferred (§14).
        # A failed job leaves none of it behind; the rule candidate still has it.
        "corrective_turn": (None if outcome != reasoning.SUCCESS else
                            (corrective_turn if corrective_turn is not None
                             else (diagnosed or {}).get("corrective_turn"))),
        "confidence_adjusted": adjustment if outcome == reasoning.SUCCESS else None,
        "evidence_for": _strings((body or {}).get("evidence_for")),
        "evidence_against": _strings((body or {}).get("evidence_against")),
        "risks": _strings((body or {}).get("risks")),
        "engine": {"name": engine_name, "model": model, "isolated": isolated},
        "budget": {"max_tokens": evidence.MAX_BYTES, "max_items": evidence.MAX_ITEMS,
                   "items_sent": len(selection.items),
                   "items_skipped": selection.skipped,
                   "estimated_tokens": selection.bytes + overhead_bytes},
        "evidence_refs": selection.refs(),
        # Earlier commits the model judged the same problem (Phase 6 candidates).
        "history": list(history or []) if outcome == reasoning.SUCCESS else [],
        "failure": {"reason": failure[:1000]} if failure else None,
    }


def diagnoses_dir(checkout_id: str) -> Path:
    return paths.checkout_dir(checkout_id) / "diagnoses"


def save(checkout_id: str, rec: dict) -> Path:
    artifacts.require_artifact_id("diagnosis", rec.get("diagnosis_id"))
    directory = paths.ensure(diagnoses_dir(checkout_id))
    path = directory / f"{rec['diagnosis_id']}.json"
    return artifacts.save(path, rec, "diagnosis",
                          expected_checkout_id=checkout_id,
                          expected_artifact_id=rec["diagnosis_id"])


def load_all(checkout_id: str) -> List[dict]:
    listed = artifacts.members(diagnoses_dir(checkout_id))
    return artifacts.load_all([p for p in listed or [] if p.name.startswith("dx_")],
                              "diagnosis",
                              expected_checkout_id=checkout_id)


@dataclass
class Prepared:
    """Everything a diagnosis would send, computed without sending it."""
    record: dict
    corrective_turn: Optional[int]
    repository_id: Optional[str]
    developer_id: Optional[str]
    existing: Any
    selection: evidence.Selection
    prompt: str
    sent_bytes: int
    sendable: bool


def prepare(root, checkout_id: str, session_id: str,
            rule_statement: Optional[str] = None,
            corrective_turn: Optional[int] = None) -> Prepared:
    rec = record.load(checkout_id, session_id)
    if rec is None:
        raise FileNotFoundError(f"no session record for {session_id}")

    corrective_turn = evidence.validated_turn(rec, corrective_turn)
    repository_id = (rec.get("repository") or {}).get("id")
    existing = fingerprint.known(checkout_id, repository_id)

    # A pre-estimate only, so large sessions do not load evidence that
    # fit_to_budget would drop again; the ceiling itself is enforced there.
    overhead = (static_prompt_bytes()
                + evidence.measure_bytes(fingerprint.summarise(existing)))
    selection = evidence.select(rec, root, rule_statement,
                                max_bytes=max(1, evidence.MAX_BYTES - overhead),
                                corrective_turn=corrective_turn)
    prompt, sent_bytes = fit_to_budget(selection, existing)
    return Prepared(record=rec, corrective_turn=corrective_turn,
                    repository_id=repository_id,
                    developer_id=(rec.get("developer") or {}).get("id"),
                    existing=existing, selection=selection, prompt=prompt,
                    sent_bytes=sent_bytes,
                    sendable=evidence.is_sufficient(selection))


def run(root, checkout_id: str, session_id: str, engine,
        rule_statement: Optional[str] = None,
        corrective_turn: Optional[int] = None) -> dict:
    """One reasoning job, one terminal state. The record is written whatever
    happens: a failed diagnosis is evidence too, and never a proposal."""
    prepared = prepare(root, checkout_id, session_id, rule_statement, corrective_turn)
    corrective_turn = prepared.corrective_turn
    repository_id, developer_id = prepared.repository_id, prepared.developer_id
    existing, selection = prepared.existing, prepared.selection
    prompt, sent_bytes = prepared.prompt, prepared.sent_bytes

    engine_name = getattr(engine, "name", type(engine).__name__)
    created_at = record.now()
    diagnosis_id = new_id(session_id, created_at)

    sent_history = {item.ref: item.meta for item in selection.items
                    if item.kind == "history" and item.meta}

    def finish(outcome, body=None, failure=None, model=None, isolated=True,
               mark=None, seen=None, adjustment=None, history=None):
        built = build_record(
            session_id, checkout_id, outcome, selection, engine_name, model, isolated,
            body=body, failure=failure, created_at=created_at,
            diagnosis_id=diagnosis_id, mark=mark,
            recurrence={"sessions": seen.sessions if seen else 0,
                        "checkouts": seen.checkouts if seen else 0,
                        "developers": seen.developers if seen else 0,
                        "known_fingerprints": len(existing),
                        "scope": "local-checkouts",
                        "complete": seen.complete if seen else existing.complete,
                        "historical": seen.historical if seen else 0},
            overhead_bytes=max(0, sent_bytes - selection.bytes),
            corrective_turn=corrective_turn, adjustment=adjustment,
            repository_id=repository_id, developer_id=developer_id,
            history=[sent_history[ref] for ref in history or []])
        save(checkout_id, built)
        return built

    if not prepared.sendable:
        return finish(reasoning.INSUFFICIENT_EVIDENCE,
                      failure="selected evidence lacks either developer intent or "
                              "observed agent behaviour; expanding context is not the answer")

    try:
        reply = engine.run(prompt)
    except reasoning.Unavailable as exc:
        return finish(reasoning.MODEL_UNAVAILABLE, failure=str(exc))
    except reasoning.Interrupted as exc:
        return finish(reasoning.INTERRUPTED, failure=str(exc))

    verdict = interpret(reasoning.extract_json(reply.text), sent_history)
    if verdict["outcome"] != reasoning.SUCCESS:
        return finish(verdict["outcome"], body=verdict.get("body"),
                      failure=verdict.get("failure"), model=reply.model,
                      isolated=reply.isolated)

    body = verdict["body"]
    proposed = body["fingerprint"]
    published = {}

    def publish(matched, created, seen):
        mark = {"fingerprint_id": matched.fingerprint_id, "area": matched.area,
                "behavior": matched.behavior, "created": created}
        published["record"] = finish(
            reasoning.SUCCESS, body=body, adjustment=verdict.get("adjustment"),
            model=reply.model, isolated=reply.isolated, mark=mark, seen=seen,
            history=verdict["history"])

    fingerprint.record_occurrence(
        checkout_id, repository_id, body["root_cause"]["class"],
        proposed["area"], proposed["behavior"], session_id, diagnosis_id, created_at,
        developer_id=developer_id, publish=publish, history=verdict["history"])
    return published["record"]
