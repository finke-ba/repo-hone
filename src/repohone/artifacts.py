"""One strict codec for RepoHone's persisted JSON artifacts.

Every semantic reader goes through this module. A JSON file being readable is
not enough: its artifact kind, shape version and semantic contract must all be
understood before Core may use it.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .paths import entries, read_regular


class ArtifactError(RuntimeError):
    """A persisted artifact cannot safely be interpreted."""


class MalformedArtifact(ArtifactError):
    """The bytes do not form the artifact they claim to be."""


class UnsupportedVersion(ArtifactError):
    """The artifact uses semantics this Core does not understand."""


# Kind -> (schema identifier, supported shape versions, first compatible
# contract). Derived artifacts written before contract 1.7 are regenerated;
# bootstrap arrived in 1.8.
REGISTRY: Dict[str, Tuple[str, frozenset, str]] = {
    "session": ("repohone.session/v1", frozenset({1}), "1.4"),
    "rule": ("repohone.rule_candidate/v1", frozenset({1}), "1.7"),
    "diagnosis": ("repohone.diagnosis/v1", frozenset({1}), "1.7"),
    "mechanisms": ("repohone.mechanisms/v1", frozenset({1}), "1.7"),
    "proposal": ("repohone.proposal/v1", frozenset({1}), "1.7"),
    "bootstrap": ("repohone.bootstrap/v1", frozenset({1}), "1.8"),
}

SUPPORTED_CONTRACTS = frozenset(
    {"1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "1.10", "1.11",
     "1.12", "1.13", "1.14", "1.15", "1.16", "1.17", "1.18", "1.19", "1.20",
     "1.21", "1.22", "1.23", "1.24", "1.25"})

# Artifact ids that are encoded in their containing filename. Fixed-name
# artifacts (mechanisms and bootstrap) still carry and verify checkout_id.
ID_FIELDS = {
    "session": "session_id",
    "rule": "candidate_id",
    "diagnosis": "diagnosis_id",
    "proposal": "proposal_id",
}
ID_PATTERNS = {
    "session": re.compile(r"^rh_[0-9a-f]{8,}$"),
    "rule": re.compile(r"^rule_[0-9a-f]{16}$"),
    "diagnosis": re.compile(r"^dx_[0-9a-f]{8,}$"),
    "proposal": re.compile(r"^prop_[0-9a-f]{8,}$"),
}


def require_artifact_id(kind: str, value: object) -> str:
    """Reject a caller-controlled path component before any filesystem use."""
    pattern = ID_PATTERNS.get(kind)
    if pattern is None or not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise MalformedArtifact(f"invalid {kind} artifact id {value!r}")
    return value


def _version(value: object) -> Tuple[int, ...]:
    try:
        return tuple(int(part) for part in str(value).split("."))
    except ValueError:
        return ()


def _migrate(value: dict, kind: str) -> dict:
    """Return the current in-memory shape for an explicitly supported contract.

    These fields were additive safety metadata.  Their historical absence has a
    precise conservative meaning, so older records can be read without
    inventing evidence.  Current-contract records are never repaired here:
    missing fields in current output remain corruption.
    """
    if kind != "proposal":
        return value
    contract = _version(value.get("contract_version"))
    if not contract or contract >= _version("1.16"):
        return value
    migrated = deepcopy(value)
    validation = migrated.get("validation")
    if contract < _version("1.8") and isinstance(validation, dict):
        validation.setdefault("estimate", None)
    rollback = migrated.get("rollback")
    if contract < _version("1.12") and isinstance(rollback, dict):
        for entry in rollback.get("files") or []:
            if isinstance(entry, dict):
                entry.setdefault("previous_mode", None)
    change = migrated.get("change")
    if contract < _version("1.13") and isinstance(change, dict):
        for entry in change.get("files") or []:
            if isinstance(entry, dict):
                entry.setdefault("baseline_mode", None)
    privacy = migrated.get("privacy")
    if contract < _version("1.15") and isinstance(privacy, dict):
        privacy.setdefault("egress", None)
    return migrated


def validate(value, kind: str) -> dict:
    """Return an understood artifact or raise a precise refusal."""
    if kind not in REGISTRY:
        raise ValueError(f"unknown artifact kind {kind!r}")
    if not isinstance(value, dict):
        raise MalformedArtifact(f"{kind} artifact must be a JSON object")

    schema, shapes, minimum_contract = REGISTRY[kind]
    if value.get("schema") != schema:
        raise MalformedArtifact(
            f"expected {schema!r}; got artifact schema {value.get('schema')!r}")

    shape = value.get("schema_version")
    if type(shape) is not int or shape not in shapes:
        supported = ", ".join(str(item) for item in sorted(shapes))
        raise UnsupportedVersion(
            f"schema_version {shape!r} is not supported for {kind} (expected {supported})")

    contract = value.get("contract_version")
    if (not isinstance(contract, str) or contract not in SUPPORTED_CONTRACTS
            or _version(contract) < _version(minimum_contract)):
        raise UnsupportedVersion(
            f"contract_version {contract!r} is not supported for {kind}; "
            f"minimum {minimum_contract}, newest supported "
            f"{max(SUPPORTED_CONTRACTS, key=_version)}")

    value = _migrate(value, kind)

    # Imported lazily so package import remains cheap. Every persisted artifact
    # is a semantic boundary: a valid header alone is never sufficient.
    from .session_validation import validate_artifact, validate_session
    problems = (validate_session(value) if kind == "session"
                else validate_artifact(value, kind))
    if problems:
        summary = "; ".join(problems[:5])
        more = f"; and {len(problems) - 5} more" if len(problems) > 5 else ""
        raise MalformedArtifact(f"{kind} artifact is invalid: {summary}{more}")
    return value


def validate_context(value: dict, kind: str, *,
                     expected_checkout_id: Optional[str] = None,
                     expected_artifact_id: Optional[str] = None) -> None:
    """Bind semantic contents to the checkout and filename that selected them."""
    checkout = ((value.get("repository") or {}).get("checkout_id")
                if kind == "session" else value.get("checkout_id"))
    if expected_checkout_id is not None and checkout != expected_checkout_id:
        raise MalformedArtifact(
            f"embedded checkout_id {checkout!r} does not match "
            f"{expected_checkout_id!r}")
    id_field = ID_FIELDS.get(kind)
    if expected_artifact_id is not None and id_field:
        actual = value.get(id_field)
        if actual != expected_artifact_id:
            raise MalformedArtifact(
                f"artifact claims {id_field} {actual!r}; filename identifies "
                f"{expected_artifact_id!r}")


def load(path: Path, kind: str, *, strict: bool = True,
         expected_checkout_id: Optional[str] = None,
         expected_artifact_id: Optional[str] = None) -> Optional[dict]:
    """Read one artifact. Missing is None; anything else wrong is an error in
    strict mode — including a directory or dangling link where the file was."""
    path = Path(path)
    try:
        text = read_regular(path)
        if text is None:
            return None
        value = json.loads(text)
    except (ValueError, OSError) as exc:
        if strict:
            raise MalformedArtifact(f"{path.name}: unreadable ({exc})") from exc
        return None
    if strict:
        try:
            checked = validate(value, kind)
            validate_context(checked, kind,
                             expected_checkout_id=expected_checkout_id,
                             expected_artifact_id=expected_artifact_id)
            return checked
        except ArtifactError as exc:
            raise type(exc)(f"{path.name}: {exc}") from exc
    return value if isinstance(value, dict) else None


# A write stages its bytes under this prefix and renames them into place. A
# process killed in between leaves the staging file behind.
STAGING_PREFIX = ".rh-"


def members(directory: Path) -> Optional[List[Path]]:
    """The JSON artifacts in `directory`, or None when it is absent. A leftover
    staging file is not a member: reading one as an artifact made a single
    interrupted write block every later listing."""
    listed = entries(directory)
    if listed is None:
        return None
    return [p for p in listed
            if p.name.endswith(".json") and not p.name.startswith(STAGING_PREFIX)]


def load_all(paths: Iterable[Path], kind: str, *, strict: bool = True,
             expected_checkout_id: Optional[str] = None) -> List[dict]:
    """Read a collection without silently dropping a bad member."""
    out = []
    for path in paths:
        path = Path(path)
        value = load(
            path, kind, strict=strict,
            expected_checkout_id=expected_checkout_id,
            expected_artifact_id=path.stem if kind in ID_FIELDS else None)
        if value is not None:
            out.append(value)
    return out


def atomic_write(path: Path, value) -> Path:
    """Replace one JSON file only after its complete bytes are durable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=STAGING_PREFIX,
                                     suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(str(path.parent), getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            # Some filesystems/platforms do not allow directory fsync. The file
            # is still atomically visible; durability uses the strongest local
            # primitive available.
            pass
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def save(path: Path, value, kind: str, *,
         expected_checkout_id: Optional[str] = None,
         expected_artifact_id: Optional[str] = None) -> Path:
    """Validate shape and provenance before atomically publishing an artifact."""
    checked = validate(value, kind)
    validate_context(checked, kind,
                     expected_checkout_id=expected_checkout_id,
                     expected_artifact_id=expected_artifact_id)
    return atomic_write(path, checked)
