"""Small dependency-free runtime validator for RepoHone's JSON Schemas.

Capture cannot depend on a test-only package, but every persisted artefact is a
semantic boundary. This module implements exactly the JSON Schema vocabulary
used by RepoHone's packaged schemas and rejects a schema if a new unsupported
keyword is introduced.
"""
from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any, Dict, Iterable, List, Optional

from .record_invariants import check_record

SCHEMA_FILES = {
    "session": "session-record.v1.schema.json",
    "rule": "rule-candidate.v1.schema.json",
    "diagnosis": "diagnosis.v1.schema.json",
    "mechanisms": "mechanisms.v1.schema.json",
    "proposal": "proposal.v1.schema.json",
    "bootstrap": "bootstrap.v1.schema.json",
}


def _load_schema(filename: str) -> dict:
    target = resources.files("repohone").joinpath(f"schemas/{filename}")
    return json.loads(target.read_text(encoding="utf-8"))


SCHEMAS = {kind: _load_schema(filename) for kind, filename in SCHEMA_FILES.items()}
# Kept for callers and tests that inspect the session schema directly.
SCHEMA = SCHEMAS["session"]

SUPPORTED_KEYWORDS = {
    "$schema", "$id", "$defs", "$ref", "$comment", "title", "description",
    "format", "type", "properties", "required", "additionalProperties", "items",
    "enum", "const", "pattern", "minimum", "minLength", "maxLength", "minItems",
    "uniqueItems", "oneOf", "allOf", "not", "if", "then",
}


def unsupported_keywords(kind: Optional[str] = None) -> List[str]:
    """Name schema vocabulary the runtime validator does not understand."""
    found = []

    def visit(node: dict, path: str) -> None:
        for keyword in node:
            if keyword not in SUPPORTED_KEYWORDS:
                found.append(f"{path}.{keyword}")
        for mapping in ("$defs", "properties"):
            for name, child in (node.get(mapping) or {}).items():
                if isinstance(child, dict):
                    visit(child, f"{path}.{mapping}.{name}")
        for name in ("items", "additionalProperties", "not", "if", "then"):
            child = node.get(name)
            if isinstance(child, dict):
                visit(child, f"{path}.{name}")
        for name in ("oneOf", "allOf"):
            for index, child in enumerate(node.get(name) or []):
                if isinstance(child, dict):
                    visit(child, f"{path}.{name}[{index}]")

    selected = {kind: SCHEMAS[kind]} if kind else SCHEMAS
    for name, schema in selected.items():
        visit(schema, f"{name}:$")
    return sorted(found)


def validate_artifact(value: Any, kind: str) -> List[str]:
    """Return every shape violation for one persisted artefact."""
    schema = SCHEMAS[kind]
    return list(_validate(value, schema, "$", schema))


def validate_session(value: Any) -> List[str]:
    """Return every session shape or cross-field violation without raising."""
    errors = validate_artifact(value, "session")
    # Invariants intentionally assume the normative shape. Running them after a
    # type error turned malformed evidence into TypeError and crashed Doctor.
    if not errors and isinstance(value, dict):
        try:
            errors.extend(check_record(value))
        except Exception as exc:  # a validator is a total error boundary
            errors.append(f"$: invariant validation failed: {type(exc).__name__}: {exc}")
    return errors


def _resolve(ref: str, root_schema: dict) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported schema reference {ref!r}")
    node: Any = root_schema
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _json_equal(left: Any, right: Any) -> bool:
    """JSON booleans are not integers, while JSON numbers share one domain."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return (left.keys() == right.keys()
                and all(_json_equal(left[key], right[key]) for key in left))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }[expected]()


def _validate(value: Any, schema: Dict[str, Any], path: str,
              root_schema: dict) -> Iterable[str]:
    if "$ref" in schema:
        yield from _validate(value, _resolve(schema["$ref"], root_schema), path,
                             root_schema)

    for branch in schema.get("allOf", []):
        yield from _validate(value, branch, path, root_schema)

    if "oneOf" in schema:
        matches = [not list(_validate(value, branch, path, root_schema))
                   for branch in schema["oneOf"]]
        if sum(matches) != 1:
            yield f"{path}: must match exactly one allowed shape"

    if "not" in schema and not list(_validate(value, schema["not"], path, root_schema)):
        yield f"{path}: matches a forbidden shape"

    condition = schema.get("if")
    if condition is not None and not list(_validate(value, condition, path, root_schema)):
        yield from _validate(value, schema.get("then", {}), path, root_schema)

    if "const" in schema and not _json_equal(value, schema["const"]):
        yield f"{path}: expected {schema['const']!r}; got {value!r}"
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        yield f"{path}: {value!r} is not an allowed value"

    expected = schema.get("type")
    if expected is not None:
        choices = [expected] if isinstance(expected, str) else expected
        if not any(_matches_type(value, choice) for choice in choices):
            yield f"{path}: expected {' or '.join(choices)}; got {type(value).__name__}"
            return

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                yield f"{path}: missing required property {name!r}"
        if schema.get("additionalProperties") is False:
            for name in value.keys() - properties.keys():
                yield f"{path}: unexpected property {name!r}"
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            for name in value.keys() - properties.keys():
                yield from _validate(value[name], additional, f"{path}.{name}", root_schema)
        for name, child in properties.items():
            if name in value:
                yield from _validate(value[name], child, f"{path}.{name}", root_schema)

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            yield f"{path}: has fewer than {schema['minItems']} items"
        if schema.get("uniqueItems"):
            for index, item in enumerate(value):
                if any(_json_equal(item, earlier) for earlier in value[:index]):
                    yield f"{path}: items must be unique"
                    break
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                yield from _validate(item, items, f"{path}[{index}]", root_schema)

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            yield f"{path}: must contain at least {schema['minLength']} characters"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            yield f"{path}: must contain at most {schema['maxLength']} characters"
        pattern = schema.get("pattern")
        if pattern and re.search(pattern, value) is None:
            yield f"{path}: does not match {pattern!r}"

    minimum = schema.get("minimum")
    if minimum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < minimum:
            yield f"{path}: must be at least {minimum}"
