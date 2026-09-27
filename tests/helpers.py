"""Shared builders for tests that need canonical recurrence evidence."""
from __future__ import annotations

import hashlib

from repohone import artifacts, diagnosis, evidence, fingerprint, identity, reasoning, record, state


def ensure_session(checkout: str, host_session_id: str, repository_id=None,
                   developer_id=None) -> str:
    state.initialize(checkout)
    session_id = identity.session_id("fixture", host_session_id)
    if record.load(checkout, session_id) is None:
        rec = record.new_record(
            session_id, "fixture", host_session_id, "fixture", "test", "redacted",
            {"root": "/fixture", "id": repository_id, "checkout_id": checkout,
             "start_head": None, "start_branch": None, "branches_observed": []},
            developer_id, "fixture", None, record.now(), "/fixture")
        artifacts.save(record.record_path(checkout, session_id), rec, "session",
                       expected_checkout_id=checkout,
                       expected_artifact_id=session_id)
    return session_id


def canonical_occurrence(checkout: str, repository_id, cls: str, area: str,
                         behavior: str, session_id: str, developer_id=None,
                         diagnosis_id: str | None = None,
                         corrective_turn: int | None = None):
    if diagnosis_id is None:
        token = hashlib.sha256(
            f"{checkout}\0{session_id}\0{area}\0{behavior}".encode()).hexdigest()[:16]
        diagnosis_id = f"dx_{token}"
    created_at = record.now()

    def publish(matched, created, seen):
        mark = {"fingerprint_id": matched.fingerprint_id, "area": matched.area,
                "behavior": matched.behavior, "created": created}
        body = {
            "root_cause": {"class": cls, "summary": "fixture occurrence"},
            "confidence": "medium",
            "required_property": "fixture property",
            "corrective_turn": corrective_turn,
            "fingerprint": {"area": area, "behavior": behavior},
            "evidence_for": ["fixture evidence"],
            "evidence_against": ["fixture counter-evidence"],
            "risks": [],
        }
        rec = diagnosis.build_record(
            session_id, checkout, reasoning.SUCCESS, evidence.Selection(),
            "fixture", None, True, body=body, created_at=created_at,
            diagnosis_id=diagnosis_id, mark=mark,
            recurrence={"sessions": seen.sessions, "checkouts": seen.checkouts,
                        "developers": seen.developers, "known_fingerprints": 0,
                        "scope": "local-checkouts"},
            corrective_turn=corrective_turn, repository_id=repository_id,
            developer_id=developer_id)
        diagnosis.save(checkout, rec)

    return fingerprint.record_occurrence(
        checkout, repository_id, cls, area, behavior, session_id, diagnosis_id,
        created_at, developer_id=developer_id, publish=publish)
