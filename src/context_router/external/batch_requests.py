"""One request contract and chat-message layout for execution and token counting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from context_router.external.longmemeval_answers import object_hash

Stage = Literal["generation", "judge"]


def request_messages(request: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": request.get(
                "instructions", "Grade the provided candidate. Return only yes or no."
            ),
        },
        {"role": "user", "content": request["prompt"]},
    ]


def policy_zero(request: dict[str, Any], stage: Stage) -> bool:
    return stage == "judge" and (
        request.get("answer_completed") is False or request.get("answer_empty") is True
    )


def read_requests(path: Path, stage: Stage) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = (
        {"request_id", "prompt_sha256", "instructions", "prompt"}
        if stage == "generation"
        else {"request_id", "answer_sha256", "prompt"}
    )
    allowed = required | ({"answer_completed", "answer_empty"} if stage == "judge" else set())
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not required <= row.keys() or row.keys() - allowed:
            raise ValueError(
                "request file has the wrong stage/schema; use the exported public requests"
            )
        if any(not isinstance(row[key], str) or not row[key] for key in required):
            raise ValueError("request fields must be non-empty strings")
        key = row["request_id"]
        if key in seen:
            raise ValueError("duplicate request_id")
        if stage == "generation" and row["prompt_sha256"] != object_hash(
            {"instructions": row["instructions"], "prompt": row["prompt"]}
        ):
            raise ValueError("generation prompt failed its hash check")
        for field in ("answer_completed", "answer_empty"):
            if field in row and type(row[field]) is not bool:
                raise ValueError("completion flags must be JSON booleans")
        rows.append(row)
        seen.add(key)
    if not rows:
        raise ValueError("request file is empty")
    return rows, hashlib.sha256(data).hexdigest()
