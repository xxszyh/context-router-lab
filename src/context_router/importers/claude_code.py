from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from context_router.domain import RawEvent
from context_router.storage import SQLiteEventStore


@dataclass(frozen=True)
class ClaudeImportReport:
    discovered_sessions: int
    imported_sessions: int
    imported_events: int
    existing_events: int
    skipped_thinking: int
    skipped_meta_records: int
    malformed_records: int
    source_bytes: int


@dataclass(frozen=True)
class _ImportedBlock:
    actor: str
    kind: str
    content: str
    payload: dict[str, Any]
    tool_use_id: str | None = None
    parent_tool_use_id: str | None = None


def _timestamp(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return fallback


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _visible_blocks(record: dict[str, Any]) -> tuple[list[_ImportedBlock], int]:
    record_type = record.get("type")
    message = record.get("message")
    if record_type not in {"user", "assistant"} or not isinstance(message, dict):
        return [], 0
    content = message.get("content")
    if isinstance(content, str):
        if not content.strip():
            return [], 0
        actor = "assistant" if record_type == "assistant" else "user"
        return [_ImportedBlock(actor, "message", content, {})], 0
    if not isinstance(content, list):
        return [], 0

    blocks: list[_ImportedBlock] = []
    skipped_thinking = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "thinking":
            skipped_thinking += 1
        elif block_type == "text" and isinstance(block.get("text"), str):
            actor = "assistant" if record_type == "assistant" else "user"
            blocks.append(_ImportedBlock(actor, "message", block["text"], {}))
        elif block_type == "tool_use":
            name = str(block.get("name") or "unknown_tool")
            tool_input = block.get("input") or {}
            content_text = f"{name}\n{json.dumps(tool_input, ensure_ascii=False, sort_keys=True)}"
            blocks.append(
                _ImportedBlock(
                    "assistant",
                    "tool_call",
                    content_text,
                    {
                        "tool_name": name,
                        "claude_tool_use_id": block.get("id"),
                    },
                    tool_use_id=str(block.get("id")) if block.get("id") else None,
                )
            )
        elif block_type == "tool_result":
            content_text = _content_text(block.get("content"))
            if not content_text:
                content_text = "[empty tool result]"
            blocks.append(
                _ImportedBlock(
                    "tool",
                    "tool_result",
                    content_text,
                    {
                        "claude_tool_use_id": block.get("tool_use_id"),
                        "is_error": bool(block.get("is_error", False)),
                    },
                    parent_tool_use_id=(
                        str(block.get("tool_use_id")) if block.get("tool_use_id") else None
                    ),
                )
            )
        elif block_type == "image":
            source_value = block.get("source")
            source: dict[str, Any] = source_value if isinstance(source_value, dict) else {}
            blocks.append(
                _ImportedBlock(
                    "user" if record_type == "user" else "assistant",
                    "artifact",
                    "[image attachment]",
                    {
                        "media_type": source.get("media_type"),
                        "source_type": source.get("type"),
                        "binary_omitted": True,
                    },
                )
            )
    return blocks, skipped_thinking


def _transcript_files(root: Path, include_subagents: bool) -> list[Path]:
    search_root = root / "projects" if (root / "projects").is_dir() else root
    files = []
    for path in search_root.rglob("*.jsonl"):
        if not include_subagents and "subagents" in path.parts:
            continue
        files.append(path)
    return sorted(files)


def import_claude_code(
    root: str | Path,
    store: SQLiteEventStore,
    *,
    include_subagents: bool = False,
    dry_run: bool = False,
) -> ClaudeImportReport:
    """Stream Claude Code project transcripts into append-only RawEvents."""

    files = _transcript_files(Path(root), include_subagents)
    imported_events = existing_events = skipped_thinking = 0
    skipped_meta_records = malformed_records = imported_sessions = 0
    source_bytes = sum(path.stat().st_size for path in files)

    for path in files:
        fallback = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        session_id: str | None = None
        existing_ids: set[str] = set()
        emitted_position = 0
        tool_event_ids: dict[str, str] = {}
        uuid_event_ids: dict[str, str] = {}
        emitted_in_file = 0

        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_records += 1
                    continue
                if not isinstance(record, dict):
                    malformed_records += 1
                    continue
                record_session = record.get("sessionId") or record.get("session_id") or path.stem
                if session_id is None:
                    session_id = f"claude:{record_session}"
                    existing_ids = {event.event_id for event in store.list_events(session_id)}
                blocks, thinking_count = _visible_blocks(record)
                skipped_thinking += thinking_count
                if not blocks:
                    skipped_meta_records += 1
                    continue
                record_uuid = str(record.get("uuid") or f"line-{emitted_position}")
                parent_uuid = str(record.get("parentUuid") or "")
                occurred_at = _timestamp(record.get("timestamp"), fallback)
                record_first_event: str | None = None
                for block_index, block in enumerate(blocks):
                    emitted_position += 1
                    event_id = f"claude:{record_session}:{record_uuid}:{block_index}:{block.kind}"
                    if event_id in existing_ids:
                        existing_events += 1
                        if block.tool_use_id:
                            tool_event_ids[block.tool_use_id] = event_id
                        record_first_event = record_first_event or event_id
                        continue
                    parent_event_id = None
                    if block.parent_tool_use_id:
                        parent_event_id = tool_event_ids.get(block.parent_tool_use_id)
                    elif parent_uuid:
                        parent_event_id = uuid_event_ids.get(parent_uuid)
                    payload = {
                        **block.payload,
                        "source": "claude-code",
                        "source_file": str(path),
                        "claude_uuid": record_uuid,
                        "cwd": record.get("cwd"),
                        "git_branch": record.get("gitBranch"),
                        "model": (record.get("message") or {}).get("model"),
                        "slug": record.get("slug"),
                    }
                    event = RawEvent.create(
                        event_id=event_id,
                        session_id=session_id,
                        sequence=emitted_position,
                        occurred_at=occurred_at,
                        actor=block.actor,  # type: ignore[arg-type]
                        kind=block.kind,  # type: ignore[arg-type]
                        content=block.content,
                        payload=payload,
                        parent_event_id=parent_event_id,
                    )
                    if not dry_run:
                        store.append_event(event)
                    existing_ids.add(event_id)
                    imported_events += 1
                    emitted_in_file += 1
                    record_first_event = record_first_event or event_id
                    if block.tool_use_id:
                        tool_event_ids[block.tool_use_id] = event_id
                if record_first_event:
                    uuid_event_ids[record_uuid] = record_first_event
        if emitted_in_file:
            imported_sessions += 1

    return ClaudeImportReport(
        discovered_sessions=len(files),
        imported_sessions=imported_sessions,
        imported_events=imported_events,
        existing_events=existing_events,
        skipped_thinking=skipped_thinking,
        skipped_meta_records=skipped_meta_records,
        malformed_records=malformed_records,
        source_bytes=source_bytes,
    )
