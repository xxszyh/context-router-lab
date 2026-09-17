from __future__ import annotations

import json

from context_router.importers.claude_code import import_claude_code
from context_router.storage import SQLiteEventStore


def write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_claude_import_preserves_visible_turns_and_tool_lineage(tmp_path) -> None:
    source = tmp_path / "projects" / "C--repo"
    source.mkdir(parents=True)
    transcript = source / "session-1.jsonl"
    common = {
        "sessionId": "session-1",
        "cwd": "C:\\repo",
        "gitBranch": "main",
        "timestamp": "2026-09-17T01:02:03.000Z",
    }
    write_jsonl(
        transcript,
        [
            {
                **common,
                "type": "user",
                "uuid": "user-1",
                "message": {"role": "user", "content": "修复 migration.py"},
            },
            {
                **common,
                "type": "assistant",
                "uuid": "assistant-1",
                "message": {
                    "role": "assistant",
                    "model": "claude-pinned",
                    "content": [
                        {"type": "thinking", "thinking": "hidden reasoning"},
                        {"type": "text", "text": "我先检查文件。"},
                        {
                            "type": "tool_use",
                            "id": "toolu-1",
                            "name": "Read",
                            "input": {"file_path": "migration.py"},
                        },
                    ],
                },
            },
            {
                **common,
                "type": "user",
                "uuid": "tool-result-record",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu-1",
                            "content": "SQLITE_BUSY",
                        },
                        {"type": "image", "source": {"type": "base64", "data": "ignored"}},
                    ],
                },
            },
        ],
    )
    store = SQLiteEventStore(tmp_path / "events.db")

    first = import_claude_code(source.parent, store)
    second = import_claude_code(source.parent, store)
    events = store.list_events("claude:session-1")

    assert first.imported_events == 5
    assert first.skipped_thinking == 1
    assert second.imported_events == 0
    assert [event.kind for event in events] == [
        "message",
        "message",
        "tool_call",
        "tool_result",
        "artifact",
    ]
    assert "hidden reasoning" not in "\n".join(event.content for event in events)
    tool_call = next(event for event in events if event.kind == "tool_call")
    tool_result = next(event for event in events if event.kind == "tool_result")
    assert tool_result.parent_event_id == tool_call.event_id
    assert tool_call.payload["claude_tool_use_id"] == "toolu-1"
    assert all(event.payload["source"] == "claude-code" for event in events)
