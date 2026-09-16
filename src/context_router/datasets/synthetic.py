from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from context_router.domain import (
    BenchmarkQuery,
    EventContextAssignment,
    FlatContext,
    RawEvent,
)


@dataclass(frozen=True)
class SyntheticDataset:
    events: list[RawEvent]
    contexts: list[FlatContext]
    assignments: list[EventContextAssignment]
    queries: list[BenchmarkQuery]

    def write(self, directory: str | Path) -> dict[str, int]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        records_path = destination / "records.jsonl"
        with records_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record_type, records in (
                ("event", self.events),
                ("context", self.contexts),
                ("assignment", self.assignments),
            ):
                for record in records:
                    line = {"record_type": record_type, "data": record.model_dump(mode="json")}
                    handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
        benchmark_path = destination / "benchmark.jsonl"
        with benchmark_path.open("w", encoding="utf-8", newline="\n") as handle:
            for query in self.queries:
                handle.write(query.model_dump_json() + "\n")
        return {
            "sessions": len({event.session_id for event in self.events}),
            "events": len(self.events),
            "contexts": len(self.contexts),
            "assignments": len(self.assignments),
            "queries": len(self.queries),
        }


def generate_synthetic_dataset(session_count: int = 60) -> SyntheticDataset:
    if session_count < 1:
        raise ValueError("session_count must be positive")
    events: list[RawEvent] = []
    contexts: list[FlatContext] = []
    assignments: list[EventContextAssignment] = []
    queries: list[BenchmarkQuery] = []
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    for session_index in range(session_count):
        session_id = f"syn-{session_index:03d}"
        prefix = session_id
        router_id = f"{prefix}-router"
        database_id = f"{prefix}-database"
        visual_id = f"{prefix}-visual"
        context_specs = [
            (
                router_id,
                "Context Router",
                "实现关系感知的上下文路由",
                "RouteDecision 使用 RRF、校准概率和 soft routing",
                ["RouteDecision", "routing.py", "RRF"],
                2,
            ),
            (
                database_id,
                "SQLite migration",
                "修复数据库迁移和锁竞争",
                "migration.py 遇到 SQLITE_BUSY，需要检查事务与 WAL",
                ["migration.py", "SQLITE_BUSY", "WAL"],
                4,
            ),
            (
                visual_id,
                "Figure rendering",
                "生成论文图并修复坐标轴",
                "plot.py 使用 matplotlib，当前检查 legend 和 axis labels",
                ["plot.py", "matplotlib", "legend"],
                6,
            ),
        ]
        for context_id, name, goal, summary, entities, last_sequence in context_specs:
            contexts.append(
                FlatContext(
                    context_id=context_id,
                    name=name,
                    goal=goal,
                    summary=summary,
                    entities=entities,
                    lexical_terms=[name, goal],
                    status="active",
                    created_at_event=f"{prefix}-evt-{last_sequence - 1}",
                    last_active_sequence=last_sequence,
                    version=1,
                )
            )

        base_events = [
            (1, "user", "message", "我们先实现 Context Router 的候选召回。", router_id),
            (2, "assistant", "decision", "RouteDecision 将保留 RRF 和来源理由。", router_id),
            (3, "user", "message", "migration.py 报 SQLITE_BUSY。", database_id),
            (4, "assistant", "decision", "锁竞争需要检查事务范围和 WAL 配置。", database_id),
            (5, "user", "message", "plot.py 的图例遮挡坐标轴。", visual_id),
            (6, "assistant", "decision", "调整 matplotlib legend 和 axis labels。", visual_id),
        ]
        for sequence, actor, kind, content, context_id in base_events:
            raw = RawEvent.create(
                event_id=f"{prefix}-evt-{sequence}",
                session_id=session_id,
                sequence=sequence,
                occurred_at=base_time + timedelta(days=session_index, minutes=sequence),
                ingested_at=base_time + timedelta(days=session_index, minutes=sequence),
                actor=actor,  # type: ignore[arg-type]
                kind=kind,  # type: ignore[arg-type]
                content=content,
            )
            events.append(raw)
            assignments.append(
                EventContextAssignment(
                    event_id=raw.event_id,
                    context_id=context_id,
                    source="oracle",
                    relevance=1.0,
                    annotation_version=1,
                    created_at=raw.ingested_at,
                )
            )

        patterns = [
            (
                "继续这个，图例应该放哪里？",
                [visual_id],
                "continue",
                "short_coreference",
                [f"{prefix}-evt-6"],
            ),
            (
                "回到之前的 Context Router，RRF 怎么算？",
                [router_id],
                "switch_or_return",
                "return",
                [f"{prefix}-evt-2"],
            ),
            (
                "Continue tuning RouteDecision calibration.",
                [router_id],
                "continue",
                "continue",
                [f"{prefix}-evt-2"],
            ),
            (
                "现在切换到 migration.py 的 SQLITE_BUSY。",
                [database_id],
                "switch_or_return",
                "switch",
                [f"{prefix}-evt-3", f"{prefix}-evt-4"],
            ),
            (
                "把 SQLite 的检索思路应用到 Context Router。",
                [database_id, router_id],
                "cross_context",
                "cross_context",
                [f"{prefix}-evt-2", f"{prefix}-evt-4"],
            ),
            (
                "Go back to plot.py and fix the legend.",
                [visual_id],
                "switch_or_return",
                "return",
                [f"{prefix}-evt-5", f"{prefix}-evt-6"],
            ),
            (
                "Can the WAL result be shown in the matplotlib figure?",
                [database_id, visual_id],
                "cross_context",
                "cross_context",
                [f"{prefix}-evt-4", f"{prefix}-evt-6"],
            ),
            ("我想开始研究一个全新的量子化学项目。", [], "new_context", "new_context", []),
            ("那个尚未讨论的部署密钥是多少？", [], "unknown", "unanswerable", []),
            (
                "回到 RouteDecision，继续完成 soft routing。",
                [router_id],
                "switch_or_return",
                "return",
                [f"{prefix}-evt-2"],
            ),
        ]
        primary: str | None = visual_id
        recent_contexts = [visual_id, database_id, router_id]
        for offset, (text, required, relation, query_type, evidence) in enumerate(
            patterns, start=7
        ):
            event_id = f"{prefix}-evt-{offset}"
            raw = RawEvent.create(
                event_id=event_id,
                session_id=session_id,
                sequence=offset,
                occurred_at=base_time + timedelta(days=session_index, minutes=offset),
                ingested_at=base_time + timedelta(days=session_index, minutes=offset),
                actor="user",
                kind="message",
                content=text,
            )
            events.append(raw)
            queries.append(
                BenchmarkQuery(
                    sample_id=f"{session_id}-q-{offset - 6:02d}",
                    session_id=session_id,
                    query_event_id=event_id,
                    as_of_sequence=offset,
                    language="en" if text.isascii() else "mixed",
                    query_type=query_type,  # type: ignore[arg-type]
                    required_context_ids=required,
                    acceptable_evidence_sets=[evidence] if evidence else [],
                    forbidden_future_event_ids=[f"{prefix}-evt-20"],
                    relation_label=relation,  # type: ignore[arg-type]
                    answer_requirements=[],
                    must_abstain=not required,
                    difficulty="medium" if len(required) > 1 else "easy",
                    primary_context_id=primary,
                    recent_context_ids=recent_contexts[:],
                )
            )
            if required:
                primary = required[0]
                recent_contexts = list(dict.fromkeys(required + recent_contexts))[:3]

        future = RawEvent.create(
            event_id=f"{prefix}-evt-20",
            session_id=session_id,
            sequence=20,
            occurred_at=base_time + timedelta(days=session_index, minutes=20),
            ingested_at=base_time + timedelta(days=session_index, minutes=20),
            actor="assistant",
            kind="decision",
            content="未来事件：部署密钥问题在此后才被处理。",
        )
        events.append(future)

    return SyntheticDataset(events, contexts, assignments, queries)
