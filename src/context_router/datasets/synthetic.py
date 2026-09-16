"""Deterministic synthetic generator for interleaved bilingual coding conversations.

The generator authors every context up front, which is the Phase 0 model: contexts
are pre-labelled and there is no Memory Writer yet. A descriptor is therefore
available from the moment its context is created, and ``last_active_sequence``
records that creation sequence rather than a running "last touched" value. A
running value would need maintained context state and would leak later turns into
earlier checkpoints, which is the one thing this benchmark must not do.

Sessions interleave six contexts over many episodes, so that:

* a query's evidence is usually several episodes behind the recent window, which
  is what makes ``query_recent_only`` and ``sliding_window`` genuinely lose;
* full history reaches thousands of tokens, so a 2 048-token memory budget
  actually binds instead of being an unreachable ceiling.

Every episode owns a distinct sub-topic, so a query names exactly one episode and
the labelled evidence stays small and precise. Every third episode answers through
a tool call/result pair, so those stay indivisible evidence. Checkpoints carry the
labels a leakage-aware evaluation needs: ``required_context_ids``, at least two
equivalent minimal evidence sets wherever the conclusion alone already answers,
``relation_label`` and the events that must not be visible yet.
"""

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
    Relation,
)

DEFAULT_CONTEXTS = 6
DEFAULT_EPISODES = 9
#: Emit one query checkpoint after this many context episodes.
QUERY_EVERY = 4
#: How many upcoming events to list as forbidden for a checkpoint.
FORBIDDEN_SAMPLE = 3
#: Trailing events kept after the last checkpoint so every checkpoint has a future.
TRAILING_EVENTS = 3


@dataclass(frozen=True)
class _Blueprint:
    key: str
    name: str
    goal: str
    summary: str
    entities: tuple[str, str, str]
    subtopics: tuple[str, ...]

    @property
    def anchor(self) -> str:
        return self.entities[0]


_BLUEPRINTS: tuple[_Blueprint, ...] = (
    _Blueprint(
        key="router",
        name="Context Router",
        goal="实现关系感知的上下文路由",
        summary="RouteDecision 使用 RRF、校准概率和 soft routing",
        entities=("routing.py", "RouteDecision", "RRF"),
        subtopics=(
            "候选召回",
            "RRF 融合",
            "概率校准",
            "soft routing",
            "弃答阈值",
            "Relation Lite",
            "因果截断",
            "证据来源标记",
            "候选重排",
        ),
    ),
    _Blueprint(
        key="database",
        name="SQLite migration",
        goal="修复数据库迁移和锁竞争",
        summary="migration.py 遇到 SQLITE_BUSY，需要检查事务与 WAL",
        entities=("migration.py", "SQLITE_BUSY", "WAL"),
        subtopics=(
            "事务范围",
            "锁升级",
            "重试退避",
            "连接池上限",
            "迁移回滚",
            "索引重建",
            "批处理提交",
            "schema 版本表",
            "长事务拆分",
        ),
    ),
    _Blueprint(
        key="visual",
        name="Figure rendering",
        goal="生成论文图并修复坐标轴",
        summary="plot.py 使用 matplotlib，当前检查 legend 和 axis labels",
        entities=("plot.py", "matplotlib", "legend"),
        subtopics=(
            "图例遮挡",
            "坐标轴标签",
            "子图间距",
            "配色可读性",
            "字体嵌入",
            "矢量导出",
            "误差棒",
            "双轴刻度",
            "归一化色标",
        ),
    ),
    _Blueprint(
        key="forecast",
        name="Demand forecast",
        goal="修正需求预测的滞后特征",
        summary="forecast.py 的 lag_feature 存在泄漏，需要重做切分",
        entities=("forecast.py", "lag_feature", "leakage"),
        subtopics=(
            "特征泄漏",
            "滚动窗口",
            "季节性差分",
            "缺失值填充",
            "目标编码",
            "交叉验证切分",
            "残差诊断",
            "外推边界",
            "评估口径",
        ),
    ),
    _Blueprint(
        key="solver",
        name="MIP solver tuning",
        goal="调优混合整数规划的求解器参数",
        summary="solver.py 的 MIP_gap 与 timeout 互相冲突",
        entities=("solver.py", "MIP_gap", "timeout"),
        subtopics=(
            "时间上限",
            "间隙容差",
            "分支策略",
            "割平面",
            "预处理强度",
            "对称性破除",
            "初始可行解",
            "数值稳定性",
            "并行线程数",
        ),
    ),
    _Blueprint(
        key="report",
        name="LaTeX report build",
        goal="修复 LaTeX 编译与参考文献",
        summary="main.tex 的 bibtex 引用顺序错乱，交叉引用未解析",
        entities=("main.tex", "bibtex", "reference"),
        subtopics=(
            "引用顺序",
            "编译链",
            "中文断行",
            "浮动体位置",
            "交叉引用",
            "参考文献去重",
            "图表编号",
            "页边距",
            "拼写检查",
        ),
    ),
)

#: Query templates cycled per session. Every query type in the plan appears.
_QUERY_CYCLE: tuple[str, ...] = (
    "return",
    "continue",
    "cross_context",
    "switch",
    "short_coreference",
    "return",
    "new_context",
    "unanswerable",
)

_RELATION_BY_TYPE: dict[str, Relation] = {
    "continue": "continue",
    "return": "switch_or_return",
    "switch": "switch_or_return",
    "short_coreference": "continue",
    "cross_context": "cross_context",
    "new_context": "new_context",
    "unanswerable": "unknown",
}

_DIFFICULTY_BY_TYPE = {
    "continue": "easy",
    "short_coreference": "easy",
    "switch": "medium",
    "return": "hard",
    "cross_context": "hard",
    "new_context": "hard",
    "unanswerable": "hard",
}


@dataclass(frozen=True)
class _QuerySlot:
    sequence: int
    query_type: str
    target_keys: list[str]
    evidence_ids: list[str]


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


def queries_per_session(
    contexts_per_session: int = DEFAULT_CONTEXTS,
    episodes_per_context: int = DEFAULT_EPISODES,
) -> int:
    """Checkpoints one session produces, so callers never hard-code the count."""

    return len(range(QUERY_EVERY, contexts_per_session * episodes_per_context + 1, QUERY_EVERY))


def _blueprints_for(contexts_per_session: int) -> tuple[_Blueprint, ...]:
    if not 1 <= contexts_per_session <= len(_BLUEPRINTS):
        raise ValueError(f"contexts_per_session must be between 1 and {len(_BLUEPRINTS)}")
    return _BLUEPRINTS[:contexts_per_session]


def _topic(blueprint: _Blueprint, episode_index: int) -> str:
    return blueprint.subtopics[episode_index % len(blueprint.subtopics)]


def _episode_turns(
    blueprint: _Blueprint, episode_index: int, english: bool
) -> tuple[str, str, str]:
    """Return the user turn, the topic, and the assistant reply for one episode.

    Turns are deliberately multi-clause, the way real agent transcripts are. Short
    one-liners would leave full history under the memory budget, and a budget that
    never binds cannot separate the arms.
    """

    topic = _topic(blueprint, episode_index)
    anchor, marker, cross = blueprint.entities
    if english:
        user = (
            f"Back to {anchor}: how should we handle the {topic} step? "
            f"{marker} is still unstable, so I want the blast radius before touching code."
        )
        reply = (
            f"Check {topic} in {anchor} against {cross} first; if {marker} stays "
            f"unstable, roll back to the last good revision instead of patching it."
        )
    else:
        user = (
            f"{anchor} 的{topic}这一步怎么处理？{marker} 目前还不稳定，"
            f"我想先确认影响范围再动手改代码。"
        )
        reply = (
            f"先确认 {anchor} 在{topic}上的约束，再对照 {cross} 调整；"
            f"{marker} 若仍然不稳定，就回退到上一个可用版本，不要继续打补丁。"
        )
    return user, topic, reply


def _session_plan(
    *,
    session_id: str,
    session_index: int,
    blueprints: tuple[_Blueprint, ...],
    episodes_per_context: int,
) -> tuple[list[tuple[str, int, bool]], list[_QuerySlot]]:
    """Decide the emission order and which checkpoints to take, before emitting.

    Checkpoint targets are always contexts that have already appeared and are as
    cold as possible, so a "return" genuinely reaches back and always has evidence.
    """

    order = [
        blueprints[(index + session_index) % len(blueprints)] for index in range(len(blueprints))
    ]
    emissions: list[tuple[str, int, bool]] = []
    slots: list[_QuerySlot] = []
    last_touched: dict[str, int] = {}
    emitted = 0
    for episode_index in range(episodes_per_context):
        for position, blueprint in enumerate(order):
            english = (episode_index + position) % 2 == 1
            emissions.append((blueprint.key, episode_index, english))
            emitted += 1
            last_touched[blueprint.key] = emitted
            if emitted % QUERY_EVERY:
                continue
            coldest = sorted(
                (key for key in last_touched if key != blueprint.key),
                key=lambda key: (last_touched[key], key),
            )
            query_type = _QUERY_CYCLE[(emitted // QUERY_EVERY - 1) % len(_QUERY_CYCLE)]
            if query_type in ("continue", "short_coreference"):
                targets = [blueprint.key]
            elif query_type == "cross_context":
                # Both halves must be old contexts, otherwise the recent window
                # alone would satisfy a query that is supposed to need two.
                targets = coldest[:2] if len(coldest) >= 2 else ([blueprint.key] + coldest)[:2]
            elif query_type in ("new_context", "unanswerable"):
                targets = []
            elif query_type == "return":
                targets = coldest[:1]
            else:
                targets = coldest[1:2] or coldest[:1]
            slots.append(
                _QuerySlot(sequence=0, query_type=query_type, target_keys=targets, evidence_ids=[])
            )
    return emissions, slots


def _query_text(
    query_type: str,
    target_keys: list[str],
    topics: dict[str, str],
    blueprint_by_key: dict[str, _Blueprint],
) -> str:
    if query_type == "new_context":
        return "我想开始研究一个全新的量子化学项目，先看反应路径。"
    if query_type == "unanswerable":
        return "那个尚未讨论的部署密钥是多少？"
    if query_type == "cross_context":
        left = blueprint_by_key[target_keys[0]]
        right = blueprint_by_key[target_keys[1]]
        left_topic = topics.get(left.key, left.subtopics[0])
        right_topic = topics.get(right.key, right.subtopics[0])
        # Both halves must name what is wanted from them. Naming only the context would
        # make the labelled evidence unanswerable: every episode of a context shares the
        # same anchor file and marker, so nothing in the query would distinguish the
        # labelled episode from its siblings.
        return f"把 {left.name} 的{left_topic}思路用到 {right.name} 的{right_topic}上，可行吗？"
    target = blueprint_by_key[target_keys[0]]
    topic = topics.get(target.key, target.subtopics[0])
    if query_type == "return":
        return f"回到 {target.name}，{topic}这一步最终怎么定？"
    if query_type == "switch":
        return f"现在切换到 {target.name}，先看 {topic} 的结论。"
    if query_type == "short_coreference":
        return f"继续这个，{target.entities[2]} 应该放在哪里？"
    return f"Continue on {target.name}: {topic} still needs a decision."


def _build_session(
    *,
    session_id: str,
    session_index: int,
    blueprints: tuple[_Blueprint, ...],
    episodes_per_context: int,
    base_time: datetime,
) -> tuple[
    list[RawEvent],
    list[EventContextAssignment],
    list[tuple[_QuerySlot, str]],
    dict[str, RawEvent],
]:
    """Emit one interleaved session plus its labelled checkpoints."""

    blueprint_by_key = {blueprint.key: blueprint for blueprint in blueprints}
    emissions, slots = _session_plan(
        session_id=session_id,
        session_index=session_index,
        blueprints=blueprints,
        episodes_per_context=episodes_per_context,
    )
    events: list[RawEvent] = []
    assignments: list[EventContextAssignment] = []
    first_event: dict[str, RawEvent] = {}
    latest_episode: dict[str, list[str]] = {}
    latest_topic: dict[str, str] = {}
    labelled: list[tuple[_QuerySlot, str]] = []
    sequence = 0
    emitted = 0
    slot_index = 0

    def emit(actor: str, kind: str, content: str, *, parent: str | None = None) -> RawEvent:
        nonlocal sequence
        sequence += 1
        event = RawEvent.create(
            event_id=f"{session_id}-evt-{sequence}",
            session_id=session_id,
            sequence=sequence,
            occurred_at=base_time + timedelta(minutes=sequence),
            ingested_at=base_time + timedelta(minutes=sequence),
            actor=actor,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            content=content,
            parent_event_id=parent,
        )
        events.append(event)
        return event

    def assign(event: RawEvent, key: str, relevance: float) -> None:
        assignments.append(
            EventContextAssignment(
                event_id=event.event_id,
                context_id=f"{session_id}-{key}",
                source="oracle",
                relevance=relevance,
                annotation_version=1,
                created_at=event.ingested_at,
            )
        )

    for key, episode_index, english in emissions:
        blueprint = blueprint_by_key[key]
        user_text, topic, reply_text = _episode_turns(blueprint, episode_index, english)
        user_event = emit("user", "message", user_text)
        assign(user_event, key, 1.0)
        first_event.setdefault(key, user_event)
        episode_ids = [user_event.event_id]

        if episode_index % 3 == 2:
            # Every third episode answers through a tool, so call/result pairs
            # stay indivisible evidence in the benchmark.
            call = emit(
                "assistant",
                "tool_call",
                f"读取 {blueprint.anchor} 中与{topic}相关的日志与最近三次改动，"
                f"重点定位 {blueprint.entities[1]} 的出现位置",
            )
            assign(call, key, 1.0)
            result = emit(
                "tool",
                "tool_result",
                f"{topic}检查完成：命中 {blueprint.entities[1]} 相关记录两条，"
                f"结论是 {blueprint.entities[2]} 需要先收敛，再判断是否回退。",
                parent=call.event_id,
            )
            assign(result, key, 1.0)
            episode_ids.extend([call.event_id, result.event_id])
        else:
            reply = emit("assistant", "decision", reply_text)
            assign(reply, key, 1.0)
            episode_ids.append(reply.event_id)

        latest_episode[key] = episode_ids
        latest_topic[key] = topic
        emitted += 1

        if emitted % QUERY_EVERY:
            continue
        slot = slots[slot_index]
        slot_index += 1
        text = _query_text(slot.query_type, slot.target_keys, latest_topic, blueprint_by_key)
        query_event = emit("user", "message", text)
        evidence = [
            event_id for target in slot.target_keys for event_id in latest_episode.get(target, [])
        ]
        labelled.append(
            (
                _QuerySlot(
                    sequence=query_event.sequence,
                    query_type=slot.query_type,
                    target_keys=list(slot.target_keys),
                    evidence_ids=evidence,
                ),
                text,
            )
        )

    for index in range(TRAILING_EVENTS):
        blueprint = blueprints[index % len(blueprints)]
        tail = emit(
            "assistant",
            "decision",
            f"{blueprint.anchor}：本轮{_topic(blueprint, index)}的修改已合并并通过本地检查，"
            f"等待复核；下一步要确认 {blueprint.entities[1]} 没有回归。",
        )
        assign(tail, blueprint.key, 0.6)

    return events, assignments, labelled, first_event


def generate_synthetic_dataset(
    session_count: int = 60,
    *,
    contexts_per_session: int = DEFAULT_CONTEXTS,
    episodes_per_context: int = DEFAULT_EPISODES,
    session_start: int = 0,
) -> SyntheticDataset:
    """Generate sessions ``session_start`` .. ``session_start + session_count - 1``.

    ``session_start`` exists so a train / dev / test split can be generated as disjoint
    session ranges instead of being hoped for. A ranker trained on one range and scored on
    another is the only way the numbers mean anything.
    """

    if session_count < 1:
        raise ValueError("session_count must be positive")
    if session_start < 0:
        raise ValueError("session_start must not be negative")
    if episodes_per_context < QUERY_EVERY:
        raise ValueError(f"episodes_per_context must be at least {QUERY_EVERY}")
    blueprints = _blueprints_for(contexts_per_session)
    events: list[RawEvent] = []
    contexts: list[FlatContext] = []
    assignments: list[EventContextAssignment] = []
    queries: list[BenchmarkQuery] = []
    base_time = datetime(2026, 1, 1, tzinfo=UTC)

    for session_index in range(session_start, session_start + session_count):
        session_id = f"syn-{session_index:03d}"
        session_events, session_assignments, labelled, first_event = _build_session(
            session_id=session_id,
            session_index=session_index,
            blueprints=blueprints,
            episodes_per_context=episodes_per_context,
            base_time=base_time + timedelta(days=session_index),
        )
        events.extend(session_events)
        assignments.extend(session_assignments)

        primary: str | None = None
        recent_contexts: list[str] = []
        for position, (slot, text) in enumerate(labelled, start=1):
            required = [f"{session_id}-{key}" for key in slot.target_keys]
            causal_evidence = [event_id for event_id in slot.evidence_ids]
            evidence_sets: list[list[str]] = []
            if causal_evidence:
                evidence_sets.append(list(causal_evidence))
                # The conclusion alone is an equivalent minimal answer, so the
                # benchmark never assumes a single correct evidence path.
                evidence_sets.append([causal_evidence[-1]])
            future = [event.event_id for event in session_events if event.sequence > slot.sequence]
            queries.append(
                BenchmarkQuery(
                    sample_id=f"{session_id}-q-{position:02d}",
                    session_id=session_id,
                    query_event_id=f"{session_id}-evt-{slot.sequence}",
                    as_of_sequence=slot.sequence,
                    language="en" if text.isascii() else "mixed",
                    query_type=slot.query_type,  # type: ignore[arg-type]
                    required_context_ids=required,
                    acceptable_evidence_sets=evidence_sets,
                    forbidden_future_event_ids=future[:FORBIDDEN_SAMPLE],
                    relation_label=_RELATION_BY_TYPE[slot.query_type],
                    answer_requirements=[],
                    must_abstain=not required,
                    difficulty=_DIFFICULTY_BY_TYPE[slot.query_type],  # type: ignore[arg-type]
                    primary_context_id=primary,
                    recent_context_ids=list(recent_contexts),
                )
            )
            if required:
                primary = required[0]
                recent_contexts = list(dict.fromkeys(required + recent_contexts))[:3]

        for blueprint in blueprints:
            created = first_event[blueprint.key]
            contexts.append(
                FlatContext(
                    context_id=f"{session_id}-{blueprint.key}",
                    name=blueprint.name,
                    goal=blueprint.goal,
                    summary=blueprint.summary,
                    entities=list(blueprint.entities),
                    lexical_terms=[blueprint.name, blueprint.goal],
                    status="active",
                    created_at_event=created.event_id,
                    last_active_sequence=created.sequence,
                    version=1,
                )
            )

    return SyntheticDataset(events, contexts, assignments, queries)
