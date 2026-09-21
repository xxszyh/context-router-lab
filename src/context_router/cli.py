from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from context_router.assembly import assemble_context
from context_router.datasets import (
    AnswerAdjudicationSet,
    SecondaryAnswerAnnotationSet,
    audit_answer_adjudication,
    audit_secondary_answer_annotations,
    generate_synthetic_dataset,
    validate_dataset,
)
from context_router.datasets.real_replay import RealReplayAnnotation, answer_after
from context_router.domain import (
    AssemblyRequest,
    BenchmarkQuery,
    FlatContext,
    RawEvent,
    RouteRequest,
)
from context_router.evaluation import (
    ARM_NAMES,
    AnswerRecord,
    ArmCase,
    ArmName,
    RouteCaseResult,
    answer_one,
    build_answer_cases,
    build_arm_cases,
    describe_router,
    evaluate_answers,
    evaluate_arms,
    evaluate_routes,
    first_gate,
    quality_token_frontier,
    run_arm,
    run_pairwise_judging,
    summarise_judge_strata,
    summarise_wins,
)
from context_router.evaluation.judge import (
    CoverageJudge,
    Judge,
    JudgePair,
    LLMJudge,
    RefusalGatedJudge,
)
from context_router.evaluation.necessity import (
    context_material,
    control_failures,
    required_from_ablations,
    run_ablations,
)
from context_router.importers import import_claude_code
from context_router.providers import HashEmbeddingProvider
from context_router.providers.anthropic import (
    AnthropicCompatibleAnswerProvider,
    AnthropicCompatibleVerdictModel,
)
from context_router.retrieval import BM25Index, LexicalAnalyzer
from context_router.routing import ContextRouter
from context_router.routing.calibration import PlattContextRanker
from context_router.routing.policy import PolicyObservation, SelectionPolicyTuner
from context_router.routing.router import RoutingPolicy
from context_router.storage import SQLiteEventStore

app = typer.Typer(no_args_is_help=True, help="Context routing research harness.")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_benchmark(path: Path) -> list[BenchmarkQuery]:
    with path.open(encoding="utf-8") as handle:
        return [BenchmarkQuery.model_validate_json(line) for line in handle if line.strip()]


@app.command("audit-secondary-annotations")
def audit_secondary_annotations_command(
    annotations: Annotated[Path, typer.Argument(help="Primary real-replay annotation JSON")],
    secondary: Annotated[Path, typer.Argument(help="Independent answer-annotation JSON")],
    database: Annotated[Path, typer.Argument(help="Local event store")],
    output: Annotated[Path, typer.Argument(help="Audit JSON output")],
) -> None:
    """Audit an independent label set without overwriting or adjudicating the gold labels."""

    primary = RealReplayAnnotation.model_validate_json(annotations.read_text(encoding="utf-8"))
    second = SecondaryAnswerAnnotationSet.model_validate_json(secondary.read_text(encoding="utf-8"))
    events = SQLiteEventStore(database).list_events(primary.source_session_id)
    answers = {
        checkpoint.sample_id: answer_after(events, checkpoint.as_of_sequence)
        for checkpoint in primary.checkpoints
    }
    audit = audit_secondary_answer_annotations(second, primary, answers)
    _write_json(output, audit.model_dump(mode="json"))
    typer.echo(str(output))


@app.command("audit-answer-adjudication")
def audit_answer_adjudication_command(
    annotations: Annotated[Path, typer.Argument(help="Primary real-replay annotation JSON")],
    secondary: Annotated[Path, typer.Argument(help="Independent answer-annotation JSON")],
    adjudication: Annotated[Path, typer.Argument(help="Completed adjudication JSON")],
    database: Annotated[Path, typer.Argument(help="Local event store")],
    output: Annotated[Path, typer.Argument(help="Audit JSON output")],
) -> None:
    """Check final labels and expose decision/text contradictions without rewriting them."""

    primary = RealReplayAnnotation.model_validate_json(annotations.read_text(encoding="utf-8"))
    second = SecondaryAnswerAnnotationSet.model_validate_json(secondary.read_text(encoding="utf-8"))
    final = AnswerAdjudicationSet.model_validate_json(adjudication.read_text(encoding="utf-8"))
    events = SQLiteEventStore(database).list_events(primary.source_session_id)
    answers = {
        checkpoint.sample_id: answer_after(events, checkpoint.as_of_sequence)
        for checkpoint in primary.checkpoints
    }
    audit = audit_answer_adjudication(final, primary, second, answers)
    _write_json(output, audit.model_dump(mode="json"))
    typer.echo(str(output))


def _session_contexts(
    store: SQLiteEventStore, session_id: str, as_of_sequence: int | None = None
) -> list[FlatContext]:
    """Contexts of a session that already exist at the checkpoint.

    Filtering on the whole session would expose contexts created in the future of
    the checkpoint, which is exactly the leak causal replay exists to prevent.
    """

    sequence_of = {event.event_id: event.sequence for event in store.list_events(session_id)}
    cutoff = as_of_sequence if as_of_sequence is not None else max(sequence_of.values(), default=0)
    return [
        context
        for context in store.list_contexts()
        if sequence_of.get(context.created_at_event, cutoff + 1) <= cutoff
    ]


@app.command("generate-synthetic")
def generate_synthetic(
    output: Annotated[Path, typer.Argument(help="Output dataset directory")],
    sessions: Annotated[int, typer.Option(min=1)] = 60,
    session_start: Annotated[int, typer.Option(min=0, help="First session index")] = 0,
) -> None:
    """Generate deterministic bilingual coding conversations.

    Use a distinct --session-start per split so train and test conversations never
    overlap.
    """

    counts = generate_synthetic_dataset(sessions, session_start=session_start).write(output)
    typer.echo(json.dumps(counts, ensure_ascii=False, sort_keys=True))


@app.command()
def ingest(
    database: Annotated[Path, typer.Argument()],
    jsonl: Annotated[Path, typer.Argument()],
) -> None:
    counts = SQLiteEventStore(database).import_jsonl(jsonl)
    typer.echo(json.dumps(counts, ensure_ascii=False, sort_keys=True))


@app.command("ingest-claude")
def ingest_claude(
    database: Annotated[Path, typer.Argument(help="Local append-only SQLite database")],
    source: Annotated[
        Path,
        typer.Argument(help="Claude home, projects directory, or one project directory"),
    ],
    include_subagents: Annotated[bool, typer.Option()] = False,
    dry_run: Annotated[bool, typer.Option()] = False,
) -> None:
    """Import visible Claude Code turns and tool lineage; omit hidden thinking."""

    report = import_claude_code(
        source,
        SQLiteEventStore(database),
        include_subagents=include_subagents,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(asdict(report), ensure_ascii=False, indent=2))


@app.command("validate-dataset")
def validate_dataset_command(
    database: Annotated[Path, typer.Argument()],
    benchmark: Annotated[Path, typer.Argument()],
) -> None:
    report = validate_dataset(SQLiteEventStore(database), _read_benchmark(benchmark))
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise typer.Exit(1)


@app.command("build-index")
def build_index(
    database: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
) -> None:
    store = SQLiteEventStore(database)
    contexts = store.list_contexts()
    provider = HashEmbeddingProvider()
    vectors = provider.embed([context.searchable_text() for context in contexts])
    payload = {
        "schema_version": "1.0",
        "model_version": provider.model_version,
        "context_ids": [context.context_id for context in contexts],
        "vectors": vectors,
    }
    payload["index_version"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    _write_json(output, payload)
    typer.echo(str(output))


@app.command("route")
def route_command(
    database: Annotated[Path, typer.Argument()],
    session_id: Annotated[str, typer.Argument()],
    query: Annotated[str, typer.Argument()],
    as_of: Annotated[int | None, typer.Option()] = None,
    primary: Annotated[str | None, typer.Option()] = None,
) -> None:
    store = SQLiteEventStore(database)
    events = store.list_events(session_id, as_of_sequence=as_of)
    cutoff = as_of if as_of is not None else (events[-1].sequence if events else 0)
    request = RouteRequest(
        query_event_id="cli-query",
        query=query,
        recent_events=events[-6:],
        primary_context_id=primary,
        recent_context_ids=[],
        context_catalog=_session_contexts(store, session_id, cutoff),
        as_of_sequence=cutoff,
    )
    typer.echo(ContextRouter().route(request).model_dump_json(indent=2))


@app.command("assemble")
def assemble_command(
    database: Annotated[Path, typer.Argument()],
    session_id: Annotated[str, typer.Argument()],
    query: Annotated[str, typer.Argument()],
    token_budget: Annotated[int, typer.Option(min=32)] = 2048,
    as_of: Annotated[int | None, typer.Option()] = None,
    primary: Annotated[str | None, typer.Option()] = None,
) -> None:
    store = SQLiteEventStore(database)
    events = store.list_events(session_id, as_of_sequence=as_of)
    cutoff = as_of if as_of is not None else (events[-1].sequence if events else 0)
    contexts = _session_contexts(store, session_id, cutoff)
    route_request = RouteRequest(
        query_event_id="cli-query",
        query=query,
        recent_events=events[-6:],
        primary_context_id=primary,
        recent_context_ids=[],
        context_catalog=contexts,
        as_of_sequence=cutoff,
    )
    route_decision = ContextRouter().route(route_request)
    assembly_request = AssemblyRequest(
        query=query,
        recent_events=events[-6:],
        event_pool=events,
        context_catalog=contexts,
        assignments=store.list_assignments(session_id=session_id, latest_only=True),
        as_of_sequence=cutoff,
        token_budget=token_budget,
    )
    typer.echo(assemble_context(assembly_request, route_decision).model_dump_json(indent=2))


def _stratified_sample(cases: list[ArmCase], limit: int) -> list[ArmCase]:
    """Round-robin across query types so a small run still covers every kind of request.

    Taking the first N checkpoints would be deterministic but would cover the query cycle
    once and a half, and taking every k-th would cover the same phase of it every time,
    since each session runs the same cycle. Sampling across types is what makes a 20-
    checkpoint run informative rather than merely cheap.
    """

    if limit >= len(cases):
        return list(cases)
    buckets: dict[str, list[ArmCase]] = {}
    for case in cases:
        buckets.setdefault(case.query_type, []).append(case)
    # Spread within each type as well as across types. Taking each bucket from the front
    # would balance the query types while quietly drawing every one of them from the first
    # few sessions.
    per_type = max(1, limit // len(buckets)) + 1
    spread: dict[str, list[ArmCase]] = {}
    for key, bucket in buckets.items():
        take = min(per_type, len(bucket))
        if take <= 1:
            spread[key] = list(bucket)
            continue
        step = (len(bucket) - 1) / (take - 1)
        spread[key] = [bucket[round(index * step)] for index in range(take)]
    sampled: list[ArmCase] = []
    depth = 0
    while len(sampled) < limit:
        progressed = False
        for bucket in spread.values():
            if depth < len(bucket) and len(sampled) < limit:
                sampled.append(bucket[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    # Sparse query-type buckets can exhaust their quota before ``limit``. Fill the tail from
    # the still-unselected timeline instead of silently running fewer paid calls than reported.
    selected_ids = {case.sample_id for case in sampled}
    sampled.extend(
        case for case in cases if case.sample_id not in selected_ids and len(sampled) < limit
    )
    return sampled


@app.command("answer-experiment")
def answer_experiment_command(
    database: Annotated[Path, typer.Argument()],
    benchmark: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
    arms: Annotated[str, typer.Option(help="Comma-separated arm names")] = (
        "query_recent_only,hybrid_router,oracle_router"
    ),
    limit: Annotated[int, typer.Option(min=1, help="Checkpoints per arm")] = 20,
    model: Annotated[str | None, typer.Option(help="Pinned model id")] = None,
    base_url: Annotated[str | None, typer.Option(help="Endpoint base url")] = None,
    auth_style: Annotated[str, typer.Option(help="bearer or x-api-key")] = "bearer",
    max_tokens: Annotated[int, typer.Option(min=64)] = 800,
    token_budget: Annotated[int, typer.Option(min=32)] = 2048,
    index_head_chars: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Body characters the indexed arm keeps per event; it does not change selection",
        ),
    ] = None,
    ranker_file: Annotated[Path | None, typer.Option()] = None,
    policy_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Ask a real main model to answer each arm's memory, and score what comes back.

    The credential is read from the environment (ANTHROPIC_AUTH_TOKEN by default) and is
    deliberately not a command-line option: flags land in shell history and in the process
    list, and a research harness should not be the reason a key leaks.
    """

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
    resolved_model = model or os.environ.get("ANTHROPIC_MODEL", "")
    if not (api_key and resolved_base and resolved_model):
        raise typer.BadParameter(
            "need ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL and a pinned model "
            "(env or --model/--base-url)"
        )
    provider = AnthropicCompatibleAnswerProvider(
        base_url=resolved_base,
        api_key=api_key,
        model=resolved_model,
        auth_style=auth_style,  # type: ignore[arg-type]
        max_tokens=max_tokens,
    )
    store = SQLiteEventStore(database)
    cases = _stratified_sample(
        build_answer_cases(store, _read_benchmark(benchmark), budget=token_budget), limit
    )
    ranker = PlattContextRanker.load(ranker_file) if ranker_file else None
    policy = RoutingPolicy(**json.loads(policy_file.read_text())) if policy_file else None
    router = ContextRouter(ranker=ranker, policy=policy)
    chosen = [name.strip() for name in arms.split(",") if name.strip()]
    typer.echo(
        f"{len(chosen)} arms x {len(cases)} checkpoints = {len(chosen) * len(cases)} calls; "
        f"model={resolved_model} max_tokens={max_tokens}"
    )
    records: list[AnswerRecord] = []
    failures: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        for name in chosen:
            try:
                record = answer_one(
                    case,
                    cast(ArmName, name),
                    provider,
                    router=router,
                    index_head_chars=index_head_chars,
                )
            except Exception as error:  # noqa: BLE001 - one failed call must not lose the run
                typer.echo(f"  [{index}/{len(cases)}] {name} FAILED: {type(error).__name__}")
                response = getattr(error, "response", None)
                failure: dict[str, object] = {
                    "sample_id": case.sample_id,
                    "arm": name,
                    "error_type": type(error).__name__,
                }
                if response is not None:
                    failure["http_status"] = getattr(response, "status_code", None)
                    try:
                        payload = response.json()
                    except Exception:  # noqa: BLE001 - an HTML error page is still a failure
                        payload = None
                    if isinstance(payload, dict):
                        failure["provider_error"] = payload.get("error", payload.get("type"))
                failures.append(failure)
                continue
            records.append(record)
            typer.echo(
                f"  [{index}/{len(cases)}] {name:<18} mem={record.memory_tokens:>5} "
                f"in={record.input_tokens:>5} out={record.output_tokens:>5} "
                f"cov={record.coverage:.2f}{' TRUNCATED' if record.truncated else ''}"
            )
    if not records:
        raise typer.BadParameter("no answer records were produced")
    summary = evaluate_answers(records)
    payload = {
        "schema_version": "1.0",
        "model": resolved_model,
        "max_tokens": max_tokens,
        "token_budget": token_budget,
        "arms": chosen,
        "checkpoints": len(cases),
        "expected_calls": len(chosen) * len(cases),
        "completed_calls": len(records),
        "failures": failures,
        "summary": summary,
        "frontier": [point.__dict__ for point in quality_token_frontier(summary)],
        "records": [record.model_dump(mode="json") for record in records],
    }
    _write_json(output, payload)
    typer.echo(str(output))


@app.command("judge-answers")
def judge_answers_command(
    answers: Annotated[Path, typer.Argument(help="Output of answer-experiment")],
    output: Annotated[Path, typer.Argument()],
    arm_a: Annotated[str, typer.Option(help="First arm in each pair")] = "hybrid_router",
    arm_b: Annotated[str, typer.Option(help="Second arm in each pair")] = "query_recent_only",
    limit: Annotated[int, typer.Option(min=1)] = 20,
    model: Annotated[str | None, typer.Option(help="Pinned model id")] = None,
    base_url: Annotated[str | None, typer.Option()] = None,
    auth_style: Annotated[str, typer.Option(help="bearer or x-api-key")] = "bearer",
    max_tokens: Annotated[int, typer.Option(min=64)] = 4096,
    judge_kind: Annotated[str, typer.Option("--judge", help="llm or coverage")] = "llm",
    refusal_gate: Annotated[
        bool,
        typer.Option(
            "--refusal-gate/--no-refusal-gate",
            help="Tie two explicit refusals before calling the judge",
        ),
    ] = True,
) -> None:
    """Blind-judge two arms' answers pairwise, in both orders, and tally the wins.

    Reads the saved answer records rather than the dataset, so re-judging after fixing the
    judge costs calls but never re-runs the answers.
    """

    payload = json.loads(answers.read_text(encoding="utf-8"))
    by_sample: dict[str, dict[str, Any]] = {}
    for row in payload["records"]:
        by_sample.setdefault(row["sample_id"], {})[row["arm"]] = row
    pairs: list[JudgePair] = []
    for sample_id, arms in by_sample.items():
        if arm_a in arms and arm_b in arms:
            left, right = arms[arm_a], arms[arm_b]
            # Read from `expects_refusal`, not `must_abstain`: the latter is set whenever the
            # reference reply drew on no labelled context, which includes the case where it
            # answered anyway. Older run files predate the field and carry no refusal
            # expectation, which is the correct default for them -- no real checkpoint in this
            # dataset is `unanswerable`.
            left_expects = left.get("expects_refusal", False)
            right_expects = right.get("expects_refusal", False)
            if not isinstance(left_expects, bool) or not isinstance(right_expects, bool):
                raise typer.BadParameter(f"{sample_id}: expects_refusal labels must be boolean")
            if left_expects != right_expects:
                raise typer.BadParameter(
                    f"{sample_id}: expects_refusal labels disagree between arms"
                )
            pairs.append(
                JudgePair(
                    sample_id=sample_id,
                    query=left["query"],
                    requirements=list(left["answer_requirements"]),
                    arm_a=arm_a,
                    arm_b=arm_b,
                    answer_a=left["answer"],
                    answer_b=right["answer"],
                    expects_refusal=left_expects,
                )
            )
        if len(pairs) >= limit:
            break
    if not pairs:
        raise typer.BadParameter(f"no checkpoint has both {arm_a} and {arm_b}")
    # The deterministic coverage judge is the control: it runs the identical swap
    # protocol with no model and no credential, so "is the paid judge adding anything"
    # is answerable for free.
    base_judge: Judge
    if judge_kind == "coverage":
        base_judge = CoverageJudge()
    else:
        api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL", "")
        if not (api_key and resolved_base and resolved_model):
            raise typer.BadParameter(
                "need ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL and a model (or --judge coverage)"
            )
        base_judge = LLMJudge(
            AnthropicCompatibleVerdictModel(
                base_url=resolved_base,
                api_key=api_key,
                model=resolved_model,
                auth_style=auth_style,  # type: ignore[arg-type]
                max_tokens=max_tokens,
            )
        )
    judge: Judge = RefusalGatedJudge(base_judge) if refusal_gate else base_judge
    typer.echo(
        f"{len(pairs)} pairs x 2 orders <= {len(pairs) * 2} judge calls; "
        f"judge={judge.model_version}"
    )
    outcomes = run_pairwise_judging(judge, pairs, swap=True)
    answerable_outcomes = [
        outcome for pair, outcome in zip(pairs, outcomes, strict=True) if not pair.expects_refusal
    ]
    tally = summarise_wins(answerable_outcomes)
    strata = summarise_judge_strata(pairs, outcomes)
    for outcome in outcomes:
        typer.echo(
            f"  {outcome.sample_id:<16} winner={outcome.winner:<5} agreement={outcome.agreement}"
        )
    _write_json(
        output,
        {
            "schema_version": "1.0",
            "judge_model": judge.model_version,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "pairs": len(pairs),
            "judged_pairs": sum(outcome.agreement is not None for outcome in outcomes),
            "agreed": sum(1 for outcome in outcomes if outcome.agreement),
            "ties": sum(1 for outcome in outcomes if outcome.winner == "tie"),
            "refusal_gate": refusal_gate,
            "refusal_gate_hits": getattr(judge, "gate_hits", 0),
            "refusal_gated_pairs": getattr(judge, "gate_hits", 0),
            "parse_failures": getattr(base_judge, "parse_failures", 0),
            "judge_input_tokens": getattr(base_judge, "input_tokens", 0),
            "judge_output_tokens": getattr(base_judge, "output_tokens", 0),
            "tally": tally,
            "tally_scope": "answerable_checkpoints_only",
            "strata": strata,
            "judge_calls": [c.model_dump(mode="json") for c in getattr(base_judge, "calls", [])],
            "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
        },
    )
    typer.echo(str(output))
    failures = getattr(base_judge, "parse_failures", 0)
    if failures:
        typer.echo(f"WARNING: {failures} unparseable judge replies")


@app.command("annotate-necessity")
def annotate_necessity_command(
    annotations: Annotated[Path, typer.Argument(help="Real-replay annotation file")],
    database: Annotated[Path, typer.Argument(help="Local event store")],
    output: Annotated[Path, typer.Argument()],
    model: Annotated[str | None, typer.Option(help="Pinned model id")] = None,
    base_url: Annotated[str | None, typer.Option()] = None,
    auth_style: Annotated[str, typer.Option(help="bearer or x-api-key")] = "bearer",
    max_tokens: Annotated[int, typer.Option(min=64)] = 512,
    limit: Annotated[int, typer.Option(min=1, help="Checkpoints to process")] = 26,
    per_context: Annotated[int, typer.Option(min=1, help="Events per context")] = 6,
) -> None:
    """Derive a required-context set by ablation, for the record.

    One call per (checkpoint, active context). This is the instrument that produced the
    negative result in `docs/v0.3-necessity-is-circular.md`, kept so the result can be re-run
    rather than taken on trust.

    It no longer compares against human labels, because there are none to compare with: the
    annotation has no `required_context_ids`, since that field cannot be labelled -- which is
    the finding. `derived_required` is therefore an output with no ground truth attached, and
    v0.3 is the thing to read before reading anything into it. The control failures are still
    worth watching: they are the ablation calling a context necessary when removing it cost
    nothing, which is the failure mode that was observed.
    """

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL", "")
    resolved_model = model or os.environ.get("ANTHROPIC_MODEL", "")
    if not (api_key and resolved_base and resolved_model):
        raise typer.BadParameter("need ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL and a model")
    judge = AnthropicCompatibleAnswerProvider(
        base_url=resolved_base,
        api_key=api_key,
        model=resolved_model,
        auth_style=auth_style,  # type: ignore[arg-type]
        max_tokens=max_tokens,
    )
    annotation = RealReplayAnnotation.model_validate_json(annotations.read_text(encoding="utf-8"))
    store = SQLiteEventStore(database)
    events = store.list_events(annotation.source_session_id)
    by_seq = {event.sequence: event for event in events}
    sequence_of = {event.event_id: event.sequence for event in events}
    member_of: dict[str, list[RawEvent]] = {c.context_id: [] for c in annotation.contexts}
    for context_id, event_ids in annotation.context_members.items():
        member_of[context_id] = sorted(
            (by_seq[sequence_of[e]] for e in event_ids if e in sequence_of),
            key=lambda event: event.sequence,
        )
    names = {c.context_id: c.name for c in annotation.contexts}
    analyzer = LexicalAnalyzer()

    rows: list[dict[str, Any]] = []
    totals = {"calls": 0, "unreadable": 0, "control_failures": 0}
    unreadable_replies: list[str] = []
    started = time.time()
    for index, checkpoint in enumerate(annotation.checkpoints[:limit], start=1):
        seq = checkpoint.as_of_sequence
        query = by_seq[seq].content
        recent = [
            event
            for event in events
            if event.sequence < seq and event.actor in ("user", "assistant")
        ][-6:]
        recent_window = "\n".join(f"{e.actor}: {e.content[:300]}" for e in recent)

        materials: dict[str, str] = {}
        controls: set[str] = set()
        plausibility: dict[str, float] = {}
        for context_id, context_events in member_of.items():
            if not any(event.sequence < seq for event in context_events):
                continue  # the context does not exist yet at this checkpoint
            usable = [event for event in context_events if event.sequence < seq]
            bm25 = BM25Index({e.event_id: e.content for e in usable}, analyzer=analyzer)
            if not bm25.rank(query, per_context):
                controls.add(context_id)
            plausibility[context_id] = max(bm25.scores(query).values(), default=0.0)
            materials[context_id] = context_material(
                usable, query, analyzer=analyzer, limit=per_context
            )
        if not materials:
            continue
        # The weakest candidate is the control. Almost every context overlaps the query
        # lexically, so "no overlap" almost never fires; the least plausible candidate is the
        # one an ablation should be able to remove without cost, and a "no" there is the alarm.
        if len(materials) > 1:
            weakest = min(plausibility, key=lambda key: plausibility[key])
            controls.add(weakest)
        run = run_ablations(
            sample_id=checkpoint.sample_id,
            query=query,
            recent_window=recent_window,
            materials=materials,
            context_names=names,
            control_context_ids=controls,
            judge=judge,
        )
        derived = required_from_ablations(run.verdicts)
        failed_controls = control_failures(run.verdicts)
        totals["calls"] += len(materials)
        totals["unreadable"] += run.unreadable
        unreadable_replies.extend(run.unreadable_replies)
        totals["control_failures"] += len(failed_controls)
        rows.append(
            {
                "sample_id": checkpoint.sample_id,
                "query_type": checkpoint.query_type,
                "derived_required": derived,
                "control_failures": failed_controls,
                "candidate_contexts": sorted(materials),
                "unreadable": run.unreadable,
            }
        )
        typer.echo(
            f"  [{index}/{min(limit, len(annotation.checkpoints))}] {checkpoint.sample_id} "
            f"cand={len(materials)} derived={len(derived)}"
            + (f" control-failures={failed_controls}" if failed_controls else "")
        )

    _write_json(
        output,
        {
            "schema_version": "1.0",
            "judge_model": resolved_model,
            "checkpoints": len(rows),
            "calls": totals["calls"],
            "unreadable": totals["unreadable"],
            "unreadable_replies": unreadable_replies[:20],
            "control_failures": totals["control_failures"],
            "seconds": round(time.time() - started, 1),
            "rows": rows,
        },
    )
    typer.echo(
        f"\n{resolved_model}: {totals['calls']} calls, unreadable {totals['unreadable']}, "
        f"control failures {totals['control_failures']} -- no agreement rate, see v0.3"
    )
    typer.echo(str(output))


@app.command("benchmark")
def benchmark_command(
    database: Annotated[Path, typer.Argument()],
    benchmark: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
    ranker_file: Annotated[Path | None, typer.Option()] = None,
    policy_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    store = SQLiteEventStore(database)
    ranker = PlattContextRanker.load(ranker_file) if ranker_file else None
    policy = RoutingPolicy(**json.loads(policy_file.read_text())) if policy_file else None
    router = ContextRouter(ranker=ranker, policy=policy)
    benchmark_queries = _read_benchmark(benchmark)
    cases: list[RouteCaseResult] = []
    traces: list[dict[str, object]] = []
    for sample in benchmark_queries:
        events = store.list_events(sample.session_id, as_of_sequence=sample.as_of_sequence)
        recent = [event for event in events if event.sequence < sample.as_of_sequence][-6:]
        query_event = store.get_event(sample.query_event_id)
        if query_event is None:
            raise typer.BadParameter(f"missing query event: {sample.query_event_id}")
        decision = router.route(
            RouteRequest(
                query_event_id=sample.query_event_id,
                query=query_event.content,
                recent_events=recent,
                primary_context_id=sample.primary_context_id,
                recent_context_ids=sample.recent_context_ids,
                context_catalog=_session_contexts(store, sample.session_id, sample.as_of_sequence),
                as_of_sequence=sample.as_of_sequence,
            )
        )
        cases.append(
            RouteCaseResult(
                sample_id=sample.sample_id,
                session_id=sample.session_id,
                required_context_ids=sample.required_context_ids,
                selected_context_ids=decision.selected_context_ids,
                candidate_order=[candidate.context_id for candidate in decision.candidates],
                decision=decision.decision,
                confidence=decision.confidence,
                relation_expected=sample.relation_label,
                relation_predicted=decision.relation,
                query_type=sample.query_type,
            )
        )
        traces.append(
            {
                "sample_id": sample.sample_id,
                "session_id": sample.session_id,
                "query_type": sample.query_type,
                "relation_expected": sample.relation_label,
                "relation_predicted": decision.relation,
                "required_context_ids": sample.required_context_ids,
                "decision": decision.model_dump(mode="json"),
            }
        )
    payload = {
        "schema_version": "1.1",
        "metrics": evaluate_routes(cases),
        "traces": traces,
    }
    _write_json(output, payload)
    typer.echo(str(output))


@app.command("compare-baselines")
def compare_baselines_command(
    database: Annotated[Path, typer.Argument()],
    benchmark: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
    token_budget: Annotated[int, typer.Option(min=32)] = 2048,
    ranker_file: Annotated[Path | None, typer.Option()] = None,
    policy_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compare every memory strategy on tokens and evidence recall, entirely offline.

    Pass a ranker and/or policy trained on a disjoint session range to report the
    router's configured performance rather than only its defaults.
    """

    store = SQLiteEventStore(database)
    ranker = PlattContextRanker.load(ranker_file) if ranker_file else None
    policy = RoutingPolicy(**json.loads(policy_file.read_text())) if policy_file else None
    router = ContextRouter(ranker=ranker, policy=policy)
    cases = build_arm_cases(store, _read_benchmark(benchmark), token_budget=token_budget)
    results = [run_arm(name, case, router=router) for case in cases for name in ARM_NAMES]
    summary = evaluate_arms(results)
    payload = {
        "schema_version": "1.0",
        "token_budget": token_budget,
        "router_profile": describe_router(router),
        "summary": summary,
        "gate": first_gate(summary),
        "results": [result.model_dump(mode="json") for result in results],
    }
    _write_json(output, payload)
    typer.echo(str(output))


@app.command("train-ranker")
def train_ranker(
    benchmark_results: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
) -> None:
    """Train by conversation group and calibrate on held-out conversations."""

    payload = json.loads(benchmark_results.read_text(encoding="utf-8"))
    train_rows: list[dict[str, float]] = []
    train_labels: list[int] = []
    calibration_rows: list[dict[str, float]] = []
    calibration_labels: list[int] = []
    for trace in payload["traces"]:
        required = set(trace["required_context_ids"])
        bucket = int(hashlib.sha256(trace["session_id"].encode()).hexdigest(), 16) % 5
        for candidate in trace["decision"]["candidates"]:
            row = candidate["feature_values"]
            label = int(candidate["context_id"] in required)
            if bucket == 0:
                calibration_rows.append(row)
                calibration_labels.append(label)
            else:
                train_rows.append(row)
                train_labels.append(label)
    ranker = PlattContextRanker.fit(
        train_rows=train_rows,
        train_labels=train_labels,
        calibration_rows=calibration_rows,
        calibration_labels=calibration_labels,
    )
    ranker.save(output)
    typer.echo(ranker.model_version)


@app.command("tune-policy")
def tune_policy(
    benchmark_results: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
) -> None:
    payload = json.loads(benchmark_results.read_text(encoding="utf-8"))
    observations = [
        PolicyObservation(
            probabilities={
                candidate["context_id"]: candidate["calibrated_probability"]
                for candidate in trace["decision"]["candidates"]
            },
            required_context_ids=frozenset(trace["required_context_ids"]),
            relation=trace["decision"]["relation"],
        )
        for trace in payload["traces"]
    ]
    result = SelectionPolicyTuner().tune(observations)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(result.policy), indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(asdict(result), ensure_ascii=False, default=str))


def _arms_report(payload: dict[str, Any]) -> list[str]:
    summary: dict[str, dict[str, float]] = payload["summary"]
    gate: dict[str, Any] = payload["gate"]
    lines = [
        "# Context Router - Baseline Arm Comparison",
        "",
        "Memory tokens and evidence recall are offline measurements. Answer quality is not",
        "measured here and requires a run against a pinned main model.",
        "",
        f"- Token budget per arm: {payload['token_budget']}",
        f"- Router profile: {payload.get('router_profile', 'default')}",
        f"- Checkpoints: {int(summary['oracle_router']['count'])}",
        "",
        "| Arm | Mean memory tokens | Median | vs Full history | Evidence recall | Leakage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ARM_NAMES:
        row = summary.get(name)
        if row is None:
            continue
        reduction = row.get("token_reduction_vs_full_history", 0.0)
        lines.append(
            f"| `{name}` | {row['mean_memory_tokens']:.1f} | {row['median_memory_tokens']:.1f} "
            f"| {reduction:.1%} | {row['evidence_set_recall']:.3f} "
            f"| {int(row['future_leakage_total'])} |"
        )
    lines += [
        "",
        "## Gate one: is selective context worth a real router?",
        "",
        f"- Oracle memory-token reduction vs full history: {gate['oracle_token_reduction']:.1%} "
        f"(target >= {gate['token_reduction_target']:.0%})",
        f"- Token gate passed: {gate['token_gate_passed']}",
        "- Oracle evidence recall not worse than full history: "
        f"{gate['oracle_recall_not_worse_than_full_history']}",
        f"- Answer quality verified: {gate['answer_quality_verified']}",
        f"- Verdict: **{gate['verdict']}**",
    ]
    return lines


@app.command("report")
def report_command(
    results: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Argument()],
) -> None:
    payload = json.loads(results.read_text(encoding="utf-8"))
    if "summary" in payload:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(_arms_report(payload)) + "\n", encoding="utf-8")
        typer.echo(str(output))
        return
    metrics = payload["metrics"]
    lines = [
        "# Context Router Benchmark",
        "",
        f"- Samples: {metrics['count']}",
        f"- Exact context set accuracy: {metrics['exact_context_set_accuracy']:.3f}",
        f"- Micro precision: {metrics['micro_precision']:.3f}",
        f"- Micro recall: {metrics['micro_recall']:.3f}",
        f"- Cross-context recall: {metrics['cross_context_recall']:.3f}",
        f"- Relation accuracy: {metrics['relation_accuracy']:.3f}",
        f"- Brier score: {metrics['brier_score']:.3f}",
        f"- ECE: {metrics['ece']:.3f}",
        "",
        "The report contains routing metrics only; "
        "answer-quality claims require a pinned model run.",
    ]
    stages = metrics.get("stage_diagnostics")
    if isinstance(stages, dict):
        candidate_recall = stages.get("candidate_micro_recall")
        selection_recall = stages.get("selection_micro_recall")

        def display(value: object) -> str:
            return f"{value:.3f}" if isinstance(value, float) else "n/a"

        def count(value: object) -> str:
            return str(value) if isinstance(value, int) else "n/a"

        def append_stage_table(title: str, label: str, groups: object) -> None:
            if not isinstance(groups, dict):
                return
            lines.extend(
                [
                    "",
                    f"### {title}",
                    "",
                    f"| {label} | Cases | Candidate recall | Selection recall | "
                    "Candidate misses | Selection losses | Excess contexts |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for name, raw_row in sorted(groups.items()):
                if not isinstance(raw_row, dict):
                    continue
                lines.append(
                    f"| `{name}` | {count(raw_row.get('cases'))} "
                    f"| {display(raw_row.get('candidate_micro_recall'))} "
                    f"| {display(raw_row.get('selection_micro_recall'))} "
                    f"| {count(raw_row.get('candidate_miss_count'))} "
                    f"| {count(raw_row.get('selection_loss_count'))} "
                    f"| {count(raw_row.get('selection_extra_count'))} |"
                )

        lines += [
            "",
            "## Routing stage diagnosis",
            "",
            f"- Candidate required-context recall: {display(candidate_recall)}",
            f"- Selected required-context recall: {display(selection_recall)}",
            f"- Candidate misses: {int(stages.get('candidate_miss_count', 0))}",
            f"- Selection losses: {int(stages.get('selection_loss_count', 0))}",
            f"- Excess selected contexts: {int(stages.get('selection_extra_count', 0))}",
            f"- Mean selected contexts: {display(stages.get('mean_selected_contexts'))}",
            "",
            "A candidate miss points to retrieval. A selection loss means the required context",
            "was available and was later dropped by ranking, calibration or policy. Excess",
            "contexts point to over-selection or under-calibration.",
        ]
        append_stage_table("By query type", "Query type", stages.get("by_query_type"))
        append_stage_table("By expected relation", "Expected relation", stages.get("by_relation"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(str(output))


if __name__ == "__main__":
    app()
