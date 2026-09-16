from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from context_router.assembly import assemble_context
from context_router.datasets import generate_synthetic_dataset, validate_dataset
from context_router.domain import (
    AssemblyRequest,
    BenchmarkQuery,
    FlatContext,
    RouteRequest,
)
from context_router.evaluation import (
    ARM_NAMES,
    RouteCaseResult,
    build_arm_cases,
    describe_router,
    evaluate_arms,
    evaluate_routes,
    first_gate,
    run_arm,
)
from context_router.providers import HashEmbeddingProvider
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
            )
        )
        traces.append(
            {
                "sample_id": sample.sample_id,
                "session_id": sample.session_id,
                "required_context_ids": sample.required_context_ids,
                "decision": decision.model_dump(mode="json"),
            }
        )
    payload = {
        "schema_version": "1.0",
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(str(output))


if __name__ == "__main__":
    app()
