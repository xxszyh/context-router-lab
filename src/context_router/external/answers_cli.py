"""Commands for an offline LongMemEval answer-quality exchange."""

import json
from pathlib import Path
from typing import Annotated

import typer

from context_router.external.answer_runner import Stage, TokenLimitField, run_requests
from context_router.external.longmemeval import write_report
from context_router.external.longmemeval_answers import (
    SUPPORTED_ARMS,
    MemoryBudget,
    prepare_answer_plan,
    prepare_judge_requests,
    score_answer_plan,
)
from context_router.external.longmemeval_rendering import LEGACY_RENDERING
from context_router.external.quality_comparison import (
    QualityCriteria,
    prepare_quality_comparison,
    prepare_quality_judgments,
    score_quality_comparison,
)
from context_router.external.rendering_audit import audit_answer_plans


def register_answer_commands(app: typer.Typer) -> None:
    @app.command("longmemeval-quality-plan")
    def quality_plan_command(
        baseline: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        candidate: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
        arm: str = "joint",
        seed: int = 20261007,
        minimum_accuracy_gain: Annotated[float, typer.Option(min=0, max=1)] = 0.0,
        maximum_failed_completion_increase: Annotated[float, typer.Option(min=0, max=1)] = 0.0,
        maximum_mean_memory_ratio: Annotated[float, typer.Option(min=0.01)] = 1.0,
        maximum_exact_mcnemar_p: Annotated[float, typer.Option(min=0, max=1)] = 0.05,
    ) -> None:
        """Freeze one cross-rendering comparison and mix its public generation requests."""
        try:
            result = prepare_quality_comparison(
                baseline,
                candidate,
                output,
                arm=arm,
                seed=seed,
                criteria=QualityCriteria(
                    minimum_accuracy_gain,
                    maximum_failed_completion_increase,
                    maximum_mean_memory_ratio,
                    maximum_exact_mcnemar_p,
                ),
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        typer.echo("Answer quality: not measured. The inspected sample remains exploratory.")

    @app.command("longmemeval-quality-judge-plan")
    def quality_judge_command(
        comparison: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        answers: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
    ) -> None:
        """Mix both renderings into one blinded, reference-based judging batch."""
        try:
            result = prepare_quality_judgments(comparison, answers, output)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))

    @app.command("longmemeval-quality-score")
    def quality_score_command(
        comparison: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        answers: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        judgments: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
    ) -> None:
        """Compare complete matched answers using the frozen primary arm and criteria."""
        protected = [
            comparison,
            answers,
            judgments,
            *comparison.parent.glob("*.private.json"),
            comparison.parent / "generation.requests.jsonl",
        ]
        if output.exists() or {
            output.resolve(),
            output.with_name(output.name + ".tmp").resolve(),
        } & {path.resolve() for path in protected}:
            raise typer.BadParameter("choose a new output file; inputs must remain intact")
        try:
            result = score_quality_comparison(comparison, answers, judgments)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        write_report(output, result)
        typer.echo(json.dumps(result["sample_gate"], ensure_ascii=False, indent=2))
        typer.echo(
            "Scope: exploratory. Independent validation is required before default promotion."
        )

    @app.command("longmemeval-render-audit")
    def render_audit_command(
        dataset: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
        compare_plan: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        tokenizer_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        seed: int = 20261007,
    ) -> None:
        """Audit rendered source spans; optionally compare two matched rendering plans."""
        inputs = [dataset, plan, *([compare_plan] if compare_plan is not None else [])]
        if output.exists() or {
            output.resolve(),
            output.with_name(output.name + ".tmp").resolve(),
        } & {path.resolve() for path in inputs}:
            raise typer.BadParameter("choose a new output file; inputs must remain intact")
        try:
            result = audit_answer_plans(
                dataset,
                [plan, *([compare_plan] if compare_plan is not None else [])],
                tokenizer_file=tokenizer_file,
                seed=seed,
                progress=lambda count: (
                    typer.echo(f"Audited {count} questions") if count % 25 == 0 else None
                ),
            )
        except (ValueError, ImportError) as error:
            raise typer.BadParameter(str(error)) from error
        write_report(output, result)
        typer.echo(
            json.dumps(
                [
                    {"rendering": item["rendering"], "summary": item["summary"]}
                    for item in result["plans"]
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        typer.echo(
            "Answer quality: not measured. This audits source-span retention, not correctness."
        )

    @app.command("longmemeval-run")
    def run_command(
        requests: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
        model: Annotated[
            str, typer.Option(help="Explicit model/version served by the local endpoint")
        ],
        stage: str = "generation",
        base_url: str = "http://127.0.0.1:11434/v1",
        execute: bool = False,
        resume: bool = False,
        retry_failures: bool = False,
        max_requests: Annotated[int, typer.Option(min=1)] = 20,
        max_output_tokens: Annotated[int, typer.Option(min=1)] = 256,
        temperature: Annotated[float, typer.Option(min=0, max=2)] = 0.0,
        token_limit_field: str = "max_completion_tokens",
        timeout: Annotated[float, typer.Option(min=0.01, max=60)] = 60.0,
    ) -> None:
        """Preview or explicitly execute a bounded local batch; journal every attempt."""
        if stage not in ("generation", "judge"):
            raise typer.BadParameter("stage must be generation or judge")
        if token_limit_field not in ("max_tokens", "max_completion_tokens"):
            raise typer.BadParameter("unsupported token-limit-field")
        selected_stage: Stage = "generation" if stage == "generation" else "judge"
        selected_limit: TokenLimitField = (
            "max_tokens" if token_limit_field == "max_tokens" else "max_completion_tokens"
        )
        try:
            result = run_requests(
                requests,
                output,
                stage=selected_stage,
                model=model,
                base_url=base_url,
                execute=execute,
                resume=resume,
                retry_failures=retry_failures,
                max_requests=max_requests,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                token_limit_field=selected_limit,
                timeout=timeout,
            )
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
        if result.get("failed", 0):
            raise typer.Exit(code=1)

    @app.command("longmemeval-answer-plan")
    def plan_command(
        dataset: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        retrieval: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
        memory_budget: Annotated[int, typer.Option(min=128)] = 12000,
        tokenizer_file: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
        tokenizer_model: str | None = None,
        arm: Annotated[list[str] | None, typer.Option()] = None,
        limit: Annotated[int | None, typer.Option(min=1)] = None,
        seed: int = 20261007,
        include_abstention: bool = False,
        rendering: str = LEGACY_RENDERING,
    ) -> None:
        """Export dated, budgeted prompts. Makes no model calls."""
        try:
            result = prepare_answer_plan(
                dataset,
                retrieval,
                output,
                budget=MemoryBudget(
                    memory_budget, tokenizer_file=tokenizer_file, model=tokenizer_model
                ),
                arms=tuple(arm or SUPPORTED_ARMS),
                limit=limit,
                seed=seed,
                include_abstention=include_abstention,
                rendering=rendering,
                progress=lambda count: (
                    typer.echo(f"Prepared {count} questions") if count % 25 == 0 else None
                ),
            )
        except (ValueError, ImportError) as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(f"Prepared {result['requests']} requests for {result['questions']} questions.")
        typer.echo(
            "Answer quality: not measured. Generation and independent judgments are pending."
        )

    @app.command("longmemeval-judge-plan")
    def judge_command(
        plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        answers: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
    ) -> None:
        """Export blinded reference-based judging prompts for completed answers."""
        try:
            result = prepare_judge_requests(plan, answers, output)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        typer.echo(f"Prepared {result['requests']} blinded judging requests.")
        typer.echo("Answer quality: not measured. Independent judgments are pending.")

    @app.command("longmemeval-answer-score")
    def score_command(
        plan: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        answers: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        judgments: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
        output: Path,
        seed: int = 20261007,
    ) -> None:
        """Score a complete paired set; incomplete or mismatched imports fail."""
        if output.exists() or output.resolve() in {
            path.resolve() for path in (plan, answers, judgments)
        }:
            raise typer.BadParameter("choose a new output file; inputs must remain intact")
        try:
            result = score_answer_plan(plan, answers, judgments, seed=seed)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        write_report(output, result)
        typer.echo(
            f"Scored {len(result['records'])} independently judged answers. Report: {output}"
        )
        typer.echo("Scope: exploratory; answerable and abstention results are reported separately.")
