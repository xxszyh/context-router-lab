"""Commands for an offline LongMemEval answer-quality exchange."""

from pathlib import Path
from typing import Annotated

import typer

from context_router.external.longmemeval import write_report
from context_router.external.longmemeval_answers import (
    SUPPORTED_ARMS,
    MemoryBudget,
    prepare_answer_plan,
    prepare_judge_requests,
    score_answer_plan,
)


def register_answer_commands(app: typer.Typer) -> None:
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
        typer.echo(
            "Scope: exploratory, answerable questions only; no held-out or abstention claim."
        )
