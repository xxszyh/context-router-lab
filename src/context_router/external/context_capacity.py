"""Offline chat-template capacity checks, bound to the complete request file.

No weights or service are loaded. A passing check concerns the declared local
template and context limit; the runner separately checks observed service usage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from context_router.external.batch_requests import (
    Stage,
    policy_zero,
    read_requests,
    request_messages,
)
from context_router.external.longmemeval import write_report
from context_router.external.longmemeval_answers import file_hash, object_hash, text_hash
from context_router.providers.pinned import require_pinned_model

ASSET_SUFFIXES = {".json", ".txt", ".model", ".jinja", ".tiktoken", ".vocab", ".merges"}


def _assets(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise ValueError("tokenizer directory must exist locally")
    files = {
        path.relative_to(directory).as_posix(): file_hash(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in ASSET_SUFFIXES
    }
    if not files or "tokenizer_config.json" not in files:
        raise ValueError("a local tokenizer directory with tokenizer_config.json is required")
    return files


def _limits(context_window: int, max_output_tokens: int) -> None:
    if (
        type(context_window) is not int
        or type(max_output_tokens) is not int
        or context_window < 1
        or not 0 < max_output_tokens < context_window
    ):
        raise ValueError("context window must exceed the positive output-token reserve")


def prepare_capacity_report(
    source: Path,
    output: Path,
    *,
    model: str,
    tokenizer_directory: Path,
    context_window: int,
    max_output_tokens: int = 256,
    stage: Stage = "generation",
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    _limits(context_window, max_output_tokens)
    if stage not in ("generation", "judge"):
        raise ValueError("stage must be generation or judge")
    model = require_pinned_model(model)
    tokenizer_directory = tokenizer_directory.resolve()
    if output.exists() or any(
        path == source.resolve() or path.is_relative_to(tokenizer_directory)
        for path in (output.resolve(), output.with_name(output.name + ".tmp").resolve())
    ):
        raise ValueError("choose a new output outside the request source and tokenizer directory")
    requests, digest = read_requests(source, stage)
    assets = _assets(tokenizer_directory)
    from transformers import AutoTokenizer, __version__

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_directory), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if not tokenizer.is_fast:
        raise ValueError("capacity counting requires a local fast tokenizer")
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template.strip():
        raise ValueError("the tokenizer must declare a non-empty chat template")
    rows = []
    for index, request in enumerate(requests, 1):
        skip = policy_zero(request, stage)
        ids = (
            []
            if skip
            else tokenizer.apply_chat_template(
                request_messages(request),
                tokenize=True,
                add_generation_prompt=True,
                truncation=False,
                padding=False,
                return_dict=False,
            )
        )
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("chat template must yield one flat token sequence")
        tokens = len(ids)
        required = 0 if skip else tokens + max_output_tokens
        rows.append(
            {
                "request_id": request["request_id"],
                "request_sha256": object_hash(request),
                "input_tokens": tokens,
                "required_tokens": required,
                "requires_call": not skip,
                "fits": required <= context_window,
            }
        )
        if progress is not None:
            progress(index)
    if _assets(tokenizer_directory) != assets:
        raise ValueError("tokenizer assets changed during counting")
    overflows = sum(not row["fits"] for row in rows)
    report = {
        "schema_version": "1.0",
        "kind": "local-chat-capacity-report",
        "status": "blocked_context_overflow" if overflows else "fits_declared_context",
        "requests_sha256": digest,
        "stage": stage,
        "model": model,
        "context_window": context_window,
        "max_output_tokens": max_output_tokens,
        "requests": len(rows),
        "overflow_requests": overflows,
        "all_fit": overflows == 0,
        "counter": {
            "tokenizer_directory": str(tokenizer_directory),
            "assets_sha256": assets,
            "transformers_version": __version__,
            "chat_template_sha256": text_hash(template),
            "add_generation_prompt": True,
            "truncation": False,
            "padding": False,
            "template_options": "tokenizer_defaults",
        },
        "service_tokenization_verified": False,
        "generator_token_budget_verified": False,
        "model_weights_loaded": False,
        "network_calls": 0,
        "rows": rows,
    }
    report["capacity_sha256"] = object_hash(report)
    write_report(output, report)
    return {key: value for key, value in report.items() if key not in ("rows", "counter")}


def validate_capacity_report(
    path: Path,
    requests: list[dict[str, Any]],
    *,
    requests_sha256: str,
    stage: Stage,
    model: str,
    max_output_tokens: int,
) -> tuple[str, dict[str, int]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    digest = report.pop("capacity_sha256", None)
    if report.get("kind") != "local-chat-capacity-report" or object_hash(report) != digest:
        raise ValueError("capacity report failed its integrity check")
    _limits(report["context_window"], report["max_output_tokens"])
    if any(
        report[field] != value
        for field, value in {
            "requests_sha256": requests_sha256,
            "stage": stage,
            "model": model,
            "max_output_tokens": max_output_tokens,
        }.items()
    ):
        raise ValueError("capacity report does not match requests, stage, model or output reserve")
    counter = report["counter"]
    if (
        counter["add_generation_prompt"] is not True
        or counter["truncation"] is not False
        or counter["padding"] is not False
        or counter["template_options"] != "tokenizer_defaults"
        or _assets(Path(counter["tokenizer_directory"])) != counter["assets_sha256"]
    ):
        raise ValueError("capacity tokenizer assets or counting settings changed")
    indexed = {row["request_id"]: row for row in report["rows"]}
    if len(indexed) != len(report["rows"]) or set(indexed) != {r["request_id"] for r in requests}:
        raise ValueError("capacity report must cover the complete request set exactly once")
    overflows = 0
    for request in requests:
        row = indexed[request["request_id"]]
        skip = policy_zero(request, stage)
        tokens = row["input_tokens"]
        if type(tokens) is not int or tokens < 0:
            raise ValueError("capacity token counts must be nonnegative integers")
        required = 0 if skip else tokens + max_output_tokens
        if (
            type(tokens) is not int
            or tokens < 0
            or (skip and tokens != 0)
            or row["request_sha256"] != object_hash(request)
            or type(row["requires_call"]) is not bool
            or row["requires_call"] != (not skip)
            or type(row["required_tokens"]) is not int
            or row["required_tokens"] != required
            or type(row["fits"]) is not bool
            or row["fits"] != (required <= report["context_window"])
        ):
            raise ValueError("invalid capacity count or request binding")
        overflows += not row["fits"]
    if (
        report["requests"] != len(requests)
        or report["overflow_requests"] != overflows
        or type(report["all_fit"]) is not bool
        or report["all_fit"] != (overflows == 0)
    ):
        raise ValueError("capacity summary does not match its complete rows")
    if overflows:
        raise ValueError(f"{overflows} requests exceed the declared context window; no calls made")
    return digest, {key: row["input_tokens"] for key, row in indexed.items()}
