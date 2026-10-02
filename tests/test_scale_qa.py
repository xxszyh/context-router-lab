from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from context_router.assembly.builder import TokenCounter
from context_router.external.scale_qa import (
    ORDERING_FIELDS,
    ScaleQaPackage,
    ScaleQaQuestion,
    ScaleQaWindow,
    WindowRetriever,
    arrange,
    arrangement_reachability,
    baseline_solved_mask,
    candidate_order_attribution,
    difficulty_strata,
    evidence_recall,
    load_package,
    ordering_hits,
    ordering_report,
    paired_ordering_comparison,
    retrieve,
    routing_miss_decomposition,
    window_blocks,
)
from context_router.retrieval import LexicalAnalyzer

#: A miniature runtime package in the shape SCALE-QA's builder emits. Built here rather than
#: vendored, so the tests exercise the adapter without depending on the external repository.
_TRUTH = [
    [
        ("The Aegis-7 protocol forbids acidified silver stains.", "user"),
        ("Confirmed: GMS reacts with X-122 to give false positives.", "model"),
    ],
    [
        ("Ledger Q3 shows a 12% margin on the Helios line.", "user"),
        ("The margin was restated after the Nova-Life write-off.", "model"),
    ],
]
_NOISE = [
    [
        ("Edinburgh Castle was rebuilt across several centuries.", "user"),
        (
            "The 12th-century fortification was wooden and the later curtain wall was stone.",
            "model",
        ),
    ],
    [
        ("Sourdough needs a stable starter temperature.", "user"),
        ("Keep the levain between 24 and 26 degrees and feed it twice daily.", "model"),
    ],
    [
        ("Which tram lines serve the old town?", "user"),
        ("Lines 3, 5 and 7 stop beside the castle esplanade every ten minutes.", "model"),
    ],
    [
        ("How do I season a cast iron pan?", "user"),
        (
            "Warm it, wipe a thin film of flaxseed oil over the whole surface, then bake it.",
            "model",
        ),
    ],
    [
        ("What is the boiling point of water at altitude?", "user"),
        (
            "It falls roughly one degree Celsius for every three hundred metres of elevation.",
            "model",
        ),
    ],
]


def _write_package(root: Path) -> Path:
    runtime = root / "runtime" / "full_corpus"
    runtime.mkdir(parents=True)
    (runtime / "test_cases.py").write_text(
        "GROUND_TRUTH_HISTORY = " + repr(_TRUTH) + "\n"
        "EVALUATION_QUERIES = [\n"
        "    {'name': 'q1', 'query': 'Which stain does the Aegis-7 protocol forbid?',\n"
        "     'expected_doc': ['The Aegis-7 protocol forbids acidified silver stains.'],\n"
        "     'expected_answer_keywords': 'A', 'topic': 'Biomed', 'creator': 'codex',\n"
        "     'global_id': 'codex:Biomed-001'},\n"
        "    {'name': 'q2', 'query': 'What happened to the Helios line margin?',\n"
        "     'expected_doc': ['The margin was restated after the Nova-Life write-off.'],\n"
        "     'expected_answer_keywords': 'C', 'topic': 'Finance', 'creator': 'claude-code',\n"
        "     'global_id': 'claude-code:Finance-001'},\n"
        "]\n",
        encoding="utf-8",
    )
    (runtime / "processed_noise.py").write_text("NOISE = " + repr(_NOISE) + "\n", encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> ScaleQaPackage:
    root = _write_package(tmp_path_factory.mktemp("scaleqa"))
    return load_package(root)


def test_load_reads_truth_noise_and_questions(package: ScaleQaPackage) -> None:
    assert len(package.truth) == 2
    assert len(package.noise) == 5
    assert [q.global_id for q in package.questions] == [
        "codex:Biomed-001",
        "claude-code:Finance-001",
    ]
    assert package.questions[0].answer_key == "A"
    assert package.questions[0].expected_doc == (
        "The Aegis-7 protocol forbids acidified silver stains.",
    )


def test_load_accepts_the_runtime_directory_itself(package: ScaleQaPackage) -> None:
    """A reader of SCALE-QA's README may hold either the package root or runtime/<mode>."""

    runtime = Path(package.source)
    assert load_package(runtime).questions == package.questions


def test_arrangement_is_seeded_and_reproducible(package: ScaleQaPackage) -> None:
    for arrangement in ("blocks", "interleaved", "shuffled"):
        first = [block.text for block in arrange(package, arrangement=arrangement, seed=7)]
        second = [block.text for block in arrange(package, arrangement=arrangement, seed=7)]
        assert first == second, arrangement


def test_blocks_arrangement_is_the_naive_concatenation(package: ScaleQaPackage) -> None:
    ordered = arrange(package, arrangement="blocks", seed=0)
    assert [block.kind for block in ordered] == ["truth", "truth"] + ["noise"] * 5


def test_the_naive_arrangement_is_reported_as_degenerate(package: ScaleQaPackage) -> None:
    """The measurement that motivates the whole module.

    Truth-then-noise puts every question's evidence at the far end of the stream, so a trailing
    window of any size reaches none of it. That is the mirror of this project's own generator,
    where some query types are reachable 100% of the time. Both decide the recency question in
    advance, and the point of `arrangement_reachability` is to say so out loud.
    """

    counter = TokenCounter()
    ordered = arrange(package, arrangement="blocks", seed=0)
    report = arrangement_reachability(
        ordered,
        package.questions,
        counter=counter,
        arrangement="blocks",
        seed=0,
        # Window sizes small against this fixture's stream, so the assertion is about the
        # arrangement and not about the fixture happening to be shorter than the window.
        window_sizes=(8, 16, 32),
    )
    assert report.located == len(package.questions)
    # Truth first, noise after: every question's evidence sits at the far end of the stream, so a
    # trailing window of any size below reaches none of it.
    assert report.median_distance_from_end > 32
    for size, share in report.reachable_within_window.items():
        assert share == 0.0, size


def test_the_reachability_report_names_the_size_of_the_evidence_corpus(
    package: ScaleQaPackage,
) -> None:
    """The number a reader needs to interpret a miss decomposition, and the one nobody reports.

    Swept from 1,000 to 3,000 evidence dialogues at a fixed truth share, the router's
    candidate-miss rate goes 0.131 -> 0.246. Swept from 27% to 98% truth at a fixed corpus it does
    not move. "Candidate generation is not the bottleneck" is a statement about this number.
    """

    counter = TokenCounter()
    ordered = arrange(package, arrangement="blocks", seed=0)
    report = arrangement_reachability(
        ordered, package.questions, counter=counter, arrangement="blocks", seed=0
    )
    assert report.truth_blocks == len(package.truth) == 2


def test_windows_never_span_two_dialogues(package: ScaleQaPackage) -> None:
    """A window is a retrieved unit inside one source dialogue, not a slice of the concatenation."""

    counter = TokenCounter()
    ordered = arrange(package, arrangement="blocks", seed=0)
    windows = window_blocks(ordered, size=8, counter=counter)
    assert windows
    # Windows are emitted in block order and never merge two blocks, so every window's lines
    # appear inside the single block it names. `start_block` is a unique block index -- noise
    # blocks continue the truth blocks' numbering rather than restarting it.
    assert [w.start_block for w in windows] == sorted(w.start_block for w in windows)
    by_index = {block.index: block.text for block in ordered}
    for window in windows:
        owner = by_index[window.start_block]
        for line in window.text.splitlines():
            assert line in owner
    # A window holds at least one sentence, so it may exceed the requested size when a single
    # sentence does; what it may never do is span a boundary.
    assert any(window.tokens > 8 for window in windows)


def test_evidence_recall_is_exact_containment(package: ScaleQaPackage) -> None:
    question = package.questions[0]
    assert (
        evidence_recall(
            question, "prefix The Aegis-7 protocol forbids acidified silver stains. suffix"
        )
        == 1.0
    )
    assert evidence_recall(question, "nothing relevant here") == 0.0


def test_evidence_recall_ignores_whitespace_and_case(package: ScaleQaPackage) -> None:
    """The runtime package renders turns on separate lines, so the evidence may be re-wrapped."""

    question = package.questions[1]
    assembled = "The   margin was restated\nafter the Nova-Life write-off."
    assert evidence_recall(question, assembled) == 1.0


def test_lexical_retrieval_finds_the_window_holding_the_evidence(package: ScaleQaPackage) -> None:
    counter = TokenCounter()
    analyzer = LexicalAnalyzer()
    ordered = arrange(package, arrangement="blocks", seed=0)
    windows = window_blocks(ordered, size=64, counter=counter)
    chosen = retrieve(
        windows,
        package.questions[0].query,
        mode="lexical",
        analyzer=analyzer,
        embedder=_embedder(analyzer),
        depth=1,
    )
    assert chosen
    assert "acidified silver" in chosen[0].text, chosen[0].text


def test_router_mode_returns_units_from_the_same_catalog(package: ScaleQaPackage) -> None:
    """The router arm must return windows, not invent any -- the comparison is over one unit set."""

    counter = TokenCounter()
    analyzer = LexicalAnalyzer()
    ordered = arrange(package, arrangement="interleaved", seed=0)
    windows = window_blocks(ordered, size=64, counter=counter)
    chosen = retrieve(
        windows,
        package.questions[0].query,
        mode="router",
        analyzer=analyzer,
        embedder=_embedder(analyzer),
        depth=2,
    )
    known = {window.window_id for window in windows}
    assert chosen
    assert all(window.window_id in known for window in chosen)
    assert len(chosen) <= 2


def _embedder(analyzer: LexicalAnalyzer):  # type: ignore[no-untyped-def]
    from context_router.external.scale_qa import default_embedder

    return default_embedder(analyzer)


def test_the_module_docstring_names_the_arrangement_problem() -> None:
    """A reader who only opens the file must learn that the arrangement is a choice, not a given."""

    from context_router.external import scale_qa

    doc = textwrap.dedent(scale_qa.__doc__ or "")
    assert "arrangement" in doc.lower()
    assert "degenerate" in doc.lower()


# --- the two additions of 2026-10-02: a real embedder, and a cached router index ---


def test_lsa_embedding_carries_semantics_the_hash_placeholder_does_not() -> None:
    """The whole reason the dense path was never measurable.

    A hashing embedder gives two texts nothing in common just because they share no tokens. A
    fitted distributional embedding places texts that share *context* near each other, which is
    what makes a dense arm a measurement rather than a formality.
    """

    from context_router.providers.embedding import LsaEmbeddingProvider, cosine

    analyzer = LexicalAnalyzer()
    corpus = [
        "The Aegis-7 protocol forbids acidified silver stains for biopsy samples.",
        "The Aegis-7 protocol prohibits silver-based staining of biopsy material.",
        "Edinburgh Castle was rebuilt across several centuries of Scottish history.",
        "Sourdough starter should be kept between 24 and 26 degrees Celsius.",
    ]
    provider = LsaEmbeddingProvider(corpus, dimensions=16, analyzer=analyzer)
    query, related, unrelated = provider.embed([corpus[0], corpus[1], corpus[2]])
    assert cosine(query, related) > cosine(query, unrelated)


def test_lsa_is_deterministic_for_a_fixed_corpus() -> None:
    from context_router.providers.embedding import LsaEmbeddingProvider

    analyzer = LexicalAnalyzer()
    corpus = ["alpha beta gamma", "beta gamma delta", "gamma delta epsilon"]
    first = LsaEmbeddingProvider(corpus, dimensions=4, analyzer=analyzer).embed(corpus)
    second = LsaEmbeddingProvider(corpus, dimensions=4, analyzer=analyzer).embed(corpus)
    assert first == second


def test_lsa_refuses_an_empty_corpus() -> None:
    from context_router.providers.embedding import LsaEmbeddingProvider

    with pytest.raises(ValueError, match="non-empty corpus"):
        LsaEmbeddingProvider(["", "   "], dimensions=4)


def test_build_embedder_names_itself_and_rejects_unknown_names() -> None:
    from context_router.external.scale_qa import build_embedder

    analyzer = LexicalAnalyzer()
    windows = [ScaleQaWindow("w0", "a window of text about routing", 0, 8)]
    assert build_embedder("hash", windows, analyzer=analyzer).model_version == "hash-embedding-v1"
    lsa = build_embedder("lsa", windows, analyzer=analyzer)
    assert lsa.model_version.startswith("lsa-")
    # A one-window fixture has fewer features than the default width, so the provider clamps and
    # says so rather than raising or claiming a dimension it did not use.
    assert "svd256-" not in lsa.model_version
    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("bge", windows, analyzer=analyzer)


def test_the_router_index_cache_does_not_change_a_decision() -> None:
    """A cache that changes an answer is a bug, not an optimisation."""

    from context_router.domain import FlatContext, RouteRequest
    from context_router.routing import ContextRouter

    contexts = [
        FlatContext(
            context_id=f"c{index}",
            name=f"context {index}",
            goal="",
            summary=text,
            entities=[],
            lexical_terms=[],
            status="active",
            created_at_event=f"c{index}",
            last_active_sequence=index,
            version=1,
        )
        for index, text in enumerate(
            [
                "routing the query to the right logical context in a long conversation",
                "sea ice thermodynamics and the mushy layer equations",
                "sqlite migration and write-ahead logging",
            ]
        )
    ]
    request = RouteRequest(
        query_event_id="q",
        query="how does write-ahead logging work in sqlite?",
        recent_events=[],
        context_catalog=contexts,
        as_of_sequence=3,
    )
    uncached = ContextRouter().route(request)
    cached = ContextRouter(index_cache_size=2)
    first = cached.route(request, cache_key="catalog")
    second = cached.route(request, cache_key="catalog")
    assert first.selected_context_ids == uncached.selected_context_ids
    assert second.selected_context_ids == uncached.selected_context_ids
    assert first.decision == second.decision


def test_the_router_index_cache_is_off_unless_asked_for() -> None:
    """On by default would be a correctness hazard for every existing caller."""

    from context_router.routing import ContextRouter

    assert ContextRouter().index_cache_size == 0
    assert ContextRouter(index_cache_size=4).index_cache_size == 4


def test_the_router_index_cache_evicts_beyond_its_size() -> None:
    from context_router.domain import FlatContext, RouteRequest
    from context_router.routing import ContextRouter

    def catalog(marker: str) -> list[FlatContext]:
        return [
            FlatContext(
                context_id="c0",
                name="c0",
                goal="",
                summary=f"{marker} about sqlite write-ahead logging",
                entities=[],
                lexical_terms=[],
                status="active",
                created_at_event="c0",
                last_active_sequence=0,
                version=1,
            )
        ]

    router = ContextRouter(index_cache_size=1)
    for marker in ("first", "second", "third"):
        router.route(
            RouteRequest(
                query_event_id="q",
                query="sqlite write-ahead logging",
                recent_events=[],
                context_catalog=catalog(marker),
                as_of_sequence=0,
            ),
            cache_key=marker,
        )
    assert list(router._prepared) == ["third"]


def test_static_neural_embedding_is_semantic_and_pinned() -> None:
    """The only provider here that is actually neural, so it is the one that can answer the
    dense-channel question. Skipped where the model cannot be fetched: it needs the `neural`
    extra and a reachable model hub (set HF_ENDPOINT to a mirror where the default times out).
    """

    pytest.importorskip("model2vec")
    from context_router.providers.embedding import StaticNeuralEmbeddingProvider, cosine

    try:
        provider = StaticNeuralEmbeddingProvider()
    except Exception as error:  # pragma: no cover - network dependent
        pytest.skip(f"model unavailable: {error}")

    assert provider.model_version.startswith("static-neural:")
    query, related, unrelated = provider.embed(
        [
            "Which stain does the protocol forbid for biopsy samples?",
            "The protocol forbids acidified silver stains on biopsy material.",
            "Sourdough starter should be kept between 24 and 26 degrees.",
        ]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_the_placeholder_embedder_is_measurably_not_a_dense_retriever() -> None:
    """The property behind README limitation 2, made executable.

    Hashing places a text by the identity of its tokens, so two texts that share no token are
    exactly orthogonal however related they are. A real embedding places them by meaning. Measured
    on the 50-question package the dense arm scores **0.300** with the placeholder against 0.900
    with a pinned static encoder and 0.960 with a corpus-fitted LSA -- which is why no dense or
    hybrid number from this project was quotable while the placeholder was the default.
    """

    from context_router.providers.embedding import HashEmbeddingProvider, cosine

    analyzer = LexicalAnalyzer()
    hashing = HashEmbeddingProvider(analyzer=analyzer)
    # Same topic, no shared token: a paraphrase.
    left, right = "silver staining prohibited", "acidified reagent forbidden"
    assert not (set(analyzer.tokens(left)) & set(analyzer.tokens(right)))
    vectors = hashing.embed([left, right])
    assert cosine(vectors[0], vectors[1]) == 0.0

    # It is not that hashing is silent -- it responds to shared tokens, which is the whole of it.
    same = hashing.embed(["silver staining prohibited", "silver staining prohibited"])
    assert cosine(same[0], same[1]) > 0.9


# --- the difficulty split of 2026-10-02: the instrument for what a pooled score hides ---

#: Six unrelated windows. `b`'s evidence shares no token with `b`'s query, so a lexical lookup
#: answers `b` with a decoy -- which is exactly the case a pooled score cannot see.
_STRATUM_WINDOWS = [
    "notes on lichen growth rates",
    "the falcon decoy appears in this unrelated window",
    "a discussion of tidal barrages",
    "a registry entry about the northern goshawk",
    "secondary material concerning otters",
    "an appendix about volcanic ash",
]


def _question(identifier: str, query: str, evidence: str) -> ScaleQaQuestion:
    return ScaleQaQuestion(
        global_id=identifier,
        query=query,
        expected_doc=(evidence,),
        answer_key="A",
        topic="",
        creator="",
    )


def _lexical_retriever(texts: list[str], analyzer: LexicalAnalyzer) -> WindowRetriever:
    windows = [
        ScaleQaWindow(f"w{index}", text, index, len(text.split()))
        for index, text in enumerate(texts)
    ]
    return WindowRetriever(windows, mode="lexical", analyzer=analyzer, embedder=_embedder(analyzer))


def test_difficulty_strata_split_the_pooled_score_into_what_a_lookup_already_answers() -> None:
    """The split is a decomposition of the pooled number, not a second opinion about it."""

    analyzer = LexicalAnalyzer()
    retriever = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    questions = (
        _question("a", "otters lichen", "secondary material concerning otters"),
        _question("b", "falcon decoy", "a registry entry about the northern goshawk"),
        _question("c", "tidal barrages", "a discussion of tidal barrages"),
    )
    report = difficulty_strata(questions, {"lexical": retriever}, depth=2)

    assert report.baseline == "lexical"
    assert report.baseline_depth == 1
    assert report.questions == 3
    # `b` is the one the lookup does not answer: its top hit is the decoy, not its evidence.
    assert report.solved_by_baseline == 2
    assert report.solved_share == pytest.approx(2 / 3)

    stratum = report.strata[0]
    assert stratum.mode == "lexical"
    assert stratum.recall_on_solved == 1.0
    assert stratum.recall_on_unsolved == 0.0
    assert stratum.recall_pooled == pytest.approx(2 / 3)
    # Weighted back together the strata have to reproduce the pooled mean exactly.
    recombined = (
        report.solved_share * stratum.recall_on_solved
        + (1 - report.solved_share) * stratum.recall_on_unsolved
    )
    assert recombined == pytest.approx(stratum.recall_pooled)


def test_an_empty_stratum_is_unmeasured_rather_than_zero() -> None:
    """Same rule as `requirement_reachability`: no questions in a subset means no recall."""

    analyzer = LexicalAnalyzer()
    retriever = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    questions = (
        _question("a", "otters lichen", "secondary material concerning otters"),
        _question("c", "tidal barrages", "a discussion of tidal barrages"),
    )
    report = difficulty_strata(questions, {"lexical": retriever}, depth=2)
    assert report.solved_by_baseline == 2
    assert report.strata[0].recall_on_unsolved is None


def test_difficulty_strata_rejects_a_baseline_it_was_not_given() -> None:
    """A split defined by an arm that was never run would be a number with no measurement."""

    analyzer = LexicalAnalyzer()
    retriever = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    with pytest.raises(ValueError, match="baseline"):
        difficulty_strata((), {"dense": retriever})


def test_the_cli_reports_the_difficulty_split_beside_the_pooled_score(
    package: ScaleQaPackage, tmp_path: Path
) -> None:
    """The pooled score is what the report used to be; the split exists to qualify it."""

    from typer.testing import CliRunner

    from context_router.cli import app

    output = tmp_path / "scale-qa.json"
    result = CliRunner().invoke(
        app,
        [
            "scale-qa",
            package.source,
            str(output),
            "--window-size",
            "64",
            "--mode",
            "lexical",
            "--mode",
            "router",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "difficulty strata" in result.output
    assert "already answers" in result.output
    assert "selection loss" in result.output
    assert "router ordering over the same candidates" in result.output

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1"
    assert payload["difficulty"][0]["questions"] == 2
    assert payload["difficulty"][0]["baseline"] == "lexical"
    # Both requested modes, plus `lexical` even when it is not requested, which it is here.
    assert sorted(s["mode"] for s in payload["difficulty"][0]["strata"]) == ["lexical", "router"]
    assert payload["results"][0]["mode"] == "lexical"

    detail = payload["router_decomposition"][0]
    assert (
        detail["recovered"] + detail["selection_loss"] + detail["candidate_miss"]
        == detail["questions"]
    )

    ordering = payload["router_ordering"][0]
    assert {row["field"] for row in ordering["orderings"]} == set(ORDERING_FIELDS)
    assert ordering["oracle_at_depth"] >= max(
        row["recall_at_depth"] for row in ordering["orderings"]
    )
    # A point estimate without its paired interval is what got a published sentence withdrawn.
    assert "paired " in result.output
    assert ordering["paired"], "the report must carry the comparisons, not only the rows"
    for comparison in ordering["paired"]:
        assert comparison["bootstrap_low"] <= comparison["bootstrap_high"]


# --- the decomposition of 2026-10-02: a retrieval miss and a ranking miss are different ---


def test_retrieve_with_decision_returns_the_units_retrieve_returns() -> None:
    """The decision is extra information, not a different answer."""

    analyzer = LexicalAnalyzer()
    windows = [
        ScaleQaWindow(f"w{index}", text, index, len(text.split()))
        for index, text in enumerate(_STRATUM_WINDOWS)
    ]
    question = _question("a", "otters lichen", "secondary material concerning otters")

    router = WindowRetriever(
        windows, mode="router", analyzer=analyzer, embedder=_embedder(analyzer)
    )
    units, decision = router.retrieve_with_decision(question.query, depth=2)
    assert units == router.retrieve(question.query, depth=2)
    assert decision is not None
    assert set(decision.selected_context_ids) <= {window.window_id for window in units}

    baseline = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    units, decision = baseline.retrieve_with_decision(question.query, depth=2)
    assert units == baseline.retrieve(question.query, depth=2)
    assert decision is None


def test_the_decomposition_accounts_for_every_question_it_was_given() -> None:
    analyzer = LexicalAnalyzer()
    windows = [
        ScaleQaWindow(f"w{index}", text, index, len(text.split()))
        for index, text in enumerate(_STRATUM_WINDOWS)
    ]
    baseline = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    router = WindowRetriever(
        windows, mode="router", analyzer=analyzer, embedder=_embedder(analyzer)
    )
    questions = (
        _question("a", "otters lichen", "secondary material concerning otters"),
        _question("b", "falcon decoy", "a registry entry about the northern goshawk"),
        _question("c", "tidal barrages", "a discussion of tidal barrages"),
    )
    solved = baseline_solved_mask(
        questions, {"lexical": baseline, "router": router}, baseline="lexical"
    )
    hard = tuple(
        question for question, is_solved in zip(questions, solved, strict=True) if not is_solved
    )
    assert len(hard) == 1, "only `b` is not already answered by one lookup"

    report = routing_miss_decomposition(hard, router, depth=2)
    assert report.questions == 1
    # Every question is in exactly one bucket, and the recovered bucket has to agree with the
    # arm's own recall -- otherwise the decomposition is a second, unrelated measurement.
    assert report.recovered + report.selection_loss + report.candidate_miss == 1
    assert report.no_evidence_window == 0
    assert sum(report.selected_counts.values()) == 1
    pooled = router.evaluate(hard, depth=2)
    assert report.recovered == round(pooled["all_evidence_recall"])


def test_the_decomposition_refuses_an_arm_that_has_no_decision() -> None:
    """A selection loss is only meaningful for an arm that selects rather than ranks."""

    analyzer = LexicalAnalyzer()
    retriever = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    with pytest.raises(ValueError, match="router"):
        routing_miss_decomposition((), retriever)


def test_the_ordering_attribution_holds_the_candidate_list_fixed() -> None:
    """Every row is the same candidates, so a difference between rows belongs to the ordering."""

    analyzer = LexicalAnalyzer()
    windows = [
        ScaleQaWindow(f"w{index}", text, index, len(text.split()))
        for index, text in enumerate(_STRATUM_WINDOWS)
    ]
    router = WindowRetriever(
        windows, mode="router", analyzer=analyzer, embedder=_embedder(analyzer)
    )
    questions = (
        _question("a", "otters lichen", "secondary material concerning otters"),
        _question("b", "falcon decoy", "a registry entry about the northern goshawk"),
        _question("c", "tidal barrages", "a discussion of tidal barrages"),
    )
    report = candidate_order_attribution(questions, router, depth=2)

    assert report.questions == 3
    assert report.depth == 2
    assert {ordering.field for ordering in report.orderings} == set(ORDERING_FIELDS)
    assert report.candidate_size_median is not None
    for ordering in report.orderings:
        assert ordering.questions == 3
        assert 0.0 <= ordering.recall_at_depth <= 1.0
        # No ordering can surface evidence the list does not contain, so the ceiling holds.
        assert ordering.recall_at_depth <= report.oracle_at_depth


def test_the_ordering_attribution_refuses_an_arm_that_ranks_nothing() -> None:
    analyzer = LexicalAnalyzer()
    retriever = _lexical_retriever(_STRATUM_WINDOWS, analyzer)
    with pytest.raises(ValueError, match="router"):
        candidate_order_attribution((), retriever)


def _router_over_stratum(
    analyzer: LexicalAnalyzer,
) -> tuple[WindowRetriever, tuple[ScaleQaQuestion, ...]]:
    windows = [
        ScaleQaWindow(f"w{index}", text, index, len(text.split()))
        for index, text in enumerate(_STRATUM_WINDOWS)
    ]
    router = WindowRetriever(
        windows, mode="router", analyzer=analyzer, embedder=_embedder(analyzer)
    )
    questions = (
        _question("a", "otters lichen", "secondary material concerning otters"),
        _question("b", "falcon decoy", "a registry entry about the northern goshawk"),
        _question("c", "tidal barrages", "a discussion of tidal barrages"),
    )
    return router, questions


def test_the_paired_comparison_agrees_with_the_rows_it_came_from() -> None:
    """The interval and the point estimate have to come out of the same routing pass."""

    analyzer = LexicalAnalyzer()
    router, questions = _router_over_stratum(analyzer)
    hits = ordering_hits(questions, router, depth=2)
    report = ordering_report(hits, depth=2)
    by_field = {o.field: o.recall_at_depth for o in report.orderings}
    comparisons = paired_ordering_comparison(hits, resamples=200)

    assert comparisons, "the default comparisons should all have been measured"
    for comparison in comparisons:
        assert comparison.difference == pytest.approx(
            by_field[comparison.left] - by_field[comparison.right]
        )
        assert comparison.questions == report.questions
        assert 0.0 <= comparison.exact_mcnemar_p <= 1.0
        assert comparison.bootstrap_low <= comparison.bootstrap_high
        # Only the disagreements carry information.
        assert comparison.left_wins + comparison.right_wins <= comparison.questions


def test_an_ordering_compared_with_itself_is_not_a_result() -> None:
    analyzer = LexicalAnalyzer()
    router, questions = _router_over_stratum(analyzer)
    hits = ordering_hits(questions, router, depth=2)
    comparison = paired_ordering_comparison(hits, (("rrf_score", "rrf_score"),), resamples=100)[0]

    assert comparison.difference == 0.0
    assert comparison.left_wins == 0
    assert comparison.right_wins == 0
    assert comparison.exact_mcnemar_p == 1.0


def test_a_comparison_needs_both_orderings_measured() -> None:
    analyzer = LexicalAnalyzer()
    router, questions = _router_over_stratum(analyzer)
    hits = ordering_hits(questions, router, depth=2, fields=("rrf_score",))
    with pytest.raises(ValueError, match="measured"):
        paired_ordering_comparison(hits, (("rrf_score", "lexical_rank"),))


def test_held_out_fusion_reports_only_on_folds_it_did_not_fit() -> None:
    """A parameter fitted on the fold it is reported on has not been tested, only memorised."""

    from context_router.external.scale_qa import held_out_fusion

    analyzer = LexicalAnalyzer()
    windows = [
        ScaleQaWindow(f"w{index}", text, index, len(text.split()))
        for index, text in enumerate(_STRATUM_WINDOWS)
    ]
    questions = (
        _question("a", "otters lichen", "secondary material concerning otters"),
        _question("b", "falcon decoy", "a registry entry about the northern goshawk"),
        _question("c", "tidal barrages", "a discussion of tidal barrages"),
        _question("d", "volcanic ash", "an appendix about volcanic ash"),
        _question("e", "lichen growth", "notes on lichen growth rates"),
        _question("f", "decoy window", "the falcon decoy appears in this unrelated window"),
    )
    result = held_out_fusion(
        questions, windows, analyzer=analyzer, embedder=_embedder(analyzer), depth=2, seed=7
    )

    assert result.train + result.calibration + result.holdout == len(questions)
    assert set(result.channel_recall) == {"lexical", "dense", "entity"}
    assert set(result.gated_weights) == {"lexical", "dense", "entity"}
    # Gating keeps a channel only if its own recall is within tolerance of the best one.
    best = max(result.channel_recall.values())
    for name, recall in result.channel_recall.items():
        assert result.gated_weights[name] == float(recall >= best - result.tolerance)

    labels = {ordering.field for ordering in result.orderings}
    assert "lexical_rank (best member)" in labels
    assert "rrf_score (equal weight)" in labels
    assert "gated fusion + hand-set ranker" in labels
    for ordering in result.orderings:
        assert 0.0 <= ordering.recall_at_depth <= 1.0
        assert ordering.questions == result.holdout_hard
