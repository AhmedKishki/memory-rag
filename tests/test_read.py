"""The read: which of an answer's statements are the same statement.

Two tests decide it, in order: the words, and then the meaning. The second only applies
where there is a vector to compare, so these tests hold the vectors out where that is
what is under test.
"""

from __future__ import annotations

from typing import Any

from memory_rag.read import collapse_repetitions


def _ranked(
    *statements: tuple[str, tuple[float, ...]],
) -> list[tuple[float, int, str, str, tuple[float, ...] | None, dict[str, Any]]]:
    """Build the candidates a merge hands over: score, place, key, words, vector, stated."""

    return [
        (
            1.0 / (index + 1),
            len(statements) - index,
            f"key-{index}",
            text,
            vector,
            {"scope": "local", "text": text, "kind": "RULE", "recalls": 0},
        )
        for index, (text, vector) in enumerate(statements)
    ]


def test_a_statement_with_no_vector_is_never_collapsed_by_cosine() -> None:
    """There is nothing to compare, so only the words test applies to it.

    A vector that is missing arrives as an empty tuple, and an empty tuple compared with
    another one has a similarity of nothing. At a threshold of zero — the setting that
    turns the comparison off — every statement without a vector folded into whichever
    such statement came before it, reported as a repetition of it.
    """

    kept, collapsed = collapse_repetitions(
        _ranked(
            ("the build runs from one command", ()),
            ("the deploy runs from one command", ()),
        ),
        threshold=0.0,
        limit=10,
    )
    assert [row["text"] for row in kept] == [
        "the build runs from one command",
        "the deploy runs from one command",
    ]
    assert collapsed == []


def test_a_vector_above_the_threshold_still_collapses() -> None:
    """The comparison still does its job where there is something to compare."""

    kept, collapsed = collapse_repetitions(
        _ranked(
            ("the build runs from one command", (1.0, 0.0, 0.0)),
            ("the deploy runs from one command", (1.0, 0.0, 0.0)),
        ),
        threshold=0.85,
        limit=10,
    )
    assert len(kept) == 1
    assert collapsed[0]["collapsed_by"] == "same meaning"
    assert collapsed[0]["similarity"] == 1.0


def test_a_vector_below_the_threshold_does_not_collapse() -> None:
    """Two statements that are only a little alike are two statements."""

    kept, collapsed = collapse_repetitions(
        _ranked(
            ("the build runs from one command", (1.0, 0.0, 0.0)),
            ("the deploy runs from one command", (0.6, 0.8, 0.0)),
        ),
        threshold=0.85,
        limit=10,
    )
    assert len(kept) == 2
    assert collapsed == []


def test_the_same_words_are_one_statement_whatever_the_vectors_say() -> None:
    """The words test runs first and does not need a vector to be sure."""

    kept, collapsed = collapse_repetitions(
        _ranked(
            ("the build runs from one command", ()),
            ("the build runs from one command", ()),
        ),
        threshold=0.0,
        limit=10,
    )
    assert len(kept) == 1
    assert collapsed[0]["collapsed_by"] == "same words"
