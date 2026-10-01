"""The document beside the record, and the words of a statement.

The record is the table in ``memory.sqlite3``; this module is the file written out
from it, and the two shapes a statement takes on the way: a row, and a block of
prose.

**The document is gone; what is here renders one.** The record is a table, and a
table is the memory: one row per statement, holding its words, its kind, its place,
when it was added, and how often it has been recalled. There is no file beside it,
because a second copy of the same statements is a second thing to keep in step and
one of them is eventually wrong.

What survives of a file is the shape. ``render_document`` turns a record into the
prose a person reads — which is what the browser view shows, and what a caller
could be shown — and ``parse_document`` turns that prose back into statements,
which is how the page saves an edit and how a memory of an older version is
recovered. A document left by an earlier version is read once, for anything the
record lacks, and then removed; that is what ``index.retire_superseded`` does, and
it is the only path that reads a file.

**A statement is one line of prose, and its type is a row, not a prefix.** Upstream
never writes a statement, so the block format is this package's; the file it seeds
with is upstream's, byte for byte. The type used to be written into the block as a
``RULE: `` prefix, and it is not any more: a file of tagged lines reads as data
rather than as something a person wrote, and a type is a property of the row that
holds the statement. A statement imported from a file that still has the prefix
keeps the type it had there, and the next export drops the prefix.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "DEFAULT_KIND",
    "EXPORT_FILENAME",
    "HANDOFF_KIND",
    "MAX_KIND_LENGTH",
    "SEED",
    "STOPWORDS",
    "TEMPLATE",
    "Statement",
    "StoreError",
    "normalise",
    "parse_document",
    "query_terms",
    "read_document",
    "render_document",
    "statement_kind",
    "unit_key",
]

#: The file a previous version wrote its statements into, and upstream's own
#: name. It is read once, for a migration, and then removed.
EXPORT_FILENAME = "MEMORY.md"

#: The document an earlier version created, byte for byte as upstream seeds it.
#: Kept so a file written by that version is recognised and never mistaken for a
#: statement.
TEMPLATE = "# MEMORY\ni am jack. i like LLMs.\n"

#: The line under the heading in that seed. It is upstream's placeholder, not a
#: memory, and an export does not carry it — a file a person opens should hold
#: what was remembered and nothing else. A reader still skips it by name, because
#: a file written before this version has it.
SEED = "i am jack. i like LLMs."

#: What a statement is filed under when the caller names none. Every statement
#: carries a type, so the default is a real one rather than an absence: a query
#: naming `ITEM` finds the statements nobody else filed, which is the only way they
#: can be found at all.
DEFAULT_KIND = "ITEM"

#: The type a session handoff is filed under. It is reserved: a handoff replaces
#: the previous one instead of accumulating beside it, and the replacement is found
#: by this type rather than by anything the caller has to remember to pass.
HANDOFF_KIND = "HANDOFF"

#: Long enough for a word, short enough that a type cannot become a phrase.
MAX_KIND_LENGTH = 64

_HEADING_PATTERN = re.compile(r"^#\s+\S")

#: The type a statement imported from a file that still writes it into the block.
#: Read on import only; never written.
_LEGACY_KIND_PATTERN = re.compile(r"^([A-Z]{1,64}): ")


class StoreError(ValueError):
    """Raised when a scope cannot be written the way this store writes one."""


def normalise(value: str) -> str:
    """Return text with its white space collapsed, for an exact comparison.

    Forgetting matches words and never meaning, so the comparison has to be one a
    caller can predict from the text a read gave them. Collapsing white space is
    the only slack in it, and it is slack about spacing rather than about words.
    """

    return " ".join(str(value or "").split())


def unit_key(text: str) -> str:
    """Return the stable identity one statement is known by.

    A digest of the statement's own words rather than of its place, so a statement
    inserted above another does not change its identity and a statement reworded
    becomes a different one. A row, its vector, and its history are the same row,
    and this is how they are found.
    """

    return hashlib.sha256(normalise(text).casefold().encode("utf-8")).hexdigest()


def statement_kind(value: str) -> str:
    """Return one statement's label as it is written on its row.

    A type is the caller's own category for a statement, and it carries no meaning
    here: the record holds it and never interprets it. It exists to help a read
    find the statement, so it is one run of block letters and nothing else — no
    white space, no punctuation, no digits. A single word is a category; a phrase
    would put a sentence where a word belongs, and anything but letters would make
    a query for it unwriteable.

    Case is the one thing normalised rather than refused: a type given in any case
    is recorded in block letters, so what the record holds is always the shape the
    read side recognises. An empty type is not refused either: every statement
    carries one, so a caller that names none gets `ITEM`.
    """

    raw = str(value or "").strip()
    if not raw:
        return DEFAULT_KIND
    if any(character.isspace() for character in raw):
        raise StoreError(
            "type must be one word with no white space in it: a type is a single "
            "category, and a phrase would be part of the statement."
        )
    if not (raw.isascii() and raw.isalpha()):
        raise StoreError(
            "type must be block letters only, as in RULE or PLAN: a type carries "
            "no meaning beyond helping a read find its statements."
        )
    if len(raw) > MAX_KIND_LENGTH:
        raise StoreError(
            f"type must be at most {MAX_KIND_LENGTH} characters: it named {len(raw)}."
        )
    return raw.upper()


@dataclass(frozen=True, slots=True)
class Statement:
    """One statement, as the record holds it.

    ``text`` is the statement and ``type`` is what it was filed under, which are
    now two fields rather than one tagged line. ``position`` counts from the top
    of the document, so zero is the statement recorded last, and it is the order
    both the export and a read's preference for new information follow.
    """

    kind: str
    text: str
    key: str
    position: int
    added_at: str | None = None
    recalls: int = 0

    @property
    def normalized(self) -> str:
        """Return the text as an exact match compares it."""

        return normalise(self.text)


def parse_document(
    document: str, *, newest_first: bool | None = None
) -> list[Statement]:
    """Return the statements a document holds, with their positions.

    Read once, on import, for a file written when the document was the record. The
    heading and the seed are not statements, and a block that opens with a kind and
    a colon carries that kind into the row.

    A file that carries a ``KIND: `` prefix was written by a version that appended,
    so its last statement is its newest: importing one in file order would file the
    oldest statement as the most recent and hand it the preference for new
    information. That is the one signal in the file itself, so it is what the
    order is decided by. A file without prefixes was written newest-first, or
    typed by a person, and is read in the order it holds.

    The heading is a line of its own and is taken off first, because a document
    whose statement follows the title on the next line rather than after a blank
    one is a document a person wrote, and one block that happens to begin with a
    title must not swallow the statement under it.
    """

    body = document
    first, _newline, rest = document.partition("\n")
    if _HEADING_PATTERN.match(first.strip()):
        body = rest

    found: list[Statement] = []
    prefixed = False
    for chunk in body.split("\n\n"):
        block = chunk.strip()
        if not block or block in (SEED, TEMPLATE.strip()):
            continue
        match = _LEGACY_KIND_PATTERN.match(block)
        prefixed = prefixed or match is not None
        if match is not None:
            label, body = match.group(1), block[match.end() :].strip()
        else:
            label, body = DEFAULT_KIND, block
        found.append(
            Statement(
                kind=label,
                text=body,
                key=unit_key(f"{label}: {body}"),
                position=len(found) if newest_first else -1,
            )
        )
    if newest_first is None:
        # A prefixed file appended, so its last statement is its newest; anything
        # else was written newest-first or typed, and is read as it stands.
        newest_first = not prefixed
    if not newest_first:
        found = [
            replace(statement, position=position)
            for position, statement in enumerate(reversed(found))
        ]
    return found


def render_document(statements: list[Statement]) -> str:
    """Return the document a record is written out as: prose, newest first.

    Plain statements, one per block, under the heading, and nothing else. No type
    is written, because a type is a field of the row and a file of tagged lines
    reads as data rather than as something a person wrote; and no seed, because
    upstream's placeholder is not a memory and a file a person opens should hold
    what was remembered.
    """

    parts = ["# MEMORY", *(statement.text for statement in statements)]
    return "\n\n".join(parts) + "\n"


def read_document(scope_directory: Path) -> str | None:
    """Return the document a previous version left, or None when there is none.

    The only read of a file this server ever makes, and it is a migration: a
    document written when the document was the record, read once for anything the
    record lacks and then removed by ``MemoryIndex.retire_superseded``.
    """

    document = scope_directory / EXPORT_FILENAME
    if not document.is_file():
        return None
    try:
        return document.read_text(encoding="utf-8")
    except OSError:
        return None


#: The English words a lookup does not need. This memory is English, and the word
#: side requires every term it is given, so a term in no statement would not narrow
#: the answer, it would empty it.
STOPWORDS = frozenset(
    [
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "but",
        "by",
        "can",
        "come",
        "could",
        "day",
        "do",
        "even",
        "first",
        "for",
        "from",
        "get",
        "give",
        "go",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "know",
        "like",
        "make",
        "man",
        "many",
        "me",
        "more",
        "most",
        "my",
        "new",
        "no",
        "not",
        "now",
        "of",
        "on",
        "one",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "people",
        "say",
        "see",
        "she",
        "so",
        "some",
        "take",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "thing",
        "think",
        "this",
        "those",
        "three",
        "to",
        "two",
        "up",
        "us",
        "use",
        "very",
        "want",
        "was",
        "way",
        "we",
        "well",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "year",
        "you",
        "your",
    ]
)


def query_terms(query: str) -> tuple[str, ...]:
    """Return the distinct case-folded terms one query searches for.

    Words are split on whitespace and stripped of surrounding punctuation, then
    English stop words are dropped: the word side requires every remaining term,
    so a term that decides nothing can only lose the answer. A query made
    entirely of stop words keeps them, because dropping all of them would leave
    the index nothing to search on and the read would report an empty memory
    rather than an unusual question.
    """

    terms: list[str] = []
    for word in str(query or "").casefold().split():
        cleaned = word.strip(".,;:!?\"'`()[]{}<>*_#-")
        if cleaned and cleaned not in terms:
            terms.append(cleaned)
    meaningful = [term for term in terms if term not in STOPWORDS]
    return tuple(meaningful or terms)
