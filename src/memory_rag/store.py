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
    "KINDS",
    "MAX_KIND_LENGTH",
    "SCOPE_EITHER",
    "SCOPE_GLOBAL",
    "SCOPE_LOCAL",
    "SEED",
    "STOPWORDS",
    "TEMPLATE",
    "Kind",
    "Statement",
    "StoreError",
    "admission_failure",
    "kind_named",
    "kind_names",
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

#: What a statement is filed under when it is imported from a document that does not
#: say. It is not a kind an agent may choose: a document written by an earlier version
#: carries no kind at all, so importing one has to file its statements under something,
#: and the frozen product's own parser reads a kindless line as ``ITEM``.
DEFAULT_KIND = "ITEM"

#: The type a session handoff is filed under. It is reserved: a handoff replaces
#: the previous one instead of accumulating beside it, and the replacement is found
#: by this type rather than by anything the caller has to remember to pass.
HANDOFF_KIND = "HANDOFF"

#: Long enough for a word, short enough that a type cannot become a phrase.
MAX_KIND_LENGTH = 64

#: The scope a kind may be filed in: the caller's choice.
SCOPE_EITHER = "either"

#: The scope a kind is always filed in, because the statement is true of the account
#: rather than of a repository. What is true of the user is true in every project, so
#: filing it in one would be filing a statement that is wrong everywhere else.
SCOPE_GLOBAL = "global"

#: The scope a kind is always filed in, because the statement is about this project's
#: own work and is not true of any other.
SCOPE_LOCAL = "local"


@dataclass(frozen=True, slots=True)
class Kind:
    """One kind of statement, and what filing under it means.

    ``summary`` is what goes in the tool that records it, so the one sentence an agent
    reads before choosing is written here beside the behaviour it chooses, and the two
    cannot drift apart. ``scope`` is the memory the kind is filed in, and a kind that
    is not ``SCOPE_EITHER`` takes no scope argument at all rather than being given one
    it may contradict. ``replaces`` is a bounded lifetime: the kind holds one statement
    and the next one of that kind removes it, which is what a handoff is.
    """

    name: str
    summary: str
    scope: str = SCOPE_EITHER
    replaces: bool = False

    @property
    def tool(self) -> str:
        """Return this kind as it appears in a tool name."""

        return self.name.casefold()


#: Every kind this product files a statement under, and the order they are offered in.
#: The set is closed on the write path: a word that is not here is refused by name,
#: because a kind nothing interprets is a category a later recall cannot ask for and
#: a statement filed under one of them is found only by accident.
#:
#: A read stays permissive. A record may already hold a kind this set does not name,
#: written by an earlier version or by the product this one replaces, and a recall
#: that refused to search it would make those statements unreadable rather than tidy.
KINDS: tuple[Kind, ...] = (
    Kind(
        "RULE",
        "An instruction that was given, or a standing fact that holds. Call it when "
        "the next request in three weeks would still be governed by what you write.",
    ),
    Kind(
        "GUIDELINE",
        "How to do something here: a convention, an order of preference, a way of "
        "working. Softer than a rule, because a guideline is revised rather than "
        "broken.",
    ),
    Kind(
        "PERSONALITY",
        "What is true of the user: how they think, what they care about, what they "
        "are like.",
        scope=SCOPE_GLOBAL,
    ),
    Kind(
        "GENERAL",
        "Anything meant to hold that is none of the other kinds. Choose it when the "
        "statement is plainly durable and plainly not one of the rest.",
    ),
    Kind(
        "DECISION",
        "A decision that was made and why. The reasoning goes in as well as the "
        "choice, because the reasoning is what a later session needs and the choice "
        "alone does not say when it stops holding.",
    ),
    Kind(
        "HANDOFF",
        "This session's handoff: what is done, what is in flight, and what the "
        "next session does first.",
        scope=SCOPE_LOCAL,
        replaces=True,
    ),
    Kind(
        "CORRECTION",
        "Something to stop doing. Record what was wrong rather than what is right now, "
        "so a later session recognises the same mistake when it meets it again.",
    ),
    Kind(
        "PREFERENCE",
        "What the user likes or wants: a way of being spoken to, a form an answer "
        "should take.",
        scope=SCOPE_GLOBAL,
    ),
    Kind(
        "PLAN",
        "What the project is meant to be, or what it is achieving. A destination "
        "rather than a task, and it survives the work that serves it.",
    ),
    Kind(
        "LIMITS",
        "A boundary: what must not be done here, and what is out of scope. Write the "
        "boundary rather than the rule that replaces it.",
    ),
)

_KINDS_BY_NAME: dict[str, Kind] = {kind.name: kind for kind in KINDS}


def kind_names() -> tuple[str, ...]:
    """Return every kind this product files a statement under, in its own order."""

    return tuple(kind.name for kind in KINDS)


def kind_named(name: str) -> Kind:
    """Return the kind this word names, or refuse a word that names none.

    The refusal names every kind rather than saying the word was unknown, because a
    caller that cannot tell which words are accepted cannot choose one, and the set is
    the whole of what there is to choose from.
    """

    entry = _KINDS_BY_NAME.get(str(name or "").strip().upper())
    if entry is None:
        raise StoreError(
            f"{str(name or '').strip()!r} is not a kind this memory files under. "
            f"The kinds are: {', '.join(kind_names())}."
        )
    return entry


_HEADING_PATTERN = re.compile(r"^#\s+\S")

#: The type a statement imported from a file that still writes it into the block.
#: Read on import only; never written.
_LEGACY_KIND_PATTERN = re.compile(r"^([A-Z]{1,64}): ")

#: A calendar date or a clock time written in a statement's own words. The record
#: dates every statement itself, so one that carries a date says the same thing
#: twice, and the copy in the prose is the one that goes stale: it is the token
#: that differs between two accounts of one event, and a statement's identity is a
#: digest of its words. One approval recorded per session therefore becomes one
#: statement per session rather than one approval.
#:
#: A year alone is not matched, because "(HUP 1971)" names the edition meant and
#: that has to hold next year.
_MOMENT_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b"
)


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

    A kind is one word in block letters, and this returns it in that shape: it is the
    record's own spelling of a category, and a word in any other case is recorded in
    block letters so what the record holds is always the shape the read side
    recognises. Whether the word is one of them is a separate question, answered by
    :func:`kind_named`, because this says how a word is written rather than whether
    it is allowed.

    An empty kind is refused rather than defaulted. Every statement carries a kind,
    and with the set closed a caller that names none has named one that is not in it.
    """

    raw = str(value or "").strip()
    if not raw:
        raise StoreError(
            "kind must be named: every statement is filed under one, and the kinds "
            f"are {', '.join(kind_names())}."
        )
    if any(character.isspace() for character in raw):
        raise StoreError(
            "kind must be one word with no white space in it: a kind is a single "
            "category, and a phrase would be part of the statement."
        )
    if not (raw.isascii() and raw.isalpha()):
        raise StoreError(
            "kind must be block letters only, as in RULE or PLAN: a kind names a "
            "category and carries no meaning beyond helping a read find its statements."
        )
    if len(raw) > MAX_KIND_LENGTH:
        raise StoreError(
            f"kind must be at most {MAX_KIND_LENGTH} characters: it named {len(raw)}."
        )
    return raw.upper()


def admission_failure(text: str) -> str | None:
    """Return why a statement may not be recorded, or ``None`` when it may.

    One rule, and it is the one the record's own shape demands. Every statement is
    dated when it is recorded: ``added_at`` holds it and a read reports it, so a
    statement that carries a date or a clock time in its own words is reporting a
    moment rather than saying what holds. Two such statements about one event are
    two rows, because a statement's identity is a digest of its words and the date
    is the part that differs, so a memory recorded across sessions fills with one
    row per session saying that the work was done.

    The refusal names where the statement belongs rather than restating the rule,
    because a statement about what was done is not a bad statement. It is a handoff:
    ``record_memory_handoff`` holds exactly one and replaces it with the next
    session's, so
    progress can be recorded without becoming permanent history.
    """

    found = _MOMENT_PATTERN.search(str(text or ""))
    if found is None:
        return None
    return (
        f"a statement states what holds, not when it happened, and this one carries "
        f"{found.group(0)!r}. The memory dates every statement itself and a recall "
        f"reports that date, so a statement that writes the date out says the same "
        f"thing twice — and because a statement is known by a digest of its words, "
        f"the date is exactly what makes two accounts of one event two statements "
        f"instead of one. Leave the date out. If this is what was done in this "
        f"session rather than what holds afterwards, record_memory_handoff is the "
        f"tool for "
        f"it, and it holds one handoff and replaces it with the next."
    )


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
