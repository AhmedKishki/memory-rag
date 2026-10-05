"""The SQL console: what a read may do, what a write may do, and what it repairs.

These are the rules that keep a hand edit from leaving a memory that answers wrongly.
A read is unrestricted and barely needs testing beyond working. A write is bounded, so
every refusal is stated here and checked, because a refusal that is not tested is a
refusal that a later change can quietly widen.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory_rag.index import MemoryIndex, index_path
from memory_rag.service import MemoryService
from memory_rag.sql import (
    MAXIMUM_STATEMENT,
    SqlRefusal,
    drop_vectors,
    sql_execute,
    sql_query,
)

RECORDED = "A project's local memory lives inside its repository."


@pytest.fixture
async def recorded(service: MemoryService):
    """A service holding one statement in its project memory."""

    await service.record(RECORDED, kind="RULE", scope="local")
    return service


def _ref(directory) -> object:
    from memory_rag.service import ScopeRef

    return ScopeRef(
        name="local", label="demo memory", directory=directory, kind="local"
    )


def _embed(statement, service: MemoryService) -> None:
    """Store one statement's vector, so a test can watch it be dropped.

    The worker embeds on its own thread, so waiting for it would make the test about
    timing. Writing the vector directly states the precondition instead.
    """

    from memory_rag.vectors import VectorStore

    identity = service.retrieval.embedder.identity
    with VectorStore(
        service.scope("local").directory, identity.name, identity.dimension
    ) as store:
        store.put(
            statement.key,
            (0.1, 0.2, 0.3),
            stamp=statement.added_at or "",
            kind=statement.kind,
        )


@pytest.fixture
def recorded_scope_directory(service: MemoryService, project_root: Path) -> Path:
    return project_root / ".memory-rag"


# -- a read is unrestricted ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_read_returns_the_columns_and_the_rows(recorded: MemoryService) -> None:
    answer = await recorded.sql_query("local", "SELECT kind, text FROM unit")
    assert answer["columns"] == ["kind", "text"]
    assert answer["rows"] == [["RULE", RECORDED]]
    assert answer["row_count"] == 1
    assert answer["scope"] == "demo"


@pytest.mark.asyncio
async def test_a_read_may_ask_anything_of_the_record(recorded: MemoryService) -> None:
    """A read is a read: the console does not second-guess what is being asked."""

    for statement in (
        "SELECT count(*) FROM unit",
        "SELECT * FROM meta",
        "SELECT name FROM sqlite_master ORDER BY name",
        "EXPLAIN SELECT 1",
    ):
        answer = await recorded.sql_query("local", statement)
        assert "rows" in answer, statement


@pytest.mark.asyncio
async def test_a_read_of_a_memory_with_nothing_in_it_is_an_answer(
    service: MemoryService,
) -> None:
    """A project that has never recorded anything has no file, and that is not a fault."""

    answer = await service.sql_query("global", "SELECT 1")
    assert answer["rows"] == []
    assert answer["empty_memory"] is True


@pytest.mark.asyncio
async def test_a_read_is_capped_and_says_so(recorded: MemoryService) -> None:
    answer = await recorded.sql_query(
        "local", "SELECT text FROM unit UNION ALL SELECT text FROM unit"
    )
    assert answer["row_count"] <= 500


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT nope FROM unit",
        "SELECT text FROM unit WHERE ??? = 1",
        "SELECT text FROM no_such_table",
    ],
)
@pytest.mark.asyncio
async def test_a_read_sqlite_cannot_run_is_a_refusal_that_names_why(
    recorded: MemoryService, statement: str
) -> None:
    """A read is unrestricted, and an impossible one is answered rather than raised.

    SQLite's own error is the useful part of the answer, and it names the statement that
    would not run. Letting it out of this module reaches the browser as a bare 500 and
    the terminal as a traceback, which tells a reader neither what they wrote nor why.
    """

    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_query("local", statement)
    assert "could not run that read" in str(raised.value)


@pytest.mark.asyncio
async def test_a_write_sent_as_a_query_is_refused_by_the_console(
    recorded: MemoryService,
) -> None:
    """The classifier refuses it, naming what it was."""

    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_query("local", "INSERT INTO unit (text) VALUES ('x')")
    assert "not a read" in str(raised.value)


def test_sqlite_itself_refuses_a_write_on_the_connection_a_read_uses(
    recorded_scope_directory: Path,
) -> None:
    """The guarantee does not rest on the classifier agreeing with the parser.

    The read path opens its connection read-only, so a statement that reached it by a
    route the classifier does not classify would still fail in the engine rather than
    change a memory.
    """

    from memory_rag.index import MemoryIndex

    with MemoryIndex(recorded_scope_directory) as index:
        index.insert("a statement", "RULE")
    connection = sqlite3.connect(
        f"file:{index_path(recorded_scope_directory)}?mode=ro", uri=True
    )
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM unit")
    finally:
        connection.close()


# -- a write is bounded to the statements --------------------------------------


@pytest.mark.asyncio
async def test_an_update_to_a_statement_is_accepted(recorded: MemoryService) -> None:
    answer = await recorded.sql_execute(
        "local", "UPDATE unit SET text = 'edited by hand' WHERE kind = 'RULE'"
    )
    assert answer["rows_affected"] == 1
    assert answer["reindexed"] is True
    assert answer["statements_changed"] == 1
    assert (await recorded.sql_query("local", "SELECT text FROM unit"))["rows"] == [
        ["edited by hand"]
    ]


@pytest.mark.asyncio
async def test_an_insert_and_a_delete_are_accepted(recorded: MemoryService) -> None:
    added = await recorded.sql_execute(
        "local", "INSERT INTO unit (text, unit_key, kind) VALUES ('new', 'k1', 'ITEM')"
    )
    assert added["rows_affected"] == 1
    assert (await recorded.sql_query("local", "SELECT count(*) FROM unit"))["rows"] == [
        [2]
    ]

    removed = await recorded.sql_execute(
        "local", "DELETE FROM unit WHERE unit_key = 'k1'"
    )
    assert removed["rows_affected"] == 1
    assert (await recorded.sql_query("local", "SELECT count(*) FROM unit"))["rows"] == [
        [1]
    ]


@pytest.mark.asyncio
async def test_a_replace_into_the_statements_is_accepted(
    recorded: MemoryService,
) -> None:
    """`REPLACE` is one of the write keywords this console names, so it is one it runs.

    The classifier's own list admits `replace`, and refusing the statement with a reason
    about an unnamed table would refuse a write that touches nothing but the statements.
    """

    answer = await recorded.sql_execute(
        "local",
        "REPLACE INTO unit (text, unit_key, kind, stamp) VALUES ('r', 'k9', 'ITEM', '3')",
    )
    assert answer["rows_affected"] == 1
    assert (
        await recorded.sql_query("local", "SELECT text FROM unit WHERE unit_key = 'k9'")
    )["rows"] == [["r"]]


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO unit (text, unit_key, kind, stamp) VALUES ('x', 'bad', 'RULE', 'first')",
        (
            "INSERT INTO unit (text, unit_key, kind, stamp, recalls) "
            "VALUES ('x', 'bad', 'RULE', '0', 'many')"
        ),
        "INSERT INTO unit (text, unit_key, kind) VALUES (NULL, 'bad', 'RULE')",
    ],
)
@pytest.mark.asyncio
async def test_a_write_leaving_a_row_the_record_cannot_read_is_refused(
    recorded: MemoryService, statement: str
) -> None:
    """A statement the record cannot count is refused rather than saved.

    The record reads a statement's place and its recall count as whole numbers, so a row
    holding words in either of them makes every later read of that memory fail with a
    conversion error that names nothing a reader can act on. The check runs on the
    connection the write already holds and before it is committed, so the write is
    rolled back and the memory is left as it was.
    """

    before = await recorded.sql_query("local", "SELECT text, kind FROM unit")
    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_execute("local", statement)
    assert "cannot read back" in str(raised.value)
    assert "rolled back" in str(raised.value)
    assert await recorded.sql_query("local", "SELECT text, kind FROM unit") == before


@pytest.mark.asyncio
async def test_a_write_that_counts_the_statements_it_changes_is_accepted(
    recorded: MemoryService,
) -> None:
    """The rule refuses rows the record cannot count, not rows it counts differently.

    A repair that sets a place or adds to a recall count in its own words is the ordinary
    way an operator reorders or recounts, so the numbers a statement legitimately holds
    have to be accepted.
    """

    answer = await recorded.sql_execute(
        "local",
        "UPDATE unit SET stamp = CAST(stamp AS INTEGER) + 5, recalls = 3",
    )
    assert answer["rows_affected"] == 1
    rows = (await recorded.sql_query("local", "SELECT stamp, recalls FROM unit"))[
        "rows"
    ]
    assert [int(stamp) for stamp, _recalls in rows] == [5]
    assert [int(recalls) for _stamp, recalls in rows] == [3]


@pytest.mark.asyncio
async def test_a_deleted_statement_stays_deleted_after_the_maintain(
    recorded: MemoryService,
) -> None:
    """The reindex that follows an accepted write must not put the statement back.

    A re-rendered document still holds whatever it held when it was written, so reading
    it as a source of statements would restore every statement a delete had just
    removed, and the write the user made by hand would be reported as done and undone
    in the same call.
    """

    removed = await recorded.sql_execute(
        "local", f"DELETE FROM unit WHERE text = {RECORDED!r}"
    )
    assert removed["rows_affected"] == 1
    assert (await recorded.sql_query("local", "SELECT count(*) FROM unit"))["rows"] == [
        [0]
    ]
    answer = await recorded.recall("repository", limit=5)
    assert answer["units"] == []


@pytest.mark.asyncio
async def test_a_forgotten_statement_is_not_brought_back_by_a_later_read(
    recorded: MemoryService,
) -> None:
    """The same rule on the path an agent takes, through the same reindex."""

    answer = await recorded.forget(RECORDED)
    assert answer["status"] == "forgotten"
    with MemoryIndex(recorded.scope("local").directory) as index:
        index.write_render()
        index.retire_superseded(render=True)
        index.write_render()
    assert (await recorded.recall("repository", limit=5))["units"] == []


@pytest.mark.asyncio
async def test_a_write_to_another_table_is_refused_by_name(
    recorded: MemoryService,
) -> None:
    for statement in (
        "UPDATE meta SET value = '9'",
        "DELETE FROM meta",
        "DELETE FROM vector",
    ):
        with pytest.raises(SqlRefusal) as raised:
            await recorded.sql_execute("local", statement)
        assert "may only touch the unit table" in str(raised.value)


@pytest.mark.parametrize(
    "statement",
    [
        "DROP TABLE unit",
        "CREATE TABLE other (a int)",
        "ALTER TABLE unit ADD COLUMN extra text",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE '/tmp/other.sqlite3' AS other",
        "VACUUM",
        "DELETE FROM unit_data",
    ],
)
@pytest.mark.asyncio
async def test_a_write_that_changes_the_file_is_refused_by_name(
    recorded: MemoryService, statement: str
) -> None:
    with pytest.raises(SqlRefusal):
        await recorded.sql_execute("local", statement)


@pytest.mark.asyncio
async def test_a_write_naming_a_protected_table_anywhere_is_refused(
    recorded: MemoryService,
) -> None:
    """The table a write claims is checked, and so is every table it also mentions.

    A statement that targets `unit` and reads from `vector` in the same breath would
    otherwise reach past the bound the target alone appears to respect.
    """

    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_execute(
            "local",
            "INSERT INTO unit (text, unit_key) SELECT text, unit_key FROM vector",
        )
    assert "protected" in str(raised.value)


@pytest.mark.asyncio
async def test_two_statements_at_once_are_refused(recorded: MemoryService) -> None:
    """One statement per run, because a write is reindexed once."""

    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_execute(
            "local",
            "INSERT INTO unit (text, unit_key) VALUES ('a','k1'); "
            "INSERT INTO unit (text, unit_key) VALUES ('b','k2')",
        )
    assert "one statement at a time" in str(raised.value)


@pytest.mark.asyncio
async def test_a_comment_cannot_hide_a_second_statement(
    recorded: MemoryService,
) -> None:
    """The comment is removed before the statement is counted, so it cannot hide one.

    A `;` inside a comment is the ordinary way a second statement gets in, and counting
    the statement with the comment still attached would see two.
    """

    answer = await recorded.sql_execute(
        "local",
        "INSERT INTO unit (text, unit_key) VALUES ('a','k1') -- ; DROP TABLE unit",
    )
    assert answer["rows_affected"] == 1
    # The table is still there, which is the point: the DROP was in a comment and was
    # never part of the statement that ran.
    assert (await recorded.sql_query("local", "SELECT count(*) FROM unit"))["rows"] == [
        [2]
    ]


@pytest.mark.parametrize("statement", ["", "   ", "-- just a comment", "SELECT"])
@pytest.mark.asyncio
async def test_nothing_to_run_is_refused(
    recorded: MemoryService, statement: str
) -> None:
    with pytest.raises(SqlRefusal):
        await recorded.sql_execute("local", statement)


@pytest.mark.asyncio
async def test_a_statement_over_the_limit_is_refused_rather_than_truncated(
    recorded: MemoryService,
) -> None:
    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_execute(
            "local",
            "INSERT INTO unit (text) VALUES ('" + "x" * MAXIMUM_STATEMENT + "')",
        )
    assert str(MAXIMUM_STATEMENT) in str(raised.value)


@pytest.mark.asyncio
async def test_a_write_sqlite_itself_refuses_leaves_the_memory_as_it_was(
    recorded: MemoryService,
) -> None:
    """A statement that passes the console and then fails in the engine changes nothing.

    The console checks what a statement may do; SQLite decides whether it can be done.
    A failure from the engine rolls the write back, and the record is left as it was
    rather than half changed.
    """

    before = await recorded.sql_query("local", "SELECT text, kind FROM unit")
    with pytest.raises(SqlRefusal) as raised:
        await recorded.sql_execute("local", "INSERT INTO unit (nope) VALUES ('x')")
    assert "SQLite refused" in str(raised.value)
    assert await recorded.sql_query("local", "SELECT text, kind FROM unit") == before


# -- what an accepted write repairs --------------------------------------------


@pytest.mark.asyncio
async def test_an_accepted_write_drops_the_vector_of_the_statement_it_changed(
    recorded: MemoryService,
) -> None:
    """The one thing a hand edit can leave behind is a vector about old words.

    A vector describes the text its statement held when the model last saw it, so a
    statement whose text changed has a vector that now answers a question nobody asked.
    """

    scope = recorded.scope("local")
    directory = scope.directory
    recorded.retrieval.maintain(directory)
    with MemoryIndex(directory) as index:
        statements = index.statements()
        index.collect_vectors()
    _embed(statements[0], recorded)

    def _vectors() -> int:
        with sqlite3.connect(index_path(directory)) as connection:
            return connection.execute("SELECT count(*) FROM vector").fetchone()[0]

    assert _vectors() >= 1

    await recorded.sql_execute("local", "UPDATE unit SET text = 'a different claim'")

    assert _vectors() == 0, "the vector describing the old words was kept"


@pytest.mark.asyncio
async def test_an_accepted_write_rewrites_the_standing_document(
    recorded: MemoryService,
) -> None:
    answer = await recorded.sql_execute(
        "local", "UPDATE unit SET text = 'the document says this now'"
    )
    rendered = Path(answer["rendered"])
    assert "the document says this now" in rendered.read_text(encoding="utf-8")
    assert rendered.name == "MEMORY.md"


@pytest.mark.asyncio
async def test_a_hand_edited_statement_is_still_found_by_a_recall(
    recorded: MemoryService,
) -> None:
    """The end the whole write rule exists for.

    A statement edited by hand has to answer a recall afterwards. If the write left
    the word index or the vectors describing the old text, the memory would answer
    wrongly rather than fail, and nothing else in the product would notice.
    """

    await recorded.sql_execute(
        "local", "UPDATE unit SET text = 'An unused letterpress'"
    )
    answer = await recorded.recall("letterpress", limit=5)
    assert [unit["text"] for unit in answer["units"]] == ["An unused letterpress"]


@pytest.mark.asyncio
async def test_dropping_vectors_is_a_no_op_without_keys(tmp_path) -> None:
    assert drop_vectors(tmp_path, []) == 0
    assert drop_vectors(tmp_path / "absent", ["k"]) == 0


def test_sql_execute_reports_what_it_touched_even_without_a_service(tmp_path) -> None:
    """The reindex decision is made from the record before and after, not from the
    statement's own WHERE clause, so it holds for a write the classifier never read."""

    from memory_rag.index import MemoryIndex

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("first", "RULE")
        index.insert("second", "RULE")
    answer = sql_execute(_ref(directory), "DELETE FROM unit WHERE kind = 'RULE'")
    assert len(answer["touched"]) == 2
    assert sql_query(_ref(directory), "SELECT count(*) FROM unit")["rows"] == [[0]]


def test_rebuilding_a_scope_keeps_the_statements(tmp_path) -> None:
    """`rebuild` settles the derived state and never removes the record.

    A rebuild that deleted `memory.sqlite3` would destroy every statement in the
    memory, and the only recovery would be retyping them from a rendering that is
    written from the record and never read back.
    """

    from memory_rag.index import MemoryIndex, index_path, rebuild

    directory = tmp_path / "scope"
    with MemoryIndex(directory) as index:
        index.insert("a statement worth keeping", "RULE")
        index.insert("another one", "RULE")

    report = rebuild(directory)

    assert index_path(directory).is_file()
    assert report["statements"] == 2
    assert sql_query(_ref(directory), "SELECT count(*) FROM unit")["rows"] == [[2]]
