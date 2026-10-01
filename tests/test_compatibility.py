"""The compatibility this product owes the frozen product that is still installed.

`memory-ultra-rag-mcp-server` is frozen and still installed, and it reads and writes the
same files this one serves. That is the whole reason its repository stays in the
collection, and it is a promise that decays quietly: a schema version bumped here alone
makes a memory this product writes unreadable to that one, and nothing in either test
suite would fail.

So the contract is stated here as a set of values rather than as a check against the
other repository. A test may not import the frozen product or read its files — that would
couple two release histories and make this product depend on a sibling's private state —
so the contract is pinned here and the frozen side is verified by reading its committed
revision, which is what this collection's own documentation points at.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from memory_rag import index, store
from memory_rag.config import GLOBAL_MEMORY_DIRNAME, GLOBAL_SCOPE_DIRECTORY
from memory_rag.config import LOCAL_STATE_DIRNAME as CONFIG_LOCAL_DIRNAME
from memory_rag.index import INDEX_FILENAME, SCHEMA_VERSION
from memory_rag.settings import (
    PROJECT_CONFIG_RELATIVE,
    SETTINGS,
    USER_CONFIG_DIRECTORY,
)

#: The prefix every environment variable this product reads carries. A user's existing
#: `MEMORY_ULTRARAG_*` settings have to keep applying, so the prefix is the frozen
#: product's rather than one of this product's own.
EXPECTED_ENVIRONMENT_PREFIX = "MEMORY_ULTRARAG_"

#: The revision of the frozen product whose formats this one keeps.
FROZEN_REVISION = "86243e108ac394a98b1eeb68029f7065e36b6c5c"

#: The record schema version the frozen product writes and reads. A change here is a
#: change to a file both products open, so it cannot be made in one of them.
EXPECTED_SCHEMA_VERSION = "8"


def test_the_schema_version_is_the_one_the_frozen_product_writes() -> None:
    """A version bumped in one product alone is a memory the other cannot serve.

    The frozen product refuses a record whose `schema_version` it does not know, so the
    cost of this number moving in one direction is every statement in every memory that
    product can still read.
    """

    assert SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION, (
        f"the record schema is now {SCHEMA_VERSION} and the frozen product reads "
        f"{EXPECTED_SCHEMA_VERSION}. Changing it here makes a memory this product writes "
        "unreadable to a product that is frozen and still installed. Both sides have to "
        "move together, and the frozen side cannot move at all, so this number is fixed "
        "for as long as that product is installed."
    )


def test_the_record_file_is_named_what_the_frozen_product_opens() -> None:
    """The file name is how either product finds the record at all."""

    assert INDEX_FILENAME == "memory.sqlite3"


def test_the_scope_directories_are_named_what_the_frozen_product_opens() -> None:
    """A rename strands every existing memory, because the old path stops being read.

    The names are the upstream layout's own, so a reader that expects that layout finds
    it unaltered, and the frozen product keeps writing the path it already knows.
    """

    assert CONFIG_LOCAL_DIRNAME == ".memory-rag", (
        "a project's local memory lives inside its repository; renaming the directory "
        "leaves every statement behind"
    )
    assert GLOBAL_MEMORY_DIRNAME == "memory"
    assert GLOBAL_SCOPE_DIRECTORY == "default", (
        "the global memory is one per account and the directory is a constant rather "
        "than a parameter, because the frozen product resolves exactly this path"
    )
    assert f"{GLOBAL_MEMORY_DIRNAME}/{GLOBAL_SCOPE_DIRECTORY}" == "memory/default"


def test_the_account_settings_directory_keeps_the_frozen_products_name() -> None:
    """It holds the project record, the model cache reference, and the settings.

    A rename would strand every existing memory's settings and split the project list
    between two files that could disagree about the same project, while the frozen
    product keeps reading the old one.
    """

    assert USER_CONFIG_DIRECTORY == "memory-ultra-rag-mcp"


def test_the_environment_prefix_keeps_the_frozen_products_name() -> None:
    """A user's existing `MEMORY_ULTRARAG_*` settings have to keep applying.

    The prefix is read off the declared names rather than from a constant, because each
    setting names its own environment variable and the shared stack must not be able to
    impose one. A setting added with a different prefix would therefore be a setting a
    user of the frozen product could not set, which is why every name is checked.
    """

    declared = {setting.env for setting in SETTINGS if setting.env}
    assert declared, "no setting declares an environment variable"
    for name in sorted(declared):
        assert name.startswith(EXPECTED_ENVIRONMENT_PREFIX), (
            f"{name} does not carry {EXPECTED_ENVIRONMENT_PREFIX}. A user who set it "
            "for the frozen product would find this product ignoring it."
        )


def test_the_project_settings_layer_is_still_the_projects_own_file() -> None:
    """A project's `config.toml` is inside the repository, where the frozen product
    looks for it."""

    assert Path(".memory-rag") / "config.toml" == PROJECT_CONFIG_RELATIVE


def test_a_vectors_shape_the_frozen_product_can_read_is_what_is_written() -> None:
    """The vector table's columns are a contract, not an implementation detail.

    A column the frozen product does not read is a statement it cannot serve
    semantically, and a changed width is a file it will refuse to open: `MemoryIndex`
    drops a vector table whose shape does not match and rebuilds it, which on a record
    the other product still serves means silently losing every meaning it held.
    """

    assert index.VECTOR_COLUMNS == (
        "unit_key",
        "stamp",
        "kind",
        "model",
        "dimension",
        "components",
    )
    normalised = " ".join(index.VECTOR_SCHEMA.split())
    for column in index.VECTOR_COLUMNS:
        assert column in normalised, f"the vector schema does not declare {column}"
    assert "PRIMARY KEY" in normalised
    assert "components BLOB" in normalised


def test_a_statement_shape_the_frozen_product_can_parse_is_what_is_written() -> None:
    """The document is prose under a heading, one statement per block, and no kind.

    That is what `read_document` parses when it adopts a document left by an earlier
    version, so the line format is a contract rather than a presentation choice: adding
    a kind to each line would change a file the frozen product reads.
    """

    document = store.render_document(
        [
            store.Statement(kind="RULE", text="first", key="a", position=1),
            store.Statement(kind="ITEM", text="second", key="b", position=0),
        ]
    )
    assert document == "# MEMORY\n\nfirst\n\nsecond\n", (
        "the rendered document is what the frozen product parses when it adopts one, so "
        "its line format is a contract"
    )
    assert "[RULE]" not in document and "[ITEM]" not in document, (
        "a kind in the rendered line would be a tag the frozen product's parser does "
        "not expect"
    )


def test_every_statement_column_still_names_its_reader() -> None:
    """A column without a reader is a field nothing can account for, and adding one is
    a change to a file both products open."""

    assert set(index.PRODUCTIVE_COLUMNS) == {"kind", "stamp", "added_at", "recalls"}
    for column, reader in index.PRODUCTIVE_COLUMNS.items():
        assert reader, f"the {column} column names no reader"


def test_the_unit_table_is_itself_the_word_index() -> None:
    """The property the whole SQL write rule rests on.

    `unit` is an FTS5 virtual table, so the words that find a statement and the
    statement itself are the same rows. A hand edit to `unit.text` therefore cannot put
    the word index out of step with the record, and the only derived state an edit can
    leave behind is a vector. If this ever stopped being true, the bounded write rule
    would be bounding the wrong thing.
    """

    source = (
        Path(__file__).resolve().parent.parent / "src" / "memory_rag" / "index.py"
    ).read_text(encoding="utf-8")
    assert re.search(r"CREATE VIRTUAL TABLE IF NOT EXISTS unit USING fts5\(", source), (
        "the statement table is no longer an FTS5 table"
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("the schema version", EXPECTED_SCHEMA_VERSION),
        ("the record file", "memory.sqlite3"),
        ("the local directory", ".memory-rag"),
        ("the global scope directory", "memory/default"),
        ("the account settings directory", USER_CONFIG_DIRECTORY),
    ],
)
def test_the_contract_is_stated_where_a_change_will_be_looked_for(
    name: str, value: str
) -> None:
    """The numbers are also written down, so a reader has somewhere to look.

    A contract that exists only as a passing test is one nobody reads before changing
    it, and it is the change to the record layer that breaks both products at once.
    """

    storage = (Path(__file__).resolve().parent.parent / "STORAGE.md").read_text(
        encoding="utf-8"
    )
    agents = (Path(__file__).resolve().parent.parent / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    assert value in storage, f"STORAGE.md does not name {name}: {value}"
    assert value in agents or name == "the record file", (
        f"AGENTS.md does not name {name}: {value}"
    )


def test_the_frozen_revision_this_contract_refers_to_is_a_full_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", FROZEN_REVISION), FROZEN_REVISION
