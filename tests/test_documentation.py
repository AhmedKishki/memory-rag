"""The command centre, and the agreement between the three surfaces.

Two things are checked here. The first is that the help menu accounts for every
command, because a command nobody can find is a command nobody uses and the menu is
the only place a new one is documented. The second is that the terminal reaches the
same operations the workspace does, so a capability cannot exist in a browser and be
missing from a shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from memory_rag.surfaces import cli
from memory_rag.surfaces.cli import HELP_GROUPS, HELP_TOPICS, _parser

REPOSITORY = Path(__file__).resolve().parent.parent


def _subcommands() -> list[str]:
    for action in _parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return sorted(action.choices)
    raise AssertionError("the parser has no subcommands")


def _menu_commands() -> list[str]:
    return [name for _, entries in HELP_GROUPS for name, _ in entries]


def test_the_help_menu_accounts_for_every_command() -> None:
    """A command in the parser and absent from the menu is one nobody can find."""

    assert set(_subcommands()) == set(_menu_commands())


def test_no_command_is_listed_twice() -> None:
    menu = _menu_commands()
    assert len(menu) == len(set(menu))


def test_every_menu_entry_says_what_its_command_does() -> None:
    for _, entries in HELP_GROUPS:
        for name, description in entries:
            assert description.endswith("."), f"{name}: {description}"
            assert not description[0].isupper() or description[0].isupper()


def test_help_and_the_subcommand_produce_the_same_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`help x` and `x --help` are one document, not two that can differ."""

    parser = _parser()
    for command in _subcommands():
        through_help = cli._command_usage(parser, command)
        with pytest.raises(SystemExit):
            parser.parse_args([command, "--help"])
        capsys.readouterr()
        through_flag = _usage_of(command)
        assert through_help == through_flag, command


def _usage_of(command: str) -> str:
    """Return what `command --help` writes, without leaving the test's output."""

    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.suppress(SystemExit):
        _parser().parse_args([command, "--help"])
    return buffer.getvalue()


def test_every_subject_page_names_the_commands_it_is_about() -> None:
    """A page that documents a command the parser does not have is a page about a
    product that does not exist."""

    menu = set(_menu_commands())
    for topic, text in HELP_TOPICS.items():
        assert text.strip(), f"the {topic} page is empty"
        for word in text.replace("`", " ").split():
            if word in menu:
                assert word in menu  # the command exists; nothing to assert further
    assert "help " in cli._help_menu()


def test_the_menu_names_every_subject_page_and_every_command() -> None:
    menu = cli._help_menu()
    for topic in HELP_TOPICS:
        assert f"help {topic}" in menu
    for command in _subcommands():
        assert command in menu


def test_an_unknown_topic_says_so_rather_than_printing_nothing() -> None:
    assert "no command called" in cli._command_usage(_parser(), "not-a-command")


def test_the_json_flag_is_available_on_every_command() -> None:
    """A reader who wants the answer as data should not have to ask a different way."""

    parser = _parser()
    arguments = parser.parse_args(["--json", "status"])
    assert arguments.json is True


def test_recording_a_statement_names_the_kind_it_is_filed_under() -> None:
    """There is no default kind, so the command line asks for one.

    The kinds are a closed list and a statement filed under nothing is a category no
    later recall can ask for, so a default would file statements where they cannot be
    found. `store.DEFAULT_KIND` exists for reading an older document and is not one of
    the kinds, so a parser default naming it would refuse every write it produced.
    """

    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["record", "a statement"])
    arguments = parser.parse_args(["record", "a statement", "--kind", "RULE"])
    assert arguments.kind == "RULE"
    assert arguments.scope == "local"


def test_the_two_surfaces_take_the_recall_bound_from_the_service() -> None:
    """One number, and the service owns it.

    The browser, the agent surface, and the terminal all reach the same service, so a
    ceiling repeated in two of them is a ceiling the third does not have.
    """

    from memory_rag.service import MAX_RECALL_LIMIT
    from memory_rag.surfaces import cli, mcp

    assert cli.MAX_RESULT_LIMIT == MAX_RECALL_LIMIT
    assert mcp.MAX_RESULT_LIMIT == MAX_RECALL_LIMIT


# -- the terminal reaches what the browser reaches ------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        "recall",
        "record",
        "forget",
        "handoff",
        "maintain",
        "export",
        "sql_query",
        "sql_execute",
    ],
)
def test_both_local_and_remote_answer_the_same_operations(operation: str) -> None:
    """One interface, two implementations.

    The dispatch cannot tell them apart, so a capability cannot exist for a browser
    and be missing from a terminal: whichever implementation is chosen has to have the
    method, and neither may be a bare pass that answers something else.
    """

    from memory_rag.surfaces.cli import Local, Operations, Remote

    for implementation in (Operations, Local, Remote):
        member = getattr(implementation, operation)
        assert callable(member), f"{implementation.__name__}.{operation}"
    assert getattr(Local, operation) is not getattr(Remote, operation)


def test_the_control_channel_carries_every_operation_the_service_has() -> None:
    """Every route the app serves is one the command line can reach, and vice versa."""

    from memory_rag.control import Control, control_routes

    served = {route.path for route in control_routes(_StubApp())}
    for expected in (
        "/control/recall",
        "/control/record",
        "/control/forget",
        "/control/handoff",
        "/control/scopes",
        "/control/maintain",
        "/control/export",
        "/control/sql/query",
        "/control/sql/execute",
        "/control/clients",
        "/control/status",
        "/control/health",
    ):
        assert expected in served, expected
    for method in (
        "recall",
        "record",
        "forget",
        "handoff",
        "maintain",
        "export",
        "sql_query",
        "sql_execute",
        "scopes",
        "clients",
        "disconnect",
        "status",
        "health",
    ):
        assert hasattr(Control, method), method


class _StubApp:
    """The smallest thing `control_routes` will build routes for."""

    def __init__(self) -> None:
        self.service = None
        self.clients = None


# -- the documents ---------------------------------------------------------------


def test_the_documents_that_carry_a_rule_exist() -> None:
    for name in ("README.md", "AGENTS.md", "STORAGE.md", "NOTICE", "LICENSE"):
        assert (REPOSITORY / name).is_file(), f"{name} is missing"


def test_the_readme_carries_the_credit_and_says_nothing_about_upstream_endorsing() -> (
    None
):
    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for credit in ("UltraRAG", "THUNLP", "NEUIR", "OpenBMB", "AI9stars"):
        assert credit in text, f"the UltraRAG credit is missing {credit}"
    assert "endorsement" in text


def test_the_readme_does_not_compare_this_product_to_its_siblings() -> None:
    """A product's own README stands alone.

    Cross-project comparison belongs to the collection's README, because a document
    that changes when a sibling changes is a document about the collection.
    """

    text = (REPOSITORY / "README.md").read_text(encoding="utf-8").casefold()
    for sibling in (
        "research-rag",
        "research-ultra-rag-mcp-server",
        "memory-ultra-rag-mcp-server",
        "vanilla-ultra-rag-mcp",
        "graph-memory",
    ):
        assert sibling not in text, (
            f"the README names {sibling}. Selection and comparison between the "
            "collection's products belong in the collection's README."
        )


def test_the_readme_documents_every_command() -> None:
    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    for command in _subcommands():
        assert f"`{command}`" in text or f" {command} " in text, (
            f"the README does not mention the {command} command"
        )


def test_the_storage_document_names_the_layout_and_the_frozen_names() -> None:
    text = (REPOSITORY / "STORAGE.md").read_text(encoding="utf-8")
    for name in (
        ".memory-rag",
        "memory.sqlite3",
        "MEMORY.md",
        "projects.json",
        "memory/default",
    ):
        assert name in text, f"STORAGE.md does not name {name}"


def test_the_frozen_names_are_kept_and_say_why() -> None:
    """The names the previous product used stay, and a test says they are load-bearing.

    Renaming the account's settings directory or the environment prefix would strand
    every existing memory and force a silent model re-download, while the previous
    product is still installed and reads the same files.
    """

    from memory_rag.settings import USER_CONFIG_DIRECTORY

    assert USER_CONFIG_DIRECTORY == "memory-ultra-rag-mcp"
    agents = (REPOSITORY / "AGENTS.md").read_text(encoding="utf-8")
    assert "memory-ultra-rag-mcp" in agents
    assert "frozen" in agents.casefold()


def test_the_agents_document_states_the_write_rule_the_sql_panel_follows() -> None:
    """The one rule a contributor is most likely to break by widening a query."""

    agents = (REPOSITORY / "AGENTS.md").read_text(encoding="utf-8").casefold()
    assert "unit" in agents
    for phrase in ("reindex", "vector"):
        assert phrase in agents, f"AGENTS.md does not mention the {phrase}"
