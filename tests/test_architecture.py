"""The boundaries this product keeps, checked by reading the source rather than running it.

A boundary that is only a convention is a boundary the next change crosses, and these
fail when it does. The checks parse imports with `ast` and never import the module they
read, so a test cannot be satisfied by an import that succeeds and an import that is
never reached.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parent.parent / "src" / "memory_rag"
PACKAGE = SOURCE.parent

#: The web and agent stack. The engine must not import any of it: a memory is a record
#: on disk, and a module that can reach a socket is a module whose behaviour depends on
#: whether a server happens to be running.
WEB_MODULES = {
    "fastmcp",
    "mcp",
    "pydantic",
    "starlette",
    "uvicorn",
    "ui_ultra_rag_mcp",
}

#: The modules allowed to import the stack above. `surfaces/cli.py` is deliberately
#: absent so `config` and `doctor` answer on a machine where the web stack is not
#: importable, which is the whole reason the split exists. The names are paths
#: relative to the package, because a surface is a file and not a package: `app.py`
#: sits beside `surfaces/`, not inside it.
SURFACE_MODULES = {
    "app.py",
    "bridge.py",
    "control.py",
    "surfaces/mcp.py",
    "surfaces/ui.py",
}

#: The engine: everything that opens a memory. It may not import a surface, because a
#: surface is a way of presenting an answer and a record does not know how it is read.
ENGINE_MODULES = {
    "config",
    "index",
    "models",
    "read",
    "registry",
    "reindex",
    "retrieval",
    "runtime",
    "service",
    "sql",
    "settings",
    "store",
    "support",
    "vectors",
}

#: The agent surface, by name. Every kind has a recorder and a recaller, and two tools
#: sit beside them that are not kinds: a recall that answers across every kind, and a
#: forget that removes a statement whatever kind filed it. The test below compares this
#: against the server the product actually builds, so a tool cannot be added without
#: being written here, and cannot be dropped without this failing.
AGENT_TOOLS = {
    "recall_memory",
    "forget_memory",
    *(
        f"{verb}_memory_{kind}"
        for verb in ("record", "recall")
        for kind in (
            "rule",
            "guideline",
            "personality",
            "general",
            "decision",
            "handoff",
            "correction",
            "preference",
            "plan",
            "limits",
        )
    ),
}


def _module_path(module: str) -> Path:
    return SOURCE / f"{module}.py"


def _imported_names(path: Path) -> set[str]:
    """Return every module name a file imports, read without executing it."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _relative_imports(path: Path) -> set[str]:
    """Return the first segment of every relative import a file makes."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.module:
                found.add(node.module.split(".")[0])
            for alias in node.names:
                found.add(alias.name.split(".")[0])
    return found


def _engine_files() -> list[Path]:
    return [
        path
        for path in sorted(SOURCE.rglob("*.py"))
        if path.relative_to(SOURCE).as_posix() not in SURFACE_MODULES
        and path.relative_to(SOURCE).as_posix() != "__main__.py"
    ]


@pytest.mark.parametrize("path", _engine_files(), ids=lambda p: p.name)
def test_the_engine_imports_neither_the_web_stack_nor_a_surface(path: Path) -> None:
    relative = path.relative_to(SOURCE).as_posix()
    third_party = _imported_names(path)
    assert not (third_party & WEB_MODULES), (
        f"{relative} imports {sorted(third_party & WEB_MODULES)}. The engine opens a "
        "memory and must not be able to reach a socket: a record that behaves "
        "differently depending on whether a server is running is a record whose answer "
        "depends on something a reader cannot see."
    )
    assert not (_relative_imports(path) & SURFACE_MODULES), (
        f"{relative} imports a surface. A surface presents an answer; the record it "
        "presents has no opinion about how it is read, and depending on one here would "
        "put the web stack back inside the engine."
    )


def test_no_surface_under_surfaces_imports_a_sibling_surface() -> None:
    """Two front ends must not share an implementation by reaching for each other.

    The command line may use the app, the control channel, and the launcher, because
    those are the machinery a terminal drives rather than another front end. What it
    may not do is ask the agent surface or the workspace to do its work, and the two of
    those may not ask each other: a capability reached from two surfaces is a
    capability implemented in the place both of them can see, which is how the copies
    start to differ.
    """

    for path in sorted((SOURCE / "surfaces").glob("*.py")):
        if path.name == "__init__.py":
            continue
        siblings = {
            target
            for target in _relative_imports(path)
            if f"surfaces/{target}.py" in SURFACE_MODULES
        }
        assert not siblings, (
            f"surfaces/{path.name} imports {sorted(siblings)}, another surface. Each "
            "surface calls the service; none of them calls another."
        )


def test_only_the_agent_surface_declares_a_tool() -> None:
    """A tool is declared once, in one place.

    A second copy of a tool is a second place for it to be wrong, and would let an
    agent and a browser disagree about what a memory holds.
    """

    declaring = [
        path.relative_to(SOURCE).as_posix()
        for path in SOURCE.rglob("*.py")
        if "server.tool(" in path.read_text(encoding="utf-8")
    ]
    assert declaring == ["surfaces/mcp.py"], (
        f"tools are declared in {declaring}; they belong in surfaces/mcp.py alone."
    )


async def test_the_agent_surface_declares_exactly_the_memory_tools(service) -> None:
    """The surface is one tool per kind, and the test reads the server rather than the source.

    Grepping for a decorator would miss a tool declared in a loop, which is how these
    are declared, so this builds the server and asks it what it offers. A kind added to
    the registry without being written above fails here.
    """

    from memory_rag.surfaces.mcp import create_mcp

    server = create_mcp(service)
    declared = {tool.name for tool in await server.list_tools()}
    assert declared == AGENT_TOOLS, (
        f"the agent surface offers {sorted(declared - AGENT_TOOLS)} unexpectedly and "
        f"is missing {sorted(AGENT_TOOLS - declared)}. Everything else belongs to the "
        "command center."
    )


async def test_every_kind_has_a_recorder_and_a_recaller(service) -> None:
    """The registry is the surface's source, so a kind and a tool cannot come apart."""

    from memory_rag.store import KINDS
    from memory_rag.surfaces.mcp import agent_tool_names

    names = set(agent_tool_names())
    for kind in KINDS:
        assert f"record_memory_{kind.tool}" in names, kind.name
        assert f"recall_memory_{kind.tool}" in names, kind.name


def test_the_entry_point_runs_the_command_line() -> None:
    entry = SOURCE / "__main__.py"
    assert entry.is_file()
    assert "from .surfaces.cli import main" in entry.read_text(encoding="utf-8")


def test_the_console_script_names_the_surfaces_entry_point() -> None:
    manifest = (PACKAGE.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert 'memory-rag = "memory_rag.surfaces.cli:main"' in manifest
    assert "memory-rag-mcp" not in manifest, (
        "the entry point is this product's own name. The previous product keeps its "
        "entry point; two products sharing one name cannot be told apart by a client "
        "configuration, and a frozen product is still installed."
    )


def test_no_module_registers_a_tool_outside_the_agent_surface() -> None:
    """A tool-shaped registration anywhere else is a second agent surface."""

    for path in SOURCE.rglob("*.py"):
        relative = path.relative_to(SOURCE).as_posix()
        text = path.read_text(encoding="utf-8")
        for marker in ("FastMCP(", "create_proxy("):
            if marker in text:
                assert relative in {
                    "surfaces/mcp.py",
                    "bridge.py",
                }, (
                    f"{relative} builds an MCP server or proxy ({marker}). The served "
                    "tools live in surfaces/mcp.py and the stdio bridge in bridge.py, "
                    "because those are the only two things an agent talks to."
                )


def test_the_workspace_is_never_copied_in() -> None:
    """The shared workspace stays a pinned dependency, not a vendored copy.

    A copy is a second implementation of the same interface that stops receiving the
    fixes the first one gets, and this product's workspace would then be the one left
    behind.
    """

    for name in ("static", "index.html", "app.js", "app.css"):
        assert not (SOURCE / name).exists(), (
            f"{name} was found inside the package. The workspace is served by the "
            "pinned ui-ultra-rag-mcp dependency and is never copied in."
        )
