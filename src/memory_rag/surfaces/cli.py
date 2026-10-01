"""The command centre: one command, and every operation on this installation.

The command line is the third front end to the same process the browser and the agent
use. It does not open a memory of its own when the app is up: it asks the app over the
control API, so a terminal answer and a workspace answer are the same object. When no
app is up it builds the service in process rather than refusing, because a read of
local state must not refuse itself because no daemon is running.

The help is a menu of groups and a set of subject pages rather than argparse's own flat
list, because the order a person works in is the useful part and argparse cannot carry
it. A test fails when a command exists and the menu does not name it, so the two cannot
drift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..app import App
from ..config import AccountConfig, ConfigurationError, resolve_account
from ..control import Control, ControlError, connect
from ..launcher import (
    DEFAULT_UI_PORT,
    ensure_launcher,
    ensure_link,
    log_path,
    running_url,
    start_app,
    state_root,
    stop_app,
)
from ..models import ModelError, describe_environment
from ..registry import RegistryError, register
from ..registry import load as load_projects
from ..runtime import build_service
from ..service import MemoryService
from ..sql import SqlRefusal

CLI_NAME = "memory-rag"
DEFAULT_RESULT_LIMIT = 10
MAX_RESULT_LIMIT = 50
#: The verb the command line uses to settle a memory, named the way the reader thinks
#: about it rather than the way the index calls it.
REINDEX_LIMIT_SECONDS = 1.0

HELP_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "First time",
        [
            ("init", "Record a project and create its local memory directory."),
            ("projects", "List every project this installation serves."),
        ],
    ),
    (
        "What is remembered",
        [
            ("scopes", "The account's memory and every project's, with their sizes."),
            ("recall", "What is remembered that matches some words."),
            ("record", "Record one statement in a memory."),
            ("forget", "Remove the one statement a recall returned exactly."),
            ("handoff", "File this session's handoff, replacing the last one."),
        ],
    ),
    (
        "Editing a record",
        [
            ("sql", "Run SQL against a memory; --execute writes statements."),
            ("reindex", "Settle a memory: re-embed what is missing, re-render it."),
            ("export", "Write the standing document for a memory."),
        ],
    ),
    (
        "Running the app",
        [
            ("start", "Bring the app up and print where it is."),
            ("ui", "Open the workspace in a browser against the running app."),
            ("serve", "Run the app in this process; the launcher calls this."),
            ("stop", "Stop the app and what it started."),
            ("status", "Whether the app is up, and what it holds."),
        ],
    ),
    (
        "Agents and settings",
        [
            ("clients", "The MCP clients attached to the app."),
            ("disconnect", "End one attached client's session."),
            ("mcp", "Serve an agent over stdio, proxying to the app."),
            ("mcp-entry", "Print the client entry a project makes for itself."),
            ("config", "Every setting in force, and the layer that supplied it."),
            ("doctor", "Diagnose this installation without writing to it."),
            ("help", "This menu, or a page per subject."),
        ],
    ),
]

HELP_TOPICS: dict[str, str] = {
    "scopes": """\
Two kinds of memory, and the difference is who can read them.

  global  the account's own memory, in the storage root. Every project on this
          machine reads it, so a rule about how the user wants to be spoken to
          belongs here rather than in any one repository.
  local   one project's memory, inside that project's repository under
          .memory-rag. It travels with the project and no other project can
          reach it.

A project's memory is named by the project's recorded name, so `memory-rag
--project research-rag recall ...` and the SQL panel's scope selector read the same
memory. A project is created by `memory-rag init` before it holds anything.
""",
    "sql": """\
`memory-rag sql` runs SQL against one memory's record, and the two paths differ.

A read is unrestricted. Any SELECT runs, and the answer is the rows it returned, so
you can ask the record anything you would ask a database.

A write is bounded to the statements themselves. INSERT, UPDATE and DELETE against
the `unit` table are accepted, and nothing else is: DDL, PRAGMA, ATTACH, and writes
to `meta` or `vector` are refused by name. An accepted write rebuilds the standing
document, drops the vectors of the statements it touched, and embeds them again, so
a hand edit cannot leave a memory whose words and whose meaning disagree.

`memory.sqlite3` is the record, not a cache. Deleting it costs every statement in it.
""",
    "agents": """\
An agent gets one tool per kind, and they are the whole of what an agent may do to a memory:
record_memory_<kind> and recall_memory_<kind> for each of the ten kinds, plus a
recall_memory across all of them and a forget_memory. Everything else
belongs to the command center.

A client that speaks streamable HTTP reaches the app at `<app>/mcp`. A client that
speaks only stdio runs `memory-rag --project <name> mcp`, which starts the app if it
is not up and proxies stdio to that endpoint. Both name themselves, so
`memory-rag clients` lists them by the name they declared, and `disconnect` ends
one.

The client entry names a project, never a directory: a configuration is written once
and copied between machines, and an absolute path in it is true on exactly one of
them.
""",
    "settings": """\
Every tunable is declared in code and shipped in `default.toml`, so a key the code
knows and a key a file may set cannot drift apart. An undeclared key is an error in
every layer rather than a silent default.

Layers win per key, lowest first:

  default.toml  <  the account's config.toml  <  the project's own config.toml
               <  --config PATH  <  MEMORY_ULTRARAG_*  <  --set key=value

The account's directory and the environment prefix keep the names the previous
product used, so a rename would strand every existing memory and force a silent
model re-download. `memory-rag config` prints the merged result and names the layer
each value came from.
""",
}


@dataclass
class CommandResult:
    """What a command answered: prose to print, or a payload to print as JSON."""

    text: str = ""
    payload: Any = None
    exit_code: int = 0


class Operations:
    """The one interface the command dispatch uses, local or over the control API.

    Both implementations answer the same coroutines with the same payloads, so the
    dispatch cannot tell them apart, and a capability cannot exist in one surface and
    not the other.
    """

    async def status(self) -> dict[str, Any]:
        raise NotImplementedError

    async def health(self) -> dict[str, Any]:
        raise NotImplementedError

    async def scopes(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def recall(self, query: str, kind: str | None, limit: int) -> dict[str, Any]:
        raise NotImplementedError

    async def record(
        self, content: str, kind: str | None, scope: str
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def forget(self, text: str, scope: str | None) -> dict[str, Any]:
        raise NotImplementedError

    async def handoff(self, content: str) -> dict[str, Any]:
        raise NotImplementedError

    async def maintain(self, scope: str) -> dict[str, Any]:
        raise NotImplementedError

    async def export(self, scope: str) -> dict[str, Any]:
        raise NotImplementedError

    async def sql_query(self, scope: str, statement: str) -> dict[str, Any]:
        raise NotImplementedError

    async def sql_execute(self, scope: str, statement: str) -> dict[str, Any]:
        raise NotImplementedError


class Local(Operations):
    """The operations answered in process, for a read of local state."""

    def __init__(self, service: MemoryService) -> None:
        self.service = service

    async def status(self) -> dict[str, Any]:
        return {**self.service.describe(), "app": {"running": False, "url": None}}

    async def health(self) -> dict[str, Any]:
        return {"ready": False, "note": "No app is running; this answered in process."}

    async def scopes(self) -> list[dict[str, Any]]:
        return [ref.as_dict() for ref in self.service.scopes()]

    async def recall(self, query: str, kind: str | None, limit: int) -> dict[str, Any]:
        return await self.service.recall(query, kind=kind, limit=limit)

    async def record(
        self, content: str, kind: str | None, scope: str
    ) -> dict[str, Any]:
        return await self.service.record(content, kind=kind, scope=scope)

    async def forget(self, text: str, scope: str | None) -> dict[str, Any]:
        return await self.service.forget(text, scope=scope)

    async def handoff(self, content: str) -> dict[str, Any]:
        return await self.service.handoff(content)

    async def maintain(self, scope: str) -> dict[str, Any]:
        return await self.service.maintain(scope)

    async def export(self, scope: str) -> dict[str, Any]:
        return await self.service.export(scope)

    async def sql_query(self, scope: str, statement: str) -> dict[str, Any]:
        return await self.service.sql_query(scope, statement)

    async def sql_execute(self, scope: str, statement: str) -> dict[str, Any]:
        return await self.service.sql_execute(scope, statement)


class Remote(Operations):
    """The same operations, asked of the running app."""

    def __init__(self, control: Control) -> None:
        self.control = control

    async def status(self) -> dict[str, Any]:
        return self.control.status()

    async def health(self) -> dict[str, Any]:
        return self.control.health()

    async def scopes(self) -> list[dict[str, Any]]:
        return self.control.scopes()

    async def recall(self, query: str, kind: str | None, limit: int) -> dict[str, Any]:
        return self.control.recall(query, kind=kind, limit=limit)

    async def record(
        self, content: str, kind: str | None, scope: str
    ) -> dict[str, Any]:
        return self.control.record(content, kind=kind, scope=scope)

    async def forget(self, text: str, scope: str | None) -> dict[str, Any]:
        return self.control.forget(text, scope=scope)

    async def handoff(self, content: str) -> dict[str, Any]:
        return self.control.handoff(content)

    async def maintain(self, scope: str) -> dict[str, Any]:
        return self.control.maintain(scope)

    async def export(self, scope: str) -> dict[str, Any]:
        return self.control.export(scope)

    async def sql_query(self, scope: str, statement: str) -> dict[str, Any]:
        return self.control.sql_query(scope, statement)

    async def sql_execute(self, scope: str, statement: str) -> dict[str, Any]:
        return self.control.sql_execute(scope, statement)


def _service(account: AccountConfig, arguments: argparse.Namespace) -> MemoryService:
    return build_service(
        account.storage_root,
        project_root=arguments.project_root,
        project_name=arguments.project,
        config_path=arguments.config,
        overrides=tuple(arguments.set_overrides or ()),
        account=account,
    )


def _operations(
    account: AccountConfig, arguments: argparse.Namespace
) -> tuple[Operations, MemoryService | None]:
    """Return the running app if there is one, and a service only if there is not.

    The service is built lazily because building it opens the local models, and a
    command that only prints the app's state should not pay for that.
    """

    handle = connect(account.storage_root)
    if handle is not None:
        return Remote(handle), None
    return Local(_service(account, arguments)), None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_NAME,
        description=(
            "A command center for an account's global memory and every project's "
            "local memory, with a browser workspace, an agent surface, and this "
            "command line on one loopback port."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"{CLI_NAME} {__version__}"
    )
    parser.add_argument(
        "--storage-root",
        default=None,
        help=(
            "Where the account's global memory lives, as a normal directory. "
            "Defaults to $MEMORY_ULTRARAG_STORAGE_ROOT, then "
            "$ULTRARAG_UI_STORAGE_ROOT, then this account's home data directory."
        ),
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help=(
            "The repository whose local memory this command acts on. It must be a "
            "project this installation has recorded; run 'init' for it first."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help=(
            "The recorded name of the project this command acts on, instead of "
            "repeating its path. A client configuration carries a name because a "
            "path is true on one machine only."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help=(
            "A config.toml whose settings apply to this run. It is a layer: it names "
            "only what it changes, and it loses to the environment and to --set."
        ),
    )
    parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        dest="set_overrides",
        help=(
            "Override one setting for this run, e.g. --set "
            "retrieval.recency_bonus=0. Repeatable, and the strongest layer."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the answer as JSON rather than as prose.",
    )

    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    initialise = commands.add_parser(
        "init", help="Record a project and create its local memory directory."
    )
    initialise.add_argument(
        "--name",
        required=True,
        help=(
            "The name this project is addressed by, in an agent's client entry and "
            "in --project. It travels with the configuration; the path does not."
        ),
    )
    initialise.add_argument(
        "--no-link",
        action="store_true",
        help=(
            "Do not write the double-clickable name into the project root. The file "
            "is machine-local and holds nothing of yours."
        ),
    )

    commands.add_parser("projects", help="List every project this installation serves.")

    commands.add_parser(
        "scopes",
        help="The account's memory and every project's, with their sizes.",
    )

    recall = commands.add_parser(
        "recall", help="What is remembered that matches some words."
    )
    recall.add_argument("query", help="A few words, not a sentence.")
    recall.add_argument(
        "--kind",
        default=None,
        help="Return only the statements filed under this kind, one word in capitals.",
    )
    recall.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RESULT_LIMIT,
        help=f"How many statements to return, at most {MAX_RESULT_LIMIT}.",
    )

    record = commands.add_parser("record", help="Record one statement in a memory.")
    record.add_argument(
        "content", help="The statement, in the user's words where it is theirs."
    )
    record.add_argument(
        "--kind",
        default=None,
        help=(
            "The kind to file it under. RULE, PLAN, PREFERENCE, CORRECTION, or any "
            "other word in capitals; the default is ITEM."
        ),
    )
    record.add_argument(
        "--scope",
        default="local",
        help="'local' is the active project's memory; 'global' is the account's.",
    )

    forget = commands.add_parser(
        "forget", help="Remove the one statement a recall returned exactly."
    )
    forget.add_argument("text", help="The exact text of the statement to remove.")
    forget.add_argument(
        "--scope",
        default=None,
        help="Remove it from this memory only; without it, both are searched.",
    )

    handoff = commands.add_parser(
        "handoff", help="File this session's handoff, replacing the last one."
    )
    handoff.add_argument(
        "content", help="What is done, what is in flight, what is next."
    )

    sql = commands.add_parser(
        "sql", help="Run SQL against a memory; --execute writes statements."
    )
    sql.add_argument(
        "--scope",
        required=True,
        help="The memory to run against: global or a project's name.",
    )
    sql.add_argument("statement", help="One SQL statement.")
    sql.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run it as a write. It may only touch the unit table, and the memory is "
            "reindexed afterwards."
        ),
    )

    reindex = commands.add_parser(
        "reindex", help="Settle a memory: re-embed what is missing, re-render it."
    )
    reindex.add_argument(
        "--scope",
        default="local",
        help="The memory to settle; 'local' is the active project.",
    )

    export = commands.add_parser(
        "export", help="Write the standing document for a memory."
    )
    export.add_argument(
        "--scope",
        default="local",
        help="The memory to render; 'local' is the active project.",
    )

    commands.add_parser("status", help="Whether the app is up, and what it holds.")

    start = commands.add_parser("start", help="Bring the app up and print where it is.")
    start.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"The port to ask for first; the app walks forward from {DEFAULT_UI_PORT}.",
    )
    start.add_argument(
        "--open", action="store_true", help="Open the workspace in a browser."
    )

    ui = commands.add_parser(
        "ui", help="Open the workspace in a browser against the running app."
    )
    ui.add_argument("--port", type=int, default=None, help="The port to ask for first.")

    serve = commands.add_parser(
        "serve", help="Run the app in this process; the launcher calls this."
    )
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_UI_PORT,
        help=f"The loopback port to serve on. Default {DEFAULT_UI_PORT}.",
    )
    serve.add_argument(
        "--open", action="store_true", help="Open the workspace once it is serving."
    )

    stop = commands.add_parser("stop", help="Stop the app and what it started.")
    stop.add_argument(
        "--servers",
        action="store_true",
        help="Also report the stdio servers this machine still runs over these memories.",
    )

    clients = commands.add_parser(
        "clients", help="The MCP clients attached to the app."
    )
    clients.set_defaults(limit=0)

    disconnect = commands.add_parser(
        "disconnect", help="End one attached client's session."
    )
    disconnect.add_argument("session_id", help="The session id 'clients' printed.")
    disconnect.add_argument(
        "--reason",
        default="disconnected from the command line",
        help="Why it is ended.",
    )

    bridge = commands.add_parser(
        "mcp", help="Serve an agent over stdio, proxying to the app."
    )
    bridge.add_argument(
        "--project-name",
        default=None,
        help="The project this client works on. It must be recorded on this machine.",
    )

    entry = commands.add_parser(
        "mcp-entry", help="Print the client entry a project makes for itself."
    )
    entry.add_argument(
        "--project",
        dest="entry_project",
        default=None,
        help="The project to print an entry for. Defaults to the active one.",
    )
    entry.add_argument(
        "--check",
        metavar="PATH",
        default=None,
        help=(
            "Check an entry that is already in place against this installation, and "
            "print the one this machine would use. Nothing is written."
        ),
    )

    commands.add_parser(
        "config", help="Every setting in force, and the layer that supplied it."
    )
    commands.add_parser(
        "doctor", help="Diagnose this installation without writing to it."
    )
    help_parser = commands.add_parser("help", help="This menu, or a page per subject.")
    help_parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help=f"One of: {', '.join(sorted(HELP_TOPICS))}, or a command name.",
    )
    return parser


def _help_menu() -> str:
    lines = [f"{CLI_NAME} — every operation on this installation's memories", ""]
    for title, entries in HELP_GROUPS:
        lines.append(f"{title}:")
        for name, description in entries:
            lines.append(f"  {name:<12} {description}")
        lines.append("")
    lines.append(
        "Every subject page: "
        + ", ".join(f"help {name}" for name in sorted(HELP_TOPICS))
    )
    lines.append(
        "One command's own usage: "
        + ", ".join(f"help {name}" for name, _ in _all_commands())
    )
    return "\n".join(lines)


def _all_commands() -> list[tuple[str, str]]:
    return [
        (name, description)
        for _, entries in HELP_GROUPS
        for name, description in entries
    ]


def _command_usage(parser: argparse.ArgumentParser, topic: str) -> str:
    """Return one command's own help, so `help x` and `x --help` are one document."""

    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        subparser = action.choices.get(topic)
        if subparser is not None:
            return subparser.format_help()
    return f"{CLI_NAME} has no command called {topic!r}. Run '{CLI_NAME} help'."


def _init(arguments: argparse.Namespace, account: AccountConfig) -> CommandResult:
    """Record a project, create its memory directory, and write the launcher."""

    if not arguments.project_root:
        raise ConfigurationError(
            "init needs --project-root: a project's local memory lives inside its own "
            "repository, and the command center is told where that repository is."
        )
    root = Path(arguments.project_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(f"project root is not a directory: {root}")
    name = arguments.name.strip()
    if not name:
        raise ConfigurationError("a project's recorded name must not be empty")

    state = root / ".memory-rag"
    state.mkdir(parents=True, exist_ok=True)
    descriptor = state / "project.json"
    project_id = _project_id(descriptor, root)
    if descriptor.is_file():
        existing = _existing_project_id(descriptor)
        if existing:
            project_id = existing
    _write_descriptor(descriptor, project_id, name, root)
    entry = register(project_id, name, root)
    ensure_launcher(account.storage_root)
    link = "not written"
    if not arguments.no_link:
        link = ensure_link(root, account.storage_root)
    return CommandResult(
        text=(
            f"Recorded {name} at {root}.\n"
            f"  its memory: {state}\n"
            f"  double-click: {link}\n"
            f"  the app: memory-rag start, then memory-rag ui"
        ),
        payload={
            "project": entry.as_record(),
            "state_directory": str(state),
            "link": link,
        },
    )


def _project_id(descriptor: Path, root: Path) -> str:
    """Return a stable identity for a project, derived from where it lives.

    Derived rather than random so a project that is initialised twice on two machines
    gets the same id, which is what lets one client entry name it on both.
    """

    import hashlib

    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]


def _existing_project_id(descriptor: Path) -> str | None:
    try:
        document = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = document.get("project_id")
    return value if isinstance(value, str) and value else None


def _write_descriptor(descriptor: Path, project_id: str, name: str, root: Path) -> None:
    """Write the project's identity, additively.

    The frozen memory server reads nothing from this file, so adding one is invisible
    to it, and a project that already has one keeps the id it was recorded under.
    """

    from datetime import UTC, datetime

    document = {
        "schema_version": 1,
        "project_id": project_id,
        "project_name": name,
        "project_root": str(root),
        "written_by": "memory-rag",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    temporary = descriptor.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(descriptor)


def _mcp_entry(arguments: argparse.Namespace, account: AccountConfig) -> CommandResult:
    """Print the client entry a project makes for itself.

    A configuration is written once and copied between machines, so it has to name a
    project rather than a path, and the executable it names has to be this
    installation's own. Both are facts of the machine, which is why the entry is
    printed here rather than copied out of this document.
    """

    import shutil

    from .. import registry as registry_module

    wanted = arguments.entry_project
    known = registry_module.load()
    if wanted:
        project = registry_module.resolve(wanted)
    elif known:
        project = known[0]
    else:
        raise RegistryError(
            "no project is recorded yet, so there is no entry to print. Run "
            "'memory-rag init --project-root <path> --name <name>' first."
        )
    executable = shutil.which(CLI_NAME) or str(Path(sys.executable).parent / CLI_NAME)
    entry = {
        "mcpServers": {
            "memory-rag": {
                "command": executable,
                "args": [
                    "--storage-root",
                    str(account.storage_root),
                    "--project",
                    project.project_name,
                    "mcp",
                ],
                "timeout": 3600,
            }
        }
    }
    text = json.dumps(entry, indent=2)
    if arguments.check:
        text = _check_entry(Path(arguments.check), entry, project) + "\n\n" + text
    return CommandResult(text=text, payload=entry)


def _check_entry(path: Path, entry: dict[str, Any], project: Any) -> str:
    """Compare an entry already in place with the one this machine would use."""

    expected = entry["mcpServers"]["memory-rag"]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"{path} could not be read as JSON ({error}). Nothing was written."
    found = (document.get("mcpServers") or {}).get("memory-rag")
    if not isinstance(found, dict):
        return f"{path} has no 'memory-rag' entry under 'mcpServers'."
    differences = []
    if found.get("command") != expected["command"]:
        differences.append(
            f"command: {found.get('command')!r} here, {expected['command']!r} on this "
            "machine"
        )
    if found.get("args") != expected["args"]:
        differences.append(
            f"args: {found.get('args')!r} here, {expected['args']!r} on this machine"
        )
    if not differences:
        return f"{path} matches this installation."
    return "\n".join(
        [f"{path} differs from this installation:", *(f"  {d}" for d in differences)]
    )


def _projects(account: AccountConfig) -> CommandResult:
    projects = load_projects()
    if not projects:
        return CommandResult(
            text=(
                "No project is recorded yet. The account's global memory is served "
                "regardless, and a project is added with:\n"
                "  memory-rag init --project-root /path/to/repo --name its-name"
            ),
            payload={"projects": []},
        )
    lines = []
    payload = []
    for project in projects:
        marker = "initialised" if project.initialised() else "no descriptor"
        lines.append(f"  {project.project_name:<20} {project.project_root}  ({marker})")
        payload.append(
            {
                "project_id": project.project_id,
                "project_name": project.project_name,
                "project_root": str(project.project_root),
                "initialised": project.initialised(),
            }
        )

    active = None
    if projects:
        active = projects[0].project_name
    return CommandResult(
        text=(
            f"{len(projects)} project(s) recorded"
            + (f"; the active one is {active}" if active else "")
            + ":\n"
            + "\n".join(lines)
        ),
        payload={"projects": payload, "active_project": active},
    )


def _doctor(account: AccountConfig) -> CommandResult:
    """Report what this installation can do, and what stands in the way.

    Every check reads: no memory is written, no model is fetched, and no process is
    started, so the report is safe to run on a machine that cannot serve yet. A check
    that did not run reads as unknown rather than as healthy.
    """

    from ..index import fts5_available

    checks: list[dict[str, Any]] = []

    def add(name: str, state: str, detail: str, remedy: str = "") -> None:
        entry: dict[str, Any] = {"check": name, "state": state, "detail": detail}
        if remedy:
            entry["remedy"] = remedy
        checks.append(entry)

    storage = account.storage_root
    add(
        "storage root",
        "ok" if storage.is_dir() else "warn",
        str(storage),
        "" if storage.is_dir() else "It is made when a memory is first recorded.",
    )
    add(
        "global memory",
        "ok" if account.global_directory.is_dir() else "warn",
        str(account.global_directory),
        ""
        if account.global_directory.is_dir()
        else "It is made when a statement is first recorded.",
    )
    add(
        "FTS5",
        "ok" if fts5_available() else "blocked",
        "SQLite has the full-text module this memory needs."
        if fts5_available()
        else "This SQLite has no FTS5, and a read would otherwise pass over every "
        "statement a memory holds.",
        ""
        if fts5_available()
        else "Install a Python built against a SQLite with FTS5.",
    )
    launcher = ensure_launcher(account.storage_root)
    add("launcher", "ok", str(launcher))
    from ..launcher import running_pid

    pid = running_pid(storage)
    add(
        "app",
        "ok" if pid is not None else "warn",
        f"running with pid {pid}" if pid else "not running",
        "" if pid else "memory-rag start",
    )
    projects = load_projects()
    missing = [
        project.project_name for project in projects if not project.initialised()
    ]
    add(
        "projects",
        "ok" if not missing else "warn",
        f"{len(projects)} recorded"
        if not missing
        else f"{len(missing)} recorded project(s) have no descriptor: "
        + ", ".join(missing),
        ""
        if not missing
        else "Run 'memory-rag init' for each, so the app can serve it.",
    )
    add("models", "ok", json.dumps(describe_environment(), default=str))
    add("log", "ok", str(log_path(storage)))

    blocked = [entry for entry in checks if entry["state"] == "blocked"]
    warn = [entry for entry in checks if entry["state"] == "warn"]
    lines = [f"{CLI_NAME} doctor — {len(blocked)} blocked, {len(warn)} to note", ""]
    for entry in checks:
        lines.append(f"  [{entry['state']:<7}] {entry['check']}: {entry['detail']}")
        if "remedy" in entry:
            lines.append(f"            fix: {entry['remedy']}")
    return CommandResult(
        text="\n".join(lines),
        payload={"checks": checks, "blocked_by": blocked, "degraded": warn},
        exit_code=1 if blocked else 0,
    )


def _config(account: AccountConfig) -> CommandResult:
    from config_ultra_rag_mcp import describe_settings

    from ..settings import SETTINGS

    values = {
        setting.field: getattr(account.settings, setting.field)
        for setting in SETTINGS
        if account.settings is not None and hasattr(account.settings, setting.field)
    }
    provenance = account.settings_provenance or {}
    return CommandResult(
        text=describe_settings(SETTINGS, values, provenance), payload=None
    )


async def _serve(
    arguments: argparse.Namespace, account: AccountConfig
) -> CommandResult:
    """Run the app in this process. The launcher calls this; a person may too.

    This is awaited rather than run with its own event loop, because the dispatch is
    already inside one: a second loop would be refused, and the app's own surfaces
    need the loop this command is standing in.
    """

    service = _service(account, arguments)
    app = App(service, port=arguments.port)
    await _serve_forever(app)
    return CommandResult(text="")


async def _serve_forever(app: App) -> None:
    await app.start()
    if app.error is not None:
        print(f"{CLI_NAME}: {app.error}", file=sys.stderr, flush=True)
        return
    print(app.url, flush=True)
    await app.wait()


def _format_scopes(scopes: list[dict[str, Any]]) -> str:
    lines = [f"{'SCOPE':<24} {'KIND':<7} {'STATEMENTS':>10}  RECORD"]
    for entry in scopes:
        lines.append(
            f"{entry['scope']:<24} {entry['kind']:<7} {entry['statements']:>10}  "
            f"{entry.get('record', '')}"
        )
    return "\n".join(lines)


def _format_units(payload: dict[str, Any]) -> str:
    units = payload.get("units") or []
    if not units:
        lines = ["Nothing matched."]
    else:
        lines = []
        for unit in units:
            scope = unit.get("scope", "?")
            kind = unit.get("kind", "ITEM")
            recalls = unit.get("recalls", 0)
            lines.append(f"[{scope} {kind} recalled {recalls}x] {unit.get('text', '')}")
    for field, label in (
        ("units_pending", "statements still waiting for a vector"),
        ("semantic_available", "the semantic side is unavailable"),
        ("superseded_removed", "files read once and removed"),
        ("collapsed_repetitions", "repetitions collapsed in this answer"),
        ("truncated", "more statements matched than were returned"),
    ):
        if field in payload:
            lines.append(f"\n({label}: {payload[field]})")
    if "hint" in payload:
        lines.append(f"\n{payload['hint']}")
    return "\n".join(lines)


def _format_rows(payload: dict[str, Any]) -> str:
    columns = [str(name) for name in payload.get("columns") or []]
    rows = payload.get("rows") or []
    if not columns:
        return "That statement returned no columns."
    lines = [" | ".join(columns), "-" * 3 * " | ".join(columns)]
    for row in rows:
        lines.append(" | ".join("" if cell is None else str(cell) for cell in row))
    lines.append(f"\n{len(rows)} row(s)")
    if payload.get("truncated"):
        lines.append("truncated at the console's limit; narrow the statement")
    if payload.get("empty_memory"):
        lines.append(
            "That memory holds no statements yet, so it has no record to read."
        )
    return "\n".join(lines)


async def _run(arguments: argparse.Namespace) -> CommandResult:
    command = arguments.command
    account = resolve_account(
        arguments.storage_root,
        config_path=arguments.config,
        overrides=tuple(arguments.set_overrides or ()),
    )

    if command == "help":
        topic = arguments.topic
        if topic is None:
            return CommandResult(text=_help_menu())
        if topic in HELP_TOPICS:
            return CommandResult(text=HELP_TOPICS[topic])
        return CommandResult(text=_command_usage(_parser(), topic))
    if command == "init":
        return _init(arguments, account)
    if command == "projects":
        return _projects(account)
    if command == "mcp-entry":
        return _mcp_entry(arguments, account)
    if command == "config":
        return _config(account)
    if command == "doctor":
        return _doctor(account)
    if command == "start":
        return CommandResult(
            text=_start_text(
                start_app(
                    account.storage_root,
                    port=arguments.port,
                    open_browser=arguments.open,
                )
            ),
            payload=None,
        )
    if command == "ui":
        started = start_app(
            account.storage_root, port=arguments.port, open_browser=True
        )
        return CommandResult(text=_start_text(started, opened=True), payload=None)
    if command == "serve":
        return await _serve(arguments, account)
    if command == "stop":
        return _stop_text(account, arguments)

    operations, _ = _operations(account, arguments)
    try:
        return await _dispatch(command, arguments, operations, account)
    finally:
        if isinstance(operations, Local):
            operations.service.close()


def _start_text(started: dict[str, Any], *, opened: bool = False) -> str:
    lines = []
    if started.get("url"):
        lines.append(f"The app is at {started['url']}")
        lines.append(f"  its agent endpoint: {started['url']}/mcp")
        lines.append(f"  its log: {started.get('log', '')}")
    else:
        lines.append(
            f"The app did not start: {started.get('stderr') or started.get('note')}"
        )
    if not started.get("started", False) and started.get("note"):
        lines.append(started["note"])
    if opened and started.get("url"):
        lines.append("The workspace was asked to open in your browser.")
    return "\n".join(lines)


def _stop_text(account: AccountConfig, arguments: argparse.Namespace) -> CommandResult:
    stopped = stop_app(account.storage_root)
    lines = [
        "The app is stopped."
        if stopped.get("stopped")
        else f"The app was not stopped: {stopped.get('stderr') or stopped.get('note')}"
    ]
    if arguments.servers:
        lines.append(
            "\nThe stdio servers on this machine are not stopped by this command: each "
            "belongs to the MCP client that started it. The collection's "
            "scripts/stop-servers.sh stops those by explicit pid."
        )
    return CommandResult(text="\n".join(lines), payload=stopped)


async def _dispatch(
    command: str,
    arguments: argparse.Namespace,
    operations: Operations,
    account: AccountConfig,
) -> CommandResult:
    if command == "status":
        payload = await operations.status()
        return CommandResult(text=_format_status(payload), payload=payload)
    if command == "scopes":
        scopes = await operations.scopes()
        return CommandResult(text=_format_scopes(scopes), payload={"scopes": scopes})
    if command == "recall":
        payload = await operations.recall(
            arguments.query, arguments.kind, arguments.limit
        )
        return CommandResult(text=_format_units(payload), payload=payload)
    if command == "record":
        payload = await operations.record(
            arguments.content, arguments.kind, arguments.scope
        )
        return CommandResult(
            text=f"Recorded in {payload.get('scope', arguments.scope)} as "
            f"{payload.get('kind', 'ITEM')}.",
            payload=payload,
        )
    if command == "forget":
        payload = await operations.forget(arguments.text, arguments.scope)
        return CommandResult(
            text=f"Forgot one statement from {payload.get('scope', 'a memory')}.",
            payload=payload,
        )
    if command == "handoff":
        payload = await operations.handoff(arguments.content)
        return CommandResult(
            text="Recorded this session's handoff"
            + (
                f", replacing {payload['replaced']} previous one(s)."
                if payload.get("replaced")
                else "."
            ),
            payload=payload,
        )
    if command == "reindex":
        payload = await operations.maintain(arguments.scope)
        return CommandResult(text=_format_maintain(payload), payload=payload)
    if command == "export":
        payload = await operations.export(arguments.scope)
        return CommandResult(
            text=f"Wrote the standing document for {payload.get('scope')}: "
            f"{payload.get('rendered')}",
            payload=payload,
        )
    if command == "sql":
        if arguments.execute:
            payload = await operations.sql_execute(arguments.scope, arguments.statement)
            return CommandResult(text=_format_execute(payload), payload=payload)
        payload = await operations.sql_query(arguments.scope, arguments.statement)
        return CommandResult(text=_format_rows(payload), payload=payload)
    if command == "clients":
        return await _clients(operations, account)
    if command == "disconnect":
        return await _disconnect(operations, account, arguments)
    if command == "mcp":
        # Reached only when a caller runs the dispatch directly rather than through
        # main, which hands this to the bridge before the loop exists.
        from ..bridge import run

        return CommandResult(text="", exit_code=run(arguments.project_name))
    raise ConfigurationError(f"{command!r} is not a command this build has.")


def _format_status(payload: dict[str, Any]) -> str:
    """Say where the app is and what it holds.

    The app reports its own address under ``ui_url`` and the local path carries no
    address at all, so both are read here rather than one of them being assumed.
    """

    app = payload.get("app") or {}
    url = app.get("url") or app.get("ui_url")
    lines = []
    if url:
        pid = app.get("pid")
        lines.append(f"The app is at {url}" + (f" (pid {pid})" if pid else ""))
    else:
        lines.append("No app is running; this answered in process.")
    lines.append(f"Storage root: {payload.get('storage_root')}")
    lines.append(f"Projects: {payload.get('project_count', 0)}")
    if "scopes" in payload:
        lines.append("")
        lines.append(_format_scopes(payload["scopes"]))
    return "\n".join(lines)


def _format_maintain(payload: dict[str, Any]) -> str:
    return (
        f"Settled {payload.get('scope', 'the memory')}: "
        f"{payload.get('units_queued', 0)} statement(s) queued for a vector, "
        f"{payload.get('adopted', 0)} adopted from a document, "
        f"{len(payload.get('superseded_removed') or [])} superseded file(s) removed."
    )


def _format_execute(payload: dict[str, Any]) -> str:
    return (
        f"Applied to {payload.get('scope')}: {payload.get('rows_affected', 0)} row(s) "
        f"changed, {payload.get('statements_changed', 0)} statement(s) re-embedded, "
        f"standing document rewritten at {payload.get('rendered')}."
    )


async def _clients(operations: Operations, account: AccountConfig) -> CommandResult:
    if isinstance(operations, Remote):
        rows = operations.control.clients()
    else:
        from ..app import ClientRegistry

        rows = ClientRegistry().report()
    if not rows:
        return CommandResult(
            text=(
                "No client is attached. Start the app first, then connect an agent to "
                "<app>/mcp or run 'memory-rag mcp' for a stdio client."
            ),
            payload={"clients": []},
        )
    lines = [f"{'SESSION':<40} {'NAME':<20} {'STATE':<10} REQUESTS"]
    payload = []
    for row in rows:
        state = (
            "attached"
            if row.get("attached")
            else ("dropped" if row.get("dropped") else "idle")
        )
        lines.append(
            f"{row['session_id']:<40} {row['name']:<20} {state:<10} "
            f"{row.get('requests', 0)}"
        )
        payload.append(row)
    return CommandResult(text="\n".join(lines), payload={"clients": payload})


async def _disconnect(
    operations: Operations, account: AccountConfig, arguments: argparse.Namespace
) -> CommandResult:
    if not isinstance(operations, Remote):
        raise ControlError(
            "No app is running, so there is no client to disconnect. Run "
            "'memory-rag start' first."
        )
    payload = operations.control.disconnect(arguments.session_id, arguments.reason)
    return CommandResult(
        text=f"Detached {arguments.session_id}: {payload.get('detached_reason', '')}",
        payload=payload,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if not arguments.command:
        print(_help_menu())
        raise SystemExit(0)
    if arguments.command == "mcp":
        # The bridge owns stdio and runs its own event loop, so it is handed the
        # process rather than a coroutine: a loop inside the loop the dispatch already
        # stands in is refused, and the refusal would reach the client as a closed
        # connection with nothing in its log to explain it.
        from ..bridge import run

        raise SystemExit(
            run(arguments.project_name, storage_root=arguments.storage_root)
        )
    try:
        result = asyncio.run(_run(arguments))
    except (
        ConfigurationError,
        ControlError,
        ModelError,
        RegistryError,
        SqlRefusal,
        OSError,
        ValueError,
    ) as error:
        # A refusal is an answer, not a crash. Every service refusal is a ModelError,
        # and one of them reaching the terminal as a traceback tells an operator that
        # the command line is broken rather than that their statement was refused and
        # what to do about it.
        print(f"{CLI_NAME}: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    if result.payload is not None and getattr(arguments, "json", False):
        print(json.dumps(result.payload, ensure_ascii=False, indent=2, default=str))
    elif result.text:
        print(result.text)
    if result.exit_code:
        raise SystemExit(result.exit_code)


__all__ = [
    "CLI_NAME",
    "HELP_GROUPS",
    "HELP_TOPICS",
    "CommandResult",
    "Local",
    "Operations",
    "Remote",
    "main",
    "running_url",
    "state_root",
]
