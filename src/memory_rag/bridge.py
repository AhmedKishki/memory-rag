"""`memory-rag mcp`: a stdio front end to the app that holds the account's memories.

An MCP client that speaks only stdio cannot open a socket, so this is the command that
connects one: it makes sure the app is up, then proxies stdio to the app's agent
endpoint on the app's own port. The proxy adds nothing — the tools, the
instructions, the scope resolution, and the models are the app's, so a stdio client and
a browser cannot see two different states.

The bridge names a project and never a directory. A client configuration is written
once and copied between machines, a phone, and a repository, and an absolute path in
it is true on exactly one of them; a project's recorded name is resolvable on every
machine where that project was initialised. The resolution belongs to this
installation's own record, so the bridge asks the record for the directory and refuses
to accept one itself.

A project that this machine has not initialised is answered, not refused: the
connection is established, `recall_memory` reports that the name resolves to nothing
here, and the answer carries the command that creates the project. That is a
declaration about a machine, not a refusal to speak, because an agent that gets a
connection error has nothing to report to the user and a client entry that would have
to be edited afterwards is a client entry that will not be edited.

The bridge names itself, so a disconnect is legible: the app lists clients by name, and
dropping one ends its session, which ends the pipe and therefore the client.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server import create_proxy

from . import registry
from .app import CLIENT_NAME_HEADER
from .config import AccountConfig, resolve_account
from .control import Control, ControlError, ensure_running
from .surfaces.mcp import AGENT_BRIDGE_NAME, MCP_PATH

#: The env var a client configuration sets so a reader of the app's client list can
#: tell which agent is which, rather than seeing "stdio-bridge" four times.
CLIENT_NAME_ENV = "MEMORY_ULTRARAG_CLIENT_NAME"
DEFAULT_CLIENT_NAME = "stdio-bridge"


def client_name() -> str:
    """The name this bridge reports itself under, or a stated default."""

    return os.environ.get(CLIENT_NAME_ENV) or DEFAULT_CLIENT_NAME


def build_proxy(url: str, *, name: str) -> Any:
    """Return a stdio server that forwards everything to the app at ``url``.

    The tools are proxied, not re-declared: a second copy of the four operations would
    be a second place for them to be wrong, and would give the agent a different answer
    than the workspace for the same question. The transport is built here rather than
    from a URL so the bridge can name itself, which makes it identifiable in the app's
    client list.
    """

    return create_proxy(
        StreamableHttpTransport(url, headers={CLIENT_NAME_HEADER: name}),
        name=AGENT_BRIDGE_NAME,
    )


def resolve_project_name(
    name: str | None,
    account: AccountConfig,
) -> registry.RegisteredProject | None:
    """Return the project a client entry named, or None when this machine has none.

    A name that matches nothing is not an error here. The bridge answers with a status
    that says the name resolves to nothing on this machine and what would create it,
    because an agent that cannot connect has nothing to tell the user.
    """

    if name:
        found = registry.named(name)
        if found:
            return found[0]
        return None
    projects = registry.load()
    if len(projects) == 1:
        return projects[0]
    return projects[0] if projects else None


def agent_url(control: Control) -> str:
    """Return where the app serves its agent endpoint.

    The app is asked rather than assembled from its base address, because the path its
    agent endpoint sits at is the app's own fact and a hardcoded one would become a
    second place for it to be wrong. A base address on its own is not an MCP endpoint,
    and proxying to it fails as a 405 with nothing in the client's log to explain it.
    """

    try:
        health = control.health()
    except ControlError:
        return f"{control.base_url}{MCP_PATH}"
    reported = str(health.get("mcp_url") or "").strip()
    return reported or f"{control.base_url}{MCP_PATH}"


def run(
    project_name: str | None = None,
    *,
    account: AccountConfig | None = None,
    storage_root: str | None = None,
) -> int:
    """Serve this account's memory to one stdio client, until the client stops."""

    resolved = account or resolve_account(storage_root)
    try:
        control = ensure_running(resolved.storage_root)
    except ControlError as error:
        print(f"memory-rag: {error}", flush=True)
        return 1
    proxy = build_proxy(agent_url(control), name=client_name())
    proxy.run(transport="stdio", show_banner=False)
    return 0


def main(project_name: str | None = None, *, storage_root: str | None = None) -> int:
    """The entry the command line hands this bridge."""

    return run(project_name, storage_root=storage_root)


__all__ = [
    "CLIENT_NAME_ENV",
    "DEFAULT_CLIENT_NAME",
    "agent_url",
    "build_proxy",
    "client_name",
    "main",
    "resolve_project_name",
    "run",
]
