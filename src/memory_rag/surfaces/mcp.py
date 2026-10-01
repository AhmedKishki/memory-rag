"""The agent surface: one tool per kind, and nothing else.

Each kind a statement can be filed under is its own recording tool and its own recall
tool, so what an agent is doing is named by the tool it calls rather than by a word it
passes. That buys two things. A kind that must behave differently behaves differently
— ``PERSONALITY`` and ``PREFERENCE`` take no memory argument because what is true of
the user is true of the user in every project, and ``HANDOFF`` holds one rather than a
list — and an agent cannot file a statement under a kind nothing interprets, because
there is no kind parameter to guess at.

The tools are declared here and nowhere else, and they are declared from
:data:`memory_rag.store.KINDS` rather than written out one by one, so a kind's summary
and the tool that offers it are one fact in one place. A surface that re-declared one
of them would be a second place for it to be wrong, and the workspace and the command
line would then be able to disagree with an agent about what a memory holds.

Two tools sit beside the per-kind ones and are not kinds. ``recall_memory`` answers a
question across every kind, because an agent asking what is remembered about a subject
does not know which kind filed it and should not have to guess. ``forget_memory``
removes a statement by its exact text, because a statement that is no longer true is
removed whatever kind it was filed under.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from ..instructions import SERVER_INSTRUCTIONS
from ..models import ModelError
from ..registry import RegistryError
from ..service import MemoryService
from ..store import KINDS, SCOPE_EITHER, Kind, StoreError

SERVER_NAME = "memory-rag"
AGENT_BRIDGE_NAME = "memory-rag"
#: Where the app serves its agent endpoint, on its own port.
MCP_PATH = "/mcp"

#: How many statements one read may return. A memory grows by appending, so a read is
#: capped and says so; the cap is a parameter so a caller that needs more can ask for
#: more deliberately.
DEFAULT_RESULT_LIMIT = 10
MAX_RESULT_LIMIT = 50

ContentParameter = Annotated[
    str,
    Field(
        description=(
            "The one statement to remember, in the user's own words where it is "
            "theirs. A line or a short paragraph. It states what holds rather than "
            "when it happened: the memory dates every statement itself, and a "
            "statement carrying a date is refused."
        )
    ),
]
ScopeParameter = Annotated[
    str,
    Field(
        default="local",
        description=(
            'Where the statement belongs. "local" is this project, inside the '
            'repository, and is the default. "global" is across projects: the '
            "account's memory, shared by every project on this machine. Read it "
            "off what the user means and how far it reaches rather than off the "
            "wording, because most prompts carry no marker of where a statement "
            "should go."
        ),
    ),
]
ForgetParameter = Annotated[
    str,
    Field(
        description=(
            "The exact `text` of the one statement to remove, as a recall returned "
            "it: not a summary, not a fragment, and not the kind. The match is on "
            "those words and never on meaning, so it removes that statement or "
            "nothing."
        )
    ),
]
QueryParameter = Annotated[
    str,
    Field(
        description=(
            "What to recall, as words. Answered from the statements that match, by "
            "their words and by their meaning. Required: this server answers a "
            "question rather than returning a memory whole."
        )
    ),
]
LimitParameter = Annotated[
    int,
    Field(
        description=(
            f"How many statements to return across both memories, at most "
            f"{MAX_RESULT_LIMIT}. This caps matches, not the size of a file read."
        )
    ),
]

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True, idempotentHint=True, openWorldHint=False
)
WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
)


def agent_tool_names() -> tuple[str, ...]:
    """Return every tool the agent surface offers, in the order it offers them.

    Exposed so a test can pin the surface against the registry rather than against a
    list written out twice, which is the thing this module exists to avoid.
    """

    return (
        *(f"record_memory_{kind.tool}" for kind in KINDS),
        "recall_memory",
        *(f"recall_memory_{kind.tool}" for kind in KINDS),
        "forget_memory",
    )


def create_mcp(
    service: MemoryService,
    *,
    app_state: Callable[[], dict[str, Any]] | None = None,
) -> FastMCP[Any]:
    """Return the server that answers an agent about this account's memories.

    ``app_state`` is the app's own state, which the recall answer carries so an agent
    learns where the workspace is and how many agents are attached to the same process.
    It is a callable rather than a value because the app is not serving when this is
    built.
    """

    server = FastMCP(
        name=SERVER_NAME,
        version=_version(),
        instructions=SERVER_INSTRUCTIONS,
    )

    for kind in KINDS:
        _declare_recorder(server, service, kind)
    for kind in KINDS:
        _declare_recall(server, service, kind)
    _declare_recall_any(server, service, app_state)
    _declare_forget(server, service)
    return server


# -- the tools, one declaration each -----------------------------------------


def _declare_recorder(server: FastMCP[Any], service: MemoryService, kind: Kind) -> None:
    """Add the tool that records one statement under one kind.

    A kind whose memory is not the caller's to choose takes no memory argument, rather
    than taking one and ignoring it: a tool that shows a choice and then makes it is a
    tool the caller has to learn the hard way.
    """

    takes_scope = kind.scope == SCOPE_EITHER
    summary = kind.summary
    if takes_scope:
        summary += (
            "\n\nChoose the memory with `scope`: \"local\" is this project and "
            "\"global\" is the account's, shared by every project on this machine."
        )
    else:
        summary += (
            f"\n\nIt takes no memory argument: a {kind.name} statement always goes "
            f"in the {kind.scope} memory."
        )

    if takes_scope:

        async def record(
            content: ContentParameter,
            scope: ScopeParameter = "local",
        ) -> dict[str, Any]:
            return await _guarded(
                service.record(content, kind=kind.name, scope=scope)
            )

    else:

        async def record(content: ContentParameter) -> dict[str, Any]:
            return await _guarded(service.record(content, kind=kind.name))

    record.__doc__ = summary
    record.__name__ = f"record_memory_{kind.tool}"
    server.tool(
        name=record.__name__,
        annotations=WRITE_ANNOTATIONS,
        description=summary,
    )(record)


def _declare_recall(server: FastMCP[Any], service: MemoryService, kind: Kind) -> None:
    """Add the tool that recalls from one kind."""

    async def recall(query: QueryParameter, limit: LimitParameter = DEFAULT_RESULT_LIMIT):
        return await _guarded(service.recall(query, kind=kind.name, limit=limit))

    summary = (
        f"Returns what is remembered under {kind.name} that matches these words.\n\n"
        f"{kind.summary}\n\n"
        "Give it a few words, not a sentence. When nothing comes back, that is what "
        "the search found."
    )
    recall.__doc__ = summary
    recall.__name__ = f"recall_memory_{kind.tool}"
    server.tool(
        name=recall.__name__,
        annotations=READ_ONLY_ANNOTATIONS,
        description=summary,
    )(recall)


def _declare_recall_any(
    server: FastMCP[Any],
    service: MemoryService,
    app_state: Callable[[], dict[str, Any]] | None,
) -> None:
    """Add the tool that answers a question across every kind."""

    async def recall_memory(
        query: QueryParameter, limit: LimitParameter = DEFAULT_RESULT_LIMIT
    ) -> dict[str, Any]:
        answer = await _guarded(service.recall(query, limit=limit))
        # Where the app is, which memories it serves, and how many agents are attached
        # to this same process are facts about the installation rather than about a
        # memory, and no question can make a recall return them. They ride on the one
        # answer an agent always asks for rather than on a tool of its own.
        return {**answer, "app": _app_state(service, app_state)}

    summary = (
        "Returns what is remembered that matches these words, in every kind at once.\n\n"
        "It searches every memory this installation serves — this project's and the "
        "account's — and ranks the results by how well they match, so the best "
        "statement wins whichever memory and whichever kind it is in. Each one names "
        "its scope and its kind.\n\n"
        "Reach for a `recall_memory_<kind>` tool when you know which kind you want; "
        "reach for this one when you are asking about a subject rather than a "
        "category.\n\n"
        "Give it a few words, not a sentence. When nothing comes back, that is what "
        "the search found: try other words."
    )
    server.tool(
        name="recall_memory",
        annotations=READ_ONLY_ANNOTATIONS,
        description=summary,
    )(recall_memory)


def _declare_forget(server: FastMCP[Any], service: MemoryService) -> None:
    """Add the tool that removes one statement."""

    async def forget_memory(
        text: ForgetParameter, scope: ScopeParameter | None = None
    ) -> dict[str, Any]:
        return await _guarded(service.forget(text, scope=scope))

    server.tool(
        name="forget_memory",
        annotations=WRITE_ANNOTATIONS,
        description=(
            "Removes one statement that is no longer true.\n\n"
            "Pass the exact `text` a recall returned. It matches words and never "
            "meaning, so it removes that statement or nothing: if the text matches "
            "more than one, nothing is removed and the answer names them.\n\n"
            "It takes the text and not a kind, because a statement that has stopped "
            "being true is removed whatever kind it was filed under."
        ),
    )(forget_memory)


def _app_state(
    service: MemoryService,
    app_state: Callable[[], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return the installation facts a recall answer carries."""

    return {**service.describe(), **(app_state() if app_state else {})}


def _with_app(
    answer: dict[str, Any],
    service: MemoryService,
    app_state: Callable[[], dict[str, Any]] | None,
) -> dict[str, Any]:
    return answer


async def _guarded(awaitable: Any) -> dict[str, Any]:
    """Await one service call, and turn a refusal into a message the agent can act on.

    A ``ToolError`` is what a client shows its user. An exception that is not one
    becomes a generic failure, which tells the agent nothing and hides which of the
    two it caused — a memory that cannot be opened and a statement that is not
    remembered are different problems with different fixes.
    """

    try:
        return await awaitable
    except (ModelError, StoreError, RegistryError, ValueError) as error:
        raise ToolError(str(error)) from error


def _version() -> str:
    from .. import __version__

    return __version__


__all__ = [
    "AGENT_BRIDGE_NAME",
    "DEFAULT_RESULT_LIMIT",
    "MAX_RESULT_LIMIT",
    "MCP_PATH",
    "SERVER_NAME",
    "agent_tool_names",
    "create_mcp",
]
