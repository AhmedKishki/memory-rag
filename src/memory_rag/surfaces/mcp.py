"""The agent surface: the four memory tools, and nothing else.

These four verbs are the whole of what an agent may do to a memory. The command center
holds everything else — the project list, the SQL console, the clients, the lifecycle —
and an agent is given none of it, because a tool that is not memory would be a feature
this product does not own and a tool that lets an agent edit the record by hand would be
a second writer beside the one that maintains it.

The tools are declared here and nowhere else. A surface that re-declared one of them
would be a second place for it to be wrong, and the workspace and the command line
would then be able to disagree with an agent about what a memory holds.
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
from ..store import DEFAULT_KIND, StoreError

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
            "theirs. A line or a short paragraph."
        )
    ),
]
KindParameter = Annotated[
    str,
    Field(
        default=DEFAULT_KIND,
        description=(
            "The kind to file it under, chosen from what the statement means. "
            "RULE for an instruction or a standing fact, PLAN for what the project "
            "is meant to be or achieve, PREFERENCE for what the user likes, "
            "CORRECTION for something to stop doing, and any other word in block "
            "letters if the statement fits one better. The same word for the same "
            "kind of thing, because a recall filtered by kind returns everything "
            "filed under it. If nothing fits, ITEM: it is a kind like any other, "
            "and a statement filed under it is found by asking for it. Nothing "
            "here interprets the word."
        ),
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
HandoffParameter = Annotated[
    str,
    Field(
        description=(
            "This session's handoff, in a few sentences: what is done, what is in "
            "flight, and what the next session does first. It replaces the previous "
            "handoff in this project, which is a project's own state."
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
KindFilterParameter = Annotated[
    str | None,
    Field(
        default=None,
        description=(
            "Return only the statements filed under this kind, one word in block "
            "letters. A kind is a column of the record rather than part of a "
            "statement's words, so it is asked for here rather than searched for."
        ),
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

    @server.tool(name="record_memory", annotations=WRITE_ANNOTATIONS)
    async def record_memory(
        content: ContentParameter,
        kind: KindParameter = DEFAULT_KIND,
        scope: ScopeParameter = "local",
    ) -> dict[str, Any]:
        """Records one statement in this account's or this project's memory.

        Call it for something meant to hold beyond this reply: a rule, a principle, a
        decision, a correction, a preference, a path. Not for a request that is
        finished when it is answered.

        Recall first, with words covering the same thing. If a statement comes back
        that says the same, record nothing. If one comes back that contradicts it,
        forget that one first.

        `scope` decides where it goes. "local" is this project, inside the
        repository. "global" is across projects, in the account's memory. Choose it
        from what the user means and how far it reaches — a fact about this code, a
        path, a convention here is local; a preference about how they want to be
        spoken to, or a rule about their own work rather than this repository, is
        global. Prompts rarely say so, so judge the substance and not the wording.
        """

        return await _guarded(service.record(content, kind=kind, scope=scope))

    @server.tool(name="recall_memory", annotations=READ_ONLY_ANNOTATIONS)
    async def recall_memory(
        query: QueryParameter,
        kind: KindFilterParameter = None,
        limit: LimitParameter = DEFAULT_RESULT_LIMIT,
    ) -> dict[str, Any]:
        """Returns what is remembered that matches these words.

        It searches every memory this installation serves — this project's and the
        account's — and ranks the results by how well they match, so the best
        statement wins whichever memory it is in. Each one names its scope.

        Give it a few words, not a sentence. When nothing comes back, that is what the
        search found: try other words, or pass a kind.
        """

        answer = await _guarded(service.recall(query, kind=kind, limit=limit))
        # Where the app is, which memories it serves, and how many agents are attached
        # to this same process are facts about the installation rather than about a
        # memory, and no question can make a recall return them. They ride on the one
        # answer an agent always asks for rather than on a fifth tool it would have to
        # decide to call.
        answer["app"] = {**service.describe(), **(app_state() if app_state else {})}
        return answer

    @server.tool(name="forget_memory", annotations=WRITE_ANNOTATIONS)
    async def forget_memory(
        text: ForgetParameter, scope: ScopeParameter | None = None
    ) -> dict[str, Any]:
        """Removes one statement that is no longer true.

        Pass the exact `text` a recall returned. It matches words and never meaning,
        so it removes that statement or nothing: if the text matches more than one,
        nothing is removed and the answer names them.

        Use it when a recalled statement is contradicted, and when a statement has
        simply stopped being true.
        """

        return await _guarded(service.forget(text, scope=scope))

    @server.tool(name="record_handoff", annotations=WRITE_ANNOTATIONS)
    async def record_handoff(content: HandoffParameter) -> dict[str, Any]:
        """Records this session's handoff for the next one, replacing the last.

        One statement at the top of this project's memory. The previous handoff is
        removed as part of the same call, so a project holds one handoff rather than a
        list of them, and `replaced` says how many went.

        Write what the next session needs to pick the work up: what is done, what is
        in flight, and what to do first.
        """

        return await _guarded(service.handoff(content))

    return server


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
    "create_mcp",
]
