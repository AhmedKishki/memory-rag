"""The browser workspace: its profile, its adapter, and the SQL panel behind it.

This module is the whole boundary between the shared workspace package and this app's
memories. The package is given results and never a path, so it cannot read a record
even if a request asked it to; the adapter decides which memory a request names and
answers from the service.

Three capabilities are declared on and the rest are off, because this app serves no
documents: the memory view, the attached-clients panel, and the SQL console. The
workspace gates its controls on capabilities rather than on a caller's arguments, so a
control this app does not serve is not rendered rather than rendered and refused.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ..app import ClientError, ClientRegistry
from ..index import MemoryIndex
from ..service import MemoryService
from ..sql import SqlRefusal
from .workspace import UICapabilities, UIProfile, create_ui_app
from .workspace.contracts import SourceFile, UIRequestError

PROFILE = UIProfile(
    application_name="memory-rag",
    project_label="Memory scope",
    project_fallback_name="Memory",
    navigation_label="Memory views",
    source_types_label="stored memories",
    ingest_intro=(
        "Every memory this installation serves, the account's own and each project's, "
        "in one place. Read one directly, record into it, or write SQL against its "
        "record."
    ),
    ingest_busy_message="Settling the memory. This can take a few seconds…",
    footer_text=(
        "The account's global memory and every registered project's local memory. "
        "Local memory never leaves its repository."
    ),
    result_text_label="Statement",
    copy_text_label="Copy statement",
    memory_label="Memory",
    memory_standing_label="Standing document",
    memory_rounds_label="Statements",
    memory_add_label="Record a statement",
    memory_note=(
        "A statement recorded here is recorded through the same operation an agent "
        "uses, so a page and an agent cannot hold two different memories."
    ),
    capabilities=UICapabilities(
        # This app serves no documents, so every corpus capability is off: there is no
        # source list, no passage, no ingestion, and no bibliography to filter on. A
        # capability that is off is a control that is not rendered.
        documents=False,
        sources=False,
        passage_context=False,
        ingestion=False,
        metadata=False,
        source_inclusion=False,
        source_files=False,
        metadata_filters=False,
        bibliographic_filters=False,
        project_metadata=False,
        source_selection=False,
        category_partitions=False,
        retrieval_modes=False,
        reranking=False,
        chunk_settings=False,
        force_recompute=False,
        bundle_export=False,
        bundle_import=False,
        # The three this app does serve.
        memory=True,
        memory_writes=True,
        clients=True,
        sql_console=True,
    ),
    version_label="memory-rag",
)


def require_known_scope(service: MemoryService, scope: Any) -> str:
    """Return the scope a request named, or refuse the name before it means anything.

    A scope arrives from the browser, so it is checked here rather than where it is
    used: a name that is not a memory this installation serves must not reach the
    service, and must not become a filesystem path on the way.
    """

    name = str(scope or "").strip()
    if not name:
        raise UIRequestError("This request needs a memory scope.", status_code=400)
    try:
        return service.scope(name).name
    except (ValueError, OSError) as error:
        raise UIRequestError(str(error), status_code=404) from error


class MemoryUIAdapter:
    """Map the shared workspace contract onto this app's memories.

    Every method here calls the service, so the page and the agent surface read the
    same records through the same code. The adapter decides nothing about a memory: a
    scope it resolves, an operation it forwards, and a refusal it translates.
    """

    def __init__(self, service: MemoryService, clients: ClientRegistry) -> None:
        self.service = service
        self.clients = clients

    async def health(self) -> Mapping[str, Any]:
        from ..models import describe_environment

        return {"status": "ok", "environment": describe_environment()}

    async def source_file(self, source_path: str) -> SourceFile:
        """Refuse, because a memory has no original to serve.

        This app stores statements rather than documents, so there is no file a reader
        could open beside a statement. The route is declared off, so this is reached
        only by a request that ignored the capability, and the answer names that.
        """

        raise UIRequestError(
            "This app serves memories rather than documents, so it has no original "
            "file to open.",
            status_code=404,
        )

    async def call(
        self, operation: str, arguments: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """Forward one workspace operation to the service."""

        given = arguments or {}
        if operation == "status":
            return self._status()
        if operation == "memory_status":
            return self._memory_status()
        if operation == "memory_rounds":
            return await self._memory_rounds(given)
        if operation == "memory_standing":
            return await self._memory_standing(given)
        if operation == "memory_append":
            return self._memory_append(given)
        if operation == "memory_standing_save":
            return self._memory_standing_save(given)
        raise UIRequestError(
            f"This app does not serve {operation!r} in a browser.", status_code=404
        )

    def _status(self) -> Mapping[str, Any]:
        """The status view: every memory this installation serves.

        This is where the SQL panel learns which scopes exist, so the scope list, the
        statement counts, and the reader's scope selector all come from one answer and
        cannot disagree about what there is.
        """

        described = self.service.describe()
        return {
            "ready": True,
            "blocked_by": [],
            "degraded": [],
            "sources": described["scopes"],
            "scopes": described["scopes"],
            "sql_scopes": described["scopes"],
            "project_count": described["project_count"],
            "storage_root": described["storage_root"],
            "active_project": self.service.active_project_name(),
        }

    def _memory_status(self) -> Mapping[str, Any]:
        described = self.service.describe()
        return {
            "scopes": described["scopes"],
            "sql_scopes": described["scopes"],
        }

    async def _memory_rounds(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Every statement in one memory, for a person to read rather than to search.

        A page is not an agent: it has no question, so it is answered with what a
        memory holds rather than with what matches. This is the one answer that scales
        with the size of a memory, and it is the workspace's rather than the agent's,
        which is why the agent surface has no equivalent.
        """

        name = require_known_scope(self.service, arguments.get("scope"))
        ref = self.service.scope(name)
        limit = arguments.get("limit")
        statements = self._statements(ref)
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            statements = statements[:limit]
        return {
            "scope": name,
            "rounds": [
                {
                    "round_id": statement.key,
                    "statements": [{"speaker": "memory", "text": statement.text}],
                    "source_file": f"{ref.label} — {statement.kind}",
                }
                for statement in statements
            ],
        }

    def _statements(self, ref: Any) -> list[Any]:
        with MemoryIndex(ref.directory) as index:
            return index.statements()

    async def _memory_standing(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """The standing document for one memory, rendered from the record.

        It is a rendering rather than the record, so what a person edits in the page is
        not what the memory reads back. The answer carries the digest the write below
        would be checked against, and the note that says the file is regenerated.
        """

        name = require_known_scope(self.service, arguments.get("scope"))
        ref = self.service.scope(name)
        with MemoryIndex(ref.directory) as index:
            rendered = index.write_render()
        text = rendered.read_text(encoding="utf-8")
        return {
            "scope": name,
            "content": text,
            "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "path": str(rendered),
            "note": (
                "A rendering of the record, not the record. It is rewritten from the "
                "record on every render, so edit a statement through SQL instead."
            ),
        }

    def _memory_append(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Refuse a dated exchange, and say what a page may record instead.

        The shared workspace's memory view was written for a store that keeps dated
        exchanges. This app's memory keeps statements, which have no speaker and no
        date, so a round is not a shape this store can hold. The refusal names the
        operation that does work rather than accepting something it cannot store.
        """

        require_known_scope(self.service, arguments.get("scope"))
        raise UIRequestError(
            "This memory keeps statements, not dated exchanges, so a round cannot be "
            "recorded. Use the SQL panel to run an INSERT against the unit table, or "
            "the command line's 'memory-rag record'.",
            status_code=409,
        )

    def _memory_standing_save(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Refuse a hand edit of the standing document, and name the way to change one.

        The document is written from the record on every render, so a write here would
        be a change a user believes they made and the next read discards.
        """

        require_known_scope(self.service, arguments.get("scope"))
        raise UIRequestError(
            "The standing document is a rendering of the record and is rewritten from "
            "it, so an edit here would be discarded on the next read. Use the SQL panel "
            "to edit the unit table, or the command line's 'memory-rag record' and "
            "'memory-rag forget'.",
            status_code=409,
        )

    # -- the SQL console ------------------------------------------------------

    async def sql_query(self, scope: str, statement: str) -> Mapping[str, Any]:
        """Run a read against one memory's record."""

        name = require_known_scope(self.service, scope)
        try:
            return await self.service.sql_query(name, statement)
        except SqlRefusal as error:
            raise UIRequestError(str(error), status_code=400) from error

    async def sql_execute(self, scope: str, statement: str) -> Mapping[str, Any]:
        """Run a write against one memory's statements, then reindex what it touched."""

        name = require_known_scope(self.service, scope)
        try:
            return await self.service.sql_execute(name, statement)
        except SqlRefusal as error:
            raise UIRequestError(str(error), status_code=400) from error

    # -- the attached clients -------------------------------------------------

    async def list_clients(self) -> list[Mapping[str, Any]]:
        return self.clients.report()

    async def disconnect_client(
        self, session_id: str, reason: str | None = None
    ) -> Mapping[str, Any]:
        try:
            client = self.clients.drop(
                session_id, reason or "disconnected from the workspace"
            )
        except ClientError as error:
            raise UIRequestError(str(error), status_code=404) from error
        return client.report()


def create_app(service: MemoryService, clients: ClientRegistry) -> Any:
    """Return the workspace this app serves, bound to its adapter."""

    return create_ui_app(
        profile=PROFILE,
        adapter=MemoryUIAdapter(service, clients),
    )


__all__ = ["PROFILE", "MemoryUIAdapter", "create_app", "require_known_scope"]
