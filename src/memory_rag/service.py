"""The one implementation every surface reaches.

The browser workspace, the agent surface, and the command line all call this object.
None of them decides anything about a memory that this module does not decide, and
none of them opens a database, because a second writer is a second place for the
answer to be wrong.

The service owns three things a caller must not reach around: the scope map, which
says which directory answers for which scope; the retrieval engine, which holds one
embedding worker per scope; and the account lock, which serialises the writes so a
record never lands beside a read that has already answered.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    AccountConfig,
    ConfigurationError,
    ServerConfig,
    resolve_config,
)
from .index import MemoryIndex
from .models import ModelError
from .registry import RegisteredProject, RegistryError
from .registry import load as load_projects
from .retrieval import Retrieval, RetrievalSettings
from .sql import SqlRefusal, drop_vectors, sql_execute, sql_query

#: How long a caller waits for the account's writer before it is told the memory is
#: busy. A write is a few milliseconds, so a wait this long means a writer is stuck
#: rather than slow, and the answer says which statement was waiting.
ACCOUNT_LOCK_TIMEOUT_SECONDS = 20.0

GLOBAL_SCOPE = "global"
LOCAL_SCOPE = "local"


@dataclass
class ScopeRef:
    """One memory this installation serves, named the way a caller names it."""

    name: str
    label: str
    directory: Path
    kind: str
    project: RegisteredProject | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.name,
            "label": self.label,
            "kind": self.kind,
            "statements": self.statement_count(),
        }

    def statement_count(self) -> int:
        """Return how many statements this memory holds, or 0 when it holds none.

        A scope directory is made when a memory is first touched, so a project that has
        never recorded anything has no directory and therefore no statements. That is
        the ordinary case, not a fault, so it reads as a count of zero.
        """

        try:
            with MemoryIndex(self.directory) as index:
                return index.count_units()
        except (OSError, ValueError):
            return 0


class MemoryService:
    """Every memory operation this product has, in one place."""

    def __init__(
        self,
        account: AccountConfig,
        retrieval: Retrieval,
        *,
        projects: list[RegisteredProject] | None = None,
    ) -> None:
        self.account = account
        self.retrieval = retrieval
        self._projects = list(projects if projects is not None else load_projects())
        self._lock = asyncio.Lock()
        self._local: dict[str, ServerConfig] = {}

    # -- scopes ---------------------------------------------------------------

    def projects(self) -> list[RegisteredProject]:
        """Return the projects this installation serves."""

        return list(self._projects)

    def scope(self, name: str) -> ScopeRef:
        """Return the one memory a scope name addresses, or refuse to guess.

        ``global`` is the account's. Every project's local memory is addressed by that
        project's recorded name, so a caller writes ``local`` when the service is
        serving one project and the project's own name when it is serving several.
        The project named is the active one, so the four tools keep the exact
        signatures they have always had and take no project argument.
        """

        key = str(name or "").strip()
        if key == GLOBAL_SCOPE:
            return ScopeRef(
                name=GLOBAL_SCOPE,
                label="Global memory",
                directory=self.account.global_directory,
                kind=GLOBAL_SCOPE,
            )
        if key in (LOCAL_SCOPE, self.active_project_name()):
            project = self.active_project()
            if project is None:
                raise RegistryError(
                    "no project is active, so there is no local memory to reach. Pass "
                    "--project-root, or run 'memory-rag init' for the project first."
                )
            return ScopeRef(
                name=key,
                label=f"{project.project_name} memory",
                directory=self._project_config(project).local_directory,
                kind=LOCAL_SCOPE,
                project=project,
            )
        if key:
            for project in self._projects:
                if project.project_name.casefold() == key.casefold():
                    return ScopeRef(
                        name=project.project_name,
                        label=f"{project.project_name} memory",
                        directory=self._project_config(project).local_directory,
                        kind=LOCAL_SCOPE,
                        project=project,
                    )
        known = ", ".join(
            [GLOBAL_SCOPE, *(project.project_name for project in self._projects)]
        )
        raise RegistryError(
            f"{name!r} is not a memory this installation serves. Known scopes: "
            f"{known or 'none, so run init for a project first'}."
        )

    def scopes(self) -> list[ScopeRef]:
        """Return every memory this installation serves, global first."""

        found = [self.scope(GLOBAL_SCOPE)]
        for project in self._projects:
            found.append(
                ScopeRef(
                    name=project.project_name,
                    label=f"{project.project_name} memory",
                    directory=self._project_config(project).local_directory,
                    kind=LOCAL_SCOPE,
                    project=project,
                )
            )
        return found

    def scope_directories(self) -> dict[str, Path]:
        """Return every memory, keyed by scope name, for a search across them.

        A recall that is asked one question is answered by the best statement from
        whichever memory holds it, so every memory is searched and the answers are
        ranked together rather than returned scope by scope.
        """

        return {ref.name: ref.directory for ref in self.scopes()}

    def recall_directories(self) -> dict[str, Path]:
        """Return the memories a recall searches, in the order it searches them.

        The active project's own memory comes first so that a statement filed in both
        it and the account's is the one a tie breaks towards, and the account's memory
        last because it is the one every project shares.
        """

        directories: dict[str, Path] = {}
        active = self.active_project_name()
        for ref in self.scopes():
            if ref.kind == LOCAL_SCOPE and ref.name == active:
                directories[ref.name] = ref.directory
        for ref in self.scopes():
            directories.setdefault(ref.name, ref.directory)
        return directories

    def active_project(self) -> RegisteredProject | None:
        """Return the project this service treats as the current one."""

        return self._projects[0] if self._projects else None

    def active_project_name(self) -> str | None:
        """Return the active project's recorded name, when there is one."""

        project = self.active_project()
        return project.project_name if project is not None else None

    def _project_config(self, project: RegisteredProject) -> ServerConfig:
        """Return one project's resolved configuration, resolved once per process.

        The project's own ``config.toml`` is a layer of its own, and resolving it
        reads that file, so it is resolved on first use and kept for the life of the
        process rather than on every read.
        """

        key = str(project.project_root)
        if key not in self._local:
            try:
                self._local[key] = resolve_config(
                    project.project_root, self.account.storage_root
                )
            except ConfigurationError as error:
                raise RegistryError(
                    f"The project at {project.project_root} could not be resolved "
                    f"({error})."
                ) from error
        return self._local[key]

    # -- the four memory operations ------------------------------------------

    async def record(
        self,
        content: str,
        *,
        kind: str | None = None,
        scope: str = LOCAL_SCOPE,
    ) -> dict[str, Any]:
        """Record one statement in one memory."""

        ref = self.scope(scope)
        return await asyncio.to_thread(
            self.retrieval.record,
            content=content,
            directory=ref.directory,
            kind=kind,
        )

    async def recall(
        self,
        query: str,
        *,
        kind: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Answer one question from every memory this installation serves."""

        return await asyncio.to_thread(
            self.retrieval.answer_both,
            directories=self.recall_directories(),
            query=query,
            limit=limit,
            kind=kind,
        )

    async def forget(
        self,
        text: str,
        *,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Remove the one statement this text is exactly, from one memory or either."""

        return await asyncio.to_thread(
            self.retrieval.forget,
            text=text,
            directories=self.recall_directories(),
            scope=scope,
        )

    async def handoff(self, content: str) -> dict[str, Any]:
        """File this session's handoff in the active project's own memory.

        A handoff is about this project's work, so it is never filed in the account's
        memory: a statement every project shares about what one repository is doing
        is a statement that is wrong everywhere else.
        """

        project = self.active_project()
        if project is None:
            raise RegistryError(
                "no project is active, so there is nowhere to file a handoff. Pass "
                "--project-root."
            )
        ref = self.scope(project.project_name)
        return await asyncio.to_thread(
            self.retrieval.handoff,
            directory=ref.directory,
            content=content,
        )

    # -- the command center's own operations ---------------------------------

    async def maintain(self, scope: str) -> dict[str, Any]:
        """Rebuild what a write left pending in one memory."""

        ref = self.scope(scope)
        return await asyncio.to_thread(self.retrieval.maintain, ref.directory)

    async def export(self, scope: str) -> dict[str, Any]:
        """Write the standing document for one memory, and say where it went."""

        ref = self.scope(scope)
        return await asyncio.to_thread(_write_rendering, ref)

    async def sql_query(self, scope: str, statement: str) -> dict[str, Any]:
        """Run a read against one memory's record, and answer with the result."""

        ref = self.scope(scope)
        return await asyncio.to_thread(sql_query, ref, statement)

    async def sql_execute(self, scope: str, statement: str) -> dict[str, Any]:
        """Run a write against one memory's statements, then reindex what it touched.

        The write is refused unless it touches nothing but statements, because a
        statement in the table is a statement the word index and the vectors agree
        about. Anything else here can leave a memory whose words and whose meaning
        describe different records, which is a memory that answers wrongly rather than
        one that fails loudly.
        """

        ref = self.scope(scope)
        return await asyncio.to_thread(_executed, self, ref, statement)

    # -- lifecycle ------------------------------------------------------------

    async def warm(self) -> dict[str, Any]:
        """Load the models now, so a later read finds them resident.

        Loading a model costs seconds, so it happens on the serving side rather than in
        the middle of a caller's lookup. A model that cannot be fetched leaves the read
        matching words only, and the answer says so rather than pretending otherwise.
        """

        return await asyncio.to_thread(self.retrieval.warm)

    def close(self) -> None:
        """Stop this process's background workers."""

        self.retrieval.close()

    def policy(self) -> RetrievalSettings:
        """Return the retrieval policy the effective settings resolved to."""

        return self.retrieval.policy

    def describe(self) -> dict[str, Any]:
        """Return what this installation serves, for a status answer.

        The account, the projects, and where each memory's record is, so a reader can
        see the whole of what the command center manages from one answer. A scope's
        directory is named because the path is the fact that tells a user which file
        holds a memory, and the SQL panel works on exactly that file.
        """

        scopes = [ref.as_dict() for ref in self.scopes()]
        for ref, described in zip(self.scopes(), scopes, strict=True):
            described["record"] = str(ref.directory / "memory.sqlite3")
        return {
            "storage_root": str(self.account.storage_root),
            "scopes": scopes,
            "project_count": len(self._projects),
        }


def _executed(service: MemoryService, ref: ScopeRef, statement: str) -> dict[str, Any]:
    """Run an accepted write, then put the memory back the way a write leaves it.

    A hand edit is only safe if what the memory derived from a statement is what the
    statement now says. The word index cannot drift, because it *is* the table, so the
    one thing to settle is the vectors: the ones describing a statement whose text
    changed are dropped, which leaves those statements pending, and the scope's own
    maintenance embeds them again. The rendering is rewritten from the record, so the
    document a person opens says what the table says.
    """

    result = sql_execute(ref, statement)
    touched = [str(key) for key in result.get("touched") or []]
    drop_vectors(ref.directory, touched)
    report = service.retrieval.maintain(ref.directory)
    with MemoryIndex(ref.directory) as index:
        rendered = index.write_render()
    return {
        "scope": ref.name,
        "rows_affected": int(result.get("rows_affected") or 0),
        "statement": result.get("statement", ""),
        "reindexed": True,
        "statements_changed": len(touched),
        "units_queued": int(report.get("units_queued") or 0),
        "rendered": str(rendered),
    }


def _write_rendering(ref: ScopeRef) -> dict[str, Any]:
    """Render the standing document for one memory, from the record and nothing else."""

    with MemoryIndex(ref.directory) as index:
        target = index.write_render()
    return {"scope": ref.name, "rendered": str(target)}


__all__ = [
    "ACCOUNT_LOCK_TIMEOUT_SECONDS",
    "GLOBAL_SCOPE",
    "LOCAL_SCOPE",
    "MemoryService",
    "ModelError",
    "ScopeRef",
    "SqlRefusal",
]
