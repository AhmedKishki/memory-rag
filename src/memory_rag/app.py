"""The one running instance: an account's memories, a port, and three HTTP surfaces.

The browser workspace, the agent surface, and the control API the command line speaks
are all served here, on one loopback port, from one process. That is what makes the app
the command center rather than one of several readers: there is exactly one place where
a memory is written, and every surface reaches it.

The service is built before the port is claimed, so a misconfigured account fails before
anything is listening rather than after a user opens a browser to a page that cannot
answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from .config import ConfigurationError
from .service import MemoryService

LOGGER = logging.getLogger(__name__)

UI_HOST = "127.0.0.1"
CONTROL_PREFIX = "/control"
#: Matches uvicorn's own default accept backlog, so a claimed socket is in the state
#: uvicorn would have created for itself.
_CLAIM_BACKLOG = 2048
#: A session that has said nothing for this long is reported as attached no longer. It
#: is kept rather than dropped, so a client that reconnects keeps one identity in the
#: list and a reader can see that it was here.
CLIENT_IDLE_SECONDS = 90.0
#: The header the stdio bridge sets so an agent is identifiable by name. A direct HTTP
#: client that does not set it is identified by its peer address.
CLIENT_NAME_HEADER = "x-memory-rag-client"
PORT_FILE = "memory-rag.port"
PID_FILE = "memory-rag.pid"


class ClientError(Exception):
    """A control request that named something this app does not have."""


def _now() -> float:
    return time.monotonic()


@dataclass
class Client:
    """One MCP client this app has seen.

    ``attached`` is a property of recency and of an explicit drop, never a stored flag.
    """

    session_id: str
    name: str
    first_seen: float
    last_seen: float
    requests: int = 0
    streams: int = 1
    dropped: bool = False
    detached_reason: str | None = None
    # A sighting without a session id: the client opened a connection that has not
    # carried one yet. It is never counted as attached, because a connection is not a
    # client, and the MCP SDK opens the notification stream before the session exists.
    pending: bool = False

    @property
    def attached(self) -> bool:
        if self.pending or self.dropped:
            return False
        return _now() - self.last_seen < CLIENT_IDLE_SECONDS

    @property
    def idle_seconds(self) -> float:
        return round(_now() - self.last_seen, 1)

    def report(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "attached": self.attached,
            "pending": self.pending,
            "dropped": self.dropped,
            # One client opens as many connections as it has streams, and the count is
            # reported rather than hidden so a row is not read as one socket.
            "streams": self.streams,
            "requests": self.requests,
            "idle_seconds": self.idle_seconds,
            "detached_reason": self.detached_reason,
        }


class ClientRegistry:
    """The clients attached to this app, and the authority to drop one.

    Identity is the MCP session id once a request carries one. The initialize request
    has no id yet, so it counts against the peer address, which keeps a client that
    never initializes visible.
    """

    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        # A client's initialize request carries its name but no session id, so it is
        # recorded against the peer address and the entry is carried over when the id
        # appears. Without that, one client reads as two.
        self._awaiting: dict[str, str] = {}

    def _touch(self, session_id: str, name: str, *, pending: bool = False) -> Client:
        existing = self._clients.get(session_id)
        if existing is None:
            existing = Client(
                session_id=session_id,
                name=name or session_id,
                first_seen=_now(),
                last_seen=_now(),
                pending=pending,
            )
            self._clients[session_id] = existing
        existing.last_seen = _now()
        existing.requests += 1
        if name:
            existing.name = name
        return existing

    def observe(self, request: Any) -> str | None:
        """Record one request and return the session id it belongs to."""

        session_id = (request.headers.get("mcp-session-id") or "").strip()
        name = (request.headers.get(CLIENT_NAME_HEADER) or "").strip()
        peer = f"{request.client.host}:{request.client.port}" if request.client else "?"
        if not session_id:
            self._awaiting[peer] = peer
            return self._touch(peer, name, pending=True).session_id
        client = self._touch(session_id, name)
        if client.streams == 1:
            # A client opens one connection per stream, and the streams arrive before
            # the session id exists, so they are folded onto the session by the name
            # the client declared. Two clients that declare the same name are one
            # client as far as this app can tell, which is why the row reports its
            # stream count rather than pretending to be one socket.
            folded = [
                key
                for key, sighting in self._clients.items()
                if sighting.pending
                and key != client.session_id
                and sighting.name == client.name
            ]
            for key in folded:
                other = self._clients.pop(key)
                client.streams += other.streams
        return client.session_id

    def get(self, session_id: str | None) -> Client | None:
        return self._clients.get(session_id) if session_id else None

    def drop(self, session_id: str, reason: str) -> Client:
        """Detach one client, and every stream it opened.

        The streams are matched by the name the client gave itself, and dropping them
        makes the client end rather than sit on an open stream with a session the app
        has already refused. A client that named nothing cannot have its stream
        identified, and only its session is refused.
        """

        client = self._clients.get(session_id)
        if client is None:
            raise ClientError(f"No client is attached with session {session_id!r}.")
        if client.dropped:
            return client
        client.dropped = True
        client.detached_reason = reason
        for other in self._clients.values():
            if other is not client and other.pending and other.name == client.name:
                other.dropped = True
                other.detached_reason = reason
        return client

    def report(self) -> list[dict[str, Any]]:
        return [client.report() for client in self._clients.values()]

    @property
    def attached(self) -> list[Client]:
        return [client for client in self._clients.values() if client.attached]


class ClientGate:
    """Refuse a dropped session before its requests reach the agent.

    The registry is the only authority, so a forced detach is enforced here rather than
    at the tools: a disconnected client sees its session fail rather than a silent
    no-op.
    """

    def __init__(self, app: Any, registry: ClientRegistry) -> None:
        self.app = app
        self.registry = registry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        from starlette.requests import Request

        session_id = self.registry.observe(Request(scope, receive=receive))
        client = self.registry.get(session_id)
        if client is not None and client.dropped:
            response = JSONResponse(
                {
                    "error": (
                        f"This session was disconnected from the app: "
                        f"{client.detached_reason}"
                    ),
                    "session_id": session_id,
                },
                status_code=404,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class Surfaces:
    """Route one port to the agent surface or the workspace.

    A prefix dispatcher rather than three mounts: mounting the agent surface under its
    own path would prefix its routes twice, and a catch-all mount does not fall through
    on a 404.
    """

    def __init__(
        self,
        *,
        mcp: Any,
        workspace: Any,
        registry: ClientRegistry,
        mcp_prefix: str,
    ) -> None:
        self.mcp = ClientGate(mcp, registry)
        self.workspace = workspace
        self.mcp_prefix = mcp_prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if path == self.mcp_prefix or path.startswith(f"{self.mcp_prefix}/"):
            await self.mcp(scope, receive, send)
            return
        await self.workspace(scope, receive, send)


def _claim_loopback_port(host: str, port: int) -> socket.socket:
    """Return the socket holding the claimed loopback port.

    A bound socket that is not listening can be bound again under ``SO_REUSEADDR``,
    which Linux allows, so the claim is the listen and not a probe followed by a bind.
    The loser of a race gets an ``OSError`` here, before any server exists.
    ``SO_REUSEADDR`` is set anyway, matching uvicorn, so a port in ``TIME_WAIT`` after a
    stop is not mistaken for one another process holds.
    """

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(_CLAIM_BACKLOG)
    except OSError:
        sock.close()
        raise
    return sock


def alive(pid: int) -> bool:
    """Whether a process still exists, without signalling it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class App:
    """The one running instance: a service, a port, and three HTTP surfaces.

    The service is built on construction and the models are warmed on start, so a
    caller that arrives before the first read finds the models already resident rather
    than paying for them inside its own lookup.
    """

    def __init__(self, service: MemoryService, *, port: int) -> None:
        if not 1 <= port <= 65535:
            raise ConfigurationError("port must be between 1 and 65535")
        self.service = service
        self.host = UI_HOST
        self.port = port
        self.error: str | None = None
        self.started_at: float | None = None
        self.clients = ClientRegistry()
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._socket: socket.socket | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def mcp_url(self) -> str:
        from .surfaces.mcp import MCP_PATH

        return f"{self.url}{MCP_PATH}"

    @property
    def ready(self) -> bool:
        """Whether the app is serving right now.

        uvicorn sets ``Server.started`` once and never clears it, so the live task is
        part of the test: after ``stop`` or a failed claim the task is finished.
        """

        return (
            self._server is not None
            and bool(self._server.started)
            and self._task is not None
            and not self._task.done()
        )

    def state(self) -> dict[str, Any]:
        """The app's own state, which the agent surface reports and the CLI reads."""

        return {
            "ui_url": self.url,
            "ui_ready": self.ready,
            "ui_error": self.error,
            "mcp_url": self.mcp_url,
            "mcp_clients": len(self.clients.attached),
        }

    def build(self) -> Starlette:
        """Compose the one application every front end is served by.

        Starlette does not run a mounted application's lifespan, so both are entered
        here: the agent app's for its session machinery and the workspace's to install
        the adapter it was handed.
        """

        from .control import control_routes
        from .surfaces.mcp import MCP_PATH, create_mcp
        from .surfaces.ui import create_app as create_workspace

        mcp_http = create_mcp(self.service, app_state=self.state).http_app(
            path=MCP_PATH
        )
        workspace = create_workspace(self.service, self.clients)

        @asynccontextmanager
        async def lifespan(_: Starlette) -> AsyncIterator[None]:
            async with (
                mcp_http.router.lifespan_context(mcp_http),
                workspace.router.lifespan_context(workspace),
            ):
                try:
                    yield
                finally:
                    self.service.close()

        return Starlette(
            routes=[
                *control_routes(self),
                Mount(
                    "/",
                    app=Surfaces(
                        mcp=mcp_http,
                        workspace=workspace,
                        registry=self.clients,
                        mcp_prefix=MCP_PATH,
                    ),
                ),
            ],
            middleware=[Middleware(BaseHTTPMiddleware, dispatch=_security_headers)],
            lifespan=lifespan,
            exception_handlers={Exception: _unexpected_error},
        )

    async def start(self) -> None:
        """Claim the port, warm the models, and serve until stopped.

        The models are warmed here rather than on the first read because loading one
        costs seconds, and a read that has to load a model pays that inside a caller's
        lookup. A model that cannot be fetched is reported and the read falls back to
        words, loudly, rather than the app refusing to serve.
        """

        try:
            claim = _claim_loopback_port(self.host, self.port)
        except OSError as exc:
            reason = exc.strerror or str(exc)
            self.error = (
                f"Port {self.port} is already in use on {self.host}, so the app was "
                f"not started; choose another --port ({reason})."
            )
            return
        self._socket = claim
        self._server = uvicorn.Server(
            uvicorn.Config(
                self.build(),
                host=self.host,
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self.started_at = _now()
        self._task = asyncio.create_task(self._server.serve(sockets=[claim]))
        self._task.add_done_callback(self._task_finished)
        await self._await_serving()
        if self.error is not None:
            return
        report = await self.service.warm()
        if not report.get("embedding_loaded", False):
            LOGGER.warning(
                "the semantic side is unavailable (%s); reads match words only until "
                "the model is fetched",
                report.get("embedding_error", "unknown"),
            )

    async def _await_serving(self, limit: float = 10.0) -> None:
        """Wait until the port is really accepting, or say that it never did.

        `start` returning while the server task has not yet reported itself started
        would let the command line print an address that refuses the first connection
        a user makes to it, and a caller that tests `ready` immediately after `start`
        would read that refusal as a failed start. The wait is bounded, and a server
        that ended in the meantime is reported rather than waited on.
        """

        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            if self._server is not None and self._server.started:
                return
            task = self._task
            if task is None or task.done():
                return
            await asyncio.sleep(0.02)
        if not self.ready and self.error is None:
            self.error = (
                f"The app on {self.host}:{self.port} did not start serving within "
                f"{limit:.0f} seconds. Read the log for where it stopped."
            )

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        """Record an app that ended on its own, and release its claim."""

        if not task.cancelled():
            failure = task.exception()
            if failure is not None:
                self.error = (
                    f"The app on {self.host}:{self.port} stopped: "
                    f"{failure.__class__.__name__}: {failure}"
                )
        self._release_claim()

    def _release_claim(self) -> None:
        """Close the claimed socket, which uvicorn also closes on shutdown."""

        claim, self._socket = self._socket, None
        if claim is not None:
            with contextlib.suppress(OSError):
                claim.close()

    async def wait(self) -> None:
        """Block until the serving task ends, whether it served or it failed."""

        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def stop(self) -> None:
        """Ask the app to stop and wait for its task to finish.

        A failure must not take this process down: ``_task_finished`` records whatever
        ended the task and does not re-raise it.
        """

        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            task, self._task = self._task, None
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._release_claim()
        self.service.close()


@asynccontextmanager
async def running(service: MemoryService, port: int) -> AsyncIterator[App]:
    """Serve this account for the life of this block, then stop it."""

    app = App(service, port=port)
    await app.start()
    try:
        yield app
    finally:
        await app.stop()


async def _security_headers(
    request: Any, call_next: Any
) -> Any:  # pragma: no cover - exercised through the app
    """Answer every request with the headers a browser workspace needs.

    The workspace loads scripts and styles from this origin only, so a content policy
    of ``default-src 'self'`` is what stops one page of it being used to run another's
    script. The control API is loopback-only and same-origin at the route level; these
    headers are the second half of that, not the first.
    """

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' "
        "data:; connect-src 'self'; form-action 'none'; frame-ancestors 'none'"
    )
    if request.url.path.startswith("/api/") or request.url.path.startswith(
        CONTROL_PREFIX
    ):
        response.headers["Cache-Control"] = "no-store"
    return response


async def _unexpected_error(request: Any, exc: Exception) -> JSONResponse:
    """Turn an unhandled failure into a message that names where the rest is.

    A bare ``Internal Server Error`` tells a user nothing they can act on, and this
    app's failures are nearly always a memory that could not be opened, which the
    operation that needed it is the right place to report.
    """

    LOGGER.exception("unhandled failure serving %s", request.url.path, exc_info=exc)
    return JSONResponse(
        {
            "error": (
                f"The app failed while serving this request "
                f"({exc.__class__.__name__}: {exc}). The app log holds the traceback."
            )
        },
        status_code=500,
    )


__all__ = [
    "CLIENT_IDLE_SECONDS",
    "CLIENT_NAME_HEADER",
    "CONTROL_PREFIX",
    "PID_FILE",
    "PORT_FILE",
    "UI_HOST",
    "App",
    "Client",
    "ClientError",
    "ClientGate",
    "ClientRegistry",
    "Surfaces",
    "alive",
    "running",
]
