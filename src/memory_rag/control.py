"""The loopback control API the command line speaks, and its client.

The command line does not open a memory itself when the app is up: it asks the app,
so a terminal answer and a workspace answer are the same object and there is one
process holding the memories rather than one per command. The routes are declared
before the workspace mount, so a URL under ``/control`` is answered here.

Every route is loopback-only and same-origin, and every one of them answers with the
service's own payload rather than a rendering of it, which is what makes the two
surfaces unable to disagree.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .app import CONTROL_PREFIX, App, ClientError
from .config import ConfigurationError
from .registry import RegistryError
from .service import MemoryService
from .sql import SqlRefusal
from .surfaces.workspace.write_guard import (
    json_content_type,
    served_authority,
    write_refusal,
)

#: A recall or a write over a loopback socket with a model behind it is bounded by the
#: model, not by the socket, so the timeout is generous and a caller that needs longer
#: is told why rather than left guessing.
CONTROL_TIMEOUT_SECONDS = 3600.0
#: How long `ensure_running` waits for a launcher that has just been asked to start.
START_TIMEOUT_SECONDS = 180.0


class ControlError(Exception):
    """A refusal from the app, carrying the app's own message."""


async def _body(request: Request) -> dict[str, Any]:
    """Return the request's JSON body, refusing anything that is not JSON.

    The control channel is this app's own, so a body it cannot parse is a caller
    mistake rather than a browser to be lenient with.
    """

    try:
        payload = await request.json()
    except ValueError as exc:
        raise ConfigurationError(f"the control request was not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("the control request body must be a JSON object")
    return payload


def _json(payload: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code)


def _text(value: Any, field: str) -> str:
    """Return a required string field, refusing anything that is not one."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"this request needs a {field} string")
    return value.strip()


async def _status(app: App, request: Request) -> JSONResponse:
    """Everything this installation holds, and what the app is doing with it.

    One answer carries the account, the memories, the app's own address, and the
    clients attached to it, so a reader learns the whole of the command center's state
    from a single call and no two of those can be from different moments.
    """

    from .launcher import state as launcher_state

    return _json(
        {
            **app.service.describe(),
            "app": {**launcher_state(app.service.account.storage_root), **app.state()},
            "clients": app.clients.report(),
        }
    )


async def _health(app: App, request: Request) -> JSONResponse:
    """Answer whether the app is serving, and what it holds.

    This reads no memory and opens no model, so it is the answer a user can get from a
    terminal before anything expensive has happened.
    """

    return _json(
        {
            "ready": app.ready,
            "error": app.error,
            "uptime_seconds": (
                round(time.monotonic() - app.started_at, 1)
                if app.started_at is not None
                else None
            ),
            **app.state(),
        }
    )


async def _clients(app: App, request: Request) -> JSONResponse:
    return _json({"clients": app.clients.report()})


async def _disconnect(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    session_id = _text(body.get("session_id"), "session_id")
    reason = _text(
        body.get("reason") or "disconnected from the command centre", "reason"
    )
    client = app.clients.drop(session_id, reason)
    return _json({"detached": client.report()})


async def _scopes(app: App, request: Request) -> JSONResponse:
    return _json({"scopes": [ref.as_dict() for ref in app.service.scopes()]})


async def _recall(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    query = _text(body.get("query"), "query")
    limit = body.get("limit", 10)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
        raise ConfigurationError("limit must be an integer between 1 and 50")
    kind = body.get("kind")
    if kind is not None and not isinstance(kind, str):
        raise ConfigurationError("kind must be a string when it is given")
    return _json(await app.service.recall(query, kind=kind, limit=limit))


async def _record(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    content = _text(body.get("content"), "content")
    kind = body.get("kind")
    scope = body.get("scope") or "local"
    if kind is not None and not isinstance(kind, str):
        raise ConfigurationError("kind must be a string when it is given")
    return _json(await app.service.record(content, kind=kind, scope=scope))


async def _forget(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    text = _text(body.get("text"), "text")
    scope = body.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ConfigurationError("scope must be a string when it is given")
    return _json(await app.service.forget(text, scope=scope))


async def _handoff(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    return _json(await app.service.handoff(_text(body.get("content"), "content")))


async def _maintain(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    return _json(await app.service.maintain(_text(body.get("scope"), "scope")))


async def _export(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    return _json(await app.service.export(_text(body.get("scope"), "scope")))


async def _sql_query(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    return _json(
        await app.service.sql_query(
            _text(body.get("scope"), "scope"), _text(body.get("statement"), "statement")
        )
    )


async def _sql_execute(app: App, request: Request) -> JSONResponse:
    body = await _body(request)
    return _json(
        await app.service.sql_execute(
            _text(body.get("scope"), "scope"), _text(body.get("statement"), "statement")
        )
    )


async def _handle(app: App, endpoint: Any, request: Request) -> JSONResponse:
    """Run one endpoint, turning its own refusals into answers a caller can act on.

    A refusal the caller can act on is a 400 with the reason; a fault it cannot is the
    app's own 500, which names the log.
    """

    try:
        return await endpoint(app, request)
    except (
        ClientError,
        ConfigurationError,
        RegistryError,
        SqlRefusal,
        ValueError,
    ) as exc:
        return _json({"error": str(exc)}, status_code=400)


def _foreign_host(request: Request) -> JSONResponse | None:
    """Refuse a request whose ``Host`` does not name this app, or None to let it through.

    This API is for the command line on the same machine, and it both writes and reads
    every memory. Binding it to loopback stops a remote host, but a browser page on a
    domain that resolves to this machine is not remote: its connection arrives from
    loopback, and the ``Host`` and ``Origin`` it sends both name its own domain, so an
    origin comparison is satisfied by the page rather than by the user. The name is what
    cannot be forged that way, and it is checked before any route runs, on a read as well
    as on a write: a rebound page reads the answer to a read just as it can make one.
    """

    if served_authority(request) is not None:
        return None
    return _json(
        {
            "error": (
                "This API answers requests that name this app's own loopback address. "
                "A request naming another host was refused before it reached a route; "
                "run 'memory-rag status' on this machine instead of opening the API "
                "from a browser."
            )
        },
        status_code=403,
    )


def _write_refusal(request: Request) -> JSONResponse | None:
    """Refuse a write this app cannot show came from its own page on this machine.

    Every control route that is not a ``GET`` changes what an account holds, so each
    takes the gates the workspace's own routes take, from one implementation rather than
    a line each endpoint remembers to copy. A cross-origin form post is a simple request:
    the browser sends it with no preflight and any content type it likes, so the JSON
    requirement is a gate rather than a description.
    """

    refusal = write_refusal(request)
    if refusal is not None:
        status, reason = refusal
        return _json({"error": reason}, status_code=status)
    if not json_content_type(request):
        return _json(
            {"error": "Write requests require application/json"}, status_code=415
        )
    return None


def control_routes(app: App) -> list[Route]:
    """The routes the command line talks to, on the app's own port.

    They are declared before the workspace mount, so a URL under ``/control`` is
    answered here rather than by the workspace.
    """

    def route(path: str, endpoint: Any, methods: list[str]) -> Route:
        async def bound(request: Request) -> JSONResponse:
            refused = _foreign_host(request)
            if refused is not None:
                return refused
            if "GET" not in methods:
                refused = _write_refusal(request)
                if refused is not None:
                    return refused
            return await _handle(app, endpoint, request)

        bound.__name__ = getattr(endpoint, "__name__", "control")
        return Route(f"{CONTROL_PREFIX}{path}", bound, methods=methods)

    return [
        route("/status", _status, ["GET"]),
        route("/health", _health, ["GET"]),
        route("/clients", _clients, ["GET"]),
        route("/clients/disconnect", _disconnect, ["POST"]),
        route("/scopes", _scopes, ["GET"]),
        route("/recall", _recall, ["POST"]),
        route("/record", _record, ["POST"]),
        route("/forget", _forget, ["POST"]),
        route("/handoff", _handoff, ["POST"]),
        route("/maintain", _maintain, ["POST"]),
        route("/export", _export, ["POST"]),
        route("/sql/query", _sql_query, ["POST"]),
        route("/sql/execute", _sql_execute, ["POST"]),
    ]


class Control:
    """A command line's handle on the running app.

    Every method answers with the app's own payload, so a terminal answer and a
    workspace answer are the same object. A refusal comes back as ``ControlError``
    carrying the app's message.
    """

    def __init__(
        self, base_url: str, *, timeout: float = CONTROL_TIMEOUT_SECONDS
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, f"{CONTROL_PREFIX}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise ControlError(
                f"The app at {self.base_url} did not answer ({exc}); if it has "
                "stopped, start it with 'memory-rag start'."
            ) from exc
        if response.status_code >= 400:
            try:
                message = response.json().get("error") or response.text
            except ValueError:
                message = response.text
            raise ControlError(str(message))
        return response.json()

    def status(self) -> dict[str, Any]:
        return self._call("GET", "/status")

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def clients(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/clients")["clients"])

    def disconnect(self, session_id: str, reason: str) -> dict[str, Any]:
        return self._call(
            "POST",
            "/clients/disconnect",
            json={"session_id": session_id, "reason": reason},
        )

    def scopes(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/scopes")["scopes"])

    def recall(
        self, query: str, *, kind: str | None = None, limit: int = 10
    ) -> dict[str, Any]:
        return self._call(
            "POST", "/recall", json={"query": query, "kind": kind, "limit": limit}
        )

    def record(
        self, content: str, *, kind: str | None = None, scope: str = "local"
    ) -> dict[str, Any]:
        return self._call(
            "POST", "/record", json={"content": content, "kind": kind, "scope": scope}
        )

    def forget(self, text: str, *, scope: str | None = None) -> dict[str, Any]:
        return self._call("POST", "/forget", json={"text": text, "scope": scope})

    def handoff(self, content: str) -> dict[str, Any]:
        return self._call("POST", "/handoff", json={"content": content})

    def maintain(self, scope: str) -> dict[str, Any]:
        return self._call("POST", "/maintain", json={"scope": scope})

    def export(self, scope: str) -> dict[str, Any]:
        return self._call("POST", "/export", json={"scope": scope})

    def sql_query(self, scope: str, statement: str) -> dict[str, Any]:
        return self._call(
            "POST", "/sql/query", json={"scope": scope, "statement": statement}
        )

    def sql_execute(self, scope: str, statement: str) -> dict[str, Any]:
        return self._call(
            "POST", "/sql/execute", json={"scope": scope, "statement": statement}
        )


def connect(storage_root: Path) -> Control | None:
    """Return a handle on the running app, or None when there is not one.

    Reading local state through a daemon that is not up would mean opening a second
    service over the same memories, so a command that only reads answers in process.
    """

    from .launcher import running_url

    url = running_url(storage_root)
    return Control(url) if url is not None else None


def ensure_running(
    storage_root: Path, *, timeout: float = START_TIMEOUT_SECONDS
) -> Control:
    """Return a handle on the app, starting it if the account has none.

    The launcher does the starting, because it owns the free-port choice, the lock,
    the pid file, and the log. This waits for the port it recorded rather than
    probing, so the app reached is the one the launcher reported.
    """

    from .launcher import start_app

    existing = connect(storage_root)
    if existing is not None:
        return existing
    started = start_app(storage_root)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = connect(storage_root)
        if handle is not None:
            return handle
        if not started.get("running", True):
            break
        time.sleep(0.2)
    from .launcher import log_path

    raise ControlError(f"The app did not start; read {log_path(storage_root)}")


__all__ = [
    "CONTROL_PREFIX",
    "CONTROL_TIMEOUT_SECONDS",
    "START_TIMEOUT_SECONDS",
    "Control",
    "ControlError",
    "MemoryService",
    "connect",
    "control_routes",
    "ensure_running",
]
