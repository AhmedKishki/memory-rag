"""The one process, and what has to be true of it from outside.

The product's central claim is that a browser, an agent, and a terminal all reach the
same records through one process. That claim is only worth anything if it is checked
against a real app on a real port, so these tests start one and talk to it over HTTP
rather than calling the service directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator

import httpx
import pytest

from memory_rag.app import App, ClientRegistry, alive, host_names_this_app
from memory_rag.service import MemoryService


def _free_port() -> int:
    """Return a port nothing is listening on, for this test's own app."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture
async def running(service: MemoryService) -> AsyncIterator[App]:
    """A real app on its own port, serving for the life of one test."""

    port = _free_port()
    app = App(service, port=port)
    await app.start()
    try:
        assert app.ready, app.error
        yield app
    finally:
        await app.stop()


async def test_the_app_claims_its_port_before_anything_listens(
    service: MemoryService,
) -> None:
    """Two apps over one account would each hold their own copy of every memory."""

    port = _free_port()
    first = App(service, port=port)
    await first.start()
    try:
        assert first.ready, first.error
        second = App(service, port=port)
        await second.start()
        assert not second.ready
        assert "already in use" in (second.error or "")
        assert second.url != first.url or second.error
    finally:
        await first.stop()


async def test_the_workspace_the_agent_endpoint_and_the_control_api_share_one_port(
    running: App,
) -> None:
    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        assert (await client.get("/")).status_code == 200
        assert (await client.get("/api/status")).status_code == 200
        assert (await client.get(f"{running.mcp_url}")).status_code in (200, 405, 406)
        health = await client.get("/control/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True
        assert health.json()["ui_url"] == running.url


async def test_a_control_request_is_refused_with_a_reason_rather_than_a_crash(
    running: App,
) -> None:
    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        refused = await client.post("/control/recall", json={"query": ""})
        assert refused.status_code == 400
        assert "query" in refused.json()["error"]

        wrong_type = await client.post(
            "/control/recall", json={"query": "a", "limit": 0}
        )
        assert wrong_type.status_code == 400

        # A cross-origin form post is a simple request, so the browser sends it with
        # whatever content type it likes and no preflight. The answer is the guard's,
        # not a body the endpoint refuses to read.
        not_json = await client.post(
            "/control/recall", content=b"{}", headers={"content-type": "text/plain"}
        )
        assert not_json.status_code == 415
        assert "application/json" in not_json.json()["error"]


async def test_a_write_through_the_control_api_is_seen_by_a_read_through_it(
    running: App,
) -> None:
    """One process, one set of records: a write and a read cannot disagree."""

    async with httpx.AsyncClient(base_url=running.url, timeout=60.0) as client:
        written = await client.post(
            "/control/record",
            json={"content": "the child is pushed before the pointer", "kind": "RULE"},
        )
        assert written.status_code == 200
        assert written.json()["kind"] == "RULE"

        read = await client.post("/control/recall", json={"query": "child pointer"})
        assert read.status_code == 200
        assert any(
            unit["text"] == "the child is pushed before the pointer"
            for unit in read.json()["units"]
        )


async def test_the_status_answer_carries_the_account_the_memories_and_the_app(
    running: App,
) -> None:
    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        body = (await client.get("/control/status")).json()
    assert body["storage_root"] == str(running.service.account.storage_root)
    assert {scope["scope"] for scope in body["scopes"]} == {"global", "demo"}
    assert body["app"]["ui_url"] == running.url
    for scope in body["scopes"]:
        assert scope["record"].endswith("memory.sqlite3"), (
            "the record's path is named because it is the fact that tells a user which "
            "file holds a memory, and the SQL panel works on exactly that file"
        )


async def test_the_app_reports_itself_not_ready_after_it_stops(
    service: MemoryService,
) -> None:
    app = App(service, port=_free_port())
    await app.start()
    assert app.ready
    await app.stop()
    assert not app.ready


# -- the client registry --------------------------------------------------------


def test_a_client_is_attached_until_it_has_been_idle_or_dropped() -> None:
    registry = ClientRegistry()
    client = registry._touch("s1", "an-agent")
    assert client.attached
    assert client.report()["name"] == "an-agent"

    client.dropped = True
    assert not client.attached
    assert client.report()["dropped"] is True


def test_a_client_that_has_not_sent_a_session_id_is_never_counted_as_attached() -> None:
    """A connection is not a client, and the SDK opens one before the session exists."""

    registry = ClientRegistry()
    client = registry._touch("127.0.0.1:5000", "half-open", pending=True)
    assert not client.attached
    assert client.report()["pending"] is True
    assert registry.attached == []


def test_dropping_a_session_ends_the_streams_that_client_opened() -> None:
    """A dropped client must end rather than sit on a stream the app has refused."""

    registry = ClientRegistry()
    registry._touch("127.0.0.1:5000", "an-agent", pending=True)
    registry._touch("s1", "an-agent")
    registry._touch("s2", "another-agent")

    registry.drop("s1", "ended by the command line")
    states = {row["session_id"]: row for row in registry.report()}
    assert states["s1"]["dropped"] is True
    assert states["127.0.0.1:5000"]["dropped"] is True
    assert states["s2"]["dropped"] is False


def test_dropping_a_session_that_is_not_attached_is_a_refusal() -> None:
    from memory_rag.app import ClientError

    with pytest.raises(ClientError):
        ClientRegistry().drop("never-seen", "no reason")


def test_alive_reports_a_process_that_is_gone_without_signalling_it() -> None:
    """The check is a liveness question, and a pid that is not in use answers it."""

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    assert alive(process.pid) is False
    assert alive(os.getpid()) is True


# -- the launcher ---------------------------------------------------------------


def test_a_port_another_process_holds_is_refused_with_a_reason(
    service: MemoryService,
) -> None:
    port = _free_port()
    with socket.socket() as held:
        held.bind(("127.0.0.1", port))
        held.listen(1)
        app = App(service, port=port)
        asyncio.run(app.start())
        assert not app.ready
        assert "--port" in (app.error or "")


def test_a_port_outside_the_valid_range_is_refused_before_anything_is_built(
    service: MemoryService,
) -> None:
    from memory_rag.config import ConfigurationError

    for port in (0, 70000):
        with pytest.raises(ConfigurationError):
            App(service, port=port)


def test_the_control_client_refuses_a_refusal_with_the_apps_own_message() -> None:
    from memory_rag.control import Control, ControlError

    control = Control("http://127.0.0.1:1", timeout=1.0)
    try:
        with pytest.raises(ControlError) as raised:
            control.status()
        assert "did not answer" in str(raised.value)
    finally:
        control.close()


# -- whose request this process answers -------------------------------------------


@pytest.mark.parametrize(
    "host,allowed",
    [
        # What a browser reaching the app on its own address sends.
        ("127.0.0.1:8765", True),
        ("127.0.0.1", True),
        ("localhost:8765", True),
        ("[::1]:8765", True),
        ("LOCALHOST:8765", True),
        # A page on a domain that resolves to this machine sends its own name, and both
        # its Host and its Origin name it, so an origin comparison alone is satisfied
        # by the attacker.
        ("attacker.example", False),
        ("attacker.example:8765", False),
        ("memory-rag.attacker.example:8765", False),
        ("127.0.0.1.attacker.example", False),
        # Unusable headers are answers, not failures.
        ("", False),
        (None, False),
        ("127.0.0.1:not-a-port", False),
        ("[::1", False),
    ],
)
def test_only_a_loopback_name_is_this_app(host, allowed) -> None:
    assert host_names_this_app(host) is allowed


async def test_a_request_naming_another_host_is_refused_before_the_route(
    service: MemoryService,
) -> None:
    """The control API writes and reads every memory, so nothing reaches a route first.

    A domain that resolves to this machine is not a remote host: the connection arrives
    from loopback and the page chooses its own Host and Origin. The name is what it does
    not choose.
    """

    from starlette.testclient import TestClient

    port = _free_port()
    app = App(service, port=port)
    await app.start()
    try:
        assert app.ready, app.error
        with TestClient(app.build(), base_url="http://127.0.0.1") as client:
            allowed = client.get("/control/status")
            assert allowed.status_code == 200, allowed.text
            refused = client.get(
                "/control/status", headers={"host": "attacker.example"}
            )
            assert refused.status_code == 403
            assert "loopback" in refused.json()["error"]
    finally:
        await app.stop()


async def test_a_malformed_host_is_refused_rather_than_raising(
    service: MemoryService,
) -> None:
    from starlette.testclient import TestClient

    port = _free_port()
    app = App(service, port=port)
    await app.start()
    try:
        assert app.ready, app.error
        with TestClient(app.build(), base_url="http://127.0.0.1") as client:
            refused = client.get(
                "/control/status", headers={"host": "127.0.0.1:not-a-port"}
            )
            assert refused.status_code == 403
    finally:
        await app.stop()


# -- the control channel a rebound page reaches ----------------------------------

#: Every route the control API answers that is not a `GET`, with a body that would
#: succeed if it were read. A simple cross-origin post carries no preflight, so this is
#: the exact shape a page on a rebound domain can send, and each one has to be refused.
CONTROL_WRITES = [
    ("/control/clients/disconnect", {"session_id": "s-1", "reason": "closed"}),
    ("/control/recall", {"query": "the child is pushed before the pointer"}),
    ("/control/record", {"content": "a rebound page wrote this", "kind": "RULE"}),
    ("/control/forget", {"text": "the child is pushed before the pointer"}),
    ("/control/handoff", {"content": "a rebound page wrote this session's handoff"}),
    ("/control/maintain", {"scope": "local"}),
    ("/control/export", {"scope": "local"}),
    (
        "/control/sql/query",
        {"scope": "local", "statement": "SELECT count(*) FROM unit"},
    ),
    (
        "/control/sql/execute",
        {"scope": "local", "statement": "DELETE FROM unit"},
    ),
]

#: What a page served from a domain that resolves to this machine sends: its own name in
#: both headers, which agree, and any content type it likes, which the browser chooses
#: because the request is simple and needs no preflight.
REBOUND = {
    "origin": "http://memory-rag.attacker.example",
    "sec-fetch-site": "cross-site",
    "content-type": "text/plain;charset=UTF-8",
}


async def test_every_control_write_is_refused_to_a_rebound_page(running: App) -> None:
    """The control API writes and reads every memory, so the guard is on all of it.

    The old wrapper checked the `Host` and nothing else, which a rebound page satisfies
    by naming its own domain: the connection arrives from loopback, and the browser adds
    no preflight to a simple request, so `Origin`, `Sec-Fetch-Site`, and the content type
    were the only things left and none of them was read.
    """

    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        recorded = await client.post(
            "/control/record",
            json={"content": "the child is pushed before the pointer", "kind": "RULE"},
        )
        assert recorded.status_code == 200
        before = (await client.get("/control/status")).json()

        answers = [
            await client.post(path, content=json.dumps(body), headers=REBOUND)
            for path, body in CONTROL_WRITES
        ]

    assert [answer.status_code for answer in answers] == [403] * len(CONTROL_WRITES)
    for answer in answers:
        assert "application/json" not in answer.json()["error"], (
            "the refusal has to name what was wrong with the request, not where it came "
            "from, or a legitimate caller cannot act on it"
        )
    # Nothing was written, nothing was removed, and no client was dropped.
    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        after = (await client.get("/control/status")).json()
    assert after["scopes"] == before["scopes"]
    assert after["clients"] == before["clients"], "a refused request attached a client"
    recall = await _recall(running, "rebound")
    assert [unit["text"] for unit in recall["units"]] == [
        "the child is pushed before the pointer"
    ]


async def _recall(app: App, query: str) -> dict:
    async with httpx.AsyncClient(base_url=app.url, timeout=60.0) as client:
        return (await client.post("/control/recall", json={"query": query})).json()


async def test_a_malformed_origin_is_a_refusal_and_not_a_crash(running: App) -> None:
    """A header a client controls must not be able to reach an exception.

    `urlsplit` raises on an unbalanced IPv6 bracket, so an `Origin` a page chose could
    reach the app's own 500 rather than an answer.
    """

    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        for origin in ("http://[::1", "://", "http://a b"):
            answer = await client.post(
                "/control/record",
                json={"content": "never written", "kind": "RULE"},
                headers={"origin": origin},
            )
            assert answer.status_code == 403, origin


async def test_a_terminal_post_without_an_origin_is_allowed(running: App) -> None:
    """A terminal has no origin, so requiring one would refuse this product's own CLI.

    `control.Control` posts JSON and sends no `Origin`, which is what this request is.
    """

    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        answer = await client.post(
            "/control/record",
            json={"content": "a terminal writes without an origin", "kind": "RULE"},
        )
    assert answer.status_code == 200
    assert answer.json()["kind"] == "RULE"


async def test_a_read_is_refused_to_a_foreign_host_as_well_as_a_write(
    running: App,
) -> None:
    """A rebound page reads the answer to a read as readily as it makes a write."""

    async with httpx.AsyncClient(base_url=running.url, timeout=30.0) as client:
        for path in ("/control/status", "/control/health", "/control/clients"):
            answer = await client.get(path, headers={"host": "attacker.example"})
            assert answer.status_code == 403, path
            assert "loopback" in answer.json()["error"]
