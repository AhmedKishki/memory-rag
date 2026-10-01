"""The generated launcher, run as written.

The launcher is a shell script this package writes at runtime, so nothing about it is
checked by importing it. Its failures are quiet by nature: a mistake in the argument
order it builds, or a process that does not outlive the shell that started it, appears
only in a log nobody opens, and the command that started it reported success.

So these tests write the script, run it, and check what came back. They are the only
tests here that spawn a process, and they are the reason the suite is slower than the
rest.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from memory_rag import launcher

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_app(storage_root: Path):
    """Stop anything this account recorded, so a test starts from nothing."""

    launcher.stop_app(storage_root)
    yield storage_root
    launcher.stop_app(storage_root)


def _wait_for_port(port: int, seconds: float = 60.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def test_the_launcher_is_written_with_this_build_in_its_marker(clean_app: Path) -> None:
    from memory_rag import __version__

    script = launcher.ensure_launcher(clean_app)
    assert script.is_file()
    assert launcher.launcher_is_current(script)
    first = script.read_text(encoding="utf-8")
    assert f"# {launcher.LAUNCHER_MARKER} {__version__}:" in first


def test_a_launcher_from_another_build_is_replaced_not_run(clean_app: Path) -> None:
    """A template that changed between two builds of the same product has to be
    rewritten.

    A marker that named only the product would pass for a file whose command this
    build no longer accepts, and the failure would appear only in a log.
    """

    script = launcher.ensure_launcher(clean_app)
    script.write_text(
        "# memory-rag-app 0.0.0: written by a build that is not this one\n",
        encoding="utf-8",
    )
    assert not launcher.launcher_is_current(script)
    launcher.ensure_launcher(clean_app)
    assert launcher.launcher_is_current(script)


def test_the_launcher_keeps_the_app_alive_after_its_own_shell_exits(
    clean_app: Path,
) -> None:
    """`start` promises an app that outlives the command that started it.

    A child that dies with the shell that spawned it is the failure this checks, and it
    is why the script puts the app in its own session where the system allows it.
    """

    from memory_rag.config import global_directory

    global_directory(clean_app).mkdir(parents=True, exist_ok=True)
    started = launcher.start_app(clean_app)
    assert started["started"] is True, started.get("stderr") or started.get("stdout")
    port = started["port"]
    assert port is not None
    assert _wait_for_port(port), "the app did not answer on the port it recorded"

    # The launching shell is long gone by now; the process must still be there.
    assert launcher.running_pid(clean_app) == started["pid"]
    assert launcher.process_is_ours(started["pid"])
    assert launcher.running_url(clean_app) == f"http://127.0.0.1:{port}"


def test_the_app_the_launcher_starts_serves_the_launchers_own_account(
    clean_app: Path,
) -> None:
    """A launcher that chose a port for one account and started an app configured for
    another is two installations agreeing about nothing.

    This is the check that catches a global option placed after the subcommand, which
    the parser rejects outright and which otherwise fails only in the log.
    """

    import httpx

    started = launcher.start_app(clean_app)
    assert started["started"] is True, started.get("stderr")
    port = started["port"]
    assert _wait_for_port(port)
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30.0) as client:
        body = client.get("/control/status").json()
    assert body["storage_root"] == str(clean_app), (
        "the app answered for a different account than the one the launcher was "
        "written under"
    )
    assert body["app"]["ui_url"] == f"http://127.0.0.1:{port}"


def test_stopping_the_app_removes_its_record(clean_app: Path) -> None:
    started = launcher.start_app(clean_app)
    assert started["started"] is True, started.get("stderr")
    port = started["port"]
    assert _wait_for_port(port)

    stopped = launcher.stop_app(clean_app)
    assert stopped["stopped"] is True, stopped.get("stderr")
    assert launcher.running_pid(clean_app) is None
    assert launcher.running_url(clean_app) is None


def test_a_stale_pid_whose_number_was_reused_is_not_believed(clean_app: Path) -> None:
    """A pid file outlives its process and the number is reused.

    Believing the number would make a stopped app look running, so `start` would refuse
    to start, and `stop` would signal a process that belongs to somebody else.
    """

    script = launcher.ensure_launcher(clean_app)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        launcher.pid_file(clean_app).write_text(str(unrelated.pid), encoding="utf-8")
        assert launcher.running_pid(clean_app) is None
        assert launcher.running_url(clean_app) is None

        stopped = launcher.stop_app(clean_app)
        assert unrelated.poll() is None, "the unrelated process was signalled"
        assert stopped["stopped"] is True
    finally:
        unrelated.send_signal(signal.SIGTERM)
        unrelated.wait(timeout=30)
    assert script.is_file()


def test_starting_twice_leaves_the_first_app_alone(clean_app: Path) -> None:
    first = launcher.start_app(clean_app)
    assert first["started"] is True, first.get("stderr")
    assert _wait_for_port(first["port"])

    second = launcher.start_app(clean_app)
    assert second["started"] is False
    assert second["note"] == "The app was already running."
    assert second["pid"] == first["pid"]


def test_the_port_asked_for_is_used_when_it_is_free(clean_app: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        wanted = int(probe.getsockname()[1])
    started = launcher.start_app(clean_app, port=wanted)
    assert started["started"] is True, started.get("stderr")
    assert started["port"] == wanted
    assert _wait_for_port(wanted)


def test_a_taken_port_is_walked_forward_rather_than_failed(clean_app: Path) -> None:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = int(held.getsockname()[1])
        started = launcher.start_app(clean_app, port=taken)
        assert started["started"] is True, started.get("stderr")
        assert started["port"] > taken
        assert _wait_for_port(started["port"])


def test_the_port_and_pid_files_name_this_product(clean_app: Path) -> None:
    """Two products serving one machine must not read or remove each other's runtime."""

    assert launcher.port_file(clean_app).name == "memory-rag.port"
    assert launcher.pid_file(clean_app).name == "memory-rag.pid"
    assert launcher.log_path(clean_app).name == "memory-rag.log"


def test_the_link_inside_a_project_points_at_the_accounts_launcher(
    clean_app: Path, tmp_path: Path
) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    launcher.ensure_launcher(clean_app)
    assert launcher.ensure_link(project, clean_app) == "written"
    assert launcher.link_path(project).is_symlink()
    assert os.readlink(launcher.link_path(project)) == str(
        launcher.launcher_path(clean_app)
    )
    assert launcher.ensure_link(project, clean_app) == "current"


def test_a_file_of_the_same_name_is_left_alone(clean_app: Path, tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    (project / launcher.LINK_NAME).write_text("mine", encoding="utf-8")
    assert "left alone" in launcher.ensure_link(project, clean_app)
    assert (project / launcher.LINK_NAME).read_text(encoding="utf-8") == "mine"


def test_the_state_of_a_stopped_account_says_so(clean_app: Path) -> None:
    state = launcher.state(clean_app)
    assert state["running"] is False
    assert state["url"] is None
    assert state["log"].endswith("memory-rag.log")


def test_the_entry_a_project_makes_names_its_name_and_not_its_path(
    clean_app: Path, tmp_path: Path
) -> None:
    from memory_rag import registry
    from memory_rag.surfaces.cli import _mcp_entry

    project = tmp_path / "a project with spaces"
    project.mkdir()
    registry.register("abc123", "my-project", project)

    class _Arguments:
        entry_project = "my-project"
        check = None

    entry = _mcp_entry(_Arguments(), _account(clean_app))
    text = json.dumps(entry.payload)
    assert "my-project" in text
    assert str(project) not in text, "the entry carries a path, so it is not portable"
    assert entry.payload["mcpServers"]["memory-rag"]["args"][-1] == "mcp"


def _account(storage_root: Path):
    from memory_rag.config import AccountConfig, global_directory

    return AccountConfig(
        storage_root=storage_root, global_directory=global_directory(storage_root)
    )
