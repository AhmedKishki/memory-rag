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
    assert launcher.process_is_ours(started["pid"], clean_app)
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


# -- whose process a pid file may name -------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["/bin/sleep", "30"],
        # An unrelated process whose arguments mention this product. The repository
        # directory itself is called `memory-rag`, so any command run against a path in
        # it mentions the product by accident, and a substring match would read all of
        # these as the app.
        ["/bin/sleep", "30", "/home/ahmed/Documents/ultra-rag-mcp-servers/memory-rag"],
        ["/bin/sleep", "30", "/home/ahmed/Documents/memory-ragged/notes.md"],
        ["/bin/sleep", "30", "memory-rag-notes.md"],
        ["vim", "/tmp/memory_rag_notes.md"],
        ["git", "-C", "/srv/memory-rag/app", "status"],
        # A `-m` naming another module is not this app either.
        ["/usr/bin/python3", "-m", "memory_ragged"],
        ["/usr/bin/python3", "-m", "some_other_module", "serve"],
        # A shell that would run the console script is the shell, not the app.
        ["/bin/sh", "-c", "memory-rag serve"],
    ],
)
def test_a_command_line_that_only_mentions_this_product_is_not_its_own(
    arguments,
) -> None:
    """Mentioning this product is not the same as being it.

    `start` decides by this check and `stop` signals what it believes, so reading a
    process that merely carries the product's name as this app's would make a stopped
    app look running and hand `stop` a process that belongs to somebody else.
    """

    assert launcher.arguments_are_ours(arguments) is False


@pytest.mark.parametrize(
    "arguments",
    [
        # The console script, which is how a user starts this product.
        ["/home/ahmed/.venv/bin/memory-rag", "serve", "--port", "8765"],
        ["/usr/bin/bin/memory_rag"],
        ["/opt/memory-rag/bin/memory-rag"],
        # `python -m`, which is what the generated launcher builds.
        ["/usr/bin/python3", "-m", "memory_rag", "serve", "--port", "8765"],
        ["/usr/bin/python3", "-m", "memory-rag"],
        ["python", "-m", "memory_rag", "--project-root", "/srv/app"],
    ],
)
def test_a_command_line_that_starts_this_product_is_its_own(arguments) -> None:
    """Both shapes the product is started in are accepted, so `start` is not blocked."""

    assert launcher.arguments_are_ours(arguments) is True


def test_a_live_process_of_another_program_is_not_believed(clean_app: Path) -> None:
    """The same rule against a real process, so the reading of `/proc` is exercised."""

    unrelated = subprocess.Popen(["/bin/sleep", "30"])
    try:
        assert launcher.process_is_ours(unrelated.pid, clean_app) is False
    finally:
        unrelated.kill()
        unrelated.wait(timeout=30)


def test_a_pid_that_names_no_process_is_not_believed(clean_app: Path) -> None:
    assert launcher.process_is_ours(0, clean_app) is False


def test_the_generated_script_reports_a_start_that_never_answered(
    clean_app: Path,
) -> None:
    """An app that never opened its port is not an app at an address.

    The script waited for the port rather than for the process, and then printed the
    address anyway, so a start that did not work reported success and the reader found
    a refused connection. The failure names the log, and the record is kept because the
    process is alive and `stop` has to find it.
    """

    script = launcher.ensure_launcher(clean_app)
    text = script.read_text(encoding="utf-8")
    assert "served=no" in text, "the script does not track whether the port answered"
    assert 'if [ "$served" != yes ]; then' in text
    # The branch has to come before the address is printed, or the address is still a
    # claim the script has not earned.
    assert text.index('if [ "$served" != yes ]') < text.index("the app is at")


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


# -- whose process a record may name ------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        # Code from a string is not this app, however the string names it.
        ["/usr/bin/python3", "-c", "import memory_rag; memory_rag.main()"],
        # `-m` is only read as a flag of the interpreter, never as an option value.
        ["/usr/bin/python3", "-W", "-m", "memory_rag"],
        ["/usr/bin/python3", "-X", "dev", "memory_rag"],
        # An interactive interpreter names no program.
        ["/usr/bin/python3", "-i", "memory_rag"],
        # After `--` the interpreter runs a file, whatever it is called.
        ["/usr/bin/python3", "--", "memory_rag"],
        # The product's name as an option value.
        ["/usr/bin/python3", "-m", "pytest", "-k", "memory_rag"],
        # A wrapper that would run the app is the wrapper.
        ["/usr/bin/env", "memory_rag", "serve"],
    ],
)
def test_a_command_line_that_only_carries_the_name_is_not_the_program(
    arguments, clean_app: Path
) -> None:
    """The program is the first argument or the module an interpreter was given.

    Looking along the line for the product's name reads whatever mentions it as this
    app, and `stop` signals what it believes. The old check accepted any argument equal
    to `-m` followed by the module, so a flag that takes a value could hand it one.
    """

    assert launcher.arguments_are_ours(arguments) is False
    assert launcher._program_of(arguments) is None


def test_the_account_a_process_serves_is_read_off_that_process(
    clean_app: Path,
) -> None:
    """The same rule the app applies to its own inputs, read off the process.

    A pid file can name a live app of another installation, and that one holds another
    account's memories, so a pid alone does not say which app this is.
    """

    other = clean_app / "another-account"
    other.mkdir()
    arguments = ["/usr/bin/python3", "-m", "memory_rag", "--storage-root", str(other)]
    assert launcher.process_storage_root(arguments, "") == other.resolve()

    named = ["/usr/bin/python3", "-m", "memory_rag", f"--storage-root={other}"]
    assert launcher.process_storage_root(named, "") == other.resolve()

    from_environment = ["/usr/bin/python3", "-m", "memory_rag"]
    environ = f"MEMORY_ULTRARAG_STORAGE_ROOT={other}\0PATH=/usr/bin\0"
    assert launcher.process_storage_root(from_environment, environ) == other.resolve()

    # With neither the option nor the environment, the account is the default one, and
    # that is a fact rather than a reason to assume this account.
    from memory_rag.config import default_storage_root

    assert launcher.process_storage_root(from_environment, "") == default_storage_root()


def test_a_live_app_of_another_account_is_neither_believed_nor_stopped(
    clean_app: Path, tmp_path: Path
) -> None:
    """Two installations on one machine must not read or signal each other's app.

    The record is copied from one account to the other, which is what a restored
    backup, a copied home directory, or a stale record does. The process behind it is
    a real app serving the account that started it, so it has to be left running.
    """

    started = launcher.start_app(clean_app)
    assert started["started"] is True, started.get("stderr")
    assert _wait_for_port(started["port"])
    pid = int(started["pid"])

    other = tmp_path / "another-account"
    other.mkdir()
    launcher.ensure_launcher(other)
    launcher.pid_file(other).write_text(str(pid), encoding="utf-8")
    launcher.port_file(other).write_text(str(started["port"]), encoding="utf-8")

    assert launcher.process_is_ours(pid, clean_app) is True
    assert launcher.process_is_ours(pid, other) is False
    assert launcher.running_pid(other) is None
    assert launcher.running_url(other) is None

    stopped = launcher.stop_app(other)
    assert stopped["stopped"] is True
    assert "not this account's app" in stopped["note"]
    assert launcher.process_exists(pid), "another account's app was signalled"
    assert launcher.running_pid(clean_app) == pid


def test_a_command_that_would_signal_itself_or_an_ancestor_refuses(
    clean_app: Path,
) -> None:
    """A stale record can name the reader's own shell.

    `stop` runs from a terminal, and a record that names that terminal's process would
    end the session that asked for it. A group is never signalled either, for the same
    reason: one signal to a negative pid takes the reader's whole shell with it.
    """

    assert launcher._may_be_signalled(os.getpid()) is False
    assert launcher._may_be_signalled(os.getppid()) is False
    assert launcher._may_be_signalled(1) is False

    unrelated = subprocess.Popen(["/bin/sleep", "30"])
    try:
        assert launcher._may_be_signalled(unrelated.pid) is True
    finally:
        unrelated.kill()
        unrelated.wait(timeout=30)


def test_a_stop_with_no_process_handle_refuses_rather_than_signalling_a_number(
    clean_app: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a handle there is no proof left between the check and the signal.

    A pidfd names the process rather than the number. Without one, a number reused
    between the check and the signal belongs to somebody else, so the app is left
    running and the reason is reported rather than a signal being sent on a guess.
    """

    from memory_rag.config import global_directory

    global_directory(clean_app).mkdir(parents=True, exist_ok=True)
    started = launcher.start_app(clean_app)
    assert started["started"] is True, started.get("stderr")
    assert _wait_for_port(started["port"])

    monkeypatch.setattr(launcher, "_open_handle", lambda pid: None)
    stopped = launcher.stop_app(clean_app)

    assert stopped["stopped"] is False
    assert "no process handle" in stopped["note"]
    assert launcher.process_exists(int(started["pid"]))
    assert launcher.running_pid(clean_app) == int(started["pid"])
