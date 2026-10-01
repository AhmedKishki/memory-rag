"""The UltraRAG revision this server's memory behaviour is taken from.

Nothing here is configurable. The two memory kinds this server serves follow one
behaviour — a standing document written in a fixed format — and this module
records which upstream revision that behaviour and that format were taken from,
so a change to either is a deliberate change to this file.

Upstream also keeps a dated exchange log beside the standing document. This
package does not read or write one, and the divergence is recorded in the
README's choice table.
"""

from __future__ import annotations

__all__ = [
    "BASELINE_COMMIT",
    "BASELINE_VERSION",
    "REFERENCE_FILES",
    "UPSTREAM_REPOSITORY",
]

#: The UltraRAG revision the behaviour and formats were taken from.
BASELINE_COMMIT = "3a709a2aea3fbe46acca59c422621c94b6e86857"

#: The version that revision declares.
BASELINE_VERSION = "0.3.0.2"

#: The upstream repository.
UPSTREAM_REPOSITORY = "https://github.com/OpenBMB/UltraRAG"

#: The files the behaviour was taken from, and the fixture holding their output.
REFERENCE_FILES = ("servers/memory/src/memory.py",)

#: Where this package keeps the bytes upstream produced, for the format tests.
FIXTURE_DIRECTORY = "tests/fixtures/upstream"
