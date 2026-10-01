"""A command center for an account's global memory and every project's local memory.

One process serves a browser workspace, an agent surface, and a command line on one
loopback port. The engine is the memory server's own code, and the four memory tools
are the whole of what an agent may do; everything else belongs to the command center.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
