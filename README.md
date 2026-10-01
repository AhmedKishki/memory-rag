# memory-rag

A command center for an account's global memory and every project's local memory, served from one process on one loopback port.

- An agent's memory is usually only reachable by an agent. `memory-rag` makes the same records readable and editable by a person, and gives an agent no more power than it already had.
- One process holds the account's global memory and the local memory of every registered project, and serves all of them at once.
- An agent gets four tools, unchanged from the product this replaces: record, recall, forget, and a handoff. It cannot see the SQL panel, the project list, or the clients.
- A command line reaches everything the workspace does, against the same process.

## What it does

- **Two kinds of memory.** `global` is the account's, in the storage root, and every project reads it. `local` is one project's, inside that project's repository, and no other project reaches it.
- **A recall that ranks across both.** A question is answered by the best statement from whichever memory holds it, not by one memory's answer followed by the other's.
- **A workspace that shows every memory at once**, with the statement count and the file each memory's record lives in.
- **A SQL panel** that reads freely and writes statements only. See [The SQL panel](#the-sql-panel).
- **A client list** of the agents attached to this process, with the ability to end one.
- **A project registry**, so a client's configuration names a project rather than a path and the command center can say what it serves.

## Requirements

- Python 3.11 or 3.12.
- `uv`.
- A SQLite built with FTS5, which the memory needs to match a statement by its words.
- Roughly 150 MB of model binaries on the first semantic read, cached in the account's model cache and shared by every project on this machine.

## Install

```bash
git clone https://github.com/AhmedKishki/memory-rag.git
cd memory-rag
uv sync
```

## First use

Register a project so the command center knows where its local memory is:

```bash
memory-rag init --project-root /path/to/your/repository --name your-project
```

`init` creates `.memory-rag/` inside the repository, writes the project's identity, adds it to the account's project list, and leaves an `open-memory-rag-ui.sh` in the repository root so the workspace can be opened by double-clicking it. That file is a symlink to a machine-local launcher and is safe to delete.

Bring the app up:

```bash
memory-rag start
```

It prints where the app is and where its log is. Then open the workspace:

```bash
memory-rag ui
```

## Read a memory

```bash
memory-rag scopes
```

```
SCOPE                    KIND    STATEMENTS  RECORD
global                   global           3  ~/.local/share/memory-ultra-rag-mcp/memory/default/memory.sqlite3
your-project             local           11  /path/to/your/repository/.memory-rag/memory.sqlite3
```

A recall searches every memory the app serves and ranks the answers together:

```bash
memory-rag recall "how this repository pins its children"
```

```
[your-project RULE recalled 4x] Every child repository is pinned by commit, never by a version number.
[global PREFERENCE recalled 1x] Prefer a direct, plain sentence over a hedged one.
```

Record, forget, and hand off:

```bash
memory-rag record "The collection pushes the child before the pointer." --kind RULE
memory-rag forget "the exact text a recall returned"
memory-rag handoff "Done: the SQL console. In flight: the workspace panel. Next: release 0.1.0."
```

`forget` matches the exact text of one statement and refuses an ambiguous one, naming the statements that matched.

## The SQL panel

The workspace and the command line can both read a memory's record directly, and a read is unrestricted:

```bash
memory-rag sql --scope your-project "SELECT kind, added_at, text FROM unit ORDER BY added_at DESC"
```

A write is bounded, because `memory.sqlite3` is the record and not a cache:

```bash
memory-rag sql --scope your-project --execute \
  "UPDATE unit SET kind = 'CORRECTION' WHERE kind = 'RULE'"
```

- `INSERT`, `UPDATE`, and `DELETE` against the `unit` table are accepted.
- `DROP`, `CREATE`, `ALTER`, `PRAGMA`, `ATTACH`, `VACUUM`, and anything writing `meta` or `vector` are refused by name.
- One statement at a time.
- An accepted write rewrites the standing document, drops the vectors of the statements it changed so they are embedded again, and says what it reindexed.

## The command line

| Command | What it answers |
|---|---|
| `init` | record a project and create its local memory directory |
| `projects` | every project this installation serves |
| `scopes` | the account's memory and every project's, with their sizes |
| `recall` | what is remembered that matches some words |
| `record` | record one statement in a memory |
| `forget` | remove the one statement a recall returned exactly |
| `handoff` | file this session's handoff, replacing the last one |
| `sql` | run SQL against a memory; `--execute` writes statements |
| `reindex` | settle a memory: re-embed what is missing, re-render it |
| `export` | write the standing document for a memory |
| `start` | bring the app up and print where it is |
| `ui` | open the workspace in a browser |
| `serve` | run the app in this process |
| `stop` | stop the app and what it started |
| `status` | whether the app is up, and what it holds |
| `clients` | the MCP clients attached to the app |
| `disconnect` | end one attached client's session |
| `mcp` | serve an agent over stdio |
| `mcp-entry` | print the client entry a project makes for itself |
| `config` | every setting in force, and the layer that supplied it |
| `doctor` | diagnose this installation without writing to it |
| `help` | the grouped menu, or a page per subject |

- `memory-rag help` prints the menu.
- `memory-rag help sql`, `help scopes`, `help agents`, and `help settings` print the subject pages.
- Add `--json` to any command to get its answer as JSON.

## Connect an agent

- A client that speaks streamable HTTP reaches the app at `<app-url>/mcp`.
- A client that speaks only stdio runs the bridge, which starts the app if it is not up and proxies stdio to the same endpoint.

```jsonc
{
  "mcpServers": {
    "memory-rag": {
      "command": "memory-rag",
      "args": ["--storage-root", "/home/you/.local/share/memory-ultra-rag-mcp",
               "--project", "your-project", "mcp"]
    }
  }
}
```

The entry names a project rather than a path, because a configuration is written once and copied between machines. Either transport names itself, so `memory-rag clients` shows which agent is attached:

```bash
memory-rag clients
```

The command prints the entry this machine needs, with its own executable and storage root filled in:

```bash
memory-rag mcp-entry --project your-project
```

`memory-rag mcp-entry --project your-project --check <path>` compares an entry that is already in place with the one this machine would use and writes nothing.

## Settings

Every tunable is declared in code and shipped in `default.toml`, so a key the code knows and a key a file may set cannot drift apart. An undeclared key is an error in every layer rather than a silent default.

```bash
memory-rag config
```

Layers win per key, lowest first: the packaged defaults, the account's `config.toml`, a project's own `.memory-rag/config.toml`, `--config PATH`, the `MEMORY_ULTRARAG_*` environment, then `--set key=value`.

The account's settings directory and the environment prefix keep the names the product this replaces used, so a rename would strand every existing memory. See [STORAGE.md](STORAGE.md).

## Diagnose an installation

```bash
memory-rag doctor
```

It reads only: it writes nothing, fetches no model, and starts no process, so it is safe on a machine that cannot serve yet. Every check is `ok`, `warn`, or `blocked`, and a blocked check names the command that fixes it.

## Limitations

- A memory holds statements, not documents. There is no corpus, no ingestion, and no original file to open beside a statement.
- The standing document is a rendering. Editing it is refused, because the next read rewrites it from the record.
- The SQL panel is for repairing and inspecting a memory by hand. The four tools remain the way a statement is normally recorded.
- One app per account. A second process over the same account would hold its own copy of every memory, so the port claim refuses a second one.
- The semantic side is unmeasured here. A read that falls back to matching words alone says so in its answer.
- The project record is a pointer. A project whose directory is moved has to be registered again, because the record points at where it was.

## UltraRAG credit and licensing

This product builds on [UltraRAG](https://github.com/OpenBMB/UltraRAG) and its ecosystem, including THUNLP, NEUIR, OpenBMB, and AI9stars. See [NOTICE](NOTICE) for the full attribution, the model licences, and the statement that this is an independent project with no endorsement from any of them.

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
