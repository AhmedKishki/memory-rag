# AGENTS.md

Rules this product holds that are not derivable from its code or its tests. A rule
stated only in code will be found by reading the code; a rule stated only in a test
will be found when the test fails. The rules here are the ones a change has to be
argued against.

## What this project is

One process holds an account's global memory and every project's local memory, and
serves three front ends from one loopback port: a browser workspace, an agent surface
of four tools, and a command line. The four tools are the whole of what an agent may do
to a memory. Everything else belongs to the command center.

The engine is the memory server's own code, byte for byte, and the on-disk layout is
frozen while that server is installed. This product changed what the memory is *for* —
a place a person manages, not only a place an agent writes — and changed nothing about
what a memory *is*.

## Documentation responsibilities

One home per fact, or the two homes will disagree:

| File | Holds |
|---|---|
| `README.md` | the user manual: installing, the first command, every command, the limits |
| `STORAGE.md` | the state format: every directory, file, column, and field name |
| `AGENTS.md` | these rules |
| a module's own docstring | why that module does what it does, and what it may never do |

A module's docstring is load-bearing rather than decorative. A rule stated in two
places will be found disagreeing with itself, and the disagreement will be settled
whichever copy a reader happened to open.

Markdown describes the present. There are no review logs, finding lists, change
histories, before-and-after narratives, or recorded decisions; a finished item leaves
no trace except the code and the commit. No "previously", "used to", "was", or "before
this change". Cross-references name a file by path, never a section number.

Write every sentence in direct language: one fact per sentence, no filler, no selling,
no restating a title as its own first sentence.

## Rules

### The contract a reader relies on

- The four memory tools are the whole agent surface. A fifth tool is a feature this
  product does not own, and a tool that edits a record by hand is a second writer
  beside the one that maintains it.
- The account's global memory and each project's local memory are the only two kinds.
  Global memory is one per account and has no user dimension: no tool, page, or
  argument takes a user identifier.
- A project's local memory lives inside its repository under `.memory-rag` and never
  leaves it. Nothing in this product copies a statement out of a project's directory.
- The record is the record. `MEMORY.md` is a rendering, written from the record and
  never read back as memory. A page that edits the rendering is refused, and told
  where to edit the record instead.
- The four tools take no project argument. The session that records a rule is working
  in one repository, and a statement filed into another is a statement that is wrong
  everywhere else.
- An answer carries a field when the field has news, and nothing else.
- A write is durable before derived, and never fails because the lookup layer is
  unavailable. A missing model leaves statements pending and the read says so.
- Reranking is not optional and no setting turns it off.
- The semantic side is unmeasured. No document may claim a quality gain from it.

### The state a reader must be able to trust

- One app per account, in one process, holding one set of memories. Two apps over one
  account would each hold their own copy of every memory and neither would know.
- Every path this product writes inside a project carries `memory-rag`, so two
  products serving one repository cannot stop each other or read each other's runtime.
- The account's settings directory (`memory-ultra-rag-mcp`), the `MEMORY_ULTRARAG_*`
  environment prefix, the model cache, and the scope directory names keep the names the
  frozen server uses. Renaming any of them strands every existing memory and forces a
  silent model re-download. `tests/test_registry.py` and `tests/test_documentation.py`
  state each one and fail if it moves.
- The on-disk contract is frozen for as long as `memory-ultra-rag-mcp-server` is
  installed, because that server reads and writes the same files. `project.json` may
  only be extended additively; `source-metadata.json` and `source-catalog.json` belong
  to the research app and are not this product's to change.
- The project record (`projects.json`) is a pointer and never a state cache. It holds
  an id, a name, and a root, and nothing a memory owns. Deleting it is how a project
  is unregistered; nothing in it has to be rebuilt.
- A project's recorded name is what an agent's client entry carries, never a path. A
  configuration is written once and copied between machines, and a path in it is true on
  exactly one of them.
- `project_id` is derived from the project's resolved path rather than generated, so a
  project initialised on two machines gets the same id on both.
- A pid file outlives its process and the number in it is reused, so a recorded pid is
  believed only when the process it names still says it is this app's. `start` decides
  by that check and `stop` signals nothing it cannot prove it owns.
- `registry.py` resolves a name exactly, never as a substring. A substring match would
  put a session's memory in a repository its caller did not ask for.

### The SQL console, which is the one bounded writer

- A read is unrestricted. Any `SELECT` runs, and the read path holds its connection
  read-only, so the guarantee is SQLite's rather than this code's judgement.
- A write may touch the `unit` table and nothing else. `DROP`, `CREATE`, `ALTER`,
  `PRAGMA`, `ATTACH`, `VACUUM`, and writes naming `meta` or `vector` are refused by
  name, and so is any statement that mentions a protected table anywhere.
- One statement per run. Comments are removed before the statement is counted, because
  a comment is where a second statement hides.
- The tables behind a statement are not separate from it. `unit` is an FTS5 virtual
  table, so the words that find a statement and the statement itself are the same rows
  and a hand edit cannot put them out of step. The one thing an edit can leave behind
  is a vector describing text the statement no longer holds.
- So an accepted write does exactly three things: it re-renders the standing document,
  it drops the vectors of the statements it touched, and it queues them again. Which
  statements it touched is found by comparing the record before and after, never by
  reading the write's own `WHERE` clause, because a `WHERE` that was wrong once will be
  wrong again and the consequence is a vector left describing a statement that is gone.
- Nothing here is a second implementation of record, forget, or handoff. It is a
  bounded way to repair a statement, and the tests exist to keep it bounded.

### Boundaries a contributor must not cross

- Keep the engine free of the web stack. `tests/test_architecture.py` fails when a
  module outside `surfaces/` imports `fastmcp`, `mcp`, `pydantic`, `starlette`,
  `uvicorn`, or `ui_ultra_rag_mcp`, and when any module reaches for a surface.
- Declare each tool once, in `surfaces/mcp.py`, and let the stdio bridge proxy them.
  A re-declared tool is a second place for it to be wrong.
- Serve the workspace from the pinned `ui-ultra-rag-mcp` and never copy its static
  files in. A copy stops receiving the fixes the pinned package gets.
- Keep browser and control writes same-origin, JSON-only, and loopback-only. A
  capability that is off is a control that is not rendered rather than one that is
  rendered and refused.
- Keep the package's `surfaces/` split: no surface imports another. A capability
  reached from two surfaces is a capability implemented where both can see it.
- Never accept a path from a caller as a memory's location. A scope name is resolved
  against what the service serves, and an unrecognised one is refused before it can
  become a filesystem path.
- Do not let a reader believe a hand edit is a tool call. Every surface reaches the same
  service, and a write through the SQL panel says what it reindexed.
- A global option goes before the subcommand and a subcommand's own option after it.
  The generated launcher gets this order wrong silently and the failure appears only in
  a log nobody opens, which is why an integration test runs that script as written.

### Reporting a condition

- Name every actionable condition with its remedy beside it, and say a check did not
  run rather than letting it read as healthy.
- `doctor` reads only. It writes nothing, fetches no model, and starts no process, so
  the report exists on a machine that cannot serve yet.
- A failure message names the reason and where to read more. `Connection closed` is
  never the whole answer.
- A gateway that cannot start is reported by the operation that needed it, not by a
  process that exits quietly.

## Architecture

```text
src/memory_rag/
  app.py        the one process: a port, the workspace, /mcp, the client registry
  control.py    the loopback control API the command line speaks, and its client
  bridge.py     `memory-rag mcp`: stdio to the app's agent endpoint
  registry.py   the projects one installation knows about
  launcher.py   the generated script: the port claim, the pid, the log, the stop
  runtime.py    building the service from an account and its settings
  service.py    MemoryService: the one implementation every surface calls
  sql.py        the SQL console: the write rule and the reindex it owes
  config.py     the account, a project's roots, and the settings layers
  surfaces/
    cli.py      the command centre
    mcp.py      the four tools and the instructions
    ui.py       the workspace profile, the adapter, and the SQL panel
  <engine>      index, store, vectors, read, retrieval, maintenance, models,
                settings, instructions, reference
```

Dependencies run downward only. The engine knows nothing about a surface; a surface
knows the service and nothing else; the app composes them.

## Working rule

`pathlib.Path`, type hints, and JSON-serializable payloads. Commit and push after
every change without being asked, and push this repository before a parent that pins
it.
