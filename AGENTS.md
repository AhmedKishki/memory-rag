# AGENTS.md

Rules a change to this product has to be argued against: what the product is, what must not move, and where each kind of fact is written.

## What this project is

- One process holds an account's global memory and every project's local memory, and serves three front ends from one loopback port: a browser workspace, an agent surface of four tools, and a command line.
- The four tools are the whole of what an agent may do to a memory. Everything else belongs to the command center.
- The engine is the memory server's own code, byte for byte, and the on-disk layout is frozen while that server is installed.
- The change this product makes is to what a memory is for — a place a person manages, not only a place an agent writes — and not to what a memory is.

## Documentation responsibilities

- One home per fact, or the two homes will disagree:
  - `README.md` holds the user manual: installing, the first command, every command, and the limits.
  - `STORAGE.md` holds the state format: every directory, file, column, and field name.
  - `AGENTS.md` holds these rules.
  - A module's own docstring holds why that module does what it does, and what it may never do.
- A module's docstring is load-bearing rather than decorative. A rule stated in two places will be found disagreeing with itself, and the disagreement will be settled by whichever copy a reader happened to open.
- Markdown describes the present, so it carries no review log, finding list, change history, before-and-after narrative, or recorded decision.
  - A finished item leaves no trace except the code and the commit.
  - The words "previously", "used to", "was", and "before this change" do not appear in it.
- Cross-references name a file by path, never a section number.
- Every sentence is direct: one fact per sentence, no filler, no selling, and no title restated as its own first sentence.

## Rules

### The contract a reader relies on

- The four memory tools are the whole agent surface.
  - A fifth tool is a feature this product does not own.
  - A tool that edits a record by hand is a second writer beside the one that maintains it.
- A statement states what holds, and carries no date or time.
  - The record dates every statement in `added_at`, and a read reports that date, so a statement repeating it says the same thing twice.
  - `store.admission_failure` refuses one, and it judges the statement's own words, so the same text is admitted or refused every time rather than depending on what has been embedded.
  - A statement's identity is a digest of its words, and the date is the token that differs between two accounts of one event, so without this rule one approval recorded per session becomes one statement per session.
  - A year on its own is not a date, because naming the edition meant still holds next year.
  - The refusal names `record_handoff`, because a statement about what was done is not a bad statement: it is a handoff, and `HANDOFF` is reserved so only `record_handoff` writes it.
    - A handoff replaces the previous one, so an ordinary record filed under that kind would be removed by the next session's handoff without ever being read.
- Nothing is refused for being similar to what is already there.
  - A write does not wait for a vector, so judging it by meaning would judge the same statement differently depending on what had been embedded.
  - The repetition is resolved in the read, over the whole answer, where it collapses rather than refuses.
- The account's global memory and each project's local memory are the only two kinds.
  - Global memory is one per account and has no user dimension: no tool, page, or argument takes a user identifier.
- A project's local memory lives inside its repository under `.memory-rag` and never leaves it. Nothing in this product copies a statement out of a project's directory.
- The record is the record. `MEMORY.md` is a rendering, written from the record and never read back as memory.
  - A page that edits the rendering is refused, and told where to edit the record instead.
- The four tools take no project argument.
  - The session that records a rule is working in one repository, and a statement filed into another is a statement that is wrong everywhere else.
- An answer carries a field when the field has news, and nothing else.
- A write is durable before derived, and never fails because the lookup layer is unavailable.
  - A missing model leaves statements pending rather than lost, the read says so, and the worker embeds them when a model is available.
  - A missing model costs one retry per interval rather than a core, and the next write or an explicit `reindex` clears the wait.
- A write drops the vectors of statements the record no longer holds, and says how many.
  - A statement is known by a digest of its words, so a reworded statement is a new identity and the vector of the old one describes text that is gone.
  - A live memory held 35 vectors for 23 statements, because only `forget` and `replace_all` collected them and an ordinary write did not.
  - `MemoryIndex.collect_vectors` runs inside the connection the write already holds, so it is a set difference over keys rather than a second open.
- Reranking is not optional and no setting turns it off.
- The semantic side is unmeasured. No document may claim a quality gain from it.

### The state a reader must be able to trust

- One app per account, in one process, holding one set of memories.
  - Two apps over one account would each hold their own copy of every memory and neither would know.
- Every path this product writes inside a project carries `memory-rag`, so two products serving one repository cannot stop each other or read each other's runtime.
- These four names keep what the frozen server looks up: the account's settings directory, `memory-ultra-rag-mcp`; the `MEMORY_ULTRARAG_*` environment prefix; the model cache; and the scope directory names.
  - Renaming any of them strands every existing memory and forces a silent model re-download, and `tests/test_registry.py` and `tests/test_documentation.py` state each one and fail if it moves.
- The on-disk contract is frozen for as long as `memory-ultra-rag-mcp-server` is installed, because that server reads and writes the same files.
  - The record is `memory.sqlite3`, under `<project-root>/.memory-rag/` and `<storage-root>/memory/default/`.
  - Its schema version is `8`.
  - The statement table is an FTS5 table, so the word index and the record are the same rows.
  - The vector table's columns are `unit_key`, `stamp`, `kind`, `model`, `dimension`, and `components`, and a changed shape is dropped and rebuilt, which on a record the other product serves means losing every meaning it held.
  - The rendered document is prose under a heading with no kind in each line, because that is what the other product's parser reads.
  - A value here cannot move while that server is installed, because it cannot move there either. `STORAGE.md` gives each value in full and `tests/test_compatibility.py` pins them.
- `project.json` may only be extended additively; `source-metadata.json` and `source-catalog.json` belong to the research app and are not this product's to change.
- The project record (`projects.json`) is a pointer and never a state cache.
  - It holds an id, a name, and a root, and nothing a memory owns.
  - Deleting it is how a project is unregistered, and nothing in it has to be rebuilt.
- A project's recorded name is what an agent's client entry carries, never a path.
  - A configuration is written once and copied between machines, and a path in it is true on exactly one of them.
- `project_id` is derived from the project's resolved path rather than generated, so a project initialised on two machines gets the same id on both.
- A pid file outlives its process and the number in it is reused, so a recorded pid is believed only when the process it names still says it is this app's.
  - `start` decides by that check and `stop` signals nothing it cannot prove it owns.
- `registry.py` resolves a name exactly, never as a substring. A substring match would put a session's memory in a repository its caller did not ask for.

### The SQL console, which is the one bounded writer

- A read is unrestricted. Any `SELECT` runs, and the read path holds its connection read-only, so the guarantee is SQLite's rather than this code's judgement.
- A write may touch the `unit` table and nothing else.
  - `DROP`, `CREATE`, `ALTER`, `PRAGMA`, `ATTACH`, `VACUUM`, and writes naming `meta` or `vector` are refused by name, and so is any statement that mentions a protected table anywhere.
- One statement per run. Comments are removed before the statement is counted, because a comment is where a second statement hides.
- The tables behind a statement are not separate from it.
  - `unit` is an FTS5 virtual table, so the words that find a statement and the statement itself are the same rows, and a hand edit cannot put them out of step.
  - The one thing an edit can leave behind is a vector describing text the statement no longer holds.
- An accepted write does exactly three things: it re-renders the standing document, it drops the vectors of the statements it touched, and it queues them again.
  - Which statements it touched is found by comparing the record before and after, never by reading the write's own `WHERE` clause, because a `WHERE` that was wrong once will be wrong again and the consequence is a vector left describing a statement that is gone.
- Nothing here is a second implementation of record, forget, or handoff. It is a bounded way to repair a statement, and the tests exist to keep it bounded.

### Boundaries a contributor must not cross

- Keep the engine free of the web stack. `tests/test_architecture.py` fails when a module outside `surfaces/` imports `fastmcp`, `mcp`, `pydantic`, `starlette`, `uvicorn`, or `ui_ultra_rag_mcp`, and when any module reaches for a surface.
- Declare each tool once, in `surfaces/mcp.py`, and let the stdio bridge proxy them. A re-declared tool is a second place for it to be wrong.
- Serve the workspace from the pinned `ui-ultra-rag-mcp` and never copy its static files in. A copy stops receiving the fixes the pinned package gets.
- Keep browser and control writes same-origin, JSON-only, and loopback-only.
  - A capability that is off is a control that is not rendered, rather than one that is rendered and refused.
- Keep the package's `surfaces/` split: no surface imports another. A capability reached from two surfaces is a capability implemented where both can see it.
- Never accept a path from a caller as a memory's location. A scope name is resolved against what the service serves, and an unrecognised one is refused before it can become a filesystem path.
- Do not let a reader believe a hand edit is a tool call. Every surface reaches the same service, and a write through the SQL panel says what it reindexed.
- A global option goes before the subcommand and a subcommand's own option after it.
  - The generated launcher gets this order wrong silently and the failure appears only in a log nobody opens, which is why an integration test runs that script as written.

### Reporting a condition

- Name every actionable condition with its remedy beside it, and say a check did not run rather than letting it read as healthy.
- `doctor` reads only. It writes nothing, fetches no model, and starts no process, so the report exists on a machine that cannot serve yet.
- A failure message names the reason and where to read more. `Connection closed` is never the whole answer.
- A gateway that cannot start is reported by the operation that needed it, not by a process that exits quietly.

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

- Dependencies run downward only.
  - The engine knows nothing about a surface.
  - A surface knows the service and nothing else.
  - The app composes them.

## Working rule

`pathlib.Path`, type hints, and JSON-serializable payloads. Commit and push after every change without being asked, and push this repository before a parent that pins it.
