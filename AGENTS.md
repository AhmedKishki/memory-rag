---
name: AGENTS.md
description: Memory app boundaries, shared-storage compatibility, and contribution rules.
---

# Memory app rules

## What this project is

- One account-wide process serves global and registered-project memories over one loopback port.
  - Browser, CLI, and MCP use the same service; agents get only memory tools.
- Ship this app's engine and workspace here, independently of any sibling product.
  - Preserve the storage contract rather than identical source code.
- Preserve UltraRAG attribution, licences, `NOTICE`, and the independent-project disclaimer.

## Documentation responsibilities

- Give each fact one owner:
  - `README.md`: installation, commands, usage, and limits.
  - `STORAGE.md`: directories, files, columns, and fields.
  - `AGENTS.md`: engineering rules.
  - Module docstrings: local mechanisms and boundaries.
- Document current state, not review logs, completed work, or superseded decisions; Git retains history.
- Reference file paths, not section numbers.
- Write one fact per sentence, without filler, sales language, or repeated headings.

## Rules

### The contract a reader relies on

- Generate per-kind record and recall tools from `store.KINDS`, without a `kind` argument or default kind.
  - `tests/test_architecture.py` inspects registered tools at runtime, not by source grep.
  - `recall_memory` searches all kinds; `forget_memory` requires exact text.
  - No non-memory tools or agent-side hand editing.
- Kind-specific scope rules:
  - `PERSONALITY` and `PREFERENCE`: global, without a scope argument.
  - `HANDOFF`: one local statement, replacing the previous handoff, without a scope argument.
  - Other kinds: caller-selected local or global memory.
- Reject unknown kinds on writes, naming allowed kinds; keep reads permissive for existing records.
  - Kindless legacy imports use `store.DEFAULT_KIND = ITEM`, never new tool writes.
- Statements describe standing facts, not dates or times; `added_at` records the date separately.
  - `store.admission_failure` judges text deterministically, never embeddings.
  - A standalone year is allowed; dated updates receive a `record_memory_handoff` remedy.
- Do not reject writes for similarity or wait for vectors; collapse repetition across the merged recall answer.
- Global memory is account-wide, with no user identifier in tools, pages, or arguments.
- Local memory stays under the project's `.memory-rag`; never copy it out.
- `MEMORY.md` is a rendering, not authoritative memory or an editable page.
  - Refuse page edits and direct the reader to the record; legacy recovery is defined in `STORAGE.md`.
- Tools take no project argument and must not file local statements in another project's memory.
- Emit only actionable or explanatory response fields.
- Commit statements before derived work; lookup/model failure must not lose or fail a write.
  - Disclose pending statements; retry missing models once per interval, reset by a write or `reindex`.
- Remove obsolete vectors after writes and report the count.
  - `MemoryIndex.collect_vectors` uses the write's existing connection.
- Reranking is not optional and no setting turns it off.
- The semantic side is unmeasured. No document may claim a quality gain from it.

### The state a reader must be able to trust

- Run one app per account; namespace project runtime paths with `memory-rag`.
- Preserve the settings directory `memory-ultra-rag-mcp`, the `MEMORY_ULTRARAG_*` prefix, the model cache, and the scope names.
  - `tests/test_registry.py` and `tests/test_documentation.py` pin these names.
- The on-disk contract is frozen while an installed copy reads and writes the same files.
  - The record is `memory.sqlite3`, under `<project-root>/.memory-rag/` and `<storage-root>/memory/default/`.
  - Its schema version is `8`.
  - FTS5 statement rows are the word index and record.
  - Preserve vector columns: `unit_key`, `stamp`, `kind`, `model`, `dimension`, `components`.
  - Render prose under a heading, without kind prefixes.
  - `STORAGE.md` owns the layout; `tests/test_compatibility.py` pins it.
- Extend `project.json` only additively; never alter research-owned metadata or catalog files.
- `projects.json` holds id, name, and root pointers, never cached memory state; removing an entry unregisters it.
- Client entries carry recorded names, never paths; `registry.py` resolves exactly and refuses ambiguity.
- New `project_id` values derive from resolved paths; preserve existing descriptor IDs.
  - Different checkout paths can produce different IDs; the recorded name is the portable address.
- Prove PID ownership before `start` trusts a record or `stop` signals it.

### The SQL console, which is the one bounded writer

- Run reads through SQLite read-only connections; allow `SELECT` without a query allowlist.
- Allow one statement per run after removing comments.
- Writes may touch only `unit`; reject protected-table references anywhere in a write.
  - Refuse `DROP`, `CREATE`, `ALTER`, `PRAGMA`, `ATTACH`, `VACUUM`, and writes to `meta` or `vector`.
- Accepted writes re-render `MEMORY.md`, remove touched vectors, and queue re-embedding.
  - Determine touched statements from before/after records, never by guessing from `WHERE`.
- Keep SQL a bounded repair path, not another implementation of record, forget, or handoff.

### Boundaries a contributor must not cross

- Keep the engine free of `fastmcp`, `mcp`, `pydantic`, `starlette`, `uvicorn`, and surface imports.
  - `tests/test_architecture.py` owns composition exceptions; surfaces must not import one another.
- Declare tools once in `surfaces/mcp.py`; the bridge proxies rather than re-declares them.
- Ship workspace code and assets only in `surfaces/workspace/`; hide disabled controls.
- Keep browser and control writes same-origin, JSON-only, and loopback-only.
- Resolve scope names against served memories; reject unknown names and caller-selected filesystem paths.
- Distinguish SQL repairs from tool writes and report what was reindexed.
- Place global options before subcommands and command options after them; execute the generated launcher in integration tests.

### Reporting a condition

- Name actionable conditions, remedies, and log locations; unchecked does not mean healthy.
- `doctor` reads only: no write, model fetch, or process startup.
- Never report only `Connection closed`.

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
    mcp.py      the per-kind tools and the instructions
    ui.py       the workspace profile, the adapter, and the SQL panel
    workspace/  the browser workspace: the app, its contracts, its assets
  <engine>      index, store, vectors, read, retrieval, maintenance, models,
                settings, settings_layers, instructions, reference
```

- Dependencies run downward only.
  - The engine knows nothing about a surface.
  - A surface knows the service and nothing else.
  - The app composes them.

## Working rule

- Use `pathlib.Path`, type hints, and JSON-serializable payloads.
- Validate, commit, and push each change before updating a parent's pointer.
