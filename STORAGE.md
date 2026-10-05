# STORAGE

The on-disk format: every directory, file, column, and field name. The only document that defines a field name.

- The format is frozen while an installed copy reads and writes the same files.
  - A change may add to the format, but no change may rename a path or a column.
  - `AGENTS.md` states the rule and `tests/test_compatibility.py` pins the values.

## The two memories

| | Global | Local |
|---|---|---|
| Path | `<storage-root>/memory/default/` | `<project-root>/.memory-rag/` |
| Read by | every project on this machine | that project only |
| Travels with | the account | the repository |
| User dimension | none | none |

- The directory is named as the upstream layout names it, so a reader that expects that layout finds it unaltered.
- No tool, page, or argument carries a user identifier.

## Where the roots come from

| Root | Selected by | Default |
|---|---|---|
| `storage_root` | `--storage-root`, then `$MEMORY_ULTRARAG_STORAGE_ROOT`, then `$ULTRARAG_UI_STORAGE_ROOT` | `platformdirs.user_data_path("memory-ultra-rag-mcp")` |
| `project_root` | `--project-root` | the working directory |
| settings | the layers below | `default.toml` beside `settings.py` |
| model cache | `runtime.model_cache_root`, then `$MEMORY_ULTRARAG_MODEL_CACHE_ROOT` | `platformdirs.user_cache_path("memory-ultra-rag-mcp")/models` |
| project record | — | `platformdirs.user_config_path("memory-ultra-rag-mcp")/projects.json` |

The settings directory, the environment prefix, the model cache, and the scope directory names keep the names the installed server looks up. `AGENTS.md` says why each one is load-bearing.

## A project's directory

```text
<project-root>/
├── open-memory-rag-ui.sh -> <storage-root>/app/bin/memory-rag-app.sh
└── .memory-rag/
    ├── project.json      this product's: the project's identity
    ├── config.toml       this project's settings layer
    ├── memory.sqlite3    the record
    ├── MEMORY.md         a rendering, written from the record
    └── memory.sqlite3-wal, -shm
```

- `open-memory-rag-ui.sh` is a symlink, is machine-local, and holds nothing of yours.
  - It is the only file this product writes at a project's root.
  - It carries the product's own name, so two products serving one repository cannot stop each other.
- `project.json` is additive, and the installed server reads nothing from this directory but `memory.sqlite3`, so a field added here is invisible to it.

```json
{
  "schema_version": 1,
  "project_id": "5f2c1a9e4b7d8036",
  "project_name": "your-project",
  "project_root": "/home/you/your-project",
  "written_by": "memory-rag",
  "written_at": "2026-10-01T09:41:00+00:00"
}
```

- `project_id` is a digest of the project's resolved path, and a descriptor that already holds one keeps it. Two machines whose checkout sits at different paths therefore get different ids, and `AGENTS.md` says why the recorded name is what travels.

## The project record

`projects.json`, beside the account's settings.

```json
{
  "schema_version": 1,
  "projects": [
    {
      "project_id": "5f2c1a9e4b7d8036",
      "project_name": "your-project",
      "project_root": "/home/you/your-project",
      "registered_at": "2026-10-01T09:41:00+00:00"
    }
  ]
}
```

- It holds an id, a name, and a root, and nothing a memory owns. Deleting an entry is how a project is unregistered; `AGENTS.md` says why it is a pointer and not a state cache.
- A record whose project directory has gone is still answerable, and `projects` reports it as having no descriptor.
- Writes are atomic: a temporary file beside it, then a rename, so a reader never sees half a record.

## The app's own state

```text
<storage-root>/app/
├── bin/memory-rag-app.sh   the generated launcher
├── memory-rag.pid          the process this launcher started
├── memory-rag.port         the port it claimed
├── memory-rag.log          everything the app wrote
├── start.lock/             one start at a time
└── logs/
```

Every name carries `memory-rag`, so this product's runtime and another product's cannot be confused for one another.

- The launcher is a generated POSIX script that owns the free-port choice, the lock that makes that choice exclusive, the pid and port files, and the log.
  - Its first line names the version that wrote it, so a template change rewrites it rather than leaving an older command in place.
- `memory-rag.pid` is believed only when the process it names still identifies itself as this app's own. `AGENTS.md` says why a pid number alone is not enough.

## The record

`memory.sqlite3`, one per memory. It is the record, not a derived cache, and the following is all of it.

```sql
CREATE VIRTUAL TABLE unit USING fts5(
    text,                 -- the statement
    unit_key UNINDEXED,   -- the identity the vector and the history are found by
    kind UNINDEXED,       -- the kind it is filed under
    stamp UNINDEXED,      -- the order it was written in
    added_at,             -- the date, and the recency bonus
    recalls               -- how often it has been handed back
);

CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);   -- holds schema_version

CREATE TABLE vector(
    unit_key TEXT PRIMARY KEY,
    stamp    TEXT NOT NULL,
    kind     TEXT NOT NULL,
    model    TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    components BLOB NOT NULL
);
```

`schema_version` is currently `8`, recorded in `meta`.

Every column earns its place by naming its reader:

| Column | Read by |
|---|---|
| `text` | the record |
| `unit_key` | the vector and the history, which are found by it |
| `kind` | a read narrowed by kind, and the label its identity is computed from |
| `stamp` | the order the document is written in, and the tie-break between memories |
| `added_at` | the recency bonus, and the date a caller is told |
| `recalls` | how often the memory has handed the statement back, which a caller judges by |

- `unit` is an FTS5 virtual table, so the words that find a statement and the statement itself are the same rows, and a hand edit to `unit.text` cannot put the word index out of step with the record. `AGENTS.md` says what an edit can still leave behind.
- `index.sqlite3` and `index-vectors.sqlite3` are older files.
  - They are read once, for what they hold, and then removed, and the answer says how many were found.

## The standing document

`MEMORY.md`, written from the record and never read back as memory.

- The statement is newest at the top.
- The document is prose under a heading, with no kind and no date in each line, because that is what the installed server's parser reads. Both are columns of the row, and a file that carried them in its text would be a file of data rather than something a person wrote.
- A file beside the record is read exactly once, and only ever read.
  - Legacy document recovery applies only to an uninitialized record, not one deliberately emptied by deleting or forgetting statements.
  - Commit imported statements before removing a superseded file; a failed commit leaves the source intact.
  - Anything the document still holds that the record lacks is imported and then removed.

## Durability

- A write reaches the `unit` table and is committed before anything derived from it is touched. The rendering and the vectors follow the record, never the other way round.
- A statement with no vector is pending, not lost: a read says so, and the worker embeds it when a model is available. A missing model costs one retry per interval, and the next write or an explicit `reindex` clears the wait.
- A write that drops the vectors of statements the record no longer holds answers with the count as `vectors_removed`. A vector is never the record, and losing one costs a re-embedding rather than a statement. `AGENTS.md` says why the drop happens on the write.
