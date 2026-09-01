# cow_btree.py

A small, single-file, SSD-friendly, copy-on-write B+tree storage engine, written
in pure Python with no dependencies beyond the standard library.

It exists to demonstrate — with real, runnable code — how a modern embedded
storage engine (in the spirit of LMDB) gets crash safety, MVCC snapshots, and
concurrent readers essentially for free from one rule: **never modify a page,
only ever write new ones.**

## Why copy-on-write

Flash storage can't cheaply overwrite a page in place — it has to erase a whole
block first. So instead of mutating pages, every write creates new pages and
copies the path from the modified leaf up to the root ("copy-on-write"). A
commit is then one atomic pointer swap: a small meta page says which page id
is the current root. Old pages stay untouched and valid until nothing
references them, which is what gives this engine:

- **Crash safety without a write-ahead log** — a crash mid-write just leaves
  the previous meta page/root intact.
- **MVCC snapshots for free** — a reader just remembers the root id it saw;
  new writes never disturb it, because writes never touch existing pages.

For the full story — page-by-page, byte-by-byte, with real output from the
engine at every step — see `cow_btree_tutorial.pdf`.

## Features

| Feature | What it means here |
|---|---|
| Real binary page encoding | Hand-packed `struct` layouts for every page type — no pickle |
| Overflow pages | Values over 64 bytes spill into a chained overflow page |
| Persistent freelist | Reclaimed space is tracked in a durable, self-hosting page chain and actually reused |
| Crash safety | Ping-pong meta pages + CRC32; a torn write just falls back to the last good commit |
| MVCC | Lock-free concurrent reads via `Reader` snapshots |
| `delete()` | Full removal with underflow handling (collapses empty nodes, shrinks tree height) |
| Byte-size-aware splitting | A node splits when its *encoded size* exceeds one page, not at a fixed key count |
| Group commit | Concurrent `put()`/`delete()` callers are batched into shared commits, amortizing fsync cost |

## Quick start

```python
from cow_btree import Engine

db = Engine("mydata.cbt")
db.put("key", "value")
db.get("key")          # b'value'
db.delete("key")       # True
db.close()
```

Keys and values may be given as `str` or `bytes`; everything is stored and
returned as `bytes`.

## API

| Call | Description |
|---|---|
| `Engine(path)` | Open or create a database file |
| `.put(key, value)` | Insert or update a key |
| `.get(key)` | Point lookup; `None` if not found |
| `.delete(key) -> bool` | Remove a key; `True` if it existed |
| `.read() -> Reader` | Open an explicit MVCC snapshot (context manager) |
| `.stats() -> dict` | `txn_id`, `commits`, `file_pages`, `free_pages`, `pending_free`, `active_readers` |
| `.close()` | Close the underlying file |

```python
with db.read() as r:      # a consistent snapshot, unaffected by concurrent writes
    r.get("key")
```

## Concurrency model

- **Many readers, always safe, never blocked.** `Reader` snapshots are
  lock-free reads of pages that copy-on-write guarantees won't be mutated
  while referenced.
- **Many writers, safely batched.** Concurrent `put()`/`delete()` calls are
  grouped: whichever thread finds no commit in flight becomes that round's
  leader, applies everyone's pending operations in one pass, and performs one
  fsync + meta swap for the whole batch (group commit).
- **Single process only.** One `Engine` per file, in one OS process. Two
  processes opening the same file is not supported.

## Running the built-in demo

```
python3 cow_btree.py
```

Runs eight self-contained checks against a real database file: basic
roundtrip + crash-safe reopen, overflow values, MVCC snapshot isolation with
delayed reclamation, page reuse, recovery from a corrupted meta page,
byte-size-aware splitting at scale, `delete()` including shrink-to-empty, and
concurrent multi-threaded writers — printing `stats()` at each stage so the
behavior is visible, not just asserted.

## Known limitations

- **No proactive rebalancing on delete.** Only fully-empty nodes are removed
  from their parent; a sparse-but-nonempty node is never merged with a
  sibling. Always correct, not always space-optimal under heavy
  interleaved insert/delete traffic.
- **No key-side overflow.** Only values spill to overflow pages; an
  enormous key can still raise `ValueError` from the split-finder.
- **Group commit is in-memory batching, not a WAL.** It amortizes fsync cost
  under concurrent load; it doesn't add durability beyond what a single
  commit already provides. A crash mid-batch loses that batch's operations,
  same as any unsynced write would.
- **No range scans** — point `get`/`put`/`delete` only.
- **No per-page checksums** on data pages — only the meta page carries a CRC.
- **No compaction.** The file never shrinks; reclaimed pages are reused in
  place, but total file size is a high-water mark.

## File format, in brief

The file is a flat array of fixed 4096-byte pages, addressed as
`page_id * PAGE_SIZE`. Pages 0 and 1 are a ping-pong pair of meta pages (the
only pages ever written in place); every other page is immutable once
written. Every non-meta page is self-describing — its first body byte is a
type tag (`PAGE_LEAF`, `PAGE_INTERNAL`, `PAGE_OVERFLOW`, `PAGE_FREELIST`) — and
is framed as `[4-byte length][body][zero padding to 4096]`, since bodies vary
in size but the on-disk slot doesn't. See the tutorial PDF (lessons 2–3) for
a full byte-by-byte walkthrough.

## Status

This is an educational/demo engine, not a production-hardened one. It has
been exercised with randomized insert/update/delete stress tests checked
against a plain-`dict` model, and with repeated multi-threaded reader/writer
races — but it hasn't had the scrutiny a real embedded database needs
(fuzzing, actual power-loss testing, multi-platform I/O behavior, etc.).
Treat it as a clear illustration of the mechanisms, not a `pip install`-ready
dependency.
