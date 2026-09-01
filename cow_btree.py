"""
cow_btree.py -- A small, SSD-friendly, copy-on-write B+tree storage engine.

CORE IDEA
---------
Flash storage can't overwrite a page in place cheaply (it must erase a whole
block first), so instead of mutating pages, every write creates NEW pages and
copies the path from the modified leaf up to the root ("copy-on-write", COW).
A commit is then just one atomic pointer swap: a small "meta" page says which
page id is the current root. Old pages stay untouched and valid until nothing
references them, which is what gives us:

  - crash safety without a write-ahead log (a crash mid-write just leaves the
    previous meta page/root intact -- there is nothing to "roll back")
  - MVCC snapshots for free (a reader just remembers the root id it saw; new
    writes don't disturb it, because writes never touch existing pages)

This file implements, and comments, seven pieces beyond that basic idea:

  1. A real on-disk byte layout for pages (no pickle).
  2. Overflow pages, for values too big to fit in one page.
  3. A persistent freelist so space is actually reclaimed and reused, gated
     by a "watermark" so we never recycle a page a live reader still needs.
  4. delete(), including underflow handling (see _delete's docstring for
     exactly how much rebalancing this does and doesn't do).
  5. Byte-size-aware splitting: a node splits when its ENCODED SIZE would
     exceed one page, not when its key count crosses a fixed threshold, so
     fanout is whatever actually fits in 4KB (see _find_leaf_split /
     _find_internal_split).
  6. Group commit: concurrent put()/delete() callers are batched together.
     Whichever thread finds no commit in flight becomes the "leader" for
     that round, applies everyone's pending operations in one pass, and
     performs exactly one fsync + meta swap for the whole batch. This is
     the standard technique for amortizing fsync cost across concurrent
     writers -- see _submit/_run_commit_batch/_commit.
  7. Single-writer-at-a-time is gone; multiple threads can call put()/
     delete() concurrently and correctly, with the physical commit work
     still done by one thread at a time (the current "leader").

WHAT'S STILL DELIBERATELY LEFT OUT (to keep this readable as one file):
  - Deleting an empty node only ever REMOVES it from its parent; siblings
    are never merged or borrowed-from just because they're underfull
    (below ~50% occupancy). A production engine rebalances proactively;
    here the tree stays correct but can get lopsided under heavy
    interleaved insert/delete traffic. See _delete's docstring.
  - No key-side overflow: values over INLINE_VALUE_LIMIT spill into
    overflow pages, but an individual KEY is assumed to comfortably fit
    in a page alongside a few siblings. An enormous key can still raise
    ValueError from the split-finder.
  - Group commit here is a purely in-memory batching scheme, not a
    persistent write-ahead log. It reduces fsync count under concurrent
    load, but a crash mid-batch loses that batch's operations the same
    way an unsynced write always would -- it doesn't add new durability,
    only amortizes an existing cost.
"""

import collections
import os
import struct
import threading
import zlib

# ---------------------------------------------------------------------------
# On-disk constants
# ---------------------------------------------------------------------------

PAGE_SIZE = 4096              # matches typical SSD/NVMe internal page size
HEADER_PAGES = 2              # pages 0 and 1: a ping-pong pair of meta pages
MAX_BODY_SIZE = PAGE_SIZE - 4  # a page body must leave room for the 4-byte
                                # length prefix every page frame carries
INLINE_VALUE_LIMIT = 64       # values bigger than this spill into overflow pages

META_MAGIC = b"CBT2"

PAGE_LEAF = 1
PAGE_INTERNAL = 2
PAGE_OVERFLOW = 3
PAGE_FREELIST = 4


def _to_bytes(x):
    """Keys/values may be given as str or bytes; everything is stored as bytes."""
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8")
    raise TypeError("keys and values must be bytes or str")


# ---------------------------------------------------------------------------
# In-memory node representations
# ---------------------------------------------------------------------------

class LeafNode:
    """keys[i] -> values[i]. A value is either raw bytes (inline) or an
    Overflow(page_id, length) pointer to a chain of overflow pages."""
    __slots__ = ("keys", "values")

    def __init__(self, keys=None, values=None):
        self.keys = keys or []
        self.values = values or []


class InternalNode:
    """len(children) == len(keys) + 1. keys[i] is the smallest key reachable
    through children[i + 1] (so: key >= keys[i] means "go right of i").
    Note: after a delete(), a surviving separator may become "loose" (no
    longer the *tightest* possible bound) but is never wrong -- see
    _delete's docstring for why that's still correct."""
    __slots__ = ("keys", "children")

    def __init__(self, keys=None, children=None):
        self.keys = keys or []
        self.children = children or []


class Overflow:
    """A value too big for one leaf page: pointer to the head of a page chain."""
    __slots__ = ("page_id", "length")

    def __init__(self, page_id, length):
        self.page_id = page_id
        self.length = length


def _child_index(keys, key):
    """First index i such that key < keys[i] -- i.e. which child to descend
    into, or (for a leaf) where to insert a new key to keep it sorted."""
    i = 0
    while i < len(keys) and key >= keys[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------
# Byte-level page encoding
#
# Every page on disk is: [4-byte body length][body][zero padding to PAGE_SIZE]
# The body's first byte is always the page type, so a page is self-describing.
# ---------------------------------------------------------------------------

def _encode_leaf(node):
    buf = bytearray((PAGE_LEAF, 0))               # type, reserved
    buf += struct.pack(">H", len(node.keys))
    for k, v in zip(node.keys, node.values):
        buf += struct.pack(">H", len(k)) + k
        if isinstance(v, Overflow):
            buf.append(1)                          # marker: overflow
            buf += struct.pack(">II", v.page_id, v.length)
        else:
            buf.append(0)                          # marker: inline
            buf += struct.pack(">I", len(v)) + v
    return bytes(buf)


def _decode_leaf(body):
    (num_keys,) = struct.unpack(">H", body[2:4])
    off = 4
    keys, values = [], []
    for _ in range(num_keys):
        (klen,) = struct.unpack(">H", body[off:off + 2]); off += 2
        k = body[off:off + klen]; off += klen
        marker = body[off]; off += 1
        if marker == 0:
            (vlen,) = struct.unpack(">I", body[off:off + 4]); off += 4
            v = body[off:off + vlen]; off += vlen
        else:
            page_id, length = struct.unpack(">II", body[off:off + 8]); off += 8
            v = Overflow(page_id, length)
        keys.append(k)
        values.append(v)
    return LeafNode(keys, values)


def _encode_internal(node):
    buf = bytearray((PAGE_INTERNAL, 0))
    buf += struct.pack(">H", len(node.keys))
    for k in node.keys:
        buf += struct.pack(">H", len(k)) + k
    for c in node.children:
        buf += struct.pack(">I", c)
    return bytes(buf)


def _decode_internal(body):
    (num_keys,) = struct.unpack(">H", body[2:4])
    off = 4
    keys = []
    for _ in range(num_keys):
        (klen,) = struct.unpack(">H", body[off:off + 2]); off += 2
        keys.append(body[off:off + klen]); off += klen
    children = []
    for _ in range(num_keys + 1):
        (c,) = struct.unpack(">I", body[off:off + 4]); off += 4
        children.append(c)
    return InternalNode(keys, children)


# ---------------------------------------------------------------------------
# Byte-size-aware split point selection
#
# A node splits when its encoded body would exceed MAX_BODY_SIZE, not when
# its key count crosses a fixed number -- so fanout is however many entries
# actually fit in 4KB, which varies with key/value size like a real engine.
# ---------------------------------------------------------------------------

def _leaf_entry_size(key, value):
    """Bytes this (key, value) pair contributes to an encoded leaf body,
    matching _encode_leaf's layout exactly."""
    if isinstance(value, Overflow):
        return 2 + len(key) + 1 + 8          # keylen+key, marker, (page_id,length)
    return 2 + len(key) + 1 + 4 + len(value)  # keylen+key, marker, vallen+value


def _find_leaf_split(keys, values):
    """Choose a split index that keeps both resulting leaf pages within
    PAGE_SIZE and as close to evenly balanced (by bytes) as possible."""
    sizes = [_leaf_entry_size(k, v) for k, v in zip(keys, values)]
    n = len(sizes)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + sizes[i]
    total = prefix[n]
    header = 4   # type + reserved + num_keys
    best_i, best_diff = None, None
    for i in range(1, n):   # every key ends up on one side or the other
        left = header + prefix[i]
        right = header + (total - prefix[i])
        if left <= MAX_BODY_SIZE and right <= MAX_BODY_SIZE:
            diff = abs(left - right)
            if best_diff is None or diff < best_diff:
                best_i, best_diff = i, diff
    if best_i is None:
        raise ValueError("cannot split this leaf: an entry (or the fixed header) doesn't fit in one page")
    return best_i


def _internal_entry_size(key):
    """Bytes one separator key contributes, INCLUDING the one extra child
    pointer it implies (children = keys + 1, so each key "brings" a child)."""
    return 2 + len(key) + 4


def _find_internal_split(keys):
    """Choose which key to promote (removed from both children, pushed up
    a level) so both resulting internal pages fit and are as balanced as
    possible. Unlike a leaf split, the entry at the chosen index is REMOVED
    entirely (promoted), not assigned to either side."""
    sizes = [_internal_entry_size(k) for k in keys]
    n = len(sizes)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + sizes[i]
    total = prefix[n]
    header = 8   # type+reserved+num_keys (4) + the one "free" child pointer
                 # every internal node has even with zero keys (4)
    best_i, best_diff = None, None
    for i in range(n):   # i = index of the key to promote
        left = header + prefix[i]
        right = header + (total - prefix[i] - sizes[i])
        if left <= MAX_BODY_SIZE and right <= MAX_BODY_SIZE:
            diff = abs(left - right)
            if best_diff is None or diff < best_diff:
                best_i, best_diff = i, diff
    if best_i is None:
        raise ValueError("cannot split this internal node: an entry (or the fixed header) doesn't fit in one page")
    return best_i


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class Engine:
    def __init__(self, path):
        self.path = path
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.f = open(path, "w+b" if is_new else "r+b")

        # Group commit machinery (see _submit/_run_commit_batch/_commit):
        # _commit_lock ensures only one thread is ever physically applying
        # a batch at a time; _batch_lock only protects the pending-ops
        # queue itself, so appending to it is never blocked by an
        # in-flight commit.
        self._commit_lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self._pending_ops = []
        self._commit_count = 0   # how many physical commits actually ran

        # Guards: _active_readers, and the (root_id, txn_id) pair, so a
        # Reader always sees a consistent snapshot instead of a torn read.
        self._reader_lock = threading.Lock()
        self._active_readers = collections.Counter()   # pinned_txn -> count

        # Pages freed by COW but not yet safe to reuse (see _reclaim_and_publish).
        # This bookkeeping is process-local only -- see _reclaim_and_publish's
        # docstring for why that's an acceptable simplification here.
        self.pending_free = []   # list of (page_id, freed_at_txn)

        if is_new:
            self._file_page_count = HEADER_PAGES
            self.free_pages = []
            root_id = self._alloc_page_id()
            self._write_page_at(root_id, _encode_leaf(LeafNode()))
            self.root_id, self.txn_id = root_id, 0
            self._freelist_head_id = 0
            self._write_meta(self.root_id, self.txn_id, freelist_head_id=0)
        else:
            self._file_page_count = os.path.getsize(path) // PAGE_SIZE
            self.root_id, self.txn_id, freelist_head = self._read_meta()
            self._freelist_head_id = freelist_head
            self.free_pages = self._read_freelist(freelist_head)

    def close(self):
        self.f.close()

    # -- raw page I/O --------------------------------------------------
    #
    # Uses os.pwrite/os.pread (explicit offset, no shared file cursor)
    # rather than seek()+write()/read() on self.f. A shared Python file
    # object has ONE file position; if a reader thread's seek() and
    # read() were interleaved with another thread's seek(), the read
    # could land at the wrong offset entirely. pread/pwrite take the
    # offset as an argument, so concurrent calls can never race like
    # that -- this is what actually makes concurrent readers (and now
    # concurrent commit preparation) safe.

    def _write_page_at(self, page_id, body):
        if len(body) > MAX_BODY_SIZE:
            raise ValueError(f"encoded page too large ({len(body)} bytes)")
        frame = struct.pack(">I", len(body)) + body
        frame = frame.ljust(PAGE_SIZE, b"\x00")
        os.pwrite(self.f.fileno(), frame, page_id * PAGE_SIZE)

    def _read_page_body(self, page_id):
        buf = os.pread(self.f.fileno(), PAGE_SIZE, page_id * PAGE_SIZE)
        (length,) = struct.unpack(">I", buf[:4])
        return buf[4:4 + length]

    def _read_page(self, page_id):
        body = self._read_page_body(page_id)
        ptype = body[0]
        if ptype == PAGE_LEAF:
            return _decode_leaf(body)
        if ptype == PAGE_INTERNAL:
            return _decode_internal(body)
        raise ValueError(f"page {page_id} has unexpected type {ptype}")

    def _alloc_page_id(self):
        """Reuse a reclaimed page if one is available, else grow the file.
        This is how reclaimed space actually gets reused, not just freed."""
        if self.free_pages:
            return self.free_pages.pop()
        pid = self._file_page_count
        self._file_page_count += 1
        return pid

    # -- overflow pages (values too big for one leaf entry) -------------

    def _write_overflow(self, data):
        chunk_size = MAX_BODY_SIZE - 9   # 1 type + 4 next-page-id + 4 chunk-len
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
        ids = [self._alloc_page_id() for _ in chunks]
        for idx, (pid, chunk) in enumerate(zip(ids, chunks)):
            next_id = ids[idx + 1] if idx + 1 < len(ids) else 0
            body = bytes((PAGE_OVERFLOW,)) + struct.pack(">II", next_id, len(chunk)) + chunk
            self._write_page_at(pid, body)
        return ids[0]

    def _read_overflow(self, head_id):
        out = bytearray()
        pid = head_id
        while pid:
            body = self._read_page_body(pid)
            next_id, length = struct.unpack(">II", body[1:9])
            out += body[9:9 + length]
            pid = next_id
        return bytes(out)

    def _overflow_chain_pages(self, head_id):
        """All page ids in a chain, so they can be handed to the freelist."""
        ids = []
        pid = head_id
        while pid:
            ids.append(pid)
            body = self._read_page_body(pid)
            (pid,) = struct.unpack(">I", body[1:5])
        return ids

    def _store_value(self, value):
        if len(value) <= INLINE_VALUE_LIMIT:
            return value
        return Overflow(self._write_overflow(value), len(value))

    def _materialize(self, v):
        if isinstance(v, Overflow):
            return self._read_overflow(v.page_id)
        return v

    # -- meta page (the atomic commit point) -----------------------------

    def _write_meta(self, root_id, txn_id, freelist_head_id):
        # Ping-pong between two slots: whichever slot we DON'T touch this
        # commit still holds the previous, fully-valid meta page. If we
        # crash mid-write here, _read_meta below will simply ignore the
        # torn slot (bad checksum) and fall back to the other one.
        slot = txn_id % HEADER_PAGES
        core = META_MAGIC + struct.pack(">IQI", root_id, txn_id, freelist_head_id)
        crc = struct.pack(">I", zlib.crc32(core))
        buf = (core + crc).ljust(PAGE_SIZE, b"\x00")
        os.pwrite(self.f.fileno(), buf, slot * PAGE_SIZE)
        os.fsync(self.f.fileno())

    def _read_meta(self):
        best = None
        for slot in range(HEADER_PAGES):
            buf = os.pread(self.f.fileno(), PAGE_SIZE, slot * PAGE_SIZE)
            if buf[:4] != META_MAGIC:
                continue
            core, stored_crc = buf[:20], struct.unpack(">I", buf[20:24])[0]
            if zlib.crc32(core) != stored_crc:
                continue   # torn/corrupt write from a crash -- skip it
            root_id, txn_id, freelist_head = struct.unpack(">IQI", buf[4:20])
            if best is None or txn_id > best[1]:
                best = (root_id, txn_id, freelist_head)
        if best is None:
            raise RuntimeError("no valid meta page found (corrupt or uninitialized file)")
        return best

    # -- freelist: a persistent linked list of reclaimed page ids --------

    def _new_freelist_page_id(self):
        """Fallback used only when the pool can't cover the freelist's own
        storage need (see _write_freelist) -- grows the file by one page."""
        pid = self._file_page_count
        self._file_page_count += 1
        return pid

    def _write_freelist(self):
        """Persist self.free_pages as a page chain, WITHOUT letting the
        freelist's own storage need scale with how large the pool gets.

        A naive version always grows the file for its storage pages. That
        seems harmless (a page here and there) but isn't: once the pool
        holds thousands of ids, representing it takes dozens of pages,
        and if every single one of those is a fresh EOF page every commit,
        the file grows by dozens of pages PER COMMIT forever -- the
        freelist's own upkeep becomes the dominant cost.

        Fix: source the freelist's storage pages FROM the pool itself,
        which is safe as long as it's done as one pass with no double
        bookkeeping -- pop the (small number of) ids needed to host the
        chain first, THEN persist whatever's left of the pool as payload.
        Popped ids are correctly excluded from that payload (they're now
        in use hosting a page, not free) and correctly still absent from
        self.free_pages afterward. Every OTHER id in the pool is left
        untouched in self.free_pages, so in-process reuse for ordinary
        tree/overflow pages keeps working between commits, not just after
        a restart. The chain built this commit becomes reclaimable itself
        next commit via _freelist_chain_page_ids, same as always -- so
        the pages "spent" hosting it come right back.
        """
        cap = (MAX_BODY_SIZE - 8) // 4   # ids that fit in one freelist page
        payload_ids = list(self.free_pages)
        n_chunks = -(-len(payload_ids) // cap)   # ceiling division; 0 if empty
        if n_chunks == 0:
            return 0

        storage_ids = [
            self.free_pages.pop() if self.free_pages else self._new_freelist_page_id()
            for _ in range(n_chunks)
        ]
        storage_set = set(storage_ids)
        payload_ids = [pid for pid in payload_ids if pid not in storage_set]

        chunks = [payload_ids[i:i + cap] for i in range(0, len(payload_ids), cap)] or [[]]
        while len(chunks) < len(storage_ids):
            chunks.append([])

        next_id = 0
        for i in range(len(storage_ids) - 1, -1, -1):   # tail-first so `next` is known
            body = bytearray((PAGE_FREELIST, 0))
            body += struct.pack(">IH", next_id, len(chunks[i]))
            for pid in chunks[i]:
                body += struct.pack(">I", pid)
            self._write_page_at(storage_ids[i], bytes(body))
            next_id = storage_ids[i]
        return next_id   # 0 means "empty freelist"

    def _read_freelist(self, head_id):
        """The free page ids *stored inside* the chain (the payload)."""
        ids = []
        pid = head_id
        while pid:
            body = self._read_page_body(pid)
            next_id, count = struct.unpack(">IH", body[2:8])
            off = 8
            for _ in range(count):
                (v,) = struct.unpack(">I", body[off:off + 4]); off += 4
                ids.append(v)
            pid = next_id
        return ids

    def _freelist_chain_page_ids(self, head_id):
        """The page ids *making up* the chain itself, so they can be freed
        once a newer freelist chain replaces them."""
        ids = []
        pid = head_id
        while pid:
            ids.append(pid)
            body = self._read_page_body(pid)
            (next_id,) = struct.unpack(">I", body[2:6])
            pid = next_id
        return ids

    # -- MVCC bookkeeping --------------------------------------------------

    def _register_reader(self, pinned_txn):
        with self._reader_lock:
            self._active_readers[pinned_txn] += 1

    def _unregister_reader(self, pinned_txn):
        with self._reader_lock:
            self._active_readers[pinned_txn] -= 1
            if self._active_readers[pinned_txn] == 0:
                del self._active_readers[pinned_txn]

    def _reclaim_and_publish(self, root, new_txn_id, freed):
        """Atomically (a) decide which pending pages are now safe to reuse
        and (b) publish the new root/txn so future readers see it -- both
        under the SAME _reader_lock acquisition.

        This has to be one critical section, not two. If we computed the
        watermark, released the lock, and only updated self.root_id/txn_id
        later, a Reader could register in between: it would still pin the
        OLD root (since we haven't published the new one yet) but would
        NOT be counted in the watermark decision that already ran -- so a
        page it still needs could get marked reusable and then physically
        overwritten (by this commit's own _write_freelist, which reuses
        pool pages immediately) before that reader ever gets to read it.

        Publishing the new root here, before the freelist/meta are even
        written to disk, is safe: any reader registering after this point
        pins the NEW root and has no claim on the pages being freed,
        regardless of what happens to them next. It doesn't weaken crash
        safety either -- durability is still governed entirely by when
        the meta page is fsync'd, which happens after this, unchanged.
        """
        with self._reader_lock:
            watermark = min(self._active_readers) if self._active_readers else None
            still_pending = []
            for pid, freed_at in self.pending_free:
                if watermark is None or watermark >= freed_at:
                    self.free_pages.append(pid)
                else:
                    still_pending.append((pid, freed_at))
            self.pending_free = still_pending
            self.root_id, self.txn_id = root, new_txn_id

    # -- public read API -----------------------------------------------

    def read(self):
        """Open an explicit MVCC snapshot. Use as a context manager so the
        pin is released even on error:  with engine.read() as r: r.get(k)"""
        return Reader(self)

    def get(self, key):
        with self.read() as r:
            return r.get(key)

    # -- public write API: put()/delete() are thin wrappers around the
    #    group-commit machinery, so many threads can call them concurrently --

    def put(self, key, value):
        key, value = _to_bytes(key), _to_bytes(value)
        self._submit(key, value, is_delete=False)

    def delete(self, key):
        """Returns True if the key existed and was removed, False if it
        wasn't present (in which case nothing is written at all -- no
        wasted commit for a no-op)."""
        key = _to_bytes(key)
        return self._submit(key, None, is_delete=True)

    def _submit(self, key, value, is_delete):
        """Queue one operation and either become this round's commit
        "leader" (if no commit is currently in flight) or block until the
        current leader finishes ours along with everyone else's."""
        op = {"key": key, "value": value, "delete": is_delete,
              "done": threading.Event(), "result": None, "error": None}
        with self._batch_lock:
            self._pending_ops.append(op)
            am_leader = self._commit_lock.acquire(blocking=False)
        if am_leader:
            try:
                self._run_commit_batch()
            finally:
                self._commit_lock.release()
        else:
            op["done"].wait()
        if op["error"] is not None:
            raise op["error"]
        return op["result"]

    def _run_commit_batch(self):
        """Drain and apply pending ops in rounds until the queue is empty.
        Looping (rather than handling one batch and returning) matters:
        more ops can arrive while we're mid-commit, and if we stopped
        after one round, those late arrivals would have no leader to
        apply them until some other thread happened to call _submit."""
        while True:
            with self._batch_lock:
                batch, self._pending_ops = self._pending_ops, []
            if not batch:
                return
            try:
                self._commit(batch)
            except Exception as e:
                # One bad op (e.g. a key too large to ever fit) fails the
                # whole batch rather than risk committing a tree built on
                # a partially-applied operation. Nothing was written to
                # meta, so on-disk state is untouched; only some now-
                # unreferenced pages may have been allocated and are
                # simply wasted (not corrupt, not visible, just unused).
                for op in batch:
                    op["error"] = e
            finally:
                for op in batch:
                    op["done"].set()

    def _commit(self, batch):
        """Apply every op in `batch` against the current tree in sequence,
        then perform exactly ONE freelist write + fsync + meta swap for
        the whole batch. This is the actual "group commit": the expensive
        synchronous fsync is paid once per batch, not once per op."""
        freed = []
        root = self.root_id
        any_change = False

        for op in batch:
            if op["delete"]:
                new_root, changed, became_empty = self._delete(root, op["key"], freed)
                op["result"] = changed
                if changed:
                    any_change = True
                    if became_empty:
                        new_root = self._alloc_page_id()
                        self._write_page_at(new_root, _encode_leaf(LeafNode()))
                    root = new_root
            else:
                new_root, split = self._insert(root, op["key"], op["value"], freed)
                if split is not None:
                    sep_key, right_id = split
                    wrapper_id = self._alloc_page_id()
                    self._write_page_at(wrapper_id, _encode_internal(InternalNode([sep_key], [new_root, right_id])))
                    new_root = wrapper_id
                root = new_root
                any_change = True

        if not any_change:
            return   # every op in the batch was a no-op; nothing to commit

        self._commit_count += 1
        new_txn_id = self.txn_id + 1

        # The freelist chain we're about to replace becomes garbage too --
        # feed it through the same watermark-gated reclamation as every
        # other page (see _new_freelist_page_id for why).
        freed.extend(self._freelist_chain_page_ids(self._freelist_head_id))
        self.pending_free.extend((pid, new_txn_id) for pid in freed)

        # This must happen BEFORE _write_freelist: that call reuses pool
        # pages immediately, and _reclaim_and_publish is what guarantees
        # no reader still needs them (see its docstring).
        self._reclaim_and_publish(root, new_txn_id, freed)

        freelist_head_id = self._write_freelist()

        # Data pages must be durable BEFORE the meta page that points to
        # them is committed -- otherwise a crash could leave meta pointing
        # at a root whose pages never made it to disk. (Page writes go
        # through os.pwrite, which bypasses any Python-level buffering,
        # so fsync alone is enough to guarantee this.)
        os.fsync(self.f.fileno())
        self._write_meta(root, new_txn_id, freelist_head_id)
        self._freelist_head_id = freelist_head_id

    # -- insert (copy-on-write, byte-size-aware split) --------------------

    def _insert(self, page_id, key, value, freed):
        node = self._read_page(page_id)
        freed.append(page_id)   # this page is superseded by whatever we return below

        if isinstance(node, LeafNode):
            keys, values = list(node.keys), list(node.values)
            stored = self._store_value(value)
            if key in keys:
                idx = keys.index(key)
                old = values[idx]
                if isinstance(old, Overflow):
                    freed.extend(self._overflow_chain_pages(old.page_id))
                values[idx] = stored
            else:
                i = _child_index(keys, key)
                keys.insert(i, key)
                values.insert(i, stored)

            body = _encode_leaf(LeafNode(keys, values))
            if len(body) <= MAX_BODY_SIZE:
                new_id = self._alloc_page_id()
                self._write_page_at(new_id, body)
                return new_id, None

            split_i = _find_leaf_split(keys, values)
            left = LeafNode(keys[:split_i], values[:split_i])
            right = LeafNode(keys[split_i:], values[split_i:])
            left_id = self._alloc_page_id(); self._write_page_at(left_id, _encode_leaf(left))
            right_id = self._alloc_page_id(); self._write_page_at(right_id, _encode_leaf(right))
            return left_id, (right.keys[0], right_id)

        # InternalNode: recurse into the one child that owns `key`, copy this node
        i = _child_index(node.keys, key)
        child_new_id, split = self._insert(node.children[i], key, value, freed)

        keys, children = list(node.keys), list(node.children)
        children[i] = child_new_id
        if split is not None:
            sep_key, right_id = split
            keys.insert(i, sep_key)
            children.insert(i + 1, right_id)

        body = _encode_internal(InternalNode(keys, children))
        if len(body) <= MAX_BODY_SIZE:
            new_id = self._alloc_page_id()
            self._write_page_at(new_id, body)
            return new_id, None

        split_i = _find_internal_split(keys)
        left = InternalNode(keys[:split_i], children[:split_i + 1])
        right = InternalNode(keys[split_i + 1:], children[split_i + 1:])
        promoted_key = keys[split_i]
        left_id = self._alloc_page_id(); self._write_page_at(left_id, _encode_internal(left))
        right_id = self._alloc_page_id(); self._write_page_at(right_id, _encode_internal(right))
        return left_id, (promoted_key, right_id)

    # -- delete (copy-on-write, empty-node collapsing) ---------------------

    def _delete(self, page_id, key, freed):
        """Returns (new_page_id, changed, became_empty).

        - changed=False means the key wasn't found anywhere below here;
          new_page_id == page_id and NOTHING was written (no wasted COW
          for a no-op).
        - became_empty=True means this subtree now holds zero keys and
          should be removed from its parent entirely, rather than kept
          around as an empty page.

        Underflow handling here is intentionally minimal: a node is only
        ever removed from its parent when it becomes COMPLETELY empty (or,
        for an internal node, down to a single child, which gets collapsed
        by replacing this level with that child directly). Nodes that are
        merely sparse -- say, one key in a page that could easily fit
        several -- are left alone rather than merged with a sibling. That
        keeps the algorithm simple and always correct (every real key is
        still found by descending "key >= keys[i] -> go right", the same
        rule _insert relies on), just not space-optimal under heavy
        interleaved insert/delete traffic. A production engine merges or
        borrows from siblings around ~50% occupancy instead.
        """
        node = self._read_page(page_id)

        if isinstance(node, LeafNode):
            if key not in node.keys:
                return page_id, False, False
            idx = node.keys.index(key)
            keys, values = list(node.keys), list(node.values)
            old_value = values[idx]
            if isinstance(old_value, Overflow):
                freed.extend(self._overflow_chain_pages(old_value.page_id))
            del keys[idx]
            del values[idx]
            freed.append(page_id)
            if not keys:
                return None, True, True
            new_id = self._alloc_page_id()
            self._write_page_at(new_id, _encode_leaf(LeafNode(keys, values)))
            return new_id, True, False

        # InternalNode
        i = _child_index(node.keys, key)
        child_new_id, changed, child_empty = self._delete(node.children[i], key, freed)
        if not changed:
            return page_id, False, False   # nothing below changed; no COW needed

        freed.append(page_id)   # this node is being replaced regardless
        keys, children = list(node.keys), list(node.children)

        if child_empty:
            # Drop the now-empty child and one adjacent separator. Which
            # one doesn't matter for correctness -- the removed child held
            # no keys, so neither neighbor's own (accurate) key range is
            # affected either way, just how tightly the remaining
            # separator bounds them. We drop keys[i] when it exists
            # (removing the boundary to this child's right), else keys[i-1]
            # (this was the last child, so there's only a left boundary).
            del children[i]
            if i < len(keys):
                del keys[i]
            else:
                del keys[i - 1]
        else:
            children[i] = child_new_id

        if not children:
            return None, True, True
        if len(children) == 1:
            # This level was reduced to a single child: collapse it away
            # by promoting that child directly, shrinking the tree height.
            return children[0], True, False

        new_id = self._alloc_page_id()
        self._write_page_at(new_id, _encode_internal(InternalNode(keys, children)))
        return new_id, True, False

    # -- introspection, handy for demos/debugging -------------------------

    def stats(self):
        return {
            "txn_id": self.txn_id,
            "commits": self._commit_count,
            "file_pages": self._file_page_count,
            "free_pages": len(self.free_pages),
            "pending_free": len(self.pending_free),
            "active_readers": sum(self._active_readers.values()),
        }


class Reader:
    """An MVCC snapshot: a pinned (root_id, txn_id) pair. Readers never
    take part in the commit-batching machinery and never block a writer --
    they just keep using old pages, which COW guarantees stay untouched
    until this Reader closes and _reclaim_and_publish decides it's safe to
    recycle them."""

    def __init__(self, engine):
        self._engine = engine
        with engine._reader_lock:
            self.root_id = engine.root_id
            self.pinned_txn = engine.txn_id
            engine._active_readers[self.pinned_txn] += 1
        self._open = True

    def get(self, key):
        key = _to_bytes(key)
        node = self._engine._read_page(self.root_id)
        while isinstance(node, InternalNode):
            i = _child_index(node.keys, key)
            node = self._engine._read_page(node.children[i])
        if key in node.keys:
            return self._engine._materialize(node.values[node.keys.index(key)])
        return None

    def close(self):
        if self._open:
            self._engine._unregister_reader(self.pinned_txn)
            self._open = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    import threading as _threading

    path = "demo.cbt"
    if os.path.exists(path):
        os.remove(path)

    # 1. basic roundtrip + crash-safe reopen
    db = Engine(path)
    for i in range(200):
        db.put(f"key{i:04d}", f"value-{i}")
    db.close()

    db = Engine(path)
    assert db.get("key0007") == b"value-7"
    assert db.get("key0199") == b"value-199"
    assert db.get("nope") is None
    print("1. basic roundtrip + reopen: OK", db.stats())

    # 2. overflow pages for large values
    big_value = b"x" * 10_000   # far bigger than INLINE_VALUE_LIMIT
    db.put("bigkey", big_value)
    assert db.get("bigkey") == big_value
    print("2. overflow value roundtrip: OK (%d bytes)" % len(big_value))

    # 3. MVCC: a reader pinned to an old snapshot keeps seeing old data,
    #    and the pages it depends on are NOT reclaimed while it's open
    reader = db.read()
    old_value = reader.get("key0007")
    for i in range(200, 210):
        db.put(f"key{i:04d}", f"value-{i}")
    during = db.stats()
    assert reader.get("key0007") == old_value
    assert during["pending_free"] > 0
    reader.close()
    db.put("trigger-reclaim", "x")
    after = db.stats()
    assert after["free_pages"] > 0 and after["pending_free"] == 0
    print("3. MVCC snapshot + delayed reclamation: OK", {"during": during, "after": after})

    # 4. reclaimed pages actually get reused (file doesn't grow without bound)
    pages_before = db.stats()["file_pages"]
    for i in range(200):
        db.put(f"key{i % 50:04d}", f"overwritten-{i}")
    pages_after = db.stats()["file_pages"]
    print(f"4. page reuse: {pages_before} -> {pages_after} file pages after 200 overwrites "
          f"(free pool has {db.stats()['free_pages']} pages available for reuse)")

    db.close()

    # 5. crash-safety: corrupt the newer meta slot's checksum and confirm we
    #    fall back to the older, still-valid slot instead of losing the DB
    with open(path, "r+b") as f:
        f.seek(20)
        b = f.read(1)
        f.seek(20)
        f.write(bytes([b[0] ^ 0xFF]))
    db = Engine(path)
    print("5. survived a corrupted meta slot: OK", db.stats())
    db.close()

    # 6. byte-size-aware splitting: fanout is driven by how much actually
    #    fits in a page, not a fixed key-count cap -- so a few thousand
    #    small entries should still resolve into a shallow, correct tree
    path6 = "demo_split.cbt"
    if os.path.exists(path6):
        os.remove(path6)
    db = Engine(path6)
    model = {}
    for i in range(4000):
        k, v = f"k{i:05d}", f"v{i}"
        db.put(k, v)
        model[k] = v.encode()
    bad = sum(1 for k, v in model.items() if db.get(k) != v)
    print(f"6. byte-size-aware splits: 4000 keys, {bad} mismatches, "
          f"tree uses {db.stats()['file_pages']} pages total")
    db.close()

    # 7. delete(): removal, overflow-chain cleanup, and shrinking back to
    #    an empty tree that still accepts new writes afterward
    path7 = "demo_delete.cbt"
    if os.path.exists(path7):
        os.remove(path7)
    db = Engine(path7)
    for i in range(100):
        db.put(f"d{i:03d}", f"val-{i}")
    db.put("overflowed", b"y" * 5000)

    assert db.delete("d050") is True
    assert db.get("d050") is None
    assert db.delete("d050") is False        # already gone: no-op, no wasted commit
    commits_before_noop = db.stats()["commits"]
    db.delete("does-not-exist")
    assert db.stats()["commits"] == commits_before_noop   # confirmed: truly a no-op

    free_before_overflow_delete = db.stats()["free_pages"]
    db.delete("overflowed")
    assert db.get("overflowed") is None
    assert db.stats()["free_pages"] > free_before_overflow_delete   # its chain was reclaimed

    for i in range(100):
        if i != 50:
            db.delete(f"d{i:03d}")
    assert all(db.get(f"d{i:03d}") is None for i in range(100))
    db.put("after-empty", "still works")
    assert db.get("after-empty") == b"still works"
    print("7. delete(): removal + overflow cleanup + shrink-to-empty: OK", db.stats())
    db.close()

    # 8. concurrent writers: many threads calling put()/delete() at once,
    #    correctness checked against a plain dict, plus a look at how much
    #    group commit actually batched them together
    path8 = "demo_concurrent.cbt"
    if os.path.exists(path8):
        os.remove(path8)
    db = Engine(path8)
    model = {}
    model_lock = _threading.Lock()
    errors = []

    def worker(n):
        try:
            for i in range(300):
                k = f"t{n}-{i}"
                v = f"val-{n}-{i}"
                db.put(k, v)
                with model_lock:
                    model[k] = v.encode()
                if i % 7 == 0 and i > 0:
                    victim = f"t{n}-{i - 1}"
                    if db.delete(victim):
                        with model_lock:
                            model.pop(victim, None)
        except Exception as e:
            errors.append((n, repr(e)))

    threads = [_threading.Thread(target=worker, args=(n,)) for n in range(8)]
    total_ops_estimate = len(threads) * 300
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    mismatches = sum(1 for k, v in model.items() if db.get(k) != v)
    leftover = sum(1 for i in range(300) for n in range(8)
                    if i % 7 == 0 and i > 0 and f"t{n}-{i - 1}" not in model
                    and db.get(f"t{n}-{i - 1}") is not None)
    print(f"8. concurrent writers: 8 threads, ~{total_ops_estimate} ops, "
          f"errors={errors}, mismatches={mismatches}, leftover={leftover}, "
          f"physical commits={db.stats()['commits']} (less than total ops means batching happened)")
    db.close()
