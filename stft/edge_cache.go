package stft

import (
	"container/list"
	"sync"
)

// edgeWindowCacheBytes bounds the window data the process-wide edge window cache
// keeps: each entry counts 4*FrameSize bytes (float32 values). The map slot, list
// element, entry struct and ready channel are not counted (NOT MEASURED), so the
// heap held is above this figure.
// One full set of edge windows for a geometry is at most 2*edgeCacheWindows+1
// windows: 1.06 MB at FrameSize 4096, 2.1 MB at 8192, 8.5 MB at the Dolph-Chebyshev
// cap of 32768. 16 MiB holds several sets at the common sizes and stays small on
// a single-board host. Windows an analyzer still references outlive their
// eviction until the analyzer drops them.
const edgeWindowCacheBytes = 16 << 20

// edgeKey identifies one edge window. buildWindow's output depends on exactly
// these four values and nothing else: FrameSize and the shape (family and
// attenuation) fix the scale and the points, m is the symmetric window length,
// and drain selects the placement and point count (m points at slot 0 for a drain
// window, m-1 points at slot N-m+1 for a lead window). Hop, drain zero count and
// clip length do not enter, which is why windows are shared across geometries.
// The full window is the lead rule at m = N+1, a length no edge window reaches.
type edgeKey struct {
	n     int
	shape WindowShape
	m     int
	drain bool
}

// edgeWindow is one cache entry. Its fields other than key and elem are written
// by the single goroutine that built it and published by closing ready; after
// that nothing writes them, so holders read w lock-free.
type edgeWindow struct {
	w     []float32     // length n
	sum   float64       // buildWindow's sum; the full window's sets FullWindowSum
	ok    bool          // buildWindow's ok
	done  bool          // true once published; false after ready closes means the build was abandoned
	ready chan struct{} // closed by the builder after publishing or abandoning
	key   edgeKey
	elem  *list.Element // guarded by edgeWindowCache.mu; nil once evicted or abandoned
}

// edgeWindowCache is a byte-bounded LRU of immutable edge windows with one build
// per key at a time. mu guards the map, the list, the byte count and every
// entry's elem; it is never held while a window is built.
type edgeWindowCache struct {
	mu       sync.Mutex
	maxBytes int
	bytes    int
	byKey    map[edgeKey]*edgeWindow
	lru      list.List // front is most recently used; Value is *edgeWindow
}

// sharedEdgeWindows is the cache NewEdgeAnalyzer uses; tests build private ones.
var sharedEdgeWindows = newEdgeWindowCache(edgeWindowCacheBytes)

func newEdgeWindowCache(maxBytes int) *edgeWindowCache {
	return &edgeWindowCache{maxBytes: maxBytes, byKey: make(map[edgeKey]*edgeWindow)}
}

// acquire returns the entry for k. When build is true the entry is new and the
// caller must fill and publish it, or abandon it. When build is false the entry
// exists: wait on e.ready, then use it if e.done, else acquire again (its builder
// gave up). The hit path allocates nothing.
func (c *edgeWindowCache) acquire(k edgeKey) (e *edgeWindow, build bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if e, ok := c.byKey[k]; ok {
		c.lru.MoveToFront(e.elem)
		return e, false
	}
	e = &edgeWindow{key: k, ready: make(chan struct{})}
	e.elem = c.lru.PushFront(e)
	c.byKey[k] = e
	c.bytes += 4 * k.n
	// Evict from the least recently used end, always keeping the entry just added
	// so its builder has a slot even when one window alone exceeds the budget.
	for c.bytes > c.maxBytes && c.lru.Len() > 1 {
		victim := c.lru.Back().Value.(*edgeWindow)
		delete(c.byKey, victim.key)
		c.lru.Remove(victim.elem)
		victim.elem = nil
		c.bytes -= 4 * victim.key.n
	}
	return e, true
}

// abandon drops an entry whose builder is giving up (a panic mid-build) and
// wakes its waiters, which then see done == false and acquire again. The entry may
// already have been evicted, and another goroutine may have since acquired the
// same key as a new entry, so it is removed only if the map still holds e itself.
func (c *edgeWindowCache) abandon(e *edgeWindow) {
	c.mu.Lock()
	if c.byKey[e.key] == e {
		delete(c.byKey, e.key)
		c.lru.Remove(e.elem)
		e.elem = nil
		c.bytes -= 4 * e.key.n
	}
	c.mu.Unlock()
	close(e.ready)
}
