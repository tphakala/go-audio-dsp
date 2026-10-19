package stft

import (
	"testing"
	"testing/synctest"
	"time"
)

// These tests pin that nothing the process-wide edge window cache hands out is
// bound to the testing/synctest bubble that created it (see edgeWindow).

// A window built inside a bubble is reused outside it and in a later bubble, over
// a private cache and over the shared one NewEdgeAnalyzer uses.
func TestEdgeWindowCacheAcrossBubbles(t *testing.T) {
	const n = 512
	shape := dolphShape(100)
	t.Run("private", func(t *testing.T) {
		cache := newEdgeWindowCache(edgeWindowCacheBytes)
		synctest.Test(t, func(t *testing.T) { newEdgeCached(t, n, shape, cache) })
		a := newEdgeCached(t, n, shape, cache)
		sameInt(t, a.windowBuilds, 0, "builds outside the bubble after it built every window")
		synctest.Test(t, func(t *testing.T) {
			a := newEdgeCached(t, n, shape, cache)
			sameInt(t, a.windowBuilds, 0, "builds in a second bubble")
		})
	})
	t.Run("shared", func(t *testing.T) {
		// An attenuation no other test in the package uses, so the bubble builds the
		// entries rather than hitting ones another test left in the shared cache.
		cfg := EdgeConfig{FrameSize: 256, Shape: dolphShape(101.25)}
		synctest.Test(t, func(t *testing.T) {
			_, err := NewEdgeAnalyzer(cfg)
			noErr(t, err)
		})
		_, err := NewEdgeAnalyzer(cfg)
		noErr(t, err)
	})
}

// An analyzer that finds another goroutine's entry in flight waits for it. If
// the owner abandons the entry the analyzer builds the window itself, and if the
// owner publishes it the analyzer returns the owner's window without building.
// This holds with no bubble and whichever side of a bubble the owner and the
// waiter are on. The owner acts only once awaitEdgeFront has seen the waiter's
// lookup, and acts even if it timed out so a stuck waiter is released.
func TestEdgeWindowWaiterRetries(t *testing.T) {
	const n, m = 64, 50
	sides := []struct {
		name                    string
		ownerBubble, waitBubble bool
	}{{"no bubble", false, false}, {"owner in bubble", true, false}, {"waiter in bubble", false, true}}
	for _, side := range sides {
		for _, publish := range []bool{false, true} {
			name := side.name + "/abandoned"
			if publish {
				name = side.name + "/published"
			}
			t.Run(name, func(t *testing.T) {
				cache := newEdgeWindowCache(edgeWindowCacheBytes)
				a := newEdgeCached(t, n, hannShape, cache)
				builds := a.windowBuilds
				got := make(chan *edgeWindow, 1)
				lookup := func() { got <- a.window(m, true) }
				// own acquires the owner's entry, then another so the owner's is not
				// the list front and only the waiter's hit can move it back there.
				own := func(t *testing.T) *edgeWindow {
					t.Helper()
					owner, build := cache.acquire(edgeKey{n: n, shape: hannShape, m: m, drain: true})
					if !build {
						t.Fatal("the owner's acquire is a hit")
					}
					if _, build := cache.acquire(testEdgeKey(n, 7)); !build {
						t.Fatal("the second acquire is a hit")
					}
					return owner
				}
				var owner *edgeWindow
				looked := true
				switch {
				case side.ownerBubble:
					owners := make(chan *edgeWindow, 1)
					seen := make(chan bool, 1)
					go func() {
						o := <-owners
						go lookup()
						seen <- awaitEdgeFront(cache, o)
					}()
					synctest.Test(t, func(t *testing.T) {
						owner = own(t)
						owners <- owner
						looked = <-seen
						finishEdgeOwner(cache, owner, publish)
					})
				case side.waitBubble:
					owner = own(t)
					seen := make(chan bool, 1)
					go func() {
						ok := awaitEdgeFront(cache, owner)
						finishEdgeOwner(cache, owner, publish)
						seen <- ok
					}()
					synctest.Test(t, func(*testing.T) { lookup() })
					looked = <-seen
				default:
					owner = own(t)
					go lookup()
					looked = awaitEdgeFront(cache, owner)
					finishEdgeOwner(cache, owner, publish)
				}
				if !looked {
					t.Fatal("the analyzer never looked the owner's entry up")
				}
				checkEdgeWaiter(t, got, a, owner, builds, publish)
				// Whatever entry the waiter ended with, built inside a bubble or not,
				// is a plain hit for a later lookup outside any bubble.
				after := a.windowBuilds
				if e := a.window(m, true); !e.done {
					t.Error("a later lookup returned an unfinished entry")
				}
				sameInt(t, a.windowBuilds, after, "builds of a later lookup")
			})
		}
	}
}

// finishEdgeOwner publishes the owner's entry with every value 3, or abandons it.
func finishEdgeOwner(c *edgeWindowCache, owner *edgeWindow, publish bool) {
	if publish {
		publishEdgeTest(owner, 3)
		return
	}
	c.abandon(owner)
}

// awaitEdgeFront reports whether e became the front of c's list within 5 s of
// real time; after the owner put another entry in front, only a hit on e moves it
// there. It must run outside any bubble, where its timers are real.
func awaitEdgeFront(c *edgeWindowCache, e *edgeWindow) bool {
	deadline := time.After(5 * time.Second)
	for {
		c.mu.Lock()
		front := c.lru.Front().Value.(*edgeWindow)
		c.mu.Unlock()
		if front == e {
			return true
		}
		select {
		case <-deadline:
			return false
		case <-time.After(time.Millisecond):
		}
	}
}

// checkEdgeWaiter checks the entry a waiting window lookup returned: the owner's
// when it published, a fresh build when it abandoned.
func checkEdgeWaiter(tb testing.TB, got <-chan *edgeWindow, a *EdgeAnalyzer, owner *edgeWindow, builds uint64, publish bool) {
	tb.Helper()
	var e *edgeWindow
	select {
	case e = <-got:
	case <-time.After(5 * time.Second):
		tb.Fatal("the analyzer did not return after the owner finished")
	}
	if !e.done || len(e.w) != a.n {
		tb.Fatalf("window returned an unfinished entry (done %v, %d values)", e.done, len(e.w))
	}
	if publish {
		if e != owner || e.w[0] != 3 {
			tb.Error("the waiter did not return the owner's published window")
		}
		sameInt(tb, a.windowBuilds, builds, "builds when the owner publishes")
		return
	}
	if e == owner {
		tb.Error("the waiter returned the abandoned entry")
	}
	sameInt(tb, a.windowBuilds, builds+1, "builds when the owner abandons")
}
