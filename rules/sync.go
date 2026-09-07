//go:build ruleguard

package gorules

import "github.com/quasilyte/go-ruleguard/dsl"

// WaitGroupGo detects the old sync.WaitGroup pattern and suggests using Go 1.25's wg.Go().
//
// The old pattern:
//
//	wg.Add(1)
//	go func() {
//	    defer wg.Done()
//	    doSomething()
//	}()
//
// Can be simplified to:
//
//	wg.Go(func() {
//	    doSomething()
//	})
//
// Benefits:
//   - Cleaner, less error-prone (no Add/Done mismatch)
//   - Single function call
//
// Note: wg.Go still calls Done on panic (it is deferred), but it does not
// recover the panic; a panicking task crashes the program just as before.
//
// See: https://pkg.go.dev/sync#WaitGroup.Go
//
// NOTE: the goroutine body is captured as $*body (a NodeSlice). Do NOT interpolate
// it into Report or Suggest: go-ruleguard renders both templates via go/printer,
// which panics on *gogrep.NodeSlice ("unsupported node type") and crashes gocritic
// on any file that matches. That is why these rules interpolate only the
// single-node $wg (never the $*body NodeSlice) and provide no Suggest quickfix.
func WaitGroupGo(m dsl.Matcher) {
	// Pattern 1: wg.Add(1) followed by go func() with defer wg.Done()
	// This matches when the defer is the first statement
	m.Match(
		`$wg.Add(1); go func() { defer $wg.Done(); $*body }()`,
	).
		Where(m["wg"].Type.Is("*sync.WaitGroup") || m["wg"].Type.Is("sync.WaitGroup")).
		Report("use $wg.Go(func() { ... }) instead of the manual Add/Done goroutine pattern (Go 1.25+)")

	// Pattern 2: same match, but for a named type whose underlying type is
	// sync.WaitGroup
	m.Match(
		`$wg.Add(1); go func() { defer $wg.Done(); $*body }()`,
	).
		Where(m["wg"].Type.Underlying().Is("sync.WaitGroup")).
		Report("use $wg.Go(func() { ... }) instead of the manual Add/Done goroutine pattern (Go 1.25+)")

	// Pattern 3: When wg is passed by reference to the closure
	m.Match(
		`$wg.Add(1); go func($param $typ) { defer $param.Done(); $*body }($wg)`,
		`$wg.Add(1); go func($param $typ) { defer $param.Done(); $*body }(&$wg)`,
	).
		Where(m["wg"].Type.Is("*sync.WaitGroup") || m["wg"].Type.Is("sync.WaitGroup")).
		Report("use $wg.Go(func() { ... }) instead of the manual Add/Done goroutine pattern (Go 1.25+)")
}
