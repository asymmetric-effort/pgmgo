package identification

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// IsValidFrontdoorSet checks whether frontdoorSet satisfies the front-door
// criterion for estimating the causal effect of treatment on outcome in g.
//
// The front-door criterion requires:
//  1. Treatment intercepts all directed paths from treatment to outcome
//     (every directed path from treatment to outcome passes through at least
//     one variable in frontdoorSet).
//  2. No unblocked back-door path from treatment to any variable in frontdoorSet.
//  3. All back-door paths from each variable in frontdoorSet to outcome are
//     blocked by treatment.
func IsValidFrontdoorSet(g *graphgo.DiGraph, treatment, outcome string, frontdoorSet []string) bool {
	if len(frontdoorSet) == 0 {
		return false
	}

	fdSet := make(map[string]bool, len(frontdoorSet))
	for _, v := range frontdoorSet {
		fdSet[v] = true
	}

	// Condition 1: frontdoorSet intercepts all directed paths from treatment to outcome.
	if !interceptsAllDirectedPaths(g, treatment, outcome, fdSet) {
		return false
	}

	// Condition 2: no unblocked back-door path from treatment to any variable
	// in frontdoorSet. This means treatment and each front-door variable are
	// d-separated in the graph with edges out of treatment removed, given
	// the empty set — OR equivalently, there is no back-door path. We check
	// d-separation with empty conditioning in the manipulated graph.
	manipulated := g.Copy()
	for _, child := range g.Successors(treatment) {
		_ = manipulated.RemoveEdge(treatment, child)
	}

	xSet := map[string]bool{treatment: true}
	for _, m := range frontdoorSet {
		ySet := map[string]bool{m: true}
		if !graphgo.DSeparation(manipulated, xSet, ySet, map[string]bool{}) {
			return false
		}
	}

	// Condition 3: all back-door paths from each front-door variable to outcome
	// are blocked by {treatment}. For each variable m in frontdoorSet, remove
	// edges out of m, and check d-separation of m and outcome given {treatment}.
	treatmentSet := map[string]bool{treatment: true}
	for _, m := range frontdoorSet {
		manipulatedM := g.Copy()
		for _, child := range g.Successors(m) {
			_ = manipulatedM.RemoveEdge(m, child)
		}
		mSet := map[string]bool{m: true}
		ySet := map[string]bool{outcome: true}
		if !graphgo.DSeparation(manipulatedM, mSet, ySet, treatmentSet) {
			return false
		}
	}

	return true
}

// GetFrontdoorSet finds a valid front-door set for estimating the causal
// effect of treatment on outcome. It searches among mediators (nodes on
// directed paths from treatment to outcome).
//
// Returns an error if no valid front-door set can be found.
func GetFrontdoorSet(g *graphgo.DiGraph, treatment, outcome string) ([]string, error) {
	// Find all nodes on directed paths from treatment to outcome.
	mediators := findMediators(g, treatment, outcome)
	sort.Strings(mediators)

	if len(mediators) == 0 {
		return nil, fmt.Errorf("identification: no mediators found between %s and %s", treatment, outcome)
	}

	// Try the full set of mediators first.
	if IsValidFrontdoorSet(g, treatment, outcome, mediators) {
		// Try to minimize.
		minimal := make([]string, len(mediators))
		copy(minimal, mediators)
		for i := 0; i < len(minimal); {
			candidate := make([]string, 0, len(minimal)-1)
			candidate = append(candidate, minimal[:i]...)
			candidate = append(candidate, minimal[i+1:]...)
			if len(candidate) > 0 && IsValidFrontdoorSet(g, treatment, outcome, candidate) {
				minimal = candidate
			} else {
				i++
			}
		}
		return minimal, nil
	}

	// Try subsets of mediators from largest to smallest.
	n := len(mediators)
	for size := n - 1; size >= 1; size-- {
		subsets := combinations(mediators, size)
		for _, subset := range subsets {
			if IsValidFrontdoorSet(g, treatment, outcome, subset) {
				return subset, nil
			}
		}
	}

	return nil, fmt.Errorf("identification: no valid front-door set found for effect of %s on %s", treatment, outcome)
}

// interceptsAllDirectedPaths checks whether every directed path from src to
// dst in g passes through at least one node in the interceptSet.
func interceptsAllDirectedPaths(g *graphgo.DiGraph, src, dst string, interceptSet map[string]bool) bool {
	// DFS from src to dst, but stop exploring when hitting an intercept node.
	// If we reach dst without hitting an intercept node, return false.
	visited := make(map[string]bool)

	var dfs func(string) bool
	dfs = func(node string) bool {
		if node == dst {
			return true // Found an unintercepted path.
		}
		visited[node] = true
		for _, child := range g.Successors(node) {
			if visited[child] {
				continue
			}
			// If child is in intercept set, don't continue through it.
			if interceptSet[child] {
				continue
			}
			if dfs(child) {
				return true // Found an unintercepted path.
			}
		}
		return false
	}

	// Start DFS but skip through treatment itself (we already know src is treatment).
	visited[src] = true
	for _, child := range g.Successors(src) {
		if interceptSet[child] {
			continue
		}
		if dfs(child) {
			return false
		}
	}
	return true
}

// findMediators returns all nodes that lie on at least one directed path from
// src to dst (excluding src and dst themselves).
func findMediators(g *graphgo.DiGraph, src, dst string) []string {
	// A node is a mediator if it is both a descendant of src and an ancestor of dst.
	descSrc := graphgo.Descendants(g, src)
	ancDst := graphgo.Ancestors(g, dst)

	var mediators []string
	for n := range descSrc {
		if n == src || n == dst {
			continue
		}
		if ancDst[n] {
			mediators = append(mediators, n)
		}
	}
	return mediators
}

// combinations generates all combinations of size k from the given slice.
func combinations(items []string, k int) [][]string {
	var result [][]string
	var combo func(start int, current []string)
	combo = func(start int, current []string) {
		if len(current) == k {
			c := make([]string, k)
			copy(c, current)
			result = append(result, c)
			return
		}
		for i := start; i < len(items); i++ {
			combo(i+1, append(current, items[i]))
		}
	}
	combo(0, nil)
	return result
}
