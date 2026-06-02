package identification

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// IsValidAdjustmentSet checks whether adjustmentSet satisfies the back-door
// criterion for estimating the causal effect of treatment on outcome in g.
//
// The back-door criterion requires:
//  1. No variable in adjustmentSet is a descendant of treatment.
//  2. adjustmentSet blocks all back-door paths from treatment to outcome
//     (i.e., d-separates treatment and outcome in the manipulated graph
//     where all edges out of treatment are removed).
func IsValidAdjustmentSet(g *graphgo.DiGraph, treatment, outcome string, adjustmentSet []string) bool {
	// Condition 1: no variable in adjustmentSet is a descendant of treatment.
	descendants := graphgo.Descendants(g, treatment)
	for _, v := range adjustmentSet {
		if descendants[v] {
			return false
		}
	}

	// Condition 2: adjustmentSet d-separates treatment and outcome in the
	// graph with all edges out of treatment removed.
	manipulated := g.Copy()
	for _, child := range g.Successors(treatment) {
		_ = manipulated.RemoveEdge(treatment, child)
	}

	xSet := map[string]bool{treatment: true}
	ySet := map[string]bool{outcome: true}
	zSet := make(map[string]bool, len(adjustmentSet))
	for _, v := range adjustmentSet {
		zSet[v] = true
	}

	return graphgo.DSeparation(manipulated, xSet, ySet, zSet)
}

// GetMinimalAdjustmentSet finds a minimal valid adjustment set for estimating
// the causal effect of treatment on outcome. It starts with the parents of
// treatment, checks validity, and then tries removing variables one at a time.
//
// Returns an error if no valid adjustment set can be found starting from
// the parents of treatment.
func GetMinimalAdjustmentSet(g *graphgo.DiGraph, treatment, outcome string) ([]string, error) {
	parents := g.Parents(treatment)
	sort.Strings(parents)

	if !IsValidAdjustmentSet(g, treatment, outcome, parents) {
		return nil, fmt.Errorf("identification: parents of %s do not form a valid adjustment set", treatment)
	}

	// Greedily try to remove each variable while maintaining validity.
	minimal := make([]string, len(parents))
	copy(minimal, parents)

	for i := 0; i < len(minimal); {
		candidate := make([]string, 0, len(minimal)-1)
		candidate = append(candidate, minimal[:i]...)
		candidate = append(candidate, minimal[i+1:]...)

		if IsValidAdjustmentSet(g, treatment, outcome, candidate) {
			minimal = candidate
			// Don't increment i — the next variable is now at position i.
		} else {
			i++
		}
	}

	return minimal, nil
}

// GetAllAdjustmentSets enumerates all valid adjustment sets from the set of
// non-descendant, non-treatment, non-outcome nodes. This is only feasible
// for small graphs due to exponential enumeration.
func GetAllAdjustmentSets(g *graphgo.DiGraph, treatment, outcome string) [][]string {
	// Collect candidate variables: all nodes except treatment, outcome,
	// and descendants of treatment.
	descendants := graphgo.Descendants(g, treatment)
	var candidates []string
	for _, n := range g.Nodes() {
		if n == treatment || n == outcome {
			continue
		}
		if descendants[n] {
			continue
		}
		candidates = append(candidates, n)
	}
	sort.Strings(candidates)

	var results [][]string
	n := len(candidates)

	// Enumerate all 2^n subsets.
	for mask := 0; mask < (1 << n); mask++ {
		var subset []string
		for i := 0; i < n; i++ {
			if mask&(1<<i) != 0 {
				subset = append(subset, candidates[i])
			}
		}
		if subset == nil {
			subset = []string{}
		}
		if IsValidAdjustmentSet(g, treatment, outcome, subset) {
			results = append(results, subset)
		}
	}

	return results
}
