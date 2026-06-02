package learning

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// MMHC implements the Max-Min Hill Climbing algorithm for Bayesian network
// structure learning. It combines a constraint-based phase (MMPC) with a
// score-based phase (hill climbing).
type MMHC struct {
	data         *tabgo.DataFrame
	scoreFn      ScoreFunc
	ciTest       CITestFunc
	significance float64
}

// NewMMHC creates a new MMHC instance.
//
// Parameters:
//   - data: the observational dataset
//   - scoreFn: scoring function for the hill climbing phase
//   - ciTest: conditional independence test function for the MMPC phase
//   - significance: significance level for CI tests
func NewMMHC(data *tabgo.DataFrame, scoreFn ScoreFunc, ciTest CITestFunc, significance float64) *MMHC {
	return &MMHC{
		data:         data,
		scoreFn:      scoreFn,
		ciTest:       ciTest,
		significance: significance,
	}
}

// Estimate runs the MMHC algorithm and returns the learned Bayesian network.
//
// The algorithm has two phases:
//  1. MMPC phase: For each variable, find candidate parents/children using the
//     max-min heuristic with conditional independence tests. This produces an
//     undirected skeleton of candidate edges.
//  2. Hill climbing phase: Restrict the search space to the candidate edges
//     found in the MMPC phase, then use greedy hill climbing to find the best
//     DAG within that space.
func (m *MMHC) Estimate() (*models.BayesianNetwork, error) {
	columns := m.data.Columns()
	if len(columns) < 2 {
		return nil, fmt.Errorf("learning: MMHC requires at least 2 variables, got %d", len(columns))
	}

	sort.Strings(columns)

	// Phase 1: MMPC — find candidate parents/children for each variable.
	candidatePC := m.mmpcPhase(columns)

	// Build the set of candidate edges (symmetrized).
	candidateEdges := make(map[[2]string]bool)
	for x, candidates := range candidatePC {
		for _, y := range candidates {
			candidateEdges[[2]string{x, y}] = true
			candidateEdges[[2]string{y, x}] = true
		}
	}

	// Phase 2: Hill climbing restricted to candidate edges.
	// Build a blacklist of all non-candidate edges.
	var blackList [][2]string
	for _, u := range columns {
		for _, v := range columns {
			if u == v {
				continue
			}
			if !candidateEdges[[2]string{u, v}] {
				blackList = append(blackList, [2]string{u, v})
			}
		}
	}

	hc := NewHillClimbSearch(m.data, m.scoreFn, WithBlackList(blackList))
	return hc.Estimate()
}

// mmpcPhase runs the MMPC (Max-Min Parents and Children) algorithm for each
// variable. Returns a map from variable to its candidate parent/child set.
func (m *MMHC) mmpcPhase(columns []string) map[string][]string {
	candidatePC := make(map[string][]string)

	for _, target := range columns {
		candidates := m.mmpc(target, columns)
		candidatePC[target] = candidates
	}

	return candidatePC
}

// mmpc runs the Max-Min Parents and Children algorithm for a single target
// variable. It iteratively adds the variable that maximizes the minimum
// association (conditional on already selected variables), subject to
// conditional independence testing.
func (m *MMHC) mmpc(target string, allVars []string) []string {
	// Candidate parent/child set.
	var cpc []string
	cpcSet := make(map[string]bool)

	// Remaining variables (excluding target).
	remaining := make(map[string]bool)
	for _, v := range allVars {
		if v != target {
			remaining[v] = true
		}
	}

	// Forward phase: grow the CPC set.
	for len(remaining) > 0 {
		bestVar := ""
		bestMinAssoc := -1.0

		for v := range remaining {
			// Compute min association of v with target given subsets of CPC.
			minAssoc := m.minAssociation(target, v, cpc)
			if minAssoc > bestMinAssoc {
				bestMinAssoc = minAssoc
				bestVar = v
			}
		}

		if bestVar == "" || bestMinAssoc <= 0 {
			break
		}

		// Check if the best variable is conditionally independent of target
		// given CPC.
		_, _, indep := m.ciTest(target, bestVar, cpc, m.data, m.significance)
		if indep {
			// Remove it from consideration; it's independent.
			delete(remaining, bestVar)
			continue
		}

		// Add to CPC.
		cpc = append(cpc, bestVar)
		cpcSet[bestVar] = true
		delete(remaining, bestVar)
	}

	// Backward phase: prune false positives.
	pruned := make([]string, 0, len(cpc))
	for _, v := range cpc {
		// Check if v is independent of target given CPC \ {v}.
		condSet := make([]string, 0, len(cpc)-1)
		for _, u := range cpc {
			if u != v {
				condSet = append(condSet, u)
			}
		}

		_, _, indep := m.ciTest(target, v, condSet, m.data, m.significance)
		if !indep {
			pruned = append(pruned, v)
		}
	}

	_ = cpcSet
	sort.Strings(pruned)
	return pruned
}

// minAssociation computes the minimum association between target and candidate
// variable x, conditioning on subsets of the current CPC. This implements the
// "max-min" heuristic: we look at the minimum p-value (used as a proxy for
// association strength) over conditioning subsets. A higher value means stronger
// association.
func (m *MMHC) minAssociation(target, x string, cpc []string) float64 {
	if len(cpc) == 0 {
		// No conditioning set: just test marginal association.
		stat, _, _ := m.ciTest(target, x, nil, m.data, m.significance)
		return stat
	}

	// Test conditioning on each subset of CPC of size 0..len(cpc).
	// For efficiency, we limit to subsets up to a reasonable size.
	maxSubsetSize := len(cpc)
	if maxSubsetSize > 3 {
		maxSubsetSize = 3
	}

	minStat := -1.0
	first := true

	for d := 0; d <= maxSubsetSize; d++ {
		subsets := combinations(cpc, d)
		for _, subset := range subsets {
			stat, _, _ := m.ciTest(target, x, subset, m.data, m.significance)
			if first || stat < minStat {
				minStat = stat
				first = false
			}
		}
	}

	return minStat
}
