package inference

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// MinNeighborsOrder returns an elimination ordering for the given variables
// using the min-neighbors (min-degree) heuristic. At each step the variable
// that appears in the fewest remaining factors is chosen for elimination.
//
// This is a greedy heuristic that tends to produce smaller intermediate
// factors, which keeps variable elimination tractable for many practical
// networks.
func MinNeighborsOrder(factorList []*factors.DiscreteFactor, eliminateVars []string) []string {
	if len(eliminateVars) == 0 {
		return nil
	}

	// Build a mutable copy of the variable set to eliminate.
	remaining := make(map[string]bool, len(eliminateVars))
	for _, v := range eliminateVars {
		remaining[v] = true
	}

	// Build a mutable list of variable sets for each factor.
	factorVarSets := make([]map[string]bool, len(factorList))
	for i, f := range factorList {
		factorVarSets[i] = make(map[string]bool)
		for _, v := range f.Variables() {
			factorVarSets[i][v] = true
		}
	}

	order := make([]string, 0, len(eliminateVars))

	for len(remaining) > 0 {
		// Pick the variable appearing in the fewest factor sets.
		bestVar := ""
		bestCount := int(^uint(0) >> 1) // max int

		for v := range remaining {
			count := 0
			for _, vs := range factorVarSets {
				if vs[v] {
					count++
				}
			}
			if count < bestCount {
				bestCount = count
				bestVar = v
			}
		}

		order = append(order, bestVar)
		delete(remaining, bestVar)

		// Simulate elimination: merge all factor sets containing bestVar
		// into one combined set, and remove bestVar from it.
		var merged map[string]bool
		var keepIndices []int

		for i, vs := range factorVarSets {
			if vs[bestVar] {
				if merged == nil {
					merged = make(map[string]bool)
				}
				for v := range vs {
					merged[v] = true
				}
			} else {
				keepIndices = append(keepIndices, i)
			}
		}

		if merged != nil {
			delete(merged, bestVar)
			newFactorVarSets := make([]map[string]bool, 0, len(keepIndices)+1)
			for _, idx := range keepIndices {
				newFactorVarSets = append(newFactorVarSets, factorVarSets[idx])
			}
			newFactorVarSets = append(newFactorVarSets, merged)
			factorVarSets = newFactorVarSets
		}
	}

	return order
}

// buildInteractionGraph builds an adjacency set from the factor variable sets.
// Two variables are neighbors if they appear together in at least one factor.
func buildInteractionGraph(factorVarSets []map[string]bool) map[string]map[string]bool {
	graph := make(map[string]map[string]bool)
	for _, vs := range factorVarSets {
		vars := make([]string, 0, len(vs))
		for v := range vs {
			vars = append(vars, v)
		}
		for i := 0; i < len(vars); i++ {
			if graph[vars[i]] == nil {
				graph[vars[i]] = make(map[string]bool)
			}
			for j := i + 1; j < len(vars); j++ {
				if graph[vars[j]] == nil {
					graph[vars[j]] = make(map[string]bool)
				}
				graph[vars[i]][vars[j]] = true
				graph[vars[j]][vars[i]] = true
			}
		}
	}
	return graph
}

// eliminateFromGraph simulates variable elimination in the interaction graph:
// connect all neighbors of the eliminated variable to each other, then remove
// the variable.
func eliminateFromGraph(graph map[string]map[string]bool, v string) {
	neighbors := graph[v]
	// Add fill edges: connect all pairs of neighbors.
	nList := make([]string, 0, len(neighbors))
	for n := range neighbors {
		nList = append(nList, n)
	}
	for i := 0; i < len(nList); i++ {
		for j := i + 1; j < len(nList); j++ {
			graph[nList[i]][nList[j]] = true
			graph[nList[j]][nList[i]] = true
		}
	}
	// Remove v from all neighbor adjacency lists and delete v's entry.
	for n := range neighbors {
		delete(graph[n], v)
	}
	delete(graph, v)
}

// buildCardinalityMap builds a map from variable name to its cardinality,
// using the first occurrence found in the factor list.
func buildCardinalityMap(factorList []*factors.DiscreteFactor) map[string]int {
	cardMap := make(map[string]int)
	for _, f := range factorList {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			if _, ok := cardMap[v]; !ok {
				cardMap[v] = card[i]
			}
		}
	}
	return cardMap
}

// copyGraph returns a deep copy of an interaction graph.
func copyGraph(graph map[string]map[string]bool) map[string]map[string]bool {
	c := make(map[string]map[string]bool, len(graph))
	for v, neighbors := range graph {
		n := make(map[string]bool, len(neighbors))
		for nb := range neighbors {
			n[nb] = true
		}
		c[v] = n
	}
	return c
}

// initFactorVarSets creates the mutable factor variable sets used by
// several heuristics.
func initFactorVarSets(factorList []*factors.DiscreteFactor) []map[string]bool {
	sets := make([]map[string]bool, len(factorList))
	for i, f := range factorList {
		sets[i] = make(map[string]bool)
		for _, v := range f.Variables() {
			sets[i][v] = true
		}
	}
	return sets
}

// MinFillOrder returns an elimination ordering using the min-fill heuristic.
// At each step the variable whose elimination requires the fewest fill edges
// (edges added to the interaction graph to connect its non-adjacent neighbors)
// is chosen.
func MinFillOrder(factorList []*factors.DiscreteFactor, eliminateVars []string) []string {
	if len(eliminateVars) == 0 {
		return nil
	}

	remaining := make(map[string]bool, len(eliminateVars))
	for _, v := range eliminateVars {
		remaining[v] = true
	}

	factorVarSets := initFactorVarSets(factorList)
	graph := buildInteractionGraph(factorVarSets)

	order := make([]string, 0, len(eliminateVars))

	for len(remaining) > 0 {
		bestVar := ""
		bestFill := int(^uint(0) >> 1)

		for v := range remaining {
			neighbors := graph[v]
			nList := make([]string, 0, len(neighbors))
			for n := range neighbors {
				nList = append(nList, n)
			}
			fill := 0
			for i := 0; i < len(nList); i++ {
				for j := i + 1; j < len(nList); j++ {
					if !graph[nList[i]][nList[j]] {
						fill++
					}
				}
			}
			if fill < bestFill {
				bestFill = fill
				bestVar = v
			}
		}

		order = append(order, bestVar)
		delete(remaining, bestVar)
		eliminateFromGraph(graph, bestVar)
	}

	return order
}

// MinWeightOrder returns an elimination ordering using the min-weight
// heuristic. The "weight" of eliminating a variable is the product of
// cardinalities of all variables that would appear in the resulting factor
// (i.e., the union of variable scopes of all factors containing that
// variable, excluding the variable itself). At each step the variable with
// the smallest weight is eliminated.
func MinWeightOrder(factorList []*factors.DiscreteFactor, eliminateVars []string) []string {
	if len(eliminateVars) == 0 {
		return nil
	}

	remaining := make(map[string]bool, len(eliminateVars))
	for _, v := range eliminateVars {
		remaining[v] = true
	}

	cardMap := buildCardinalityMap(factorList)
	factorVarSets := initFactorVarSets(factorList)
	graph := buildInteractionGraph(factorVarSets)

	order := make([]string, 0, len(eliminateVars))

	for len(remaining) > 0 {
		bestVar := ""
		bestWeight := -1.0

		for v := range remaining {
			// The resulting factor's scope is all neighbors of v in the
			// interaction graph (v itself is marginalized out).
			weight := 1.0
			for n := range graph[v] {
				weight *= float64(cardMap[n])
			}
			if bestWeight < 0 || weight < bestWeight {
				bestWeight = weight
				bestVar = v
			}
		}

		order = append(order, bestVar)
		delete(remaining, bestVar)
		eliminateFromGraph(graph, bestVar)
	}

	return order
}

// WeightedMinFillOrder returns an elimination ordering using the weighted
// min-fill heuristic. Like MinFill, it counts fill edges needed per
// candidate variable, but each fill edge is weighted by the product of the
// cardinalities of the two variables it would connect.
func WeightedMinFillOrder(factorList []*factors.DiscreteFactor, eliminateVars []string) []string {
	if len(eliminateVars) == 0 {
		return nil
	}

	remaining := make(map[string]bool, len(eliminateVars))
	for _, v := range eliminateVars {
		remaining[v] = true
	}

	cardMap := buildCardinalityMap(factorList)
	factorVarSets := initFactorVarSets(factorList)
	graph := buildInteractionGraph(factorVarSets)

	order := make([]string, 0, len(eliminateVars))

	for len(remaining) > 0 {
		bestVar := ""
		bestCost := -1.0

		for v := range remaining {
			neighbors := graph[v]
			nList := make([]string, 0, len(neighbors))
			for n := range neighbors {
				nList = append(nList, n)
			}
			cost := 0.0
			for i := 0; i < len(nList); i++ {
				for j := i + 1; j < len(nList); j++ {
					if !graph[nList[i]][nList[j]] {
						cost += float64(cardMap[nList[i]]) * float64(cardMap[nList[j]])
					}
				}
			}
			if bestCost < 0 || cost < bestCost {
				bestCost = cost
				bestVar = v
			}
		}

		order = append(order, bestVar)
		delete(remaining, bestVar)
		eliminateFromGraph(graph, bestVar)
	}

	return order
}

// GetEliminationOrder dispatches to the elimination-order heuristic
// identified by name. Supported heuristics:
//   - "min_neighbors" (min-degree)
//   - "min_fill"
//   - "min_weight"
//   - "weighted_min_fill"
//
// Returns an error for an unrecognized heuristic name.
func GetEliminationOrder(factorList []*factors.DiscreteFactor, eliminateVars []string, heuristic string) ([]string, error) {
	switch heuristic {
	case "min_neighbors":
		return MinNeighborsOrder(factorList, eliminateVars), nil
	case "min_fill":
		return MinFillOrder(factorList, eliminateVars), nil
	case "min_weight":
		return MinWeightOrder(factorList, eliminateVars), nil
	case "weighted_min_fill":
		return WeightedMinFillOrder(factorList, eliminateVars), nil
	default:
		return nil, fmt.Errorf("inference: unknown elimination heuristic %q", heuristic)
	}
}
