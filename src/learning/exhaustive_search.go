package learning

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// maxExhaustiveVars is the maximum number of variables allowed for exhaustive
// search. The number of possible DAGs grows super-exponentially, so this is
// only feasible for very small networks.
const maxExhaustiveVars = 4

// ExhaustiveSearch performs brute-force structure learning by enumerating all
// possible DAGs over the variables, scoring each, and returning the best.
// Only feasible for up to 4 variables.
type ExhaustiveSearch struct {
	data    *tabgo.DataFrame
	scoreFn ScoreFunc
}

// NewExhaustiveSearch creates a new ExhaustiveSearch instance.
func NewExhaustiveSearch(data *tabgo.DataFrame, scoreFn ScoreFunc) *ExhaustiveSearch {
	return &ExhaustiveSearch{
		data:    data,
		scoreFn: scoreFn,
	}
}

// Estimate enumerates all DAGs over the variables in the data, scores each,
// and returns the BayesianNetwork with the highest score. Returns an error if
// there are more than maxExhaustiveVars variables.
func (es *ExhaustiveSearch) Estimate() (*models.BayesianNetwork, error) {
	columns := es.data.Columns()
	if len(columns) == 0 {
		return nil, fmt.Errorf("learning: exhaustive search requires at least one column")
	}
	if len(columns) > maxExhaustiveVars {
		return nil, fmt.Errorf("learning: exhaustive search supports at most %d variables, got %d", maxExhaustiveVars, len(columns))
	}

	sort.Strings(columns)

	allDAGs := enumerateDAGs(columns)

	bestScore := math.Inf(-1)
	var bestEdges [][2]string

	for _, edges := range allDAGs {
		score := es.scoreDAG(columns, edges)
		if score > bestScore {
			bestScore = score
			bestEdges = edges
		}
	}

	bn := models.NewBayesianNetwork()
	for _, col := range columns {
		if err := bn.AddNode(col); err != nil {
			return nil, fmt.Errorf("learning: %w", err)
		}
	}
	for _, e := range bestEdges {
		if err := bn.AddEdge(e[0], e[1]); err != nil {
			return nil, fmt.Errorf("learning: %w", err)
		}
	}

	return bn, nil
}

// AllScores returns a map from DAG description (edge list string) to its total
// score for every possible DAG over the data's variables. Returns an error if
// there are more than maxExhaustiveVars variables.
func (es *ExhaustiveSearch) AllScores() (map[string]float64, error) {
	columns := es.data.Columns()
	if len(columns) == 0 {
		return nil, fmt.Errorf("learning: exhaustive search requires at least one column")
	}
	if len(columns) > maxExhaustiveVars {
		return nil, fmt.Errorf("learning: exhaustive search supports at most %d variables, got %d", maxExhaustiveVars, len(columns))
	}

	sort.Strings(columns)
	allDAGs := enumerateDAGs(columns)

	scores := make(map[string]float64, len(allDAGs))
	for _, edges := range allDAGs {
		key := dagKey(edges)
		score := es.scoreDAG(columns, edges)
		scores[key] = score
	}
	return scores, nil
}

// scoreDAG computes the total score for a DAG defined by the given edge set.
func (es *ExhaustiveSearch) scoreDAG(columns []string, edges [][2]string) float64 {
	// Build parent map.
	parentMap := make(map[string][]string)
	for _, col := range columns {
		parentMap[col] = nil
	}
	for _, e := range edges {
		parentMap[e[1]] = append(parentMap[e[1]], e[0])
	}

	total := 0.0
	for _, col := range columns {
		parents := parentMap[col]
		sort.Strings(parents)
		total += es.scoreFn(col, parents, es.data)
	}
	return total
}

// dagKey returns a canonical string representation of a DAG's edge list.
func dagKey(edges [][2]string) string {
	if len(edges) == 0 {
		return "[]"
	}
	parts := make([]string, len(edges))
	for i, e := range edges {
		parts[i] = e[0] + "->" + e[1]
	}
	sort.Strings(parts)
	return "[" + strings.Join(parts, ",") + "]"
}

// enumerateDAGs generates all possible DAGs over the given variable names.
// Each DAG is represented as a sorted list of directed edges.
//
// Strategy: for n variables, there are n*(n-1) possible directed edges. We
// enumerate all 2^(n*(n-1)) subsets and keep only those that form a DAG.
func enumerateDAGs(vars []string) [][][2]string {
	n := len(vars)
	if n == 0 {
		return [][][2]string{{}}
	}

	// Build the list of all possible directed edges.
	var allEdges [][2]string
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				allEdges = append(allEdges, [2]string{vars[i], vars[j]})
			}
		}
	}

	numEdges := len(allEdges)
	var result [][][2]string

	// Enumerate all subsets of edges.
	total := 1 << numEdges
	for mask := 0; mask < total; mask++ {
		var edges [][2]string
		for b := 0; b < numEdges; b++ {
			if mask&(1<<b) != 0 {
				edges = append(edges, allEdges[b])
			}
		}

		// Check for mutual edges (both u->v and v->u).
		if hasMutualEdge(edges) {
			continue
		}

		// Check if it's a DAG.
		if isDAGFromEdges(vars, edges) {
			// Sort edges for canonical form.
			sorted := make([][2]string, len(edges))
			copy(sorted, edges)
			sort.Slice(sorted, func(i, j int) bool {
				if sorted[i][0] != sorted[j][0] {
					return sorted[i][0] < sorted[j][0]
				}
				return sorted[i][1] < sorted[j][1]
			})
			result = append(result, sorted)
		}
	}

	return result
}

// hasMutualEdge returns true if the edge set contains both u->v and v->u.
func hasMutualEdge(edges [][2]string) bool {
	set := make(map[[2]string]bool, len(edges))
	for _, e := range edges {
		set[e] = true
	}
	for _, e := range edges {
		if set[[2]string{e[1], e[0]}] {
			return true
		}
	}
	return false
}

// isDAGFromEdges checks whether the given edges over the given variables form a DAG.
func isDAGFromEdges(vars []string, edges [][2]string) bool {
	g := graphgo.NewDiGraph()
	for _, v := range vars {
		g.AddNode(v)
	}
	for _, e := range edges {
		g.AddEdge(e[0], e[1])
	}
	return graphgo.IsDAG(g)
}
