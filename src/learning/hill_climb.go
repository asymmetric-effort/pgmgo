package learning

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ScoreFunc computes a local score for a variable given its parents and data.
type ScoreFunc func(variable string, parents []string, data *tabgo.DataFrame) float64

// operationType represents the type of graph operation.
type operationType int

const (
	opAdd operationType = iota
	opDelete
	opReverse
)

// operation represents a single graph modification.
type operation struct {
	opType operationType
	from   string
	to     string
	delta  float64 // score improvement
}

// HillClimbOption is a functional option for configuring HillClimbSearch.
type HillClimbOption func(*HillClimbSearch)

// HillClimbSearch implements greedy hill-climbing structure learning for
// Bayesian networks with tabu list support.
type HillClimbSearch struct {
	data        *tabgo.DataFrame
	scoreFn     ScoreFunc
	maxIndegree int
	tabuSize    int
	whiteList   map[[2]string]bool
	blackList   map[[2]string]bool
}

// WithMaxIndegree sets the maximum number of parents any node may have.
func WithMaxIndegree(n int) HillClimbOption {
	return func(h *HillClimbSearch) {
		h.maxIndegree = n
	}
}

// WithTabuSize sets the size of the tabu list (number of recent operations to
// remember). Once the list is full, the oldest entry is evicted.
func WithTabuSize(n int) HillClimbOption {
	return func(h *HillClimbSearch) {
		h.tabuSize = n
	}
}

// WithWhiteList specifies edges that must be present in the final model.
// The search will only add (never remove) these edges.
func WithWhiteList(edges [][2]string) HillClimbOption {
	return func(h *HillClimbSearch) {
		for _, e := range edges {
			h.whiteList[e] = true
		}
	}
}

// WithBlackList specifies edges that must not appear in the final model.
func WithBlackList(edges [][2]string) HillClimbOption {
	return func(h *HillClimbSearch) {
		for _, e := range edges {
			h.blackList[e] = true
		}
	}
}

// NewHillClimbSearch creates a new HillClimbSearch with the given data, scoring
// function, and options.
func NewHillClimbSearch(data *tabgo.DataFrame, scoreFn ScoreFunc, opts ...HillClimbOption) *HillClimbSearch {
	h := &HillClimbSearch{
		data:        data,
		scoreFn:     scoreFn,
		maxIndegree: 0, // 0 means unlimited
		tabuSize:    100,
		whiteList:   make(map[[2]string]bool),
		blackList:   make(map[[2]string]bool),
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// Estimate runs the hill-climbing algorithm and returns the learned Bayesian
// network structure. CPDs are not fitted; only the DAG structure is learned.
func (h *HillClimbSearch) Estimate() (*models.BayesianNetwork, error) {
	columns := h.data.Columns()
	if len(columns) == 0 {
		return nil, fmt.Errorf("learning: hill climb requires at least one column")
	}

	// Build a working DiGraph (we use graphgo directly to allow temporary
	// mutations that DAG would reject).
	g := graphgo.NewDiGraph()
	for _, col := range columns {
		g.AddNode(col)
	}

	// Add whitelist edges to the initial graph (if they don't create cycles).
	for edge := range h.whiteList {
		g.AddEdge(edge[0], edge[1])
		if !graphgo.IsDAG(g) {
			_ = g.RemoveEdge(edge[0], edge[1])
			return nil, fmt.Errorf("learning: whitelist edges create a cycle")
		}
	}

	// Compute the initial total score.
	currentScore := h.totalScore(g, columns)

	// Tabu list: ring buffer of recent operations.
	tabu := make([]operation, 0, h.tabuSize)

	for {
		bestOp, found := h.bestOperation(g, columns, currentScore, tabu)
		if !found {
			break
		}

		// Apply the best operation.
		h.applyOperation(g, bestOp)
		currentScore += bestOp.delta

		// Update tabu list.
		if h.tabuSize > 0 {
			if len(tabu) >= h.tabuSize {
				tabu = tabu[1:]
			}
			tabu = append(tabu, bestOp)
		}
	}

	// Build the BayesianNetwork from the learned graph.
	bn := models.NewBayesianNetwork()
	for _, col := range columns {
		if err := bn.AddNode(col); err != nil {
			return nil, fmt.Errorf("learning: %w", err)
		}
	}
	edges := g.Edges()
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Src != edges[j].Src {
			return edges[i].Src < edges[j].Src
		}
		return edges[i].Dst < edges[j].Dst
	})
	for _, e := range edges {
		if err := bn.AddEdge(e.Src, e.Dst); err != nil {
			return nil, fmt.Errorf("learning: %w", err)
		}
	}
	return bn, nil
}

// totalScore computes the sum of local scores for all variables.
func (h *HillClimbSearch) totalScore(g *graphgo.DiGraph, columns []string) float64 {
	total := 0.0
	for _, col := range columns {
		parents := sortedParents(g, col)
		total += h.scoreFn(col, parents, h.data)
	}
	return total
}

// bestOperation finds the legal operation with the best score improvement.
// Returns the operation and true if an improving operation exists.
func (h *HillClimbSearch) bestOperation(
	g *graphgo.DiGraph,
	columns []string,
	currentScore float64,
	tabu []operation,
) (operation, bool) {
	var best operation
	bestDelta := 0.0
	found := false

	nodes := make([]string, len(columns))
	copy(nodes, columns)
	sort.Strings(nodes)

	for _, u := range nodes {
		for _, v := range nodes {
			if u == v {
				continue
			}

			// --- Add edge u→v ---
			if !g.HasEdge(u, v) && !g.HasEdge(v, u) {
				if !h.blackList[[2]string{u, v}] && h.indegreeOK(g, v) {
					op := operation{opType: opAdd, from: u, to: v}
					if !h.isTabu(tabu, op) {
						delta := h.scoreDeltaAdd(g, u, v)
						if delta > bestDelta {
							op.delta = delta
							best = op
							bestDelta = delta
							found = true
						}
					}
				}
			}

			// --- Delete edge u→v ---
			if g.HasEdge(u, v) {
				// Cannot delete whitelist edges.
				if !h.whiteList[[2]string{u, v}] {
					op := operation{opType: opDelete, from: u, to: v}
					if !h.isTabu(tabu, op) {
						delta := h.scoreDeltaDelete(g, u, v)
						if delta > bestDelta {
							op.delta = delta
							best = op
							bestDelta = delta
							found = true
						}
					}
				}
			}

			// --- Reverse edge u→v (becomes v→u) ---
			if g.HasEdge(u, v) {
				// Cannot reverse if the reversed edge is blacklisted.
				if !h.blackList[[2]string{v, u}] && !h.whiteList[[2]string{u, v}] && h.indegreeOK(g, u) {
					op := operation{opType: opReverse, from: u, to: v}
					if !h.isTabu(tabu, op) {
						delta := h.scoreDeltaReverse(g, u, v)
						if delta > bestDelta {
							op.delta = delta
							best = op
							bestDelta = delta
							found = true
						}
					}
				}
			}
		}
	}

	return best, found
}

// scoreDeltaAdd computes the score change from adding edge u→v.
// It temporarily adds the edge, checks acyclicity, and reverts.
func (h *HillClimbSearch) scoreDeltaAdd(g *graphgo.DiGraph, u, v string) float64 {
	oldParents := sortedParents(g, v)
	oldScore := h.scoreFn(v, oldParents, h.data)

	// Temporarily add.
	g.AddEdge(u, v)
	if !graphgo.IsDAG(g) {
		_ = g.RemoveEdge(u, v)
		return -1 // indicates infeasible
	}

	newParents := sortedParents(g, v)
	newScore := h.scoreFn(v, newParents, h.data)

	// Revert.
	_ = g.RemoveEdge(u, v)

	return newScore - oldScore
}

// scoreDeltaDelete computes the score change from deleting edge u→v.
func (h *HillClimbSearch) scoreDeltaDelete(g *graphgo.DiGraph, u, v string) float64 {
	oldParents := sortedParents(g, v)
	oldScore := h.scoreFn(v, oldParents, h.data)

	_ = g.RemoveEdge(u, v)
	newParents := sortedParents(g, v)
	newScore := h.scoreFn(v, newParents, h.data)

	// Revert.
	g.AddEdge(u, v)

	return newScore - oldScore
}

// scoreDeltaReverse computes the score change from reversing edge u→v to v→u.
func (h *HillClimbSearch) scoreDeltaReverse(g *graphgo.DiGraph, u, v string) float64 {
	// Score of v with current parents (including u).
	oldParentsV := sortedParents(g, v)
	oldScoreV := h.scoreFn(v, oldParentsV, h.data)

	// Score of u with current parents (not including v).
	oldParentsU := sortedParents(g, u)
	oldScoreU := h.scoreFn(u, oldParentsU, h.data)

	// Remove u→v, add v→u.
	_ = g.RemoveEdge(u, v)
	g.AddEdge(v, u)

	if !graphgo.IsDAG(g) {
		// Revert.
		_ = g.RemoveEdge(v, u)
		g.AddEdge(u, v)
		return -1
	}

	newParentsV := sortedParents(g, v)
	newScoreV := h.scoreFn(v, newParentsV, h.data)

	newParentsU := sortedParents(g, u)
	newScoreU := h.scoreFn(u, newParentsU, h.data)

	// Revert.
	_ = g.RemoveEdge(v, u)
	g.AddEdge(u, v)

	return (newScoreV + newScoreU) - (oldScoreV + oldScoreU)
}

// applyOperation mutates the graph according to the operation.
func (h *HillClimbSearch) applyOperation(g *graphgo.DiGraph, op operation) {
	switch op.opType {
	case opAdd:
		g.AddEdge(op.from, op.to)
	case opDelete:
		_ = g.RemoveEdge(op.from, op.to)
	case opReverse:
		_ = g.RemoveEdge(op.from, op.to)
		g.AddEdge(op.to, op.from)
	}
}

// indegreeOK returns true if adding one more parent to the node would not
// violate the max indegree constraint.
func (h *HillClimbSearch) indegreeOK(g *graphgo.DiGraph, node string) bool {
	if h.maxIndegree <= 0 {
		return true
	}
	return g.InDegree(node) < h.maxIndegree
}

// isTabu returns true if the operation is in the tabu list.
func (h *HillClimbSearch) isTabu(tabu []operation, op operation) bool {
	for _, t := range tabu {
		if t.opType == op.opType && t.from == op.from && t.to == op.to {
			return true
		}
	}
	return false
}

// sortedParents returns the sorted parent list for a node in the graph.
func sortedParents(g *graphgo.DiGraph, node string) []string {
	p := g.Parents(node)
	sort.Strings(p)
	return p
}
