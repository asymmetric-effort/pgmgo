package models

import (
	"fmt"
	"sort"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/src/base"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/independencies"
)

// MarkovNetwork represents a Markov random field (MRF) — an undirected
// graphical model where each factor (potential) is defined over a clique
// of the graph.
type MarkovNetwork struct {
	graph        *base.UndirectedGraph
	factorList   []*factors.DiscreteFactor
	varToFactors map[string][]*factors.DiscreteFactor
}

// NewMarkovNetwork creates a new empty MarkovNetwork.
func NewMarkovNetwork() *MarkovNetwork {
	return &MarkovNetwork{
		graph:        base.NewUndirectedGraph(),
		factorList:   nil,
		varToFactors: make(map[string][]*factors.DiscreteFactor),
	}
}

// AddNode adds a node (variable) to the network.
func (mn *MarkovNetwork) AddNode(node string) error {
	return mn.graph.AddNode(node)
}

// AddEdge adds an undirected edge between u and v. Both nodes must exist.
func (mn *MarkovNetwork) AddEdge(u, v string) error {
	return mn.graph.AddEdge(u, v)
}

// Nodes returns a sorted list of all nodes in the network.
func (mn *MarkovNetwork) Nodes() []string {
	return mn.graph.Nodes()
}

// Edges returns all undirected edges in canonical form (A < B), sorted
// lexicographically.
func (mn *MarkovNetwork) Edges() [][2]string {
	raw := mn.graph.Edges()
	result := make([][2]string, len(raw))
	for i, e := range raw {
		result[i] = [2]string{e.A, e.B}
	}
	return result
}

// Neighbors returns the sorted neighbors of a node in the undirected graph.
func (mn *MarkovNetwork) Neighbors(node string) []string {
	return mn.graph.Neighbors(node)
}

// AddFactor adds a factor to the network. All variables in the factor's scope
// must be nodes in the graph. Returns an error if any variable is missing or
// if the factor is nil.
func (mn *MarkovNetwork) AddFactor(f *factors.DiscreteFactor) error {
	if f == nil {
		return fmt.Errorf("models: factor must not be nil")
	}
	vars := f.Variables()
	for _, v := range vars {
		if !mn.graph.HasNode(v) {
			return fmt.Errorf("models: factor references unknown node %q", v)
		}
	}
	mn.factorList = append(mn.factorList, f)
	for _, v := range vars {
		mn.varToFactors[v] = append(mn.varToFactors[v], f)
	}
	return nil
}

// RemoveFactor removes all factors whose variable set exactly matches the
// given variables (order-independent).
func (mn *MarkovNetwork) RemoveFactor(variables []string) {
	target := make([]string, len(variables))
	copy(target, variables)
	sort.Strings(target)
	targetKey := strings.Join(target, "\x00")

	var kept []*factors.DiscreteFactor
	for _, f := range mn.factorList {
		fvars := f.Variables()
		sorted := make([]string, len(fvars))
		copy(sorted, fvars)
		sort.Strings(sorted)
		if strings.Join(sorted, "\x00") == targetKey {
			continue // remove this factor
		}
		kept = append(kept, f)
	}
	mn.factorList = kept

	// Rebuild varToFactors index.
	mn.varToFactors = make(map[string][]*factors.DiscreteFactor)
	for _, f := range mn.factorList {
		for _, v := range f.Variables() {
			mn.varToFactors[v] = append(mn.varToFactors[v], f)
		}
	}
}

// GetFactors returns all factors, in insertion order.
func (mn *MarkovNetwork) GetFactors() []*factors.DiscreteFactor {
	result := make([]*factors.DiscreteFactor, len(mn.factorList))
	copy(result, mn.factorList)
	return result
}

// GetFactorsOf returns all factors that include the given variable in their
// scope. Returns nil if the variable has no factors.
func (mn *MarkovNetwork) GetFactorsOf(variable string) []*factors.DiscreteFactor {
	fs := mn.varToFactors[variable]
	if len(fs) == 0 {
		return nil
	}
	result := make([]*factors.DiscreteFactor, len(fs))
	copy(result, fs)
	return result
}

// CheckModel validates the Markov network:
//  1. Every factor's scope variables must be connected by edges: for every
//     pair of variables in a factor, the edge must exist in the graph.
//  2. Every node must be covered by at least one factor.
func (mn *MarkovNetwork) CheckModel() error {
	if len(mn.factorList) == 0 {
		return fmt.Errorf("models: Markov network has no factors")
	}

	// Check that every pair of variables within each factor has an edge
	// (or the factor is unary).
	for i, f := range mn.factorList {
		vars := f.Variables()
		for j := 0; j < len(vars); j++ {
			if !mn.graph.HasNode(vars[j]) {
				return fmt.Errorf("models: factor %d references unknown node %q", i, vars[j])
			}
			for k := j + 1; k < len(vars); k++ {
				if !mn.graph.HasEdge(vars[j], vars[k]) {
					return fmt.Errorf("models: factor %d has variables %q and %q but no edge exists between them",
						i, vars[j], vars[k])
				}
			}
		}
	}

	// Check that every node is covered by at least one factor.
	nodes := mn.graph.Nodes()
	for _, node := range nodes {
		if len(mn.varToFactors[node]) == 0 {
			return fmt.Errorf("models: node %q is not covered by any factor", node)
		}
	}

	return nil
}

// GetPartitionFunction computes Z, the partition function, by summing the
// product of all factors over all joint assignments. This is only feasible
// for small models.
func (mn *MarkovNetwork) GetPartitionFunction() (float64, error) {
	if len(mn.factorList) == 0 {
		return 0, fmt.Errorf("models: no factors in the network")
	}

	// Compute the full joint factor by multiplying all factors together.
	product, err := factors.FactorProduct(mn.factorList...)
	if err != nil {
		return 0, fmt.Errorf("models: partition function: %w", err)
	}

	// Sum all values in the product factor.
	data := product.Values().Data()
	z := 0.0
	for _, v := range data {
		z += v
	}
	return z, nil
}

// MarkovBlanket returns the Markov blanket of a node, which in an undirected
// model is simply the set of neighbors.
func (mn *MarkovNetwork) MarkovBlanket(node string) []string {
	return mn.graph.Neighbors(node)
}

// ToJunctionTree constructs a junction tree from the Markov network by:
//  1. Building a graphgo.Graph from the undirected graph
//  2. Finding a min-degree elimination order
//  3. Triangulating the graph
//  4. Finding maximal cliques
//  5. Building the junction tree
//  6. Assigning factors to cliques
func (mn *MarkovNetwork) ToJunctionTree() (*JunctionTree, error) {
	if err := mn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: cannot build junction tree: %w", err)
	}

	// Build a graphgo.Graph from the UndirectedGraph's public API.
	g := graphgo.NewGraph()
	for _, node := range mn.graph.Nodes() {
		g.AddNode(node)
	}
	for _, e := range mn.graph.Edges() {
		g.AddEdge(e.A, e.B)
	}

	// Find elimination order using min-degree heuristic.
	order := minDegreeOrder(g)

	// Triangulate.
	triangulated := graphgo.Triangulate(g, order)

	// Find maximal cliques.
	cliques := graphgo.MaxCliques(triangulated)
	if len(cliques) == 0 {
		return &JunctionTree{
			cliques:       nil,
			tree:          graphgo.NewGraph(),
			separators:    make(map[string][]string),
			cliqueFactors: make(map[int][]*factors.DiscreteFactor),
		}, nil
	}

	// Build junction tree.
	tree, separators := graphgo.BuildJunctionTree(cliques)

	// Assign factors to cliques.
	cliqueFactors := assignFactorsToCliques(mn.factorList, cliques)

	return &JunctionTree{
		cliques:       cliques,
		tree:          tree,
		separators:    separators,
		cliqueFactors: cliqueFactors,
	}, nil
}

// States returns a map from each variable name to its cardinality, as
// extracted from the factors in the network. Variables with no factors are
// not included.
func (mn *MarkovNetwork) States() map[string]int {
	result := make(map[string]int)
	for _, f := range mn.factorList {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			if _, ok := result[v]; !ok {
				result[v] = card[i]
			}
		}
	}
	return result
}

// GetCardinality returns the cardinality of a node as determined from its
// factors. Returns an error if no factor covers the node.
func (mn *MarkovNetwork) GetCardinality(node string) (int, error) {
	if !mn.graph.HasNode(node) {
		return 0, fmt.Errorf("models: node %q not in network", node)
	}
	fs := mn.varToFactors[node]
	if len(fs) == 0 {
		return 0, fmt.Errorf("models: node %q has no factors to determine cardinality", node)
	}
	for _, f := range fs {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			if v == node {
				return card[i], nil
			}
		}
	}
	return 0, fmt.Errorf("models: node %q not found in any factor scope", node)
}

// Triangulate returns a triangulated copy of the Markov network. The
// heuristic parameter selects the elimination ordering heuristic:
// "min_degree" (default), "min_fill".
func (mn *MarkovNetwork) Triangulate(heuristic string) (*MarkovNetwork, error) {
	if heuristic == "" {
		heuristic = "min_degree"
	}

	// Build a graphgo.Graph from the undirected graph.
	g := graphgo.NewGraph()
	for _, node := range mn.graph.Nodes() {
		g.AddNode(node)
	}
	for _, e := range mn.graph.Edges() {
		g.AddEdge(e.A, e.B)
	}

	// Get elimination order.
	var order []string
	switch heuristic {
	case "min_degree":
		order = minDegreeOrder(g)
	case "min_fill":
		order = minFillOrder(g)
	default:
		return nil, fmt.Errorf("models: unsupported triangulation heuristic %q", heuristic)
	}

	// Triangulate.
	triangulated := graphgo.Triangulate(g, order)

	// Build a new MarkovNetwork with the triangulated graph.
	result := NewMarkovNetwork()
	for _, node := range triangulated.Nodes() {
		result.AddNode(node)
	}
	for _, e := range triangulated.Edges() {
		result.AddEdge(e.A, e.B)
	}

	// Copy factors.
	for _, f := range mn.factorList {
		result.AddFactor(f.Copy())
	}

	return result, nil
}

// minFillOrder returns an elimination ordering using the min-fill heuristic
// on a graphgo.Graph.
func minFillOrder(g *graphgo.Graph) []string {
	nodes := g.Nodes()
	sort.Strings(nodes)

	adj := make(map[string]map[string]bool, len(nodes))
	for _, n := range nodes {
		adj[n] = make(map[string]bool)
	}
	for _, e := range g.Edges() {
		adj[e.A][e.B] = true
		adj[e.B][e.A] = true
	}

	remaining := make(map[string]bool, len(nodes))
	for _, n := range nodes {
		remaining[n] = true
	}

	order := make([]string, 0, len(nodes))
	for len(remaining) > 0 {
		best := ""
		bestFill := -1
		for _, n := range nodes {
			if !remaining[n] {
				continue
			}
			nbs := make([]string, 0, len(adj[n]))
			for nb := range adj[n] {
				nbs = append(nbs, nb)
			}
			fill := 0
			for i := 0; i < len(nbs); i++ {
				for j := i + 1; j < len(nbs); j++ {
					if !adj[nbs[i]][nbs[j]] {
						fill++
					}
				}
			}
			if best == "" || fill < bestFill || (fill == bestFill && n < best) {
				best = n
				bestFill = fill
			}
		}

		order = append(order, best)
		delete(remaining, best)

		// Add fill edges and eliminate.
		nbs := make([]string, 0, len(adj[best]))
		for nb := range adj[best] {
			nbs = append(nbs, nb)
		}
		for i := 0; i < len(nbs); i++ {
			for j := i + 1; j < len(nbs); j++ {
				adj[nbs[i]][nbs[j]] = true
				adj[nbs[j]][nbs[i]] = true
			}
		}
		for nb := range adj[best] {
			delete(adj[nb], best)
		}
		delete(adj, best)
	}

	return order
}

// ToFactorGraph converts the Markov network to a factor graph.
func (mn *MarkovNetwork) ToFactorGraph() (*FactorGraph, error) {
	fg := NewFactorGraph()

	// Add all variables with cardinalities from factors.
	for _, node := range mn.graph.Nodes() {
		card, err := mn.GetCardinality(node)
		if err != nil {
			return nil, fmt.Errorf("models: ToFactorGraph: %w", err)
		}
		if err := fg.AddVariable(node, card); err != nil {
			return nil, fmt.Errorf("models: ToFactorGraph: %w", err)
		}
	}

	// Add all factors.
	for _, f := range mn.factorList {
		if err := fg.AddFactor(f.Copy()); err != nil {
			return nil, fmt.Errorf("models: ToFactorGraph: %w", err)
		}
	}

	return fg, nil
}

// ToBayesianModel converts the Markov network to a Bayesian network by
// triangulating the graph and finding a topological ordering of the
// resulting DAG. The CPDs are derived from the factors.
func (mn *MarkovNetwork) ToBayesianModel() (*BayesianNetwork, error) {
	if err := mn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: ToBayesianModel: %w", err)
	}

	// Build graphgo.Graph.
	g := graphgo.NewGraph()
	for _, node := range mn.graph.Nodes() {
		g.AddNode(node)
	}
	for _, e := range mn.graph.Edges() {
		g.AddEdge(e.A, e.B)
	}

	// Triangulate.
	order := minDegreeOrder(g)
	triangulated := graphgo.Triangulate(g, order)

	// Find maximal cliques from the triangulated graph.
	cliques := graphgo.MaxCliques(triangulated)

	// Build a DAG using the elimination order. Nodes earlier in the order
	// become parents of later nodes when they share a clique.
	bn := NewBayesianNetwork()
	allNodes := mn.graph.Nodes()
	for _, node := range allNodes {
		bn.AddNode(node)
	}

	// Create an ordering: use elimination order (reversed gives a
	// topological ordering of the resulting DAG).
	posInOrder := make(map[string]int, len(order))
	for i, n := range order {
		posInOrder[n] = i
	}
	// Nodes not in elimination order (shouldn't happen, but handle gracefully)
	for _, n := range allNodes {
		if _, ok := posInOrder[n]; !ok {
			posInOrder[n] = len(order)
		}
	}

	// For each clique, create directed edges from earlier nodes to later ones.
	for _, clique := range cliques {
		sorted := make([]string, len(clique))
		copy(sorted, clique)
		sort.Slice(sorted, func(i, j int) bool {
			return posInOrder[sorted[i]] < posInOrder[sorted[j]]
		})
		// Last node in elimination order within the clique gets parents = all earlier nodes.
		for i := 1; i < len(sorted); i++ {
			child := sorted[i]
			parent := sorted[i-1]
			// Add edges from all earlier nodes to later ones.
			for j := 0; j < i; j++ {
				bn.AddEdge(sorted[j], child)
			}
			_ = parent // suppress unused
		}
	}

	// Build cardinality map.
	cardMap := make(map[string]int)
	for _, f := range mn.factorList {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			if _, ok := cardMap[v]; !ok {
				cardMap[v] = card[i]
			}
		}
	}

	// Compute the joint factor.
	if len(mn.factorList) == 0 {
		return nil, fmt.Errorf("models: ToBayesianModel: no factors in network")
	}
	joint, err := factors.FactorProduct(mn.factorList...)
	if err != nil {
		return nil, fmt.Errorf("models: ToBayesianModel: joint product failed: %w", err)
	}

	// For each node, compute P(node | parents) = P(node, parents) / P(parents)
	// by marginalizing the joint.
	nodes := bn.Nodes()
	for _, node := range nodes {
		parents := bn.Parents(node)
		card := cardMap[node]
		evidenceVars := parents
		evidenceCards := make([]int, len(parents))
		for i, p := range parents {
			evidenceCards[i] = cardMap[p]
		}

		numParentConfigs := 1
		for _, ec := range evidenceCards {
			numParentConfigs *= ec
		}

		// Scope of CPD: [node] + parents
		scopeVars := make([]string, 0, 1+len(parents))
		scopeVars = append(scopeVars, node)
		scopeVars = append(scopeVars, parents...)

		// Marginalize joint to scope variables.
		jointVars := joint.Variables()
		var margOutVars []string
		scopeSet := make(map[string]bool, len(scopeVars))
		for _, v := range scopeVars {
			scopeSet[v] = true
		}
		for _, v := range jointVars {
			if !scopeSet[v] {
				margOutVars = append(margOutVars, v)
			}
		}

		var scopeFactor *factors.DiscreteFactor
		if len(margOutVars) > 0 {
			scopeFactor, err = joint.Marginalize(margOutVars)
			if err != nil {
				return nil, fmt.Errorf("models: ToBayesianModel: marginalize for %q failed: %w", node, err)
			}
		} else {
			scopeFactor = joint.Copy()
		}

		// Build the CPD values table: [childState][parentConfig]
		values := make([][]float64, card)
		for cs := 0; cs < card; cs++ {
			values[cs] = make([]float64, numParentConfigs)
		}

		if len(parents) > 0 {
			// Compute P(parents) by marginalizing out node.
			parentFactor, err := scopeFactor.Marginalize([]string{node})
			if err != nil {
				return nil, fmt.Errorf("models: ToBayesianModel: parent marginalize for %q failed: %w", node, err)
			}

			// Fill in CPD values.
			for cs := 0; cs < card; cs++ {
				for pc := 0; pc < numParentConfigs; pc++ {
					// Decompose parent config.
					parentAssignment := make(map[string]int, len(parents))
					rem := pc
					for i := len(parents) - 1; i >= 0; i-- {
						parentAssignment[parents[i]] = rem % evidenceCards[i]
						rem /= evidenceCards[i]
					}

					assignment := make(map[string]int, len(scopeVars))
					for k, v := range parentAssignment {
						assignment[k] = v
					}
					assignment[node] = cs

					jointVal := scopeFactor.GetValue(assignment)
					parentVal := parentFactor.GetValue(parentAssignment)
					if parentVal > 0 {
						values[cs][pc] = jointVal / parentVal
					}
				}
			}
		} else {
			// No parents: normalize the marginal.
			sum := 0.0
			for cs := 0; cs < card; cs++ {
				values[cs][0] = scopeFactor.GetValue(map[string]int{node: cs})
				sum += values[cs][0]
			}
			if sum > 0 {
				for cs := 0; cs < card; cs++ {
					values[cs][0] /= sum
				}
			}
		}

		// Build TabularCPD.
		cpd, err := factors.NewTabularCPD(node, card, values, evidenceVars, evidenceCards)
		if err != nil {
			return nil, fmt.Errorf("models: ToBayesianModel: CPD creation for %q failed: %w", node, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			return nil, fmt.Errorf("models: ToBayesianModel: AddCPD for %q failed: %w", node, err)
		}
	}

	return bn, nil
}

// GetLocalIndependencies returns the local Markov property independence
// assertions for a node. In a Markov network, a node is conditionally
// independent of all non-neighbors given its neighbors (Markov blanket).
func (mn *MarkovNetwork) GetLocalIndependencies(node string) ([]*independencies.IndependenceAssertion, error) {
	if !mn.graph.HasNode(node) {
		return nil, fmt.Errorf("models: node %q not in network", node)
	}

	neighbors := mn.graph.Neighbors(node)
	neighborSet := make(map[string]bool, len(neighbors))
	neighborSet[node] = true
	for _, nb := range neighbors {
		neighborSet[nb] = true
	}

	// Non-neighbors (excluding the node itself).
	var nonNeighbors []string
	for _, n := range mn.graph.Nodes() {
		if !neighborSet[n] {
			nonNeighbors = append(nonNeighbors, n)
		}
	}

	if len(nonNeighbors) == 0 {
		return nil, nil
	}

	assertion := independencies.NewIndependenceAssertion(
		[]string{node},
		nonNeighbors,
		neighbors,
	)
	return []*independencies.IndependenceAssertion{assertion}, nil
}

// Copy returns a deep copy of the MarkovNetwork.
func (mn *MarkovNetwork) Copy() *MarkovNetwork {
	newFactors := make([]*factors.DiscreteFactor, len(mn.factorList))
	for i, f := range mn.factorList {
		newFactors[i] = f.Copy()
	}

	newMN := &MarkovNetwork{
		graph:        mn.graph.Copy(),
		factorList:   newFactors,
		varToFactors: make(map[string][]*factors.DiscreteFactor),
	}

	// Rebuild varToFactors from the copied factors.
	for _, f := range newMN.factorList {
		for _, v := range f.Variables() {
			newMN.varToFactors[v] = append(newMN.varToFactors[v], f)
		}
	}

	return newMN
}
