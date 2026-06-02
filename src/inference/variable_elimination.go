package inference

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// VariableElimination performs exact inference on a set of discrete factors
// using the variable elimination algorithm. It operates on factors obtained
// from a Bayesian network's ToMarkovFactors (or constructed directly),
// avoiding any direct dependency on the models package.
type VariableElimination struct {
	factors   []*factors.DiscreteFactor
	heuristic string // elimination-order heuristic; empty means "min_neighbors"
}

// NewVariableElimination creates a new VariableElimination engine from the
// given factor list. Each factor is deep-copied so the caller's originals
// are not modified during inference.
//
// An optional heuristic name may be provided to control the elimination
// order. Supported values: "min_neighbors" (default), "min_fill",
// "min_weight", "weighted_min_fill". If more than one heuristic is given
// only the first is used.
func NewVariableElimination(factorList []*factors.DiscreteFactor, heuristic ...string) *VariableElimination {
	copied := make([]*factors.DiscreteFactor, len(factorList))
	for i, f := range factorList {
		copied[i] = f.Copy()
	}
	h := ""
	if len(heuristic) > 0 {
		h = heuristic[0]
	}
	return &VariableElimination{factors: copied, heuristic: h}
}

// Query computes P(queryVars | evidence) via variable elimination.
//
// Steps:
//  1. Reduce all factors by the observed evidence.
//  2. Determine the elimination order: all variables except query and evidence.
//  3. For each variable to eliminate, multiply all factors containing it,
//     marginalize it out, and replace those factors with the result.
//  4. Multiply remaining factors and normalize.
func (ve *VariableElimination) Query(queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("inference: queryVars must not be empty")
	}

	// Step 1: reduce all factors by evidence.
	workingFactors, err := reduceAll(ve.factors, evidence)
	if err != nil {
		return nil, fmt.Errorf("inference: evidence reduction failed: %w", err)
	}

	// Collect all variables present in the working factors.
	allVars := collectVariables(workingFactors)

	// Build the set of variables to keep (query + evidence).
	keepSet := make(map[string]bool, len(queryVars)+len(evidence))
	for _, v := range queryVars {
		keepSet[v] = true
	}
	for v := range evidence {
		keepSet[v] = true
	}

	// Variables to eliminate.
	var eliminateVars []string
	for v := range allVars {
		if !keepSet[v] {
			eliminateVars = append(eliminateVars, v)
		}
	}

	// Step 2: determine elimination order.
	heuristic := ve.heuristic
	if heuristic == "" {
		heuristic = "min_neighbors"
	}
	order, err := GetEliminationOrder(workingFactors, eliminateVars, heuristic)
	if err != nil {
		return nil, fmt.Errorf("inference: elimination order failed: %w", err)
	}

	// Step 3: eliminate each variable in order.
	for _, elimVar := range order {
		workingFactors, err = eliminateVariable(workingFactors, elimVar)
		if err != nil {
			return nil, fmt.Errorf("inference: failed to eliminate %q: %w", elimVar, err)
		}
	}

	// Step 4: multiply remaining factors.
	if len(workingFactors) == 0 {
		return nil, fmt.Errorf("inference: no factors remain after elimination")
	}

	result, err := factors.FactorProduct(workingFactors...)
	if err != nil {
		return nil, fmt.Errorf("inference: final product failed: %w", err)
	}

	result.Normalize()
	return result, nil
}

// MAP finds the maximum a posteriori assignment for queryVars given evidence.
// It returns a map from variable name to the state index that maximizes
// P(queryVars | evidence).
func (ve *VariableElimination) MAP(queryVars []string, evidence map[string]int) (map[string]int, error) {
	result, err := ve.Query(queryVars, evidence)
	if err != nil {
		return nil, err
	}

	// Iterate over all assignments in the result factor and find the maximum.
	vars := result.Variables()
	card := result.Cardinality()
	totalSize := 1
	for _, c := range card {
		totalSize *= c
	}

	bestVal := -1.0
	bestAssignment := make(map[string]int, len(vars))

	for flat := 0; flat < totalSize; flat++ {
		assignment := flatToAssignment(vars, card, flat)
		val := result.GetValue(assignment)
		if val > bestVal {
			bestVal = val
			for k, v := range assignment {
				bestAssignment[k] = v
			}
		}
	}

	return bestAssignment, nil
}

// reduceAll applies evidence reduction to each factor. Factors that don't
// contain any evidence variable are copied unchanged.
func reduceAll(factorList []*factors.DiscreteFactor, evidence map[string]int) ([]*factors.DiscreteFactor, error) {
	result := make([]*factors.DiscreteFactor, 0, len(factorList))
	for _, f := range factorList {
		// Build a subset of evidence that applies to this factor.
		fVars := varSet(f)
		applicable := make(map[string]int)
		for v, val := range evidence {
			if fVars[v] {
				applicable[v] = val
			}
		}
		reduced, err := f.Reduce(applicable)
		if err != nil {
			return nil, err
		}
		result = append(result, reduced)
	}
	return result, nil
}

// eliminateVariable finds all factors containing the given variable,
// multiplies them together, marginalizes the variable out, and returns
// the updated factor list.
func eliminateVariable(factorList []*factors.DiscreteFactor, variable string) ([]*factors.DiscreteFactor, error) {
	var containing []*factors.DiscreteFactor
	var remaining []*factors.DiscreteFactor

	for _, f := range factorList {
		if varSet(f)[variable] {
			containing = append(containing, f)
		} else {
			remaining = append(remaining, f)
		}
	}

	if len(containing) == 0 {
		// Variable not in any factor; nothing to do.
		return factorList, nil
	}

	product, err := factors.FactorProduct(containing...)
	if err != nil {
		return nil, err
	}

	// If the variable to eliminate is the only variable in the product,
	// marginalizing it would remove all variables. The sum is a scalar
	// constant that scales the entire distribution uniformly and does not
	// affect the normalized result, so we simply drop the factor.
	prodVars := product.Variables()
	if len(prodVars) == 1 && prodVars[0] == variable {
		return remaining, nil
	}

	marginalized, err := product.Marginalize([]string{variable})
	if err != nil {
		return nil, err
	}

	return append(remaining, marginalized), nil
}

// collectVariables returns the set of all variable names across all factors.
func collectVariables(factorList []*factors.DiscreteFactor) map[string]bool {
	vars := make(map[string]bool)
	for _, f := range factorList {
		for _, v := range f.Variables() {
			vars[v] = true
		}
	}
	return vars
}

// varSet returns the set of variables in a single factor.
func varSet(f *factors.DiscreteFactor) map[string]bool {
	m := make(map[string]bool)
	for _, v := range f.Variables() {
		m[v] = true
	}
	return m
}

// flatToAssignment converts a flat index to a variable assignment map
// given variable names and cardinalities (row-major order).
func flatToAssignment(vars []string, card []int, flat int) map[string]int {
	assignment := make(map[string]int, len(vars))
	rem := flat
	for i := len(vars) - 1; i >= 0; i-- {
		assignment[vars[i]] = rem % card[i]
		rem /= card[i]
	}
	return assignment
}

// MaxMarginal computes the max-marginal of queryVars given evidence.
// Like Query, but during elimination each variable is maximized out
// rather than summed out, yielding the factor whose values are
// max_{hidden} Product(factors | evidence).
func (ve *VariableElimination) MaxMarginal(queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("inference: queryVars must not be empty")
	}

	workingFactors, err := reduceAll(ve.factors, evidence)
	if err != nil {
		return nil, fmt.Errorf("inference: evidence reduction failed: %w", err)
	}

	allVars := collectVariables(workingFactors)

	keepSet := make(map[string]bool, len(queryVars)+len(evidence))
	for _, v := range queryVars {
		keepSet[v] = true
	}
	for v := range evidence {
		keepSet[v] = true
	}

	var eliminateVars []string
	for v := range allVars {
		if !keepSet[v] {
			eliminateVars = append(eliminateVars, v)
		}
	}

	heuristic := ve.heuristic
	if heuristic == "" {
		heuristic = "min_neighbors"
	}
	order, err := GetEliminationOrder(workingFactors, eliminateVars, heuristic)
	if err != nil {
		return nil, fmt.Errorf("inference: elimination order failed: %w", err)
	}

	for _, elimVar := range order {
		workingFactors, err = maxEliminateVariable(workingFactors, elimVar)
		if err != nil {
			return nil, fmt.Errorf("inference: failed to max-eliminate %q: %w", elimVar, err)
		}
	}

	if len(workingFactors) == 0 {
		return nil, fmt.Errorf("inference: no factors remain after elimination")
	}

	result, err := factors.FactorProduct(workingFactors...)
	if err != nil {
		return nil, fmt.Errorf("inference: final product failed: %w", err)
	}

	result.Normalize()
	return result, nil
}

// maxEliminateVariable is like eliminateVariable but takes the max over
// the eliminated variable instead of summing.
func maxEliminateVariable(factorList []*factors.DiscreteFactor, variable string) ([]*factors.DiscreteFactor, error) {
	var containing []*factors.DiscreteFactor
	var remaining []*factors.DiscreteFactor

	for _, f := range factorList {
		if varSet(f)[variable] {
			containing = append(containing, f)
		} else {
			remaining = append(remaining, f)
		}
	}

	if len(containing) == 0 {
		return factorList, nil
	}

	product, err := factors.FactorProduct(containing...)
	if err != nil {
		return nil, err
	}

	prodVars := product.Variables()
	if len(prodVars) == 1 && prodVars[0] == variable {
		return remaining, nil
	}

	maximized, err := maxMarginalize(product, variable)
	if err != nil {
		return nil, err
	}

	return append(remaining, maximized), nil
}

// maxMarginalize returns a new factor with the given variable maximized out.
// For each assignment to the remaining variables, the value is the maximum
// over all states of the eliminated variable.
func maxMarginalize(f *factors.DiscreteFactor, variable string) (*factors.DiscreteFactor, error) {
	idx := -1
	fVars := f.Variables()
	fCard := f.Cardinality()
	for i, v := range fVars {
		if v == variable {
			idx = i
			break
		}
	}
	if idx == -1 {
		return nil, fmt.Errorf("inference: variable %q not in factor", variable)
	}

	var newVars []string
	var newCard []int
	for i, v := range fVars {
		if i != idx {
			newVars = append(newVars, v)
			newCard = append(newCard, fCard[i])
		}
	}

	newSize := 1
	for _, c := range newCard {
		newSize *= c
	}

	newValues := make([]float64, newSize)
	for i := range newValues {
		newValues[i] = -1
	}

	totalSize := 1
	for _, c := range fCard {
		totalSize *= c
	}
	data := f.Values().Data()

	for flat := 0; flat < totalSize; flat++ {
		assignment := flatToAssignment(fVars, fCard, flat)
		// Compute flat index in new factor.
		newFlat := 0
		stride := 1
		for i := len(newVars) - 1; i >= 0; i-- {
			newFlat += assignment[newVars[i]] * stride
			stride *= newCard[i]
		}
		val := data[flat]
		if newValues[newFlat] < 0 || val > newValues[newFlat] {
			newValues[newFlat] = val
		}
	}

	return factors.NewDiscreteFactor(newVars, newCard, newValues)
}

// InducedGraph returns the induced graph (also called the filled graph) that
// results from eliminating variables in the given order. Two variables are
// connected in the induced graph if they appear together in the same factor
// at any point during elimination.
func (ve *VariableElimination) InducedGraph(eliminationOrder []string) (*graphgo.Graph, error) {
	if len(eliminationOrder) == 0 {
		return graphgo.NewGraph(), nil
	}

	// Build factor variable sets.
	factorVarSets := make([]map[string]bool, len(ve.factors))
	for i, f := range ve.factors {
		factorVarSets[i] = make(map[string]bool)
		for _, v := range f.Variables() {
			factorVarSets[i][v] = true
		}
	}

	// Build interaction graph (adjacency).
	adj := make(map[string]map[string]bool)
	for _, vs := range factorVarSets {
		vars := make([]string, 0, len(vs))
		for v := range vs {
			vars = append(vars, v)
		}
		for i := 0; i < len(vars); i++ {
			if adj[vars[i]] == nil {
				adj[vars[i]] = make(map[string]bool)
			}
			for j := i + 1; j < len(vars); j++ {
				if adj[vars[j]] == nil {
					adj[vars[j]] = make(map[string]bool)
				}
				adj[vars[i]][vars[j]] = true
				adj[vars[j]][vars[i]] = true
			}
		}
	}

	// Collect all edges that ever appear (initial + fill edges).
	allEdges := make(map[string]bool)
	allNodes := make(map[string]bool)
	for v := range adj {
		allNodes[v] = true
	}

	// Record initial edges.
	for v, neighbors := range adj {
		for nb := range neighbors {
			key := v + "\x00" + nb
			if nb < v {
				key = nb + "\x00" + v
			}
			allEdges[key] = true
		}
	}

	// Simulate elimination.
	for _, node := range eliminationOrder {
		neighbors := make([]string, 0, len(adj[node]))
		for nb := range adj[node] {
			neighbors = append(neighbors, nb)
		}
		// Connect all pairs of neighbors (fill edges).
		for i := 0; i < len(neighbors); i++ {
			for j := i + 1; j < len(neighbors); j++ {
				a, b := neighbors[i], neighbors[j]
				adj[a][b] = true
				adj[b][a] = true
				key := a + "\x00" + b
				if b < a {
					key = b + "\x00" + a
				}
				allEdges[key] = true
			}
		}
		// Remove node.
		for nb := range adj[node] {
			delete(adj[nb], node)
		}
		delete(adj, node)
	}

	// Build the resulting graphgo.Graph.
	g := graphgo.NewGraph()
	for v := range allNodes {
		g.AddNode(v)
	}
	for key := range allEdges {
		parts := splitEdgeKey(key)
		g.AddEdge(parts[0], parts[1])
	}

	return g, nil
}

// splitEdgeKey splits a NUL-separated edge key into its two parts.
func splitEdgeKey(key string) [2]string {
	for i := 0; i < len(key); i++ {
		if key[i] == '\x00' {
			return [2]string{key[:i], key[i+1:]}
		}
	}
	return [2]string{key, ""}
}

// InducedWidth returns the induced width (treewidth) for the given
// elimination order, which is the maximum clique size minus 1 in the
// induced graph.
func (ve *VariableElimination) InducedWidth(eliminationOrder []string) (int, error) {
	if len(eliminationOrder) == 0 {
		return 0, nil
	}

	// Build factor variable sets.
	factorVarSets := make([]map[string]bool, len(ve.factors))
	for i, f := range ve.factors {
		factorVarSets[i] = make(map[string]bool)
		for _, v := range f.Variables() {
			factorVarSets[i][v] = true
		}
	}

	// Build interaction graph.
	adj := make(map[string]map[string]bool)
	for _, vs := range factorVarSets {
		vars := make([]string, 0, len(vs))
		for v := range vs {
			vars = append(vars, v)
		}
		for i := 0; i < len(vars); i++ {
			if adj[vars[i]] == nil {
				adj[vars[i]] = make(map[string]bool)
			}
			for j := i + 1; j < len(vars); j++ {
				if adj[vars[j]] == nil {
					adj[vars[j]] = make(map[string]bool)
				}
				adj[vars[i]][vars[j]] = true
				adj[vars[j]][vars[i]] = true
			}
		}
	}

	maxWidth := 0

	for _, node := range eliminationOrder {
		neighbors := make([]string, 0, len(adj[node]))
		for nb := range adj[node] {
			neighbors = append(neighbors, nb)
		}
		// The clique formed is {node} + neighbors; width = len(neighbors).
		if len(neighbors) > maxWidth {
			maxWidth = len(neighbors)
		}
		// Fill edges.
		for i := 0; i < len(neighbors); i++ {
			for j := i + 1; j < len(neighbors); j++ {
				a, b := neighbors[i], neighbors[j]
				if adj[a] == nil {
					adj[a] = make(map[string]bool)
				}
				if adj[b] == nil {
					adj[b] = make(map[string]bool)
				}
				adj[a][b] = true
				adj[b][a] = true
			}
		}
		// Remove node.
		for nb := range adj[node] {
			delete(adj[nb], node)
		}
		delete(adj, node)
	}

	return maxWidth, nil
}
