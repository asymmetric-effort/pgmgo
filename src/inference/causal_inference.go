package inference

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// CausalInference provides causal reasoning over a Bayesian network using
// the do-calculus. It supports interventional queries (do-operator), average
// treatment effect estimation, and backdoor adjustment.
type CausalInference struct {
	bn *models.BayesianNetwork
}

// NewCausalInference creates a new CausalInference engine from a validated
// BayesianNetwork. The network is deep-copied so the caller's original is
// not modified during interventions.
func NewCausalInference(bn *models.BayesianNetwork) (*CausalInference, error) {
	if bn == nil {
		return nil, fmt.Errorf("inference: BayesianNetwork must not be nil")
	}
	if err := bn.CheckModel(); err != nil {
		return nil, fmt.Errorf("inference: invalid BayesianNetwork: %w", err)
	}
	return &CausalInference{bn: bn.Copy()}, nil
}

// Query computes P(queryVars | do(doVars), evidence) using the truncated
// factorization (graph mutilation) approach:
//  1. Mutilate the graph: for each do-variable, remove all incoming edges.
//  2. Replace the do-variable's factor with a delta function.
//  3. Reduce remaining factors by evidence.
//  4. Use variable elimination on the mutilated model.
func (ci *CausalInference) Query(queryVars []string, doVars map[string]int, evidence map[string]int) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("inference: queryVars must not be empty")
	}

	// Build the mutilated factor list directly from the CPDs, replacing
	// do-variable CPDs with delta factors (which also implicitly removes
	// parent dependencies, achieving graph mutilation at the factor level).
	doSet := make(map[string]bool, len(doVars))
	for v := range doVars {
		doSet[v] = true
	}

	var mutilatedFactors []*factors.DiscreteFactor
	for _, node := range ci.bn.Nodes() {
		cpd := ci.bn.GetCPD(node)
		if cpd == nil {
			return nil, fmt.Errorf("inference: no CPD for node %q", node)
		}

		if doSet[node] {
			// Replace this CPD with a delta factor over just the do-variable.
			doVal := doVars[node]
			card := cpd.VariableCard()
			if doVal < 0 || doVal >= card {
				return nil, fmt.Errorf("inference: do-value %d out of range for variable %q (card %d)", doVal, node, card)
			}
			deltaVals := make([]float64, card)
			deltaVals[doVal] = 1.0
			deltaFactor, err := factors.NewDiscreteFactor([]string{node}, []int{card}, deltaVals)
			if err != nil {
				return nil, fmt.Errorf("inference: failed to create delta factor for %q: %w", node, err)
			}
			mutilatedFactors = append(mutilatedFactors, deltaFactor)
		} else {
			mutilatedFactors = append(mutilatedFactors, cpd.ToFactor())
		}
	}

	ve := NewVariableElimination(mutilatedFactors)
	result, err := ve.Query(queryVars, evidence)
	if err != nil {
		return nil, fmt.Errorf("inference: variable elimination failed: %w", err)
	}

	return result, nil
}

// ATE computes the Average Treatment Effect of a binary treatment on a
// discrete outcome variable:
//
//	ATE = E[outcome | do(treatment=treatmentValues[1])] - E[outcome | do(treatment=treatmentValues[0])]
//
// For a discrete outcome with states 0, 1, ..., k-1, the expected value
// E[outcome | do(treatment=v)] = sum_i ( i * P(outcome=i | do(treatment=v)) ).
func (ci *CausalInference) ATE(treatment string, outcome string, treatmentValues [2]int) (float64, error) {
	expectations := [2]float64{}

	for idx, tVal := range treatmentValues {
		doVars := map[string]int{treatment: tVal}
		result, err := ci.Query([]string{outcome}, doVars, nil)
		if err != nil {
			return 0, fmt.Errorf("inference: ATE query failed for do(%s=%d): %w", treatment, tVal, err)
		}

		// Compute expected value: sum_i(i * P(outcome=i))
		vars := result.Variables()
		card := result.Cardinality()
		if len(vars) != 1 {
			return 0, fmt.Errorf("inference: ATE expected single-variable result, got %v", vars)
		}
		ev := 0.0
		assignment := make(map[string]int, 1)
		for i := 0; i < card[0]; i++ {
			assignment[vars[0]] = i
			ev += float64(i) * result.GetValue(assignment)
		}
		expectations[idx] = ev
	}

	return expectations[1] - expectations[0], nil
}

// BackdoorAdjustment estimates the ATE using the backdoor adjustment formula
// from observational data. The adjustment set must satisfy the backdoor
// criterion (use IsValidBackdoor to verify).
//
// The formula is:
//
//	P(outcome | do(treatment)) = sum_z P(outcome | treatment, z) * P(z)
//
// where z ranges over all configurations of the adjustment set variables.
// The ATE is then:
//
//	E[outcome | do(treatment=1)] - E[outcome | do(treatment=0)]
//
// using treatment values 0 and 1.
//
// All variables in the data must be integer-valued (discrete states).
func (ci *CausalInference) BackdoorAdjustment(treatment, outcome string, adjustmentSet []string, data *tabgo.DataFrame) (float64, error) {
	if data == nil {
		return 0, fmt.Errorf("inference: data must not be nil")
	}
	if !ci.IsValidBackdoor(treatment, outcome, adjustmentSet) {
		return 0, fmt.Errorf("inference: adjustment set %v does not satisfy backdoor criterion for (%s, %s)", adjustmentSet, treatment, outcome)
	}

	nRows := data.Len()
	if nRows == 0 {
		return 0, fmt.Errorf("inference: data has no rows")
	}

	// Read all relevant columns as int slices.
	treatmentData := data.Column(treatment).Int()
	outcomeData := data.Column(outcome).Int()

	adjData := make([][]int, len(adjustmentSet))
	for i, v := range adjustmentSet {
		adjData[i] = data.Column(v).Int()
	}

	// Enumerate unique adjustment set configurations and count them.
	configCount := make(map[string]int)
	configMap := make(map[string][]int)

	for row := 0; row < nRows; row++ {
		vals := make([]int, len(adjustmentSet))
		for j := range adjustmentSet {
			vals[j] = adjData[j][row]
		}
		key := fmt.Sprintf("%v", vals)
		configCount[key]++
		if _, exists := configMap[key]; !exists {
			configMap[key] = vals
		}
	}

	// For each treatment value (0 and 1), compute E[outcome | do(treatment=t)]
	// using the backdoor formula.
	expectations := [2]float64{}
	for tIdx, tVal := range [2]int{0, 1} {
		totalExpectation := 0.0

		for key, adjVals := range configMap {
			// P(Z = z) = count(z) / N
			pZ := float64(configCount[key]) / float64(nRows)

			// E[outcome | treatment=t, Z=z] from data
			sumOutcome := 0.0
			count := 0
			for row := 0; row < nRows; row++ {
				if treatmentData[row] != tVal {
					continue
				}
				match := true
				for j := range adjustmentSet {
					if adjData[j][row] != adjVals[j] {
						match = false
						break
					}
				}
				if match {
					sumOutcome += float64(outcomeData[row])
					count++
				}
			}

			if count > 0 {
				eOutcome := sumOutcome / float64(count)
				totalExpectation += eOutcome * pZ
			}
			// If count == 0, this treatment-adjustment configuration was never
			// observed; we skip it (contributes 0).
		}

		expectations[tIdx] = totalExpectation
	}

	return expectations[1] - expectations[0], nil
}

// IsValidBackdoor checks whether the given adjustment set satisfies the
// backdoor criterion for estimating the causal effect of treatment on outcome.
//
// The backdoor criterion requires:
//  1. No node in adjustmentSet is a descendant of treatment.
//  2. adjustmentSet d-separates treatment from outcome in the graph where
//     all edges out of treatment have been removed (the manipulated graph).
func (ci *CausalInference) IsValidBackdoor(treatment, outcome string, adjustmentSet []string) bool {
	// Build a DiGraph from the BN structure.
	g := bnToDigraph(ci.bn)

	// Check criterion 1: no adjustment variable is a descendant of treatment.
	descendants := allDescendants(g, treatment)
	for _, z := range adjustmentSet {
		if descendants[z] {
			return false
		}
	}

	// Build the manipulated graph: remove all edges out of treatment.
	mutilatedG := g.Copy()
	for _, child := range g.Successors(treatment) {
		_ = mutilatedG.RemoveEdge(treatment, child)
	}

	// Check criterion 2: treatment and outcome are d-separated given adjustmentSet
	// in the manipulated graph.
	xSet := map[string]bool{treatment: true}
	ySet := map[string]bool{outcome: true}
	zSet := make(map[string]bool, len(adjustmentSet))
	for _, z := range adjustmentSet {
		zSet[z] = true
	}

	return graphgo.DSeparation(mutilatedG, xSet, ySet, zSet)
}

// bnToDigraph reconstructs a graphgo.DiGraph from the BayesianNetwork's
// public API. This is needed because the BN's internal DAG is unexported.
func bnToDigraph(bn *models.BayesianNetwork) *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	for _, node := range bn.Nodes() {
		g.AddNode(node)
	}
	for _, edge := range bn.Edges() {
		g.AddEdge(edge[0], edge[1])
	}
	return g
}

// allDescendants returns the set of all descendants of the given node
// (not including the node itself) using BFS.
func allDescendants(g *graphgo.DiGraph, node string) map[string]bool {
	desc := make(map[string]bool)
	queue := g.Successors(node)
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		if desc[curr] {
			continue
		}
		desc[curr] = true
		queue = append(queue, g.Successors(curr)...)
	}
	return desc
}
