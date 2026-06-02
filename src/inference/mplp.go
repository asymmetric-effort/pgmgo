package inference

import (
	"fmt"
	"math"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// MPLP implements Max-Product Linear Programming for MAP inference.
// It uses coordinate descent on the LP dual to find the most probable
// assignment to a set of variables given evidence.
type MPLP struct {
	factors []*factors.DiscreteFactor
}

// NewMPLP creates a new MPLP engine from the given factor list.
// Each factor is deep-copied so the caller's originals are not modified.
func NewMPLP(factorList []*factors.DiscreteFactor) *MPLP {
	copied := make([]*factors.DiscreteFactor, len(factorList))
	for i, f := range factorList {
		copied[i] = f.Copy()
	}
	return &MPLP{factors: copied}
}

// factorVarEdge identifies a (factor index, variable name) pair for dual messages.
type factorVarEdge struct {
	factorIdx int
	variable  string
}

// mplpState holds the internal state of the MPLP algorithm during optimization.
type mplpState struct {
	// workingFactors are evidence-reduced factors (in log space).
	workingFactors []*factors.DiscreteFactor

	// cardMap maps variable name to cardinality.
	cardMap map[string]int

	// varToFactors maps each variable to the list of factor indices containing it.
	varToFactors map[string][]int

	// messages[factorIdx][variable] stores the dual message (log-space) from
	// factor factorIdx to variable. It is a slice of length card(variable).
	messages map[int]map[string][]float64

	// allVars is the set of all variable names in the working factors.
	allVars []string
}

// initState sets up the MPLP internal state from evidence-reduced factors.
func initState(workingFactors []*factors.DiscreteFactor) *mplpState {
	st := &mplpState{
		workingFactors: workingFactors,
		cardMap:        make(map[string]int),
		varToFactors:   make(map[string][]int),
		messages:       make(map[int]map[string][]float64),
	}

	varSet := make(map[string]bool)
	for fi, f := range workingFactors {
		vars := f.Variables()
		card := f.Cardinality()
		for vi, v := range vars {
			if _, ok := st.cardMap[v]; !ok {
				st.cardMap[v] = card[vi]
			}
			if !varSet[v] {
				varSet[v] = true
				st.allVars = append(st.allVars, v)
			}
			st.varToFactors[v] = append(st.varToFactors[v], fi)
		}
	}

	// Initialize all messages to zero.
	for fi, f := range workingFactors {
		st.messages[fi] = make(map[string][]float64)
		for _, v := range f.Variables() {
			msg := make([]float64, st.cardMap[v])
			st.messages[fi][v] = msg
		}
	}

	return st
}

// logFactor converts a factor's values to log space, returning log-values
// indexed by flat position. Zeros become -Inf.
func logFactor(f *factors.DiscreteFactor) []float64 {
	data := f.Values().Data()
	logData := make([]float64, len(data))
	for i, v := range data {
		if v <= 0 {
			logData[i] = math.Inf(-1)
		} else {
			logData[i] = math.Log(v)
		}
	}
	return logData
}

// maxOverOthers computes, for each value of targetVar, the maximum of
// (logFactor + sum of messages for other variables) over all assignments
// to the other variables in the factor.
func maxOverOthers(f *factors.DiscreteFactor, logData []float64, msgs map[string][]float64, targetVar string) []float64 {
	vars := f.Variables()
	card := f.Cardinality()
	targetCard := 0
	targetIdx := -1
	for i, v := range vars {
		if v == targetVar {
			targetCard = card[i]
			targetIdx = i
			break
		}
	}

	result := make([]float64, targetCard)
	for i := range result {
		result[i] = math.Inf(-1)
	}

	totalSize := 1
	for _, c := range card {
		totalSize *= c
	}

	for flat := 0; flat < totalSize; flat++ {
		// Decompose flat index.
		assignment := make(map[string]int, len(vars))
		rem := flat
		for i := len(vars) - 1; i >= 0; i-- {
			assignment[vars[i]] = rem % card[i]
			rem /= card[i]
		}

		val := logData[flat]
		if math.IsInf(val, -1) {
			continue
		}

		// Add messages for all variables except targetVar.
		for _, v := range vars {
			if v == targetVar {
				continue
			}
			if msg, ok := msgs[v]; ok {
				val += msg[assignment[v]]
			}
		}

		tVal := assignment[vars[targetIdx]]
		if val > result[tVal] {
			result[tVal] = val
		}
	}

	return result
}

// MAP finds the maximum a posteriori assignment for queryVars given evidence
// using the MPLP (Max-Product Linear Programming) algorithm.
//
// The algorithm performs coordinate descent on the dual of the MAP LP
// relaxation. For each variable, it updates the messages from all factors
// containing that variable to tighten the LP relaxation bound. After
// convergence, the MAP assignment is extracted from the converged messages.
//
// Returns the MAP assignment, the objective value (log-probability), and
// any error encountered.
func (m *MPLP) MAP(queryVars []string, evidence map[string]int, maxIter int, tol float64) (map[string]int, float64, error) {
	if len(queryVars) == 0 {
		return nil, 0, fmt.Errorf("mplp: queryVars must not be empty")
	}
	if maxIter <= 0 {
		return nil, 0, fmt.Errorf("mplp: maxIter must be positive")
	}

	// Step 1: Reduce factors by evidence.
	workingFactors, err := reduceAll(m.factors, evidence)
	if err != nil {
		return nil, 0, fmt.Errorf("mplp: evidence reduction failed: %w", err)
	}

	// Filter out scalar factors (all evidence variables reduced away).
	var nonScalar []*factors.DiscreteFactor
	scalarLogSum := 0.0
	for _, f := range workingFactors {
		if len(f.Variables()) == 0 {
			// Scalar factor: accumulate its log value.
			data := f.Values().Data()
			if len(data) > 0 && data[0] > 0 {
				scalarLogSum += math.Log(data[0])
			}
		} else {
			nonScalar = append(nonScalar, f)
		}
	}
	workingFactors = nonScalar

	if len(workingFactors) == 0 {
		// All factors were scalar; return empty assignment.
		assignment := make(map[string]int, len(queryVars))
		for _, v := range queryVars {
			assignment[v] = 0
		}
		return assignment, scalarLogSum, nil
	}

	// Step 2: Initialize dual state.
	st := initState(workingFactors)

	// Precompute log-space factor values.
	logFactors := make([][]float64, len(workingFactors))
	for i, f := range workingFactors {
		logFactors[i] = logFactor(f)
	}

	// Step 3: Iterate coordinate descent.
	prevObj := math.Inf(-1)
	for iter := 0; iter < maxIter; iter++ {
		for _, v := range st.allVars {
			fIndices := st.varToFactors[v]
			if len(fIndices) <= 1 {
				continue
			}
			nFactors := len(fIndices)
			card := st.cardMap[v]

			// For each factor containing v, compute max over other vars
			// of (logFactor + messages from other vars).
			beliefContribs := make([][]float64, nFactors)
			for k, fi := range fIndices {
				beliefContribs[k] = maxOverOthers(
					workingFactors[fi], logFactors[fi],
					st.messages[fi], v,
				)
			}

			// Compute the average belief across all factors for this variable.
			avgBelief := make([]float64, card)
			for xi := 0; xi < card; xi++ {
				sum := 0.0
				for k := range fIndices {
					sum += beliefContribs[k][xi]
				}
				avgBelief[xi] = sum / float64(nFactors)
			}

			// Update messages: for each factor fi, message fi->v is set to
			// avgBelief - beliefContrib[fi] (the "share" redistribution).
			for k, fi := range fIndices {
				msg := st.messages[fi][v]
				for xi := 0; xi < card; xi++ {
					msg[xi] = avgBelief[xi] - beliefContribs[k][xi]
				}
			}
		}

		// Compute dual objective for convergence check.
		obj := m.computeDualObjective(st, logFactors, scalarLogSum)
		if math.Abs(obj-prevObj) < tol {
			break
		}
		prevObj = obj
	}

	// Step 4: Extract MAP assignment from converged beliefs.
	assignment := make(map[string]int, len(queryVars))
	querySet := make(map[string]bool, len(queryVars))
	for _, v := range queryVars {
		querySet[v] = true
	}

	// For each variable, compute the belief as the sum of max-marginals
	// from all factors containing it, and pick the argmax.
	for _, v := range st.allVars {
		if !querySet[v] {
			continue
		}
		card := st.cardMap[v]
		belief := make([]float64, card)
		for i := range belief {
			belief[i] = 0
		}

		for _, fi := range st.varToFactors[v] {
			contrib := maxOverOthers(
				workingFactors[fi], logFactors[fi],
				st.messages[fi], v,
			)
			for xi := 0; xi < card; xi++ {
				belief[xi] += contrib[xi]
			}
		}

		bestVal := math.Inf(-1)
		bestState := 0
		for xi := 0; xi < card; xi++ {
			if belief[xi] > bestVal {
				bestVal = belief[xi]
				bestState = xi
			}
		}
		assignment[v] = bestState
	}

	// Compute the objective value at the MAP assignment.
	objVal := scalarLogSum
	for fi, f := range workingFactors {
		vars := f.Variables()
		card := f.Cardinality()
		totalSize := 1
		for _, c := range card {
			totalSize *= c
		}
		// Build the assignment for this factor using the MAP values.
		// For variables not in queryVars, pick the best from beliefs.
		fAssign := make(map[string]int, len(vars))
		for _, v := range vars {
			if val, ok := assignment[v]; ok {
				fAssign[v] = val
			} else {
				// Non-query variable: find its best state from beliefs.
				card := st.cardMap[v]
				belief := make([]float64, card)
				for _, fi2 := range st.varToFactors[v] {
					contrib := maxOverOthers(
						workingFactors[fi2], logFactors[fi2],
						st.messages[fi2], v,
					)
					for xi := 0; xi < card; xi++ {
						belief[xi] += contrib[xi]
					}
				}
				bestState := 0
				bestVal := math.Inf(-1)
				for xi := 0; xi < card; xi++ {
					if belief[xi] > bestVal {
						bestVal = belief[xi]
						bestState = xi
					}
				}
				assignment[v] = bestState
				fAssign[v] = bestState
			}
		}

		// Compute flat index for this assignment.
		flat := 0
		stride := 1
		for i := len(vars) - 1; i >= 0; i-- {
			flat += fAssign[vars[i]] * stride
			stride *= card[i]
		}
		objVal += logFactors[fi][flat]
	}

	// Filter assignment to only query vars.
	result := make(map[string]int, len(queryVars))
	for _, v := range queryVars {
		result[v] = assignment[v]
	}

	return result, objVal, nil
}

// computeDualObjective computes the current dual objective value.
// The dual objective is the sum over all factors of max over assignments
// of (logFactor + messages).
func (m *MPLP) computeDualObjective(st *mplpState, logFactors [][]float64, scalarLogSum float64) float64 {
	obj := scalarLogSum

	for fi, f := range st.workingFactors {
		vars := f.Variables()
		card := f.Cardinality()
		totalSize := 1
		for _, c := range card {
			totalSize *= c
		}

		bestVal := math.Inf(-1)
		for flat := 0; flat < totalSize; flat++ {
			val := logFactors[fi][flat]
			if math.IsInf(val, -1) {
				continue
			}
			// Add all messages for this factor.
			rem := flat
			for i := len(vars) - 1; i >= 0; i-- {
				xi := rem % card[i]
				rem /= card[i]
				if msg, ok := st.messages[fi][vars[i]]; ok {
					val += msg[xi]
				}
			}
			if val > bestVal {
				bestVal = val
			}
		}
		if !math.IsInf(bestVal, -1) {
			obj += bestVal
		}
	}

	return obj
}

// Tighten runs tightening iterations on the dual and returns the dual
// objective value after convergence.
func (m *MPLP) Tighten(maxIter int) float64 {
	st := initState(m.factors)
	logFactors := make([][]float64, len(m.factors))
	for i, f := range m.factors {
		logFactors[i] = logFactor(f)
	}

	prevObj := math.Inf(-1)
	for iter := 0; iter < maxIter; iter++ {
		for _, v := range st.allVars {
			fIndices := st.varToFactors[v]
			if len(fIndices) <= 1 {
				continue
			}
			nFactors := len(fIndices)
			card := st.cardMap[v]

			beliefContribs := make([][]float64, nFactors)
			for k, fi := range fIndices {
				beliefContribs[k] = maxOverOthers(
					m.factors[fi], logFactors[fi],
					st.messages[fi], v,
				)
			}

			avgBelief := make([]float64, card)
			for xi := 0; xi < card; xi++ {
				sum := 0.0
				for k := range fIndices {
					sum += beliefContribs[k][xi]
				}
				avgBelief[xi] = sum / float64(nFactors)
			}

			for k, fi := range fIndices {
				msg := st.messages[fi][v]
				for xi := 0; xi < card; xi++ {
					msg[xi] = avgBelief[xi] - beliefContribs[k][xi]
				}
			}
		}

		obj := m.computeDualObjective(st, logFactors, 0)
		if math.Abs(obj-prevObj) < 1e-10 {
			return obj
		}
		prevObj = obj
	}

	return m.computeDualObjective(st, logFactors, 0)
}

// GetDualObjective computes and returns the current dual objective without
// running any iterations. This initializes messages to zero and returns
// the resulting objective.
func (m *MPLP) GetDualObjective() float64 {
	st := initState(m.factors)
	logFactors := make([][]float64, len(m.factors))
	for i, f := range m.factors {
		logFactors[i] = logFactor(f)
	}
	return m.computeDualObjective(st, logFactors, 0)
}
