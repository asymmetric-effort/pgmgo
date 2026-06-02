package inference

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// DBNInference performs inference on Dynamic Bayesian Networks (DBNs).
// A DBN is represented by an initial set of factors (time slice 0),
// transition factors (connecting consecutive time slices), and a set
// of interface nodes that carry information between time steps.
//
// This avoids importing the models package directly; it operates on
// factors and variable names.
type DBNInference struct {
	// initialFactors define the distribution at time step 0.
	initialFactors []*factors.DiscreteFactor

	// transitionFactors define the conditional distribution for
	// time-step t given time-step t-1. Variables at time t-1 use
	// a "_prev" suffix by convention (e.g., "X_prev" for the previous
	// value of "X").
	transitionFactors []*factors.DiscreteFactor

	// interfaceNodes are the variable names that form the interface
	// between time slices (e.g., ["X"]). These are the variables
	// whose beliefs are rolled forward. The transition factors should
	// reference both "X_prev" and "X" for each interface node "X".
	interfaceNodes []string
}

// NewDBNInference creates a new DBNInference engine.
//
// Parameters:
//   - initialFactors: factors defining the prior at t=0.
//   - transitionFactors: factors defining P(t | t-1). Previous-step
//     variables should use a "_prev" suffix.
//   - interfaceNodes: variable names that connect consecutive time slices.
//
// All factors are deep-copied.
func NewDBNInference(
	initialFactors []*factors.DiscreteFactor,
	transitionFactors []*factors.DiscreteFactor,
	interfaceNodes []string,
) *DBNInference {
	initCopy := make([]*factors.DiscreteFactor, len(initialFactors))
	for i, f := range initialFactors {
		initCopy[i] = f.Copy()
	}
	transCopy := make([]*factors.DiscreteFactor, len(transitionFactors))
	for i, f := range transitionFactors {
		transCopy[i] = f.Copy()
	}
	nodes := make([]string, len(interfaceNodes))
	copy(nodes, interfaceNodes)

	return &DBNInference{
		initialFactors:    initCopy,
		transitionFactors: transCopy,
		interfaceNodes:    nodes,
	}
}

// ForwardInference performs filtering (forward inference) over a sequence
// of evidence observations, returning the posterior belief over queryVars
// at the final time step.
//
// Algorithm:
//  1. Start with initialFactors at t=0.
//  2. Incorporate evidence at t=0 and run variable elimination to get
//     the belief over the interface nodes.
//  3. For each subsequent time step t:
//     a. Rename interface node beliefs from "X" to "X_prev" to connect
//     with the transition factors.
//     b. Combine the renamed belief with transition factors.
//     c. Incorporate evidence at time t.
//     d. Run variable elimination to get the new interface belief.
//  4. At the final time step, return the query result.
//
// evidenceSequence[t] contains the evidence at time step t (variable
// names without any suffix, matching the current time step).
func (dbn *DBNInference) ForwardInference(
	queryVars []string,
	evidenceSequence []map[string]int,
) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("dbn_inference: queryVars must not be empty")
	}
	if len(evidenceSequence) == 0 {
		return nil, fmt.Errorf("dbn_inference: evidenceSequence must not be empty")
	}

	// Step 1: Process t=0 with initial factors.
	currentFactors := make([]*factors.DiscreteFactor, len(dbn.initialFactors))
	for i, f := range dbn.initialFactors {
		currentFactors[i] = f.Copy()
	}

	nSteps := len(evidenceSequence)

	for t := 0; t < nSteps; t++ {
		evidence := evidenceSequence[t]

		if t == nSteps-1 {
			// Final time step: query the desired variables.
			// Separate evidence on query vars (add as indicator factors)
			// from evidence on non-query vars (pass as VE evidence).
			querySet := make(map[string]bool, len(queryVars))
			for _, v := range queryVars {
				querySet[v] = true
			}

			augFactors := make([]*factors.DiscreteFactor, len(currentFactors))
			for i, f := range currentFactors {
				augFactors[i] = f.Copy()
			}

			cardMap := make(map[string]int)
			for _, f := range currentFactors {
				vars := f.Variables()
				card := f.Cardinality()
				for i, v := range vars {
					if _, ok := cardMap[v]; !ok {
						cardMap[v] = card[i]
					}
				}
			}

			otherEvidence := make(map[string]int)
			for v, val := range evidence {
				if querySet[v] {
					c, ok := cardMap[v]
					if !ok {
						return nil, fmt.Errorf("dbn_inference: evidence variable %q not found in factors", v)
					}
					indicatorVals := make([]float64, c)
					indicatorVals[val] = 1.0
					indicator, err := factors.NewDiscreteFactor([]string{v}, []int{c}, indicatorVals)
					if err != nil {
						return nil, fmt.Errorf("dbn_inference: failed to create indicator for %q: %w", v, err)
					}
					augFactors = append(augFactors, indicator)
				} else {
					otherEvidence[v] = val
				}
			}

			ve := NewVariableElimination(augFactors)
			result, err := ve.Query(queryVars, otherEvidence)
			if err != nil {
				return nil, fmt.Errorf("dbn_inference: query at final time step %d failed: %w", t, err)
			}
			return result, nil
		}

		// Not the final step: compute interface belief and roll forward.
		interfaceBelief, err := dbn.computeInterfaceBelief(currentFactors, evidence)
		if err != nil {
			return nil, fmt.Errorf("dbn_inference: interface belief at time step %d failed: %w", t, err)
		}

		// Rename interface belief from "X" to "X_prev".
		renamedBelief, err := dbn.renameToPrefix(interfaceBelief, "_prev")
		if err != nil {
			return nil, fmt.Errorf("dbn_inference: renaming at time step %d failed: %w", t, err)
		}

		// Combine renamed belief with transition factors for the next step.
		currentFactors = make([]*factors.DiscreteFactor, 0, 1+len(dbn.transitionFactors))
		currentFactors = append(currentFactors, renamedBelief)
		for _, f := range dbn.transitionFactors {
			currentFactors = append(currentFactors, f.Copy())
		}
	}

	// Should not reach here.
	return nil, fmt.Errorf("dbn_inference: unexpected end of forward inference")
}

// computeInterfaceBelief runs variable elimination on the given factors
// with evidence to compute the joint belief over the interface nodes.
//
// Evidence on interface nodes is handled specially: instead of passing
// them as VE evidence (which would eliminate the variable), we add
// indicator factors and query the interface nodes directly, so the
// result is always a proper factor over the interface variables.
func (dbn *DBNInference) computeInterfaceBelief(
	currentFactors []*factors.DiscreteFactor,
	evidence map[string]int,
) (*factors.DiscreteFactor, error) {
	// Collect all variables present in current factors.
	allVars := collectVariables(currentFactors)

	// Only query interface nodes that are present in the current factors.
	var queryNodes []string
	interfaceSet := make(map[string]bool)
	for _, node := range dbn.interfaceNodes {
		if allVars[node] {
			queryNodes = append(queryNodes, node)
			interfaceSet[node] = true
		}
	}

	if len(queryNodes) == 0 {
		return nil, fmt.Errorf("dbn_inference: no interface nodes found in current factors")
	}

	// Separate evidence into interface-node evidence and other evidence.
	// Interface-node evidence is incorporated as indicator factors.
	otherEvidence := make(map[string]int)
	augmentedFactors := make([]*factors.DiscreteFactor, len(currentFactors))
	for i, f := range currentFactors {
		augmentedFactors[i] = f.Copy()
	}

	// Build cardinality map from current factors.
	cardMap := make(map[string]int)
	for _, f := range currentFactors {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			if _, ok := cardMap[v]; !ok {
				cardMap[v] = card[i]
			}
		}
	}

	for v, val := range evidence {
		if interfaceSet[v] {
			// Create indicator factor for this interface node.
			c, ok := cardMap[v]
			if !ok {
				return nil, fmt.Errorf("dbn_inference: evidence variable %q not found in factors", v)
			}
			indicatorVals := make([]float64, c)
			indicatorVals[val] = 1.0
			indicator, err := factors.NewDiscreteFactor([]string{v}, []int{c}, indicatorVals)
			if err != nil {
				return nil, fmt.Errorf("dbn_inference: failed to create indicator for %q: %w", v, err)
			}
			augmentedFactors = append(augmentedFactors, indicator)
		} else {
			otherEvidence[v] = val
		}
	}

	ve := NewVariableElimination(augmentedFactors)
	result, err := ve.Query(queryNodes, otherEvidence)
	if err != nil {
		return nil, fmt.Errorf("dbn_inference: variable elimination failed: %w", err)
	}

	return result, nil
}

// renameToPrefix creates a new factor with each variable name appended with
// the given suffix. For example, "X" becomes "X_prev".
func (dbn *DBNInference) renameToPrefix(
	f *factors.DiscreteFactor,
	suffix string,
) (*factors.DiscreteFactor, error) {
	vars := f.Variables()
	card := f.Cardinality()
	data := f.Values().Data()

	newVars := make([]string, len(vars))
	for i, v := range vars {
		newVars[i] = v + suffix
	}

	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)

	return factors.NewDiscreteFactor(newVars, card, dataCopy)
}
