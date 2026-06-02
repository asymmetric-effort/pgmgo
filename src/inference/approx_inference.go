package inference

import (
	"fmt"
	"math/rand"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// ApproxInference estimates marginal distributions via likelihood-weighted
// sampling. It operates on a set of discrete factors (as obtained from a
// Bayesian network's ToMarkovFactors or constructed directly), similar to
// VariableElimination, but trades exactness for scalability.
type ApproxInference struct {
	factors []*factors.DiscreteFactor
	rng     *rand.Rand
}

// NewApproxInference creates a new ApproxInference engine from the given
// factor list. Each factor is deep-copied so the caller's originals are not
// modified during inference. The seed controls the random number generator
// for reproducibility.
func NewApproxInference(factorList []*factors.DiscreteFactor, seed int64) *ApproxInference {
	copied := make([]*factors.DiscreteFactor, len(factorList))
	for i, f := range factorList {
		copied[i] = f.Copy()
	}
	return &ApproxInference{
		factors: copied,
		rng:     rand.New(rand.NewSource(seed)),
	}
}

// Query approximates P(queryVars | evidence) using likelihood-weighted
// sampling.
//
// Steps:
//  1. Collect all variables and their cardinalities from all factors.
//  2. Reduce factors by evidence to obtain conditional factors.
//  3. For nSamples iterations, draw a uniform sample over all non-evidence
//     variables, compute the weight as the product of all (reduced) factor
//     values at that assignment, and accumulate the weighted counts for the
//     query variable assignments.
//  4. Normalize the accumulated counts to get an approximate marginal.
//  5. Return the result as a DiscreteFactor.
func (ai *ApproxInference) Query(queryVars []string, evidence map[string]int, nSamples int) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("approx_inference: queryVars must not be empty")
	}
	if nSamples <= 0 {
		return nil, fmt.Errorf("approx_inference: nSamples must be positive")
	}

	// Step 1: Collect all variables and cardinalities.
	cardMap := make(map[string]int)
	for _, f := range ai.factors {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			if _, ok := cardMap[v]; !ok {
				cardMap[v] = card[i]
			}
		}
	}

	// Validate query variables exist.
	for _, v := range queryVars {
		if _, ok := cardMap[v]; !ok {
			return nil, fmt.Errorf("approx_inference: query variable %q not found in any factor", v)
		}
	}

	// Validate evidence variables exist and values are in range.
	for v, val := range evidence {
		c, ok := cardMap[v]
		if !ok {
			return nil, fmt.Errorf("approx_inference: evidence variable %q not found in any factor", v)
		}
		if val < 0 || val >= c {
			return nil, fmt.Errorf("approx_inference: evidence value %d out of range for variable %q (card %d)", val, v, c)
		}
	}

	// Step 2: Reduce factors by evidence.
	workingFactors, err := reduceAll(ai.factors, evidence)
	if err != nil {
		return nil, fmt.Errorf("approx_inference: evidence reduction failed: %w", err)
	}

	// Build the list of free (non-evidence) variables to sample over.
	var freeVars []string
	var freeCard []int
	for _, f := range ai.factors {
		for _, v := range f.Variables() {
			if _, isEvidence := evidence[v]; isEvidence {
				continue
			}
			// Only add if not already seen.
			found := false
			for _, fv := range freeVars {
				if fv == v {
					found = true
					break
				}
			}
			if !found {
				freeVars = append(freeVars, v)
				freeCard = append(freeCard, cardMap[v])
			}
		}
	}

	// Build the query variable cardinalities and index positions.
	queryCard := make([]int, len(queryVars))
	for i, v := range queryVars {
		queryCard[i] = cardMap[v]
	}

	// Compute total size of the query result table.
	querySize := 1
	for _, c := range queryCard {
		querySize *= c
	}
	counts := make([]float64, querySize)

	// Step 3: Likelihood-weighted sampling.
	assignment := make(map[string]int, len(freeVars))
	for s := 0; s < nSamples; s++ {
		// Draw a uniform random assignment for all free variables.
		for i, v := range freeVars {
			assignment[v] = ai.rng.Intn(freeCard[i])
		}

		// Compute the weight: product of all reduced factor values at this
		// assignment.
		weight := 1.0
		for _, f := range workingFactors {
			fVars := f.Variables()
			fAssign := make(map[string]int, len(fVars))
			for _, fv := range fVars {
				fAssign[fv] = assignment[fv]
			}
			weight *= f.GetValue(fAssign)
		}

		if weight == 0 {
			continue
		}

		// Compute the flat index for the query variables.
		queryFlat := 0
		stride := 1
		for i := len(queryVars) - 1; i >= 0; i-- {
			queryFlat += assignment[queryVars[i]] * stride
			stride *= queryCard[i]
		}
		counts[queryFlat] += weight
	}

	// Step 4: Normalize.
	sum := 0.0
	for _, c := range counts {
		sum += c
	}
	if sum == 0 {
		return nil, fmt.Errorf("approx_inference: all samples had zero weight; try increasing nSamples")
	}
	for i := range counts {
		counts[i] /= sum
	}

	// Step 5: Return as DiscreteFactor.
	return factors.NewDiscreteFactor(queryVars, queryCard, counts)
}
