package sampling

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// GibbsSampling implements Gibbs sampling (a special case of MCMC) for
// approximate inference in Bayesian networks.
type GibbsSampling struct {
	bn      *models.BayesianNetwork
	factors []*factors.DiscreteFactor
	rng     *numgo.RNG
}

// NewGibbsSampling creates a new GibbsSampling sampler. The BayesianNetwork
// must pass CheckModel validation. seed controls the RNG for reproducibility.
func NewGibbsSampling(bn *models.BayesianNetwork, seed int64) (*GibbsSampling, error) {
	if bn == nil {
		return nil, fmt.Errorf("sampling: BayesianNetwork must not be nil")
	}
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		return nil, fmt.Errorf("sampling: failed to get Markov factors: %w", err)
	}
	return &GibbsSampling{
		bn:      bn,
		factors: markovFactors,
		rng:     numgo.NewRNG(seed),
	}, nil
}

// Sample runs Gibbs sampling and returns a DataFrame of samples.
//
// Parameters:
//   - n: number of samples to collect
//   - burnIn: number of initial iterations to discard
//   - thinning: collect every thinning-th sample after burn-in
//   - evidence: map of variable name to observed state index (may be nil)
//
// The returned DataFrame has one column per variable in the BN, with n rows.
func (gs *GibbsSampling) Sample(n int, burnIn int, thinning int, evidence map[string]int) (*tabgo.DataFrame, error) {
	if n <= 0 {
		return nil, fmt.Errorf("sampling: n must be positive, got %d", n)
	}
	if burnIn < 0 {
		return nil, fmt.Errorf("sampling: burnIn must be non-negative, got %d", burnIn)
	}
	if thinning <= 0 {
		return nil, fmt.Errorf("sampling: thinning must be positive, got %d", thinning)
	}
	if evidence == nil {
		evidence = make(map[string]int)
	}

	nodes := gs.bn.Nodes() // sorted

	// Build cardinality map from factors.
	cardMap := make(map[string]int)
	for _, f := range gs.factors {
		vars := f.Variables()
		card := f.Cardinality()
		for i, v := range vars {
			cardMap[v] = card[i]
		}
	}

	// Validate evidence variables.
	for v, val := range evidence {
		card, ok := cardMap[v]
		if !ok {
			return nil, fmt.Errorf("sampling: evidence variable %q not found in network", v)
		}
		if val < 0 || val >= card {
			return nil, fmt.Errorf("sampling: evidence value %d out of range for variable %q (cardinality %d)", val, v, card)
		}
	}

	// Determine non-evidence variables.
	var samplingVars []string
	for _, node := range nodes {
		if _, isEvidence := evidence[node]; !isEvidence {
			samplingVars = append(samplingVars, node)
		}
	}

	// Initialize state: evidence is fixed, others are random.
	state := make(map[string]int, len(nodes))
	for v, val := range evidence {
		state[v] = val
	}
	for _, v := range samplingVars {
		card := cardMap[v]
		// Use RNG to pick a random initial state.
		idx := gs.rng.RandInt(0, card, 1).Data()
		state[v] = int(idx[0])
	}

	// Precompute which factors contain each sampling variable.
	varFactors := make(map[string][]*factors.DiscreteFactor, len(samplingVars))
	for _, v := range samplingVars {
		for _, f := range gs.factors {
			for _, fv := range f.Variables() {
				if fv == v {
					varFactors[v] = append(varFactors[v], f)
					break
				}
			}
		}
	}

	// Allocate sample storage.
	totalIter := burnIn + n*thinning
	samples := make([]map[string]int, 0, n)

	for iter := 0; iter < totalIter; iter++ {
		// Sweep over all non-evidence variables.
		for _, v := range samplingVars {
			card := cardMap[v]
			dist := gs.computeFullConditional(v, card, state, varFactors[v])
			state[v] = sampleCategorical(gs.rng, dist)
		}

		// Collect sample after burn-in, at every thinning-th iteration.
		if iter >= burnIn && (iter-burnIn)%thinning == 0 {
			s := make(map[string]int, len(nodes))
			for k, val := range state {
				s[k] = val
			}
			samples = append(samples, s)
		}
	}

	// Build DataFrame.
	sort.Strings(nodes)
	colData := make(map[string]*tabgo.Series, len(nodes))
	for _, v := range nodes {
		vals := make([]any, len(samples))
		for i, s := range samples {
			vals[i] = s[v]
		}
		colData[v] = tabgo.NewSeries(v, vals)
	}
	return tabgo.NewDataFrame(colData), nil
}

// computeFullConditional computes the full conditional distribution of variable
// v given the current state of all other variables. It multiplies all factors
// containing v, reduces by the current assignments of all other variables in
// those factors, and normalizes.
func (gs *GibbsSampling) computeFullConditional(
	v string,
	card int,
	state map[string]int,
	relevantFactors []*factors.DiscreteFactor,
) []float64 {
	dist := make([]float64, card)
	for i := range dist {
		dist[i] = 1.0
	}

	for _, f := range relevantFactors {
		fVars := f.Variables()
		// Build evidence for reduction: all variables in this factor except v.
		reduceEvidence := make(map[string]int)
		for _, fv := range fVars {
			if fv != v {
				reduceEvidence[fv] = state[fv]
			}
		}

		reduced, err := f.Reduce(reduceEvidence)
		if err != nil {
			// Should not happen if model is valid; treat as uniform.
			continue
		}

		// The reduced factor should be over just v. Multiply into dist.
		reducedData := reduced.Values().Data()
		for i := 0; i < card && i < len(reducedData); i++ {
			dist[i] *= reducedData[i]
		}
	}

	// Normalize.
	sum := 0.0
	for _, p := range dist {
		sum += p
	}
	if sum > 0 {
		for i := range dist {
			dist[i] /= sum
		}
	} else {
		// Fallback to uniform if all zero.
		for i := range dist {
			dist[i] = 1.0 / float64(card)
		}
	}

	return dist
}

// sampleCategorical draws a single sample from a categorical distribution
// specified by the probability vector probs. Returns the index of the chosen
// category.
func sampleCategorical(rng *numgo.RNG, probs []float64) int {
	u := rng.Uniform(0, 1, 1).Data()[0]
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if u < cumulative {
			return i
		}
	}
	return len(probs) - 1
}
