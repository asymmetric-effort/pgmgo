package structure_score

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// BIC implements the Bayesian Information Criterion score.
// BIC = LL - 0.5 * k * ln(N)
// where LL is the log-likelihood, k is the number of free parameters,
// and N is the sample size.
type BIC struct{}

// NewBIC creates a new BIC scorer.
func NewBIC() *BIC {
	return &BIC{}
}

// LocalScore computes the BIC local score for a variable given its parents.
func (b *BIC) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}

	counts, parentCounts, card, states := countTable(variable, parents, data)
	numParentCfgs := numParentConfigurations(parents, data)

	// Log-likelihood: sum over all (j, k) of N_jk * log(N_jk / N_j)
	ll := 0.0
	for key, nj := range parentCounts {
		njFloat := float64(nj)
		for _, state := range states {
			njk := counts[key][state]
			if njk > 0 {
				ll += float64(njk) * math.Log(float64(njk)/njFloat)
			}
		}
	}

	// Number of free parameters: (card - 1) * numParentConfigs
	k := float64((card - 1) * numParentCfgs)

	// BIC = LL - 0.5 * k * ln(N)
	return ll - 0.5*k*math.Log(N)
}

// Score computes the total BIC score for a set of variables and their parent assignments.
func (b *BIC) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += b.LocalScore(v, parents, data)
	}
	return total
}
