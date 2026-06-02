package structure_score

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// AIC implements the Akaike Information Criterion score.
// AIC = LL - k
// where LL is the log-likelihood and k is the number of free parameters.
// Unlike BIC, the penalty does not depend on the sample size.
type AIC struct{}

// NewAIC creates a new AIC scorer.
func NewAIC() *AIC {
	return &AIC{}
}

// LocalScore computes the AIC local score for a variable given its parents.
func (a *AIC) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
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

	// AIC = LL - k
	return ll - k
}

// Score computes the total AIC score for a set of variables and their parent assignments.
func (a *AIC) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += a.LocalScore(v, parents, data)
	}
	return total
}
