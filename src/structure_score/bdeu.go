package structure_score

import (
	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// BDeu implements the Bayesian Dirichlet equivalent uniform score.
// The score uses an equivalent sample size (ESS) as a hyperparameter
// which is distributed uniformly across all parent configurations and states.
type BDeu struct {
	equivalentSampleSize float64
}

// NewBDeu creates a new BDeu scorer with the given equivalent sample size.
func NewBDeu(ess float64) *BDeu {
	return &BDeu{equivalentSampleSize: ess}
}

// LocalScore computes the BDeu local score for a variable given its parents.
// Score = sum_j [ lgamma(alpha_j) - lgamma(alpha_j + N_j) +
//
//	sum_k [ lgamma(alpha_jk + N_jk) - lgamma(alpha_jk) ] ]
//
// where alpha_jk = ESS / (numParentConfigs * numStates)
// and alpha_j = sum_k alpha_jk = ESS / numParentConfigs.
func (b *BDeu) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	counts, parentCounts, card, states := countTable(variable, parents, data)
	numParentCfgs := numParentConfigurations(parents, data)

	alphaJK := b.equivalentSampleSize / float64(numParentCfgs*card)
	alphaJ := b.equivalentSampleSize / float64(numParentCfgs)

	allConfigs := allParentConfigs(parents, data)

	score := 0.0
	for _, key := range allConfigs {
		nj := parentCounts[key] // 0 if not present
		score += scigo.Gammaln(alphaJ) - scigo.Gammaln(alphaJ+float64(nj))

		for _, state := range states {
			njk := 0
			if counts[key] != nil {
				njk = counts[key][state]
			}
			score += scigo.Gammaln(alphaJK+float64(njk)) - scigo.Gammaln(alphaJK)
		}
	}

	return score
}

// Score computes the total BDeu score for a set of variables and their parent assignments.
func (b *BDeu) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += b.LocalScore(v, parents, data)
	}
	return total
}
