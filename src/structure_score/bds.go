package structure_score

import (
	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// BDs implements the Bayesian Dirichlet sparse score (BDs).
// Like BDeu, it uses Dirichlet priors, but with pseudo-counts that favor
// sparse structures by concentrating the prior mass on fewer parent configurations.
//
// The structure prior penalizes complex structures:
//
//	score = BDeu_score - structurePrior * |parents|
//
// The pseudo-counts use a smaller equivalent sample size that is further divided
// by the number of parent configurations, which gives stronger preference to
// simpler structures compared to BDeu.
type BDs struct {
	equivalentSampleSize float64
	structurePrior       float64
}

// NewBDs creates a new BDs scorer with the given equivalent sample size and
// structure prior penalty. The structurePrior is multiplied by the number of
// parents and subtracted from the score (higher values penalize more parents).
func NewBDs(ess float64, structurePrior float64) *BDs {
	return &BDs{
		equivalentSampleSize: ess,
		structurePrior:       structurePrior,
	}
}

// LocalScore computes the BDs local score for a variable given its parents.
//
// The score uses BDeu-style computation but with sparse pseudo-counts:
//
//	alpha_jk = ESS / (numParentConfigs^2 * numStates)
//
// This quadratic scaling of parent configurations encourages sparsity.
// A structure prior penalty of structurePrior * |parents| is also applied.
func (b *BDs) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	counts, parentCounts, card, states := countTable(variable, parents, data)
	numParentCfgs := numParentConfigurations(parents, data)

	// Sparse pseudo-counts: divide by numParentCfgs^2 instead of numParentCfgs.
	// This makes the prior weaker for complex parent sets.
	alphaJK := b.equivalentSampleSize / float64(numParentCfgs*numParentCfgs*card)
	alphaJ := alphaJK * float64(card)

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

	// Structure prior penalty.
	score -= b.structurePrior * float64(len(parents))

	return score
}

// Score computes the total BDs score for a set of variables and their parent assignments.
func (b *BDs) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += b.LocalScore(v, parents, data)
	}
	return total
}
