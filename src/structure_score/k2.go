package structure_score

import (
	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// K2 implements the K2 score (Cooper and Herskovits, 1992).
// Score = sum_j [ lgamma(card) - lgamma(N_j + card) + sum_k lgamma(N_jk + 1) ]
type K2 struct{}

// NewK2 creates a new K2 scorer.
func NewK2() *K2 {
	return &K2{}
}

// LocalScore computes the K2 local score for a variable given its parents.
func (k *K2) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	counts, parentCounts, card, states := countTable(variable, parents, data)
	cardFloat := float64(card)

	allConfigs := allParentConfigs(parents, data)

	score := 0.0
	for _, key := range allConfigs {
		nj := parentCounts[key] // 0 if not present

		score += scigo.Gammaln(cardFloat) - scigo.Gammaln(float64(nj)+cardFloat)

		for _, state := range states {
			njk := 0
			if counts[key] != nil {
				njk = counts[key][state]
			}
			score += scigo.Gammaln(float64(njk) + 1)
		}
	}

	return score
}

// Score computes the total K2 score for a set of variables and their parent assignments.
func (k *K2) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += k.LocalScore(v, parents, data)
	}
	return total
}
