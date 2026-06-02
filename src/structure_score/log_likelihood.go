package structure_score

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// LogLikelihood implements the pure log-likelihood score with no penalty.
// This is the maximum likelihood score: LL = sum_j sum_k N_jk * log(N_jk / N_j).
// Without a penalty term, this score always favors more complex models.
type LogLikelihood struct{}

// NewLogLikelihood creates a new LogLikelihood scorer.
func NewLogLikelihood() *LogLikelihood {
	return &LogLikelihood{}
}

// LocalScore computes the log-likelihood local score for a variable given its parents.
func (l *LogLikelihood) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}

	counts, parentCounts, _, states := countTable(variable, parents, data)

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

	return ll
}

// Score computes the total log-likelihood score for a set of variables and their parent assignments.
func (l *LogLikelihood) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += l.LocalScore(v, parents, data)
	}
	return total
}
