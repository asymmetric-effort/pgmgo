package metrics

import "github.com/asymmetric-effort/pgmgo/lib/tabgo"

// LocalScorer is the interface expected by StructureScoreMetric.
// It matches the LocalScore method of structure_score.StructureScore.
type LocalScorer interface {
	LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64
}

// StructureScoreMetric evaluates a DAG structure by summing local scores over
// all variables. Each variable's local score is computed from the variable,
// its parents, and the data using the provided scorer.
func StructureScoreMetric(
	variables []string,
	parentMap map[string][]string,
	data *tabgo.DataFrame,
	scorer LocalScorer,
) float64 {
	total := 0.0
	for _, v := range variables {
		parents := parentMap[v]
		total += scorer.LocalScore(v, parents, data)
	}
	return total
}
