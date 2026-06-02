package structure_score

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// StructureScore defines the interface for scoring structures in structure learning.
type StructureScore interface {
	// LocalScore computes the score contribution of a single variable given its parents.
	LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64

	// Score computes the total score for a set of variables and their parent assignments.
	Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64
}

// countTable builds a table of counts for (parentConfig, variableValue) combinations.
// It returns:
//   - counts: map[parentConfigKey][variableValue] -> count
//   - parentCounts: map[parentConfigKey] -> total count for that config
//   - card: number of distinct values of the variable
//   - numParentConfigs: number of distinct parent configurations observed
func countTable(variable string, parents []string, data *tabgo.DataFrame) (
	counts map[string]map[any]int,
	parentCounts map[string]int,
	card int,
	states []any,
) {
	n := data.Len()
	varVals := data.Column(variable).Values()
	states = data.Column(variable).Unique()
	card = len(states)

	// Sort parents for deterministic config keys.
	sortedParents := make([]string, len(parents))
	copy(sortedParents, parents)
	sort.Strings(sortedParents)

	// Pre-fetch parent column values.
	parentVals := make([][]any, len(sortedParents))
	for i, p := range sortedParents {
		parentVals[i] = data.Column(p).Values()
	}

	counts = make(map[string]map[any]int)
	parentCounts = make(map[string]int)

	for row := 0; row < n; row++ {
		// Build parent config key.
		key := parentConfigKey(sortedParents, parentVals, row)
		varVal := varVals[row]

		if counts[key] == nil {
			counts[key] = make(map[any]int)
		}
		counts[key][varVal]++
		parentCounts[key]++
	}

	return counts, parentCounts, card, states
}

// parentConfigKey builds a string key representing the parent configuration for a given row.
func parentConfigKey(parents []string, parentVals [][]any, row int) string {
	if len(parents) == 0 {
		return ""
	}
	key := ""
	for i, p := range parents {
		if i > 0 {
			key += ","
		}
		key += fmt.Sprintf("%s=%v", p, parentVals[i][row])
	}
	return key
}

// allParentConfigs enumerates all possible parent configurations from the data.
// This ensures we account for configs that may have zero counts for some variable states.
func allParentConfigs(parents []string, data *tabgo.DataFrame) []string {
	if len(parents) == 0 {
		return []string{""}
	}

	sortedParents := make([]string, len(parents))
	copy(sortedParents, parents)
	sort.Strings(sortedParents)

	// Get unique values for each parent.
	parentUniques := make([][]any, len(sortedParents))
	for i, p := range sortedParents {
		parentUniques[i] = data.Column(p).Unique()
	}

	// Enumerate all combinations via cross product.
	var configs []string
	var enumerate func(depth int, current string)
	enumerate = func(depth int, current string) {
		if depth == len(sortedParents) {
			configs = append(configs, current)
			return
		}
		for _, val := range parentUniques[depth] {
			prefix := current
			if depth > 0 {
				prefix += ","
			}
			prefix += fmt.Sprintf("%s=%v", sortedParents[depth], val)
			enumerate(depth+1, prefix)
		}
	}
	enumerate(0, "")
	return configs
}

// numParentConfigurations computes the number of possible parent configurations
// as the product of cardinalities of each parent variable.
func numParentConfigurations(parents []string, data *tabgo.DataFrame) int {
	if len(parents) == 0 {
		return 1
	}
	n := 1
	for _, p := range parents {
		n *= data.Column(p).NUnique()
	}
	return n
}
