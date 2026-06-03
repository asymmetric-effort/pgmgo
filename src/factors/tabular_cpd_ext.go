package factors

import (
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// GetValues returns the CPD values as a 2D slice where rows correspond to
// child variable states and columns correspond to parent configurations.
func (cpd *TabularCPD) GetValues() [][]float64 {
	numParentConfigs := 1
	for _, ec := range cpd.evidenceCard {
		numParentConfigs *= ec
	}
	data := cpd.factor.values.Data()
	result := make([][]float64, cpd.variableCard)
	for i := 0; i < cpd.variableCard; i++ {
		row := make([]float64, numParentConfigs)
		for j := 0; j < numParentConfigs; j++ {
			row[j] = data[i*numParentConfigs+j]
		}
		result[i] = row
	}
	return result
}

// Normalize returns a new TabularCPD where each column sums to 1.
// If a column sums to zero it is left as-is.
func (cpd *TabularCPD) Normalize() *TabularCPD {
	vals := cpd.GetValues()
	numParentConfigs := 1
	for _, ec := range cpd.evidenceCard {
		numParentConfigs *= ec
	}

	newVals := make([][]float64, cpd.variableCard)
	for i := range newVals {
		newVals[i] = make([]float64, numParentConfigs)
	}

	for j := 0; j < numParentConfigs; j++ {
		sum := 0.0
		for i := 0; i < cpd.variableCard; i++ {
			sum += vals[i][j]
		}
		for i := 0; i < cpd.variableCard; i++ {
			if sum == 0 {
				newVals[i][j] = vals[i][j]
			} else {
				newVals[i][j] = vals[i][j] / sum
			}
		}
	}

	ev := make([]string, len(cpd.evidence))
	copy(ev, cpd.evidence)
	ec := make([]int, len(cpd.evidenceCard))
	copy(ec, cpd.evidenceCard)

	result, _ := NewTabularCPD(cpd.variable, cpd.variableCard, newVals, ev, ec)
	return result
}

// Marginalize sums out the specified variables from the CPD. The variables
// must be evidence (parent) variables. Returns a new TabularCPD with the
// remaining evidence variables.
func (cpd *TabularCPD) Marginalize(variables []string) (*TabularCPD, error) {
	if len(variables) == 0 {
		return cpd.Copy(), nil
	}

	// Validate: all variables must be evidence variables, not the child.
	margSet := make(map[string]bool, len(variables))
	for _, v := range variables {
		if v == cpd.variable {
			return nil, fmt.Errorf("factors: cannot marginalize the child variable %q from a CPD", v)
		}
		found := false
		for _, ev := range cpd.evidence {
			if ev == v {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("factors: variable %q not in CPD evidence", v)
		}
		margSet[v] = true
	}

	// Marginalize the underlying factor over the specified variables.
	margFactor, err := cpd.factor.Marginalize(variables)
	if err != nil {
		return nil, err
	}

	// Build new evidence and evidenceCard.
	var newEvidence []string
	var newEvidenceCard []int
	for i, ev := range cpd.evidence {
		if !margSet[ev] {
			newEvidence = append(newEvidence, ev)
			newEvidenceCard = append(newEvidenceCard, cpd.evidenceCard[i])
		}
	}

	// Extract values from the marginalized factor in the right shape.
	numParentConfigs := 1
	for _, ec := range newEvidenceCard {
		numParentConfigs *= ec
	}

	data := margFactor.values.Data()
	vals := make([][]float64, cpd.variableCard)
	for i := 0; i < cpd.variableCard; i++ {
		row := make([]float64, numParentConfigs)
		for j := 0; j < numParentConfigs; j++ {
			row[j] = data[i*numParentConfigs+j]
		}
		vals[i] = row
	}

	return NewTabularCPD(cpd.variable, cpd.variableCard, vals, newEvidence, newEvidenceCard)
}

// Reduce fixes evidence variables to specific values and returns a new
// TabularCPD with those variables removed from evidence.
func (cpd *TabularCPD) Reduce(values map[string]int) (*TabularCPD, error) {
	if len(values) == 0 {
		return cpd.Copy(), nil
	}

	// Validate: all keys must be evidence variables.
	for v := range values {
		if v == cpd.variable {
			return nil, fmt.Errorf("factors: cannot reduce the child variable %q", v)
		}
		found := false
		for _, ev := range cpd.evidence {
			if ev == v {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("factors: variable %q not in CPD evidence", v)
		}
	}

	// Reduce the underlying factor.
	reducedFactor, err := cpd.factor.Reduce(values)
	if err != nil {
		return nil, err
	}

	// Build new evidence.
	var newEvidence []string
	var newEvidenceCard []int
	for i, ev := range cpd.evidence {
		if _, ok := values[ev]; !ok {
			newEvidence = append(newEvidence, ev)
			newEvidenceCard = append(newEvidenceCard, cpd.evidenceCard[i])
		}
	}

	numParentConfigs := 1
	for _, ec := range newEvidenceCard {
		numParentConfigs *= ec
	}

	data := reducedFactor.values.Data()
	vals := make([][]float64, cpd.variableCard)
	for i := 0; i < cpd.variableCard; i++ {
		row := make([]float64, numParentConfigs)
		for j := 0; j < numParentConfigs; j++ {
			row[j] = data[i*numParentConfigs+j]
		}
		vals[i] = row
	}

	return NewTabularCPD(cpd.variable, cpd.variableCard, vals, newEvidence, newEvidenceCard)
}

// ReorderParents returns a new TabularCPD with parent variables reordered
// according to newOrder. newOrder must be a permutation of the current
// evidence variables.
func (cpd *TabularCPD) ReorderParents(newOrder []string) (*TabularCPD, error) {
	if len(newOrder) != len(cpd.evidence) {
		return nil, fmt.Errorf("factors: newOrder length %d != evidence length %d",
			len(newOrder), len(cpd.evidence))
	}

	// Build a map from evidence name to index.
	evIdx := make(map[string]int, len(cpd.evidence))
	for i, ev := range cpd.evidence {
		evIdx[ev] = i
	}

	// Validate newOrder is a permutation.
	perm := make([]int, len(newOrder))
	seen := make(map[string]bool, len(newOrder))
	for i, v := range newOrder {
		idx, ok := evIdx[v]
		if !ok {
			return nil, fmt.Errorf("factors: variable %q not in evidence", v)
		}
		if seen[v] {
			return nil, fmt.Errorf("factors: duplicate variable %q in newOrder", v)
		}
		seen[v] = true
		perm[i] = idx
	}

	// Build new evidence card.
	newEvidenceCard := make([]int, len(newOrder))
	for i, p := range perm {
		newEvidenceCard[i] = cpd.evidenceCard[p]
	}

	oldNumParentConfigs := 1
	for _, ec := range cpd.evidenceCard {
		oldNumParentConfigs *= ec
	}
	newNumParentConfigs := oldNumParentConfigs // same total

	// Compute old strides for evidence variables.
	oldStrides := make([]int, len(cpd.evidence))
	if len(cpd.evidence) > 0 {
		oldStrides[len(cpd.evidence)-1] = 1
		for i := len(cpd.evidence) - 2; i >= 0; i-- {
			oldStrides[i] = oldStrides[i+1] * cpd.evidenceCard[i+1]
		}
	}

	// Compute new strides.
	newStrides := make([]int, len(newOrder))
	if len(newOrder) > 0 {
		newStrides[len(newOrder)-1] = 1
		for i := len(newOrder) - 2; i >= 0; i-- {
			newStrides[i] = newStrides[i+1] * newEvidenceCard[i+1]
		}
	}

	oldVals := cpd.GetValues()
	newVals := make([][]float64, cpd.variableCard)
	for i := range newVals {
		newVals[i] = make([]float64, newNumParentConfigs)
	}

	// For each new parent config, compute the corresponding old parent config.
	for newFlat := 0; newFlat < newNumParentConfigs; newFlat++ {
		// Decompose newFlat into new evidence indices.
		newIndices := make([]int, len(newOrder))
		rem := newFlat
		for i := len(newOrder) - 1; i >= 0; i-- {
			newIndices[i] = rem % newEvidenceCard[i]
			rem /= newEvidenceCard[i]
		}

		// Map to old flat index.
		oldFlat := 0
		for i, ni := range newIndices {
			oldEvIdx := perm[i]
			oldFlat += ni * oldStrides[oldEvIdx]
		}

		for childState := 0; childState < cpd.variableCard; childState++ {
			newVals[childState][newFlat] = oldVals[childState][oldFlat]
		}
	}

	newEvidence := make([]string, len(newOrder))
	copy(newEvidence, newOrder)

	return NewTabularCPD(cpd.variable, cpd.variableCard, newVals, newEvidence, newEvidenceCard)
}

// String returns a pretty-printed table representation of the CPD.
func (cpd *TabularCPD) String() string {
	var b strings.Builder
	vals := cpd.GetValues()

	numParentConfigs := 1
	for _, ec := range cpd.evidenceCard {
		numParentConfigs *= ec
	}

	b.WriteString(fmt.Sprintf("TabularCPD(%s)\n", cpd.variable))

	if len(cpd.evidence) > 0 {
		// Header row: evidence variable names and their state indices.
		// Compute evidence state for each parent config column.
		evStrides := make([]int, len(cpd.evidence))
		if len(cpd.evidence) > 0 {
			evStrides[len(cpd.evidence)-1] = 1
			for i := len(cpd.evidence) - 2; i >= 0; i-- {
				evStrides[i] = evStrides[i+1] * cpd.evidenceCard[i+1]
			}
		}

		for ei, ev := range cpd.evidence {
			b.WriteString(fmt.Sprintf("%-12s", ev))
			for j := 0; j < numParentConfigs; j++ {
				state := (j / evStrides[ei]) % cpd.evidenceCard[ei]
				b.WriteString(fmt.Sprintf("  %6d", state))
			}
			b.WriteString("\n")
		}
		b.WriteString(strings.Repeat("-", 12+numParentConfigs*8) + "\n")
	}

	for i := 0; i < cpd.variableCard; i++ {
		b.WriteString(fmt.Sprintf("%-12s", fmt.Sprintf("%s=%d", cpd.variable, i)))
		for j := 0; j < numParentConfigs; j++ {
			b.WriteString(fmt.Sprintf("  %6.4f", vals[i][j]))
		}
		b.WriteString("\n")
	}

	return b.String()
}

// ToCSV writes the CPD table to a CSV file.
func (cpd *TabularCPD) ToCSV(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("factors: failed to create file %q: %w", filename, err)
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	numParentConfigs := 1
	for _, ec := range cpd.evidenceCard {
		numParentConfigs *= ec
	}

	// Compute evidence strides.
	evStrides := make([]int, len(cpd.evidence))
	if len(cpd.evidence) > 0 {
		evStrides[len(cpd.evidence)-1] = 1
		for i := len(cpd.evidence) - 2; i >= 0; i-- {
			evStrides[i] = evStrides[i+1] * cpd.evidenceCard[i+1]
		}
	}

	// Write evidence header rows.
	for ei, ev := range cpd.evidence {
		row := make([]string, 1+numParentConfigs)
		row[0] = ev
		for j := 0; j < numParentConfigs; j++ {
			state := (j / evStrides[ei]) % cpd.evidenceCard[ei]
			row[j+1] = strconv.Itoa(state)
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	// Write value rows.
	vals := cpd.GetValues()
	for i := 0; i < cpd.variableCard; i++ {
		row := make([]string, 1+numParentConfigs)
		row[0] = fmt.Sprintf("%s=%d", cpd.variable, i)
		for j := 0; j < numParentConfigs; j++ {
			row[j+1] = strconv.FormatFloat(vals[i][j], 'f', -1, 64)
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// GetRandom generates a TabularCPD with random values (each column normalized
// to sum to 1) using the provided seed. If seed is 0, a default random source
// is used.
func GetRandom(variable string, variableCard int, evidence []string, evidenceCard []int, seed int64) (*TabularCPD, error) {
	if variableCard <= 0 {
		return nil, fmt.Errorf("factors: variableCard must be positive, got %d", variableCard)
	}
	if len(evidence) != len(evidenceCard) {
		return nil, fmt.Errorf("factors: evidence length %d != evidenceCard length %d",
			len(evidence), len(evidenceCard))
	}

	numParentConfigs := 1
	for _, ec := range evidenceCard {
		if ec <= 0 {
			return nil, fmt.Errorf("factors: evidence cardinality must be positive, got %d", ec)
		}
		numParentConfigs *= ec
	}

	rng := rand.New(rand.NewSource(seed))

	vals := make([][]float64, variableCard)
	for i := range vals {
		vals[i] = make([]float64, numParentConfigs)
	}

	for j := 0; j < numParentConfigs; j++ {
		sum := 0.0
		for i := 0; i < variableCard; i++ {
			v := rng.Float64()
			vals[i][j] = v
			sum += v
		}
		for i := 0; i < variableCard; i++ {
			vals[i][j] /= sum
		}
	}

	return NewTabularCPD(variable, variableCard, vals, evidence, evidenceCard)
}

// GetUniform generates a TabularCPD with uniform values (each column has
// equal probabilities summing to 1).
func GetUniform(variable string, variableCard int, evidence []string, evidenceCard []int) (*TabularCPD, error) {
	if variableCard <= 0 {
		return nil, fmt.Errorf("factors: variableCard must be positive, got %d", variableCard)
	}
	if len(evidence) != len(evidenceCard) {
		return nil, fmt.Errorf("factors: evidence length %d != evidenceCard length %d",
			len(evidence), len(evidenceCard))
	}

	numParentConfigs := 1
	for _, ec := range evidenceCard {
		if ec <= 0 {
			return nil, fmt.Errorf("factors: evidence cardinality must be positive, got %d", ec)
		}
		numParentConfigs *= ec
	}

	p := 1.0 / float64(variableCard)
	vals := make([][]float64, variableCard)
	for i := range vals {
		row := make([]float64, numParentConfigs)
		for j := range row {
			row[j] = p
		}
		vals[i] = row
	}

	return NewTabularCPD(variable, variableCard, vals, evidence, evidenceCard)
}

// ToDataFrame converts the CPD to a DataFrame. Each column corresponds to a
// parent configuration (column header encodes the evidence combination).
// Rows correspond to the variable states.
func (cpd *TabularCPD) ToDataFrame() *tabgo.DataFrame {
	vals := cpd.GetValues()

	numParentConfigs := 1
	for _, ec := range cpd.evidenceCard {
		numParentConfigs *= ec
	}

	// Compute evidence strides for column headers.
	evStrides := make([]int, len(cpd.evidence))
	if len(cpd.evidence) > 0 {
		evStrides[len(cpd.evidence)-1] = 1
		for i := len(cpd.evidence) - 2; i >= 0; i-- {
			evStrides[i] = evStrides[i+1] * cpd.evidenceCard[i+1]
		}
	}

	// Build column names from evidence combinations.
	colNames := make([]string, numParentConfigs)
	for j := 0; j < numParentConfigs; j++ {
		if len(cpd.evidence) == 0 {
			colNames[j] = cpd.variable
		} else {
			parts := make([]string, len(cpd.evidence))
			for ei, ev := range cpd.evidence {
				state := (j / evStrides[ei]) % cpd.evidenceCard[ei]
				parts[ei] = fmt.Sprintf("%s=%d", ev, state)
			}
			colNames[j] = strings.Join(parts, ",")
		}
	}

	// Build rows: each row is one state of the child variable.
	rows := make([][]any, cpd.variableCard)
	for i := 0; i < cpd.variableCard; i++ {
		row := make([]any, numParentConfigs)
		for j := 0; j < numParentConfigs; j++ {
			row[j] = vals[i][j]
		}
		rows[i] = row
	}

	return tabgo.NewDataFrameFromRows(colNames, rows)
}

// Repr returns a detailed string representation of the CPD, including
// type information, variable, evidence, and cardinalities.
func (cpd *TabularCPD) Repr() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("TabularCPD(variable=%q, variableCard=%d", cpd.variable, cpd.variableCard))
	if len(cpd.evidence) > 0 {
		b.WriteString(fmt.Sprintf(", evidence=%v, evidenceCard=%v", cpd.evidence, cpd.evidenceCard))
	}
	b.WriteString(")\n")
	b.WriteString(cpd.String())
	return b.String()
}
