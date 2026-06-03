package factors

import (
	"fmt"
	"math"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// Equals checks whether two DiscreteFactors are approximately equal. Two
// factors are equal if they have the same variables (in the same order) and
// the same cardinalities, and every corresponding value differs by at most
// atol.
func (f *DiscreteFactor) Equals(other *DiscreteFactor, atol float64) bool {
	if other == nil {
		return false
	}
	if len(f.variables) != len(other.variables) {
		return false
	}
	for i := range f.variables {
		if f.variables[i] != other.variables[i] {
			return false
		}
		if f.cardinality[i] != other.cardinality[i] {
			return false
		}
	}
	fData := f.values.Data()
	oData := other.values.Data()
	if len(fData) != len(oData) {
		return false
	}
	for i := range fData {
		if math.Abs(fData[i]-oData[i]) > atol {
			return false
		}
	}
	return true
}

// ToDataFrame converts the DiscreteFactor to a DataFrame representation.
// Each row represents a complete assignment with one column per variable
// plus a "value" column for the factor value.
func (f *DiscreteFactor) ToDataFrame() *tabgo.DataFrame {
	totalSize := f.totalSize()
	colNames := make([]string, 0, len(f.variables)+1)
	colNames = append(colNames, f.variables...)
	colNames = append(colNames, "value")

	rows := make([][]any, totalSize)
	data := f.values.Data()
	for flat := 0; flat < totalSize; flat++ {
		assignment := f.flatToAssignment(flat)
		row := make([]any, len(f.variables)+1)
		for i, v := range f.variables {
			row[i] = assignment[v]
		}
		row[len(f.variables)] = data[flat]
		rows[flat] = row
	}
	return tabgo.NewDataFrameFromRows(colNames, rows)
}

// Divide divides this factor by other element-wise. Variables of other must
// be a subset of this factor's variables. Division by zero produces zero.
func (f *DiscreteFactor) Divide(other *DiscreteFactor) (*DiscreteFactor, error) {
	return FactorDivide(f, other)
}

// Product multiplies this factor with other. Returns a new factor whose
// variables are the union of both factors' variables.
func (f *DiscreteFactor) Product(other *DiscreteFactor) (*DiscreteFactor, error) {
	return pairwiseProduct(f, other)
}

// Scope returns the list of variable names (alias for Variables).
func (f *DiscreteFactor) Scope() []string {
	return f.Variables()
}

// StateNames returns a map from variable name to a list of state name strings.
// Since DiscreteFactor does not store state names, it returns integer labels.
func (f *DiscreteFactor) StateNames() map[string][]string {
	result := make(map[string][]string, len(f.variables))
	for i, v := range f.variables {
		states := make([]string, f.cardinality[i])
		for s := 0; s < f.cardinality[i]; s++ {
			states[s] = fmt.Sprintf("%d", s)
		}
		result[v] = states
	}
	return result
}

// TabularCPD parity methods.

// StateNames returns a map from each variable (child and evidence) to its
// state names. Since TabularCPD does not store state names, integer labels
// are used.
func (cpd *TabularCPD) StateNames() map[string][]string {
	result := make(map[string][]string, 1+len(cpd.evidence))
	childStates := make([]string, cpd.variableCard)
	for i := 0; i < cpd.variableCard; i++ {
		childStates[i] = fmt.Sprintf("%d", i)
	}
	result[cpd.variable] = childStates
	for i, ev := range cpd.evidence {
		states := make([]string, cpd.evidenceCard[i])
		for s := 0; s < cpd.evidenceCard[i]; s++ {
			states[s] = fmt.Sprintf("%d", s)
		}
		result[ev] = states
	}
	return result
}

// Scope returns all variables: child followed by evidence.
func (cpd *TabularCPD) Scope() []string {
	result := make([]string, 0, 1+len(cpd.evidence))
	result = append(result, cpd.variable)
	result = append(result, cpd.evidence...)
	return result
}

// FactorSet parity methods.

// Factors returns a copy of the factor slice.
func (fs *FactorSet) Factors() []*DiscreteFactor {
	result := make([]*DiscreteFactor, len(fs.factors))
	copy(result, fs.factors)
	return result
}

// Marginalize sums out the given variable from all factors that contain it,
// and replaces them with the marginalized result. Returns a new FactorSet.
func (fs *FactorSet) Marginalize(variable string) (*FactorSet, error) {
	result := NewFactorSet()
	for _, f := range fs.factors {
		if f.varIndex(variable) >= 0 {
			// Check if it's the only variable
			if len(f.variables) == 1 {
				// Skip: factor over only this variable; marginalization removes it.
				continue
			}
			marg, err := f.Marginalize([]string{variable})
			if err != nil {
				return nil, fmt.Errorf("factors: FactorSet.Marginalize: %w", err)
			}
			result.Add(marg)
		} else {
			result.Add(f.Copy())
		}
	}
	return result, nil
}

// Union returns a new FactorSet containing all factors from both sets.
func (fs *FactorSet) Union(other *FactorSet) *FactorSet {
	result := NewFactorSet()
	for _, f := range fs.factors {
		result.Add(f)
	}
	if other != nil {
		for _, f := range other.factors {
			result.Add(f)
		}
	}
	return result
}

// String returns a human-readable representation of the FactorSet.
func (fs *FactorSet) String() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("FactorSet(%d factors)\n", len(fs.factors)))
	for i, f := range fs.factors {
		b.WriteString(fmt.Sprintf("  [%d] %s\n", i, f.String()))
	}
	return b.String()
}

// FactorDict parity methods.

// Has returns true if the given key exists.
func (fd *FactorDict) Has(key string) bool {
	_, ok := fd.entries[key]
	return ok
}

// Delete removes the given key.
func (fd *FactorDict) Delete(key string) {
	delete(fd.entries, key)
}

// Values returns all factors in key-sorted order.
func (fd *FactorDict) Values() []*DiscreteFactor {
	keys := fd.Keys()
	result := make([]*DiscreteFactor, len(keys))
	for i, k := range keys {
		result[i] = fd.entries[k]
	}
	return result
}
