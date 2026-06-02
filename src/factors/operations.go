package factors

import "fmt"

// FactorProduct multiplies one or more discrete factors together, aligning on
// shared variables. The result contains the union of all variables. For each
// joint assignment, the result value is the product of the corresponding values
// in all input factors.
func FactorProduct(factors ...*DiscreteFactor) (*DiscreteFactor, error) {
	if len(factors) == 0 {
		return nil, fmt.Errorf("factors: FactorProduct requires at least one factor")
	}
	if len(factors) == 1 {
		return factors[0].Copy(), nil
	}

	// Accumulate pairwise.
	result := factors[0]
	for i := 1; i < len(factors); i++ {
		var err error
		result, err = pairwiseProduct(result, factors[i])
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// FactorDivide divides f1 by f2. The variables of f2 must be a subset of f1's
// variables. For each assignment in f1, the corresponding value in f2 is looked
// up and divided. Division by zero produces a zero in the result.
func FactorDivide(f1, f2 *DiscreteFactor) (*DiscreteFactor, error) {
	if f1 == nil || f2 == nil {
		return nil, fmt.Errorf("factors: FactorDivide requires non-nil factors")
	}

	// Validate that f2's variables are a subset of f1's variables with matching cardinalities.
	f1VarIdx := make(map[string]int, len(f1.variables))
	for i, v := range f1.variables {
		f1VarIdx[v] = i
	}
	for i, v := range f2.variables {
		idx, ok := f1VarIdx[v]
		if !ok {
			return nil, fmt.Errorf("factors: variable %q in f2 is not in f1", v)
		}
		if f1.cardinality[idx] != f2.cardinality[i] {
			return nil, fmt.Errorf("factors: variable %q has cardinality %d in f1 but %d in f2",
				v, f1.cardinality[idx], f2.cardinality[i])
		}
	}

	// Build strides for f2.
	f2VarIdx := make(map[string]int, len(f2.variables))
	for i, v := range f2.variables {
		f2VarIdx[v] = i
	}
	f2Strides := make([]int, len(f2.variables))
	if len(f2.variables) > 0 {
		f2Strides[len(f2.variables)-1] = 1
		for i := len(f2.variables) - 2; i >= 0; i-- {
			f2Strides[i] = f2Strides[i+1] * f2.cardinality[i+1]
		}
	}

	f1Data := f1.values.Data()
	f2Data := f2.values.Data()
	totalSize := f1.totalSize()
	newValues := make([]float64, totalSize)

	for flat := 0; flat < totalSize; flat++ {
		assignment := f1.flatToAssignment(flat)

		// Compute flat index in f2.
		f2Flat := 0
		for i, v := range f2.variables {
			f2Flat += assignment[v] * f2Strides[i]
		}

		denom := f2Data[f2Flat]
		if denom == 0 {
			newValues[flat] = 0
		} else {
			newValues[flat] = f1Data[flat] / denom
		}
	}

	vars := make([]string, len(f1.variables))
	copy(vars, f1.variables)
	card := make([]int, len(f1.cardinality))
	copy(card, f1.cardinality)
	return NewDiscreteFactor(vars, card, newValues)
}

// FactorSumProduct performs sum-product variable elimination. For each variable
// in eliminationOrder, all factors containing that variable are multiplied
// together and the variable is marginalized out. The resulting factor replaces
// those factors in the working set. After all eliminations, the remaining
// factors are multiplied together and returned.
func FactorSumProduct(factors []*DiscreteFactor, eliminationOrder []string) (*DiscreteFactor, error) {
	if len(factors) == 0 {
		return nil, fmt.Errorf("factors: FactorSumProduct requires at least one factor")
	}

	// Work on copies so we don't mutate the caller's slice.
	working := make([]*DiscreteFactor, len(factors))
	for i, f := range factors {
		working[i] = f.Copy()
	}

	for _, elimVar := range eliminationOrder {
		// Partition factors into those that contain elimVar and those that don't.
		var relevant []*DiscreteFactor
		var rest []*DiscreteFactor
		for _, f := range working {
			if f.varIndex(elimVar) >= 0 {
				relevant = append(relevant, f)
			} else {
				rest = append(rest, f)
			}
		}
		if len(relevant) == 0 {
			// Variable not in any factor; skip.
			continue
		}

		// Multiply all relevant factors together.
		product, err := FactorProduct(relevant...)
		if err != nil {
			return nil, fmt.Errorf("factors: FactorSumProduct product step for %q: %w", elimVar, err)
		}

		// Marginalize out the variable.
		marginalized, err := product.Marginalize([]string{elimVar})
		if err != nil {
			return nil, fmt.Errorf("factors: FactorSumProduct marginalize step for %q: %w", elimVar, err)
		}

		// Replace working set.
		working = append(rest, marginalized)
	}

	// Multiply remaining factors.
	if len(working) == 0 {
		return nil, fmt.Errorf("factors: FactorSumProduct produced no remaining factors")
	}
	return FactorProduct(working...)
}

// pairwiseProduct multiplies two factors together.
func pairwiseProduct(a, b *DiscreteFactor) (*DiscreteFactor, error) {
	// Validate shared variables have matching cardinalities.
	bVarCard := make(map[string]int, len(b.variables))
	for i, v := range b.variables {
		bVarCard[v] = b.cardinality[i]
	}
	for i, v := range a.variables {
		if bCard, ok := bVarCard[v]; ok {
			if a.cardinality[i] != bCard {
				return nil, fmt.Errorf("factors: variable %q has cardinality %d in first factor but %d in second",
					v, a.cardinality[i], bCard)
			}
		}
	}

	// Build the union of variables, preserving order: a's variables first,
	// then b's variables not in a.
	aVarSet := make(map[string]bool, len(a.variables))
	for _, v := range a.variables {
		aVarSet[v] = true
	}

	var newVars []string
	var newCard []int
	newVars = append(newVars, a.variables...)
	newCard = append(newCard, a.cardinality...)
	for i, v := range b.variables {
		if !aVarSet[v] {
			newVars = append(newVars, v)
			newCard = append(newCard, b.cardinality[i])
		}
	}

	newSize := 1
	for _, c := range newCard {
		newSize *= c
	}
	newValues := make([]float64, newSize)

	// Build lookup for variable positions in a and b.
	aVarIdx := make(map[string]int, len(a.variables))
	for i, v := range a.variables {
		aVarIdx[v] = i
	}
	bVarIdx := make(map[string]int, len(b.variables))
	for i, v := range b.variables {
		bVarIdx[v] = i
	}

	aData := a.values.Data()
	bData := b.values.Data()

	// Precompute strides for a and b factors.
	aStrides := make([]int, len(a.variables))
	if len(a.variables) > 0 {
		aStrides[len(a.variables)-1] = 1
		for i := len(a.variables) - 2; i >= 0; i-- {
			aStrides[i] = aStrides[i+1] * a.cardinality[i+1]
		}
	}
	bStrides := make([]int, len(b.variables))
	if len(b.variables) > 0 {
		bStrides[len(b.variables)-1] = 1
		for i := len(b.variables) - 2; i >= 0; i-- {
			bStrides[i] = bStrides[i+1] * b.cardinality[i+1]
		}
	}

	// For each new variable, precompute which axis it maps to in a and b.
	type axisMap struct {
		aAxis int // -1 if not in a
		bAxis int // -1 if not in b
	}
	mappings := make([]axisMap, len(newVars))
	for i, v := range newVars {
		aIdx, aOk := aVarIdx[v]
		bIdx, bOk := bVarIdx[v]
		m := axisMap{aAxis: -1, bAxis: -1}
		if aOk {
			m.aAxis = aIdx
		}
		if bOk {
			m.bAxis = bIdx
		}
		mappings[i] = m
	}

	// Iterate over all assignments in the product factor.
	indices := make([]int, len(newVars))
	for flat := 0; flat < newSize; flat++ {
		// Decompose flat index into indices for each variable.
		rem := flat
		for i := len(newVars) - 1; i >= 0; i-- {
			indices[i] = rem % newCard[i]
			rem /= newCard[i]
		}

		// Compute flat indices in a and b.
		aFlat := 0
		bFlat := 0
		for i := 0; i < len(newVars); i++ {
			m := mappings[i]
			if m.aAxis >= 0 {
				aFlat += indices[i] * aStrides[m.aAxis]
			}
			if m.bAxis >= 0 {
				bFlat += indices[i] * bStrides[m.bAxis]
			}
		}

		newValues[flat] = aData[aFlat] * bData[bFlat]
	}

	return NewDiscreteFactor(newVars, newCard, newValues)
}
