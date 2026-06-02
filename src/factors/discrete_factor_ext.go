package factors

import (
	"fmt"
	"math"
	"math/rand"
)

// Maximize is like Marginalize but takes the max instead of the sum over
// the eliminated variables. Returns a new factor over the remaining variables.
func (f *DiscreteFactor) Maximize(variables []string) (*DiscreteFactor, error) {
	if len(variables) == 0 {
		return f.Copy(), nil
	}

	maxSet := make(map[string]bool, len(variables))
	for _, v := range variables {
		if f.varIndex(v) == -1 {
			return nil, fmt.Errorf("factors: variable %q not in factor", v)
		}
		maxSet[v] = true
	}
	if len(maxSet) == len(f.variables) {
		return nil, fmt.Errorf("factors: cannot maximize out all variables")
	}

	var newVars []string
	var newCard []int
	for i, v := range f.variables {
		if !maxSet[v] {
			newVars = append(newVars, v)
			newCard = append(newCard, f.cardinality[i])
		}
	}

	newSize := 1
	for _, c := range newCard {
		newSize *= c
	}
	newValues := make([]float64, newSize)
	for i := range newValues {
		newValues[i] = math.Inf(-1)
	}

	data := f.values.Data()
	totalSize := f.totalSize()

	for flat := 0; flat < totalSize; flat++ {
		assignment := f.flatToAssignment(flat)
		newFlat := 0
		stride := 1
		for i := len(newVars) - 1; i >= 0; i-- {
			newFlat += assignment[newVars[i]] * stride
			stride *= newCard[i]
		}
		if data[flat] > newValues[newFlat] {
			newValues[newFlat] = data[flat]
		}
	}

	return NewDiscreteFactor(newVars, newCard, newValues)
}

// Sample draws n random assignments from the factor, treating its values as
// an unnormalized distribution. Uses the provided seed for reproducibility.
func (f *DiscreteFactor) Sample(n int, seed int64) ([]map[string]int, error) {
	if n <= 0 {
		return nil, fmt.Errorf("factors: n must be positive, got %d", n)
	}

	data := f.values.Data()
	totalSize := f.totalSize()

	// Build CDF.
	sum := 0.0
	for _, v := range data {
		if v < 0 {
			return nil, fmt.Errorf("factors: cannot sample from factor with negative values")
		}
		sum += v
	}
	if sum == 0 {
		return nil, fmt.Errorf("factors: cannot sample from factor with all-zero values")
	}

	cdf := make([]float64, totalSize)
	cumulative := 0.0
	for i, v := range data {
		cumulative += v / sum
		cdf[i] = cumulative
	}
	cdf[totalSize-1] = 1.0 // avoid floating-point edge case

	rng := rand.New(rand.NewSource(seed))
	samples := make([]map[string]int, n)

	for s := 0; s < n; s++ {
		u := rng.Float64()
		// Binary search for the index.
		lo, hi := 0, totalSize-1
		for lo < hi {
			mid := (lo + hi) / 2
			if cdf[mid] < u {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		samples[s] = f.flatToAssignment(lo)
	}

	return samples, nil
}

// Assignment converts a flat index to a variable assignment map.
func (f *DiscreteFactor) Assignment(index int) (map[string]int, error) {
	totalSize := f.totalSize()
	if index < 0 || index >= totalSize {
		return nil, fmt.Errorf("factors: index %d out of range [0, %d)", index, totalSize)
	}
	return f.flatToAssignment(index), nil
}

// IdentityFactor creates a factor with all values set to 1.
func IdentityFactor(variables []string, cardinality []int) (*DiscreteFactor, error) {
	if len(variables) != len(cardinality) {
		return nil, fmt.Errorf("factors: variables length %d != cardinality length %d",
			len(variables), len(cardinality))
	}
	size := 1
	for _, c := range cardinality {
		if c <= 0 {
			return nil, fmt.Errorf("factors: cardinality must be positive, got %d", c)
		}
		size *= c
	}
	vals := make([]float64, size)
	for i := range vals {
		vals[i] = 1.0
	}
	return NewDiscreteFactor(variables, cardinality, vals)
}

// Sum returns the element-wise addition of two factors. Both factors must have
// the same variables (in the same order) and cardinalities.
func (f *DiscreteFactor) Sum(other *DiscreteFactor) (*DiscreteFactor, error) {
	if other == nil {
		return nil, fmt.Errorf("factors: other factor must not be nil")
	}
	if len(f.variables) != len(other.variables) {
		return nil, fmt.Errorf("factors: variable count mismatch: %d vs %d",
			len(f.variables), len(other.variables))
	}
	for i := range f.variables {
		if f.variables[i] != other.variables[i] {
			return nil, fmt.Errorf("factors: variable mismatch at position %d: %q vs %q",
				i, f.variables[i], other.variables[i])
		}
		if f.cardinality[i] != other.cardinality[i] {
			return nil, fmt.Errorf("factors: cardinality mismatch for %q: %d vs %d",
				f.variables[i], f.cardinality[i], other.cardinality[i])
		}
	}

	fData := f.values.Data()
	oData := other.values.Data()
	newValues := make([]float64, len(fData))
	for i := range fData {
		newValues[i] = fData[i] + oData[i]
	}

	vars := make([]string, len(f.variables))
	copy(vars, f.variables)
	card := make([]int, len(f.cardinality))
	copy(card, f.cardinality)
	return NewDiscreteFactor(vars, card, newValues)
}

// IsValidCPD checks whether the factor represents a valid CPD. It assumes
// the first variable is the child and the remaining are parents. Each column
// (conditioned on a parent configuration) must sum to 1 (within tolerance).
func (f *DiscreteFactor) IsValidCPD() bool {
	if len(f.variables) == 0 {
		return false
	}

	const tol = 1e-6
	childCard := f.cardinality[0]
	numParentConfigs := 1
	for i := 1; i < len(f.cardinality); i++ {
		numParentConfigs *= f.cardinality[i]
	}

	data := f.values.Data()
	for j := 0; j < numParentConfigs; j++ {
		sum := 0.0
		for i := 0; i < childCard; i++ {
			sum += data[i*numParentConfigs+j]
		}
		if math.Abs(sum-1.0) > tol {
			return false
		}
	}
	return true
}
