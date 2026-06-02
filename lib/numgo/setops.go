package numgo

import "sort"

// Intersect1D returns a sorted 1D NDArray of values common to both a and b.
// Both inputs are treated as flattened 1D arrays.
func Intersect1D(a, b *NDArray) *NDArray {
	setB := make(map[float64]bool, b.Size())
	for _, v := range b.data {
		setB[v] = true
	}

	seen := make(map[float64]bool)
	var result []float64
	for _, v := range a.data {
		if setB[v] && !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}

	sort.Float64s(result)
	if len(result) == 0 {
		return NewNDArray([]int{0}, []float64{})
	}
	return NewNDArray([]int{len(result)}, result)
}

// Union1D returns a sorted 1D NDArray of the unique values from both a and b.
// Both inputs are treated as flattened 1D arrays.
func Union1D(a, b *NDArray) *NDArray {
	seen := make(map[float64]bool)
	var result []float64

	for _, v := range a.data {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}
	for _, v := range b.data {
		if !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}

	sort.Float64s(result)
	if len(result) == 0 {
		return NewNDArray([]int{0}, []float64{})
	}
	return NewNDArray([]int{len(result)}, result)
}

// SetDiff1D returns a sorted 1D NDArray of values in a that are not in b.
// Both inputs are treated as flattened 1D arrays.
func SetDiff1D(a, b *NDArray) *NDArray {
	setB := make(map[float64]bool, b.Size())
	for _, v := range b.data {
		setB[v] = true
	}

	seen := make(map[float64]bool)
	var result []float64
	for _, v := range a.data {
		if !setB[v] && !seen[v] {
			seen[v] = true
			result = append(result, v)
		}
	}

	sort.Float64s(result)
	if len(result) == 0 {
		return NewNDArray([]int{0}, []float64{})
	}
	return NewNDArray([]int{len(result)}, result)
}
