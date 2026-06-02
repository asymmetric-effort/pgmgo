package numgo

import (
	"fmt"
	"sort"
)

// Sort returns a new NDArray with elements sorted along the given axis.
// For a 1D array, axis must be 0. For an ND array, each 1D slice along
// the specified axis is sorted independently.
func Sort(a *NDArray, axis int) *NDArray {
	ndim := a.Ndim()
	if axis < 0 || axis >= ndim {
		panic(fmt.Sprintf("numgo.Sort: axis %d out of range for %d dimensions", axis, ndim))
	}

	result := a.Copy()
	axisLen := result.shape[axis]

	// Iterate over every 1D slice along the given axis.
	iterateAlongAxis(result, axis, func(indices []int) {
		slice := make([]float64, axisLen)
		for i := 0; i < axisLen; i++ {
			indices[axis] = i
			slice[i] = result.data[result.flatIndex(indices)]
		}
		sort.Float64s(slice)
		for i := 0; i < axisLen; i++ {
			indices[axis] = i
			result.data[result.flatIndex(indices)] = slice[i]
		}
	})

	return result
}

// ArgSort returns a new NDArray containing the indices that would sort the
// input array along the given axis. The result has the same shape as the input,
// with float64 index values.
func ArgSort(a *NDArray, axis int) *NDArray {
	ndim := a.Ndim()
	if axis < 0 || axis >= ndim {
		panic(fmt.Sprintf("numgo.ArgSort: axis %d out of range for %d dimensions", axis, ndim))
	}

	result := NewNDArray(a.shape, nil)
	axisLen := a.shape[axis]

	iterateAlongAxis(a, axis, func(indices []int) {
		// Build index-value pairs for this slice.
		type iv struct {
			idx int
			val float64
		}
		pairs := make([]iv, axisLen)
		for i := 0; i < axisLen; i++ {
			indices[axis] = i
			pairs[i] = iv{idx: i, val: a.data[a.flatIndex(indices)]}
		}
		sort.SliceStable(pairs, func(i, j int) bool {
			return pairs[i].val < pairs[j].val
		})
		for i := 0; i < axisLen; i++ {
			indices[axis] = i
			result.data[result.flatIndex(indices)] = float64(pairs[i].idx)
		}
	})

	return result
}

// Unique returns a 1D NDArray containing the sorted unique values from a.
func Unique(a *NDArray) *NDArray {
	if a.Size() == 0 {
		return NewNDArray([]int{0}, []float64{})
	}

	sorted := make([]float64, a.Size())
	copy(sorted, a.data)
	sort.Float64s(sorted)

	unique := []float64{sorted[0]}
	for i := 1; i < len(sorted); i++ {
		if sorted[i] != sorted[i-1] {
			unique = append(unique, sorted[i])
		}
	}
	return NewNDArray([]int{len(unique)}, unique)
}

// Where performs element-wise selection: for each element, if condition is
// nonzero (true), pick from x; otherwise pick from y.
// condition, x, and y must be broadcast-compatible.
func Where(condition, x, y *NDArray) *NDArray {
	// Broadcast all three to a common shape.
	shapeAB, err := BroadcastShapes(condition.shape, x.shape)
	if err != nil {
		panic(fmt.Sprintf("numgo.Where: %v", err))
	}
	resultShape, err := BroadcastShapes(shapeAB, y.shape)
	if err != nil {
		panic(fmt.Sprintf("numgo.Where: %v", err))
	}

	cb, err := BroadcastTo(condition, resultShape)
	if err != nil {
		panic(fmt.Sprintf("numgo.Where: %v", err))
	}
	xb, err := BroadcastTo(x, resultShape)
	if err != nil {
		panic(fmt.Sprintf("numgo.Where: %v", err))
	}
	yb, err := BroadcastTo(y, resultShape)
	if err != nil {
		panic(fmt.Sprintf("numgo.Where: %v", err))
	}

	data := make([]float64, cb.Size())
	for i := range data {
		if cb.data[i] != 0 {
			data[i] = xb.data[i]
		} else {
			data[i] = yb.data[i]
		}
	}
	return NewNDArray(resultShape, data)
}

// Nonzero returns the indices of all nonzero elements in a.
// The result is a slice of coordinate tuples, where each tuple has length
// equal to a.Ndim().
func Nonzero(a *NDArray) [][]int {
	ndim := a.Ndim()
	var result [][]int

	for flat := 0; flat < a.Size(); flat++ {
		if a.data[flat] == 0 {
			continue
		}
		coords := make([]int, ndim)
		rem := flat
		for d := 0; d < ndim; d++ {
			coords[d] = rem / a.strides[d]
			rem %= a.strides[d]
		}
		result = append(result, coords)
	}
	return result
}

// SearchSorted performs a binary search on a sorted 1D array, returning the
// indices at which the given values should be inserted to maintain sort order.
// The sorted array must be 1D. The result has the same shape as values.
func SearchSorted(sorted, values *NDArray) *NDArray {
	if sorted.Ndim() != 1 {
		panic("numgo.SearchSorted: sorted array must be 1D")
	}

	data := make([]float64, values.Size())
	for i := 0; i < values.Size(); i++ {
		v := values.data[i]
		// Binary search: find leftmost insertion point.
		idx := sort.Search(sorted.Size(), func(j int) bool {
			return sorted.data[j] >= v
		})
		data[i] = float64(idx)
	}
	return NewNDArray(values.Shape(), data)
}

// iterateAlongAxis calls fn once for each 1D slice along the given axis.
// fn receives a mutable indices slice; fn is responsible for setting indices[axis]
// to iterate within the slice. On entry, indices[axis] is 0.
func iterateAlongAxis(a *NDArray, axis int, fn func(indices []int)) {
	ndim := a.Ndim()
	if ndim == 0 {
		return
	}

	// Total number of slices = product of all dims except axis.
	totalSlices := 1
	for d := 0; d < ndim; d++ {
		if d != axis {
			totalSlices *= a.shape[d]
		}
	}

	// Build a shape for the "other" dimensions to iterate over.
	otherDims := make([]int, 0, ndim-1)
	for d := 0; d < ndim; d++ {
		if d != axis {
			otherDims = append(otherDims, d)
		}
	}

	indices := make([]int, ndim)
	for s := 0; s < totalSlices; s++ {
		// Decompose s into coordinates for otherDims.
		rem := s
		for i := len(otherDims) - 1; i >= 0; i-- {
			d := otherDims[i]
			indices[d] = rem % a.shape[d]
			rem /= a.shape[d]
		}
		indices[axis] = 0
		fn(indices)
	}
}
