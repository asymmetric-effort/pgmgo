package numgo

import (
	"fmt"
	"sort"
	"strings"
)

// einsumParsed holds the parsed notation for an einsum operation.
type einsumParsed struct {
	inputSubs [][]byte // subscript labels per operand
	outputSub []byte   // subscript labels for output
	allLabels []byte   // all unique labels in sorted order
}

// parseEinsum parses a notation string like "ij,jk->ik" into its components.
func parseEinsum(notation string) (*einsumParsed, error) {
	notation = strings.ReplaceAll(notation, " ", "")
	if notation == "" {
		return nil, fmt.Errorf("numgo.Einsum: empty notation")
	}

	var inputPart, outputPart string
	implicit := false

	if idx := strings.Index(notation, "->"); idx >= 0 {
		inputPart = notation[:idx]
		outputPart = notation[idx+2:]
	} else {
		inputPart = notation
		implicit = true
	}

	inputStrs := strings.Split(inputPart, ",")
	inputSubs := make([][]byte, len(inputStrs))
	for i, s := range inputStrs {
		for _, c := range s {
			if c < 'a' || c > 'z' {
				return nil, fmt.Errorf("numgo.Einsum: invalid subscript character '%c'", c)
			}
		}
		inputSubs[i] = []byte(s)
	}

	// Collect all labels and their counts.
	labelCount := make(map[byte]int)
	for _, sub := range inputSubs {
		for _, c := range sub {
			labelCount[c]++
		}
	}

	// Build sorted list of all unique labels.
	allLabels := make([]byte, 0, len(labelCount))
	for c := range labelCount {
		allLabels = append(allLabels, c)
	}
	sort.Slice(allLabels, func(i, j int) bool { return allLabels[i] < allLabels[j] })

	var outputSub []byte
	if implicit {
		// Implicit mode: output has labels that appear exactly once, sorted.
		for _, c := range allLabels {
			if labelCount[c] == 1 {
				outputSub = append(outputSub, c)
			}
		}
	} else {
		outputSub = []byte(outputPart)
		// Validate output labels.
		for _, c := range outputSub {
			if c < 'a' || c > 'z' {
				return nil, fmt.Errorf("numgo.Einsum: invalid output subscript character '%c'", c)
			}
			if _, ok := labelCount[c]; !ok {
				return nil, fmt.Errorf("numgo.Einsum: output label '%c' not found in inputs", c)
			}
		}
	}

	return &einsumParsed{
		inputSubs: inputSubs,
		outputSub: outputSub,
		allLabels: allLabels,
	}, nil
}

// Einsum performs Einstein summation on the given operands according to the
// notation string. It supports numpy-style einsum notation such as:
//
//   - "ij,jk->ik"  (matrix multiply)
//   - "ii->"       (trace)
//   - "ij->"       (sum all elements)
//   - "ij->i"      (row sums)
//   - "ij->j"      (column sums)
//   - "i,j->ij"    (outer product)
//   - "bij,bjk->bik" (batch matrix multiply)
//
// If no "->" is given, implicit mode outputs the sorted labels that appear
// exactly once across all inputs.
func Einsum(notation string, operands ...*NDArray) (*NDArray, error) {
	parsed, err := parseEinsum(notation)
	if err != nil {
		return nil, err
	}

	if len(operands) != len(parsed.inputSubs) {
		return nil, fmt.Errorf("numgo.Einsum: notation specifies %d operands, got %d",
			len(parsed.inputSubs), len(operands))
	}

	// Build label -> size map and validate consistency.
	labelSize := make(map[byte]int)
	for opIdx, sub := range parsed.inputSubs {
		op := operands[opIdx]
		if len(sub) != op.Ndim() {
			return nil, fmt.Errorf("numgo.Einsum: operand %d has %d dimensions but subscript '%s' has %d labels",
				opIdx, op.Ndim(), string(sub), len(sub))
		}
		shape := op.Shape()
		for dimIdx, c := range sub {
			if sz, ok := labelSize[c]; ok {
				if sz != shape[dimIdx] {
					return nil, fmt.Errorf("numgo.Einsum: label '%c' has inconsistent sizes %d and %d",
						c, sz, shape[dimIdx])
				}
			} else {
				labelSize[c] = shape[dimIdx]
			}
		}
	}

	// Try optimized paths for common cases.
	if result, ok := einsumOptimized(parsed, operands, labelSize); ok {
		return result, nil
	}

	return einsumGeneric(parsed, operands, labelSize)
}

// einsumOptimized tries direct loop implementations for common 2-operand cases.
// Returns (result, true) if an optimized path was used, (nil, false) otherwise.
func einsumOptimized(parsed *einsumParsed, operands []*NDArray, labelSize map[byte]int) (*NDArray, bool) {
	if len(operands) != 2 {
		return nil, false
	}

	inA := string(parsed.inputSubs[0])
	inB := string(parsed.inputSubs[1])
	out := string(parsed.outputSub)

	// Matrix multiply: ij,jk->ik
	if inA == "ij" && inB == "jk" && out == "ik" {
		return einsumMatmul(operands[0], operands[1], labelSize), true
	}

	// Dot product: i,i->
	if inA == "i" && inB == "i" && out == "" {
		return einsumDot(operands[0], operands[1]), true
	}

	// Outer product: i,j->ij
	if inA == "i" && inB == "j" && out == "ij" {
		return einsumOuter(operands[0], operands[1]), true
	}

	return nil, false
}

// einsumMatmul performs C[i,k] = sum_j A[i,j] * B[j,k].
func einsumMatmul(a, b *NDArray, labelSize map[byte]int) *NDArray {
	m := labelSize['i']
	n := labelSize['j']
	p := labelSize['k']
	result := Zeros(m, p)
	for i := 0; i < m; i++ {
		for k := 0; k < p; k++ {
			sum := 0.0
			for j := 0; j < n; j++ {
				sum += a.Get(i, j) * b.Get(j, k)
			}
			result.Set(sum, i, k)
		}
	}
	return result
}

// einsumDot computes the dot product of two 1-D arrays.
func einsumDot(a, b *NDArray) *NDArray {
	sum := 0.0
	for i := 0; i < a.Size(); i++ {
		sum += a.data[i] * b.data[i]
	}
	return NewNDArray([]int{}, []float64{sum})
}

// einsumOuter computes the outer product of two 1-D arrays.
func einsumOuter(a, b *NDArray) *NDArray {
	m := a.Size()
	n := b.Size()
	data := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[i*n+j] = a.data[i] * b.data[j]
		}
	}
	return NewNDArray([]int{m, n}, data)
}

// einsumGeneric is the general-purpose einsum implementation that works for
// any valid notation by iterating over the cartesian product of all index values.
func einsumGeneric(parsed *einsumParsed, operands []*NDArray, labelSize map[byte]int) (*NDArray, error) {
	allLabels := parsed.allLabels

	// Build label -> position in allLabels.
	labelPos := make(map[byte]int)
	for i, c := range allLabels {
		labelPos[c] = i
	}

	// Compute sizes for cartesian product iteration.
	nLabels := len(allLabels)
	labelSizes := make([]int, nLabels)
	totalCombinations := 1
	for i, c := range allLabels {
		labelSizes[i] = labelSize[c]
		totalCombinations *= labelSize[c]
	}

	// Build output shape.
	outShape := make([]int, len(parsed.outputSub))
	for i, c := range parsed.outputSub {
		outShape[i] = labelSize[c]
	}

	// Handle scalar output.
	if len(outShape) == 0 {
		outShape = []int{}
	}

	outSize := product(outShape)
	if len(outShape) == 0 {
		outSize = 1
	}
	outData := make([]float64, outSize)

	// Precompute: for each operand, the label positions of its subscripts.
	opLabelPositions := make([][]int, len(operands))
	for opIdx, sub := range parsed.inputSubs {
		positions := make([]int, len(sub))
		for i, c := range sub {
			positions[i] = labelPos[c]
		}
		opLabelPositions[opIdx] = positions
	}

	// Precompute: for output, the label positions of its subscripts.
	outLabelPositions := make([]int, len(parsed.outputSub))
	for i, c := range parsed.outputSub {
		outLabelPositions[i] = labelPos[c]
	}

	// Precompute output strides.
	outStrides := computeStrides(outShape)

	// Iterate over all combinations of index values.
	indices := make([]int, nLabels)
	opIndices := make([]int, maxOpDims(operands))

	for combo := 0; combo < totalCombinations; combo++ {
		// Decompose combo into multi-index.
		if combo == 0 {
			for i := range indices {
				indices[i] = 0
			}
		}

		// Compute product of elements from all operands.
		prod := 1.0
		for opIdx, op := range operands {
			positions := opLabelPositions[opIdx]
			for d, pos := range positions {
				opIndices[d] = indices[pos]
			}
			prod *= op.data[op.flatIndex(opIndices[:len(positions)])]
		}

		// Compute output flat index.
		outFlat := 0
		for i, pos := range outLabelPositions {
			outFlat += indices[pos] * outStrides[i]
		}

		outData[outFlat] += prod

		// Increment indices (odometer style, last label fastest).
		for i := nLabels - 1; i >= 0; i-- {
			indices[i]++
			if indices[i] < labelSizes[i] {
				break
			}
			indices[i] = 0
		}
	}

	return NewNDArray(outShape, outData), nil
}

// maxOpDims returns the maximum number of dimensions across operands.
func maxOpDims(operands []*NDArray) int {
	m := 0
	for _, op := range operands {
		if op.Ndim() > m {
			m = op.Ndim()
		}
	}
	if m == 0 {
		m = 1
	}
	return m
}
