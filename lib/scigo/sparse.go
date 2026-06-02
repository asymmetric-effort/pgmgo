package scigo

import (
	"fmt"
	"sort"
)

// ---------------------------------------------------------------------------
// COO — Coordinate (triplet) format
// ---------------------------------------------------------------------------

// COO stores a sparse matrix in coordinate (triplet) format.
// Each non-zero entry is represented by its (row, col, value) triple.
type COO struct {
	rows   []int
	cols   []int
	values []float64
	shape  [2]int
}

// NewCOO creates a COO sparse matrix from row indices, column indices, values,
// and the matrix shape. Returns an error if the input slices have mismatched
// lengths or if any index is out of bounds.
func NewCOO(rows, cols []int, values []float64, shape [2]int) (*COO, error) {
	if len(rows) != len(cols) || len(rows) != len(values) {
		return nil, fmt.Errorf("scigo: NewCOO: rows, cols, and values must have the same length")
	}
	if shape[0] <= 0 || shape[1] <= 0 {
		return nil, fmt.Errorf("scigo: NewCOO: shape dimensions must be positive")
	}
	for i := range rows {
		if rows[i] < 0 || rows[i] >= shape[0] {
			return nil, fmt.Errorf("scigo: NewCOO: row index %d out of bounds [0, %d)", rows[i], shape[0])
		}
		if cols[i] < 0 || cols[i] >= shape[1] {
			return nil, fmt.Errorf("scigo: NewCOO: col index %d out of bounds [0, %d)", cols[i], shape[1])
		}
	}
	r := make([]int, len(rows))
	c := make([]int, len(cols))
	v := make([]float64, len(values))
	copy(r, rows)
	copy(c, cols)
	copy(v, values)
	return &COO{rows: r, cols: c, values: v, shape: shape}, nil
}

// Get returns the value at (row, col). If the position has multiple entries
// their sum is returned (standard COO semantics). Returns 0 for absent entries.
func (m *COO) Get(row, col int) float64 {
	var sum float64
	for i := range m.rows {
		if m.rows[i] == row && m.cols[i] == col {
			sum += m.values[i]
		}
	}
	return sum
}

// Set adds a new (row, col, value) entry to the COO matrix. It does not
// de-duplicate; if an entry already exists at (row, col) the value will be
// summed on Get or during conversion.
func (m *COO) Set(row, col int, value float64) {
	m.rows = append(m.rows, row)
	m.cols = append(m.cols, col)
	m.values = append(m.values, value)
}

// Shape returns the dimensions of the matrix as [rows, cols].
func (m *COO) Shape() [2]int {
	return m.shape
}

// NNZ returns the number of stored entries (including explicit zeros and
// duplicate positions).
func (m *COO) NNZ() int {
	return len(m.values)
}

// ToCSR converts the COO matrix to CSR format. Duplicate entries at the same
// position are summed.
func (m *COO) ToCSR() *CSR {
	nrows := m.shape[0]

	// Aggregate duplicates into a map keyed by (row, col).
	type key struct{ r, c int }
	agg := make(map[key]float64)
	for i := range m.values {
		agg[key{m.rows[i], m.cols[i]}] += m.values[i]
	}

	// Build sorted entries per row.
	rowEntries := make([][]struct {
		col int
		val float64
	}, nrows)
	for k, v := range agg {
		if v == 0 {
			continue
		}
		rowEntries[k.r] = append(rowEntries[k.r], struct {
			col int
			val float64
		}{k.c, v})
	}
	for i := range rowEntries {
		sort.Slice(rowEntries[i], func(a, b int) bool {
			return rowEntries[i][a].col < rowEntries[i][b].col
		})
	}

	indptr := make([]int, nrows+1)
	var indices []int
	var data []float64
	for i := 0; i < nrows; i++ {
		for _, e := range rowEntries[i] {
			indices = append(indices, e.col)
			data = append(data, e.val)
		}
		indptr[i+1] = len(data)
	}

	return &CSR{indptr: indptr, indices: indices, data: data, shape: m.shape}
}

// ToDense converts the COO matrix to a dense 2D slice. Duplicate entries are
// summed.
func (m *COO) ToDense() [][]float64 {
	dense := make([][]float64, m.shape[0])
	for i := range dense {
		dense[i] = make([]float64, m.shape[1])
	}
	for i := range m.values {
		dense[m.rows[i]][m.cols[i]] += m.values[i]
	}
	return dense
}

// ---------------------------------------------------------------------------
// CSR — Compressed Sparse Row
// ---------------------------------------------------------------------------

// CSR stores a sparse matrix in Compressed Sparse Row format.
// indptr has length nrows+1; row i has column indices in indices[indptr[i]:indptr[i+1]]
// and corresponding values in data[indptr[i]:indptr[i+1]].
type CSR struct {
	indptr  []int
	indices []int
	data    []float64
	shape   [2]int
}

// NewCSR creates a CSR sparse matrix. Returns an error if inputs are invalid.
func NewCSR(indptr, indices []int, data []float64, shape [2]int) (*CSR, error) {
	if shape[0] <= 0 || shape[1] <= 0 {
		return nil, fmt.Errorf("scigo: NewCSR: shape dimensions must be positive")
	}
	if len(indptr) != shape[0]+1 {
		return nil, fmt.Errorf("scigo: NewCSR: indptr length must be nrows+1 (%d), got %d", shape[0]+1, len(indptr))
	}
	if len(indices) != len(data) {
		return nil, fmt.Errorf("scigo: NewCSR: indices and data must have the same length")
	}
	if indptr[0] != 0 {
		return nil, fmt.Errorf("scigo: NewCSR: indptr[0] must be 0")
	}
	nnz := len(data)
	if indptr[len(indptr)-1] != nnz {
		return nil, fmt.Errorf("scigo: NewCSR: indptr[-1] must equal len(data)")
	}
	for i := 0; i < len(indptr)-1; i++ {
		if indptr[i] > indptr[i+1] {
			return nil, fmt.Errorf("scigo: NewCSR: indptr must be non-decreasing")
		}
	}
	for _, c := range indices {
		if c < 0 || c >= shape[1] {
			return nil, fmt.Errorf("scigo: NewCSR: column index %d out of bounds [0, %d)", c, shape[1])
		}
	}

	ip := make([]int, len(indptr))
	idx := make([]int, len(indices))
	d := make([]float64, len(data))
	copy(ip, indptr)
	copy(idx, indices)
	copy(d, data)
	return &CSR{indptr: ip, indices: idx, data: d, shape: shape}, nil
}

// Get returns the value at (row, col), or 0 if the entry is not stored.
func (m *CSR) Get(row, col int) float64 {
	start, end := m.indptr[row], m.indptr[row+1]
	for i := start; i < end; i++ {
		if m.indices[i] == col {
			return m.data[i]
		}
	}
	return 0
}

// Shape returns the dimensions of the matrix as [rows, cols].
func (m *CSR) Shape() [2]int {
	return m.shape
}

// NNZ returns the number of stored non-zero entries.
func (m *CSR) NNZ() int {
	return len(m.data)
}

// Row returns the column indices and corresponding values of row i.
func (m *CSR) Row(i int) ([]int, []float64) {
	start, end := m.indptr[i], m.indptr[i+1]
	idx := make([]int, end-start)
	vals := make([]float64, end-start)
	copy(idx, m.indices[start:end])
	copy(vals, m.data[start:end])
	return idx, vals
}

// ToDense converts the CSR matrix to a dense 2D slice.
func (m *CSR) ToDense() [][]float64 {
	dense := make([][]float64, m.shape[0])
	for i := range dense {
		dense[i] = make([]float64, m.shape[1])
	}
	for i := 0; i < m.shape[0]; i++ {
		start, end := m.indptr[i], m.indptr[i+1]
		for j := start; j < end; j++ {
			dense[i][m.indices[j]] = m.data[j]
		}
	}
	return dense
}

// ToCOO converts the CSR matrix to COO format.
func (m *CSR) ToCOO() *COO {
	var rows, cols []int
	var values []float64
	for i := 0; i < m.shape[0]; i++ {
		start, end := m.indptr[i], m.indptr[i+1]
		for j := start; j < end; j++ {
			rows = append(rows, i)
			cols = append(cols, m.indices[j])
			values = append(values, m.data[j])
		}
	}
	if rows == nil {
		rows = []int{}
		cols = []int{}
		values = []float64{}
	}
	return &COO{rows: rows, cols: cols, values: values, shape: m.shape}
}

// Transpose returns the transpose of the CSR matrix (also in CSR format).
func (m *CSR) Transpose() *CSR {
	tShape := [2]int{m.shape[1], m.shape[0]}
	ncols := m.shape[1]

	// Count entries per column (which becomes rows of transpose).
	counts := make([]int, ncols)
	for _, c := range m.indices {
		counts[c]++
	}

	// Build indptr for transpose.
	tIndptr := make([]int, ncols+1)
	for i := 0; i < ncols; i++ {
		tIndptr[i+1] = tIndptr[i] + counts[i]
	}

	tIndices := make([]int, len(m.data))
	tData := make([]float64, len(m.data))
	pos := make([]int, ncols)

	for i := 0; i < m.shape[0]; i++ {
		start, end := m.indptr[i], m.indptr[i+1]
		for j := start; j < end; j++ {
			c := m.indices[j]
			dest := tIndptr[c] + pos[c]
			tIndices[dest] = i
			tData[dest] = m.data[j]
			pos[c]++
		}
	}

	return &CSR{indptr: tIndptr, indices: tIndices, data: tData, shape: tShape}
}

// MulVec computes the sparse matrix-vector product y = A*x. Panics if the
// length of x does not match the number of columns.
func (m *CSR) MulVec(x []float64) []float64 {
	if len(x) != m.shape[1] {
		panic(fmt.Sprintf("scigo: CSR.MulVec: vector length %d does not match ncols %d", len(x), m.shape[1]))
	}
	y := make([]float64, m.shape[0])
	for i := 0; i < m.shape[0]; i++ {
		start, end := m.indptr[i], m.indptr[i+1]
		var sum float64
		for j := start; j < end; j++ {
			sum += m.data[j] * x[m.indices[j]]
		}
		y[i] = sum
	}
	return y
}

// MulDense computes the product of the sparse CSR matrix (m x k) with a dense
// matrix (k x n), returning a dense (m x n) matrix. Panics on dimension
// mismatch.
func (m *CSR) MulDense(dense [][]float64) [][]float64 {
	k := m.shape[1]
	if len(dense) != k {
		panic(fmt.Sprintf("scigo: CSR.MulDense: dense rows %d does not match ncols %d", len(dense), k))
	}
	n := 0
	if len(dense) > 0 {
		n = len(dense[0])
	}
	result := make([][]float64, m.shape[0])
	for i := range result {
		result[i] = make([]float64, n)
	}
	for i := 0; i < m.shape[0]; i++ {
		start, end := m.indptr[i], m.indptr[i+1]
		for j := start; j < end; j++ {
			col := m.indices[j]
			val := m.data[j]
			for c := 0; c < n; c++ {
				result[i][c] += val * dense[col][c]
			}
		}
	}
	return result
}

// ---------------------------------------------------------------------------
// CSC — Compressed Sparse Column
// ---------------------------------------------------------------------------

// CSC stores a sparse matrix in Compressed Sparse Column format.
// indptr has length ncols+1; column j has row indices in indices[indptr[j]:indptr[j+1]]
// and corresponding values in data[indptr[j]:indptr[j+1]].
type CSC struct {
	indptr  []int
	indices []int
	data    []float64
	shape   [2]int
}

// NewCSC creates a CSC sparse matrix. Returns an error if inputs are invalid.
func NewCSC(indptr, indices []int, data []float64, shape [2]int) (*CSC, error) {
	if shape[0] <= 0 || shape[1] <= 0 {
		return nil, fmt.Errorf("scigo: NewCSC: shape dimensions must be positive")
	}
	if len(indptr) != shape[1]+1 {
		return nil, fmt.Errorf("scigo: NewCSC: indptr length must be ncols+1 (%d), got %d", shape[1]+1, len(indptr))
	}
	if len(indices) != len(data) {
		return nil, fmt.Errorf("scigo: NewCSC: indices and data must have the same length")
	}
	if indptr[0] != 0 {
		return nil, fmt.Errorf("scigo: NewCSC: indptr[0] must be 0")
	}
	nnz := len(data)
	if indptr[len(indptr)-1] != nnz {
		return nil, fmt.Errorf("scigo: NewCSC: indptr[-1] must equal len(data)")
	}
	for i := 0; i < len(indptr)-1; i++ {
		if indptr[i] > indptr[i+1] {
			return nil, fmt.Errorf("scigo: NewCSC: indptr must be non-decreasing")
		}
	}
	for _, r := range indices {
		if r < 0 || r >= shape[0] {
			return nil, fmt.Errorf("scigo: NewCSC: row index %d out of bounds [0, %d)", r, shape[0])
		}
	}

	ip := make([]int, len(indptr))
	idx := make([]int, len(indices))
	d := make([]float64, len(data))
	copy(ip, indptr)
	copy(idx, indices)
	copy(d, data)
	return &CSC{indptr: ip, indices: idx, data: d, shape: shape}, nil
}

// Get returns the value at (row, col), or 0 if the entry is not stored.
func (m *CSC) Get(row, col int) float64 {
	start, end := m.indptr[col], m.indptr[col+1]
	for i := start; i < end; i++ {
		if m.indices[i] == row {
			return m.data[i]
		}
	}
	return 0
}

// Shape returns the dimensions of the matrix as [rows, cols].
func (m *CSC) Shape() [2]int {
	return m.shape
}

// NNZ returns the number of stored non-zero entries.
func (m *CSC) NNZ() int {
	return len(m.data)
}

// Col returns the row indices and corresponding values of column j.
func (m *CSC) Col(j int) ([]int, []float64) {
	start, end := m.indptr[j], m.indptr[j+1]
	idx := make([]int, end-start)
	vals := make([]float64, end-start)
	copy(idx, m.indices[start:end])
	copy(vals, m.data[start:end])
	return idx, vals
}

// ToDense converts the CSC matrix to a dense 2D slice.
func (m *CSC) ToDense() [][]float64 {
	dense := make([][]float64, m.shape[0])
	for i := range dense {
		dense[i] = make([]float64, m.shape[1])
	}
	for j := 0; j < m.shape[1]; j++ {
		start, end := m.indptr[j], m.indptr[j+1]
		for k := start; k < end; k++ {
			dense[m.indices[k]][j] = m.data[k]
		}
	}
	return dense
}

// ToCSR converts the CSC matrix to CSR format.
func (m *CSC) ToCSR() *CSR {
	nrows := m.shape[0]

	// Count entries per row.
	counts := make([]int, nrows)
	for _, r := range m.indices {
		counts[r]++
	}

	indptr := make([]int, nrows+1)
	for i := 0; i < nrows; i++ {
		indptr[i+1] = indptr[i] + counts[i]
	}

	indices := make([]int, len(m.data))
	data := make([]float64, len(m.data))
	pos := make([]int, nrows)

	for j := 0; j < m.shape[1]; j++ {
		start, end := m.indptr[j], m.indptr[j+1]
		for k := start; k < end; k++ {
			r := m.indices[k]
			dest := indptr[r] + pos[r]
			indices[dest] = j
			data[dest] = m.data[k]
			pos[r]++
		}
	}

	// Sort column indices within each row.
	for i := 0; i < nrows; i++ {
		s, e := indptr[i], indptr[i+1]
		if e-s <= 1 {
			continue
		}
		sortCSRRow(indices[s:e], data[s:e])
	}

	return &CSR{indptr: indptr, indices: indices, data: data, shape: m.shape}
}

// sortCSRRow sorts parallel indices/data slices by index.
func sortCSRRow(indices []int, data []float64) {
	n := len(indices)
	type pair struct {
		idx int
		val float64
	}
	pairs := make([]pair, n)
	for i := 0; i < n; i++ {
		pairs[i] = pair{indices[i], data[i]}
	}
	sort.Slice(pairs, func(a, b int) bool {
		return pairs[a].idx < pairs[b].idx
	})
	for i := 0; i < n; i++ {
		indices[i] = pairs[i].idx
		data[i] = pairs[i].val
	}
}

// ---------------------------------------------------------------------------
// Dense-to-sparse conversion helpers
// ---------------------------------------------------------------------------

// DenseToCSR converts a dense 2D slice to CSR format. Zero entries are omitted.
func DenseToCSR(dense [][]float64) *CSR {
	nrows := len(dense)
	if nrows == 0 {
		return &CSR{indptr: []int{0}, indices: nil, data: nil, shape: [2]int{0, 0}}
	}
	ncols := len(dense[0])

	indptr := make([]int, nrows+1)
	var indices []int
	var data []float64

	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			if dense[i][j] != 0 {
				indices = append(indices, j)
				data = append(data, dense[i][j])
			}
		}
		indptr[i+1] = len(data)
	}

	return &CSR{indptr: indptr, indices: indices, data: data, shape: [2]int{nrows, ncols}}
}

// DenseToCOO converts a dense 2D slice to COO format. Zero entries are omitted.
func DenseToCOO(dense [][]float64) *COO {
	nrows := len(dense)
	if nrows == 0 {
		return &COO{rows: []int{}, cols: []int{}, values: []float64{}, shape: [2]int{0, 0}}
	}
	ncols := len(dense[0])

	var rows, cols []int
	var values []float64

	for i := 0; i < nrows; i++ {
		for j := 0; j < ncols; j++ {
			if dense[i][j] != 0 {
				rows = append(rows, i)
				cols = append(cols, j)
				values = append(values, dense[i][j])
			}
		}
	}
	if rows == nil {
		rows = []int{}
		cols = []int{}
		values = []float64{}
	}

	return &COO{rows: rows, cols: cols, values: values, shape: [2]int{nrows, ncols}}
}
