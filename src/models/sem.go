package models

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/src/base"
)

// SEMEquation represents a single structural equation:
//
//	variable = intercept + sum(coefficients[i] * parents[i]) + error
//
// where error ~ N(0, variance).
type SEMEquation struct {
	Variable     string
	Parents      []string
	Coefficients []float64
	Intercept    float64
	Variance     float64
}

// SEM represents a linear Structural Equation Model (SEM).
// The model is defined by a DAG and a set of linear equations, one per variable.
type SEM struct {
	dag       *base.DAG
	equations map[string]*SEMEquation
}

// NewSEM creates a new empty SEM.
func NewSEM() *SEM {
	return &SEM{
		dag:       base.NewDAG(),
		equations: make(map[string]*SEMEquation),
	}
}

// AddEquation adds a structural equation for a variable. If the variable or
// its parents do not yet exist in the DAG, they are added automatically.
// Edges are added from each parent to the variable.
func (s *SEM) AddEquation(variable string, parents []string, coefficients []float64, intercept, variance float64) error {
	if len(parents) != len(coefficients) {
		return fmt.Errorf("models: parents length %d != coefficients length %d", len(parents), len(coefficients))
	}

	// Ensure variable exists.
	if !s.dag.HasNode(variable) {
		if err := s.dag.AddNode(variable); err != nil {
			return fmt.Errorf("models: failed to add variable %q: %w", variable, err)
		}
	}

	// Ensure parents exist and add edges.
	for _, p := range parents {
		if !s.dag.HasNode(p) {
			if err := s.dag.AddNode(p); err != nil {
				return fmt.Errorf("models: failed to add parent %q: %w", p, err)
			}
		}
		if !s.dag.HasEdge(p, variable) {
			if err := s.dag.AddEdge(p, variable); err != nil {
				return fmt.Errorf("models: failed to add edge %q -> %q: %w", p, variable, err)
			}
		}
	}

	pCopy := make([]string, len(parents))
	copy(pCopy, parents)
	cCopy := make([]float64, len(coefficients))
	copy(cCopy, coefficients)

	s.equations[variable] = &SEMEquation{
		Variable:     variable,
		Parents:      pCopy,
		Coefficients: cCopy,
		Intercept:    intercept,
		Variance:     variance,
	}

	return nil
}

// GetEquation returns the equation for the given variable, or nil if none is set.
func (s *SEM) GetEquation(variable string) *SEMEquation {
	return s.equations[variable]
}

// Variables returns a sorted list of all variables in the SEM.
func (s *SEM) Variables() []string {
	return s.dag.Nodes()
}

// CheckModel validates the SEM:
//  1. Every node in the DAG has an equation.
//  2. Each equation's parents match the node's parents in the DAG.
//  3. Variance must be non-negative.
func (s *SEM) CheckModel() error {
	nodes := s.dag.Nodes()
	for _, node := range nodes {
		eq, ok := s.equations[node]
		if !ok {
			return fmt.Errorf("models: node %q has no equation", node)
		}

		if eq.Variance < 0 {
			return fmt.Errorf("models: equation for %q has negative variance %f", node, eq.Variance)
		}

		// Check parents match DAG.
		dagParents := s.dag.Parents(node) // sorted
		eqParents := make([]string, len(eq.Parents))
		copy(eqParents, eq.Parents)
		sort.Strings(eqParents)

		if len(dagParents) != len(eqParents) {
			return fmt.Errorf("models: equation for %q has %d parents but DAG has %d",
				node, len(eqParents), len(dagParents))
		}
		for i := range dagParents {
			if dagParents[i] != eqParents[i] {
				return fmt.Errorf("models: equation for %q has parents %v but DAG has %v",
					node, eq.Parents, dagParents)
			}
		}
	}
	return nil
}

// ImpliedCovarianceMatrix computes the model-implied covariance matrix using
// path tracing. For a linear SEM: Sigma = (I - B)^{-1} * Psi * ((I - B)^{-1})^T
// where B is the matrix of path coefficients and Psi is the diagonal matrix
// of error variances.
//
// Variables are ordered alphabetically. Returns the covariance matrix and an
// error if the model is invalid or the matrix is not invertible.
func (s *SEM) ImpliedCovarianceMatrix() ([][]float64, error) {
	if err := s.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: cannot compute implied covariance: %w", err)
	}

	vars := s.dag.Nodes() // sorted
	n := len(vars)
	if n == 0 {
		return nil, nil
	}

	varIdx := make(map[string]int, n)
	for i, v := range vars {
		varIdx[v] = i
	}

	// Build B matrix (path coefficients) and Psi (error variances).
	B := make([][]float64, n)
	psi := make([][]float64, n)
	for i := 0; i < n; i++ {
		B[i] = make([]float64, n)
		psi[i] = make([]float64, n)
	}

	for _, v := range vars {
		eq := s.equations[v]
		vi := varIdx[v]
		psi[vi][vi] = eq.Variance
		for j, p := range eq.Parents {
			pi := varIdx[p]
			B[vi][pi] = eq.Coefficients[j]
		}
	}

	// Compute (I - B).
	imb := make([][]float64, n)
	for i := 0; i < n; i++ {
		imb[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				imb[i][j] = 1.0 - B[i][j]
			} else {
				imb[i][j] = -B[i][j]
			}
		}
	}

	// Invert (I - B) using Gauss-Jordan elimination.
	inv, err := invertMatrix(imb)
	if err != nil {
		return nil, fmt.Errorf("models: failed to invert (I-B): %w", err)
	}

	// Sigma = inv * Psi * inv^T
	// First compute inv * Psi.
	invPsi := matMul(inv, psi, n)
	// Then compute (inv * Psi) * inv^T.
	invT := transpose(inv, n)
	sigma := matMul(invPsi, invT, n)

	return sigma, nil
}

// invertMatrix inverts a square matrix using Gauss-Jordan elimination.
func invertMatrix(a [][]float64) ([][]float64, error) {
	n := len(a)
	// Augmented matrix [A | I].
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		copy(aug[i], a[i])
		aug[i][n+i] = 1.0
	}

	for col := 0; col < n; col++ {
		// Find pivot.
		pivot := -1
		maxAbs := 0.0
		for row := col; row < n; row++ {
			abs := aug[row][col]
			if abs < 0 {
				abs = -abs
			}
			if abs > maxAbs {
				maxAbs = abs
				pivot = row
			}
		}
		if maxAbs < 1e-12 {
			return nil, fmt.Errorf("matrix is singular or near-singular")
		}

		// Swap rows.
		aug[col], aug[pivot] = aug[pivot], aug[col]

		// Scale pivot row.
		scale := aug[col][col]
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= scale
		}

		// Eliminate other rows.
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := 0; j < 2*n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Extract inverse.
	inv := make([][]float64, n)
	for i := 0; i < n; i++ {
		inv[i] = make([]float64, n)
		copy(inv[i], aug[i][n:])
	}
	return inv, nil
}

// matMul multiplies two n x n matrices.
func matMul(a, b [][]float64, n int) [][]float64 {
	c := make([][]float64, n)
	for i := 0; i < n; i++ {
		c[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return c
}

// transpose returns the transpose of an n x n matrix.
func transpose(a [][]float64, n int) [][]float64 {
	t := make([][]float64, n)
	for i := 0; i < n; i++ {
		t[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			t[i][j] = a[j][i]
		}
	}
	return t
}
