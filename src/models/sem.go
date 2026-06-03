package models

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
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

// Fit estimates SEM parameters from data using OLS (ordinary least squares)
// per equation. The DataFrame must contain columns for all variables.
// The DAG structure must be set up (via AddEquation or by constructing
// equations with zero coefficients) before calling Fit.
func (s *SEM) Fit(data *tabgo.DataFrame) error {
	if data == nil {
		return fmt.Errorf("models: data must not be nil")
	}
	if data.Len() == 0 {
		return fmt.Errorf("models: data must not be empty")
	}

	nodes := s.dag.Nodes()
	if len(nodes) == 0 {
		return fmt.Errorf("models: SEM has no variables")
	}

	nRows := data.Len()

	// Fetch all columns.
	colData := make(map[string][]float64, len(nodes))
	for _, node := range nodes {
		colData[node] = data.Column(node).Float64()
	}

	for _, node := range nodes {
		parents := s.dag.Parents(node)
		y := colData[node]

		if len(parents) == 0 {
			// No parents: estimate mean and variance.
			mean := 0.0
			for _, v := range y {
				mean += v
			}
			mean /= float64(nRows)

			variance := 0.0
			for _, v := range y {
				d := v - mean
				variance += d * d
			}
			if nRows > 1 {
				variance /= float64(nRows)
			}
			if variance <= 0 {
				variance = 1e-10
			}

			s.equations[node] = &SEMEquation{
				Variable:     node,
				Parents:      nil,
				Coefficients: nil,
				Intercept:    mean,
				Variance:     variance,
			}
			continue
		}

		// OLS: y = intercept + sum(beta_i * x_i) + error
		// Using normal equations: [X^T X] beta = X^T y
		// where X has a column of 1s for the intercept.
		nP := len(parents)
		k := nP + 1 // number of parameters (intercept + coefficients)

		// Build X^T X (k x k) and X^T y (k x 1).
		xtx := make([][]float64, k)
		for i := range xtx {
			xtx[i] = make([]float64, k)
		}
		xty := make([]float64, k)

		for row := 0; row < nRows; row++ {
			// x[0] = 1 (intercept), x[1..] = parent values
			x := make([]float64, k)
			x[0] = 1.0
			for j, p := range parents {
				x[j+1] = colData[p][row]
			}
			for i := 0; i < k; i++ {
				for j := 0; j < k; j++ {
					xtx[i][j] += x[i] * x[j]
				}
				xty[i] += x[i] * y[row]
			}
		}

		// Solve via matrix inversion.
		inv, err := invertMatrix(xtx)
		if err != nil {
			return fmt.Errorf("models: OLS for %q: %w", node, err)
		}

		// beta = inv(X^T X) * X^T y
		beta := make([]float64, k)
		for i := 0; i < k; i++ {
			for j := 0; j < k; j++ {
				beta[i] += inv[i][j] * xty[j]
			}
		}

		intercept := beta[0]
		coefficients := beta[1:]

		// Compute residual variance.
		variance := 0.0
		for row := 0; row < nRows; row++ {
			predicted := intercept
			for j, p := range parents {
				predicted += coefficients[j] * colData[p][row]
			}
			d := y[row] - predicted
			variance += d * d
		}
		if nRows > 1 {
			variance /= float64(nRows)
		}
		if variance <= 0 {
			variance = 1e-10
		}

		pCopy := make([]string, len(parents))
		copy(pCopy, parents)
		cCopy := make([]float64, len(coefficients))
		copy(cCopy, coefficients)

		s.equations[node] = &SEMEquation{
			Variable:     node,
			Parents:      pCopy,
			Coefficients: cCopy,
			Intercept:    intercept,
			Variance:     variance,
		}
	}

	return nil
}

// GenerateSamples simulates data from the SEM by sampling in topological order.
// Each variable is computed as: value = intercept + sum(coeff * parentValue) + N(0, variance).
func (s *SEM) GenerateSamples(nSamples int) (*tabgo.DataFrame, error) {
	if err := s.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: cannot generate samples: %w", err)
	}
	if nSamples <= 0 {
		return nil, fmt.Errorf("models: nSamples must be positive, got %d", nSamples)
	}

	order, err := s.dag.TopologicalOrder()
	if err != nil {
		return nil, fmt.Errorf("models: cannot get topological order: %w", err)
	}

	data := make(map[string][]float64, len(order))
	for _, v := range order {
		data[v] = make([]float64, nSamples)
	}

	for i := 0; i < nSamples; i++ {
		for _, v := range order {
			eq := s.equations[v]
			val := eq.Intercept
			for j, p := range eq.Parents {
				val += eq.Coefficients[j] * data[p][i]
			}
			// Add Gaussian noise using Box-Muller.
			val += math.Sqrt(eq.Variance) * randStdNormal()
			data[v][i] = val
		}
	}

	// Build DataFrame.
	vars := s.dag.Nodes() // sorted
	rows := make([][]any, nSamples)
	for i := 0; i < nSamples; i++ {
		row := make([]any, len(vars))
		for j, v := range vars {
			row[j] = data[v][i]
		}
		rows[i] = row
	}

	return tabgo.NewDataFrameFromRows(vars, rows), nil
}

// randStdNormal returns a sample from N(0,1) using Box-Muller.
func randStdNormal() float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	for u1 == 0 {
		u1 = rand.Float64()
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// SetParams sets the parameters for an existing equation.
func (s *SEM) SetParams(variable string, coefficients []float64, intercept, variance float64) error {
	eq, ok := s.equations[variable]
	if !ok {
		return fmt.Errorf("models: no equation for variable %q", variable)
	}
	if len(coefficients) != len(eq.Parents) {
		return fmt.Errorf("models: coefficients length %d != parents length %d", len(coefficients), len(eq.Parents))
	}
	cCopy := make([]float64, len(coefficients))
	copy(cCopy, coefficients)
	eq.Coefficients = cCopy
	eq.Intercept = intercept
	eq.Variance = variance
	return nil
}

// GetScalingIndicators returns one indicator variable per latent variable.
// In a basic SEM, this returns all root nodes (nodes with no parents) as
// they serve as natural scaling indicators for identification.
func (s *SEM) GetScalingIndicators() []string {
	return s.dag.GetRoots()
}

// ActiveTrailNodes returns the set of nodes reachable from variable via
// active trails given the observed variables, using Bayes-ball on the
// underlying DAG.
func (s *SEM) ActiveTrailNodes(variable string, observed map[string]bool) ([]string, error) {
	if !s.dag.HasNode(variable) {
		return nil, fmt.Errorf("models: variable %q not in the SEM", variable)
	}
	if observed == nil {
		observed = make(map[string]bool)
	}

	type visit struct {
		node string
		up   bool
	}

	reachable := make(map[string]bool)
	visited := make(map[visit]bool)
	queue := []visit{{variable, true}, {variable, false}}

	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		if visited[cur] {
			continue
		}
		visited[cur] = true

		node := cur.node
		isObserved := observed[node]

		if cur.up {
			if !isObserved {
				reachable[node] = true
				for _, p := range s.dag.Parents(node) {
					queue = append(queue, visit{p, true})
				}
				for _, c := range s.dag.Children(node) {
					queue = append(queue, visit{c, false})
				}
			}
		} else {
			if !isObserved {
				reachable[node] = true
				for _, c := range s.dag.Children(node) {
					queue = append(queue, visit{c, false})
				}
			} else {
				for _, p := range s.dag.Parents(node) {
					queue = append(queue, visit{p, true})
				}
			}
		}
	}

	delete(reachable, variable)
	result := make([]string, 0, len(reachable))
	for n := range reachable {
		result = append(result, n)
	}
	sort.Strings(result)
	return result, nil
}

// Moralize returns the moral graph of the SEM's DAG as a graphgo.Graph.
// The moral graph connects all co-parents and drops edge directions.
func (s *SEM) Moralize() *graphgo.Graph {
	dg := graphgo.NewDiGraph()
	for _, n := range s.dag.Nodes() {
		dg.AddNode(n)
	}
	for _, e := range s.dag.Edges() {
		dg.AddEdge(e.Src, e.Dst)
	}
	return graphgo.Moralize(dg)
}

// FromLavaan parses lavaan model syntax into an SEM. Each line of the form
// "Y ~ X1 + X2" creates an equation for Y with parents X1, X2, zero
// coefficients, zero intercept, and unit variance. Blank lines and lines
// without "~" are ignored.
func FromLavaan(syntax string) (*SEM, error) {
	if strings.TrimSpace(syntax) == "" {
		return nil, fmt.Errorf("models: empty lavaan syntax")
	}

	s := NewSEM()
	lines := strings.Split(syntax, "\n")
	parsed := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || !strings.Contains(line, "~") {
			continue
		}
		parts := strings.SplitN(line, "~", 2)
		if len(parts) != 2 {
			continue
		}
		child := strings.TrimSpace(parts[0])
		if child == "" {
			return nil, fmt.Errorf("models: empty variable in lavaan line %q", line)
		}
		parentStr := strings.TrimSpace(parts[1])
		if parentStr == "" {
			// No parents: just an intercept equation.
			if err := s.AddEquation(child, nil, nil, 0.0, 1.0); err != nil {
				return nil, fmt.Errorf("models: %w", err)
			}
			parsed = true
			continue
		}

		parentTokens := strings.Split(parentStr, "+")
		var parents []string
		for _, tok := range parentTokens {
			p := strings.TrimSpace(tok)
			if p != "" {
				parents = append(parents, p)
			}
		}

		coeffs := make([]float64, len(parents))
		if err := s.AddEquation(child, parents, coeffs, 0.0, 1.0); err != nil {
			return nil, fmt.Errorf("models: %w", err)
		}
		parsed = true
	}

	if !parsed {
		return nil, fmt.Errorf("models: no valid lavaan lines found")
	}
	return s, nil
}

// FromGraph creates an SEM from a DAG. Each variable gets an equation with
// zero coefficients and unit variance.
func FromGraph(dag *base.DAG) (*SEM, error) {
	if dag == nil {
		return nil, fmt.Errorf("models: dag must not be nil")
	}
	s := NewSEM()
	nodes := dag.Nodes()
	for _, node := range nodes {
		parents := dag.Parents(node)
		coeffs := make([]float64, len(parents))
		if err := s.AddEquation(node, parents, coeffs, 0.0, 1.0); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// FromLisrel creates an SEM from a simplified LISREL matrix specification.
// The syntax is a set of lines, each of the form:
//
//	variable: parent1=coeff1 parent2=coeff2 variance=v intercept=i
//
// If no parents are specified, the variable is exogenous. Variance defaults
// to 1.0 and intercept defaults to 0.0 if not specified.
func FromLisrel(spec string) (*SEM, error) {
	if strings.TrimSpace(spec) == "" {
		return nil, fmt.Errorf("models: empty LISREL specification")
	}

	s := NewSEM()
	lines := strings.Split(spec, "\n")
	parsed := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		colonIdx := strings.Index(line, ":")
		if colonIdx < 0 {
			continue
		}
		variable := strings.TrimSpace(line[:colonIdx])
		if variable == "" {
			continue
		}
		rest := strings.TrimSpace(line[colonIdx+1:])

		var parents []string
		var coeffs []float64
		variance := 1.0
		intercept := 0.0

		if rest != "" {
			tokens := strings.Fields(rest)
			for _, tok := range tokens {
				eqIdx := strings.Index(tok, "=")
				if eqIdx < 0 {
					continue
				}
				key := tok[:eqIdx]
				valStr := tok[eqIdx+1:]
				val := 0.0
				_, err := fmt.Sscanf(valStr, "%f", &val)
				if err != nil {
					return nil, fmt.Errorf("models: invalid value %q in LISREL spec", valStr)
				}
				switch key {
				case "variance":
					variance = val
				case "intercept":
					intercept = val
				default:
					parents = append(parents, key)
					coeffs = append(coeffs, val)
				}
			}
		}

		if err := s.AddEquation(variable, parents, coeffs, intercept, variance); err != nil {
			return nil, fmt.Errorf("models: %w", err)
		}
		parsed = true
	}
	if !parsed {
		return nil, fmt.Errorf("models: no valid LISREL lines found")
	}
	return s, nil
}

// FromRAM creates an SEM from a RAM (Reticular Action Model) specification.
// It accepts the same syntax as FromLisrel.
func FromRAM(spec string) (*SEM, error) {
	return FromLisrel(spec)
}

// ToLisrel converts the SEM to LISREL matrix representation.
// Returns a map with keys "B" (Beta, structural coefficients among endogenous),
// "Gamma" (effects of exogenous on endogenous), "Psi" (error covariances),
// and "Phi" (exogenous covariances).
// Variables are ordered alphabetically within their category.
// Exogenous variables are those with no parents in the DAG (roots).
func (s *SEM) ToLisrel() (map[string][][]float64, error) {
	if err := s.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: ToLisrel: %w", err)
	}

	allVars := s.dag.Nodes() // sorted
	if len(allVars) == 0 {
		return map[string][][]float64{
			"B":     nil,
			"Gamma": nil,
			"Psi":   nil,
			"Phi":   nil,
		}, nil
	}

	// Classify variables: exogenous (no parents) vs endogenous (has parents).
	var exoVars, endoVars []string
	for _, v := range allVars {
		if len(s.dag.Parents(v)) == 0 {
			exoVars = append(exoVars, v)
		} else {
			endoVars = append(endoVars, v)
		}
	}

	nExo := len(exoVars)
	nEndo := len(endoVars)

	exoIdx := make(map[string]int, nExo)
	for i, v := range exoVars {
		exoIdx[v] = i
	}
	endoIdx := make(map[string]int, nEndo)
	for i, v := range endoVars {
		endoIdx[v] = i
	}

	// B: nEndo x nEndo — structural coefficients among endogenous variables.
	// B[i][j] = coefficient of endogenous variable j in equation for endogenous variable i.
	B := make([][]float64, nEndo)
	for i := 0; i < nEndo; i++ {
		B[i] = make([]float64, nEndo)
	}

	// Gamma: nEndo x nExo — effects of exogenous variables on endogenous variables.
	gamma := make([][]float64, nEndo)
	for i := 0; i < nEndo; i++ {
		gamma[i] = make([]float64, nExo)
	}

	// Fill B and Gamma from equations.
	for _, v := range endoVars {
		eq := s.equations[v]
		vi := endoIdx[v]
		for j, p := range eq.Parents {
			if pi, ok := endoIdx[p]; ok {
				B[vi][pi] = eq.Coefficients[j]
			} else if pi, ok := exoIdx[p]; ok {
				gamma[vi][pi] = eq.Coefficients[j]
			}
		}
	}

	// Psi: nEndo x nEndo — diagonal matrix of error variances for endogenous variables.
	psi := make([][]float64, nEndo)
	for i := 0; i < nEndo; i++ {
		psi[i] = make([]float64, nEndo)
		eq := s.equations[endoVars[i]]
		psi[i][i] = eq.Variance
	}

	// Phi: nExo x nExo — covariance matrix of exogenous variables.
	// For a standard SEM with independent exogenous errors, Phi is diagonal
	// with the error variances.
	phi := make([][]float64, nExo)
	for i := 0; i < nExo; i++ {
		phi[i] = make([]float64, nExo)
		eq := s.equations[exoVars[i]]
		phi[i][i] = eq.Variance
	}

	return map[string][][]float64{
		"B":     B,
		"Gamma": gamma,
		"Psi":   psi,
		"Phi":   phi,
	}, nil
}

// ToStandardLisrel converts the SEM to standardized LISREL matrix representation.
// The B matrix is scaled so that implied variances are 1 (standardized coefficients).
// Returns the same keys as ToLisrel: "B", "Gamma", "Psi", "Phi".
func (s *SEM) ToStandardLisrel() (map[string][][]float64, error) {
	if err := s.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: ToStandardLisrel: %w", err)
	}

	allVars := s.dag.Nodes() // sorted
	if len(allVars) == 0 {
		return map[string][][]float64{
			"B":     nil,
			"Gamma": nil,
			"Psi":   nil,
			"Phi":   nil,
		}, nil
	}

	// Compute implied covariance matrix.
	sigma, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		return nil, fmt.Errorf("models: ToStandardLisrel: %w", err)
	}

	n := len(allVars)
	varIdx := make(map[string]int, n)
	for i, v := range allVars {
		varIdx[v] = i
	}

	// Compute standard deviations from implied variances.
	sd := make(map[string]float64, n)
	for _, v := range allVars {
		vi := varIdx[v]
		impliedVar := sigma[vi][vi]
		if impliedVar <= 0 {
			impliedVar = 1e-10
		}
		sd[v] = math.Sqrt(impliedVar)
	}

	// Classify variables.
	var exoVars, endoVars []string
	for _, v := range allVars {
		if len(s.dag.Parents(v)) == 0 {
			exoVars = append(exoVars, v)
		} else {
			endoVars = append(endoVars, v)
		}
	}

	nExo := len(exoVars)
	nEndo := len(endoVars)

	exoIdx := make(map[string]int, nExo)
	for i, v := range exoVars {
		exoIdx[v] = i
	}
	endoIdx := make(map[string]int, nEndo)
	for i, v := range endoVars {
		endoIdx[v] = i
	}

	// Standardized B: B_std[i][j] = B[i][j] * sd(endoVars[j]) / sd(endoVars[i])
	B := make([][]float64, nEndo)
	for i := 0; i < nEndo; i++ {
		B[i] = make([]float64, nEndo)
	}

	// Standardized Gamma: Gamma_std[i][j] = Gamma[i][j] * sd(exoVars[j]) / sd(endoVars[i])
	gamma := make([][]float64, nEndo)
	for i := 0; i < nEndo; i++ {
		gamma[i] = make([]float64, nExo)
	}

	for _, v := range endoVars {
		eq := s.equations[v]
		vi := endoIdx[v]
		sdV := sd[v]
		for j, p := range eq.Parents {
			sdP := sd[p]
			stdCoeff := eq.Coefficients[j] * sdP / sdV
			if pi, ok := endoIdx[p]; ok {
				B[vi][pi] = stdCoeff
			} else if pi, ok := exoIdx[p]; ok {
				gamma[vi][pi] = stdCoeff
			}
		}
	}

	// Standardized Psi: Psi_std[i][i] = Psi[i][i] / impliedVar(endoVars[i])
	psi := make([][]float64, nEndo)
	for i := 0; i < nEndo; i++ {
		psi[i] = make([]float64, nEndo)
		eq := s.equations[endoVars[i]]
		impliedVar := sd[endoVars[i]] * sd[endoVars[i]]
		if impliedVar > 0 {
			psi[i][i] = eq.Variance / impliedVar
		}
	}

	// Standardized Phi: Phi_std[i][j] = Phi[i][j] / (sd(exoVars[i]) * sd(exoVars[j]))
	// For independent exogenous variables, Phi is diagonal, so standardized Phi = I.
	phi := make([][]float64, nExo)
	for i := 0; i < nExo; i++ {
		phi[i] = make([]float64, nExo)
		sdI := sd[exoVars[i]]
		if sdI > 0 {
			phi[i][i] = s.equations[exoVars[i]].Variance / (sdI * sdI)
		}
	}

	return map[string][][]float64{
		"B":     B,
		"Gamma": gamma,
		"Psi":   psi,
		"Phi":   phi,
	}, nil
}

// ToSEMGraph returns the SEM itself, as it already is a SEM graph representation.
func (s *SEM) ToSEMGraph() *SEM {
	return s
}
