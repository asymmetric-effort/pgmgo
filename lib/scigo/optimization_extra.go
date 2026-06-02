package scigo

import (
	"errors"
	"math"
	"math/rand"
)

// CurveFit performs nonlinear curve fitting by minimizing the sum of squared residuals
// using the Nelder-Mead simplex method.
// f(x, params) is the model function, xdata and ydata are the observed data,
// and p0 is the initial guess for parameters.
// Returns the fitted parameters.
func CurveFit(f func(x float64, params []float64) float64, xdata, ydata []float64, p0 []float64) ([]float64, error) {
	if len(xdata) != len(ydata) {
		return nil, errors.New("scigo: CurveFit: xdata and ydata must have the same length")
	}
	if len(xdata) == 0 {
		return nil, errors.New("scigo: CurveFit: data must not be empty")
	}
	if len(p0) == 0 {
		return nil, errors.New("scigo: CurveFit: p0 must not be empty")
	}

	objective := func(params []float64) float64 {
		ssr := 0.0
		for i := range xdata {
			resid := ydata[i] - f(xdata[i], params)
			ssr += resid * resid
		}
		return ssr
	}

	res, err := nelderMead(objective, p0)
	if err != nil {
		return nil, err
	}
	return res.X, nil
}

// LeastSquares solves a nonlinear least-squares problem.
// f(params) returns the vector of residuals. The optimizer minimizes
// sum(f(params)^2) using Nelder-Mead.
// Returns an OptResult with the solution.
func LeastSquares(f func(params []float64) []float64, x0 []float64) (*OptResult, error) {
	if len(x0) == 0 {
		return nil, errors.New("scigo: LeastSquares: x0 must not be empty")
	}

	objective := func(params []float64) float64 {
		residuals := f(params)
		ssr := 0.0
		for _, r := range residuals {
			ssr += r * r
		}
		return ssr
	}

	return nelderMead(objective, x0)
}

// LinearSumAssignment solves the linear sum assignment problem (Hungarian algorithm).
// Given a cost matrix, it finds the assignment of rows to columns that minimizes
// the total cost. Returns row indices and corresponding column indices.
func LinearSumAssignment(costMatrix [][]float64) ([]int, []int, error) {
	n := len(costMatrix)
	if n == 0 {
		return nil, nil, errors.New("scigo: LinearSumAssignment: cost matrix must not be empty")
	}
	m := len(costMatrix[0])
	for _, row := range costMatrix {
		if len(row) != m {
			return nil, nil, errors.New("scigo: LinearSumAssignment: cost matrix must be rectangular")
		}
	}

	// Make square by padding with large values if needed
	sz := n
	if m > sz {
		sz = m
	}
	cost := make([][]float64, sz)
	bigVal := 0.0
	for _, row := range costMatrix {
		for _, v := range row {
			if math.Abs(v) > bigVal {
				bigVal = math.Abs(v)
			}
		}
	}
	bigVal = bigVal*float64(sz) + 1

	for i := 0; i < sz; i++ {
		cost[i] = make([]float64, sz)
		for j := 0; j < sz; j++ {
			if i < n && j < m {
				cost[i][j] = costMatrix[i][j]
			} else {
				cost[i][j] = bigVal
			}
		}
	}

	// Hungarian algorithm
	u := make([]float64, sz+1)
	v := make([]float64, sz+1)
	p := make([]int, sz+1) // p[j] = row assigned to column j
	way := make([]int, sz+1)

	for i := 1; i <= sz; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, sz+1)
		used := make([]bool, sz+1)
		for j := 1; j <= sz; j++ {
			minv[j] = math.Inf(1)
		}

		for {
			used[j0] = true
			i0 := p[j0]
			delta := math.Inf(1)
			j1 := -1
			for j := 1; j <= sz; j++ {
				if !used[j] {
					cur := cost[i0-1][j-1] - u[i0] - v[j]
					if cur < minv[j] {
						minv[j] = cur
						way[j] = j0
					}
					if minv[j] < delta {
						delta = minv[j]
						j1 = j
					}
				}
			}

			for j := 0; j <= sz; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}

			j0 = j1
			if p[j0] == 0 {
				break
			}
		}

		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	// Extract assignment
	rowInds := make([]int, 0, n)
	colInds := make([]int, 0, n)
	for j := 1; j <= sz; j++ {
		if p[j] > 0 && p[j]-1 < n && j-1 < m {
			rowInds = append(rowInds, p[j]-1)
			colInds = append(colInds, j-1)
		}
	}

	// Sort by row index
	for i := 0; i < len(rowInds)-1; i++ {
		for j := i + 1; j < len(rowInds); j++ {
			if rowInds[j] < rowInds[i] {
				rowInds[i], rowInds[j] = rowInds[j], rowInds[i]
				colInds[i], colInds[j] = colInds[j], colInds[i]
			}
		}
	}

	return rowInds, colInds, nil
}

// Linprog solves a linear programming problem using the simplex method:
//
//	minimize    c^T x
//	subject to  Aub * x <= bub
//	            Aeq * x == beq
//	            x >= 0
//
// Aub/bub can be nil for no inequality constraints. Aeq/beq can be nil for no equality constraints.
func Linprog(c []float64, Aub [][]float64, bub []float64, Aeq [][]float64, beq []float64) (*OptResult, error) {
	nVars := len(c)
	if nVars == 0 {
		return nil, errors.New("scigo: Linprog: c must not be empty")
	}

	nIneq := len(Aub)
	nEq := len(Aeq)

	if nIneq > 0 && len(bub) != nIneq {
		return nil, errors.New("scigo: Linprog: Aub and bub dimension mismatch")
	}
	if nEq > 0 && len(beq) != nEq {
		return nil, errors.New("scigo: Linprog: Aeq and beq dimension mismatch")
	}

	// Convert to standard form with slack variables:
	// Aub * x + s = bub, s >= 0
	// Total variables: nVars (original) + nIneq (slack) + nEq (artificial)
	nSlack := nIneq
	nArtificial := nEq
	totalVars := nVars + nSlack + nArtificial
	totalConstraints := nIneq + nEq

	if totalConstraints == 0 {
		// No constraints: if c has any negative component, unbounded
		x := make([]float64, nVars)
		return &OptResult{X: x, Fun: 0, Success: true, Nit: 0}, nil
	}

	// Build tableau
	// Rows: totalConstraints + 1 (objective)
	// Cols: totalVars + 1 (rhs)
	tableau := make([][]float64, totalConstraints+1)
	for i := range tableau {
		tableau[i] = make([]float64, totalVars+1)
	}

	// Fill inequality constraints
	for i := 0; i < nIneq; i++ {
		if len(Aub[i]) != nVars {
			return nil, errors.New("scigo: Linprog: Aub row dimension mismatch")
		}
		for j := 0; j < nVars; j++ {
			tableau[i][j] = Aub[i][j]
		}
		tableau[i][nVars+i] = 1 // slack variable
		tableau[i][totalVars] = bub[i]
	}

	// Fill equality constraints (with artificial variables)
	for i := 0; i < nEq; i++ {
		if len(Aeq[i]) != nVars {
			return nil, errors.New("scigo: Linprog: Aeq row dimension mismatch")
		}
		row := nIneq + i
		for j := 0; j < nVars; j++ {
			tableau[row][j] = Aeq[i][j]
		}
		tableau[row][nVars+nSlack+i] = 1 // artificial variable
		tableau[row][totalVars] = beq[i]
	}

	// Objective row: for phase 1 if we have artificial vars, otherwise direct
	basis := make([]int, totalConstraints)
	for i := 0; i < nIneq; i++ {
		basis[i] = nVars + i // slack variables
	}
	for i := 0; i < nEq; i++ {
		basis[nIneq+i] = nVars + nSlack + i // artificial variables
	}

	// If there are artificial variables, use two-phase simplex
	if nArtificial > 0 {
		// Phase 1: minimize sum of artificial variables
		// Set objective = sum of artificial vars
		for j := 0; j <= totalVars; j++ {
			tableau[totalConstraints][j] = 0
		}
		for i := 0; i < nArtificial; i++ {
			for j := 0; j <= totalVars; j++ {
				tableau[totalConstraints][j] -= tableau[nIneq+i][j]
			}
		}

		if err := simplexPivot(tableau, basis, totalConstraints, totalVars, 1000); err != nil {
			return nil, errors.New("scigo: Linprog: phase 1 failed: " + err.Error())
		}

		// Check if artificial variables are in basis with nonzero value
		for i, b := range basis {
			if b >= nVars+nSlack {
				if math.Abs(tableau[i][totalVars]) > 1e-10 {
					return nil, errors.New("scigo: Linprog: problem is infeasible")
				}
			}
		}

		// Phase 2: set original objective
		for j := 0; j <= totalVars; j++ {
			tableau[totalConstraints][j] = 0
		}
		for j := 0; j < nVars; j++ {
			tableau[totalConstraints][j] = c[j]
		}
		// Make artificial variable columns very expensive
		for j := nVars + nSlack; j < totalVars; j++ {
			tableau[totalConstraints][j] = 1e10
		}
		// Pivot objective row to remove basic variables
		for i, b := range basis {
			if tableau[totalConstraints][b] != 0 {
				ratio := tableau[totalConstraints][b]
				for j := 0; j <= totalVars; j++ {
					tableau[totalConstraints][j] -= ratio * tableau[i][j]
				}
			}
		}
	} else {
		// Single-phase: set objective directly
		for j := 0; j < nVars; j++ {
			tableau[totalConstraints][j] = c[j]
		}
		// Pivot out basic slack variables from objective
		for i, b := range basis {
			if tableau[totalConstraints][b] != 0 {
				ratio := tableau[totalConstraints][b]
				for j := 0; j <= totalVars; j++ {
					tableau[totalConstraints][j] -= ratio * tableau[i][j]
				}
			}
		}
	}

	if err := simplexPivot(tableau, basis, totalConstraints, totalVars, 1000); err != nil {
		return nil, errors.New("scigo: Linprog: " + err.Error())
	}

	// Extract solution
	x := make([]float64, nVars)
	for i, b := range basis {
		if b < nVars {
			x[b] = tableau[i][totalVars]
		}
	}
	fun := 0.0
	for j := 0; j < nVars; j++ {
		fun += c[j] * x[j]
	}

	return &OptResult{X: x, Fun: fun, Success: true, Nit: 0}, nil
}

// simplexPivot performs simplex pivoting on the tableau.
func simplexPivot(tableau [][]float64, basis []int, nConstraints, nVars, maxIter int) error {
	for iter := 0; iter < maxIter; iter++ {
		// Find entering variable (most negative reduced cost)
		pivotCol := -1
		minRC := -1e-10
		for j := 0; j < nVars; j++ {
			if tableau[nConstraints][j] < minRC {
				minRC = tableau[nConstraints][j]
				pivotCol = j
			}
		}
		if pivotCol == -1 {
			return nil // optimal
		}

		// Find leaving variable (minimum ratio test)
		pivotRow := -1
		minRatio := math.Inf(1)
		for i := 0; i < nConstraints; i++ {
			if tableau[i][pivotCol] > 1e-10 {
				ratio := tableau[i][nVars] / tableau[i][pivotCol]
				if ratio < minRatio {
					minRatio = ratio
					pivotRow = i
				}
			}
		}
		if pivotRow == -1 {
			return errors.New("problem is unbounded")
		}

		// Pivot
		pivotElement := tableau[pivotRow][pivotCol]
		for j := 0; j <= nVars; j++ {
			tableau[pivotRow][j] /= pivotElement
		}
		for i := 0; i <= nConstraints; i++ {
			if i != pivotRow {
				factor := tableau[i][pivotCol]
				for j := 0; j <= nVars; j++ {
					tableau[i][j] -= factor * tableau[pivotRow][j]
				}
			}
		}
		basis[pivotRow] = pivotCol
	}
	return errors.New("maximum iterations exceeded")
}

// DifferentialEvolution performs global optimization using the differential evolution algorithm.
// bounds[i] specifies the [lower, upper] bound for each dimension.
func DifferentialEvolution(f func([]float64) float64, bounds [][2]float64) (*OptResult, error) {
	n := len(bounds)
	if n == 0 {
		return nil, errors.New("scigo: DifferentialEvolution: bounds must not be empty")
	}

	const (
		popSize = 50
		maxGen  = 1000
		F       = 0.8
		CR      = 0.9
		tol     = 1e-12
	)

	rng := rand.New(rand.NewSource(42))

	// Initialize population
	np := popSize
	if np < 4 {
		np = 4
	}
	pop := make([][]float64, np)
	fitness := make([]float64, np)
	for i := 0; i < np; i++ {
		pop[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			pop[i][j] = bounds[j][0] + rng.Float64()*(bounds[j][1]-bounds[j][0])
		}
		fitness[i] = f(pop[i])
	}

	bestIdx := 0
	for i := 1; i < np; i++ {
		if fitness[i] < fitness[bestIdx] {
			bestIdx = i
		}
	}

	var gen int
	for gen = 0; gen < maxGen; gen++ {
		improved := false
		for i := 0; i < np; i++ {
			// Pick 3 distinct random individuals != i
			r := make([]int, 3)
			for k := 0; k < 3; k++ {
				for {
					r[k] = rng.Intn(np)
					ok := r[k] != i
					for l := 0; l < k && ok; l++ {
						if r[k] == r[l] {
							ok = false
						}
					}
					if ok {
						break
					}
				}
			}

			// Mutation and crossover
			trial := make([]float64, n)
			jRand := rng.Intn(n)
			for j := 0; j < n; j++ {
				if rng.Float64() < CR || j == jRand {
					trial[j] = pop[r[0]][j] + F*(pop[r[1]][j]-pop[r[2]][j])
					// Clamp to bounds
					if trial[j] < bounds[j][0] {
						trial[j] = bounds[j][0]
					}
					if trial[j] > bounds[j][1] {
						trial[j] = bounds[j][1]
					}
				} else {
					trial[j] = pop[i][j]
				}
			}

			trialFit := f(trial)
			if trialFit <= fitness[i] {
				pop[i] = trial
				fitness[i] = trialFit
				if trialFit < fitness[bestIdx] {
					bestIdx = i
					improved = true
				}
			}
		}

		// Check convergence
		fMax, fMin := fitness[0], fitness[0]
		for _, fi := range fitness[1:] {
			if fi > fMax {
				fMax = fi
			}
			if fi < fMin {
				fMin = fi
			}
		}
		if fMax-fMin < tol && !improved {
			break
		}
	}

	result := make([]float64, n)
	copy(result, pop[bestIdx])
	return &OptResult{
		X:       result,
		Fun:     fitness[bestIdx],
		Success: gen < maxGen,
		Nit:     gen,
	}, nil
}

// BasinHopping performs global optimization using basin-hopping with Nelder-Mead
// as the local minimizer.
func BasinHopping(f func([]float64) float64, x0 []float64) (*OptResult, error) {
	n := len(x0)
	if n == 0 {
		return nil, errors.New("scigo: BasinHopping: x0 must not be empty")
	}

	const (
		maxIter     = 100
		stepSize    = 1.0
		temperature = 1.0
	)

	rng := rand.New(rand.NewSource(42))

	x := make([]float64, n)
	copy(x, x0)

	// Initial local minimization
	res, err := nelderMead(f, x)
	if err != nil {
		return nil, err
	}
	copy(x, res.X)
	fBest := res.Fun
	xBest := make([]float64, n)
	copy(xBest, x)

	var nit int
	for nit = 0; nit < maxIter; nit++ {
		// Perturb
		xNew := make([]float64, n)
		for i := range xNew {
			xNew[i] = x[i] + stepSize*rng.NormFloat64()
		}

		// Local minimization
		resNew, err := nelderMead(f, xNew)
		if err != nil {
			continue
		}

		// Metropolis acceptance criterion
		df := resNew.Fun - res.Fun
		accept := false
		if df < 0 {
			accept = true
		} else {
			if rng.Float64() < math.Exp(-df/temperature) {
				accept = true
			}
		}

		if accept {
			copy(x, resNew.X)
			res = resNew
			if resNew.Fun < fBest {
				fBest = resNew.Fun
				copy(xBest, resNew.X)
			}
		}
	}

	return &OptResult{
		X:       xBest,
		Fun:     fBest,
		Success: true,
		Nit:     nit,
	}, nil
}

// DualAnnealing performs global optimization using a dual annealing algorithm.
// It combines classical simulated annealing with Nelder-Mead local search.
func DualAnnealing(f func([]float64) float64, bounds [][2]float64) (*OptResult, error) {
	n := len(bounds)
	if n == 0 {
		return nil, errors.New("scigo: DualAnnealing: bounds must not be empty")
	}

	const (
		maxIter = 1000
		T0      = 5230.0
		Tf      = 1e-12
		qv      = 2.67
	)

	rng := rand.New(rand.NewSource(42))

	// Initial point: center of bounds
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = (bounds[i][0] + bounds[i][1]) / 2
	}

	res, err := nelderMead(f, x)
	if err != nil {
		return nil, err
	}
	xBest := make([]float64, n)
	copy(xBest, res.X)
	fBest := res.Fun

	var nit int
	for nit = 0; nit < maxIter; nit++ {
		// Temperature schedule
		T := T0 * math.Pow(Tf/T0, float64(nit)/float64(maxIter))

		// Generate visiting point
		xNew := make([]float64, n)
		for i := 0; i < n; i++ {
			// Cauchy-like perturbation scaled by temperature
			scale := (bounds[i][1] - bounds[i][0]) * math.Sqrt(T/T0)
			xNew[i] = x[i] + scale*rng.NormFloat64()
			if xNew[i] < bounds[i][0] {
				xNew[i] = bounds[i][0]
			}
			if xNew[i] > bounds[i][1] {
				xNew[i] = bounds[i][1]
			}
		}

		fNew := f(xNew)
		df := fNew - f(x)

		accept := false
		if df < 0 {
			accept = true
		} else if T > 0 {
			if rng.Float64() < math.Exp(-df/T) {
				accept = true
			}
		}

		if accept {
			copy(x, xNew)
		}

		// Periodically run local search
		if nit%100 == 99 {
			resLocal, err := nelderMead(f, x)
			if err == nil && resLocal.Fun < fBest {
				fBest = resLocal.Fun
				copy(xBest, resLocal.X)
				copy(x, resLocal.X)
			}
		}

		if fNew < fBest {
			fBest = fNew
			copy(xBest, xNew)
		}
	}

	// Final local search
	resFinal, err := nelderMead(f, xBest)
	if err == nil && resFinal.Fun < fBest {
		fBest = resFinal.Fun
		copy(xBest, resFinal.X)
	}

	return &OptResult{
		X:       xBest,
		Fun:     fBest,
		Success: true,
		Nit:     nit,
	}, nil
}

// SHGO performs Simplicial Homology Global Optimization (simplified version).
// It samples points within the bounds, performs local minimizations from the best
// candidates, and returns the global minimum found.
func SHGO(f func([]float64) float64, bounds [][2]float64) (*OptResult, error) {
	n := len(bounds)
	if n == 0 {
		return nil, errors.New("scigo: SHGO: bounds must not be empty")
	}

	const nSamples = 128

	rng := rand.New(rand.NewSource(42))

	// Sobol-like sampling (simplified: use stratified random)
	type candidate struct {
		x   []float64
		val float64
	}
	candidates := make([]candidate, nSamples)
	for i := 0; i < nSamples; i++ {
		x := make([]float64, n)
		for j := 0; j < n; j++ {
			x[j] = bounds[j][0] + rng.Float64()*(bounds[j][1]-bounds[j][0])
		}
		candidates[i] = candidate{x: x, val: f(x)}
	}

	// Sort by value, take top candidates for local search
	for i := 0; i < len(candidates)-1; i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].val < candidates[i].val {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	nLocal := 5
	if nLocal > nSamples {
		nLocal = nSamples
	}

	var bestResult *OptResult
	for i := 0; i < nLocal; i++ {
		res, err := nelderMead(f, candidates[i].x)
		if err != nil {
			continue
		}
		// Clamp to bounds
		for j := 0; j < n; j++ {
			if res.X[j] < bounds[j][0] {
				res.X[j] = bounds[j][0]
			}
			if res.X[j] > bounds[j][1] {
				res.X[j] = bounds[j][1]
			}
		}
		res.Fun = f(res.X)
		if bestResult == nil || res.Fun < bestResult.Fun {
			bestResult = res
		}
	}

	if bestResult == nil {
		return nil, errors.New("scigo: SHGO: all local minimizations failed")
	}
	bestResult.Success = true
	return bestResult, nil
}

// Direct performs the DIRECT (DIviding RECTangles) algorithm for global optimization (simplified).
// It recursively subdivides the search space and evaluates the function at center points.
func Direct(f func([]float64) float64, bounds [][2]float64) (*OptResult, error) {
	n := len(bounds)
	if n == 0 {
		return nil, errors.New("scigo: Direct: bounds must not be empty")
	}

	const maxEval = 5000

	type rectangle struct {
		center []float64
		size   []float64 // half-widths
		fval   float64
	}

	// Initial rectangle: full domain
	center := make([]float64, n)
	size := make([]float64, n)
	for i := 0; i < n; i++ {
		center[i] = (bounds[i][0] + bounds[i][1]) / 2
		size[i] = (bounds[i][1] - bounds[i][0]) / 2
	}

	rects := []rectangle{{center: center, size: size, fval: f(center)}}
	bestRect := &rects[0]
	nEval := 1

	for iter := 0; iter < 100 && nEval < maxEval; iter++ {
		// Find potentially optimal rectangles
		// Simplified: subdivide the rectangle with the best function value
		// among the largest rectangles
		if len(rects) == 0 {
			break
		}

		// Group by size and find best in each size group
		// Simplified: just pick the best rectangle overall
		bestIdx := 0
		for i := 1; i < len(rects); i++ {
			if rects[i].fval < rects[bestIdx].fval {
				bestIdx = i
			}
		}

		rect := rects[bestIdx]
		// Remove the rectangle being subdivided
		rects[bestIdx] = rects[len(rects)-1]
		rects = rects[:len(rects)-1]

		// Find the longest dimension
		maxDim := 0
		maxSize := rect.size[0]
		for d := 1; d < n; d++ {
			if rect.size[d] > maxSize {
				maxSize = rect.size[d]
				maxDim = d
			}
		}

		if maxSize < 1e-15 {
			// Rectangle is too small
			rects = append(rects, rect)
			break
		}

		// Trisect along the longest dimension
		delta := rect.size[maxDim] / 3

		newSize := make([]float64, n)
		copy(newSize, rect.size)
		newSize[maxDim] = delta

		// Left child
		leftCenter := make([]float64, n)
		copy(leftCenter, rect.center)
		leftCenter[maxDim] -= 2 * delta
		leftSize := make([]float64, n)
		copy(leftSize, newSize)
		fLeft := f(leftCenter)
		nEval++

		// Right child
		rightCenter := make([]float64, n)
		copy(rightCenter, rect.center)
		rightCenter[maxDim] += 2 * delta
		rightSize := make([]float64, n)
		copy(rightSize, newSize)
		fRight := f(rightCenter)
		nEval++

		// Center child (keeps the center point)
		centerSize := make([]float64, n)
		copy(centerSize, newSize)

		rects = append(rects,
			rectangle{center: leftCenter, size: leftSize, fval: fLeft},
			rectangle{center: rect.center, size: centerSize, fval: rect.fval},
			rectangle{center: rightCenter, size: rightSize, fval: fRight},
		)

		// Update best
		for i := len(rects) - 3; i < len(rects); i++ {
			if bestRect == nil || rects[i].fval < bestRect.fval {
				bestRect = &rects[i]
			}
		}
	}

	// Find overall best
	for i := range rects {
		if rects[i].fval < bestRect.fval {
			bestRect = &rects[i]
		}
	}

	result := make([]float64, n)
	copy(result, bestRect.center)

	return &OptResult{
		X:       result,
		Fun:     bestRect.fval,
		Success: true,
		Nit:     nEval,
	}, nil
}

// MILP solves a mixed-integer linear programming problem using branch-and-bound
// on top of the existing Linprog (LP relaxation).
//
//	minimize    c^T x
//	subject to  Aub * x <= bub
//	            x >= 0
//	            x[i] is integer for all i where integrality[i] is true
//
// This is a simplified implementation suitable for small problems only.
// Aub/bub can be nil for no inequality constraints.
func MILP(c []float64, Aub [][]float64, bub []float64, integrality []bool) (*OptResult, error) {
	nVars := len(c)
	if nVars == 0 {
		return nil, errors.New("scigo: MILP: c must not be empty")
	}
	if integrality != nil && len(integrality) != nVars {
		return nil, errors.New("scigo: MILP: integrality must have same length as c")
	}

	// If no integrality constraints, just solve the LP.
	hasInt := false
	if integrality != nil {
		for _, v := range integrality {
			if v {
				hasInt = true
				break
			}
		}
	}
	if !hasInt {
		return Linprog(c, Aub, bub, nil, nil)
	}

	type node struct {
		extraAub [][]float64
		extraBub []float64
	}

	bestFun := math.Inf(1)
	var bestX []float64

	// Branch-and-bound using a stack (DFS).
	stack := []node{{nil, nil}}
	maxNodes := 10000
	visited := 0

	for len(stack) > 0 && visited < maxNodes {
		visited++
		// Pop from stack.
		cur := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		// Build full constraint set.
		totalIneq := len(Aub) + len(cur.extraAub)
		fullAub := make([][]float64, 0, totalIneq)
		fullBub := make([]float64, 0, totalIneq)
		if Aub != nil {
			fullAub = append(fullAub, Aub...)
			fullBub = append(fullBub, bub...)
		}
		fullAub = append(fullAub, cur.extraAub...)
		fullBub = append(fullBub, cur.extraBub...)

		var res *OptResult
		var err error
		if len(fullAub) == 0 {
			res, err = Linprog(c, nil, nil, nil, nil)
		} else {
			res, err = Linprog(c, fullAub, fullBub, nil, nil)
		}
		if err != nil || !res.Success {
			continue // Infeasible or unbounded, prune.
		}

		// Prune if LP relaxation is worse than best known.
		if res.Fun >= bestFun-1e-10 {
			continue
		}

		// Check if all integer variables are integral.
		fracIdx := -1
		maxFrac := 0.0
		for i := 0; i < nVars; i++ {
			if integrality[i] {
				rounded := math.Round(res.X[i])
				frac := math.Abs(res.X[i] - rounded)
				if frac > 1e-6 && frac > maxFrac {
					maxFrac = frac
					fracIdx = i
				}
			}
		}

		if fracIdx == -1 {
			// All integer constraints satisfied.
			if res.Fun < bestFun {
				bestFun = res.Fun
				bestX = make([]float64, nVars)
				copy(bestX, res.X)
			}
			continue
		}

		// Branch on fracIdx.
		val := res.X[fracIdx]
		floorVal := math.Floor(val)
		ceilVal := math.Ceil(val)

		// Left branch: x[fracIdx] <= floor(val)
		// Encode as: 1*x[fracIdx] <= floorVal
		leftRow := make([]float64, nVars)
		leftRow[fracIdx] = 1
		leftExtra := make([][]float64, len(cur.extraAub)+1)
		leftBubExtra := make([]float64, len(cur.extraBub)+1)
		copy(leftExtra, cur.extraAub)
		copy(leftBubExtra, cur.extraBub)
		leftExtra[len(cur.extraAub)] = leftRow
		leftBubExtra[len(cur.extraBub)] = floorVal

		// Right branch: x[fracIdx] >= ceil(val) => -x[fracIdx] <= -ceilVal
		rightRow := make([]float64, nVars)
		rightRow[fracIdx] = -1
		rightExtra := make([][]float64, len(cur.extraAub)+1)
		rightBubExtra := make([]float64, len(cur.extraBub)+1)
		copy(rightExtra, cur.extraAub)
		copy(rightBubExtra, cur.extraBub)
		rightExtra[len(cur.extraAub)] = rightRow
		rightBubExtra[len(cur.extraBub)] = -ceilVal

		stack = append(stack,
			node{leftExtra, leftBubExtra},
			node{rightExtra, rightBubExtra},
		)
	}

	if bestX == nil {
		return nil, errors.New("scigo: MILP: no feasible integer solution found")
	}

	// Round integer variables to clean values.
	for i := 0; i < nVars; i++ {
		if integrality[i] {
			bestX[i] = math.Round(bestX[i])
		}
	}
	fun := 0.0
	for i := 0; i < nVars; i++ {
		fun += c[i] * bestX[i]
	}

	return &OptResult{
		X:       bestX,
		Fun:     fun,
		Success: true,
		Nit:     visited,
	}, nil
}
