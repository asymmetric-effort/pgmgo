package scigo

import (
	"math"
	"math/rand"
)

// SDEResult holds the result of an SDE simulation.
// T contains the time points and X contains the sample paths.
// X[i] is the i-th sample path, so X[i][j] is the value of the i-th path at time T[j].
type SDEResult struct {
	T []float64
	X [][]float64
}

// SDESystem represents a vector-valued SDE system:
//
//	dX = Drift(X, t) dt + Diffusion(X, t) dW
//
// Drift and Diffusion take a state vector and time, returning a vector.
type SDESystem struct {
	Dim       int
	Drift     func(x []float64, t float64) []float64
	Diffusion func(x []float64, t float64) []float64
}

// EulerMaruyama solves the scalar SDE
//
//	dX = drift(X, t) dt + diffusion(X, t) dW
//
// using the Euler-Maruyama method. It generates nPaths sample paths from x0
// over the interval tSpan with time step dt. seed controls the random number generator.
func EulerMaruyama(drift, diffusion func(x, t float64) float64, x0 float64, tSpan [2]float64, dt float64, nPaths int, seed int64) *SDEResult {
	if dt <= 0 {
		panic("scigo: EulerMaruyama dt must be positive")
	}
	if nPaths <= 0 {
		panic("scigo: EulerMaruyama nPaths must be positive")
	}
	if tSpan[1] <= tSpan[0] {
		panic("scigo: EulerMaruyama tSpan[1] must be greater than tSpan[0]")
	}

	nSteps := int(math.Ceil((tSpan[1] - tSpan[0]) / dt))
	actualDt := (tSpan[1] - tSpan[0]) / float64(nSteps)
	sqrtDt := math.Sqrt(actualDt)

	tVals := make([]float64, nSteps+1)
	for i := 0; i <= nSteps; i++ {
		tVals[i] = tSpan[0] + float64(i)*actualDt
	}

	paths := make([][]float64, nPaths)
	rng := rand.New(rand.NewSource(seed))

	for p := 0; p < nPaths; p++ {
		path := make([]float64, nSteps+1)
		path[0] = x0
		for i := 0; i < nSteps; i++ {
			t := tVals[i]
			x := path[i]
			dW := rng.NormFloat64() * sqrtDt
			path[i+1] = x + drift(x, t)*actualDt + diffusion(x, t)*dW
		}
		paths[p] = path
	}

	return &SDEResult{T: tVals, X: paths}
}

// Milstein solves the scalar SDE
//
//	dX = drift(X, t) dt + diffusion(X, t) dW
//
// using the Milstein method, which includes a correction term involving
// the derivative of the diffusion coefficient. This gives strong order 1.0
// convergence compared to 0.5 for Euler-Maruyama.
func Milstein(drift, diffusion, diffusionDeriv func(x, t float64) float64, x0 float64, tSpan [2]float64, dt float64, nPaths int, seed int64) *SDEResult {
	if dt <= 0 {
		panic("scigo: Milstein dt must be positive")
	}
	if nPaths <= 0 {
		panic("scigo: Milstein nPaths must be positive")
	}
	if tSpan[1] <= tSpan[0] {
		panic("scigo: Milstein tSpan[1] must be greater than tSpan[0]")
	}

	nSteps := int(math.Ceil((tSpan[1] - tSpan[0]) / dt))
	actualDt := (tSpan[1] - tSpan[0]) / float64(nSteps)
	sqrtDt := math.Sqrt(actualDt)

	tVals := make([]float64, nSteps+1)
	for i := 0; i <= nSteps; i++ {
		tVals[i] = tSpan[0] + float64(i)*actualDt
	}

	paths := make([][]float64, nPaths)
	rng := rand.New(rand.NewSource(seed))

	for p := 0; p < nPaths; p++ {
		path := make([]float64, nSteps+1)
		path[0] = x0
		for i := 0; i < nSteps; i++ {
			t := tVals[i]
			x := path[i]
			dW := rng.NormFloat64() * sqrtDt
			sig := diffusion(x, t)
			sigPrime := diffusionDeriv(x, t)
			path[i+1] = x + drift(x, t)*actualDt + sig*dW + 0.5*sig*sigPrime*(dW*dW-actualDt)
		}
		paths[p] = path
	}

	return &SDEResult{T: tVals, X: paths}
}

// SolveSDESystem solves a vector-valued SDE system using the Euler-Maruyama method.
// x0 is the initial state vector, tSpan is the time interval, dt is the time step,
// nPaths is the number of sample paths, and seed controls the random number generator.
// The result contains nPaths * dim entries in X, organized as:
// X[p*dim + d] is the d-th component of the p-th sample path.
func SolveSDESystem(sys *SDESystem, x0 []float64, tSpan [2]float64, dt float64, nPaths int, seed int64) *SDEResult {
	if sys == nil {
		panic("scigo: SolveSDESystem sys must not be nil")
	}
	if len(x0) != sys.Dim {
		panic("scigo: SolveSDESystem x0 dimension mismatch")
	}
	if dt <= 0 {
		panic("scigo: SolveSDESystem dt must be positive")
	}
	if nPaths <= 0 {
		panic("scigo: SolveSDESystem nPaths must be positive")
	}
	if tSpan[1] <= tSpan[0] {
		panic("scigo: SolveSDESystem tSpan[1] must be greater than tSpan[0]")
	}

	nSteps := int(math.Ceil((tSpan[1] - tSpan[0]) / dt))
	actualDt := (tSpan[1] - tSpan[0]) / float64(nSteps)
	sqrtDt := math.Sqrt(actualDt)
	dim := sys.Dim

	tVals := make([]float64, nSteps+1)
	for i := 0; i <= nSteps; i++ {
		tVals[i] = tSpan[0] + float64(i)*actualDt
	}

	// X is organized as nPaths*dim paths, each of length nSteps+1.
	paths := make([][]float64, nPaths*dim)
	rng := rand.New(rand.NewSource(seed))

	for p := 0; p < nPaths; p++ {
		// Initialize paths for this sample
		for d := 0; d < dim; d++ {
			paths[p*dim+d] = make([]float64, nSteps+1)
			paths[p*dim+d][0] = x0[d]
		}

		state := make([]float64, dim)
		copy(state, x0)

		for i := 0; i < nSteps; i++ {
			t := tVals[i]
			dr := sys.Drift(state, t)
			diff := sys.Diffusion(state, t)

			for d := 0; d < dim; d++ {
				dW := rng.NormFloat64() * sqrtDt
				state[d] = state[d] + dr[d]*actualDt + diff[d]*dW
				paths[p*dim+d][i+1] = state[d]
			}
		}
	}

	return &SDEResult{T: tVals, X: paths}
}
