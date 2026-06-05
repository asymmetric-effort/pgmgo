package scigo

import (
	"math"
	"math/rand"
)

// BrownianMotion generates standard Brownian motion paths.
// T is the final time, n is the number of time steps, nPaths is the number of
// independent paths, and seed controls the random number generator.
func BrownianMotion(T float64, n, nPaths int, seed int64) *SDEResult {
	if T <= 0 {
		panic("scigo: BrownianMotion T must be positive")
	}
	if n <= 0 {
		panic("scigo: BrownianMotion n must be positive")
	}
	if nPaths <= 0 {
		panic("scigo: BrownianMotion nPaths must be positive")
	}

	dt := T / float64(n)
	sqrtDt := math.Sqrt(dt)

	tVals := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		tVals[i] = float64(i) * dt
	}

	rng := rand.New(rand.NewSource(seed))
	paths := make([][]float64, nPaths)
	for p := 0; p < nPaths; p++ {
		path := make([]float64, n+1)
		path[0] = 0
		for i := 1; i <= n; i++ {
			path[i] = path[i-1] + rng.NormFloat64()*sqrtDt
		}
		paths[p] = path
	}

	return &SDEResult{T: tVals, X: paths}
}

// GeometricBrownianMotion generates paths of the process
//
//	dS = mu * S * dt + sigma * S * dW
//
// with initial value S0. The exact solution is used:
//
//	S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
func GeometricBrownianMotion(S0, mu, sigma, T float64, n, nPaths int, seed int64) *SDEResult {
	if T <= 0 {
		panic("scigo: GeometricBrownianMotion T must be positive")
	}
	if n <= 0 {
		panic("scigo: GeometricBrownianMotion n must be positive")
	}
	if nPaths <= 0 {
		panic("scigo: GeometricBrownianMotion nPaths must be positive")
	}

	dt := T / float64(n)
	sqrtDt := math.Sqrt(dt)
	drift := (mu - 0.5*sigma*sigma) * dt

	tVals := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		tVals[i] = float64(i) * dt
	}

	rng := rand.New(rand.NewSource(seed))
	paths := make([][]float64, nPaths)
	for p := 0; p < nPaths; p++ {
		path := make([]float64, n+1)
		path[0] = S0
		for i := 1; i <= n; i++ {
			dW := rng.NormFloat64() * sqrtDt
			path[i] = path[i-1] * math.Exp(drift+sigma*dW)
		}
		paths[p] = path
	}

	return &SDEResult{T: tVals, X: paths}
}

// OrnsteinUhlenbeck generates paths of the Ornstein-Uhlenbeck process
//
//	dX = theta * (mu - X) * dt + sigma * dW
//
// with initial value x0. Uses the exact discretization.
func OrnsteinUhlenbeck(x0, theta, mu, sigma, T float64, n, nPaths int, seed int64) *SDEResult {
	if T <= 0 {
		panic("scigo: OrnsteinUhlenbeck T must be positive")
	}
	if n <= 0 {
		panic("scigo: OrnsteinUhlenbeck n must be positive")
	}
	if nPaths <= 0 {
		panic("scigo: OrnsteinUhlenbeck nPaths must be positive")
	}

	dt := T / float64(n)
	expTheta := math.Exp(-theta * dt)
	// Variance of the conditional distribution
	var stdDev float64
	if theta > 0 {
		stdDev = math.Sqrt(sigma * sigma / (2 * theta) * (1 - math.Exp(-2*theta*dt)))
	} else {
		stdDev = sigma * math.Sqrt(dt)
	}

	tVals := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		tVals[i] = float64(i) * dt
	}

	rng := rand.New(rand.NewSource(seed))
	paths := make([][]float64, nPaths)
	for p := 0; p < nPaths; p++ {
		path := make([]float64, n+1)
		path[0] = x0
		for i := 1; i <= n; i++ {
			path[i] = mu + (path[i-1]-mu)*expTheta + stdDev*rng.NormFloat64()
		}
		paths[p] = path
	}

	return &SDEResult{T: tVals, X: paths}
}

// BrownianBridge generates a Brownian bridge from startVal at time 0 to endVal at time T.
// The bridge has n time steps. seed controls the random number generator.
// Returns a slice of length n+1 containing the bridge values.
func BrownianBridge(T float64, n int, startVal, endVal float64, seed int64) []float64 {
	if T <= 0 {
		panic("scigo: BrownianBridge T must be positive")
	}
	if n <= 0 {
		panic("scigo: BrownianBridge n must be positive")
	}

	dt := T / float64(n)
	sqrtDt := math.Sqrt(dt)

	rng := rand.New(rand.NewSource(seed))

	// Generate a standard Brownian motion
	w := make([]float64, n+1)
	w[0] = 0
	for i := 1; i <= n; i++ {
		w[i] = w[i-1] + rng.NormFloat64()*sqrtDt
	}

	// Transform to bridge: B(t) = W(t) - (t/T)*W(T) + startVal + (t/T)*(endVal - startVal)
	bridge := make([]float64, n+1)
	wT := w[n]
	for i := 0; i <= n; i++ {
		tRatio := float64(i) / float64(n)
		bridge[i] = w[i] - tRatio*wT + startVal + tRatio*(endVal-startVal)
	}

	return bridge
}

// QuadraticVariation computes the quadratic variation of a discrete path
// with time step dt. For a standard Brownian motion, this should converge to T.
func QuadraticVariation(path []float64, dt float64) float64 {
	if len(path) < 2 {
		return 0
	}
	qv := 0.0
	for i := 1; i < len(path); i++ {
		diff := path[i] - path[i-1]
		qv += diff * diff
	}
	return qv
}

// Covariation computes the cross-variation (covariation) of two discrete paths
// with time step dt. Both paths must have the same length.
func Covariation(path1, path2 []float64, dt float64) float64 {
	if len(path1) != len(path2) {
		panic("scigo: Covariation paths must have equal length")
	}
	if len(path1) < 2 {
		return 0
	}
	cv := 0.0
	for i := 1; i < len(path1); i++ {
		d1 := path1[i] - path1[i-1]
		d2 := path2[i] - path2[i-1]
		cv += d1 * d2
	}
	return cv
}
