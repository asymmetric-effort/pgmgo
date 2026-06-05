//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// BrownianMotion Tests
// ---------------------------------------------------------------------------

func TestBrownianMotionStartsAtZero(t *testing.T) {
	result := BrownianMotion(1.0, 100, 10, 42)
	for p := 0; p < 10; p++ {
		if result.X[p][0] != 0 {
			t.Errorf("path %d should start at 0, got %v", p, result.X[p][0])
		}
	}
}

func TestBrownianMotionMeanZero(t *testing.T) {
	nPaths := 50000
	result := BrownianMotion(1.0, 1000, nPaths, 123)
	n := len(result.T) - 1

	sum := 0.0
	for p := 0; p < nPaths; p++ {
		sum += result.X[p][n]
	}
	mean := sum / float64(nPaths)
	// E[W(1)] = 0
	if math.Abs(mean) > 0.05 {
		t.Errorf("BrownianMotion mean at T=1: got %v, want ~0", mean)
	}
}

func TestBrownianMotionVariance(t *testing.T) {
	nPaths := 50000
	T := 2.0
	result := BrownianMotion(T, 1000, nPaths, 456)
	n := len(result.T) - 1

	sum := 0.0
	sum2 := 0.0
	for p := 0; p < nPaths; p++ {
		v := result.X[p][n]
		sum += v
		sum2 += v * v
	}
	mean := sum / float64(nPaths)
	variance := sum2/float64(nPaths) - mean*mean
	// Var[W(T)] = T
	if math.Abs(variance-T) > 0.1 {
		t.Errorf("BrownianMotion variance at T=%v: got %v, want ~%v", T, variance, T)
	}
}

func TestBrownianMotionTimePoints(t *testing.T) {
	result := BrownianMotion(2.0, 100, 1, 42)
	if len(result.T) != 101 {
		t.Errorf("expected 101 time points, got %d", len(result.T))
	}
	if result.T[0] != 0 {
		t.Errorf("T[0] = %v, want 0", result.T[0])
	}
	if math.Abs(result.T[100]-2.0) > 1e-10 {
		t.Errorf("T[100] = %v, want 2.0", result.T[100])
	}
}

func TestBrownianMotionPanics(t *testing.T) {
	assertPanics(t, "negative T", func() { BrownianMotion(-1, 100, 1, 42) })
	assertPanics(t, "zero n", func() { BrownianMotion(1, 0, 1, 42) })
	assertPanics(t, "zero nPaths", func() { BrownianMotion(1, 100, 0, 42) })
}

// ---------------------------------------------------------------------------
// GeometricBrownianMotion Tests
// ---------------------------------------------------------------------------

func TestGBMMeanConvergence(t *testing.T) {
	// E[S(T)] = S0 * exp(mu * T)
	S0 := 100.0
	mu := 0.1
	sigma := 0.3
	T := 1.0
	nPaths := 100000

	result := GeometricBrownianMotion(S0, mu, sigma, T, 1000, nPaths, 789)
	n := len(result.T) - 1

	sum := 0.0
	for p := 0; p < nPaths; p++ {
		sum += result.X[p][n]
	}
	mean := sum / float64(nPaths)
	expected := S0 * math.Exp(mu*T)

	relErr := math.Abs(mean-expected) / expected
	if relErr > 0.02 {
		t.Errorf("GBM mean = %v, want ~%v (relErr=%v)", mean, expected, relErr)
	}
}

func TestGBMPositive(t *testing.T) {
	result := GeometricBrownianMotion(100, 0.05, 0.3, 1, 100, 100, 42)
	for p := 0; p < 100; p++ {
		for i := 0; i <= 100; i++ {
			if result.X[p][i] <= 0 {
				t.Fatalf("GBM path %d step %d is non-positive: %v", p, i, result.X[p][i])
			}
		}
	}
}

func TestGBMStartsAtS0(t *testing.T) {
	result := GeometricBrownianMotion(42.0, 0.05, 0.2, 1, 100, 5, 42)
	for p := 0; p < 5; p++ {
		if result.X[p][0] != 42.0 {
			t.Errorf("GBM path %d start = %v, want 42.0", p, result.X[p][0])
		}
	}
}

func TestGBMPanics(t *testing.T) {
	assertPanics(t, "negative T", func() { GeometricBrownianMotion(100, 0.05, 0.2, -1, 100, 1, 42) })
	assertPanics(t, "zero n", func() { GeometricBrownianMotion(100, 0.05, 0.2, 1, 0, 1, 42) })
	assertPanics(t, "zero nPaths", func() { GeometricBrownianMotion(100, 0.05, 0.2, 1, 100, 0, 42) })
}

// ---------------------------------------------------------------------------
// OrnsteinUhlenbeck Tests
// ---------------------------------------------------------------------------

func TestOUMeanReversion(t *testing.T) {
	// OU process should revert to mu
	x0 := 10.0
	theta := 5.0 // Strong mean reversion
	mu := 0.0
	sigma := 0.5
	T := 5.0
	nPaths := 50000

	result := OrnsteinUhlenbeck(x0, theta, mu, sigma, T, 1000, nPaths, 42)
	n := len(result.T) - 1

	sum := 0.0
	for p := 0; p < nPaths; p++ {
		sum += result.X[p][n]
	}
	mean := sum / float64(nPaths)

	// After long time, E[X] -> mu = 0
	// Exact: E[X(T)] = mu + (x0-mu)*exp(-theta*T) which is ~0 for large theta*T
	expected := mu + (x0-mu)*math.Exp(-theta*T)
	if math.Abs(mean-expected) > 0.05 {
		t.Errorf("OU mean at T=%v: got %v, want ~%v", T, mean, expected)
	}
}

func TestOUStartsAtX0(t *testing.T) {
	result := OrnsteinUhlenbeck(5.0, 1, 0, 0.5, 1, 100, 10, 42)
	for p := 0; p < 10; p++ {
		if result.X[p][0] != 5.0 {
			t.Errorf("OU path %d start = %v, want 5.0", p, result.X[p][0])
		}
	}
}

func TestOUStationaryVariance(t *testing.T) {
	// Stationary variance = sigma^2 / (2*theta)
	theta := 2.0
	mu := 0.0
	sigma := 1.0
	T := 10.0
	nPaths := 50000

	// Start at the mean to avoid transient effects
	result := OrnsteinUhlenbeck(mu, theta, mu, sigma, T, 2000, nPaths, 555)
	n := len(result.T) - 1

	sum := 0.0
	sum2 := 0.0
	for p := 0; p < nPaths; p++ {
		v := result.X[p][n]
		sum += v
		sum2 += v * v
	}
	mean := sum / float64(nPaths)
	variance := sum2/float64(nPaths) - mean*mean
	expected := sigma * sigma / (2 * theta)

	if math.Abs(variance-expected)/expected > 0.05 {
		t.Errorf("OU stationary variance: got %v, want ~%v", variance, expected)
	}
}

func TestOUPanics(t *testing.T) {
	assertPanics(t, "negative T", func() { OrnsteinUhlenbeck(0, 1, 0, 0.5, -1, 100, 1, 42) })
	assertPanics(t, "zero n", func() { OrnsteinUhlenbeck(0, 1, 0, 0.5, 1, 0, 1, 42) })
	assertPanics(t, "zero nPaths", func() { OrnsteinUhlenbeck(0, 1, 0, 0.5, 1, 100, 0, 42) })
}

// ---------------------------------------------------------------------------
// BrownianBridge Tests
// ---------------------------------------------------------------------------

func TestBrownianBridgeEndpoints(t *testing.T) {
	start := 1.0
	end := 3.0
	bridge := BrownianBridge(1.0, 100, start, end, 42)

	if math.Abs(bridge[0]-start) > 1e-10 {
		t.Errorf("bridge start = %v, want %v", bridge[0], start)
	}
	if math.Abs(bridge[100]-end) > 1e-10 {
		t.Errorf("bridge end = %v, want %v", bridge[100], end)
	}
}

func TestBrownianBridgeLength(t *testing.T) {
	bridge := BrownianBridge(1.0, 50, 0, 0, 42)
	if len(bridge) != 51 {
		t.Errorf("bridge length = %d, want 51", len(bridge))
	}
}

func TestBrownianBridgeZeroToZero(t *testing.T) {
	// Bridge from 0 to 0 should have E[B(t)] = 0
	nTrials := 10000
	n := 100
	midSum := 0.0
	for trial := 0; trial < nTrials; trial++ {
		bridge := BrownianBridge(1.0, n, 0, 0, int64(trial))
		midSum += bridge[n/2]
	}
	midMean := midSum / float64(nTrials)
	if math.Abs(midMean) > 0.05 {
		t.Errorf("Bridge 0->0 midpoint mean = %v, want ~0", midMean)
	}
}

func TestBrownianBridgePanics(t *testing.T) {
	assertPanics(t, "negative T", func() { BrownianBridge(-1, 100, 0, 0, 42) })
	assertPanics(t, "zero n", func() { BrownianBridge(1, 0, 0, 0, 42) })
}

// ---------------------------------------------------------------------------
// QuadraticVariation Tests
// ---------------------------------------------------------------------------

func TestQuadraticVariationBM(t *testing.T) {
	// QV of Brownian motion over [0, T] should converge to T
	T := 1.0
	n := 10000
	result := BrownianMotion(T, n, 1, 42)
	qv := QuadraticVariation(result.X[0], T/float64(n))

	if math.Abs(qv-T) > 0.1 {
		t.Errorf("QV of BM = %v, want ~%v", qv, T)
	}
}

func TestQuadraticVariationDeterministic(t *testing.T) {
	// QV of a smooth function should be ~0 as dt -> 0
	// f(t) = t, f'(t) = 1, increments = dt, sum of dt^2 = n*dt^2 = T*dt -> 0
	n := 10000
	dt := 1.0 / float64(n)
	path := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		path[i] = float64(i) * dt
	}
	qv := QuadraticVariation(path, dt)
	if qv > 0.001 {
		t.Errorf("QV of linear path = %v, want ~0", qv)
	}
}

func TestQuadraticVariationShortPath(t *testing.T) {
	qv := QuadraticVariation([]float64{1.0}, 0.01)
	if qv != 0 {
		t.Errorf("QV of single point = %v, want 0", qv)
	}
	qv2 := QuadraticVariation([]float64{}, 0.01)
	if qv2 != 0 {
		t.Errorf("QV of empty path = %v, want 0", qv2)
	}
}

// ---------------------------------------------------------------------------
// Covariation Tests
// ---------------------------------------------------------------------------

func TestCovariationSamePath(t *testing.T) {
	// Covariation of a path with itself = quadratic variation
	T := 1.0
	n := 5000
	result := BrownianMotion(T, n, 1, 42)
	dt := T / float64(n)

	qv := QuadraticVariation(result.X[0], dt)
	cv := Covariation(result.X[0], result.X[0], dt)

	if math.Abs(qv-cv) > 1e-10 {
		t.Errorf("Covariation(X,X) = %v, QV(X) = %v, should be equal", cv, qv)
	}
}

func TestCovariationIndependentBM(t *testing.T) {
	// Covariation of two independent BMs should be ~0
	T := 1.0
	n := 10000
	result := BrownianMotion(T, n, 2, 42)
	dt := T / float64(n)

	cv := Covariation(result.X[0], result.X[1], dt)
	if math.Abs(cv) > 0.2 {
		t.Errorf("Covariation of independent BMs = %v, want ~0", cv)
	}
}

func TestCovariationPanic(t *testing.T) {
	assertPanics(t, "length mismatch", func() {
		Covariation([]float64{1, 2, 3}, []float64{1, 2}, 0.01)
	})
}

func TestCovariationShortPaths(t *testing.T) {
	cv := Covariation([]float64{1.0}, []float64{2.0}, 0.01)
	if cv != 0 {
		t.Errorf("Covariation of single point = %v, want 0", cv)
	}
}

// ---------------------------------------------------------------------------
// Additional edge case / coverage tests
// ---------------------------------------------------------------------------

func TestBrownianMotionSingleStep(t *testing.T) {
	result := BrownianMotion(1.0, 1, 1, 42)
	if len(result.T) != 2 {
		t.Errorf("expected 2 time points for 1 step, got %d", len(result.T))
	}
	if len(result.X[0]) != 2 {
		t.Errorf("expected 2 path points for 1 step, got %d", len(result.X[0]))
	}
}

func TestGBMLogNormal(t *testing.T) {
	// ln(S(T)/S0) should be normally distributed with mean (mu-0.5*sigma^2)*T
	S0 := 100.0
	mu := 0.1
	sigma := 0.2
	T := 1.0
	nPaths := 50000

	result := GeometricBrownianMotion(S0, mu, sigma, T, 500, nPaths, 99)
	n := len(result.T) - 1

	logSum := 0.0
	for p := 0; p < nPaths; p++ {
		logSum += math.Log(result.X[p][n] / S0)
	}
	logMean := logSum / float64(nPaths)
	expected := (mu - 0.5*sigma*sigma) * T

	if math.Abs(logMean-expected) > 0.02 {
		t.Errorf("GBM log-mean = %v, want ~%v", logMean, expected)
	}
}

func TestOUZeroTheta(t *testing.T) {
	// With theta=0, OU becomes a random walk with drift 0
	result := OrnsteinUhlenbeck(0, 0, 0, 1, 1, 100, 10, 42)
	if len(result.X) != 10 {
		t.Errorf("expected 10 paths, got %d", len(result.X))
	}
}
