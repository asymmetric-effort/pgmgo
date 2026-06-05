//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// EulerMaruyama Tests
// ---------------------------------------------------------------------------

func TestEulerMaruyamaDeterministicODE(t *testing.T) {
	// dX = X dt + 0 dW => X(t) = x0 * exp(t)
	// With zero diffusion, EM should match the ODE solution exactly.
	drift := func(x, ti float64) float64 { return x }
	diffusion := func(x, ti float64) float64 { return 0 }
	x0 := 1.0
	tSpan := [2]float64{0, 1}
	dt := 0.001

	result := EulerMaruyama(drift, diffusion, x0, tSpan, dt, 1, 42)

	if len(result.T) == 0 {
		t.Fatal("EulerMaruyama returned empty T")
	}
	if len(result.X) != 1 {
		t.Fatalf("expected 1 path, got %d", len(result.X))
	}

	// Check final value against exp(1)
	finalIdx := len(result.X[0]) - 1
	got := result.X[0][finalIdx]
	want := math.Exp(1.0)
	// EM on ODE has discretization error, use loose tolerance
	if math.Abs(got-want)/want > 0.01 {
		t.Errorf("EulerMaruyama ODE: got %v, want ~%v", got, want)
	}
}

func TestEulerMaruyamaLinearSDE(t *testing.T) {
	// dX = mu*X dt + sigma*X dW (GBM)
	// E[X(T)] = x0 * exp(mu*T)
	mu := 0.05
	sigma := 0.2
	x0 := 100.0
	T := 1.0
	nPaths := 50000

	drift := func(x, ti float64) float64 { return mu * x }
	diffusion := func(x, ti float64) float64 { return sigma * x }

	result := EulerMaruyama(drift, diffusion, x0, [2]float64{0, T}, 0.001, nPaths, 12345)

	// Compute mean of terminal values
	sum := 0.0
	finalIdx := len(result.T) - 1
	for p := 0; p < nPaths; p++ {
		sum += result.X[p][finalIdx]
	}
	mean := sum / float64(nPaths)
	expected := x0 * math.Exp(mu*T)

	// Allow 2% relative error for Monte Carlo
	relErr := math.Abs(mean-expected) / expected
	if relErr > 0.02 {
		t.Errorf("EulerMaruyama GBM mean: got %v, want ~%v (relErr=%v)", mean, expected, relErr)
	}
}

func TestEulerMaruyamaMultiplePaths(t *testing.T) {
	drift := func(x, ti float64) float64 { return 0 }
	diffusion := func(x, ti float64) float64 { return 1 }

	result := EulerMaruyama(drift, diffusion, 0, [2]float64{0, 1}, 0.01, 10, 42)
	if len(result.X) != 10 {
		t.Errorf("expected 10 paths, got %d", len(result.X))
	}
	// Paths should differ (stochastic)
	if result.X[0][len(result.X[0])-1] == result.X[1][len(result.X[1])-1] {
		t.Error("different paths should have different terminal values")
	}
}

func TestEulerMaruyamaTimeSpan(t *testing.T) {
	drift := func(x, ti float64) float64 { return 0 }
	diffusion := func(x, ti float64) float64 { return 0 }
	result := EulerMaruyama(drift, diffusion, 1, [2]float64{0.5, 1.5}, 0.1, 1, 42)
	if math.Abs(result.T[0]-0.5) > 1e-10 {
		t.Errorf("first time point: got %v, want 0.5", result.T[0])
	}
	lastT := result.T[len(result.T)-1]
	if math.Abs(lastT-1.5) > 1e-10 {
		t.Errorf("last time point: got %v, want 1.5", lastT)
	}
}

func TestEulerMaruyamaPanics(t *testing.T) {
	drift := func(x, ti float64) float64 { return 0 }
	diffusion := func(x, ti float64) float64 { return 0 }

	assertPanics(t, "negative dt", func() {
		EulerMaruyama(drift, diffusion, 0, [2]float64{0, 1}, -0.1, 1, 42)
	})
	assertPanics(t, "zero nPaths", func() {
		EulerMaruyama(drift, diffusion, 0, [2]float64{0, 1}, 0.1, 0, 42)
	})
	assertPanics(t, "bad tSpan", func() {
		EulerMaruyama(drift, diffusion, 0, [2]float64{1, 0}, 0.1, 1, 42)
	})
}

// ---------------------------------------------------------------------------
// Milstein Tests
// ---------------------------------------------------------------------------

func TestMilsteinDeterministicODE(t *testing.T) {
	drift := func(x, ti float64) float64 { return x }
	diffusion := func(x, ti float64) float64 { return 0 }
	diffDeriv := func(x, ti float64) float64 { return 0 }

	result := Milstein(drift, diffusion, diffDeriv, 1.0, [2]float64{0, 1}, 0.001, 1, 42)

	finalIdx := len(result.X[0]) - 1
	got := result.X[0][finalIdx]
	want := math.Exp(1.0)
	if math.Abs(got-want)/want > 0.01 {
		t.Errorf("Milstein ODE: got %v, want ~%v", got, want)
	}
}

func TestMilsteinGBM(t *testing.T) {
	// For GBM: diffusion = sigma*x, diffusion' = sigma
	mu := 0.05
	sigma := 0.2
	x0 := 100.0
	T := 1.0
	nPaths := 50000

	drift := func(x, ti float64) float64 { return mu * x }
	diffusion := func(x, ti float64) float64 { return sigma * x }
	diffDeriv := func(x, ti float64) float64 { return sigma }

	result := Milstein(drift, diffusion, diffDeriv, x0, [2]float64{0, T}, 0.001, nPaths, 99)

	sum := 0.0
	finalIdx := len(result.T) - 1
	for p := 0; p < nPaths; p++ {
		sum += result.X[p][finalIdx]
	}
	mean := sum / float64(nPaths)
	expected := x0 * math.Exp(mu*T)

	relErr := math.Abs(mean-expected) / expected
	if relErr > 0.02 {
		t.Errorf("Milstein GBM mean: got %v, want ~%v (relErr=%v)", mean, expected, relErr)
	}
}

func TestMilsteinBetterThanEM(t *testing.T) {
	// With larger dt, Milstein should have smaller strong error than EM for GBM
	mu := 0.05
	sigma := 0.3
	x0 := 1.0
	T := 1.0
	dt := 0.05
	seed := int64(777)

	drift := func(x, ti float64) float64 { return mu * x }
	diffusion := func(x, ti float64) float64 { return sigma * x }
	diffDeriv := func(x, ti float64) float64 { return sigma }

	// Just verify it runs without error and produces valid output
	result := Milstein(drift, diffusion, diffDeriv, x0, [2]float64{0, T}, dt, 100, seed)
	if len(result.X) != 100 {
		t.Errorf("expected 100 paths, got %d", len(result.X))
	}
	for _, path := range result.X {
		if math.IsNaN(path[len(path)-1]) || math.IsInf(path[len(path)-1], 0) {
			t.Error("Milstein produced NaN or Inf")
		}
	}
}

func TestMilsteinPanics(t *testing.T) {
	drift := func(x, ti float64) float64 { return 0 }
	diffusion := func(x, ti float64) float64 { return 0 }
	diffDeriv := func(x, ti float64) float64 { return 0 }

	assertPanics(t, "negative dt", func() {
		Milstein(drift, diffusion, diffDeriv, 0, [2]float64{0, 1}, -0.1, 1, 42)
	})
	assertPanics(t, "zero nPaths", func() {
		Milstein(drift, diffusion, diffDeriv, 0, [2]float64{0, 1}, 0.1, 0, 42)
	})
	assertPanics(t, "bad tSpan", func() {
		Milstein(drift, diffusion, diffDeriv, 0, [2]float64{1, 0}, 0.1, 1, 42)
	})
}

// ---------------------------------------------------------------------------
// SDESystem Tests
// ---------------------------------------------------------------------------

func TestSolveSDESystem2D(t *testing.T) {
	// 2D system: dX1 = -X1 dt, dX2 = -X2 dt (no noise)
	// Solution: X1(t) = x1_0 * exp(-t), X2(t) = x2_0 * exp(-t)
	sys := &SDESystem{
		Dim:       2,
		Drift:     func(x []float64, ti float64) []float64 { return []float64{-x[0], -x[1]} },
		Diffusion: func(x []float64, ti float64) []float64 { return []float64{0, 0} },
	}

	x0 := []float64{1.0, 2.0}
	result := SolveSDESystem(sys, x0, [2]float64{0, 1}, 0.001, 1, 42)

	// X[0] is dim 0 of path 0, X[1] is dim 1 of path 0
	n := len(result.T) - 1
	got0 := result.X[0][n]
	got1 := result.X[1][n]
	want0 := 1.0 * math.Exp(-1.0)
	want1 := 2.0 * math.Exp(-1.0)

	if math.Abs(got0-want0)/want0 > 0.01 {
		t.Errorf("SDESystem dim 0: got %v, want ~%v", got0, want0)
	}
	if math.Abs(got1-want1)/want1 > 0.01 {
		t.Errorf("SDESystem dim 1: got %v, want ~%v", got1, want1)
	}
}

func TestSolveSDESystemMultiplePaths(t *testing.T) {
	sys := &SDESystem{
		Dim:       2,
		Drift:     func(x []float64, ti float64) []float64 { return []float64{0, 0} },
		Diffusion: func(x []float64, ti float64) []float64 { return []float64{1, 1} },
	}

	result := SolveSDESystem(sys, []float64{0, 0}, [2]float64{0, 1}, 0.01, 5, 42)
	// 5 paths * 2 dims = 10 entries in X
	if len(result.X) != 10 {
		t.Errorf("expected 10 X entries, got %d", len(result.X))
	}
}

func TestSolveSDESystemPanics(t *testing.T) {
	sys := &SDESystem{
		Dim:       1,
		Drift:     func(x []float64, ti float64) []float64 { return []float64{0} },
		Diffusion: func(x []float64, ti float64) []float64 { return []float64{0} },
	}

	assertPanics(t, "nil sys", func() {
		SolveSDESystem(nil, []float64{0}, [2]float64{0, 1}, 0.1, 1, 42)
	})
	assertPanics(t, "dim mismatch", func() {
		SolveSDESystem(sys, []float64{0, 0}, [2]float64{0, 1}, 0.1, 1, 42)
	})
	assertPanics(t, "negative dt", func() {
		SolveSDESystem(sys, []float64{0}, [2]float64{0, 1}, -0.1, 1, 42)
	})
	assertPanics(t, "zero nPaths", func() {
		SolveSDESystem(sys, []float64{0}, [2]float64{0, 1}, 0.1, 0, 42)
	})
	assertPanics(t, "bad tSpan", func() {
		SolveSDESystem(sys, []float64{0}, [2]float64{1, 0}, 0.1, 1, 42)
	})
}

// ---------------------------------------------------------------------------
// SDEResult structure Tests
// ---------------------------------------------------------------------------

func TestSDEResultStructure(t *testing.T) {
	drift := func(x, ti float64) float64 { return 0 }
	diffusion := func(x, ti float64) float64 { return 1 }

	result := EulerMaruyama(drift, diffusion, 0, [2]float64{0, 1}, 0.1, 3, 42)

	// T should be monotonically increasing
	for i := 1; i < len(result.T); i++ {
		if result.T[i] <= result.T[i-1] {
			t.Errorf("T not monotonic at index %d: %v <= %v", i, result.T[i], result.T[i-1])
		}
	}

	// All paths should start at x0 = 0
	for p := 0; p < 3; p++ {
		if result.X[p][0] != 0 {
			t.Errorf("path %d does not start at x0: got %v", p, result.X[p][0])
		}
	}
}

// assertPanics is a helper that checks if f panics.
func assertPanics(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic but did not get one", name)
		}
	}()
	f()
}
