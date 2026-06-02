//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Minimize (Nelder-Mead) Tests
// ---------------------------------------------------------------------------

func TestMinimizeNelderMead_Quadratic1D(t *testing.T) {
	// Minimize x^2, minimum at x=0
	f := func(x []float64) float64 { return x[0] * x[0] }
	res, err := Minimize(f, []float64{5.0}, "nelder-mead")
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X[0], 0, 1e-6) {
		t.Errorf("minimum at x=%v, want 0", res.X[0])
	}
	if !approxEqual(res.Fun, 0, 1e-10) {
		t.Errorf("minimum value=%v, want 0", res.Fun)
	}
}

func TestMinimizeNelderMead_Quadratic2D(t *testing.T) {
	// Minimize (x-1)^2 + (y-2)^2, minimum at (1, 2)
	f := func(x []float64) float64 {
		return (x[0]-1)*(x[0]-1) + (x[1]-2)*(x[1]-2)
	}
	res, err := Minimize(f, []float64{10.0, -5.0}, "nelder-mead")
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X[0], 1, 1e-5) {
		t.Errorf("x=%v, want 1", res.X[0])
	}
	if !approxEqual(res.X[1], 2, 1e-5) {
		t.Errorf("y=%v, want 2", res.X[1])
	}
	if !approxEqual(res.Fun, 0, 1e-8) {
		t.Errorf("fun=%v, want 0", res.Fun)
	}
}

func TestMinimizeNelderMead_Rosenbrock(t *testing.T) {
	// Rosenbrock function, minimum at (1, 1)
	f := func(x []float64) float64 {
		return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1-x[0])*(1-x[0])
	}
	res, err := Minimize(f, []float64{-1.0, 1.0}, "nelder-mead")
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.X[0], 1, 1e-3) {
		t.Errorf("x=%v, want 1", res.X[0])
	}
	if !approxEqual(res.X[1], 1, 1e-3) {
		t.Errorf("y=%v, want 1", res.X[1])
	}
}

// ---------------------------------------------------------------------------
// Minimize (Gradient Descent) Tests
// ---------------------------------------------------------------------------

func TestMinimizeGradientDescent_Quadratic1D(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	res, err := Minimize(f, []float64{5.0}, "gradient-descent")
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X[0], 0, 1e-5) {
		t.Errorf("minimum at x=%v, want 0", res.X[0])
	}
}

func TestMinimizeGradientDescent_Quadratic2D(t *testing.T) {
	f := func(x []float64) float64 {
		return (x[0]-3)*(x[0]-3) + (x[1]+1)*(x[1]+1)
	}
	res, err := Minimize(f, []float64{0, 0}, "gradient-descent")
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X[0], 3, 1e-4) {
		t.Errorf("x=%v, want 3", res.X[0])
	}
	if !approxEqual(res.X[1], -1, 1e-4) {
		t.Errorf("y=%v, want -1", res.X[1])
	}
}

func TestMinimizeUnknownMethod(t *testing.T) {
	f := func(x []float64) float64 { return 0 }
	_, err := Minimize(f, []float64{0}, "unknown")
	if err == nil {
		t.Error("expected error for unknown method")
	}
}

func TestMinimizeEmptyX0(t *testing.T) {
	f := func(x []float64) float64 { return 0 }
	_, err := Minimize(f, []float64{}, "nelder-mead")
	if err == nil {
		t.Error("expected error for empty x0")
	}
	_, err = Minimize(f, []float64{}, "gradient-descent")
	if err == nil {
		t.Error("expected error for empty x0")
	}
}

// ---------------------------------------------------------------------------
// MinimizeScalar Tests
// ---------------------------------------------------------------------------

func TestMinimizeScalar_Quadratic(t *testing.T) {
	// Minimize x^2 on [-10, 10], minimum at x=0
	f := func(x float64) float64 { return x * x }
	res, err := MinimizeScalar(f, [2]float64{-10, 10})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X, 0, 1e-8) {
		t.Errorf("minimum at x=%v, want 0", res.X)
	}
	if !approxEqual(res.Fun, 0, 1e-14) {
		t.Errorf("minimum value=%v, want 0", res.Fun)
	}
}

func TestMinimizeScalar_Shifted(t *testing.T) {
	// Minimize (x-3)^2 on [0, 10]
	f := func(x float64) float64 { return (x - 3) * (x - 3) }
	res, err := MinimizeScalar(f, [2]float64{0, 10})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X, 3, 1e-8) {
		t.Errorf("minimum at x=%v, want 3", res.X)
	}
}

func TestMinimizeScalar_Cos(t *testing.T) {
	// Minimize cos(x) on [2, 5], minimum at x=pi
	f := func(x float64) float64 { return math.Cos(x) }
	res, err := MinimizeScalar(f, [2]float64{2, 5})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("expected Success=true")
	}
	if !approxEqual(res.X, math.Pi, 1e-8) {
		t.Errorf("minimum at x=%v, want pi=%v", res.X, math.Pi)
	}
	if !approxEqual(res.Fun, -1, 1e-10) {
		t.Errorf("minimum value=%v, want -1", res.Fun)
	}
}

func TestMinimizeScalar_InvalidBounds(t *testing.T) {
	f := func(x float64) float64 { return x }
	_, err := MinimizeScalar(f, [2]float64{5, 3})
	if err == nil {
		t.Error("expected error for invalid bounds")
	}
}

// ---------------------------------------------------------------------------
// RootScalar Tests
// ---------------------------------------------------------------------------

func TestRootScalar_Linear(t *testing.T) {
	// Root of f(x) = x - 3 on [0, 10]
	f := func(x float64) float64 { return x - 3 }
	root, err := RootScalar(f, [2]float64{0, 10})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(root, 3, 1e-12) {
		t.Errorf("root=%v, want 3", root)
	}
}

func TestRootScalar_Quadratic(t *testing.T) {
	// Root of f(x) = x^2 - 4 on [0, 10] -> root at 2
	f := func(x float64) float64 { return x*x - 4 }
	root, err := RootScalar(f, [2]float64{0, 10})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(root, 2, 1e-12) {
		t.Errorf("root=%v, want 2", root)
	}
}

func TestRootScalar_Trig(t *testing.T) {
	// Root of sin(x) near pi, bracket [3, 4]
	f := func(x float64) float64 { return math.Sin(x) }
	root, err := RootScalar(f, [2]float64{3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(root, math.Pi, 1e-12) {
		t.Errorf("root=%v, want pi=%v", root, math.Pi)
	}
}

func TestRootScalar_Exp(t *testing.T) {
	// Root of exp(x) - 2 on [-1, 2] -> root at ln(2)
	f := func(x float64) float64 { return math.Exp(x) - 2 }
	root, err := RootScalar(f, [2]float64{-1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(root, math.Ln2, 1e-12) {
		t.Errorf("root=%v, want ln(2)=%v", root, math.Ln2)
	}
}

func TestRootScalar_NotBracketed(t *testing.T) {
	f := func(x float64) float64 { return x*x + 1 }
	_, err := RootScalar(f, [2]float64{-5, 5})
	if err == nil {
		t.Error("expected error when root is not bracketed")
	}
}

func TestRootScalar_ExactBoundary(t *testing.T) {
	// Root exactly at bracket endpoint
	f := func(x float64) float64 { return x - 5 }
	root, err := RootScalar(f, [2]float64{5, 10})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(root, 5, 1e-12) {
		t.Errorf("root=%v, want 5", root)
	}
}

// ---------------------------------------------------------------------------
// Integration: OptResult and ScalarResult struct tests
// ---------------------------------------------------------------------------

func TestOptResultFields(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	res, err := Minimize(f, []float64{5}, "nelder-mead")
	if err != nil {
		t.Fatal(err)
	}
	if len(res.X) != 1 {
		t.Errorf("len(X)=%d, want 1", len(res.X))
	}
	if res.Nit <= 0 {
		t.Errorf("Nit=%d, want > 0", res.Nit)
	}
}

func TestScalarResultFields(t *testing.T) {
	f := func(x float64) float64 { return (x - 1) * (x - 1) }
	res, err := MinimizeScalar(f, [2]float64{-5, 5})
	if err != nil {
		t.Fatal(err)
	}
	if res.Nit <= 0 {
		t.Errorf("Nit=%d, want > 0", res.Nit)
	}
	if !approxEqual(res.Fun, 0, 1e-14) {
		t.Errorf("Fun=%v, want 0", res.Fun)
	}
}
