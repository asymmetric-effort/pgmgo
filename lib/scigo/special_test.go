//go:build unit

package scigo

import (
	"math"
	"testing"
)

func approxEqual(a, b, tol float64) bool {
	if math.IsInf(a, 0) && math.IsInf(b, 0) {
		return (a > 0) == (b > 0)
	}
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	return math.Abs(a-b) <= tol
}

func TestGammaln(t *testing.T) {
	tests := []struct {
		x, want float64
	}{
		{1, 0},                       // ln(Gamma(1)) = ln(1) = 0
		{2, 0},                       // ln(Gamma(2)) = ln(1) = 0
		{0.5, math.Log(math.Pi) / 2}, // ln(Gamma(0.5)) = ln(sqrt(pi))
		{5, math.Log(24)},            // ln(Gamma(5)) = ln(4!) = ln(24)
		{10, math.Log(362880)},       // ln(Gamma(10)) = ln(9!)
	}
	for _, tc := range tests {
		got := Gammaln(tc.x)
		if !approxEqual(got, tc.want, 1e-10) {
			t.Errorf("Gammaln(%v) = %v, want %v", tc.x, got, tc.want)
		}
	}
}

func TestDigamma(t *testing.T) {
	// Known values: psi(1) = -gamma (Euler-Mascheroni constant)
	euler := 0.5772156649015329
	tests := []struct {
		x, want float64
	}{
		{1, -euler},
		{2, 1 - euler},             // psi(2) = psi(1) + 1 = 1 - gamma
		{0.5, -euler - math.Ln2*2}, // psi(1/2) = -gamma - 2*ln(2)
	}
	for _, tc := range tests {
		got := Digamma(tc.x)
		if !approxEqual(got, tc.want, 1e-8) {
			t.Errorf("Digamma(%v) = %v, want %v", tc.x, got, tc.want)
		}
	}

	// psi at non-positive integers should be NaN
	if !math.IsNaN(Digamma(0)) {
		t.Error("Digamma(0) should be NaN")
	}
	if !math.IsNaN(Digamma(-1)) {
		t.Error("Digamma(-1) should be NaN")
	}
}

func TestRegularizedIncompleteGamma(t *testing.T) {
	// P(1, x) = 1 - e^{-x}
	tests := []struct {
		a, x, want float64
	}{
		{1, 0, 0},
		{1, 1, 1 - math.Exp(-1)},
		{1, 2, 1 - math.Exp(-2)},
		{1, 10, 1 - math.Exp(-10)},
		// P(a, 0) = 0 for any a
		{5, 0, 0},
		// P(1, inf) = 1
		{1, math.Inf(1), 1},
	}
	for _, tc := range tests {
		got := RegularizedIncompleteGamma(tc.a, tc.x)
		if !approxEqual(got, tc.want, 1e-10) {
			t.Errorf("RegularizedIncompleteGamma(%v, %v) = %v, want %v", tc.a, tc.x, got, tc.want)
		}
	}

	// P(0.5, x) relates to erf: P(0.5, x) = erf(sqrt(x))
	for _, x := range []float64{0.1, 0.5, 1, 2, 5} {
		got := RegularizedIncompleteGamma(0.5, x)
		want := math.Erf(math.Sqrt(x))
		if !approxEqual(got, want, 1e-8) {
			t.Errorf("RegularizedIncompleteGamma(0.5, %v) = %v, want %v (erf relation)", x, got, want)
		}
	}

	// Invalid inputs
	if !math.IsNaN(RegularizedIncompleteGamma(-1, 1)) {
		t.Error("Expected NaN for negative a")
	}
	if !math.IsNaN(RegularizedIncompleteGamma(1, -1)) {
		t.Error("Expected NaN for negative x")
	}
}

func TestErf(t *testing.T) {
	tests := []struct {
		x, want float64
	}{
		{0, 0},
		{1, 0.8427007929497149},
		{-1, -0.8427007929497149},
		{2, 0.9953222650189527},
	}
	for _, tc := range tests {
		got := Erf(tc.x)
		if !approxEqual(got, tc.want, 1e-12) {
			t.Errorf("Erf(%v) = %v, want %v", tc.x, got, tc.want)
		}
	}
}

func TestErfinv(t *testing.T) {
	// erfinv(erf(x)) should equal x
	for _, x := range []float64{-2, -1, -0.5, 0, 0.1, 0.5, 1, 1.5, 2} {
		y := math.Erf(x)
		got := Erfinv(y)
		if !approxEqual(got, x, 1e-10) {
			t.Errorf("Erfinv(Erf(%v)) = %v, want %v", x, got, x)
		}
	}

	// Boundary values
	if !math.IsInf(Erfinv(1), 1) {
		t.Error("Erfinv(1) should be +Inf")
	}
	if !math.IsInf(Erfinv(-1), -1) {
		t.Error("Erfinv(-1) should be -Inf")
	}
	if Erfinv(0) != 0 {
		t.Error("Erfinv(0) should be 0")
	}
}

func TestBetaln(t *testing.T) {
	// B(1,1) = Gamma(1)*Gamma(1)/Gamma(2) = 1*1/1 = 1, ln(1) = 0.
	got := Betaln(1, 1)
	if !approxEqual(got, 0, 1e-12) {
		t.Errorf("Betaln(1,1) = %v, want 0", got)
	}
	// B(0.5, 0.5) = pi, ln(pi) ≈ 1.1447298858494.
	got = Betaln(0.5, 0.5)
	if !approxEqual(got, math.Log(math.Pi), 1e-10) {
		t.Errorf("Betaln(0.5,0.5) = %v, want %v", got, math.Log(math.Pi))
	}
	// B(2, 3) = 1/12, ln(1/12).
	got = Betaln(2, 3)
	if !approxEqual(got, math.Log(1.0/12.0), 1e-10) {
		t.Errorf("Betaln(2,3) = %v, want %v", got, math.Log(1.0/12.0))
	}
}

func TestComb(t *testing.T) {
	tests := []struct {
		n, k int
		want float64
	}{
		{5, 0, 1},
		{5, 5, 1},
		{5, 2, 10},
		{10, 3, 120},
		{20, 10, 184756},
		{0, 0, 1},
		{5, -1, 0},
		{5, 6, 0},
	}
	for _, tc := range tests {
		got := Comb(tc.n, tc.k)
		if !approxEqual(got, tc.want, 1e-6) {
			t.Errorf("Comb(%d,%d) = %v, want %v", tc.n, tc.k, got, tc.want)
		}
	}
}

func TestFactorial(t *testing.T) {
	tests := []struct {
		n    int
		want float64
	}{
		{0, 1},
		{1, 1},
		{5, 120},
		{10, 3628800},
		{20, 2432902008176640000},
	}
	for _, tc := range tests {
		got := Factorial(tc.n)
		if !approxEqual(got, tc.want, tc.want*1e-10) {
			t.Errorf("Factorial(%d) = %v, want %v", tc.n, got, tc.want)
		}
	}
	// Negative should be NaN.
	if !math.IsNaN(Factorial(-1)) {
		t.Error("Factorial(-1) should be NaN")
	}
}

func TestSoftmax(t *testing.T) {
	// Basic test.
	vals := []float64{1, 2, 3}
	result := Softmax(vals)
	if len(result) != 3 {
		t.Fatalf("Softmax: got %d elements, want 3", len(result))
	}
	// Sum should be 1.
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if !approxEqual(sum, 1, 1e-12) {
		t.Errorf("Softmax sum = %v, want 1", sum)
	}
	// Values should be increasing since input is increasing.
	if result[0] >= result[1] || result[1] >= result[2] {
		t.Errorf("Softmax not monotonically increasing: %v", result)
	}

	// Numerical stability with large values.
	large := []float64{1000, 1001, 1002}
	resultLarge := Softmax(large)
	sumLarge := 0.0
	for _, v := range resultLarge {
		sumLarge += v
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Softmax large: got NaN/Inf: %v", resultLarge)
		}
	}
	if !approxEqual(sumLarge, 1, 1e-12) {
		t.Errorf("Softmax large sum = %v, want 1", sumLarge)
	}

	// Equal values should give uniform distribution.
	equal := []float64{5, 5, 5, 5}
	resultEq := Softmax(equal)
	for _, v := range resultEq {
		if !approxEqual(v, 0.25, 1e-12) {
			t.Errorf("Softmax equal: got %v, want 0.25", v)
		}
	}
}

func TestSoftmaxPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Softmax should panic on empty input")
		}
	}()
	Softmax([]float64{})
}

func TestLogsumexp(t *testing.T) {
	// Basic case
	vals := []float64{1, 2, 3}
	got := Logsumexp(vals)
	want := math.Log(math.Exp(1) + math.Exp(2) + math.Exp(3))
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Logsumexp([1,2,3]) = %v, want %v", got, want)
	}

	// Large values (tests numerical stability)
	largeVals := []float64{1000, 1001, 1002}
	got = Logsumexp(largeVals)
	// Should be 1002 + log(exp(-2) + exp(-1) + 1)
	want = 1002 + math.Log(math.Exp(-2)+math.Exp(-1)+1)
	if !approxEqual(got, want, 1e-10) {
		t.Errorf("Logsumexp large values = %v, want %v", got, want)
	}

	// Single value
	got = Logsumexp([]float64{5.0})
	if !approxEqual(got, 5.0, 1e-12) {
		t.Errorf("Logsumexp([5]) = %v, want 5", got)
	}

	// Empty slice
	got = Logsumexp([]float64{})
	if !math.IsInf(got, -1) {
		t.Errorf("Logsumexp([]) = %v, want -Inf", got)
	}

	// All -Inf
	got = Logsumexp([]float64{math.Inf(-1), math.Inf(-1)})
	if !math.IsInf(got, -1) {
		t.Errorf("Logsumexp([-Inf, -Inf]) = %v, want -Inf", got)
	}
}
