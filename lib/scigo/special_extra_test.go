//go:build unit

package scigo

import (
	"math"
	"testing"
)

func approxEqualSE(a, b, tol float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	if math.IsInf(a, 1) && math.IsInf(b, 1) {
		return true
	}
	if math.IsInf(a, -1) && math.IsInf(b, -1) {
		return true
	}
	return math.Abs(a-b) < tol
}

// ---------------------------------------------------------------------------
// GammaFunc
// ---------------------------------------------------------------------------

func TestGammaFunc(t *testing.T) {
	// Gamma(5) = 4! = 24
	if !approxEqualSE(GammaFunc(5), 24, 1e-10) {
		t.Errorf("GammaFunc(5)=%v, want 24", GammaFunc(5))
	}
	// Gamma(0.5) = sqrt(pi)
	if !approxEqualSE(GammaFunc(0.5), math.Sqrt(math.Pi), 1e-10) {
		t.Errorf("GammaFunc(0.5)=%v, want sqrt(pi)", GammaFunc(0.5))
	}
	// Gamma(1) = 1
	if !approxEqualSE(GammaFunc(1), 1, 1e-10) {
		t.Errorf("GammaFunc(1)=%v, want 1", GammaFunc(1))
	}
}

// ---------------------------------------------------------------------------
// BetaFunc
// ---------------------------------------------------------------------------

func TestBetaFunc(t *testing.T) {
	// B(1,1) = 1
	if !approxEqualSE(BetaFunc(1, 1), 1, 1e-10) {
		t.Errorf("BetaFunc(1,1)=%v, want 1", BetaFunc(1, 1))
	}
	// B(2,3) = Gamma(2)*Gamma(3)/Gamma(5) = 1*2/24 = 1/12
	if !approxEqualSE(BetaFunc(2, 3), 1.0/12, 1e-10) {
		t.Errorf("BetaFunc(2,3)=%v, want 1/12", BetaFunc(2, 3))
	}
	// B(0.5, 0.5) = pi
	if !approxEqualSE(BetaFunc(0.5, 0.5), math.Pi, 1e-10) {
		t.Errorf("BetaFunc(0.5,0.5)=%v, want pi", BetaFunc(0.5, 0.5))
	}
}

// ---------------------------------------------------------------------------
// Psi (alias for Digamma)
// ---------------------------------------------------------------------------

func TestPsi(t *testing.T) {
	// psi(1) = -gamma (Euler-Mascheroni constant)
	euler := 0.5772156649015329
	if !approxEqualSE(Psi(1), -euler, 1e-6) {
		t.Errorf("Psi(1)=%v, want %v", Psi(1), -euler)
	}
}

// ---------------------------------------------------------------------------
// Polygamma
// ---------------------------------------------------------------------------

func TestPolygamma(t *testing.T) {
	// polygamma(0, x) = digamma(x)
	if !approxEqualSE(Polygamma(0, 5), Digamma(5), 1e-10) {
		t.Errorf("Polygamma(0,5)=%v, want Digamma(5)=%v", Polygamma(0, 5), Digamma(5))
	}

	// polygamma(1, 1) = pi^2/6 (trigamma at 1)
	expected := math.Pi * math.Pi / 6
	got := Polygamma(1, 1)
	if !approxEqualSE(got, expected, 1e-4) {
		t.Errorf("Polygamma(1,1)=%v, want %v", got, expected)
	}

	// polygamma(2, 1) = -2*zeta(3) = -2*1.2020569... = -2.4041138...
	got = Polygamma(2, 1)
	expected = -2 * 1.2020569031595942
	if !approxEqualSE(got, expected, 1e-3) {
		t.Errorf("Polygamma(2,1)=%v, want %v", got, expected)
	}

	// Negative n should return NaN
	if !math.IsNaN(Polygamma(-1, 5)) {
		t.Errorf("Polygamma(-1,5) should be NaN")
	}
}

// ---------------------------------------------------------------------------
// Zeta
// ---------------------------------------------------------------------------

func TestZeta(t *testing.T) {
	// zeta(2) = pi^2/6
	expected := math.Pi * math.Pi / 6
	if !approxEqualSE(Zeta(2), expected, 1e-6) {
		t.Errorf("Zeta(2)=%v, want %v", Zeta(2), expected)
	}

	// zeta(4) = pi^4/90
	expected = math.Pow(math.Pi, 4) / 90
	if !approxEqualSE(Zeta(4), expected, 1e-6) {
		t.Errorf("Zeta(4)=%v, want %v", Zeta(4), expected)
	}

	// zeta(1) should be +Inf (pole)
	if !math.IsInf(Zeta(1), 1) {
		t.Errorf("Zeta(1)=%v, want +Inf", Zeta(1))
	}

	// zeta(0) = -1/2
	if !approxEqualSE(Zeta(0), -0.5, 1e-10) {
		t.Errorf("Zeta(0)=%v, want -0.5", Zeta(0))
	}

	// zeta(-1) = -1/12
	if !approxEqualSE(Zeta(-1), -1.0/12, 1e-10) {
		t.Errorf("Zeta(-1)=%v, want %v", Zeta(-1), -1.0/12)
	}

	// zeta(-2) = 0
	if !approxEqualSE(Zeta(-2), 0, 1e-10) {
		t.Errorf("Zeta(-2)=%v, want 0", Zeta(-2))
	}

	// zeta(-3) = 1/120
	if !approxEqualSE(Zeta(-3), 1.0/120, 1e-10) {
		t.Errorf("Zeta(-3)=%v, want %v", Zeta(-3), 1.0/120)
	}

	// zeta(0.5) should be finite, approximately -1.4604
	z05 := Zeta(0.5)
	if math.IsNaN(z05) || math.IsInf(z05, 0) {
		t.Errorf("Zeta(0.5) should be finite, got %v", z05)
	}
	if !approxEqualSE(z05, -1.4603545088095868, 1e-3) {
		t.Errorf("Zeta(0.5)=%v, want ~-1.4604", z05)
	}
}

// ---------------------------------------------------------------------------
// I0 and I1
// ---------------------------------------------------------------------------

func TestI0(t *testing.T) {
	// I0(0) = 1
	if !approxEqualSE(I0(0), 1, 1e-6) {
		t.Errorf("I0(0)=%v, want 1", I0(0))
	}
	// I0 is always positive and monotonically increasing for x > 0
	v1 := I0(1)
	v2 := I0(2)
	if v1 <= 1 || v2 <= v1 {
		t.Errorf("I0 should be monotonically increasing: I0(1)=%v, I0(2)=%v", v1, v2)
	}
}

func TestI1(t *testing.T) {
	// I1(0) = 0
	if !approxEqualSE(I1(0), 0, 1e-6) {
		t.Errorf("I1(0)=%v, want 0", I1(0))
	}
	// I1 should be positive for positive x
	if I1(2) <= 0 {
		t.Errorf("I1(2)=%v, should be positive", I1(2))
	}
	// I1(-x) = -I1(x)
	if !approxEqualSE(I1(-2), -I1(2), 1e-6) {
		t.Errorf("I1(-2)=%v, want %v", I1(-2), -I1(2))
	}
}

// ---------------------------------------------------------------------------
// Logit and Expit
// ---------------------------------------------------------------------------

func TestLogit(t *testing.T) {
	// logit(0.5) = 0
	if !approxEqualSE(Logit(0.5), 0, 1e-10) {
		t.Errorf("Logit(0.5)=%v, want 0", Logit(0.5))
	}
	// logit(0) = -Inf
	if !math.IsInf(Logit(0), -1) {
		t.Errorf("Logit(0)=%v, want -Inf", Logit(0))
	}
	// logit(1) = +Inf
	if !math.IsInf(Logit(1), 1) {
		t.Errorf("Logit(1)=%v, want +Inf", Logit(1))
	}
	// logit outside [0,1] = NaN
	if !math.IsNaN(Logit(-0.1)) {
		t.Errorf("Logit(-0.1) should be NaN")
	}
	if !math.IsNaN(Logit(1.1)) {
		t.Errorf("Logit(1.1) should be NaN")
	}
}

func TestExpit(t *testing.T) {
	// expit(0) = 0.5
	if !approxEqualSE(Expit(0), 0.5, 1e-10) {
		t.Errorf("Expit(0)=%v, want 0.5", Expit(0))
	}
	// expit(large) -> 1
	if !approxEqualSE(Expit(100), 1.0, 1e-10) {
		t.Errorf("Expit(100)=%v, want 1", Expit(100))
	}
	// expit(very negative) -> 0
	if !approxEqualSE(Expit(-100), 0.0, 1e-10) {
		t.Errorf("Expit(-100)=%v, want 0", Expit(-100))
	}
	// expit(logit(p)) = p (roundtrip)
	for _, p := range []float64{0.1, 0.25, 0.5, 0.75, 0.9} {
		got := Expit(Logit(p))
		if !approxEqualSE(got, p, 1e-10) {
			t.Errorf("Expit(Logit(%v))=%v, want %v", p, got, p)
		}
	}
}
