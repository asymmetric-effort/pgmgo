//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// BlackScholesCall Tests
// ---------------------------------------------------------------------------

func TestBlackScholesCallKnownValue(t *testing.T) {
	// S=100, K=100, T=1, r=0.05, sigma=0.2
	// Known value approximately 10.4506
	got := BlackScholesCall(100, 100, 1, 0.05, 0.2)
	want := 10.4506
	if math.Abs(got-want) > 0.01 {
		t.Errorf("BlackScholesCall = %v, want ~%v", got, want)
	}
}

func TestBlackScholesCallDeepITM(t *testing.T) {
	// Deep in the money call
	got := BlackScholesCall(200, 100, 1, 0.05, 0.2)
	intrinsic := 200 - 100*math.Exp(-0.05)
	if got < intrinsic {
		t.Errorf("Deep ITM call %v < discounted intrinsic %v", got, intrinsic)
	}
}

func TestBlackScholesCallDeepOTM(t *testing.T) {
	got := BlackScholesCall(50, 100, 1, 0.05, 0.2)
	if got < 0 || got > 1 {
		t.Errorf("Deep OTM call should be near zero, got %v", got)
	}
}

func TestBlackScholesCallAtExpiry(t *testing.T) {
	if got := BlackScholesCall(110, 100, 0, 0.05, 0.2); math.Abs(got-10) > 1e-10 {
		t.Errorf("Call at expiry ITM: got %v, want 10", got)
	}
	if got := BlackScholesCall(90, 100, 0, 0.05, 0.2); got != 0 {
		t.Errorf("Call at expiry OTM: got %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// BlackScholesPut Tests
// ---------------------------------------------------------------------------

func TestBlackScholesPutKnownValue(t *testing.T) {
	got := BlackScholesPut(100, 100, 1, 0.05, 0.2)
	want := 5.5735
	if math.Abs(got-want) > 0.01 {
		t.Errorf("BlackScholesPut = %v, want ~%v", got, want)
	}
}

func TestBlackScholesPutAtExpiry(t *testing.T) {
	if got := BlackScholesPut(90, 100, 0, 0.05, 0.2); math.Abs(got-10) > 1e-10 {
		t.Errorf("Put at expiry ITM: got %v, want 10", got)
	}
	if got := BlackScholesPut(110, 100, 0, 0.05, 0.2); got != 0 {
		t.Errorf("Put at expiry OTM: got %v, want 0", got)
	}
}

// ---------------------------------------------------------------------------
// Put-Call Parity Tests
// ---------------------------------------------------------------------------

func TestPutCallParity(t *testing.T) {
	// C - P = S - K*exp(-rT)
	cases := []struct {
		S, K, T, r, sigma float64
	}{
		{100, 100, 1, 0.05, 0.2},
		{100, 110, 0.5, 0.03, 0.3},
		{50, 55, 2, 0.08, 0.25},
		{200, 180, 0.25, 0.01, 0.15},
	}

	for _, c := range cases {
		call := BlackScholesCall(c.S, c.K, c.T, c.r, c.sigma)
		put := BlackScholesPut(c.S, c.K, c.T, c.r, c.sigma)
		lhs := call - put
		rhs := c.S - c.K*math.Exp(-c.r*c.T)
		if math.Abs(lhs-rhs) > 1e-8 {
			t.Errorf("Put-call parity violated for S=%v K=%v T=%v r=%v sigma=%v: C-P=%v, S-Ke^{-rT}=%v",
				c.S, c.K, c.T, c.r, c.sigma, lhs, rhs)
		}
	}
}

// ---------------------------------------------------------------------------
// Greeks Tests
// ---------------------------------------------------------------------------

func TestBlackScholesGreeksCallDelta(t *testing.T) {
	g := BlackScholesGreeks(100, 100, 1, 0.05, 0.2)
	// ATM call delta should be around 0.5-0.6
	if g.Delta < 0.4 || g.Delta > 0.8 {
		t.Errorf("ATM call delta = %v, expected 0.4-0.8", g.Delta)
	}
}

func TestBlackScholesGreeksVega(t *testing.T) {
	g := BlackScholesGreeks(100, 100, 1, 0.05, 0.2)
	// Vega should be positive
	if g.Vega <= 0 {
		t.Errorf("Vega should be positive, got %v", g.Vega)
	}
}

func TestBlackScholesGreeksGamma(t *testing.T) {
	g := BlackScholesGreeks(100, 100, 1, 0.05, 0.2)
	if g.Gamma <= 0 {
		t.Errorf("Gamma should be positive, got %v", g.Gamma)
	}
}

func TestBlackScholesGreeksRho(t *testing.T) {
	g := BlackScholesGreeks(100, 100, 1, 0.05, 0.2)
	if g.Rho <= 0 {
		t.Errorf("Rho for call should be positive, got %v", g.Rho)
	}
}

func TestBlackScholesGreeksNumerical(t *testing.T) {
	// Verify Delta numerically
	S := 100.0
	K := 100.0
	T := 1.0
	r := 0.05
	sigma := 0.2
	eps := 0.01

	g := BlackScholesGreeks(S, K, T, r, sigma)

	// Numerical delta
	cUp := BlackScholesCall(S+eps, K, T, r, sigma)
	cDown := BlackScholesCall(S-eps, K, T, r, sigma)
	numDelta := (cUp - cDown) / (2 * eps)
	if math.Abs(g.Delta-numDelta) > 1e-4 {
		t.Errorf("Delta analytical=%v, numerical=%v", g.Delta, numDelta)
	}

	// Numerical gamma
	c0 := BlackScholesCall(S, K, T, r, sigma)
	numGamma := (cUp - 2*c0 + cDown) / (eps * eps)
	if math.Abs(g.Gamma-numGamma) > 1e-3 {
		t.Errorf("Gamma analytical=%v, numerical=%v", g.Gamma, numGamma)
	}

	// Numerical vega
	sigEps := 0.001
	cSigUp := BlackScholesCall(S, K, T, r, sigma+sigEps)
	cSigDown := BlackScholesCall(S, K, T, r, sigma-sigEps)
	numVega := (cSigUp - cSigDown) / (2 * sigEps)
	if math.Abs(g.Vega-numVega) > 0.01 {
		t.Errorf("Vega analytical=%v, numerical=%v", g.Vega, numVega)
	}

	// Numerical rho
	rEps := 0.0001
	cRUp := BlackScholesCall(S, K, T, r+rEps, sigma)
	cRDown := BlackScholesCall(S, K, T, r-rEps, sigma)
	numRho := (cRUp - cRDown) / (2 * rEps)
	if math.Abs(g.Rho-numRho) > 0.01 {
		t.Errorf("Rho analytical=%v, numerical=%v", g.Rho, numRho)
	}
}

func TestBlackScholesGreeksAtExpiry(t *testing.T) {
	g := BlackScholesGreeks(110, 100, 0, 0.05, 0.2)
	if g.Delta != 1.0 {
		t.Errorf("ITM call delta at expiry = %v, want 1.0", g.Delta)
	}
	g2 := BlackScholesGreeks(90, 100, 0, 0.05, 0.2)
	if g2.Delta != 0.0 {
		t.Errorf("OTM call delta at expiry = %v, want 0.0", g2.Delta)
	}
}

// ---------------------------------------------------------------------------
// ImpliedVolatility Tests
// ---------------------------------------------------------------------------

func TestImpliedVolatilityCall(t *testing.T) {
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	price := BlackScholesCall(S, K, T, r, sigma)
	iv, err := ImpliedVolatility(price, S, K, T, r, "call")
	if err != nil {
		t.Fatalf("ImpliedVolatility: %v", err)
	}
	if math.Abs(iv-sigma) > 1e-6 {
		t.Errorf("IV = %v, want %v", iv, sigma)
	}
}

func TestImpliedVolatilityPut(t *testing.T) {
	S, K, T, r, sigma := 100.0, 110.0, 0.5, 0.03, 0.3
	price := BlackScholesPut(S, K, T, r, sigma)
	iv, err := ImpliedVolatility(price, S, K, T, r, "put")
	if err != nil {
		t.Fatalf("ImpliedVolatility: %v", err)
	}
	if math.Abs(iv-sigma) > 1e-6 {
		t.Errorf("IV = %v, want %v", iv, sigma)
	}
}

func TestImpliedVolatilityErrors(t *testing.T) {
	_, err := ImpliedVolatility(-1, 100, 100, 1, 0.05, "call")
	if err == nil {
		t.Error("expected error for negative price")
	}
	_, err = ImpliedVolatility(10, 100, 100, 0, 0.05, "call")
	if err == nil {
		t.Error("expected error for zero T")
	}
	_, err = ImpliedVolatility(10, 100, 100, 1, 0.05, "straddle")
	if err == nil {
		t.Error("expected error for invalid optionType")
	}
}

func TestImpliedVolatilityNoRoot(t *testing.T) {
	// Price too high for any vol in range - impossible price
	_, err := ImpliedVolatility(9999, 100, 100, 1, 0.05, "call")
	if err == nil {
		t.Error("expected error for price that cannot be matched")
	}
}

func TestImpliedVolatilityHighVol(t *testing.T) {
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 1.5
	price := BlackScholesCall(S, K, T, r, sigma)
	iv, err := ImpliedVolatility(price, S, K, T, r, "call")
	if err != nil {
		t.Fatalf("ImpliedVolatility high vol: %v", err)
	}
	if math.Abs(iv-sigma) > 1e-4 {
		t.Errorf("IV = %v, want %v", iv, sigma)
	}
}

// ---------------------------------------------------------------------------
// BinomialTree Tests
// ---------------------------------------------------------------------------

func TestBinomialTreeEuropeanCall(t *testing.T) {
	// Should converge to BS price with large n
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	bsPrice := BlackScholesCall(S, K, T, r, sigma)
	binPrice := BinomialTree(S, K, T, r, sigma, 500, "call", false)
	if math.Abs(binPrice-bsPrice) > 0.1 {
		t.Errorf("BinomialTree call = %v, BS = %v", binPrice, bsPrice)
	}
}

func TestBinomialTreeEuropeanPut(t *testing.T) {
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	bsPrice := BlackScholesPut(S, K, T, r, sigma)
	binPrice := BinomialTree(S, K, T, r, sigma, 500, "put", false)
	if math.Abs(binPrice-bsPrice) > 0.1 {
		t.Errorf("BinomialTree put = %v, BS = %v", binPrice, bsPrice)
	}
}

func TestBinomialTreeAmericanPut(t *testing.T) {
	// American put should be >= European put
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	eurPut := BinomialTree(S, K, T, r, sigma, 200, "put", false)
	amPut := BinomialTree(S, K, T, r, sigma, 200, "put", true)
	if amPut < eurPut-1e-10 {
		t.Errorf("American put %v < European put %v", amPut, eurPut)
	}
}

func TestBinomialTreeAmericanCall(t *testing.T) {
	// American call on non-dividend stock = European call
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	eurCall := BinomialTree(S, K, T, r, sigma, 200, "call", false)
	amCall := BinomialTree(S, K, T, r, sigma, 200, "call", true)
	if math.Abs(amCall-eurCall) > 0.1 {
		t.Errorf("American call %v != European call %v (no dividends)", amCall, eurCall)
	}
}

func TestBinomialTreePanic(t *testing.T) {
	assertPanics(t, "zero n", func() {
		BinomialTree(100, 100, 1, 0.05, 0.2, 0, "call", false)
	})
}

// ---------------------------------------------------------------------------
// MonteCarloPricing Tests
// ---------------------------------------------------------------------------

func TestMonteCarloCall(t *testing.T) {
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	bsPrice := BlackScholesCall(S, K, T, r, sigma)
	mcPrice := MonteCarloPricing(S, K, T, r, sigma, 500000, 42, func(sT float64) float64 {
		return math.Max(sT-K, 0)
	})
	if math.Abs(mcPrice-bsPrice) > 0.3 {
		t.Errorf("MonteCarlo call = %v, BS = %v", mcPrice, bsPrice)
	}
}

func TestMonteCarloPut(t *testing.T) {
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	bsPrice := BlackScholesPut(S, K, T, r, sigma)
	mcPrice := MonteCarloPricing(S, K, T, r, sigma, 500000, 42, func(sT float64) float64 {
		return math.Max(K-sT, 0)
	})
	if math.Abs(mcPrice-bsPrice) > 0.3 {
		t.Errorf("MonteCarlo put = %v, BS = %v", mcPrice, bsPrice)
	}
}

func TestMonteCarloPricingPanic(t *testing.T) {
	assertPanics(t, "zero nPaths", func() {
		MonteCarloPricing(100, 100, 1, 0.05, 0.2, 0, 42, func(sT float64) float64 { return 0 })
	})
}

func TestMonteCarloCustomPayoff(t *testing.T) {
	// Digital call: pays 1 if S > K at expiry
	S, K, T, r, sigma := 100.0, 100.0, 1.0, 0.05, 0.2
	mcPrice := MonteCarloPricing(S, K, T, r, sigma, 200000, 42, func(sT float64) float64 {
		if sT > K {
			return 1.0
		}
		return 0.0
	})
	// Should be positive and less than the discount factor
	if mcPrice <= 0 || mcPrice > 1 {
		t.Errorf("Digital call price = %v, expected in (0, 1)", mcPrice)
	}
}

// ---------------------------------------------------------------------------
// Helper function tests (internal)
// ---------------------------------------------------------------------------

func TestCdfNormalSymmetry(t *testing.T) {
	if math.Abs(cdfNormal(0)-0.5) > 1e-10 {
		t.Errorf("cdfNormal(0) = %v, want 0.5", cdfNormal(0))
	}
	if math.Abs(cdfNormal(1)+cdfNormal(-1)-1) > 1e-10 {
		t.Error("cdfNormal symmetry violated")
	}
}

func TestPdfNormalSymmetry(t *testing.T) {
	if math.Abs(pdfNormal(1)-pdfNormal(-1)) > 1e-10 {
		t.Error("pdfNormal not symmetric")
	}
	// Peak at 0
	if pdfNormal(0) < pdfNormal(1) {
		t.Error("pdfNormal should peak at 0")
	}
}
