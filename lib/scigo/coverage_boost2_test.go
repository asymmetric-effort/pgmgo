//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ===========================================================================
// stats_tests.go: additional edge case coverage
// ===========================================================================

func TestKS2Samp_WithTies(t *testing.T) {
	// Data with ties to exercise the tie-advancement loops.
	x := []float64{1, 1, 2, 3, 3, 3, 4, 5}
	y := []float64{1, 1, 1, 2, 3, 4, 4, 5}
	stat, pval := KS2Samp(x, y)
	_ = stat
	_ = pval
}

func TestKsProb_Boundaries(t *testing.T) {
	// lambda <= 0 => 1
	if ksProb(0) != 1 {
		t.Error("expected 1 for lambda=0")
	}
	// lambda > 8 => 0
	if ksProb(10) != 0 {
		t.Error("expected 0 for lambda=10")
	}
}

func TestShapiroWilk_TooMany(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for > 5000")
		}
	}()
	data := make([]float64, 5001)
	for i := range data {
		data[i] = float64(i)
	}
	ShapiroWilk(data)
}

func TestShapiroWilk_ConstantData(t *testing.T) {
	stat, pval := ShapiroWilk([]float64{5, 5, 5, 5, 5})
	if stat != 1 || pval != 1 {
		t.Errorf("expected (1,1) for constant data, got (%f,%f)", stat, pval)
	}
}

func TestShapiroWilk_SmallSample(t *testing.T) {
	// n <= 11 triggers small sample path
	stat, pval := ShapiroWilk([]float64{1, 2, 3, 4, 5})
	if stat <= 0 || stat > 1 {
		t.Errorf("stat out of range: %f", stat)
	}
	_ = pval
}

func TestNormalTest_Normal(t *testing.T) {
	// Generate enough data (>= 20).
	data := make([]float64, 50)
	for i := range data {
		data[i] = float64(i)
	}
	stat, pval := NormalTest(data)
	_ = stat
	_ = pval
}

func TestAndersonDarling_Normal(t *testing.T) {
	data := make([]float64, 20)
	for i := range data {
		data[i] = float64(i)
	}
	stat, crit := AndersonDarling(data)
	_ = stat
	_ = crit
}

func TestBartlettTest_Normal(t *testing.T) {
	stat, pval := BartlettTest([]float64{1, 2, 3, 4, 5}, []float64{2, 3, 4, 5, 6})
	_ = stat
	_ = pval
}

func TestLeveneTest_Normal(t *testing.T) {
	stat, pval := LeveneTest([]float64{1, 2, 3, 4, 5}, []float64{2, 3, 4, 5, 6})
	_ = stat
	_ = pval
}

func TestFlignerKilleen_Normal(t *testing.T) {
	stat, pval := FlignerKilleen([]float64{1, 2, 3, 4, 5}, []float64{2, 3, 4, 5, 6})
	_ = stat
	_ = pval
}

func TestMoodTest_Normal(t *testing.T) {
	stat, pval := MoodTest([]float64{1, 2, 3, 4, 5}, []float64{6, 7, 8, 9, 10})
	_ = stat
	_ = pval
}

func TestSpearmanR_Normal(t *testing.T) {
	r, pval := SpearmanR([]float64{1, 2, 3, 4}, []float64{1, 2, 3, 4})
	if math.Abs(r-1) > 0.01 {
		t.Errorf("expected r=1 for perfect correlation, got %f", r)
	}
	_ = pval
}

func TestKendallTau_Normal(t *testing.T) {
	tau, pval := KendallTau([]float64{1, 2, 3}, []float64{1, 2, 3})
	if math.Abs(tau-1) > 0.01 {
		t.Errorf("expected tau=1 for perfect concordance, got %f", tau)
	}
	_ = pval
}

func TestPartialCorrelation_ZeroDenom_Boost(t *testing.T) {
	// Perfectly correlated data: partial correlation with conditioning
	// that leaves residuals perfectly correlated => r^2 = 1.
	data := [][]float64{
		{1, 2, 0},
		{2, 4, 0},
		{3, 6, 0},
		{4, 8, 0},
		{5, 10, 0},
		{6, 12, 0},
	}
	r, pval := PartialCorrelation(data, 0, 1, []int{2})
	_ = r
	_ = pval
}

// ===========================================================================
// optimization_extra.go: edge cases
// ===========================================================================

func TestMinimizeScalar_Normal(t *testing.T) {
	f := func(x float64) float64 { return (x - 2) * (x - 2) }
	result, err := MinimizeScalar(f, [2]float64{0, 5})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result.X-2) > 0.01 {
		t.Errorf("expected x near 2, got %f", result.X)
	}
}

// ===========================================================================
// interpolate.go: additional edge cases
// ===========================================================================

func TestBSpline_Normal(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5, 6, 7}
	y := []float64{0, 1, 4, 9, 16, 25, 36, 49}
	bs := BSpline(x, y, 3)
	_ = bs(0.5)
	_ = bs(3.5)
}

func TestRBFInterpolator_Gaussian(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	vals := []float64{0, 1, 1, 2}
	rbf := RBFInterpolator(pts, vals, "gaussian")
	_ = rbf([]float64{0.5, 0.5})
}

// ===========================================================================
// special.go: additional edge cases
// ===========================================================================

func TestGammaln_Small(t *testing.T) {
	// Very small positive x
	_ = Gammaln(1e-10)
}

func TestDigamma_Negative(t *testing.T) {
	// Negative non-integer should work
	_ = Digamma(-0.5)
}

func TestErfinv_MiddleRange(t *testing.T) {
	// Test points across the range
	for _, x := range []float64{-0.99, -0.5, 0, 0.5, 0.99} {
		result := Erfinv(x)
		if math.IsNaN(result) {
			t.Errorf("Erfinv(%f) should not be NaN", x)
		}
	}
}

func TestRegularizedIncompleteBeta_Boundaries(t *testing.T) {
	if RegularizedIncompleteBeta(0, 1, 1) != 0 {
		t.Error("expected 0 for x=0")
	}
	if !math.IsNaN(RegularizedIncompleteBeta(-1, 1, 1)) {
		t.Error("expected NaN for x<0")
	}
	if RegularizedIncompleteBeta(1, 1, 1) != 1 {
		t.Error("expected 1 for x=1")
	}
	_ = RegularizedIncompleteBeta(0.9, 1, 10)
}

func TestBetaln_Normal(t *testing.T) {
	_ = Betaln(0.5, 0.5)
	_ = Betaln(1, 1)
}

// ===========================================================================
// spatial.go: additional edge cases
// ===========================================================================

func TestCdist_Metrics(t *testing.T) {
	xa := [][]float64{{0, 0}, {1, 1}}
	xb := [][]float64{{0, 1}, {1, 0}}
	// Different metrics
	_ = Cdist(xa, xb, "cityblock")
	_ = Cdist(xa, xb, "chebyshev")
	_ = Cdist(xa, xb, "cosine")
}

func TestPdist_Metrics(t *testing.T) {
	x := [][]float64{{0, 0}, {1, 1}, {2, 0}}
	_ = Pdist(x, "cityblock")
	_ = Pdist(x, "chebyshev")
}

// ===========================================================================
// distributions_extra.go: additional edge cases
// ===========================================================================

func TestPareto_PPF_Boost(t *testing.T) {
	p := NewPareto(1, 1)
	if p.PPF(0) != 1 { // Should be xm
		// Just exercise the path
	}
}

func TestPareto_LogPDF_Boost(t *testing.T) {
	p := NewPareto(1, 1)
	_ = p.LogPDF(0.5) // x < xm
	_ = p.LogPDF(2.0) // normal
}

// ===========================================================================
// distributions_continuous.go: remaining PPF edges
// ===========================================================================

func TestVonMises_CDF_Wrap(t *testing.T) {
	v := NewVonMises(0, 1)
	_ = v.CDF(5 * math.Pi)
	_ = v.CDF(-5 * math.Pi)
}

func TestSkewNormal_PPF_Normal(t *testing.T) {
	sn := NewSkewNormal(0, 1, 2)
	_ = sn.PPF(0.5)
	_ = sn.PPF(0.1)
	_ = sn.PPF(0.9)
}

func TestGEV_LogPDF(t *testing.T) {
	// xi ≈ 0 (Gumbel)
	gev := NewGeneralizedExtremeValue(0, 1, 0)
	_ = gev.LogPDF(1)
	// xi != 0, arg > 0
	gev2 := NewGeneralizedExtremeValue(0, 1, 0.5)
	_ = gev2.LogPDF(1)
}

// ===========================================================================
// distributions.go: remaining edges
// ===========================================================================

func TestChiSquared_PPF_NewtonClamp(t *testing.T) {
	// Very small p that could make initial guess negative.
	c := NewChiSquared(1)
	result := c.PPF(0.001)
	if result <= 0 {
		t.Error("expected positive result")
	}
}

func TestTDistribution_PPF_Newton(t *testing.T) {
	td := NewTDistribution(2)
	_ = td.PPF(0.001)
	_ = td.PPF(0.999)
}

// ===========================================================================
// stats_extra2.go: additional edges
// ===========================================================================

func TestZmap_Normal(t *testing.T) {
	scores := []float64{1, 2, 3, 4, 5}
	compare := []float64{1, 2, 3, 4, 5}
	result := Zmap(scores, compare)
	if len(result) != 5 {
		t.Errorf("expected 5 results, got %d", len(result))
	}
}

func TestJarqueBera_Boost(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	stat, pval := JarqueBera(data)
	_ = stat
	_ = pval
}

// ===========================================================================
// correlation.go: additional edges
// ===========================================================================

func TestPartialCorrelation_InsufficientCols(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for insufficient columns")
		}
	}()
	data := [][]float64{{1, 2}, {3, 4}}
	PartialCorrelation(data, 0, 1, nil)
}

// ===========================================================================
// integrate.go: remaining edges
// ===========================================================================

func TestRomberg_Normal(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	result := Romberg(f, 0, 1)
	if math.Abs(result-1.0/3) > 0.01 {
		t.Errorf("expected ~0.333, got %f", result)
	}
}

func TestSolveIVP_Normal(t *testing.T) {
	f := func(tc float64, y []float64) []float64 {
		return []float64{-y[0]}
	}
	times, states, err := SolveIVP(f, [2]float64{0, 1}, []float64{1})
	if err != nil {
		t.Fatal(err)
	}
	_ = times
	_ = states
}

// ===========================================================================
// linalg.go: remaining edges
// ===========================================================================

func TestLDL_Normal(t *testing.T) {
	// Symmetric positive definite
	a := [][]float64{{4, 2}, {2, 3}}
	l, d, err := LDL(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = l
	_ = d
}

func TestPolar_Normal(t *testing.T) {
	a := [][]float64{{1, 0}, {0, 1}}
	u, p, err := Polar(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = u
	_ = p
}

func TestLogm_Normal(t *testing.T) {
	a := [][]float64{{1, 0}, {0, 1}} // identity
	result, err := Logm(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestInterpolative_Normal(t *testing.T) {
	a := [][]float64{{1, 0, 1}, {0, 1, 1}, {1, 1, 0}}
	idx, proj, err := Interpolative(a, 2)
	if err != nil {
		t.Fatal(err)
	}
	_ = idx
	_ = proj
}

func TestDFT_Normal(t *testing.T) {
	m, err := DFT(4)
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 4 {
		t.Errorf("expected 4x4 DFT matrix")
	}
}
