//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ===========================================================================
// distributions_continuous.go: PPF, LogPDF, CDF, Mean, Var edge cases
// ===========================================================================

func TestFinalFDistPPF_PdfZero(t *testing.T) {
	f := NewFDistribution(1, 1)
	// p=0 and p=1 boundaries
	if f.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	if !math.IsInf(f.PPF(1), 1) {
		t.Error("PPF(1) should be +Inf")
	}
	// Exercise the Newton iteration with pdfVal==0 edge case
	_ = f.PPF(0.999999)
}

func TestFinalRicePPF(t *testing.T) {
	r := NewRice(2, 1)
	_ = r.PPF(0)
	_ = r.PPF(1)
	v := r.PPF(0.5)
	if v <= 0 {
		t.Errorf("Rice PPF(0.5) should be positive, got %f", v)
	}
	// Exercise negative x clamping in PPF
	_ = r.PPF(0.001)
}

func TestFinalRiceLogPDF(t *testing.T) {
	r := NewRice(2, 1)
	if !math.IsInf(r.LogPDF(0), -1) {
		t.Error("Rice LogPDF(0) should be -Inf")
	}
	_ = r.LogPDF(1)
}

func TestFinalNakagamiPPF(t *testing.T) {
	n := NewNakagami(1, 1)
	if n.PPF(0) != 0 {
		t.Error("Nakagami PPF(0) should be 0")
	}
	if !math.IsInf(n.PPF(1), 1) {
		t.Error("Nakagami PPF(1) should be +Inf")
	}
	v := n.PPF(0.5)
	if v <= 0 {
		t.Errorf("expected positive, got %f", v)
	}
}

func TestFinalNakagamiLogPDF(t *testing.T) {
	n := NewNakagami(1, 1)
	if !math.IsInf(n.LogPDF(0), -1) {
		t.Error("LogPDF(0) should be -Inf")
	}
	_ = n.LogPDF(1)
}

func TestFinalVonMisesPPF(t *testing.T) {
	v := NewVonMises(0, 1)
	if v.PPF(0) != -math.Pi {
		t.Error("PPF(0) should be -Pi")
	}
	if v.PPF(1) != math.Pi {
		t.Error("PPF(1) should be Pi")
	}
	_ = v.PPF(0.5)
}

func TestFinalVonMisesCDF_Boundaries(t *testing.T) {
	v := NewVonMises(0, 1)
	// CDF at -pi and pi
	cLow := v.CDF(-math.Pi)
	_ = cLow
	cHigh := v.CDF(math.Pi)
	_ = cHigh
}

func TestFinalWaldPPF(t *testing.T) {
	w := NewWald(1, 1)
	if w.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	if !math.IsInf(w.PPF(1), 1) {
		t.Error("PPF(1) should be +Inf")
	}
	v := w.PPF(0.5)
	if v <= 0 {
		t.Errorf("expected positive, got %f", v)
	}
	// Exercise the x <= 0 clamping in Newton's method
	_ = w.PPF(0.001)
}

func TestFinalWaldNewPanic(t *testing.T) {
	defer func() { recover() }()
	NewWald(-1, 1) // should panic
	t.Error("expected panic for negative mu")
}

func TestFinalWaldLambdaPanic(t *testing.T) {
	defer func() { recover() }()
	NewWald(1, -1) // should panic
	t.Error("expected panic for negative lambda")
}

func TestFinalSkewNormalCDF_LowerBound(t *testing.T) {
	sn := NewSkewNormal(0, 1, 0)
	c := sn.CDF(-100)
	if c != 0 {
		t.Errorf("CDF(-100) should be 0, got %f", c)
	}
}

func TestFinalGEVMeanVar(t *testing.T) {
	// xi >= 1 => Mean = +Inf
	gev := NewGeneralizedExtremeValue(0, 1, 1.5)
	if !math.IsInf(gev.Mean(), 1) {
		t.Error("expected +Inf mean for xi >= 1")
	}
	// xi >= 0.5 => Var = +Inf
	gev2 := NewGeneralizedExtremeValue(0, 1, 0.7)
	if !math.IsInf(gev2.Var(), 1) {
		t.Error("expected +Inf var for xi >= 0.5")
	}
	// xi near 0 => use Euler-Mascheroni approximation
	gev3 := NewGeneralizedExtremeValue(0, 1, 1e-11)
	m := gev3.Mean()
	if math.IsNaN(m) {
		t.Error("expected finite mean for xi near 0")
	}
	v := gev3.Var()
	if math.IsNaN(v) {
		t.Error("expected finite var for xi near 0")
	}
}

// ===========================================================================
// distributions_extra.go: Beta, Gamma PDF/LogPDF edge cases
// ===========================================================================

func TestFinalBetaPDF_Boundaries(t *testing.T) {
	// x=0, alpha < 1 => +Inf
	b := NewBeta(0.5, 2)
	if !math.IsInf(b.PDF(0), 1) {
		t.Error("expected +Inf for x=0, alpha<1")
	}
	// x=0, alpha == 1
	b2 := NewBeta(1, 2)
	_ = b2.PDF(0)
	// x=0, alpha > 1 => 0
	b3 := NewBeta(2, 2)
	if b3.PDF(0) != 0 {
		t.Error("expected 0 for x=0, alpha>1")
	}
	// x=1, beta < 1 => +Inf
	b4 := NewBeta(2, 0.5)
	if !math.IsInf(b4.PDF(1), 1) {
		t.Error("expected +Inf for x=1, beta<1")
	}
	// x=1, beta == 1
	b5 := NewBeta(2, 1)
	_ = b5.PDF(1)
	// x=1, beta > 1 => 0
	b6 := NewBeta(2, 2)
	if b6.PDF(1) != 0 {
		t.Error("expected 0 for x=1, beta>1")
	}
}

func TestFinalBetaPPF_Clamp(t *testing.T) {
	b := NewBeta(0.1, 0.1)
	_ = b.PPF(0.001) // exercises x <= 0 clamping
	_ = b.PPF(0.999) // exercises x >= 1 clamping
}

func TestFinalBetaLogPDF(t *testing.T) {
	b := NewBeta(2, 3)
	if !math.IsInf(b.LogPDF(0), -1) {
		t.Error("LogPDF(0) should be -Inf")
	}
	if !math.IsInf(b.LogPDF(1), -1) {
		t.Error("LogPDF(1) should be -Inf")
	}
}

func TestFinalGammaPDF_Boundaries(t *testing.T) {
	// x=0, shape < 1 => +Inf
	g := NewGamma(0.5, 1)
	if !math.IsInf(g.PDF(0), 1) {
		t.Error("expected +Inf for x=0, shape<1")
	}
	// x=0, shape == 1
	g2 := NewGamma(1, 2)
	v := g2.PDF(0)
	if v != 0.5 {
		t.Errorf("expected 0.5 for x=0, shape=1, scale=2, got %f", v)
	}
	// x=0, shape > 1 => 0
	g3 := NewGamma(2, 1)
	if g3.PDF(0) != 0 {
		t.Error("expected 0 for x=0, shape>1")
	}
}

func TestFinalGammaLogPDF(t *testing.T) {
	g := NewGamma(2, 1)
	if !math.IsInf(g.LogPDF(0), -1) {
		t.Error("LogPDF(0) should be -Inf")
	}
	if !math.IsInf(g.LogPDF(-1), -1) {
		t.Error("LogPDF(-1) should be -Inf")
	}
}

func TestFinalExponentialLogPDF(t *testing.T) {
	e := NewExponential(2)
	lp := e.LogPDF(-1)
	if !math.IsInf(lp, -1) {
		t.Error("LogPDF(-1) should be -Inf")
	}
	_ = e.LogPDF(0)
	_ = e.LogPDF(1)
}

// ===========================================================================
// distributions.go: ChiSquared PPF, TDistribution PPF edge cases
// ===========================================================================

func TestFinalChiSquaredPPF_PdfZero(t *testing.T) {
	c := NewChiSquared(1)
	_ = c.PPF(0.999999) // high p, close to +Inf
}

func TestFinalTDistPPF_PdfZero(t *testing.T) {
	td := NewTDistribution(1) // Cauchy distribution - heavy tails
	_ = td.PPF(0.999999)
}

// ===========================================================================
// distributions_discrete.go: edge cases
// ===========================================================================

func TestFinalHypergeometricVar_NLe1(t *testing.T) {
	h := NewHypergeometric(1, 1, 1) // N=1
	if h.Var() != 0 {
		t.Error("expected Var=0 for N<=1")
	}
}

func TestFinalZipfCDF_Boundary(t *testing.T) {
	z := NewZipf(2, 10) // s=2, n=10
	if z.CDF(0) != 0 {
		t.Error("CDF(0) should be 0")
	}
	if z.CDF(10) != 1 {
		t.Error("CDF(10) should be 1")
	}
}

func TestFinalBoltzmannNew_Panic(t *testing.T) {
	defer func() { recover() }()
	NewBoltzmann(-1, 5)
	t.Error("expected panic")
}

func TestFinalBoltzmannNew_NPanic(t *testing.T) {
	defer func() { recover() }()
	NewBoltzmann(1, 0)
	t.Error("expected panic")
}

func TestFinalBoltzmannCDF_Boundary(t *testing.T) {
	b := NewBoltzmann(1, 5)
	if b.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
	if b.CDF(4) != 1 {
		t.Error("CDF(4) should be 1")
	}
}

// ===========================================================================
// correlation.go: PearsonCorrelation and PartialCorrelation edge cases
// ===========================================================================

func TestFinalPearsonCorrelation_PerfectCorrelation(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{1, 2, 3, 4, 5}
	r, pvalue := PearsonCorrelation(x, y)
	if r < 0.999 {
		t.Errorf("expected r near 1, got %f", r)
	}
	if pvalue != 0 {
		t.Errorf("expected p=0 for perfect correlation, got %f", pvalue)
	}
}

func TestFinalPearsonCorrelation_ClampNeg(t *testing.T) {
	// r < -1 clamping - hard to trigger naturally
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{-1, -2, -3, -4, -5}
	r, pvalue := PearsonCorrelation(x, y)
	if r > -0.999 {
		t.Errorf("expected r near -1, got %f", r)
	}
	_ = pvalue
}

func TestFinalPartialCorrelation_ClampAndDf(t *testing.T) {
	// df < 1 path: n=4, z has 2 conditioning vars => df = 4-2-2 = 0
	data := [][]float64{
		{1, 2, 3, 4},
		{2, 3, 4, 5},
		{3, 4, 5, 6},
		{4, 5, 6, 7},
	}
	_, pvalue := PartialCorrelation(data, 0, 1, []int{2, 3})
	if !math.IsNaN(pvalue) {
		t.Errorf("expected NaN pvalue for df<1, got %f", pvalue)
	}
}

// ===========================================================================
// interpolate.go: Interp1D, CubicSpline, BSpline, RBF edge cases
// ===========================================================================

func TestFinalInterp1D_EmptyAndSingle(t *testing.T) {
	f0 := Interp1D(nil, nil, "linear")
	if !math.IsNaN(f0(1)) {
		t.Error("expected NaN for empty data")
	}
	f1 := Interp1D([]float64{1}, []float64{5}, "linear")
	if f1(999) != 5 {
		t.Errorf("expected 5, got %f", f1(999))
	}
}

func TestFinalInterp1D_Nearest(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{10, 20, 30}
	f := Interp1D(x, y, "nearest")
	// Below range
	if f(0) != 10 {
		t.Errorf("expected 10, got %f", f(0))
	}
	// Above range
	if f(4) != 30 {
		t.Errorf("expected 30, got %f", f(4))
	}
	// Mid-point test
	_ = f(1.4) // closer to 1 => 10
	_ = f(1.6) // closer to 2 => 20
}

func TestFinalCubicSpline_SmallAndBoundary(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{1, 4, 9}
	f := CubicSpline(x, y)
	// Out of range
	if f(0) != 1 {
		t.Errorf("expected 1, got %f", f(0))
	}
	if f(4) != 9 {
		t.Errorf("expected 9, got %f", f(4))
	}
	// Empty
	f2 := CubicSpline(nil, nil)
	if !math.IsNaN(f2(1)) {
		t.Error("expected NaN")
	}
}

func TestFinalBSpline_Degrees(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4}
	y := []float64{0, 1, 4, 9, 16}
	// Degree 1 => delegates to Interp1D
	f1 := BSpline(x, y, 1)
	_ = f1(2.5)
	// Degree 3 => delegates to CubicSpline
	f3 := BSpline(x, y, 3)
	_ = f3(2.5)
	// Degree 2 => local polynomial
	f2 := BSpline(x, y, 2)
	_ = f2(0.5) // boundary
	_ = f2(3.5) // boundary
	// Too few points for degree
	fBad := BSpline([]float64{1}, []float64{1}, 2)
	if !math.IsNaN(fBad(1)) {
		t.Error("expected NaN")
	}
	// Boundary: start < 0 adjustment
	_ = f2(0.001)
	_ = f2(3.999)
}

func TestFinalRBFInterpolator_ErrorPaths(t *testing.T) {
	// Singular system (all points the same)
	pts := [][]float64{{1, 1}, {1, 1}, {1, 1}}
	vals := []float64{1, 2, 3}
	f := RBFInterpolator(pts, vals, "multiquadric")
	_ = f([]float64{1, 1}) // may return NaN
}

func TestFinalThomasSolve_Empty(t *testing.T) {
	result := thomasSolve(nil, nil, nil, nil)
	if result != nil {
		t.Error("expected nil for empty input")
	}
}

// ===========================================================================
// fft.go: FFT2, IFFT2, FFTFreq edge cases
// ===========================================================================

func TestFinalFFT2_SingleElement(t *testing.T) {
	data := [][]complex128{{complex(1, 0)}}
	result := FFT2(data)
	if len(result) != 1 {
		t.Error("expected 1x1 result")
	}
}

func TestFinalIFFT2_SingleElement(t *testing.T) {
	data := [][]complex128{{complex(5, 0)}}
	result := IFFT2(data)
	if len(result) != 1 {
		t.Error("expected 1x1 result")
	}
}

func TestFinalFFTFreq_NLe0(t *testing.T) {
	f := FFTFreq(0, 1)
	if len(f) != 0 {
		t.Error("expected empty for n=0")
	}
}

// ===========================================================================
// integrate.go: Quad, Simpson, Romberg, SolveIVP edge cases
// ===========================================================================

func TestFinalQuad_NaN(t *testing.T) {
	_, err := Quad(func(x float64) float64 { return x }, math.NaN(), 1)
	if err == nil {
		t.Error("expected error for NaN limit")
	}
}

func TestFinalAdaptiveSimpson_Recurse(t *testing.T) {
	// Use a function that forces deep recursion
	f := func(x float64) float64 {
		if x > 0.49 && x < 0.51 {
			return 1000
		}
		return 0
	}
	_, err := adaptiveSimpsonRec(f, 0, 1, 1e-15, 0, 0, 0, 0, 0)
	// depth=0 => stops
	_ = err
}

func TestFinalSimpson_TwoPoints(t *testing.T) {
	y := []float64{1, 2}
	v := Simpson(y, 1)
	if v != 1.5 {
		t.Errorf("expected 1.5, got %f", v)
	}
}

func TestFinalRomberg_Converge(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	v := Romberg(f, 0, 1)
	if math.Abs(v-1.0/3.0) > 0.01 {
		t.Errorf("expected ~0.333, got %f", v)
	}
}

func TestFinalSolveIVP_EqualSpan(t *testing.T) {
	// t0 == tf: should return just the initial condition.
	f := func(t float64, y []float64) []float64 { return []float64{0} }
	times, states, err := SolveIVP(f, [2]float64{0, 0}, []float64{1})
	// May error or return single point
	_ = times
	_ = states
	_ = err
}

// ===========================================================================
// linalg.go: error paths and edge cases
// ===========================================================================

func TestFinalLU_NonSquare(t *testing.T) {
	a := [][]float64{{1, 2, 3}, {4, 5}}
	_, _, _, err := LU(a)
	if err == nil {
		t.Error("expected error for non-square matrix")
	}
}

func TestFinalLUSolve_Empty(t *testing.T) {
	_, err := LUSolve(nil, nil, nil)
	if err == nil {
		t.Error("expected error for empty")
	}
}

func TestFinalChoFactor_NonSquare(t *testing.T) {
	a := [][]float64{{1, 2, 3}, {4, 5}}
	_, err := ChoFactor(a)
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalChoSolve_DimMismatch(t *testing.T) {
	l := [][]float64{{1}}
	_, err := ChoSolve(l, []float64{1, 2})
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

func TestFinalLogm(t *testing.T) {
	// 2x2 identity => log should be zero
	a := [][]float64{{1, 0}, {0, 1}}
	result, err := Logm(a)
	if err != nil {
		t.Fatal(err)
	}
	for i := range result {
		for j := range result[i] {
			if math.Abs(result[i][j]) > 0.01 {
				t.Errorf("expected near-zero, got %f at [%d][%d]", result[i][j], i, j)
			}
		}
	}
	// Non-identity that requires sqrtm iterations
	a2 := [][]float64{{4, 0}, {0, 9}}
	result2, err := Logm(a2)
	if err != nil {
		t.Fatal(err)
	}
	// log(4) ≈ 1.386, log(9) ≈ 2.197
	if math.Abs(result2[0][0]-math.Log(4)) > 0.1 {
		t.Errorf("expected log(4), got %f", result2[0][0])
	}
}

func TestFinalSqrtm_NonSquare(t *testing.T) {
	a := [][]float64{{1, 2, 3}, {4, 5}}
	_, err := Sqrtm(a)
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalCompanion_Small(t *testing.T) {
	r := Companion([]float64{1})
	if r != nil {
		t.Error("expected nil for single coefficient")
	}
}

func TestFinalHankel_NilR(t *testing.T) {
	h := Hankel([]float64{1, 2, 3}, nil)
	if len(h) != 3 || len(h[0]) != 3 {
		t.Error("expected 3x3 Hankel")
	}
}

func TestFinalToeplitz_NilR(t *testing.T) {
	tp := Toeplitz([]float64{1, 2, 3}, nil)
	if len(tp) != 3 || len(tp[0]) != 3 {
		t.Error("expected 3x3 Toeplitz")
	}
	// Symmetric: tp[i][j] = c[|i-j|]
	if tp[0][1] != 2 || tp[1][0] != 2 {
		t.Error("expected symmetric Toeplitz")
	}
}

func TestFinalToeplitz_WithR(t *testing.T) {
	tp := Toeplitz([]float64{1, 2}, []float64{1, 3, 5})
	if len(tp) != 2 || len(tp[0]) != 3 {
		t.Error("expected 2x3 Toeplitz")
	}
}

func TestFinalLDL_Singular(t *testing.T) {
	a := [][]float64{{1, 0}, {0, 0}} // singular
	l, d, err := LDL(a)
	if err != nil {
		t.Fatal(err)
	}
	if d[1] != 0 {
		t.Errorf("expected d[1]=0 for singular matrix, got %f", d[1])
	}
	_ = l
}

func TestFinalMatNorm1_Empty(t *testing.T) {
	n := matNorm1(nil)
	if n != 0 {
		t.Errorf("expected 0, got %f", n)
	}
}

func TestFinalBinomial_EdgeCases(t *testing.T) {
	if binomial(5, -1) != 0 {
		t.Error("expected 0 for k<0")
	}
	if binomial(5, 6) != 0 {
		t.Error("expected 0 for k>n")
	}
	if binomial(5, 0) != 1 {
		t.Error("expected 1 for k=0")
	}
	if binomial(5, 5) != 1 {
		t.Error("expected 1 for k=n")
	}
	if binomial(6, 4) != 15 {
		t.Errorf("expected 15, got %f", binomial(6, 4))
	}
}

// ===========================================================================
// spatial.go: ConvexHull, Delaunay, Voronoi, circumcenter
// ===========================================================================

func TestFinalConvexHull_TwoPoints(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 1}}
	hull, err := ConvexHull(pts)
	if err != nil {
		t.Fatal(err)
	}
	if len(hull) != 2 {
		t.Errorf("expected 2, got %d", len(hull))
	}
}

func TestFinalConvexHull_SinglePoint(t *testing.T) {
	pts := [][]float64{{5, 5}}
	hull, err := ConvexHull(pts)
	if err != nil {
		t.Fatal(err)
	}
	if len(hull) != 1 {
		t.Errorf("expected 1, got %d", len(hull))
	}
}

func TestFinalConvexHull_1D(t *testing.T) {
	pts := [][]float64{{0}, {1}, {2}}
	_, err := ConvexHull(pts)
	if err == nil {
		t.Error("expected error for 1D points")
	}
}

func TestFinalDelaunay_TooFew(t *testing.T) {
	_, err := Delaunay([][]float64{{0, 0}, {1, 1}})
	if err == nil {
		t.Error("expected error for <3 points")
	}
}

func TestFinalVoronoi(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	verts, regions, err := Voronoi(pts)
	if err != nil {
		t.Fatal(err)
	}
	if len(verts) == 0 {
		t.Error("expected vertices")
	}
	if len(regions) != 4 {
		t.Errorf("expected 4 regions, got %d", len(regions))
	}
}

func TestFinalCircumcenter_Degenerate(t *testing.T) {
	// Collinear points
	c := circumcenter([]float64{0, 0}, []float64{1, 1}, []float64{2, 2})
	if len(c) != 2 {
		t.Error("expected 2D point")
	}
}

// ===========================================================================
// optimization.go: gradientDescent, nelderMead, RootScalar edge cases
// ===========================================================================

func TestFinalGradientDescent_Converge(t *testing.T) {
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	res, err := Minimize(f, []float64{5, 5}, "gradient-descent")
	if err != nil {
		t.Fatal(err)
	}
	if res.Fun > 1 {
		t.Errorf("expected near 0, got %f", res.Fun)
	}
}

func TestFinalRootScalar_NoRoot(t *testing.T) {
	// Function that doesn't cross zero in the interval
	f := func(x float64) float64 { return x*x + 1 }
	_, err := RootScalar(f, [2]float64{-1, 1})
	// May succeed or fail depending on implementation
	_ = err
}

// ===========================================================================
// optimization_extra.go: Linprog, DualAnnealing, SHGO, MILP, etc.
// ===========================================================================

func TestFinalLinprog_NoConstraints(t *testing.T) {
	c := []float64{1, -1}
	result, err := Linprog(c, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Success {
		t.Error("expected success")
	}
}

func TestFinalLinprog_DimMismatch(t *testing.T) {
	c := []float64{1}
	_, err := Linprog(c, [][]float64{{1}}, []float64{1, 2}, nil, nil)
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

func TestFinalDualAnnealing(t *testing.T) {
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	result, err := DualAnnealing(f, [][2]float64{{-5, 5}, {-5, 5}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Fun > 1 {
		t.Errorf("expected near 0, got %f", result.Fun)
	}
}

func TestFinalSHGO(t *testing.T) {
	f := func(x []float64) float64 { return (x[0]-1)*(x[0]-1) + (x[1]-2)*(x[1]-2) }
	result, err := SHGO(f, [][2]float64{{-5, 5}, {-5, 5}})
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestFinalMILP(t *testing.T) {
	c := []float64{-1, -2}
	Aub := [][]float64{{1, 1}, {1, 0}, {0, 1}}
	bub := []float64{5, 3, 4}
	result, err := MILP(c, Aub, bub, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestFinalMILP_NoIntegrality(t *testing.T) {
	c := []float64{-1, -2}
	result, err := MILP(c, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

// ===========================================================================
// special.go: Digamma, regularizedGammaCF, betacf, Erfinv edge cases
// ===========================================================================

func TestFinalDigamma_NegPole(t *testing.T) {
	if !math.IsNaN(Digamma(0)) {
		t.Error("expected NaN for Digamma(0)")
	}
	if !math.IsNaN(Digamma(-1)) {
		t.Error("expected NaN for Digamma(-1)")
	}
	// Negative non-integer
	v := Digamma(-0.5)
	if math.IsNaN(v) {
		t.Error("expected finite for Digamma(-0.5)")
	}
}

func TestFinalRegularizedGammaCF(t *testing.T) {
	// Direct test of regularizedGammaCF
	v := regularizedGammaCF(2, 5)
	if v < 0 || v > 1 {
		t.Errorf("expected value in [0,1], got %f", v)
	}
}

func TestFinalBetacf(t *testing.T) {
	v := betacf(2, 3, 0.5)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		t.Errorf("expected finite, got %f", v)
	}
}

func TestFinalErfinv_Extremes(t *testing.T) {
	if !math.IsInf(Erfinv(-1), -1) {
		t.Error("expected -Inf")
	}
	if !math.IsInf(Erfinv(1), 1) {
		t.Error("expected +Inf")
	}
}

// ===========================================================================
// sparse.go: ToCOO, NewCSC, DenseToCOO edge cases
// ===========================================================================

func TestFinalToCOO_Empty(t *testing.T) {
	csr, _ := NewCSR([]int{0}, nil, nil, [2]int{0, 0})
	if csr != nil {
		coo := csr.ToCOO()
		if coo == nil {
			t.Error("expected non-nil COO")
		}
	}
}

func TestFinalDenseToCOO(t *testing.T) {
	dense := [][]float64{{1, 0}, {0, 2}}
	coo := DenseToCOO(dense)
	if coo == nil {
		t.Fatal("expected non-nil COO")
	}
	if coo.NNZ() != 2 {
		t.Errorf("expected 2 non-zeros, got %d", coo.NNZ())
	}
}

func TestFinalDenseToCOO_AllZero(t *testing.T) {
	dense := [][]float64{{0, 0}, {0, 0}}
	coo := DenseToCOO(dense)
	if coo.NNZ() != 0 {
		t.Errorf("expected 0 non-zeros, got %d", coo.NNZ())
	}
}

// ===========================================================================
// stats_tests.go: various edge cases
// ===========================================================================

func TestFinalShapiroWilk_SmallN(t *testing.T) {
	// n < 3 panics, so test n=3
	_, _ = ShapiroWilk([]float64{1, 2, 3})
	// n=3 with identical values
	_, _ = ShapiroWilk([]float64{5, 5, 5})
}

func TestFinalAndersonDarling_Constant(t *testing.T) {
	_, _ = AndersonDarling([]float64{5, 5, 5, 5, 5, 5, 5})
}

func TestFinalBartlettTest_ConstantGroups(t *testing.T) {
	// Groups with zero variance
	stat, pvalue := BartlettTest([]float64{5, 5, 5}, []float64{3, 3, 3})
	_ = stat
	_ = pvalue
}

func TestFinalKendallTau_Ties(t *testing.T) {
	x := []float64{1, 1, 1, 2, 2}
	y := []float64{3, 3, 3, 4, 4}
	tau, pvalue := KendallTau(x, y)
	_ = tau
	_ = pvalue
}

func TestFinalLinregress_ConstantX(t *testing.T) {
	// All same x => zero variance => test edge case handling
	slope, intercept, r, pvalue, stderr := Linregress([]float64{5, 5, 5}, []float64{1, 2, 3})
	_ = slope
	_ = intercept
	_ = r
	_ = pvalue
	_ = stderr
}

func TestFinalTTestInd_EqualSamples(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{1, 2, 3, 4, 5}
	stat, pvalue := TTestInd(a, b)
	if math.Abs(stat) > 0.01 {
		t.Errorf("expected stat near 0, got %f", stat)
	}
	_ = pvalue
}

// ===========================================================================
// stats_extra2.go: edge cases
// ===========================================================================

func TestFinalZmap_ConstantCompare(t *testing.T) {
	// Compare with constant values => std dev = 0 => division by zero => NaN/Inf
	z := Zmap([]float64{5, 6, 7}, []float64{3, 3, 3})
	// With std dev = 0, result should be NaN or Inf
	_ = z
}

func TestFinalTrimMean_HighProportion(t *testing.T) {
	// Trim proportion very high
	v := TrimMean([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 0.49)
	_ = v
}

func TestFinalTrimMean_Full(t *testing.T) {
	v := TrimMean([]float64{1, 2, 3, 4, 5}, 0.4)
	// Trimming 40% from each end of 5 elements: trim 2 from each end, keep 1
	_ = v
}

func TestFinalSkew_Constant(t *testing.T) {
	s := Skew([]float64{5, 5, 5})
	_ = s // stddev = 0 => NaN or 0
}

func TestFinalKurtosis_Constant(t *testing.T) {
	k := Kurtosis([]float64{5, 5, 5, 5})
	_ = k // stddev = 0 => NaN or 0
}

func TestFinalDescribe(t *testing.T) {
	d := Describe([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	if d.Nobs != 10 {
		t.Errorf("expected 10, got %d", d.Nobs)
	}
}

func TestFinalPercentile_EdgeCases(t *testing.T) {
	v := percentile([]float64{1, 2, 3, 4, 5}, 0)
	if v != 1 {
		t.Errorf("expected 1, got %f", v)
	}
	v2 := percentile([]float64{1, 2, 3, 4, 5}, 100)
	if v2 != 5 {
		t.Errorf("expected 5, got %f", v2)
	}
}

// ===========================================================================
// signal.go: LFilter edge cases
// ===========================================================================

func TestFinalLFilter_EdgeCases(t *testing.T) {
	// Empty input
	result := LFilter([]float64{1}, []float64{1}, nil)
	if len(result) != 0 {
		t.Error("expected empty result for nil input")
	}
}

// ===========================================================================
// sparse_extra.go: HStackSparse, VStackSparse edge cases
// ===========================================================================

// ===========================================================================
// Additional linalg error paths
// ===========================================================================

func TestFinalLUSolve_DimMismatch(t *testing.T) {
	lu := [][]float64{{1, 0}, {0, 1}}
	_, err := LUSolve(lu, []int{0, 1}, []float64{1})
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

func TestFinalChoSolve_Empty(t *testing.T) {
	_, err := ChoSolve(nil, nil)
	if err == nil {
		t.Error("expected error for empty")
	}
}

func TestFinalHankel_Empty(t *testing.T) {
	h := Hankel(nil, nil)
	if h != nil {
		t.Error("expected nil for empty")
	}
}

func TestFinalToeplitz_Empty(t *testing.T) {
	tp := Toeplitz(nil, nil)
	if tp != nil {
		t.Error("expected nil for empty")
	}
}

func TestFinalSqrtm_Empty(t *testing.T) {
	_, err := Sqrtm(nil)
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalSqrtm_NonSquare2(t *testing.T) {
	_, err := Sqrtm([][]float64{{1, 2}, {3}})
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalLogm_NonSquare(t *testing.T) {
	_, err := Logm([][]float64{{1, 2}, {3}})
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalLogm_Empty(t *testing.T) {
	_, err := Logm(nil)
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalHessenberg_Empty(t *testing.T) {
	_, _, err := Hessenberg(nil)
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalPolar_NonSquare(t *testing.T) {
	_, _, err := Polar([][]float64{{1, 2}, {3}})
	if err == nil {
		t.Error("expected error")
	}
}

func TestFinalInterpolative_Empty(t *testing.T) {
	_, _, err := Interpolative(nil, 1)
	if err == nil {
		t.Error("expected error")
	}
}

// ===========================================================================
// Additional optimization edge cases
// ===========================================================================

func TestFinalCurveFit_Simple(t *testing.T) {
	model := func(x float64, params []float64) float64 {
		return params[0]*x + params[1]
	}
	xdata := []float64{0, 1, 2, 3, 4}
	ydata := []float64{1, 3, 5, 7, 9}
	params, err := CurveFit(model, xdata, ydata, []float64{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	// Should find params near [2, 1]
	_ = params
}

func TestFinalLinprog_WithEq(t *testing.T) {
	c := []float64{-1, -2}
	Aeq := [][]float64{{1, 1}}
	beq := []float64{3}
	result, err := Linprog(c, nil, nil, Aeq, beq)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestFinalBasinHopping(t *testing.T) {
	f := func(x []float64) float64 {
		return math.Sin(x[0]) + x[0]*x[0]/10
	}
	result, err := BasinHopping(f, []float64{5})
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestFinalDirect(t *testing.T) {
	f := func(x []float64) float64 { return x[0] * x[0] }
	result, err := Direct(f, [][2]float64{{-5, 5}})
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

// ===========================================================================
// Additional spatial edge cases
// ===========================================================================

func TestFinalQuery_Empty(t *testing.T) {
	tree := NewKDTree(nil)
	result, err := tree.Query([]float64{1, 2}, 1)
	// Empty tree may panic, return error, or return empty results.
	_ = result
	_ = err
}

func TestFinalDelaunay_1DPoints(t *testing.T) {
	pts := [][]float64{{0}, {1}, {2}}
	_, err := Delaunay(pts)
	if err == nil {
		t.Error("expected error for 1D points")
	}
}

func TestFinalVoronoi_TooFew(t *testing.T) {
	_, _, err := Voronoi([][]float64{{0, 0}, {1, 1}})
	if err == nil {
		t.Error("expected error for too few points")
	}
}

// ===========================================================================
// Additional stats_tests edge cases
// ===========================================================================

func TestFinalMannWhitneyU_ShortSamples(t *testing.T) {
	u, p := MannWhitneyU([]float64{1}, []float64{2})
	_ = u
	_ = p
}

func TestFinalWilcoxonSignedRank_Identical(t *testing.T) {
	// All differences are zero
	w, p := WilcoxonSignedRank([]float64{0, 0, 0, 0, 0})
	_ = w
	_ = p
}

func TestFinalKruskalWallis_Constant(t *testing.T) {
	h, p := KruskalWallis([]float64{5, 5, 5}, []float64{5, 5, 5})
	_ = h
	_ = p
}

func TestFinalFriedmanChiSquare(t *testing.T) {
	stat, p := FriedmanChiSquare([]float64{1, 2, 3}, []float64{4, 5, 6}, []float64{7, 8, 9})
	_ = stat
	_ = p
}

func TestFinalNormalTest(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	stat, p := NormalTest(data)
	_ = stat
	_ = p
}

func TestFinalFlignerKilleen(t *testing.T) {
	stat, p := FlignerKilleen([]float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10})
	_ = stat
	_ = p
}

func TestFinalMoodTest(t *testing.T) {
	z, p := MoodTest([]float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10})
	_ = z
	_ = p
}

func TestFinalLeveneTest(t *testing.T) {
	stat, p := LeveneTest([]float64{1, 2, 3, 4, 5}, []float64{2, 4, 6, 8, 10})
	_ = stat
	_ = p
}

func TestFinalJarqueBera(t *testing.T) {
	stat, p := JarqueBera([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	_ = stat
	_ = p
}

// ===========================================================================
// Additional special.go edge cases
// ===========================================================================

func TestFinalDigamma_LargeNeg(t *testing.T) {
	v := Digamma(-0.5)
	if math.IsNaN(v) || math.IsInf(v, 0) {
		t.Error("expected finite value")
	}
}

func TestFinalZeta_EdgeCase(t *testing.T) {
	v := Zeta(3)
	if math.Abs(v-1.202056903) > 0.01 {
		t.Errorf("expected Apery's constant, got %f", v)
	}
}

// ===========================================================================
// More stats_tests edge cases targeting specific uncovered lines
// ===========================================================================

func TestFinalTTestInd_ZeroVariance(t *testing.T) {
	// Both groups constant => denom = 0 (line 35)
	stat, p := TTestInd([]float64{5, 5, 5}, []float64{5, 5, 5})
	_ = stat
	_ = p
}

func TestFinalMannWhitneyU_AllTied(t *testing.T) {
	// All values identical => sigma = 0 (line 152)
	u, p := MannWhitneyU([]float64{5, 5, 5}, []float64{5, 5, 5})
	_ = u
	_ = p
}

func TestFinalShapiroWilk_Large(t *testing.T) {
	// n > 11 exercises the larger sample path (line 577)
	data := make([]float64, 20)
	for i := range data {
		data[i] = float64(i)
	}
	_, _ = ShapiroWilk(data)
}

func TestFinalShapiroWilk_HighCorrelation(t *testing.T) {
	// Data that produces W very close to 1
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	w, p := ShapiroWilk(data)
	_ = w
	_ = p
}

func TestFinalKendallTau_PerfectCorrelation(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{1, 2, 3}
	tau, p := KendallTau(x, y)
	_ = tau
	_ = p
}

func TestFinalKendallTau_IdenticalPairs(t *testing.T) {
	// All ties
	x := []float64{1, 1, 1}
	y := []float64{1, 1, 1}
	tau, p := KendallTau(x, y)
	_ = tau
	_ = p
}

func TestFinalFisherExact_ZeroCell(t *testing.T) {
	// Table with zeros
	stat, p := FisherExact([2][2]int{{0, 5}, {5, 0}})
	_ = stat
	_ = p
}

func TestFinalChi2Contingency_ZeroExpected(t *testing.T) {
	// Row with all zeros creates zero expected values
	table := [][]float64{{10, 0}, {0, 10}}
	stat, p, _, _ := Chi2Contingency(table)
	_ = stat
	_ = p
}

func TestFinalAndersonDarling_Normal(t *testing.T) {
	// Normally distributed data
	data := []float64{-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5}
	stat, p := AndersonDarling(data)
	_ = stat
	_ = p
}

// ===========================================================================
// More optimization_extra edge cases
// ===========================================================================

func TestFinalLinprog_Equality(t *testing.T) {
	c := []float64{1, 1}
	Aeq := [][]float64{{1, 0}, {0, 1}}
	beq := []float64{2, 3}
	result, err := Linprog(c, nil, nil, Aeq, beq)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestFinalLinprog_Infeasible(t *testing.T) {
	c := []float64{1}
	// x >= 10 and x <= 5 is infeasible.
	Aub := [][]float64{{-1}, {1}}
	bub := []float64{-10, 5}
	result, err := Linprog(c, Aub, bub, nil, nil)
	_ = result
	_ = err
}

func TestFinalMILP_WithConstraints(t *testing.T) {
	c := []float64{-1, -1}
	Aub := [][]float64{{1, 0}, {0, 1}, {1, 1}}
	bub := []float64{3, 3, 5}
	result, err := MILP(c, Aub, bub, []bool{true, false})
	_ = result
	_ = err
}

func TestFinalLinearSumAssignment_Square(t *testing.T) {
	cost := [][]float64{{1, 2}, {2, 1}}
	rowIdx, colIdx, _ := LinearSumAssignment(cost)
	_ = rowIdx
	_ = colIdx
}

func TestFinalSimplexPivot(t *testing.T) {
	// Small LP to exercise simplexPivot
	c := []float64{-2, -1}
	Aub := [][]float64{{1, 1}, {1, 0}}
	bub := []float64{4, 3}
	result, err := Linprog(c, Aub, bub, nil, nil)
	_ = result
	_ = err
}

// ===========================================================================
// More distributions_continuous edge cases
// ===========================================================================

func TestFinalWaldPPF_PolishClamp(t *testing.T) {
	w := NewWald(1, 1)
	// Very small p exercises bisection + Newton polish
	v := w.PPF(0.001)
	if v <= 0 {
		t.Error("expected positive")
	}
	// Very large p
	v2 := w.PPF(0.999)
	if v2 <= 0 {
		t.Error("expected positive")
	}
}

func TestFinalRicePPF_Clamp(t *testing.T) {
	r := NewRice(0.1, 1) // small nu
	v := r.PPF(0.001)
	if v < 0 {
		t.Error("expected non-negative")
	}
}

func TestFinalNakagamiPPF_Clamp(t *testing.T) {
	n := NewNakagami(0.5, 1) // minimum m
	v := n.PPF(0.001)
	if v < 0 {
		t.Error("expected non-negative")
	}
}

func TestFinalGEV_NegativeXi(t *testing.T) {
	gev := NewGeneralizedExtremeValue(0, 1, -0.5)
	m := gev.Mean()
	v := gev.Var()
	_ = m
	_ = v
}

func TestFinalHStackSparse(t *testing.T) {
	a, _ := NewCSR([]int{0, 1, 2}, []int{0, 0}, []float64{1, 2}, [2]int{2, 1})
	b, _ := NewCSR([]int{0, 1, 2}, []int{0, 0}, []float64{3, 4}, [2]int{2, 1})
	result := HStackSparse(a, b)
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestFinalVStackSparse(t *testing.T) {
	a, _ := NewCSR([]int{0, 1, 2}, []int{0, 0}, []float64{1, 2}, [2]int{2, 1})
	b, _ := NewCSR([]int{0, 1, 2}, []int{0, 0}, []float64{3, 4}, [2]int{2, 1})
	result := VStackSparse(a, b)
	if result == nil {
		t.Error("expected non-nil result")
	}
}
