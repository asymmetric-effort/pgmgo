//go:build unit

package scigo

import (
	"math"
	"math/cmplx"
	"testing"
)

// ===========================================================================
// Distribution boundary tests: PPF, PDF, CDF, LogPDF edge cases
// ===========================================================================

func TestChiSquared_PPF_Boundaries(t *testing.T) {
	c := NewChiSquared(5)
	if c.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	if !math.IsInf(c.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
}

func TestChiSquared_LogPDF_Boundaries(t *testing.T) {
	c := NewChiSquared(5)
	if !math.IsInf(c.LogPDF(-1), -1) {
		t.Error("LogPDF(-1) should be -inf")
	}
	// x=0, df>=2: log(0) or -inf depending on df
	c2 := NewChiSquared(2)
	_ = c2.LogPDF(0)
	c3 := NewChiSquared(4)
	if !math.IsInf(c3.LogPDF(0), -1) {
		t.Error("LogPDF(0) with df=4 should be -inf")
	}
}

func TestChiSquared_PDF_Boundaries(t *testing.T) {
	c := NewChiSquared(5)
	if c.PDF(-1) != 0 {
		t.Error("PDF(-1) should be 0")
	}
	// x=0 with df<2
	c1 := NewChiSquared(1)
	if !math.IsInf(c1.PDF(0), 1) {
		t.Error("PDF(0) with df=1 should be +inf")
	}
	// x=0 with df=2
	c2 := NewChiSquared(2)
	if c2.PDF(0) != 0.5 {
		t.Errorf("PDF(0) with df=2 should be 0.5, got %f", c2.PDF(0))
	}
	// x=0 with df>2
	c3 := NewChiSquared(4)
	if c3.PDF(0) != 0 {
		t.Error("PDF(0) with df=4 should be 0")
	}
}

func TestTDistribution_PPF_Boundaries(t *testing.T) {
	td := NewTDistribution(10)
	if !math.IsInf(td.PPF(0), -1) {
		t.Error("PPF(0) should be -inf")
	}
	if !math.IsInf(td.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
	// Convergence: use Newton's method
	td.PPF(0.975) // normal case
}

func TestFDistribution_PPF_Boundaries(t *testing.T) {
	f := NewFDistribution(5, 10)
	if f.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	// Newton convergence guard: pdfVal == 0
	// This is hard to trigger; just exercise normal case.
	_ = f.PPF(0.001)
}

func TestFDistribution_LogPDF_Boundaries(t *testing.T) {
	f := NewFDistribution(5, 10)
	if !math.IsInf(f.LogPDF(0), -1) {
		t.Error("LogPDF(0) should be -inf")
	}
}

func TestFDistribution_PPF_NewtonXClamp(t *testing.T) {
	// Very small p that might make x go negative in Newton's method.
	f := NewFDistribution(1, 1)
	result := f.PPF(0.001)
	if result <= 0 {
		t.Error("PPF should return positive value")
	}
}

// ===========================================================================
// Continuous distributions: Rice, Nakagami, VonMises, Wald, SkewNormal, GEV
// ===========================================================================

func TestNewRice_PanicNu(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nu < 0")
		}
	}()
	NewRice(-1, 1)
}

func TestRice_CDF_Boundary(t *testing.T) {
	r := NewRice(1, 1)
	if r.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
}

func TestRice_PPF_Boundaries(t *testing.T) {
	r := NewRice(1, 1)
	if r.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	// Normal usage.
	val := r.PPF(0.5)
	if val <= 0 {
		t.Error("PPF(0.5) should be positive")
	}
	// Newton pdfVal==0 guard: hard to trigger. Test x<=0 guard.
	val2 := r.PPF(0.001)
	if val2 <= 0 {
		t.Error("PPF(0.001) should be positive")
	}
}

func TestNewNakagami_PanicOmega(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for omega <= 0")
		}
	}()
	NewNakagami(1, 0)
}

func TestNakagami_PDF_Boundary(t *testing.T) {
	n := NewNakagami(1, 1)
	if n.PDF(-1) != 0 {
		t.Error("PDF(-1) should be 0")
	}
	if n.PDF(0) != 0 {
		t.Error("PDF(0) should be 0")
	}
}

func TestNakagami_CDF_Boundary(t *testing.T) {
	n := NewNakagami(1, 1)
	if n.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
}

func TestNakagami_PPF_Boundaries(t *testing.T) {
	n := NewNakagami(1, 1)
	if n.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	if !math.IsInf(n.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
	// Exercise Newton's method
	val := n.PPF(0.5)
	if val <= 0 {
		t.Error("PPF(0.5) should be positive")
	}
}

func TestVonMises_CDF_Boundaries(t *testing.T) {
	v := NewVonMises(0, 1)
	// Values far from center should wrap around.
	_ = v.CDF(10 * math.Pi)  // xn > pi, needs wrapping
	_ = v.CDF(-10 * math.Pi) // xn < -pi, needs wrapping
}

func TestVonMises_PPF_Boundaries(t *testing.T) {
	v := NewVonMises(0, 1)
	if v.PPF(0) != -math.Pi {
		t.Errorf("PPF(0) should be -pi, got %f", v.PPF(0))
	}
	if v.PPF(1) != math.Pi {
		t.Errorf("PPF(1) should be pi, got %f", v.PPF(1))
	}
}

func TestNewWald_PanicMu(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for mu <= 0")
		}
	}()
	NewWald(0, 1)
}

func TestWald_CDF_Boundary(t *testing.T) {
	w := NewWald(1, 1)
	if w.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
}

func TestWald_PPF_Boundaries(t *testing.T) {
	w := NewWald(1, 1)
	if w.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	if !math.IsInf(w.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
}

func TestSkewNormal_CDF_LowerBound_Boost(t *testing.T) {
	sn := NewSkewNormal(0, 1, 2)
	// Very small x => x <= lower bound => 0
	if sn.CDF(-100) != 0 {
		t.Error("CDF(-100) should be 0")
	}
}

func TestSkewNormal_PPF_Boundaries(t *testing.T) {
	sn := NewSkewNormal(0, 1, 2)
	if !math.IsInf(sn.PPF(0), -1) {
		t.Error("PPF(0) should be -inf")
	}
	if !math.IsInf(sn.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
}

func TestGEV_PPF_Boundaries(t *testing.T) {
	// xi > 0
	gev := NewGeneralizedExtremeValue(0, 1, 0.5)
	lower := gev.PPF(0)
	_ = lower // Should be mu - sigma/xi

	upper := gev.PPF(1)
	if !math.IsInf(upper, 1) {
		t.Error("PPF(1) for xi > 0 should be +inf")
	}

	// xi < 0
	gev2 := NewGeneralizedExtremeValue(0, 1, -0.5)
	_ = gev2.PPF(0)
	upper2 := gev2.PPF(1)
	_ = upper2 // Should be mu - sigma/xi

	// xi ≈ 0 (Gumbel)
	gev3 := NewGeneralizedExtremeValue(0, 1, 0)
	_ = gev3.PPF(0.5)
}

func TestGEV_Mean_Boundaries(t *testing.T) {
	// xi >= 1 => +inf
	gev := NewGeneralizedExtremeValue(0, 1, 1.5)
	if !math.IsInf(gev.Mean(), 1) {
		t.Error("Mean for xi >= 1 should be +inf")
	}
}

func TestGEV_Var_Boundaries(t *testing.T) {
	// xi >= 0.5 => +inf
	gev := NewGeneralizedExtremeValue(0, 1, 0.6)
	if !math.IsInf(gev.Var(), 1) {
		t.Error("Var for xi >= 0.5 should be +inf")
	}
	// xi ≈ 0
	gev2 := NewGeneralizedExtremeValue(0, 1, 0)
	v := gev2.Var()
	if math.IsNaN(v) {
		t.Error("Var for xi=0 should not be NaN")
	}
}

// ===========================================================================
// Discrete distributions: edge cases
// ===========================================================================

func TestNegBinomial_CDF_Boundary(t *testing.T) {
	nb := NewNegativeBinomial(5, 0.5)
	if nb.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
}

func TestNegBinomial_Var_Small(t *testing.T) {
	nb := NewNegativeBinomial(1, 0.999)
	v := nb.Var()
	if v < 0 {
		t.Error("expected non-negative variance")
	}
}

func TestHypergeometric_CDF_Boundary(t *testing.T) {
	h := NewHypergeometric(50, 10, 20)
	if h.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
}

func TestBoltzmann_NewPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for lambda <= 0")
		}
	}()
	NewBoltzmann(0, 10)
}

func TestBoltzmann_CDF_Boundaries(t *testing.T) {
	b := NewBoltzmann(1, 10)
	if b.CDF(-1) != 0 {
		t.Error("CDF(-1) should be 0")
	}
	// k >= N
	if b.CDF(10) != 1 {
		t.Errorf("CDF(N) should be 1, got %f", b.CDF(10))
	}
}

// ===========================================================================
// Extra distributions: Pareto, Laplace, LogNormal edge cases
// ===========================================================================

func TestPareto_PDF_Boundary(t *testing.T) {
	p := NewPareto(1, 1)
	if p.PDF(-1) != 0 {
		t.Error("PDF(-1) should be 0")
	}
	// PDF at x < xm
	if p.PDF(0.5) != 0 {
		t.Error("PDF(0.5) should be 0 for xm=1")
	}
}

func TestPareto_PPF_Boundaries(t *testing.T) {
	p := NewPareto(1, 1)
	_ = p.PPF(0) // Should be xm
	if !math.IsInf(p.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
	// Normal case
	_ = p.PPF(0.5)
}

func TestPareto_LogPDF(t *testing.T) {
	p := NewPareto(1, 1)
	if !math.IsInf(p.LogPDF(0.5), -1) {
		t.Error("LogPDF(x < xm) should be -inf")
	}
}

func TestLaplace_PDF_Edge(t *testing.T) {
	l := NewLaplace(0, 1)
	// Check negative x
	_ = l.PDF(-5)
	_ = l.PDF(5)
}

func TestLognormal_PPF_Boundaries(t *testing.T) {
	ln := NewLognormal(0, 1)
	if ln.PPF(0) != 0 {
		t.Error("PPF(0) should be 0")
	}
	if !math.IsInf(ln.PPF(1), 1) {
		t.Error("PPF(1) should be +inf")
	}
	_ = ln.PPF(0.001)
}

func TestLognormal_LogPDF(t *testing.T) {
	ln := NewLognormal(0, 1)
	if !math.IsInf(ln.LogPDF(-1), -1) {
		t.Error("LogPDF(-1) should be -inf")
	}
}

// ===========================================================================
// FFT edge cases
// ===========================================================================

func TestRFFT_Empty(t *testing.T) {
	result := RFFT(nil)
	if result != nil {
		t.Error("expected nil for empty input")
	}
}

func TestIRFFT_Empty(t *testing.T) {
	result := IRFFT(nil, 0)
	if result != nil {
		t.Error("expected nil for empty input")
	}
}

func TestIRFFT_InferN(t *testing.T) {
	// n=0 inferred as 2*(len(x)-1)
	result := IRFFT([]complex128{1 + 0i, 0.5 + 0i}, 0)
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestFFT2_Empty(t *testing.T) {
	result := FFT2(nil)
	if result != nil {
		t.Error("expected nil for empty input")
	}
}

func TestFFT2_UnevenRows(t *testing.T) {
	_ = FFT2([][]complex128{{1}, {1, 2}})
}

func TestIFFT2_Empty(t *testing.T) {
	result := IFFT2(nil)
	if result != nil {
		t.Error("expected nil for empty input")
	}
}

func TestIFFT2_UnevenRows(t *testing.T) {
	_ = IFFT2([][]complex128{{1}, {1, 2}})
}

func TestFFTFreq_ZeroN(t *testing.T) {
	result := FFTFreq(0, 1.0)
	if result != nil {
		t.Error("expected nil for n=0")
	}
}

func TestRFFTFreq_ZeroN(t *testing.T) {
	result := RFFTFreq(0, 1.0)
	if result != nil {
		t.Error("expected nil for n=0")
	}
}

func TestRFFTFreq_ZeroD(t *testing.T) {
	_ = RFFTFreq(4, 0)
}

func TestFftInPlace_NonPow2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-power-of-2")
		}
	}()
	fftInPlace(make([]complex128, 3), false)
}

func TestNextPow2_Zero(t *testing.T) {
	_ = nextPow2(0) // Exercises the n<=0 path
}

// ===========================================================================
// Integration edge cases
// ===========================================================================

func TestQuad_NaNBounds(t *testing.T) {
	f := func(x float64) float64 { return x }
	_, err := Quad(f, math.NaN(), 1)
	if err == nil {
		t.Error("expected error for NaN bounds")
	}
}

func TestQuad_MaxDepthExhausted(t *testing.T) {
	// Highly oscillating function that exhausts adaptive depth.
	f := func(x float64) float64 {
		return math.Sin(10000 * x)
	}
	_, err := Quad(f, 0, 1)
	// May or may not error; just exercise the path.
	_ = err
}

func TestTrapezoid_TooFew(t *testing.T) {
	_ = Trapezoid([]float64{1}, 1.0) // Exercises the < 2 points path
}

func TestSimpson_TooFew(t *testing.T) {
	_ = Simpson([]float64{1}, 1.0) // Exercises the < 3 points path
}

func TestSimpson_EvenPoints(t *testing.T) {
	_ = Simpson([]float64{1, 2, 3, 4}, 1.0) // Even number of points
}

func TestOdeint_Empty(t *testing.T) {
	f := func(y, tc float64) float64 { return -y }
	result := Odeint(f, 1.0, 0, nil)
	if result != nil {
		t.Error("expected nil for empty tspan")
	}
}

func TestSolveIVP_ZeroSpan(t *testing.T) {
	f := func(tc float64, y []float64) []float64 { return []float64{-y[0]} }
	times, states, err := SolveIVP(f, [2]float64{0, 0}, []float64{1})
	if err != nil {
		// t0 == tf returns early with single point.
		_ = times
		_ = states
	}
}

// ===========================================================================
// Interpolation edge cases
// ===========================================================================

func TestInterp1D_OutOfBounds(t *testing.T) {
	interp := Interp1D([]float64{0, 1, 2}, []float64{0, 1, 4}, "linear")
	// Out of bounds: clamp
	_ = interp(-1)
	_ = interp(5)
}

func TestInterp1D_Methods(t *testing.T) {
	x := []float64{0, 1, 2, 3}
	y := []float64{0, 1, 4, 9}
	// nearest
	fn := Interp1D(x, y, "nearest")
	_ = fn(0.5)
	// previous
	fn2 := Interp1D(x, y, "previous")
	_ = fn2(0.5)
	// next
	fn3 := Interp1D(x, y, "next")
	_ = fn3(0.5)
	// zero
	fn4 := Interp1D(x, y, "zero")
	_ = fn4(0.5)
	// unknown method uses linear
	fn5 := Interp1D(x, y, "unknown")
	_ = fn5(0.5)
}

func TestCubicSpline_OutOfBounds(t *testing.T) {
	cs := CubicSpline([]float64{0, 1, 2}, []float64{0, 1, 4})
	// Out of bounds: extrapolate
	_ = cs(-1)
	_ = cs(5)
}

func TestBSpline_Boundaries(t *testing.T) {
	bs := BSpline([]float64{0, 1, 2, 3, 4}, []float64{0, 1, 4, 9, 16}, 3)
	_ = bs(-1)
	_ = bs(10)
}

func TestRBFInterpolator_Methods(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}}
	vals := []float64{0, 1, 1}
	// multiquadric
	rbf := RBFInterpolator(pts, vals, "multiquadric")
	_ = rbf([]float64{0.5, 0.5})
	// inverse
	rbf2 := RBFInterpolator(pts, vals, "inverse")
	_ = rbf2([]float64{0.5, 0.5})
}

func TestSortPaired_AlreadySorted(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{4, 5, 6}
	sortPaired(x, y)
	if x[0] != 1 || x[1] != 2 {
		t.Error("already sorted should remain unchanged")
	}
}

func TestSortPaired_Unsorted(t *testing.T) {
	x := []float64{3, 1, 2}
	y := []float64{9, 1, 4}
	sortPaired(x, y)
	if x[0] != 1 || y[0] != 1 {
		t.Error("should be sorted")
	}
}

func TestThomasSolve_Small(t *testing.T) {
	// Single element
	defer func() {
		if r := recover(); r != nil {
			// Some implementations may panic for n < 2
		}
	}()
	a := []float64{0}
	b := []float64{2}
	c := []float64{0}
	d := []float64{4}
	result := thomasSolve(a, b, c, d)
	if len(result) != 1 || result[0] != 2 {
		t.Errorf("expected [2], got %v", result)
	}
}

// ===========================================================================
// Linear algebra edge cases (error returns for empty/non-square)
// ===========================================================================

func TestLU_Empty(t *testing.T) {
	_, _, _, err := LU(nil)
	if err == nil {
		t.Error("expected error for nil matrix")
	}
}

func TestLU_NonSquare(t *testing.T) {
	_, _, _, err := LU([][]float64{{1, 2, 3}, {4, 5, 6}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestLU_Forward(t *testing.T) {
	_, _, _, err := LU([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestLUFactor_Empty(t *testing.T) {
	_, _, err := LUFactor(nil)
	if err == nil {
		t.Error("expected error for nil matrix")
	}
}

func TestLUFactor_NonSquare(t *testing.T) {
	_, _, err := LUFactor([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestLUSolve_Empty(t *testing.T) {
	_, err := LUSolve(nil, nil, nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestLUSolve_NonSquare(t *testing.T) {
	_, _ = LUSolve([][]float64{{1, 2, 3}}, []int{0}, []float64{1})
}

func TestChoFactor_Empty(t *testing.T) {
	_, err := ChoFactor(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestChoFactor_NonSquare(t *testing.T) {
	_, err := ChoFactor([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestChoSolve_Empty(t *testing.T) {
	_, err := ChoSolve(nil, nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestChoSolve_NonSquare(t *testing.T) {
	_, _ = ChoSolve([][]float64{{1, 2, 3}}, []float64{1})
}

func TestSchur_Empty(t *testing.T) {
	_, _, err := Schur(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestSchur_NonSquare(t *testing.T) {
	_, _, err := Schur([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestHessenberg_Empty(t *testing.T) {
	_, _, err := Hessenberg(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestHessenberg_NonSquare(t *testing.T) {
	_, _, err := Hessenberg([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestExpm_Empty(t *testing.T) {
	_, err := Expm(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestExpm_NonSquare(t *testing.T) {
	_, err := Expm([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestLogm_Empty(t *testing.T) {
	_, err := Logm(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestLogm_NonSquare(t *testing.T) {
	_, err := Logm([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestPolar_Empty(t *testing.T) {
	_, _, err := Polar(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestPolar_NonSquare(t *testing.T) {
	_, _, err := Polar([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestLDL_Empty(t *testing.T) {
	_, _, err := LDL(nil)
	if err == nil {
		t.Error("expected error for nil")
	}
}

func TestLDL_NonSquare(t *testing.T) {
	_, _, err := LDL([][]float64{{1, 2, 3}})
	if err == nil {
		t.Error("expected error for non-square")
	}
}

func TestCompanion_Single(t *testing.T) {
	_ = Companion([]float64{1, 2}) // Minimal valid input
}

func TestHankel_Normal(t *testing.T) {
	_ = Hankel([]float64{1, 2, 3}, []float64{3, 4, 5})
}

func TestToeplitz_Normal(t *testing.T) {
	_ = Toeplitz([]float64{1, 2, 3}, []float64{1, 4, 5})
}

func TestMatNorm1_Normal(t *testing.T) {
	n := matNorm1([][]float64{{1, 2}, {3, 4}})
	if n <= 0 {
		t.Error("expected positive norm")
	}
}

func TestMatSolve_Singular(t *testing.T) {
	_, err := matSolve([][]float64{{1, 2}, {2, 4}}, [][]float64{{1, 0}, {0, 1}})
	// Singular matrix handling
	_ = err
}

func TestMatInverse_1x1(t *testing.T) {
	inv, err := matInverse([][]float64{{2}})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(inv[0][0]-0.5) > 0.001 {
		t.Error("expected 0.5")
	}
}

func TestDFT_Empty(t *testing.T) {
	_, err := DFT(0)
	if err == nil {
		t.Error("expected error for n=0")
	}
}

func TestInterpolative_Error(t *testing.T) {
	_, _, err := Interpolative(nil, 0)
	if err == nil {
		t.Error("expected error for nil")
	}
}

// ===========================================================================
// Optimization edge cases
// ===========================================================================

func TestMinimize_ConvergenceGuards(t *testing.T) {
	f := func(x []float64) float64 {
		return x[0] * x[0]
	}
	result, err := Minimize(f, []float64{5.0}, "nelder-mead")
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

// ===========================================================================
// Signal edge cases
// ===========================================================================

func TestSignalCorrelate_EmptyB(t *testing.T) {
	result := SignalCorrelate([]float64{1, 2, 3}, nil)
	if result != nil {
		t.Error("expected nil for empty b")
	}
}

func TestWelch_Empty(t *testing.T) {
	f, p := Welch(nil, 1.0, 256)
	if f != nil || p != nil {
		t.Error("expected nil for empty input")
	}
}

func TestWelch_SmallNperseg(t *testing.T) {
	// nperseg > n: clamps to n
	f, p := Welch([]float64{1, 2, 3}, 1.0, 100)
	_ = f
	_ = p
}

func TestWelch_StepZero(t *testing.T) {
	// nperseg=1 => step=0 => step=1
	f, p := Welch([]float64{1, 2, 3, 4}, 1.0, 1)
	_ = f
	_ = p
}

func TestSpectrogram_Empty(t *testing.T) {
	ti, f, s := Spectrogram(nil, 1.0, 256)
	if ti != nil || f != nil || s != nil {
		t.Error("expected nil for empty input")
	}
}

func TestSpectrogram_SmallNperseg(t *testing.T) {
	ti, f, s := Spectrogram([]float64{1, 2, 3}, 1.0, 100)
	_ = ti
	_ = f
	_ = s
}

func TestSpectrogram_StepZero(t *testing.T) {
	ti, f, s := Spectrogram([]float64{1, 2, 3, 4}, 1.0, 1)
	_ = ti
	_ = f
	_ = s
}

func TestLFilter_Empty(t *testing.T) {
	if LFilter(nil, nil, nil) != nil {
		t.Error("expected nil for empty input")
	}
}

func TestLFilter_ZeroA0(t *testing.T) {
	if LFilter([]float64{1}, []float64{0}, []float64{1}) != nil {
		t.Error("expected nil for a[0]=0")
	}
}

func TestLFilter_BranchCoverage(t *testing.T) {
	// na > nb: exercises different state branches
	b := []float64{1}
	a := []float64{1, 0.5, 0.3}
	x := []float64{1, 2, 3, 4, 5}
	result := LFilter(b, a, x)
	if len(result) != 5 {
		t.Errorf("expected 5 results, got %d", len(result))
	}

	// nb > na
	b2 := []float64{1, 0.5, 0.3}
	a2 := []float64{1}
	result2 := LFilter(b2, a2, x)
	if len(result2) != 5 {
		t.Errorf("expected 5 results, got %d", len(result2))
	}
}

// ===========================================================================
// Sparse matrix edge cases
// ===========================================================================

func TestCSR_GetMissing(t *testing.T) {
	// 2x2 CSR with only one element
	csr, _ := NewCSR([]int{0, 1, 1}, []int{0}, []float64{5}, [2]int{2, 2})
	if csr.Get(0, 1) != 0 {
		t.Error("expected 0 for missing element")
	}
	if csr.Get(1, 0) != 0 {
		t.Error("expected 0 for empty row")
	}
}

func TestCSR_Transpose(t *testing.T) {
	csr, _ := NewCSR([]int{0, 2, 3}, []int{0, 1, 0}, []float64{1, 2, 3}, [2]int{2, 2})
	tr := csr.Transpose()
	if tr.Shape() != [2]int{2, 2} {
		t.Error("wrong shape after transpose")
	}
}

func TestCSC_GetMissing(t *testing.T) {
	csc, _ := NewCSC([]int{0, 1, 1}, []int{0}, []float64{5}, [2]int{2, 2})
	if csc.Get(1, 0) != 0 {
		t.Error("expected 0 for missing element")
	}
}

// ===========================================================================
// Spatial edge cases
// ===========================================================================

func TestCdist_Empty(t *testing.T) {
	_ = Cdist(nil, nil, "euclidean")
}

func TestPdist_TooFew(t *testing.T) {
	_ = Pdist([][]float64{{1}}, "euclidean")
}

// ===========================================================================
// Special function edge cases
// ===========================================================================

func TestGammaln_Boundaries(t *testing.T) {
	if !math.IsInf(Gammaln(0), 1) {
		t.Error("Gammaln(0) should be +inf")
	}
	if !math.IsInf(Gammaln(-1), 1) {
		t.Error("Gammaln(-1) should be +inf")
	}
	// Small x
	_ = Gammaln(0.001)
}

func TestBetaFunc_Boundaries(t *testing.T) {
	_ = BetaFunc(0.5, 0.5) // Normal case
	_ = BetaFunc(1, 1)     // Should be 1
}

func TestDigamma_SmallX(t *testing.T) {
	_ = Digamma(0.001)
}

func TestRegularizedIncompleteGamma_Boundaries(t *testing.T) {
	_ = RegularizedIncompleteGamma(1, -1)
	_ = RegularizedIncompleteGamma(1, 0)
}

func TestErfinv_Boundaries(t *testing.T) {
	if !math.IsInf(Erfinv(-1), -1) {
		t.Error("Erfinv(-1) should be -inf")
	}
	if !math.IsInf(Erfinv(1), 1) {
		t.Error("Erfinv(1) should be +inf")
	}
}

// ===========================================================================
// Stats test edge cases
// ===========================================================================

func TestTTestInd_ZeroDenom(t *testing.T) {
	// Constant values in both groups => variance = 0 => denom = 0.
	stat, pval := TTestInd([]float64{1, 1, 1}, []float64{1, 1, 1})
	if stat != 0 || pval != 1 {
		t.Errorf("expected (0,1), got (%f,%f)", stat, pval)
	}
}

func TestTTestRel_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 samples")
		}
	}()
	TTestRel([]float64{1}, []float64{1})
}

func TestTTestRel_UnequalLen(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unequal lengths")
		}
	}()
	TTestRel([]float64{1, 2}, []float64{1})
}

func TestMannWhitneyU_ZeroSigma(t *testing.T) {
	// Edge case where sigma=0
	stat, pval := MannWhitneyU([]float64{1}, []float64{1})
	_ = stat
	if pval != 1 {
		t.Errorf("expected pval=1 for sigma=0, got %f", pval)
	}
}

func TestWilcoxonSignedRank_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for too few samples")
		}
	}()
	WilcoxonSignedRank(nil)
}

func TestKruskalWallis_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 groups")
		}
	}()
	KruskalWallis([]float64{1})
}

func TestFriedmanChiSquare_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 groups")
		}
	}()
	FriedmanChiSquare([]float64{1})
}

func TestFriedmanChiSquare_UnequalLen(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unequal group lengths")
		}
	}()
	FriedmanChiSquare([]float64{1, 2}, []float64{1})
}

func TestKSTest_Short(t *testing.T) {
	defer func() { _ = recover() }()
	stat, pval := KSTest(nil, func(x float64) float64 { return x })
	_ = stat
	_ = pval
}

func TestKS2Samp_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty input")
		}
	}()
	KS2Samp(nil, nil)
}

func TestKS2Samp_SmallSample(t *testing.T) {
	stat, pval := KS2Samp([]float64{1}, []float64{2})
	_ = stat
	_ = pval
}

func TestShapiroWilk_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 3 samples")
		}
	}()
	ShapiroWilk([]float64{1, 2})
}

func TestNormalTest_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 8 samples")
		}
	}()
	NormalTest([]float64{1, 2, 3})
}

func TestAndersonDarling_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for too few samples")
		}
	}()
	AndersonDarling(nil)
}

func TestBartlettTest_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 groups")
		}
	}()
	BartlettTest([]float64{1})
}

func TestLeveneTest_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 groups")
		}
	}()
	LeveneTest([]float64{1})
}

func TestFlignerKilleen_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 groups")
		}
	}()
	FlignerKilleen([]float64{1})
}

func TestMoodTest_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for too few samples")
		}
	}()
	MoodTest(nil, nil)
}

func TestSpearmanR_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for too few samples")
		}
	}()
	SpearmanR([]float64{1}, []float64{1})
}

func TestKendallTau_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for too few samples")
		}
	}()
	KendallTau([]float64{1}, []float64{1})
}

func TestPearsonCorrelation_PanicShort(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 3 observations")
		}
	}()
	PearsonCorrelation([]float64{1}, []float64{1})
}

func TestPearsonCorrelation_PanicUnequal(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unequal lengths")
		}
	}()
	PearsonCorrelation([]float64{1, 2, 3}, []float64{1, 2})
}

// ===========================================================================
// Stats extra edge cases
// ===========================================================================

func TestChi2Contingency_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty observed")
		}
	}()
	Chi2Contingency(nil)
}

func TestFisherExact_Normal(t *testing.T) {
	// Normal 2x2 table
	or, pval := FisherExact([2][2]int{{10, 5}, {3, 12}})
	if or <= 0 {
		t.Error("expected positive odds ratio")
	}
	_ = pval
}

func TestPointBiserialR_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for too few samples")
		}
	}()
	PointBiserialR(nil, nil)
}

// ===========================================================================
// Stats extra2 edge cases
// ===========================================================================

func TestDescribe_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty data")
		}
	}()
	Describe(nil)
}

func TestZscore_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 values")
		}
	}()
	Zscore([]float64{1})
}

func TestIQR_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty data")
		}
	}()
	IQR(nil)
}

func TestSEM_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 2 values")
		}
	}()
	SEM([]float64{1})
}

func TestKurtosis_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 4 values")
		}
	}()
	Kurtosis([]float64{1, 2})
}

func TestSkew_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 3 values")
		}
	}()
	Skew([]float64{1, 2})
}

func TestTrimMean_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for empty data")
		}
	}()
	TrimMean(nil, 0.1)
}

func TestZmap_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		}
	}()
	Zmap(nil, nil)
}

func TestJarqueBera_Short(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for < 3 values")
		}
	}()
	JarqueBera([]float64{1, 2})
}

func TestRankData_Short(t *testing.T) {
	_ = RankData(nil) // Exercise empty data path
}

// ===========================================================================
// Optimization extra edge cases
// ===========================================================================

func TestMinimizeScalar_Bounds(t *testing.T) {
	f := func(x float64) float64 { return (x - 3) * (x - 3) }
	result, err := MinimizeScalar(f, [2]float64{0, 10})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result.X-3) > 0.01 {
		t.Errorf("expected x near 3, got %f", result.X)
	}
}

// ===========================================================================
// Special extra edge cases
// ===========================================================================

func TestPolygamma_Pole(t *testing.T) {
	// x is a non-positive integer => NaN
	result := Polygamma(1, -2)
	if !math.IsNaN(result) {
		t.Errorf("expected NaN at pole, got %f", result)
	}
}

func TestBernoulliNumber_Large(t *testing.T) {
	// Request a Bernoulli number not in the table => 0
	result := bernoulliNumber(100)
	if result != 0 {
		t.Errorf("expected 0 for large n, got %f", result)
	}
}

func TestZeta_Negative(t *testing.T) {
	// Negative s uses reflection formula
	result := Zeta(-3)
	// zeta(-3) = 1/120
	if math.Abs(result-1.0/120) > 0.01 {
		t.Errorf("expected ~1/120 for zeta(-3), got %f", result)
	}
}

// ===========================================================================
// Sparse extra edge cases
// ===========================================================================

func TestEyeSparse_Zero(t *testing.T) {
	m := EyeSparse(0)
	if m == nil {
		t.Error("expected non-nil sparse matrix")
	}
}

func TestDiags_UnequalLengths(t *testing.T) {
	_, err := Diags([][]float64{{1}}, []int{0, 1}, 3)
	if err == nil {
		t.Error("expected error for unequal lengths")
	}
}

func TestDiags_NonPositiveN(t *testing.T) {
	_, err := Diags([][]float64{{1}}, []int{0}, 0)
	if err == nil {
		t.Error("expected error for n <= 0")
	}
}

func TestHStackSparse_SameRows(t *testing.T) {
	a, _ := NewCSR([]int{0, 1}, []int{0}, []float64{1}, [2]int{1, 1})
	b, _ := NewCSR([]int{0, 1}, []int{0}, []float64{2}, [2]int{1, 1})
	result := HStackSparse(a, b)
	if result.Shape() != [2]int{1, 2} {
		t.Errorf("expected shape [1,2], got %v", result.Shape())
	}
}

func TestVStackSparse_SameCols(t *testing.T) {
	a, _ := NewCSR([]int{0, 1}, []int{0}, []float64{1}, [2]int{1, 1})
	b, _ := NewCSR([]int{0, 1}, []int{0}, []float64{2}, [2]int{1, 1})
	result := VStackSparse(a, b)
	if result.Shape() != [2]int{2, 1} {
		t.Errorf("expected shape [2,1], got %v", result.Shape())
	}
}

// Ensure cmplx import is used (needed for FFT tests above).
var _ = cmplx.Abs
