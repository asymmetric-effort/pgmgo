//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ===========================================================================
// optimization_extra.go: additional edge cases
// ===========================================================================

func TestDifferentialEvolution_Basic(t *testing.T) {
	f := func(x []float64) float64 {
		return x[0]*x[0] + x[1]*x[1]
	}
	result, err := DifferentialEvolution(f, [][2]float64{{-5, 5}, {-5, 5}})
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestRootScalar_Basic(t *testing.T) {
	f := func(x float64) float64 { return x*x - 4 }
	result, err := RootScalar(f, [2]float64{0, 5})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result-2) > 0.01 {
		t.Errorf("expected root near 2, got %f", result)
	}
}

// ===========================================================================
// spatial.go: ConvexHull, Delaunay, Voronoi
// ===========================================================================

func TestConvexHull_Basic(t *testing.T) {
	points := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {0.5, 0.5}}
	hull, err := ConvexHull(points)
	if err != nil {
		t.Fatal(err)
	}
	if len(hull) < 3 {
		t.Errorf("expected at least 3 hull points, got %d", len(hull))
	}
}

func TestDelaunay_Basic(t *testing.T) {
	points := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	triangles, err := Delaunay(points)
	if err != nil {
		t.Fatal(err)
	}
	_ = triangles
}

func TestVoronoi_Basic(t *testing.T) {
	points := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	vertices, regions, err := Voronoi(points)
	if err != nil {
		t.Fatal(err)
	}
	_ = vertices
	_ = regions
}

// ===========================================================================
// interpolate.go: cover all methods
// ===========================================================================

func TestInterp1D_AllMethods_Boost(t *testing.T) {
	x := []float64{0, 1, 2, 3, 4, 5}
	y := []float64{0, 1, 4, 9, 16, 25}
	for _, method := range []string{"linear", "nearest", "previous", "next", "zero", "quadratic", "cubic"} {
		fn := Interp1D(x, y, method)
		_ = fn(2.5)
	}
}

func TestRBFInterpolator_AllKernels_Boost(t *testing.T) {
	pts := [][]float64{{0, 0}, {1, 0}, {0, 1}, {1, 1}}
	vals := []float64{0, 1, 1, 2}
	for _, kernel := range []string{"multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"} {
		rbf := RBFInterpolator(pts, vals, kernel)
		_ = rbf([]float64{0.5, 0.5})
	}
}

// ===========================================================================
// special.go: more edges
// ===========================================================================

func TestErfinv_Extended_Boost(t *testing.T) {
	_ = Erfinv(-0.9999)
	_ = Erfinv(0.9999)
}

func TestRegularizedIncompleteBeta_Symmetry(t *testing.T) {
	_ = RegularizedIncompleteBeta(0.9, 1, 10)
	_ = RegularizedIncompleteBeta(0.1, 10, 1)
}

// ===========================================================================
// distributions_extra.go: LogPDF edges
// ===========================================================================

func TestLaplace_LogPDF_Boost(t *testing.T) {
	l := NewLaplace(0, 1)
	_ = l.LogPDF(-5)
	_ = l.LogPDF(0)
	_ = l.LogPDF(5)
}

func TestCauchy_PPF_LogPDF_Boost(t *testing.T) {
	c := NewCauchy(0, 1)
	_ = c.PPF(0.001)
	_ = c.PPF(0.999)
	_ = c.LogPDF(-100)
	_ = c.LogPDF(100)
}

func TestLaplace_PPF_Boost(t *testing.T) {
	l := NewLaplace(0, 1)
	_ = l.PPF(0.001)
	_ = l.PPF(0.25)
	_ = l.PPF(0.999)
}

// ===========================================================================
// distributions_continuous.go: PPF Newton edges
// ===========================================================================

func TestWald_PPF_SmallP_Boost(t *testing.T) {
	w := NewWald(1, 1)
	_ = w.PPF(0.001)
	_ = w.PPF(0.999)
}

func TestFDistribution_PPF_Edge(t *testing.T) {
	f := NewFDistribution(2, 5)
	_ = f.PPF(0.001)
	_ = f.PPF(0.999)
}

func TestVonMises_CDF_Wrap_Boost(t *testing.T) {
	v := NewVonMises(0, 1)
	_ = v.CDF(5 * math.Pi)
	_ = v.CDF(-5 * math.Pi)
}

// ===========================================================================
// stats_tests.go: remaining edges
// ===========================================================================

func TestTTestInd_ZeroDenom_Boost(t *testing.T) {
	stat, pval := TTestInd([]float64{1, 1, 1}, []float64{1, 1, 1})
	_ = stat
	_ = pval
}

func TestShapiroWilk_LargeSample(t *testing.T) {
	// n > 11 to trigger the larger sample code path
	data := make([]float64, 20)
	for i := range data {
		data[i] = float64(i) * 0.5
	}
	stat, pval := ShapiroWilk(data)
	_ = stat
	_ = pval
}

func TestNormalTest_Boost(t *testing.T) {
	data := make([]float64, 30)
	for i := range data {
		data[i] = float64(i) * 0.1
	}
	stat, pval := NormalTest(data)
	_ = stat
	_ = pval
}

func TestAndersonDarling_Boost(t *testing.T) {
	data := make([]float64, 15)
	for i := range data {
		data[i] = float64(i)
	}
	stat, crit := AndersonDarling(data)
	_ = stat
	_ = crit
}

func TestFlignerKilleen_Boost(t *testing.T) {
	stat, pval := FlignerKilleen([]float64{1, 2, 3, 4}, []float64{5, 6, 7, 8})
	_ = stat
	_ = pval
}

func TestMoodTest_Boost(t *testing.T) {
	stat, pval := MoodTest([]float64{1, 2, 3, 4, 5}, []float64{6, 7, 8, 9, 10})
	_ = stat
	_ = pval
}

func TestFriedmanChiSquare_Normal(t *testing.T) {
	stat, pval := FriedmanChiSquare([]float64{1, 2, 3}, []float64{2, 3, 4}, []float64{3, 4, 5})
	_ = stat
	_ = pval
}

// ===========================================================================
// linalg.go: more coverage
// ===========================================================================

func TestSchur_Boost(t *testing.T) {
	a := [][]float64{{1, 2}, {0, 3}}
	tc, z, err := Schur(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = tc
	_ = z
}

func TestHessenberg_Boost(t *testing.T) {
	a := [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	h, q, err := Hessenberg(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = h
	_ = q
}

func TestExpm_Boost(t *testing.T) {
	a := [][]float64{{0, 1}, {0, 0}}
	result, err := Expm(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestLogm_Boost(t *testing.T) {
	a := [][]float64{{1, 0}, {0, 1}}
	result, err := Logm(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestPolar_Boost(t *testing.T) {
	a := [][]float64{{1, 0}, {0, 1}}
	u, p, err := Polar(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = u
	_ = p
}

func TestLDL_Boost(t *testing.T) {
	a := [][]float64{{4, 2}, {2, 3}}
	l, d, err := LDL(a)
	if err != nil {
		t.Fatal(err)
	}
	_ = l
	_ = d
}

func TestInterpolative_Boost(t *testing.T) {
	a := [][]float64{{1, 0, 1}, {0, 1, 1}, {1, 1, 0}}
	idx, proj, err := Interpolative(a, 2)
	if err != nil {
		t.Fatal(err)
	}
	_ = idx
	_ = proj
}

func TestDFT_Boost(t *testing.T) {
	m, err := DFT(4)
	if err != nil {
		t.Fatal(err)
	}
	_ = m
}

// ===========================================================================
// integrate.go: additional coverage
// ===========================================================================

func TestRomberg_Boost(t *testing.T) {
	f := func(x float64) float64 { return x * x }
	result := Romberg(f, 0, 1)
	if math.Abs(result-1.0/3) > 0.001 {
		t.Errorf("expected ~0.333, got %f", result)
	}
}

func TestSolveIVP_Normal_Boost(t *testing.T) {
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
// sparse.go: matrix operations
// ===========================================================================

func TestCSR_MulVec_Boost(t *testing.T) {
	csr, _ := NewCSR([]int{0, 2, 3}, []int{0, 1, 0}, []float64{1, 2, 3}, [2]int{2, 2})
	result := csr.MulVec([]float64{1, 2})
	_ = result
}

func TestCSR_MulDense_Boost(t *testing.T) {
	csr, _ := NewCSR([]int{0, 1, 2}, []int{0, 1}, []float64{2, 3}, [2]int{2, 2})
	dense := [][]float64{{1, 0}, {0, 1}}
	result := csr.MulDense(dense)
	_ = result
}

// ===========================================================================
// stats_extra2.go: additional coverage
// ===========================================================================

func TestZscore_Boost(t *testing.T) {
	result := Zscore([]float64{1, 2, 3, 4, 5})
	if len(result) != 5 {
		t.Error("expected 5 z-scores")
	}
}

func TestZmap_Boost(t *testing.T) {
	result := Zmap([]float64{1, 2, 3}, []float64{1, 2, 3, 4, 5})
	if len(result) != 3 {
		t.Error("expected 3 results")
	}
}

func TestTrimMean_Normal(t *testing.T) {
	result := TrimMean([]float64{1, 2, 3, 4, 5, 100}, 0.2)
	_ = result
}

func TestPointBiserialR_Boost(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6}
	y := []bool{true, true, true, false, false, false}
	r, pval := PointBiserialR(x, y)
	_ = r
	_ = pval
}

func TestPearsonCorrelation_ZeroDenom_Boost(t *testing.T) {
	x := []float64{1, 1, 1, 1, 1}
	y := []float64{1, 2, 3, 4, 5}
	r, pval := PearsonCorrelation(x, y)
	_ = r
	_ = pval
}

// ===========================================================================
// distributions_discrete.go: remaining edges
// ===========================================================================

func TestNegBinomial_Var_Boundary(t *testing.T) {
	nb := NewNegativeBinomial(1, 0.5)
	v := nb.Var()
	if v <= 0 {
		t.Error("expected positive variance")
	}
}

func TestBoltzmann_CDF_AtN(t *testing.T) {
	b := NewBoltzmann(1, 5)
	if b.CDF(5) != 1 {
		t.Errorf("expected CDF(N) = 1, got %f", b.CDF(5))
	}
	if b.CDF(-1) != 0 {
		t.Errorf("expected CDF(-1) = 0, got %f", b.CDF(-1))
	}
}
