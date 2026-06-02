//go:build unit

package ci_tests

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeContinuousDFFromMap creates a DataFrame from column name -> []float64 data.
// (Duplicated here since the continuous_test.go version is not exported.)
func makeContinuousDFFromMap(cols map[string][]float64) *tabgo.DataFrame {
	series := make(map[string]*tabgo.Series)
	for name, vals := range cols {
		anyVals := make([]any, len(vals))
		for i, v := range vals {
			anyVals[i] = v
		}
		series[name] = tabgo.NewSeries(name, anyVals)
	}
	return tabgo.NewDataFrame(series)
}

// ============================================================
// Pearsonr Tests
// ============================================================

func TestPearsonr_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := Pearsonr("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("Pearsonr should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("Pearsonr p-value should be < 0.05 for dependent data, got %f", pvalue)
	}
}

func TestPearsonr_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := Pearsonr("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("Pearsonr should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestPearsonr_ConditionalIndependence(t *testing.T) {
	n := 500
	rng := rand.New(rand.NewSource(123))
	zData := make([]float64, n)
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		zData[i] = rng.NormFloat64()
		xData[i] = zData[i]*0.8 + rng.NormFloat64()*0.5
		yData[i] = zData[i]*0.8 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData, "z": zData})

	_, pvalMarginal, indepMarginal := Pearsonr("x", "y", nil, df, 0.05)
	if indepMarginal {
		t.Errorf("Pearsonr should detect marginal dependence, pvalue=%f", pvalMarginal)
	}

	_, pvalCond, indepCond := Pearsonr("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("Pearsonr should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestPearsonr_TooFewSamples(t *testing.T) {
	xData := []float64{1, 2}
	yData := []float64{3, 4}
	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indep := Pearsonr("x", "y", nil, df, 0.05)
	if !indep {
		t.Error("Pearsonr should return independent=true with too few samples")
	}
}

func TestPearsonr_AgreesWithFisherZ(t *testing.T) {
	// Both should reach the same independence conclusion on the same data.
	n := 300
	rng := rand.New(rand.NewSource(77))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.5 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indepP := Pearsonr("x", "y", nil, df, 0.05)
	_, _, indepF := FisherZ("x", "y", nil, df, 0.05)

	if indepP != indepF {
		t.Errorf("Pearsonr and FisherZ should agree: Pearsonr independent=%v, FisherZ independent=%v", indepP, indepF)
	}
}

// ============================================================
// PearsonrEquivalence Tests
// ============================================================

func TestPearsonrEquivalence_IndependentData(t *testing.T) {
	n := 500
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	// With a reasonable epsilon, should declare equivalence (independence).
	_, pvalue, indep := PearsonrEquivalence(0.15)("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("PearsonrEquivalence should declare independence for uncorrelated data, pvalue=%f", pvalue)
	}
}

func TestPearsonrEquivalence_DependentData(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	// With a small epsilon, should not declare equivalence.
	_, _, indep := PearsonrEquivalence(0.1)("x", "y", nil, df, 0.05)
	if indep {
		t.Error("PearsonrEquivalence should not declare independence for strongly correlated data")
	}
}

func TestPearsonrEquivalence_TooFewSamples(t *testing.T) {
	xData := []float64{1, 2}
	yData := []float64{3, 4}
	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indep := PearsonrEquivalence(0.1)("x", "y", nil, df, 0.05)
	if !indep {
		t.Error("PearsonrEquivalence should return independent=true with too few samples")
	}
}

func TestPearsonrEquivalence_CompileTimeCheck(t *testing.T) {
	var _ CITest = PearsonrEquivalence(0.05)
}

// ============================================================
// GCM Tests
// ============================================================

func TestGCM_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := GCM("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("GCM should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestGCM_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := GCM("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("GCM should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestGCM_ConditionalIndependence(t *testing.T) {
	n := 500
	rng := rand.New(rand.NewSource(123))
	zData := make([]float64, n)
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		zData[i] = rng.NormFloat64()
		xData[i] = zData[i]*0.8 + rng.NormFloat64()*0.5
		yData[i] = zData[i]*0.8 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData, "z": zData})

	// Without conditioning: should detect dependence.
	_, pval, indep := GCM("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("GCM should detect marginal dependence, pvalue=%f", pval)
	}

	// Conditioning on z: should find conditional independence.
	_, pvalCond, indepCond := GCM("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("GCM should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestGCM_MultipleConditioning(t *testing.T) {
	n := 500
	rng := rand.New(rand.NewSource(456))
	z1Data := make([]float64, n)
	z2Data := make([]float64, n)
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		z1Data[i] = rng.NormFloat64()
		z2Data[i] = rng.NormFloat64()
		xData[i] = z1Data[i]*0.6 + z2Data[i]*0.4 + rng.NormFloat64()*0.3
		yData[i] = z1Data[i]*0.5 + z2Data[i]*0.5 + rng.NormFloat64()*0.3
	}

	df := makeContinuousDFFromMap(map[string][]float64{
		"x": xData, "y": yData, "z1": z1Data, "z2": z2Data,
	})

	_, pvalCond, indepCond := GCM("x", "y", []string{"z1", "z2"}, df, 0.05)
	if !indepCond {
		t.Errorf("GCM should find conditional independence given z1,z2, pvalue=%f", pvalCond)
	}
}

func TestGCM_TooFewSamples(t *testing.T) {
	xData := []float64{1, 2, 3}
	yData := []float64{4, 5, 6}
	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indep := GCM("x", "y", nil, df, 0.05)
	if !indep {
		t.Error("GCM should return independent=true with too few samples")
	}
}

// ============================================================
// Helper Tests
// ============================================================

func TestResiduals_NoPredictors(t *testing.T) {
	target := []float64{1, 2, 3, 4, 5}
	res := residuals(target, nil)

	// Should be centered: mean = 3
	mean := 0.0
	for _, v := range res {
		mean += v
	}
	mean /= float64(len(res))

	if math.Abs(mean) > 1e-10 {
		t.Errorf("residuals with no predictors should have zero mean, got %f", mean)
	}

	// res[0] should be 1-3 = -2
	if math.Abs(res[0]-(-2)) > 1e-10 {
		t.Errorf("residuals[0]: expected -2, got %f", res[0])
	}
}

func TestResiduals_PerfectFit(t *testing.T) {
	// y = 2*x + 1, so residuals should be ~0.
	n := 100
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = float64(i)
		y[i] = 2*x[i] + 1
	}

	res := residuals(y, [][]float64{x})

	for i, r := range res {
		if math.Abs(r) > 1e-8 {
			t.Errorf("residual[%d] should be ~0 for perfect fit, got %f", i, r)
		}
	}
}

func TestSolveLinearSystem(t *testing.T) {
	// 2x + y = 5
	// x + 3y = 7
	// Solution: x=1.6, y=1.8
	A := []float64{2, 1, 1, 3}
	b := []float64{5, 7}
	sol := solveLinearSystem(A, b, 2)

	if math.Abs(sol[0]-1.6) > 1e-10 || math.Abs(sol[1]-1.8) > 1e-10 {
		t.Errorf("expected [1.6, 1.8], got [%f, %f]", sol[0], sol[1])
	}
}

func TestFSurvival(t *testing.T) {
	// F(0, d1, d2) = 1
	if v := fSurvival(0, 1, 10); math.Abs(v-1) > 1e-10 {
		t.Errorf("fSurvival(0, 1, 10) should be 1, got %f", v)
	}

	// For very large F, survival should be near 0.
	if v := fSurvival(1000, 1, 10); v > 0.001 {
		t.Errorf("fSurvival(1000, 1, 10) should be very small, got %f", v)
	}

	// F with moderate value: check it's between 0 and 1.
	v := fSurvival(4.0, 1, 20)
	if v < 0 || v > 1 {
		t.Errorf("fSurvival(4.0, 1, 20) should be in [0,1], got %f", v)
	}
}
