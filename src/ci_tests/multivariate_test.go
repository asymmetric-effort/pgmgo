//go:build unit

package ci_tests

import (
	"math"
	"math/rand"
	"testing"
)

// ============================================================
// HotellingLawley Tests
// ============================================================

func TestHotellingLawley_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := HotellingLawley("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("HotellingLawley should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("HotellingLawley p-value should be < 0.05, got %f", pvalue)
	}
}

func TestHotellingLawley_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := HotellingLawley("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("HotellingLawley should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestHotellingLawley_ConditionalIndependence(t *testing.T) {
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

	_, pvalCond, indepCond := HotellingLawley("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("HotellingLawley should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestHotellingLawley_TooFewSamples(t *testing.T) {
	xData := []float64{1, 2}
	yData := []float64{3, 4}
	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indep := HotellingLawley("x", "y", nil, df, 0.05)
	if !indep {
		t.Error("HotellingLawley should return independent=true with too few samples")
	}
}

// ============================================================
// PillaiTrace Tests
// ============================================================

func TestPillaiTrace_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := PillaiTrace("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("PillaiTrace should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	// Pillai's trace should be r^2, which should be close to 1 for strongly correlated data.
	if stat < 0.5 {
		t.Errorf("PillaiTrace statistic should be high for strongly correlated data, got %f", stat)
	}
}

func TestPillaiTrace_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := PillaiTrace("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("PillaiTrace should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
	// Pillai's trace should be near 0 for independent data.
	if stat > 0.1 {
		t.Errorf("PillaiTrace statistic should be near 0 for independent data, got %f", stat)
	}
}

func TestPillaiTrace_ConditionalIndependence(t *testing.T) {
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

	_, pvalCond, indepCond := PillaiTrace("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("PillaiTrace should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestPillaiTrace_StatisticRange(t *testing.T) {
	// Pillai's trace (r^2) should be in [0, 1].
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.5 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, _, _ := PillaiTrace("x", "y", nil, df, 0.05)
	if stat < 0 || stat > 1 {
		t.Errorf("PillaiTrace statistic should be in [0,1], got %f", stat)
	}
}

// ============================================================
// RoysLargestRoot Tests
// ============================================================

func TestRoysLargestRoot_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := RoysLargestRoot("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("RoysLargestRoot should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestRoysLargestRoot_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := RoysLargestRoot("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("RoysLargestRoot should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestRoysLargestRoot_ConditionalIndependence(t *testing.T) {
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

	_, pvalCond, indepCond := RoysLargestRoot("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("RoysLargestRoot should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestRoysLargestRoot_StatisticNonnegative(t *testing.T) {
	// Roy's largest root (r^2/(1-r^2)) should be >= 0.
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, _, _ := RoysLargestRoot("x", "y", nil, df, 0.05)
	if stat < 0 {
		t.Errorf("RoysLargestRoot statistic should be >= 0, got %f", stat)
	}
}

// ============================================================
// WilksLambda Tests
// ============================================================

func TestWilksLambda_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := WilksLambda("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("WilksLambda should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	// Wilks' lambda = 1 - r^2, should be near 0 for strongly correlated data.
	if stat > 0.5 {
		t.Errorf("WilksLambda statistic should be near 0 for strongly correlated data, got %f", stat)
	}
}

func TestWilksLambda_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := WilksLambda("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("WilksLambda should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
	// Wilks' lambda should be near 1 for independent data.
	if stat < 0.9 {
		t.Errorf("WilksLambda statistic should be near 1 for independent data, got %f", stat)
	}
}

func TestWilksLambda_ConditionalIndependence(t *testing.T) {
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

	_, pvalCond, indepCond := WilksLambda("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("WilksLambda should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestWilksLambda_StatisticRange(t *testing.T) {
	// Wilks' lambda (1 - r^2) should be in [0, 1].
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.5 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, _, _ := WilksLambda("x", "y", nil, df, 0.05)
	if stat < 0 || stat > 1 {
		t.Errorf("WilksLambda statistic should be in [0,1], got %f", stat)
	}
}

// ============================================================
// Cross-test consistency tests
// ============================================================

func TestMultivariate_AllAgreeOnIndependence(t *testing.T) {
	// All multivariate tests should agree on the independence decision
	// for the same data.
	n := 300
	rng := rand.New(rand.NewSource(77))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.5 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indepHL := HotellingLawley("x", "y", nil, df, 0.05)
	_, _, indepPT := PillaiTrace("x", "y", nil, df, 0.05)
	_, _, indepRR := RoysLargestRoot("x", "y", nil, df, 0.05)
	_, _, indepWL := WilksLambda("x", "y", nil, df, 0.05)

	if indepHL != indepPT || indepPT != indepRR || indepRR != indepWL {
		t.Errorf("All multivariate tests should agree: HL=%v, PT=%v, RR=%v, WL=%v",
			indepHL, indepPT, indepRR, indepWL)
	}
}

func TestMultivariate_AllAgreeOnDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indepHL := HotellingLawley("x", "y", nil, df, 0.05)
	_, _, indepPT := PillaiTrace("x", "y", nil, df, 0.05)
	_, _, indepRR := RoysLargestRoot("x", "y", nil, df, 0.05)
	_, _, indepWL := WilksLambda("x", "y", nil, df, 0.05)

	if indepHL || indepPT || indepRR || indepWL {
		t.Errorf("All tests should detect dependence: HL=%v, PT=%v, RR=%v, WL=%v",
			indepHL, indepPT, indepRR, indepWL)
	}
}

func TestMultivariate_PvaluesConsistent(t *testing.T) {
	// All multivariate tests should produce the same p-value in the single-variable case,
	// since they all reduce to the same F-test.
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.5 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, pHL, _ := HotellingLawley("x", "y", nil, df, 0.05)
	_, pPT, _ := PillaiTrace("x", "y", nil, df, 0.05)
	_, pRR, _ := RoysLargestRoot("x", "y", nil, df, 0.05)
	_, pWL, _ := WilksLambda("x", "y", nil, df, 0.05)

	tol := 1e-10
	if math.Abs(pHL-pPT) > tol {
		t.Errorf("HotellingLawley and PillaiTrace p-values differ: %f vs %f", pHL, pPT)
	}
	if math.Abs(pHL-pRR) > tol {
		t.Errorf("HotellingLawley and RoysLargestRoot p-values differ: %f vs %f", pHL, pRR)
	}
	if math.Abs(pHL-pWL) > tol {
		t.Errorf("HotellingLawley and WilksLambda p-values differ: %f vs %f", pHL, pWL)
	}
}

func TestMultivariate_RelationshipBetweenStatistics(t *testing.T) {
	// Verify the mathematical relationships between statistics:
	// HotellingLawley = F
	// PillaiTrace = r^2
	// RoysLargestRoot = r^2/(1-r^2)
	// WilksLambda = 1-r^2
	// So: PillaiTrace + WilksLambda = 1
	// And: RoysLargestRoot = PillaiTrace / WilksLambda
	// And: HotellingLawley = RoysLargestRoot * df2

	n := 200
	rng := rand.New(rand.NewSource(55))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.6 + rng.NormFloat64()*0.4
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	statHL, _, _ := HotellingLawley("x", "y", nil, df, 0.05)
	statPT, _, _ := PillaiTrace("x", "y", nil, df, 0.05)
	statRR, _, _ := RoysLargestRoot("x", "y", nil, df, 0.05)
	statWL, _, _ := WilksLambda("x", "y", nil, df, 0.05)

	tol := 1e-10

	// PillaiTrace + WilksLambda = 1
	if math.Abs(statPT+statWL-1) > tol {
		t.Errorf("PillaiTrace + WilksLambda should be 1: %f + %f = %f", statPT, statWL, statPT+statWL)
	}

	// RoysLargestRoot = PillaiTrace / WilksLambda
	if statWL > tol {
		expected := statPT / statWL
		if math.Abs(statRR-expected) > tol {
			t.Errorf("RoysLargestRoot should be PillaiTrace/WilksLambda: %f vs %f", statRR, expected)
		}
	}

	// HotellingLawley = RoysLargestRoot * df2 where df2 = n-2-|z| = n-2
	df2 := float64(n - 2)
	expectedHL := statRR * df2
	if math.Abs(statHL-expectedHL) > tol {
		t.Errorf("HotellingLawley should be RoysLargestRoot * df2: %f vs %f", statHL, expectedHL)
	}
}
