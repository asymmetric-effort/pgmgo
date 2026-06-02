//go:build unit

package ci_tests

import (
	"math"
	"math/rand"
	"testing"
)

// ============================================================
// GeneralizedCov Tests
// ============================================================

func TestGeneralizedCov_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := GeneralizedCov("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("GeneralizedCov should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("GeneralizedCov p-value should be < 0.05 for dependent data, got %f", pvalue)
	}
}

func TestGeneralizedCov_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64() // independent
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := GeneralizedCov("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("GeneralizedCov should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestGeneralizedCov_ConditionalIndependence(t *testing.T) {
	// z confounds x and y: marginally dependent, conditionally independent.
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
	_, pvalMarg, indepMarg := GeneralizedCov("x", "y", nil, df, 0.05)
	if indepMarg {
		t.Errorf("GeneralizedCov should detect marginal dependence, pvalue=%f", pvalMarg)
	}

	// Conditioning on z: should find conditional independence.
	_, pvalCond, indepCond := GeneralizedCov("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("GeneralizedCov should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

func TestGeneralizedCov_MultipleConditioning(t *testing.T) {
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

	_, pvalCond, indepCond := GeneralizedCov("x", "y", []string{"z1", "z2"}, df, 0.05)
	if !indepCond {
		t.Errorf("GeneralizedCov should find conditional independence given z1,z2, pvalue=%f", pvalCond)
	}
}

func TestGeneralizedCov_TooFewSamples(t *testing.T) {
	xData := []float64{1, 2, 3}
	yData := []float64{4, 5, 6}
	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, _, indep := GeneralizedCov("x", "y", nil, df, 0.05)
	if !indep {
		t.Error("GeneralizedCov should return independent=true with too few samples")
	}
}

func TestGeneralizedCov_AgreesWithGCM(t *testing.T) {
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

	_, _, indepGCov := GeneralizedCov("x", "y", nil, df, 0.05)
	_, _, indepGCM := GCM("x", "y", nil, df, 0.05)

	if indepGCov != indepGCM {
		t.Errorf("GeneralizedCov and GCM should agree on strong dependence: GenCov=%v, GCM=%v", indepGCov, indepGCM)
	}
}

func TestGeneralizedCov_PerfectCorrelation(t *testing.T) {
	// y = 2*x + 1 exactly.
	n := 100
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = float64(i) / float64(n)
		yData[i] = 2*xData[i] + 1
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := GeneralizedCov("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("GeneralizedCov should detect perfect correlation: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestGeneralizedCov_StatisticIsNonNegative(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	stat, _, _ := GeneralizedCov("x", "y", nil, df, 0.05)
	if stat < 0 {
		t.Errorf("GeneralizedCov statistic should be non-negative, got %f", stat)
	}
}

func TestGeneralizedCov_PValueInRange(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDFFromMap(map[string][]float64{"x": xData, "y": yData})

	_, pvalue, _ := GeneralizedCov("x", "y", nil, df, 0.05)
	if pvalue < 0 || pvalue > 1 {
		t.Errorf("GeneralizedCov p-value should be in [0,1], got %f", pvalue)
	}
	if math.IsNaN(pvalue) {
		t.Error("GeneralizedCov p-value should not be NaN")
	}
}

func TestCenterSlice(t *testing.T) {
	vals := []float64{1, 2, 3, 4, 5}
	centered := centerSlice(vals)

	// Mean should be zero.
	sum := 0.0
	for _, v := range centered {
		sum += v
	}
	if math.Abs(sum) > 1e-10 {
		t.Errorf("centerSlice should produce zero-mean, got sum=%f", sum)
	}

	// Check specific values: mean=3, so centered = {-2, -1, 0, 1, 2}.
	expected := []float64{-2, -1, 0, 1, 2}
	for i, v := range centered {
		if math.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("centerSlice[%d]: expected %f, got %f", i, expected[i], v)
		}
	}
}
