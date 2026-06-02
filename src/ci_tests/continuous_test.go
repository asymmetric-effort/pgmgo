//go:build unit

package ci_tests

import (
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeContinuousDF creates a DataFrame from column name -> []float64 data.
func makeContinuousDF(cols map[string][]float64) *tabgo.DataFrame {
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

// TestFisherZ_DetectsDependence tests that FisherZ detects dependence
// between strongly correlated continuous variables.
func TestFisherZ_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1 // strong linear dependence
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := FisherZ("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("FisherZ should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("FisherZ p-value should be < 0.05 for dependent data, got %f", pvalue)
	}
}

// TestFisherZ_DetectsIndependence tests that FisherZ does not reject independence
// when variables are generated independently.
func TestFisherZ_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64() // independent
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData})

	stat, pvalue, indep := FisherZ("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("FisherZ should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

// TestFisherZ_ConditionalIndependence tests FisherZ with a confounding variable.
// x and y are both caused by z, so they are marginally dependent but conditionally
// independent given z.
func TestFisherZ_ConditionalIndependence(t *testing.T) {
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

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData, "z": zData})

	// Without conditioning on z: should detect dependence (since z confounds).
	_, pvalMarginal, indepMarginal := FisherZ("x", "y", nil, df, 0.05)
	if indepMarginal {
		t.Errorf("FisherZ should detect marginal dependence (confounded by z), pvalue=%f", pvalMarginal)
	}

	// Conditioning on z: should find conditional independence.
	_, pvalCond, indepCond := FisherZ("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("FisherZ should find conditional independence given z, pvalue=%f", pvalCond)
	}
}

// TestFisherZ_MultipleConditioning tests FisherZ with multiple conditioning variables.
func TestFisherZ_MultipleConditioning(t *testing.T) {
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

	df := makeContinuousDF(map[string][]float64{
		"x": xData, "y": yData, "z1": z1Data, "z2": z2Data,
	})

	// Without conditioning: should detect dependence.
	_, pval, indep := FisherZ("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("FisherZ should detect marginal dependence, pvalue=%f", pval)
	}

	// Conditioning on both z1 and z2: should find conditional independence.
	_, pvalCond, indepCond := FisherZ("x", "y", []string{"z1", "z2"}, df, 0.05)
	if !indepCond {
		t.Errorf("FisherZ should find conditional independence given z1,z2, pvalue=%f", pvalCond)
	}
}

// TestFisherZ_TooFewSamples tests that FisherZ returns independent=true when
// there are too few samples for valid inference.
func TestFisherZ_TooFewSamples(t *testing.T) {
	// With 2 conditioning variables and only 5 data points, n=5 <= k+3=5.
	xData := []float64{1, 2, 3, 4, 5}
	yData := []float64{2, 4, 6, 8, 10}
	z1Data := []float64{0, 1, 0, 1, 0}
	z2Data := []float64{1, 0, 1, 0, 1}

	df := makeContinuousDF(map[string][]float64{
		"x": xData, "y": yData, "z1": z1Data, "z2": z2Data,
	})

	_, _, indep := FisherZ("x", "y", []string{"z1", "z2"}, df, 0.05)
	if !indep {
		t.Errorf("FisherZ should return independent=true when n is too small for valid test")
	}
}
