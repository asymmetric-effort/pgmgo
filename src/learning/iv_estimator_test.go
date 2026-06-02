//go:build unit

package learning

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// generateIVData generates synthetic IV data:
//
//	Z -> X -> Y with a confounder U affecting both X and Y.
//	Z = instrument, X = treatment, Y = outcome, U = unobserved confounder.
//	X = 1.0 + 0.8*Z + 0.5*U + noise
//	Y = 2.0 + 3.0*X + 0.7*U + noise
//	True causal effect of X on Y is 3.0.
func generateIVData(rng *rand.Rand, n int) *tabgo.DataFrame {
	zVals := make([]any, n)
	xVals := make([]any, n)
	yVals := make([]any, n)

	for i := 0; i < n; i++ {
		z := rng.NormFloat64()
		u := rng.NormFloat64()
		x := 1.0 + 0.8*z + 0.5*u + 0.1*rng.NormFloat64()
		y := 2.0 + 3.0*x + 0.7*u + 0.1*rng.NormFloat64()

		zVals[i] = z
		xVals[i] = x
		yVals[i] = y
	}

	return tabgo.NewDataFrame(map[string]*tabgo.Series{
		"Z": tabgo.NewSeries("Z", zVals),
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})
}

func TestNewIVEstimator(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	if iv == nil {
		t.Fatal("expected non-nil IVEstimator")
	}
	if iv.fitted {
		t.Error("new estimator should not be fitted")
	}
}

func TestIVEstimator_Fit_CausalEffect(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	data := generateIVData(rng, 50000)

	iv := NewIVEstimator("X", "Y", []string{"Z"})
	if err := iv.Fit(data); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// True causal effect is 3.0.
	ate := iv.ATE()
	assertClose(t, "ATE", ate, 3.0, 0.15)
}

func TestIVEstimator_Fit_MultipleInstruments(t *testing.T) {
	// Two instruments Z1 and Z2.
	rng := rand.New(rand.NewSource(123))
	n := 30000
	z1Vals := make([]any, n)
	z2Vals := make([]any, n)
	xVals := make([]any, n)
	yVals := make([]any, n)

	for i := 0; i < n; i++ {
		z1 := rng.NormFloat64()
		z2 := rng.NormFloat64()
		u := rng.NormFloat64()
		x := 0.5 + 0.6*z1 + 0.4*z2 + 0.3*u + 0.1*rng.NormFloat64()
		y := 1.0 + 2.0*x + 0.5*u + 0.1*rng.NormFloat64()

		z1Vals[i] = z1
		z2Vals[i] = z2
		xVals[i] = x
		yVals[i] = y
	}

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"Z1": tabgo.NewSeries("Z1", z1Vals),
		"Z2": tabgo.NewSeries("Z2", z2Vals),
		"X":  tabgo.NewSeries("X", xVals),
		"Y":  tabgo.NewSeries("Y", yVals),
	})

	iv := NewIVEstimator("X", "Y", []string{"Z1", "Z2"})
	if err := iv.Fit(data); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	assertClose(t, "ATE", iv.ATE(), 2.0, 0.15)
}

func TestIVEstimator_Fit_NilData(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	err := iv.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestIVEstimator_Fit_NoInstruments(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{})
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0}),
		"Y": tabgo.NewSeries("Y", []any{3.0, 4.0}),
	})
	err := iv.Fit(data)
	if err == nil {
		t.Error("expected error for no instruments")
	}
}

func TestIVEstimator_ATE_NotFitted(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	if iv.ATE() != 0 {
		t.Errorf("expected 0 ATE before fitting, got %f", iv.ATE())
	}
}

func TestIVEstimator_OLS_Biased(t *testing.T) {
	// Verify that naive OLS is biased in the presence of confounding,
	// while IV estimation is not.
	rng := rand.New(rand.NewSource(77))
	data := generateIVData(rng, 50000)

	// OLS (biased due to confounder U).
	ols := NewLinearModel()
	if err := ols.Fit(data, "Y", []string{"X"}); err != nil {
		t.Fatalf("OLS Fit: %v", err)
	}
	olsCoeff := ols.Coefficients()[0]

	// IV (should be closer to 3.0).
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	if err := iv.Fit(data); err != nil {
		t.Fatalf("IV Fit: %v", err)
	}

	// OLS should be biased away from 3.0 (upward due to positive confounding).
	olsBias := math.Abs(olsCoeff - 3.0)
	ivBias := math.Abs(iv.ATE() - 3.0)

	if ivBias >= olsBias {
		t.Errorf("IV estimate (%.4f) should be less biased than OLS (%.4f) for true effect 3.0",
			iv.ATE(), olsCoeff)
	}
}

func TestIVEstimator_InstrumentsCopied(t *testing.T) {
	instruments := []string{"Z1", "Z2"}
	iv := NewIVEstimator("X", "Y", instruments)
	instruments[0] = "modified"
	if iv.instruments[0] == "modified" {
		t.Error("IVEstimator should copy instruments slice")
	}
}
