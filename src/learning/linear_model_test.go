//go:build unit

package learning

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestNewLinearModel(t *testing.T) {
	lm := NewLinearModel()
	if lm == nil {
		t.Fatal("expected non-nil LinearModel")
	}
	if lm.fitted {
		t.Error("new model should not be fitted")
	}
}

func TestLinearModel_Fit_SimpleRegression(t *testing.T) {
	// Y = 3 + 2*X (no noise)
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		xVals[i] = x
		yVals[i] = 3.0 + 2.0*x
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	lm := NewLinearModel()
	if err := lm.Fit(data, "Y", []string{"X"}); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	assertClose(t, "intercept", lm.Intercept(), 3.0, 1e-8)
	coeffs := lm.Coefficients()
	if len(coeffs) != 1 {
		t.Fatalf("expected 1 coefficient, got %d", len(coeffs))
	}
	assertClose(t, "coeff[0]", coeffs[0], 2.0, 1e-8)
}

func TestLinearModel_Fit_MultipleRegression(t *testing.T) {
	// Z = 1.0 + 2.0*X + 3.0*Y with noise
	rng := rand.New(rand.NewSource(42))
	n := 10000
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		y := 2.0 + rng.NormFloat64()
		z := 1.0 + 2.0*x + 3.0*y + 0.1*rng.NormFloat64()
		xVals[i] = x
		yVals[i] = y
		zVals[i] = z
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
		"Z": tabgo.NewSeries("Z", zVals),
	})

	lm := NewLinearModel()
	if err := lm.Fit(data, "Z", []string{"X", "Y"}); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	assertClose(t, "intercept", lm.Intercept(), 1.0, 0.1)
	coeffs := lm.Coefficients()
	if len(coeffs) != 2 {
		t.Fatalf("expected 2 coefficients, got %d", len(coeffs))
	}
	assertClose(t, "coeff[X]", coeffs[0], 2.0, 0.05)
	assertClose(t, "coeff[Y]", coeffs[1], 3.0, 0.05)
}

func TestLinearModel_Predict(t *testing.T) {
	// Fit Y = 1 + 2*X
	n := 50
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		xVals[i] = x
		yVals[i] = 1.0 + 2.0*x
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	lm := NewLinearModel()
	if err := lm.Fit(data, "Y", []string{"X"}); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// Predict on new data.
	newX := []any{0.0, 5.0, 10.0}
	newData := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", newX),
	})

	preds, err := lm.Predict(newData)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	expected := []float64{1.0, 11.0, 21.0}
	for i, want := range expected {
		assertClose(t, "prediction", preds[i], want, 1e-8)
	}
}

func TestLinearModel_Predict_NotFitted(t *testing.T) {
	lm := NewLinearModel()
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0}),
	})
	_, err := lm.Predict(data)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestLinearModel_Coefficients_NotFitted(t *testing.T) {
	lm := NewLinearModel()
	if coeffs := lm.Coefficients(); coeffs != nil {
		t.Errorf("expected nil coefficients, got %v", coeffs)
	}
}

func TestLinearModel_Fit_NilData(t *testing.T) {
	lm := NewLinearModel()
	err := lm.Fit(nil, "Y", []string{"X"})
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestLinearModel_Fit_NoPredictors(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"Y": tabgo.NewSeries("Y", []any{1.0}),
	})
	lm := NewLinearModel()
	err := lm.Fit(data, "Y", []string{})
	if err == nil {
		t.Error("expected error for empty predictors")
	}
}

func TestLinearModel_Fit_InsufficientRows(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0}),
	})
	lm := NewLinearModel()
	// Need at least 2 rows for 2 parameters (intercept + 1 beta).
	err := lm.Fit(data, "Y", []string{"X"})
	if err == nil {
		t.Error("expected error for insufficient rows")
	}
}

func TestLinearModel_Predict_NilData(t *testing.T) {
	lm := NewLinearModel()
	lm.fitted = true
	lm.predictors = []string{"X"}
	lm.coefficients = []float64{1.0}
	_, err := lm.Predict(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestLinearModel_Fit_WithNoise(t *testing.T) {
	// Y = 5 + 0.5*X with noise, check recovery.
	rng := rand.New(rand.NewSource(99))
	n := 5000
	xVals := make([]any, n)
	yVals := make([]any, n)
	for i := 0; i < n; i++ {
		x := rng.NormFloat64() * 10
		y := 5.0 + 0.5*x + rng.NormFloat64()*0.5
		xVals[i] = x
		yVals[i] = y
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
	})

	lm := NewLinearModel()
	if err := lm.Fit(data, "Y", []string{"X"}); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	assertClose(t, "intercept", lm.Intercept(), 5.0, 0.1)
	assertClose(t, "coeff", lm.Coefficients()[0], 0.5, 0.02)
}

func TestLinearModel_CoefficientsCopy(t *testing.T) {
	// Ensure Coefficients returns a copy, not the internal slice.
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0, 4.0, 6.0}),
	})
	lm := NewLinearModel()
	if err := lm.Fit(data, "Y", []string{"X"}); err != nil {
		t.Fatalf("Fit: %v", err)
	}
	c1 := lm.Coefficients()
	c2 := lm.Coefficients()
	c1[0] = 999.0
	if math.Abs(c2[0]-999.0) < 1e-9 {
		t.Error("Coefficients should return a copy")
	}
}
