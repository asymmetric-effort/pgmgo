//go:build unit

package learning

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// buildTestSEM creates a SEM: X -> Y -> Z with known parameters.
func buildTestSEM(t *testing.T) *models.SEM {
	t.Helper()
	sem := models.NewSEM()
	if err := sem.AddEquation("X", nil, nil, 0, 1.0); err != nil {
		t.Fatalf("AddEquation X: %v", err)
	}
	if err := sem.AddEquation("Y", []string{"X"}, []float64{0.5}, 2.0, 0.5); err != nil {
		t.Fatalf("AddEquation Y: %v", err)
	}
	if err := sem.AddEquation("Z", []string{"Y"}, []float64{1.5}, -1.0, 0.8); err != nil {
		t.Fatalf("AddEquation Z: %v", err)
	}
	return sem
}

// generateSEMData generates data from the test SEM.
func generateSEMData(rng *rand.Rand, n int) *tabgo.DataFrame {
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)

	for i := 0; i < n; i++ {
		x := rng.NormFloat64()
		y := 2.0 + 0.5*x + math.Sqrt(0.5)*rng.NormFloat64()
		z := -1.0 + 1.5*y + math.Sqrt(0.8)*rng.NormFloat64()
		xVals[i] = x
		yVals[i] = y
		zVals[i] = z
	}

	return tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", xVals),
		"Y": tabgo.NewSeries("Y", yVals),
		"Z": tabgo.NewSeries("Z", zVals),
	})
}

func TestNewSEMEstimator(t *testing.T) {
	sem := buildTestSEM(t)
	data := generateSEMData(rand.New(rand.NewSource(42)), 100)
	est := NewSEMEstimator(sem, data)
	if est == nil {
		t.Fatal("expected non-nil SEMEstimator")
	}
}

func TestSEMEstimator_Estimate(t *testing.T) {
	sem := buildTestSEM(t)
	rng := rand.New(rand.NewSource(42))
	data := generateSEMData(rng, 10000)

	est := NewSEMEstimator(sem, data)
	if err := est.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	// Check X equation (root node: intercept ~= 0, variance ~= 1).
	betas, intercept, variance, err := est.GetCoefficients("X")
	if err != nil {
		t.Fatalf("GetCoefficients(X): %v", err)
	}
	if len(betas) != 0 {
		t.Errorf("X should have 0 betas, got %d", len(betas))
	}
	assertClose(t, "X intercept", intercept, 0.0, 0.1)
	assertClose(t, "X variance", variance, 1.0, 0.1)

	// Check Y equation: Y = 2.0 + 0.5*X + eps, var(eps) = 0.5.
	betas, intercept, variance, err = est.GetCoefficients("Y")
	if err != nil {
		t.Fatalf("GetCoefficients(Y): %v", err)
	}
	if len(betas) != 1 {
		t.Fatalf("Y should have 1 beta, got %d", len(betas))
	}
	assertClose(t, "Y beta(X)", betas[0], 0.5, 0.05)
	assertClose(t, "Y intercept", intercept, 2.0, 0.1)
	assertClose(t, "Y variance", variance, 0.5, 0.1)

	// Check Z equation: Z = -1.0 + 1.5*Y + eps, var(eps) = 0.8.
	betas, intercept, variance, err = est.GetCoefficients("Z")
	if err != nil {
		t.Fatalf("GetCoefficients(Z): %v", err)
	}
	if len(betas) != 1 {
		t.Fatalf("Z should have 1 beta, got %d", len(betas))
	}
	assertClose(t, "Z beta(Y)", betas[0], 1.5, 0.05)
	assertClose(t, "Z intercept", intercept, -1.0, 0.2)
	assertClose(t, "Z variance", variance, 0.8, 0.15)
}

func TestSEMEstimator_Estimate_MultipleParents(t *testing.T) {
	// SEM: A, B -> C; C = 1 + 2*A + 3*B + eps
	sem := models.NewSEM()
	_ = sem.AddEquation("A", nil, nil, 0, 1.0)
	_ = sem.AddEquation("B", nil, nil, 0, 1.0)
	_ = sem.AddEquation("C", []string{"A", "B"}, []float64{2.0, 3.0}, 1.0, 0.5)

	rng := rand.New(rand.NewSource(123))
	n := 10000
	aVals := make([]any, n)
	bVals := make([]any, n)
	cVals := make([]any, n)
	for i := 0; i < n; i++ {
		a := rng.NormFloat64()
		b := rng.NormFloat64()
		c := 1.0 + 2.0*a + 3.0*b + math.Sqrt(0.5)*rng.NormFloat64()
		aVals[i] = a
		bVals[i] = b
		cVals[i] = c
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aVals),
		"B": tabgo.NewSeries("B", bVals),
		"C": tabgo.NewSeries("C", cVals),
	})

	est := NewSEMEstimator(sem, data)
	if err := est.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	betas, intercept, variance, err := est.GetCoefficients("C")
	if err != nil {
		t.Fatalf("GetCoefficients(C): %v", err)
	}
	if len(betas) != 2 {
		t.Fatalf("expected 2 betas, got %d", len(betas))
	}
	assertClose(t, "C beta(A)", betas[0], 2.0, 0.1)
	assertClose(t, "C beta(B)", betas[1], 3.0, 0.1)
	assertClose(t, "C intercept", intercept, 1.0, 0.1)
	assertClose(t, "C variance", variance, 0.5, 0.1)
}

func TestSEMEstimator_Estimate_NilSEM(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0}),
	})
	est := NewSEMEstimator(nil, data)
	if err := est.Estimate(); err == nil {
		t.Error("expected error for nil SEM")
	}
}

func TestSEMEstimator_Estimate_NilData(t *testing.T) {
	sem := buildTestSEM(t)
	est := NewSEMEstimator(sem, nil)
	if err := est.Estimate(); err == nil {
		t.Error("expected error for nil data")
	}
}

func TestSEMEstimator_Estimate_MissingColumn(t *testing.T) {
	sem := buildTestSEM(t)
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0}),
		"Y": tabgo.NewSeries("Y", []any{1.0, 2.0}),
		// Missing Z.
	})
	est := NewSEMEstimator(sem, data)
	if err := est.Estimate(); err == nil {
		t.Error("expected error for missing column")
	}
}

func TestSEMEstimator_GetCoefficients_NilSEM(t *testing.T) {
	est := NewSEMEstimator(nil, nil)
	_, _, _, err := est.GetCoefficients("X")
	if err == nil {
		t.Error("expected error for nil SEM")
	}
}

func TestSEMEstimator_GetCoefficients_UnknownVariable(t *testing.T) {
	sem := buildTestSEM(t)
	est := NewSEMEstimator(sem, nil)
	_, _, _, err := est.GetCoefficients("W")
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

func TestSEMEstimator_CheckModel(t *testing.T) {
	sem := buildTestSEM(t)
	rng := rand.New(rand.NewSource(42))
	data := generateSEMData(rng, 1000)

	est := NewSEMEstimator(sem, data)
	if err := est.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	// The SEM should still be valid after estimation.
	if err := sem.CheckModel(); err != nil {
		t.Errorf("CheckModel after Estimate: %v", err)
	}
}
