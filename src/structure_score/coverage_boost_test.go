//go:build unit

package structure_score

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// TestGaussianLL_EmptyData exercises the N==0 early return.
func TestGaussianLL_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"X"}, [][]any{})
	ll, np := gaussianLL("X", nil, data)
	if ll != 0 || np != 0 {
		t.Errorf("expected (0,0) for empty data, got (%f, %d)", ll, np)
	}
}

// TestGaussianLL_ZeroVariance exercises the variance < 1e-300 guard.
func TestGaussianLL_ZeroVariance(t *testing.T) {
	// All identical values => variance = 0 => triggers the guard.
	rows := [][]any{{5.0}, {5.0}, {5.0}, {5.0}}
	data := tabgo.NewDataFrameFromRows([]string{"X"}, rows)
	ll, np := gaussianLL("X", nil, data)
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("expected finite LL for zero variance, got %f", ll)
	}
	_ = np
}

// TestGaussianLL_ZeroVarianceWithParents exercises the variance guard in
// the regression path when all residuals are zero.
func TestGaussianLL_ZeroVarianceWithParents(t *testing.T) {
	// Y = 2*X exactly => RSS = 0 => variance = 0.
	rows := [][]any{{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {4.0, 8.0}}
	data := tabgo.NewDataFrameFromRows([]string{"X", "Y"}, rows)
	ll, _ := gaussianLL("Y", []string{"X"}, data)
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("expected finite LL for perfect fit, got %f", ll)
	}
}

// TestGaussSolve_SingularMatrix exercises the pivot < 1e-15 guard
// (singular/near-singular matrix handling).
func TestGaussSolve_SingularMatrix(t *testing.T) {
	// Singular matrix: second row is a multiple of first.
	// [1 2] [x1]   [3]
	// [2 4] [x2] = [6]
	A := []float64{1, 2, 2, 4}
	b := []float64{3, 6}
	x := gaussSolve(A, b, 2)
	// Should not panic; x[1] should be set to 0 in back-substitution.
	_ = x
}

// TestGaussSolve_SingularBackSubstitution exercises the back-substitution
// guard where a[col*n+col] is nearly zero.
func TestGaussSolve_SingularBackSubstitution(t *testing.T) {
	// 3x3 matrix where after elimination, one diagonal is near-zero.
	// [1 0 0] [x1]   [1]
	// [0 0 0] [x2] = [0]
	// [0 0 1] [x3]   [1]
	A := []float64{1, 0, 0, 0, 0, 0, 0, 0, 1}
	b := []float64{1, 0, 1}
	x := gaussSolve(A, b, 3)
	if x[1] != 0 {
		t.Errorf("expected x[1]=0 for singular column, got %f", x[1])
	}
}

// TestConditionalGaussianLL_EmptyData exercises the N==0 early return.
func TestConditionalGaussianLL_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"X", "D"}, [][]any{})
	ll, np := conditionalGaussianLL("X", []string{"D"}, nil, data)
	if ll != 0 || np != 0 {
		t.Errorf("expected (0,0) for empty data, got (%f, %d)", ll, np)
	}
}

// TestConditionalGaussianLL_MultipleDiscreteParents exercises the comma
// separator in the key construction (line 56-58).
func TestConditionalGaussianLL_MultipleDiscreteParents(t *testing.T) {
	// Two discrete parents and one continuous variable.
	rows := [][]any{
		{0, 0, 1.0},
		{0, 1, 2.0},
		{1, 0, 3.0},
		{1, 1, 4.0},
		{0, 0, 1.5},
		{0, 1, 2.5},
		{1, 0, 3.5},
		{1, 1, 4.5},
	}
	data := tabgo.NewDataFrameFromRows([]string{"D1", "D2", "Y"}, rows)
	ll, np := conditionalGaussianLL("Y", []string{"D1", "D2"}, nil, data)
	if math.IsNaN(ll) {
		t.Errorf("expected finite LL, got NaN")
	}
	if np <= 0 {
		t.Errorf("expected positive numParams, got %d", np)
	}
}

// TestConditionalGaussianLL_ZeroVarianceStratum exercises the variance guard
// in the no-continuous-parents path within a stratum.
func TestConditionalGaussianLL_ZeroVarianceStratum(t *testing.T) {
	// All values identical within each stratum => variance = 0.
	rows := [][]any{
		{0, 5.0},
		{0, 5.0},
		{1, 10.0},
		{1, 10.0},
	}
	data := tabgo.NewDataFrameFromRows([]string{"D", "Y"}, rows)
	ll, _ := conditionalGaussianLL("Y", []string{"D"}, nil, data)
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("expected finite LL for zero-variance stratum, got %f", ll)
	}
}

// TestConditionalGaussianLL_ZeroVarianceStratumWithContinuous exercises
// the variance guard in the regression path within a stratum.
func TestConditionalGaussianLL_ZeroVarianceStratumWithContinuous(t *testing.T) {
	// Y = 2*X exactly within each stratum.
	rows := [][]any{
		{0, 1.0, 2.0},
		{0, 2.0, 4.0},
		{0, 3.0, 6.0},
		{1, 1.0, 2.0},
		{1, 2.0, 4.0},
		{1, 3.0, 6.0},
	}
	data := tabgo.NewDataFrameFromRows([]string{"D", "X", "Y"}, rows)
	ll, _ := conditionalGaussianLL("Y", []string{"D"}, []string{"X"}, data)
	if math.IsNaN(ll) || math.IsInf(ll, 0) {
		t.Errorf("expected finite LL, got %f", ll)
	}
}

// TestIsDiscreteColumn_NonIntegerFloat exercises the float64 non-integer path.
func TestIsDiscreteColumn_NonIntegerFloat(t *testing.T) {
	rows := [][]any{{1.5}, {2.7}, {3.3}}
	data := tabgo.NewDataFrameFromRows([]string{"X"}, rows)
	if isDiscreteColumn(data, "X") {
		t.Error("expected non-integer float column to be identified as continuous")
	}
}

// TestIsDiscreteColumn_DefaultType exercises the default type-switch case.
func TestIsDiscreteColumn_DefaultType(t *testing.T) {
	// Use a type that isn't string, int, or float64.
	rows := [][]any{{true}, {false}, {true}}
	data := tabgo.NewDataFrameFromRows([]string{"X"}, rows)
	if !isDiscreteColumn(data, "X") {
		t.Error("expected default type to be classified as discrete")
	}
}

// TestIsDiscreteColumn_IntegerFloats exercises the all-integer-float path.
func TestIsDiscreteColumn_IntegerFloats(t *testing.T) {
	rows := [][]any{{1.0}, {2.0}, {3.0}}
	data := tabgo.NewDataFrameFromRows([]string{"X"}, rows)
	if !isDiscreteColumn(data, "X") {
		t.Error("expected integer-valued floats to be classified as discrete")
	}
}

// TestAICCondGauss_LocalScore_EmptyData exercises the N==0 guard.
func TestAICCondGauss_LocalScore_EmptyData(t *testing.T) {
	scorer := NewAICCondGauss()
	data := tabgo.NewDataFrameFromRows([]string{"X"}, [][]any{})
	score := scorer.LocalScore("X", nil, data)
	if score != 0 {
		t.Errorf("expected 0 for empty data, got %f", score)
	}
}

// TestAICGauss_LocalScore_EmptyData exercises the N==0 guard.
func TestAICGauss_LocalScore_EmptyData(t *testing.T) {
	scorer := NewAICGauss()
	data := tabgo.NewDataFrameFromRows([]string{"X"}, [][]any{})
	score := scorer.LocalScore("X", nil, data)
	if score != 0 {
		t.Errorf("expected 0 for empty data, got %f", score)
	}
}

// TestAllParentConfigs_MultipleParents exercises the depth > 0 comma
// separator path in allParentConfigs.
func TestAllParentConfigs_MultipleParents(t *testing.T) {
	// Create data with two discrete parents.
	rows := [][]any{
		{0, 0, 1.0},
		{0, 1, 2.0},
		{1, 0, 3.0},
		{1, 1, 4.0},
	}
	data := tabgo.NewDataFrameFromRows([]string{"P1", "P2", "Y"}, rows)
	configs := allParentConfigs([]string{"P1", "P2"}, data)
	// With 2 parents, each with 2 values, should have 4 configs.
	if len(configs) != 4 {
		t.Errorf("expected 4 configs, got %d: %v", len(configs), configs)
	}
	// Each config should contain a comma separator.
	for _, c := range configs {
		found := false
		for _, ch := range c {
			if ch == ',' {
				found = true
			}
		}
		if !found {
			t.Errorf("expected comma in config %q", c)
		}
	}
}
