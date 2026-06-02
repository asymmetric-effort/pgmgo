//go:build unit

package ci_tests_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/ci_tests"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// ciInput holds the input section of a CI test fixture.
type ciInput struct {
	X                 string      `json:"x"`
	Y                 string      `json:"y"`
	Z                 []string    `json:"z"`
	SignificanceLevel float64     `json:"significance_level"`
	DataColumns       []string    `json:"data_columns"`
	DataRows          [][]float64 `json:"data_rows"`
}

// ciExpected holds the expected output of a CI test fixture.
type ciExpected struct {
	Statistic   float64 `json:"statistic"`
	PValue      float64 `json:"p_value"`
	DOF         int     `json:"dof"`
	Independent bool    `json:"independent"`
}

// buildCIDataFrame creates a DataFrame with string-typed values from fixture rows.
func buildCIDataFrame(columns []string, rows [][]float64) *tabgo.DataFrame {
	anyRows := make([][]any, len(rows))
	for i, row := range rows {
		anyRow := make([]any, len(row))
		for j, v := range row {
			anyRow[j] = strconv.Itoa(int(v))
		}
		anyRows[i] = anyRow
	}
	return tabgo.NewDataFrameFromRows(columns, anyRows)
}

func TestCrossval_ChiSquareDependent(t *testing.T) {
	ff := testutil.LoadFixtures(t, "ci_tests/fixtures.json")
	tc := ff.FindTestCase(t, "chi_square_dependent")

	var input ciInput
	tc.UnmarshalInput(t, &input)

	var expected ciExpected
	tc.UnmarshalExpected(t, &expected)

	df := buildCIDataFrame(input.DataColumns, input.DataRows)

	stat, pval, indep := ci_tests.ChiSquare(input.X, input.Y, input.Z, df, input.SignificanceLevel)

	// The independence conclusion must match.
	if indep != expected.Independent {
		t.Errorf("independent: expected %v, got %v (stat=%.4f, pval=%.6g)", expected.Independent, indep, stat, pval)
	}

	// Statistic should be reasonably close (within 10% relative or 1.0 absolute).
	if expected.Statistic > 1.0 {
		relErr := math.Abs(stat-expected.Statistic) / expected.Statistic
		if relErr > 0.10 {
			t.Errorf("statistic: expected %.4f, got %.4f (relErr=%.4f)", expected.Statistic, stat, relErr)
		}
	}
}

func TestCrossval_ChiSquareIndependent(t *testing.T) {
	ff := testutil.LoadFixtures(t, "ci_tests/fixtures.json")
	tc := ff.FindTestCase(t, "chi_square_independent")

	var input ciInput
	tc.UnmarshalInput(t, &input)

	var expected ciExpected
	tc.UnmarshalExpected(t, &expected)

	df := buildCIDataFrame(input.DataColumns, input.DataRows)

	stat, pval, indep := ci_tests.ChiSquare(input.X, input.Y, input.Z, df, input.SignificanceLevel)

	// The independence conclusion must match.
	if indep != expected.Independent {
		t.Errorf("independent: expected %v, got %v (stat=%.4f, pval=%.6g)", expected.Independent, indep, stat, pval)
	}

	// For the independent case, just verify pval > significance.
	if !indep {
		t.Errorf("expected independent (pval=%.6g > %.2f), got dependent", pval, input.SignificanceLevel)
	}
	_ = stat
}

func TestCrossval_ChiSquareConditional(t *testing.T) {
	ff := testutil.LoadFixtures(t, "ci_tests/fixtures.json")
	tc := ff.FindTestCase(t, "chi_square_conditional")

	var input ciInput
	tc.UnmarshalInput(t, &input)

	var expected ciExpected
	tc.UnmarshalExpected(t, &expected)

	df := buildCIDataFrame(input.DataColumns, input.DataRows)

	stat, pval, indep := ci_tests.ChiSquare(input.X, input.Y, input.Z, df, input.SignificanceLevel)

	// The independence conclusion must match.
	if indep != expected.Independent {
		t.Errorf("independent: expected %v, got %v (stat=%.4f, pval=%.6g)", expected.Independent, indep, stat, pval)
	}

	// Statistic should be reasonably close for the conditional test.
	if expected.Statistic > 1.0 {
		relErr := math.Abs(stat-expected.Statistic) / expected.Statistic
		if relErr > 0.10 {
			t.Errorf("statistic: expected %.4f, got %.4f (relErr=%.4f)", expected.Statistic, stat, relErr)
		}
	}
	_ = pval
}
