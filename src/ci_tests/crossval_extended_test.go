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

// ciExtInput holds the input for extended CI test fixtures.
type ciExtInput struct {
	X                 string      `json:"x"`
	Y                 string      `json:"y"`
	Z                 []string    `json:"z"`
	SignificanceLevel float64     `json:"significance_level"`
	DataColumns       []string    `json:"data_columns"`
	DataRows          [][]float64 `json:"data_rows"`
	Continuous        bool        `json:"continuous"`
}

// ciExtExpected holds expected output for extended CI test fixtures.
type ciExtExpected struct {
	Statistic   float64 `json:"statistic"`
	PValue      float64 `json:"p_value"`
	DOF         int     `json:"dof"`
	Independent bool    `json:"independent"`
}

// buildCIExtDF creates a DataFrame from fixture rows. Uses string for discrete, float for continuous.
func buildCIExtDF(columns []string, rows [][]float64, continuous bool) *tabgo.DataFrame {
	anyRows := make([][]any, len(rows))
	for i, row := range rows {
		anyRow := make([]any, len(row))
		for j, v := range row {
			if continuous {
				anyRow[j] = v
			} else {
				anyRow[j] = strconv.Itoa(int(v))
			}
		}
		anyRows[i] = anyRow
	}
	return tabgo.NewDataFrameFromRows(columns, anyRows)
}

func runDiscreteCI(t *testing.T, fixtureName string, ciTest ci_tests.CITest) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "ci_tests_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input ciExtInput
	tc.UnmarshalInput(t, &input)

	var expected ciExtExpected
	tc.UnmarshalExpected(t, &expected)

	df := buildCIExtDF(input.DataColumns, input.DataRows, false)
	stat, pval, indep := ciTest(input.X, input.Y, input.Z, df, input.SignificanceLevel)

	// Independence conclusion must match
	if indep != expected.Independent {
		t.Errorf("independent: expected %v, got %v (stat=%.4f, pval=%.6g)", expected.Independent, indep, stat, pval)
	}

	// Statistic should be reasonably close for large values
	if expected.Statistic > 1.0 {
		relErr := math.Abs(stat-expected.Statistic) / math.Abs(expected.Statistic)
		if relErr > 0.15 {
			t.Errorf("statistic: expected %.4f, got %.4f (relErr=%.4f)", expected.Statistic, stat, relErr)
		}
	}
}

func runContinuousCI(t *testing.T, fixtureName string, ciTest ci_tests.CITest) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "ci_tests_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input ciExtInput
	tc.UnmarshalInput(t, &input)

	var expected ciExtExpected
	tc.UnmarshalExpected(t, &expected)

	df := buildCIExtDF(input.DataColumns, input.DataRows, true)
	stat, pval, indep := ciTest(input.X, input.Y, input.Z, df, input.SignificanceLevel)

	if indep != expected.Independent {
		t.Errorf("independent: expected %v, got %v (stat=%.4f, pval=%.6g)", expected.Independent, indep, stat, pval)
	}

	_ = stat
}

// ChiSquare extended tests
func TestCrossval_ChiSquareExtDep(t *testing.T) {
	runDiscreteCI(t, "chi_square_extended_dep", ci_tests.ChiSquare)
}

func TestCrossval_ChiSquareExtIndep(t *testing.T) {
	runDiscreteCI(t, "chi_square_extended_indep", ci_tests.ChiSquare)
}

func TestCrossval_ChiSquareCondZ(t *testing.T) {
	runDiscreteCI(t, "chi_square_cond_z", ci_tests.ChiSquare)
}

// GSq tests
func TestCrossval_GSqDependent(t *testing.T) {
	runDiscreteCI(t, "gsq_dependent", ci_tests.GSq)
}

func TestCrossval_GSqIndependent(t *testing.T) {
	runDiscreteCI(t, "gsq_independent", ci_tests.GSq)
}

func TestCrossval_GSqCondZ(t *testing.T) {
	runDiscreteCI(t, "gsq_cond_z", ci_tests.GSq)
}

// FisherZ tests
func TestCrossval_FisherZDependent(t *testing.T) {
	runDiscreteCI(t, "fisher_z_dependent", ci_tests.FisherZ)
}

func TestCrossval_FisherZIndependent(t *testing.T) {
	runDiscreteCI(t, "fisher_z_independent", ci_tests.FisherZ)
}

// Pearsonr tests (continuous data)
func TestCrossval_PearsonrDependent(t *testing.T) {
	runContinuousCI(t, "pearsonr_dependent", ci_tests.Pearsonr)
}

func TestCrossval_PearsonrIndependent(t *testing.T) {
	runContinuousCI(t, "pearsonr_independent", ci_tests.Pearsonr)
}
