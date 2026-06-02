//go:build unit

package ci_tests

import (
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// makeDiscreteDF creates a DataFrame from column name -> []any data.
func makeDiscreteDF(cols map[string][]any) *tabgo.DataFrame {
	series := make(map[string]*tabgo.Series)
	for name, vals := range cols {
		series[name] = tabgo.NewSeries(name, vals)
	}
	return tabgo.NewDataFrame(series)
}

// TestChiSquare_DetectsDependence tests that ChiSquare detects dependence
// when x and y are strongly correlated.
func TestChiSquare_DetectsDependence(t *testing.T) {
	// Create data where x and y are perfectly correlated (x == y).
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(3) // 0, 1, 2
		xVals[i] = v
		yVals[i] = v // perfect dependence
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := ChiSquare("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("ChiSquare should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("ChiSquare p-value should be < 0.05 for dependent data, got %f", pvalue)
	}
}

// TestChiSquare_DetectsIndependence tests that ChiSquare does not reject independence
// when x and y are generated independently.
func TestChiSquare_DetectsIndependence(t *testing.T) {
	n := 500
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < n; i++ {
		xVals[i] = rng.Intn(3)
		yVals[i] = rng.Intn(3) // independent of x
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := ChiSquare("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("ChiSquare should not reject independence for independent data: stat=%f, pvalue=%f", stat, pvalue)
	}
}

// TestChiSquare_WithConditioning tests ChiSquare with a conditioning variable.
func TestChiSquare_WithConditioning(t *testing.T) {
	// x and y are conditionally independent given z:
	// z determines both x and y, but given z, x and y are independent draws.
	n := 600
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < n; i++ {
		z := rng.Intn(2) // 0 or 1
		zVals[i] = z
		// Given z, x and y are independently chosen.
		xVals[i] = rng.Intn(2)
		yVals[i] = rng.Intn(2)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals, "z": zVals})

	_, pvalue, indep := ChiSquare("x", "y", []string{"z"}, df, 0.05)
	if !indep {
		t.Errorf("ChiSquare should find conditional independence given z, pvalue=%f", pvalue)
	}
}

// TestGSq_DetectsDependence tests that GSq detects dependence.
func TestGSq_DetectsDependence(t *testing.T) {
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(3)
		xVals[i] = v
		yVals[i] = v
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := GSq("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("GSq should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

// TestGSq_DetectsIndependence tests that GSq does not reject independence.
func TestGSq_DetectsIndependence(t *testing.T) {
	n := 500
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < n; i++ {
		xVals[i] = rng.Intn(3)
		yVals[i] = rng.Intn(3)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := GSq("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("GSq should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

// TestLogLikelihood_IsSameAsGSq verifies that LogLikelihood produces identical results to GSq.
func TestLogLikelihood_IsSameAsGSq(t *testing.T) {
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(77))
	for i := 0; i < n; i++ {
		xVals[i] = rng.Intn(2)
		yVals[i] = rng.Intn(2)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat1, pval1, ind1 := GSq("x", "y", nil, df, 0.05)
	stat2, pval2, ind2 := LogLikelihood("x", "y", nil, df, 0.05)

	if stat1 != stat2 || pval1 != pval2 || ind1 != ind2 {
		t.Errorf("LogLikelihood should be identical to GSq: GSq(%f,%f,%v) vs LL(%f,%f,%v)",
			stat1, pval1, ind1, stat2, pval2, ind2)
	}
}

// TestPowerDivergence_Lambda1_MatchesChiSquare tests that lambda=1 gives chi-squared results.
func TestPowerDivergence_Lambda1_MatchesChiSquare(t *testing.T) {
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(3)
		xVals[i] = v
		yVals[i] = v
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	statChi, _, _ := ChiSquare("x", "y", nil, df, 0.05)
	statPD, _, _ := PowerDivergence(1.0)("x", "y", nil, df, 0.05)

	// They should be very close (both are Pearson chi-squared).
	diff := statChi - statPD
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-6 {
		t.Errorf("PowerDivergence(1.0) should match ChiSquare: %f vs %f", statPD, statChi)
	}
}

// TestPowerDivergence_Lambda0_MatchesGSq tests that lambda=0 gives G-test results.
func TestPowerDivergence_Lambda0_MatchesGSq(t *testing.T) {
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(3)
		xVals[i] = v
		yVals[i] = v
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	statG, _, _ := GSq("x", "y", nil, df, 0.05)
	statPD, _, _ := PowerDivergence(0.0)("x", "y", nil, df, 0.05)

	diff := statG - statPD
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-6 {
		t.Errorf("PowerDivergence(0.0) should match GSq: %f vs %f", statPD, statG)
	}
}

// TestChiSquare_EmptyConditioning tests that passing empty z slice works like no conditioning.
func TestChiSquare_EmptyConditioning(t *testing.T) {
	n := 100
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(55))
	for i := 0; i < n; i++ {
		xVals[i] = rng.Intn(2)
		yVals[i] = rng.Intn(2)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat1, pval1, _ := ChiSquare("x", "y", nil, df, 0.05)
	stat2, pval2, _ := ChiSquare("x", "y", []string{}, df, 0.05)

	if stat1 != stat2 || pval1 != pval2 {
		t.Errorf("Empty z should equal nil z: (%f,%f) vs (%f,%f)", stat1, pval1, stat2, pval2)
	}
}
