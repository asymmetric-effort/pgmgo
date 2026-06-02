//go:build unit

package ci_tests

import (
	"math/rand"
	"testing"
)

// ============================================================
// ModifiedLogLikelihood Tests
// ============================================================

func TestModifiedLogLikelihood_DetectsDependence(t *testing.T) {
	// x and y are perfectly correlated.
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(3)
		xVals[i] = v
		yVals[i] = v // perfect dependence
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := ModifiedLogLikelihood("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("ModifiedLogLikelihood should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("ModifiedLogLikelihood p-value should be < 0.05, got %f", pvalue)
	}
}

func TestModifiedLogLikelihood_DetectsIndependence(t *testing.T) {
	n := 500
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < n; i++ {
		xVals[i] = rng.Intn(3)
		yVals[i] = rng.Intn(3) // independent
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := ModifiedLogLikelihood("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("ModifiedLogLikelihood should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestModifiedLogLikelihood_SmallerThanGSq(t *testing.T) {
	// Williams' correction should make the statistic smaller than the raw G statistic.
	n := 50 // small sample to make the correction meaningful
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
	statMod, _, _ := ModifiedLogLikelihood("x", "y", nil, df, 0.05)

	if statMod >= statG {
		t.Errorf("ModifiedLogLikelihood statistic (%f) should be smaller than GSq (%f) due to Williams' correction",
			statMod, statG)
	}
}

func TestModifiedLogLikelihood_ConvergesToGSqForLargeN(t *testing.T) {
	// For large samples, the correction factor q -> 1, so the statistic should
	// approach the raw G statistic.
	n := 10000
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
	statMod, _, _ := ModifiedLogLikelihood("x", "y", nil, df, 0.05)

	relDiff := (statG - statMod) / statG
	if relDiff > 0.01 {
		t.Errorf("For large N, ModifiedLogLikelihood should be very close to GSq: GSq=%f, Mod=%f, relDiff=%f",
			statG, statMod, relDiff)
	}
}

func TestModifiedLogLikelihood_WithConditioning(t *testing.T) {
	// x and y are conditionally independent given z.
	n := 600
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < n; i++ {
		z := rng.Intn(2)
		zVals[i] = z
		xVals[i] = rng.Intn(2)
		yVals[i] = rng.Intn(2)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals, "z": zVals})

	_, pvalue, indep := ModifiedLogLikelihood("x", "y", []string{"z"}, df, 0.05)
	if !indep {
		t.Errorf("ModifiedLogLikelihood should find conditional independence given z, pvalue=%f", pvalue)
	}
}

func TestModifiedLogLikelihood_EmptyData(t *testing.T) {
	df := makeDiscreteDF(map[string][]any{"x": {}, "y": {}})
	stat, pvalue, indep := ModifiedLogLikelihood("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("ModifiedLogLikelihood on empty data should return independent=true: stat=%f, pvalue=%f", stat, pvalue)
	}
}

// ============================================================
// IndependenceMatch Tests
// ============================================================

func TestIndependenceMatch_DetectsDependence(t *testing.T) {
	// x and y are perfectly correlated.
	n := 300
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(4)
		xVals[i] = v
		yVals[i] = v // perfect dependence
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := IndependenceMatch("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("IndependenceMatch should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("IndependenceMatch p-value should be < 0.05, got %f", pvalue)
	}
}

func TestIndependenceMatch_DetectsIndependence(t *testing.T) {
	n := 500
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(99))
	for i := 0; i < n; i++ {
		xVals[i] = rng.Intn(3)
		yVals[i] = rng.Intn(3) // independent
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := IndependenceMatch("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("IndependenceMatch should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestIndependenceMatch_WithConditioning(t *testing.T) {
	// z is a confounder: z determines x and y, but given z, they are independent.
	n := 600
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	rng := rand.New(rand.NewSource(123))
	for i := 0; i < n; i++ {
		z := rng.Intn(2)
		zVals[i] = z
		xVals[i] = rng.Intn(3)
		yVals[i] = rng.Intn(3)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals, "z": zVals})

	_, pvalue, indep := IndependenceMatch("x", "y", []string{"z"}, df, 0.05)
	if !indep {
		t.Errorf("IndependenceMatch should find conditional independence given z, pvalue=%f", pvalue)
	}
}

func TestIndependenceMatch_ConfoundedMarginal(t *testing.T) {
	// z confounds x and y: marginally dependent, conditionally independent.
	n := 600
	xVals := make([]any, n)
	yVals := make([]any, n)
	zVals := make([]any, n)
	rng := rand.New(rand.NewSource(77))
	for i := 0; i < n; i++ {
		z := rng.Intn(3)
		zVals[i] = z
		// x and y both depend on z.
		xVals[i] = z + rng.Intn(2)
		yVals[i] = z + rng.Intn(2)
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals, "z": zVals})

	// Marginal: should detect dependence (z confounds).
	_, pMarg, indepMarg := IndependenceMatch("x", "y", nil, df, 0.05)
	if indepMarg {
		t.Errorf("IndependenceMatch should detect marginal dependence, pvalue=%f", pMarg)
	}

	// Conditional on z: should find independence.
	_, pCond, indepCond := IndependenceMatch("x", "y", []string{"z"}, df, 0.05)
	if !indepCond {
		t.Errorf("IndependenceMatch should find conditional independence given z, pvalue=%f", pCond)
	}
}

func TestIndependenceMatch_TooFewSamples(t *testing.T) {
	df := makeDiscreteDF(map[string][]any{"x": {1, 2, 3}, "y": {4, 5, 6}})
	_, _, indep := IndependenceMatch("x", "y", nil, df, 0.05)
	if !indep {
		t.Error("IndependenceMatch should return independent=true with too few samples")
	}
}

func TestIndependenceMatch_BinaryPerfectAssociation(t *testing.T) {
	// Binary x and y that are identical.
	n := 200
	xVals := make([]any, n)
	yVals := make([]any, n)
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < n; i++ {
		v := rng.Intn(2)
		xVals[i] = v
		yVals[i] = v
	}

	df := makeDiscreteDF(map[string][]any{"x": xVals, "y": yVals})

	stat, pvalue, indep := IndependenceMatch("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("IndependenceMatch should detect perfect binary association: stat=%f, pvalue=%f", stat, pvalue)
	}
}
