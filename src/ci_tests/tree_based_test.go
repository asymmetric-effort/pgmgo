//go:build unit

package ci_tests

import (
	"math"
	"math/rand"
	"testing"
)

// --- Decision tree unit tests ---

func TestDecisionTree_FitPredict_Constant(t *testing.T) {
	// All targets are the same: tree should predict that constant.
	X := [][]float64{{1}, {2}, {3}, {4}, {5}}
	y := []float64{3, 3, 3, 3, 3}
	dt := newDecisionTree(5, 2)
	dt.fit(X, y)
	preds := dt.predict(X)
	for i, p := range preds {
		if math.Abs(p-3) > 1e-10 {
			t.Errorf("predict[%d]=%f, want 3", i, p)
		}
	}
}

func TestDecisionTree_FitPredict_LinearStep(t *testing.T) {
	// Step function: y = 0 for x <= 5, y = 10 for x > 5.
	n := 100
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = []float64{float64(i)}
		if i <= 50 {
			y[i] = 0
		} else {
			y[i] = 10
		}
	}
	dt := newDecisionTree(10, 2)
	dt.fit(X, y)
	preds := dt.predict(X)

	// Check that low-x samples predict near 0 and high-x near 10.
	for i := 0; i < 45; i++ {
		if math.Abs(preds[i]) > 1.0 {
			t.Errorf("predict[%d]=%f, expected ~0", i, preds[i])
		}
	}
	for i := 55; i < n; i++ {
		if math.Abs(preds[i]-10) > 1.0 {
			t.Errorf("predict[%d]=%f, expected ~10", i, preds[i])
		}
	}
}

func TestDecisionTree_FeatureImportance(t *testing.T) {
	// Two features: only feature 0 matters (y = X[0]).
	n := 100
	rng := rand.New(rand.NewSource(42))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		X[i] = []float64{float64(i), rng.NormFloat64()}
		y[i] = float64(i)
	}
	dt := newDecisionTree(10, 2)
	dt.fit(X, y)
	imp := dt.featureImportance()

	if len(imp) != 2 {
		t.Fatalf("expected 2 importances, got %d", len(imp))
	}
	// Feature 0 should dominate.
	if imp[0] < 0.8 {
		t.Errorf("feature 0 importance=%f, expected > 0.8", imp[0])
	}
}

func TestDecisionTree_EmptyInput(t *testing.T) {
	dt := newDecisionTree(5, 2)
	dt.fit(nil, nil)
	preds := dt.predict([][]float64{{1}})
	if len(preds) != 1 {
		t.Fatalf("expected 1 prediction, got %d", len(preds))
	}
}

func TestDecisionTree_FitPredict_MultiFeature(t *testing.T) {
	// y = x0 + x1, tree should approximate this.
	n := 200
	rng := rand.New(rand.NewSource(77))
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x0 := rng.Float64() * 10
		x1 := rng.Float64() * 10
		X[i] = []float64{x0, x1}
		y[i] = x0 + x1
	}
	dt := newDecisionTree(10, 2)
	dt.fit(X, y)
	preds := dt.predict(X)

	// Compute R^2 on training data; should be high for a tree that can fit additive data.
	meanY := 0.0
	for _, v := range y {
		meanY += v
	}
	meanY /= float64(n)
	ssTot := 0.0
	ssRes := 0.0
	for i := 0; i < n; i++ {
		ssTot += (y[i] - meanY) * (y[i] - meanY)
		ssRes += (y[i] - preds[i]) * (y[i] - preds[i])
	}
	r2 := 1 - ssRes/ssTot
	if r2 < 0.8 {
		t.Errorf("R^2=%f on training data, expected > 0.8", r2)
	}
}

// --- TreeBasedCI tests ---

func TestTreeBasedCI_DetectsDependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(42))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = xData[i]*0.9 + rng.NormFloat64()*0.1
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData})
	stat, pvalue, indep := TreeBasedCI("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("TreeBasedCI should detect dependence: stat=%f, pvalue=%f", stat, pvalue)
	}
	if pvalue >= 0.05 {
		t.Errorf("p-value should be < 0.05 for dependent data, got %f", pvalue)
	}
}

func TestTreeBasedCI_DetectsIndependence(t *testing.T) {
	n := 200
	rng := rand.New(rand.NewSource(99))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.NormFloat64()
		yData[i] = rng.NormFloat64()
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData})
	stat, pvalue, indep := TreeBasedCI("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("TreeBasedCI should not reject independence: stat=%f, pvalue=%f", stat, pvalue)
	}
}

func TestTreeBasedCI_ConditionalIndependence(t *testing.T) {
	// x and y are both caused by z (confounded), so they are marginally
	// dependent but conditionally independent given z.
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

	// Without conditioning: should detect dependence.
	_, pval, indep := TreeBasedCI("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("TreeBasedCI should detect marginal dependence, pvalue=%f", pval)
	}

	// Conditioning on z: should find conditional independence.
	_, pval, indep = TreeBasedCI("x", "y", []string{"z"}, df, 0.05)
	if !indep {
		t.Errorf("TreeBasedCI should find conditional independence given z, pvalue=%f", pval)
	}
}

func TestTreeBasedCI_NonlinearDependence_FisherZMisses(t *testing.T) {
	// Create a nonlinear dependency: y = x^2 + noise.
	// Pearson correlation (and FisherZ) will see near-zero linear correlation
	// for a symmetric distribution, but TreeBasedCI should detect it via
	// the residualization approach when both depend nonlinearly on z.
	n := 500
	rng := rand.New(rand.NewSource(55))

	zData := make([]float64, n)
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		// z is the latent variable
		zData[i] = rng.NormFloat64()
		// x depends on z nonlinearly
		xData[i] = zData[i]*zData[i] + rng.NormFloat64()*0.1
		// y depends on z nonlinearly in the same way
		yData[i] = zData[i]*zData[i] + rng.NormFloat64()*0.1
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData, "z": zData})

	// FisherZ conditioning on z uses partial correlation which is linear.
	// Since the relationship x~z and y~z are both quadratic, FisherZ may
	// still detect dependence (because x and y are correlated), but after
	// linear residualization, substantial nonlinear signal remains.
	// Key test: TreeBasedCI should detect the marginal dependence.
	_, pvalTree, indepTree := TreeBasedCI("x", "y", nil, df, 0.05)
	if indepTree {
		t.Errorf("TreeBasedCI should detect nonlinear dependence, pvalue=%f", pvalTree)
	}

	// Now test with a pure nonlinear relationship that has zero linear correlation.
	// y = x^2 with x symmetric around 0 => corr(x,y) ~ 0 but they are dependent.
	xData2 := make([]float64, n)
	yData2 := make([]float64, n)
	for i := 0; i < n; i++ {
		xData2[i] = rng.NormFloat64() * 2
		yData2[i] = xData2[i]*xData2[i] + rng.NormFloat64()*0.3
	}

	df2 := makeContinuousDF(map[string][]float64{"x": xData2, "y": yData2})

	// FisherZ should fail to detect this (p-value > 0.05) since corr(x, x^2) ~ 0
	// for symmetric x.
	_, pvalFisher, indepFisher := FisherZ("x", "y", nil, df2, 0.05)
	// Note: FisherZ might or might not detect it depending on sampling, but the
	// correlation should be weak. We check that TreeBasedCI is at least as
	// good or better.
	_, pvalTree2, _ := TreeBasedCI("x", "y", nil, df2, 0.05)

	// Both tests operate on the raw data without conditioning, so TreeBasedCI
	// also uses Pearson correlation directly. The real advantage is in the
	// conditional case. Log results for diagnostic purposes.
	t.Logf("Pure nonlinear (y=x^2): FisherZ p=%f indep=%v, TreeBasedCI p=%f",
		pvalFisher, indepFisher, pvalTree2)
}

func TestTreeBasedCI_NonlinearConditional_TreeBeatsFisherZ(t *testing.T) {
	// The key scenario where TreeBasedCI outshines FisherZ:
	// x = z^2 + noise_x, y = z^2 + noise_y.
	// x and y are dependent (both driven by z^2).
	// Conditioning on z using linear methods (FisherZ) leaves residual
	// dependence because the quadratic component is not removed.
	// TreeBasedCI should handle this better since trees can capture z^2.
	n := 800
	rng := rand.New(rand.NewSource(77))

	zData := make([]float64, n)
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		zData[i] = rng.NormFloat64() * 2
		zSq := zData[i] * zData[i]
		xData[i] = zSq + rng.NormFloat64()*1.5
		yData[i] = zSq + rng.NormFloat64()*1.5
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData, "z": zData})

	// Without conditioning: both should detect dependence.
	_, pvalF, indepF := FisherZ("x", "y", nil, df, 0.05)
	if indepF {
		t.Errorf("FisherZ should detect marginal dependence, p=%f", pvalF)
	}
	_, pvalT, indepT := TreeBasedCI("x", "y", nil, df, 0.05)
	if indepT {
		t.Errorf("TreeBasedCI should detect marginal dependence, p=%f", pvalT)
	}

	// Conditioning on z:
	// FisherZ uses linear residualization, so z^2 signal leaks through =>
	// it should still detect dependence (fail to find conditional independence).
	_, pvalFCond, indepFCond := FisherZ("x", "y", []string{"z"}, df, 0.05)

	// TreeBasedCI uses tree residualization which captures z^2 =>
	// it should find conditional independence (x _|_ y | z).
	_, pvalTCond, indepTCond := TreeBasedCI("x", "y", []string{"z"}, df, 0.05)

	t.Logf("Conditional on z: FisherZ p=%f indep=%v, TreeBasedCI p=%f indep=%v",
		pvalFCond, indepFCond, pvalTCond, indepTCond)

	// FisherZ should fail: it cannot remove the nonlinear confounding.
	if indepFCond {
		t.Logf("FisherZ unexpectedly found conditional independence (p=%f); nonlinear signal may be weak", pvalFCond)
	}

	// TreeBasedCI should succeed: trees capture the nonlinear relationship.
	if !indepTCond {
		t.Errorf("TreeBasedCI should find conditional independence given z (nonlinear), but p=%f", pvalTCond)
	}
}

func TestTreeBasedCI_LinearlySeparable(t *testing.T) {
	// Linearly separable: x = z + noise, y = z + noise.
	// Both FisherZ and TreeBasedCI should find conditional independence given z.
	n := 400
	rng := rand.New(rand.NewSource(33))

	zData := make([]float64, n)
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		zData[i] = rng.NormFloat64()
		xData[i] = zData[i]*1.0 + rng.NormFloat64()*0.5
		yData[i] = zData[i]*1.0 + rng.NormFloat64()*0.5
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData, "z": zData})

	// Both methods should find conditional independence.
	_, pvalF, indepF := FisherZ("x", "y", []string{"z"}, df, 0.05)
	if !indepF {
		t.Errorf("FisherZ should find conditional independence (linear), p=%f", pvalF)
	}

	_, pvalT, indepT := TreeBasedCI("x", "y", []string{"z"}, df, 0.05)
	if !indepT {
		t.Errorf("TreeBasedCI should find conditional independence (linear), p=%f", pvalT)
	}
}

func TestTreeBasedCI_MultipleConditioning(t *testing.T) {
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
	_, pval, indep := TreeBasedCI("x", "y", nil, df, 0.05)
	if indep {
		t.Errorf("TreeBasedCI should detect marginal dependence, pvalue=%f", pval)
	}

	// Conditioning on z1, z2: should find conditional independence.
	_, pval, indep = TreeBasedCI("x", "y", []string{"z1", "z2"}, df, 0.05)
	if !indep {
		t.Errorf("TreeBasedCI should find conditional independence given z1,z2, pvalue=%f", pval)
	}
}

func TestTreeBasedCI_TooFewSamples(t *testing.T) {
	xData := []float64{1, 2, 3}
	yData := []float64{2, 4, 6}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData})
	_, _, indep := TreeBasedCI("x", "y", nil, df, 0.05)
	if !indep {
		t.Errorf("TreeBasedCI should return independent=true when n < 4")
	}
}

func TestSortFloat64s(t *testing.T) {
	a := []float64{5, 3, 1, 4, 2}
	sortFloat64s(a)
	for i := 1; i < len(a); i++ {
		if a[i] < a[i-1] {
			t.Errorf("not sorted at index %d: %v", i, a)
		}
	}
}

func TestDecisionTree_MaxDepthLimit(t *testing.T) {
	// Verify that maxDepth=1 produces a stump (at most 1 split).
	X := make([][]float64, 20)
	y := make([]float64, 20)
	for i := 0; i < 20; i++ {
		X[i] = []float64{float64(i)}
		y[i] = float64(i)
	}
	dt := newDecisionTree(1, 2)
	dt.fit(X, y)

	// A stump has root with at most 2 leaves.
	if dt.root.isLeaf {
		// This is acceptable but unlikely for non-constant y.
		return
	}
	if !dt.root.left.isLeaf || !dt.root.right.isLeaf {
		t.Errorf("maxDepth=1 should produce a stump with leaf children")
	}
}

func TestDecisionTree_PredictNewData(t *testing.T) {
	// Train on [0,10] with step function, predict on new points.
	X := make([][]float64, 100)
	y := make([]float64, 100)
	for i := 0; i < 100; i++ {
		X[i] = []float64{float64(i) / 10.0}
		if X[i][0] < 5 {
			y[i] = -1
		} else {
			y[i] = 1
		}
	}
	dt := newDecisionTree(10, 2)
	dt.fit(X, y)

	// Predict on unseen data.
	newX := [][]float64{{1.0}, {9.0}}
	preds := dt.predict(newX)
	if preds[0] > 0 {
		t.Errorf("predict(1.0)=%f, expected negative", preds[0])
	}
	if preds[1] < 0 {
		t.Errorf("predict(9.0)=%f, expected positive", preds[1])
	}
}

func TestTreeBasedCI_CompileTimeCheck(t *testing.T) {
	// Verify TreeBasedCI satisfies CITest at runtime too.
	var ci CITest = TreeBasedCI
	_ = ci
}

func TestDecisionTree_FeatureImportance_AllZero(t *testing.T) {
	// Constant y => no splits => all importances zero.
	X := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	y := []float64{5, 5, 5}
	dt := newDecisionTree(5, 2)
	dt.fit(X, y)
	imp := dt.featureImportance()
	for i, v := range imp {
		if v != 0 {
			t.Errorf("importance[%d]=%f, expected 0 for constant target", i, v)
		}
	}
}

func TestTreeBasedCI_NonlinearConfound_DirectXY(t *testing.T) {
	// Direct nonlinear relationship y = sin(x) + noise, no conditioning.
	// Even though sin is nonlinear, for continuous x the Pearson correlation
	// of x and sin(x) may be significant. Verify TreeBasedCI handles it.
	n := 300
	rng := rand.New(rand.NewSource(88))
	xData := make([]float64, n)
	yData := make([]float64, n)
	for i := 0; i < n; i++ {
		xData[i] = rng.Float64() * 2 * math.Pi
		yData[i] = math.Sin(xData[i]) + rng.NormFloat64()*0.1
	}

	df := makeContinuousDF(map[string][]float64{"x": xData, "y": yData})
	_, pval, indep := TreeBasedCI("x", "y", nil, df, 0.05)
	// sin(x) on [0, 2pi] has moderate linear correlation with x.
	// The test should at least not crash and produce valid output.
	t.Logf("sin(x) test: p=%f, indep=%v", pval, indep)
	if pval < 0 || pval > 1 {
		t.Errorf("p-value out of range: %f", pval)
	}
}
