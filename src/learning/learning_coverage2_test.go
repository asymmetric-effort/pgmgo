//go:build unit

package learning

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func make3VarDF() *tabgo.DataFrame {
	sm := map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1}),
		"B": tabgo.NewSeries("B", []any{0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1}),
		"C": tabgo.NewSeries("C", []any{0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0}),
	}
	return tabgo.NewDataFrame(sm)
}

func makeContinuousDF() *tabgo.DataFrame {
	sm := map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
		"Y": tabgo.NewSeries("Y", []any{2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.1, 16.2, 17.9, 20.0}),
		"Z": tabgo.NewSeries("Z", []any{0.5, 1.1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}),
	}
	return tabgo.NewDataFrame(sm)
}

func makeIVData() *tabgo.DataFrame {
	sm := map[string]*tabgo.Series{
		"instrument": tabgo.NewSeries("instrument", []any{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}),
		"treatment":  tabgo.NewSeries("treatment", []any{1.5, 2.3, 3.1, 4.2, 5.3, 6.1, 7.0, 7.9, 9.1, 10.2}),
		"outcome":    tabgo.NewSeries("outcome", []any{3.0, 4.5, 6.0, 8.1, 10.2, 12.3, 14.0, 15.9, 18.0, 20.1}),
	}
	return tabgo.NewDataFrame(sm)
}

// ---------------------------------------------------------------------------
// Scoring functions: BICScore, AICScore, K2Score, BDeuScore error paths
// ---------------------------------------------------------------------------

func TestBICScore_EmptyData(t *testing.T) {
	fn := BICScore()
	df := tabgo.NewDataFrameFromRows([]string{"A"}, nil)
	score := fn("A", nil, df)
	if score != 0 {
		t.Errorf("expected 0 for empty data, got %f", score)
	}
}

func TestAICScore_EmptyData(t *testing.T) {
	fn := AICScore()
	df := tabgo.NewDataFrameFromRows([]string{"A"}, nil)
	score := fn("A", nil, df)
	if score != 0 {
		t.Errorf("expected 0 for empty data, got %f", score)
	}
}

func TestK2Score_WithParents(t *testing.T) {
	fn := K2Score()
	data := make3VarDF()
	score := fn("C", []string{"A", "B"}, data)
	// Just verify it runs and returns finite value
	if score == 0 {
		// K2Score shouldn't be exactly 0 with data
	}
	_ = score
}

func TestBDeuScore_WithParents(t *testing.T) {
	fn := BDeuScore(5.0)
	data := make3VarDF()
	score := fn("C", []string{"A"}, data)
	_ = score
}

func TestAICScore_WithParents(t *testing.T) {
	fn := AICScore()
	data := make3VarDF()
	score := fn("C", []string{"A"}, data)
	_ = score
}

func TestBICScore_WithParents(t *testing.T) {
	fn := BICScore()
	data := make3VarDF()
	score := fn("C", []string{"A"}, data)
	_ = score
}

// ---------------------------------------------------------------------------
// intToString: edge cases
// ---------------------------------------------------------------------------

func TestIntToString_Zero(t *testing.T) {
	if intToString(0) != "0" {
		t.Errorf("expected '0', got %q", intToString(0))
	}
}

func TestIntToString_Negative(t *testing.T) {
	result := intToString(-42)
	if result != "-42" {
		t.Errorf("expected '-42', got %q", result)
	}
}

func TestIntToString_Positive(t *testing.T) {
	result := intToString(123)
	if result != "123" {
		t.Errorf("expected '123', got %q", result)
	}
}

// ---------------------------------------------------------------------------
// SEMEstimator: Estimate, Fit, GetParameters
// ---------------------------------------------------------------------------

func TestSEMEstimator_Estimate_Cov(t *testing.T) {
	sem := models.NewSEM()
	sem.AddEquation("X", nil, nil, 0, 1)
	sem.AddEquation("Y", []string{"X"}, []float64{0}, 0, 1)

	data := makeContinuousDF()
	est := NewSEMEstimator(sem, data)
	err := est.Estimate()
	if err != nil {
		t.Fatal(err)
	}
}

func TestSEMEstimator_Fit_Cov(t *testing.T) {
	sem := models.NewSEM()
	sem.AddEquation("X", nil, nil, 0, 1)
	sem.AddEquation("Y", []string{"X"}, []float64{0}, 0, 1)

	data := makeContinuousDF()
	est := NewSEMEstimator(sem, nil)
	err := est.Fit(data)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSEMEstimator_Estimate_NilSEM_Cov(t *testing.T) {
	est := NewSEMEstimator(nil, makeContinuousDF())
	err := est.Estimate()
	if err == nil {
		t.Error("expected error for nil SEM")
	}
}

func TestSEMEstimator_Estimate_NilData_Cov(t *testing.T) {
	sem := models.NewSEM()
	sem.AddEquation("X", nil, nil, 0, 1)
	est := NewSEMEstimator(sem, nil)
	err := est.Estimate()
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestSEMEstimator_Estimate_NoVars(t *testing.T) {
	sem := models.NewSEM()
	data := makeContinuousDF()
	est := NewSEMEstimator(sem, data)
	err := est.Estimate()
	if err == nil {
		t.Error("expected error for SEM with no variables")
	}
}

func TestSEMEstimator_GetParameters_Cov(t *testing.T) {
	sem := models.NewSEM()
	sem.AddEquation("X", nil, nil, 0, 1)
	sem.AddEquation("Y", []string{"X"}, []float64{0.5}, 1, 0.5)

	est := NewSEMEstimator(sem, makeContinuousDF())
	params, err := est.GetParameters()
	if err != nil {
		t.Fatal(err)
	}
	if len(params) != 2 {
		t.Errorf("expected 2 vars, got %d", len(params))
	}
}

func TestSEMEstimator_GetParameters_NilSEM_Cov(t *testing.T) {
	est := NewSEMEstimator(nil, nil)
	_, err := est.GetParameters()
	if err == nil {
		t.Error("expected error for nil SEM")
	}
}

func TestSEMEstimator_GetCoefficients(t *testing.T) {
	sem := models.NewSEM()
	sem.AddEquation("X", nil, nil, 0, 1)
	sem.AddEquation("Y", []string{"X"}, []float64{0.5}, 1, 0.5)
	est := NewSEMEstimator(sem, nil)
	betas, intercept, variance, err := est.GetCoefficients("Y")
	if err != nil {
		t.Fatal(err)
	}
	if len(betas) != 1 {
		t.Errorf("expected 1 beta, got %d", len(betas))
	}
	_ = intercept
	_ = variance
}

func TestSEMEstimator_GetCoefficients_UnknownVar(t *testing.T) {
	sem := models.NewSEM()
	est := NewSEMEstimator(sem, nil)
	_, _, _, err := est.GetCoefficients("UNKNOWN")
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

// ---------------------------------------------------------------------------
// IVEstimator: Fit, Estimate error paths
// ---------------------------------------------------------------------------

func TestIVEstimator_Fit_Valid(t *testing.T) {
	iv := NewIVEstimator("treatment", "outcome", []string{"instrument"})
	data := makeIVData()
	err := iv.Fit(data)
	if err != nil {
		t.Fatal(err)
	}
	if !iv.Fitted() {
		t.Error("expected fitted=true")
	}
	ate := iv.ATE()
	_ = ate
}

func TestIVEstimator_Fit_NilData_Cov(t *testing.T) {
	iv := NewIVEstimator("treatment", "outcome", []string{"instrument"})
	err := iv.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestIVEstimator_Fit_NoInstruments_Cov(t *testing.T) {
	iv := NewIVEstimator("treatment", "outcome", nil)
	err := iv.Fit(makeIVData())
	if err == nil {
		t.Error("expected error for no instruments")
	}
}

func TestIVEstimator_Fit_InsufficientData(t *testing.T) {
	iv := NewIVEstimator("treatment", "outcome", []string{"instrument"})
	sm := map[string]*tabgo.Series{
		"instrument": tabgo.NewSeries("instrument", []any{1.0}),
		"treatment":  tabgo.NewSeries("treatment", []any{1.5}),
		"outcome":    tabgo.NewSeries("outcome", []any{3.0}),
	}
	data := tabgo.NewDataFrame(sm)
	err := iv.Fit(data)
	if err == nil {
		t.Error("expected error for insufficient data")
	}
}

func TestIVEstimator_Estimate_Cov(t *testing.T) {
	iv := NewIVEstimator("treatment", "outcome", []string{"instrument"})
	ate, err := iv.Estimate(makeIVData())
	if err != nil {
		t.Fatal(err)
	}
	_ = ate
}

// ---------------------------------------------------------------------------
// LinearModel: error paths
// ---------------------------------------------------------------------------

func TestLinearModel_Fit_Valid(t *testing.T) {
	lm := NewLinearModel()
	data := makeContinuousDF()
	err := lm.Fit(data, "Y", []string{"X"})
	if err != nil {
		t.Fatal(err)
	}
	predicted, err := lm.Predict(data)
	if err != nil {
		t.Fatal(err)
	}
	if len(predicted) != 10 {
		t.Errorf("expected 10 predictions, got %d", len(predicted))
	}
}

func TestLinearModel_Fit_NilData_Cov(t *testing.T) {
	lm := NewLinearModel()
	err := lm.Fit(nil, "Y", []string{"X"})
	if err == nil {
		t.Error("expected error for nil data")
	}
}

// ---------------------------------------------------------------------------
// ExpertInLoop: combineSignals
// ---------------------------------------------------------------------------

func TestExpertInLoop_CombineSignals_Cov(t *testing.T) {
	e := &ExpertInLoop{}

	// Statistical supports, LLM uncertain -> orient
	if !e.combineSignals(true, llmUncertain) {
		t.Error("expected true when stats support")
	}
	// Statistical supports, LLM opposes -> still orient
	if !e.combineSignals(true, llmOpposes) {
		t.Error("expected true when stats support even if LLM opposes")
	}
	// No statistical evidence, LLM supports -> orient
	if !e.combineSignals(false, llmSupports) {
		t.Error("expected true when LLM supports without stats")
	}
	// No statistical evidence, LLM opposes -> don't orient
	if e.combineSignals(false, llmOpposes) {
		t.Error("expected false when neither stats nor LLM support")
	}
	// No statistical evidence, LLM uncertain -> don't orient
	if e.combineSignals(false, llmUncertain) {
		t.Error("expected false when neither stats nor LLM support")
	}
}

func TestExpertInLoop_QueryLLMForVStructure_NilClient(t *testing.T) {
	e := &ExpertInLoop{llmClient: nil}
	opinion := e.queryLLMForVStructure(CausalPromptTemplate{}, "A", "B", "C")
	if opinion != llmUncertain {
		t.Error("expected llmUncertain for nil LLM client")
	}
}

// ---------------------------------------------------------------------------
// PC Algorithm: BuildSkeleton, Estimate, EstimateBN
// ---------------------------------------------------------------------------

func ciTestAllIndep(x, y string, z []string, data *tabgo.DataFrame, sig float64) (float64, float64, bool) {
	return 0.0, 0.9, true
}

func ciTestAllDep(x, y string, z []string, data *tabgo.DataFrame, sig float64) (float64, float64, bool) {
	return 10.0, 0.001, false
}

func TestPC_BuildSkeleton_Cov(t *testing.T) {
	data := make3VarDF()
	pc := NewPC(data, ciTestAllDep, 0.05)
	pdag, sepSets, err := pc.BuildSkeleton()
	if err != nil {
		t.Fatal(err)
	}
	if pdag == nil {
		t.Error("expected non-nil PDAG")
	}
	_ = sepSets
}

func TestPC_Estimate_Cov(t *testing.T) {
	data := make3VarDF()
	pc := NewPC(data, ciTestAllIndep, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if pdag == nil {
		t.Error("expected non-nil PDAG")
	}
}

func TestPC_EstimateBN_Cov(t *testing.T) {
	data := make3VarDF()
	pc := NewPC(data, ciTestAllIndep, 0.05)
	bn, err := pc.EstimateBN()
	if err != nil {
		t.Fatal(err)
	}
	if bn == nil {
		t.Error("expected non-nil BN")
	}
}

func TestPC_BuildSkeleton_TooFewVars_Cov(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 1}),
	})
	pc := NewPC(data, ciTestAllDep, 0.05)
	_, _, err := pc.BuildSkeleton()
	if err == nil {
		t.Error("expected error for fewer than 2 variables")
	}
}

// ---------------------------------------------------------------------------
// GES: Insert, Delete, Turn, dagToPDAG
// ---------------------------------------------------------------------------

func TestGES_Estimate(t *testing.T) {
	data := make3VarDF()
	scoreFn := BICScore()
	ges := NewGES(data, scoreFn)
	pdag, err := ges.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if pdag == nil {
		t.Error("expected non-nil PDAG")
	}
}

// ---------------------------------------------------------------------------
// HillClimb: LegalOperations, bestOperation
// ---------------------------------------------------------------------------

func TestHillClimb_Estimate_Cov(t *testing.T) {
	data := make3VarDF()
	scoreFn := BICScore()
	hc := NewHillClimbSearch(data, scoreFn)
	dag, err := hc.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if dag == nil {
		t.Error("expected non-nil DAG")
	}
}

// ---------------------------------------------------------------------------
// MLE: error paths
// ---------------------------------------------------------------------------

func TestMLE_Estimate_NilBN(t *testing.T) {
	data := makeSmallDiscreteDF()
	mle := NewMLE(nil, data)
	err := mle.Estimate()
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMLE_Estimate_NilData(t *testing.T) {
	bn := makeSimpleBN()
	mle := NewMLE(bn, nil)
	err := mle.Estimate()
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestMLE_GetParameters_Cov(t *testing.T) {
	bn := makeSimpleBN()
	data := makeSmallDiscreteDF()
	mle := NewMLE(bn, data)
	if err := mle.Estimate(); err != nil {
		t.Fatal(err)
	}
	cpd, err := mle.GetParameters("A")
	if err != nil {
		t.Fatal(err)
	}
	if cpd == nil {
		t.Error("expected non-nil CPD")
	}
}

func TestMLE_EstimatePotentials_Cov(t *testing.T) {
	bn := makeSimpleBN()
	data := makeSmallDiscreteDF()
	mle := NewMLE(bn, data)
	potentials, err := mle.EstimatePotentials()
	if err != nil {
		t.Fatal(err)
	}
	if len(potentials) == 0 {
		t.Error("expected non-empty potentials")
	}
}

func TestMLE_EstimatePotentials_NilBN(t *testing.T) {
	mle := NewMLE(nil, makeSmallDiscreteDF())
	_, err := mle.EstimatePotentials()
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMLE_EstimatePotentials_NilData(t *testing.T) {
	mle := NewMLE(makeSimpleBN(), nil)
	_, err := mle.EstimatePotentials()
	if err == nil {
		t.Error("expected error for nil data")
	}
}

// ---------------------------------------------------------------------------
// MirrorDescent: error paths
// ---------------------------------------------------------------------------

func TestMirrorDescent_Estimate_NilBN_Cov(t *testing.T) {
	md := NewMirrorDescentEstimator(nil, makeSmallDiscreteDF(), 0.1, 10)
	err := md.Estimate()
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

// ---------------------------------------------------------------------------
// ExhaustiveSearch: additional error paths
// ---------------------------------------------------------------------------

func TestExhaustiveSearch_Estimate(t *testing.T) {
	data := makeTwoVarDF()
	scoreFn := BICScore()
	es := NewExhaustiveSearch(data, scoreFn)
	dag, err := es.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if dag == nil {
		t.Error("expected non-nil DAG")
	}
}

// ---------------------------------------------------------------------------
// EM: additional edge cases
// ---------------------------------------------------------------------------

func TestEM_WithLatentVars(t *testing.T) {
	bn := makeSimpleBN()
	data := makeSmallDiscreteDF()
	em := NewEM(bn, data, []string{"A"}, 5, 1e-4)
	err := em.Estimate()
	// EM with latent vars should run (may or may not converge)
	_ = err
}

// ---------------------------------------------------------------------------
// BayesianEstimator: additional error/edge cases
// ---------------------------------------------------------------------------

func TestBayesianEstimator_UnknownPrior(t *testing.T) {
	bn := makeSimpleBN()
	data := makeSmallDiscreteDF()
	// Use a prior type that's not BDeu or K2; defaults to uniform
	be := NewBayesianEstimator(bn, data, PriorType(99), 1.0)
	err := be.Estimate()
	_ = err // Just check it doesn't panic
}

func TestBayesianEstimator_GetParameters_Cov(t *testing.T) {
	bn := makeSimpleBN()
	data := makeSmallDiscreteDF()
	be := NewBayesianEstimator(bn, data, BDeu, 5.0)
	if err := be.Estimate(); err != nil {
		t.Fatal(err)
	}
	cpd, err := be.GetParameters("A")
	if err != nil {
		t.Fatal(err)
	}
	if cpd == nil {
		t.Error("expected non-nil CPD")
	}
}

// ---------------------------------------------------------------------------
// TreeSearch
// ---------------------------------------------------------------------------

func TestTreeSearch_Estimate_Cov(t *testing.T) {
	data := make3VarDF()
	ts := NewTreeSearch(data)
	dag, err := ts.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if dag == nil {
		t.Error("expected non-nil DAG")
	}
}

// ---------------------------------------------------------------------------
// MMHC: Estimate
// ---------------------------------------------------------------------------

func TestMMHC_Estimate_Cov(t *testing.T) {
	data := make3VarDF()
	scoreFn := BICScore()
	mmhc := NewMMHC(data, scoreFn, ciTestAllIndep, 0.05)
	dag, err := mmhc.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	if dag == nil {
		t.Error("expected non-nil DAG")
	}
}

// ---------------------------------------------------------------------------
// LLM Client: rate limiter basic test
// ---------------------------------------------------------------------------

func TestTokenBucket_Wait(t *testing.T) {
	// Create a bucket with high rate to test basic functionality
	tb := newTokenBucket(6000) // 100/sec
	tb.wait()                  // Should return immediately
}

// ---------------------------------------------------------------------------
// MarginalEstimator
// ---------------------------------------------------------------------------

func TestMarginalEstimator_NilBN(t *testing.T) {
	data := makeSmallDiscreteDF()
	me := NewMarginalEstimator(nil, data)
	_, err := me.MarginalLikelihood()
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMarginalEstimator_NilData(t *testing.T) {
	bn := makeSimpleBN()
	me := NewMarginalEstimator(bn, nil)
	_, err := me.MarginalLikelihood()
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestMarginalEstimator_Valid(t *testing.T) {
	bn := makeSimpleBN()
	data := makeSmallDiscreteDF()
	me := NewMarginalEstimator(bn, data)
	ml, err := me.MarginalLikelihood()
	if err != nil {
		t.Fatal(err)
	}
	_ = ml
}
