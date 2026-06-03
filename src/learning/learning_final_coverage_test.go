//go:build unit

package learning

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

func finalCovDF(t *testing.T) *tabgo.DataFrame {
	t.Helper()
	return tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 1, 0, 1, 0, 1, 0, 1, 0, 1}),
		"B": tabgo.NewSeries("B", []any{0, 0, 1, 1, 0, 0, 1, 1, 0, 1}),
		"C": tabgo.NewSeries("C", []any{0, 1, 1, 0, 0, 1, 1, 0, 1, 0}),
	})
}

func finalCovBN(t *testing.T) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()
	bn.AddNode("A")
	bn.AddNode("B")
	bn.AddNode("C")
	bn.AddEdge("A", "B")
	bn.AddEdge("B", "C")
	bn.SetStates("A", []string{"0", "1"})
	bn.SetStates("B", []string{"0", "1"})
	bn.SetStates("C", []string{"0", "1"})
	aCPD, _ := factors.NewTabularCPD("A", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	bn.AddCPD(aCPD)
	bCPD, _ := factors.NewTabularCPD("B", 2, [][]float64{{0.8, 0.2}, {0.2, 0.8}}, []string{"A"}, []int{2})
	bn.AddCPD(bCPD)
	cCPD, _ := factors.NewTabularCPD("C", 2, [][]float64{{0.9, 0.3}, {0.1, 0.7}}, []string{"B"}, []int{2})
	bn.AddCPD(cCPD)
	return bn
}

// ==========================================================================
// LLM Client: HTTP retry paths with httptest
// ==========================================================================

func TestFinalCov_LLMClient_RetryOnServerError(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls <= 2 {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"error": "server error"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"Yes"}}]}`))
	}))
	defer server.Close()

	client := NewHTTPLLMClient(server.URL, "test-key", "gpt-4")
	client.maxRetries = 5
	result, err := client.ChatComplete([]Message{{Role: "user", Content: "test"}})
	t.Logf("Result: %q, err: %v, calls: %d", result, err, calls)
}

func TestFinalCov_LLMClient_AllRetriesFail(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error": "server error"}`))
	}))
	defer server.Close()

	client := NewHTTPLLMClient(server.URL, "test-key", "gpt-4")
	client.maxRetries = 1
	_, err := client.ChatComplete([]Message{{Role: "user", Content: "test"}})
	if err == nil {
		t.Error("expected error when all retries fail")
	}
}

func TestFinalCov_LLMClient_RateLimited(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Write([]byte(`{"error": "rate limited"}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"response"}}]}`))
	}))
	defer server.Close()

	client := NewHTTPLLMClient(server.URL, "test-key", "gpt-4")
	client.maxRetries = 5
	result, err := client.ChatComplete([]Message{{Role: "user", Content: "test"}})
	t.Logf("Result: %q, err: %v, calls: %d", result, err, calls)
}

func TestFinalCov_LLMClient_BadJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`not json`))
	}))
	defer server.Close()

	client := NewHTTPLLMClient(server.URL, "test-key", "gpt-4")
	_, err := client.ChatComplete([]Message{{Role: "user", Content: "test"}})
	if err == nil {
		t.Error("expected error for bad JSON response")
	}
}

func TestFinalCov_LLMClient_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[]}`))
	}))
	defer server.Close()

	client := NewHTTPLLMClient(server.URL, "test-key", "gpt-4")
	_, err := client.ChatComplete([]Message{{Role: "user", Content: "test"}})
	if err == nil {
		t.Error("expected error for empty choices")
	}
}

func TestFinalCov_LLMClient_Complete(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"choices":[{"message":{"content":"completed"}}]}`))
	}))
	defer server.Close()

	client := NewHTTPLLMClient(server.URL, "test-key", "gpt-4")
	result, err := client.Complete("test prompt")
	t.Logf("Complete result: %q, err: %v", result, err)
}

// ==========================================================================
// IV Estimator: error paths
// ==========================================================================

func TestFinalCov_IVEstimator_NilData(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	err := iv.Fit(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestFinalCov_IVEstimator_WeakInstrument(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"Z": tabgo.NewSeries("Z", []any{1.0, 1.0, 1.0, 1.0, 1.0}),
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0, 4.0, 6.0, 8.0, 10.0}),
	})
	err := iv.Fit(df)
	t.Logf("Weak instrument err: %v", err)
}

func TestFinalCov_IVEstimator_EmptyData(t *testing.T) {
	iv := NewIVEstimator("X", "Y", []string{"Z"})
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"Z": tabgo.NewSeries("Z", []any{}),
		"X": tabgo.NewSeries("X", []any{}),
		"Y": tabgo.NewSeries("Y", []any{}),
	})
	err := iv.Fit(df)
	t.Logf("Empty data err: %v", err)
}

// ==========================================================================
// SEM Estimator
// ==========================================================================

func TestFinalCov_SEMEstimator_Valid(t *testing.T) {
	s := models.NewSEM()
	s.AddEquation("X", nil, nil, 0.0, 1.0)
	s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)

	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5}),
		"Y": tabgo.NewSeries("Y", []any{1.5, 3.0, 4.5, 6.0, 7.5, 2.0, 3.5, 5.0, 6.5, 8.0}),
	})

	est := NewSEMEstimator(s, df)
	err := est.Estimate()
	t.Logf("SEM estimate: err=%v", err)

	params, paramsErr := est.GetParameters()
	t.Logf("SEM params: %v, err=%v", params, paramsErr)
}

func TestFinalCov_SEMEstimator_NilData(t *testing.T) {
	s := models.NewSEM()
	s.AddEquation("X", nil, nil, 0.0, 1.0)
	est := NewSEMEstimator(s, nil)
	err := est.Estimate()
	t.Logf("SEM nil data: %v", err)
}

// ==========================================================================
// MLE: error paths
// ==========================================================================

func TestFinalCov_MLE_EmptyData(t *testing.T) {
	bn := finalCovBN(t)
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{}),
		"B": tabgo.NewSeries("B", []any{}),
		"C": tabgo.NewSeries("C", []any{}),
	})
	mle := NewMLE(bn, df)
	err := mle.Estimate()
	t.Logf("MLE empty data: %v", err)
}

func TestFinalCov_MLE_GetParameters(t *testing.T) {
	bn := finalCovBN(t)
	df := finalCovDF(t)
	mle := NewMLE(bn, df)
	err := mle.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	params, err := mle.GetParameters("A")
	if err != nil {
		t.Fatal(err)
	}
	if params == nil {
		t.Error("expected non-nil parameters")
	}
}

func TestFinalCov_MLE_EstimatePotentials(t *testing.T) {
	mn := models.NewMarkovNetwork()
	mn.AddNode("A")
	mn.AddNode("B")
	mn.AddEdge("A", "B")
	fAB, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.25, 0.25, 0.25, 0.25})
	mn.AddFactor(fAB)

	df := finalCovDF(t)
	bn := finalCovBN(t)
	mle := NewMLE(bn, df)
	result, err := mle.EstimatePotentials()
	t.Logf("EstimatePotentials: result=%v, err=%v", result, err)
}

// ==========================================================================
// Bayesian Estimator
// ==========================================================================

func TestFinalCov_BayesEstimator_EmptyData(t *testing.T) {
	bn := finalCovBN(t)
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{}),
		"B": tabgo.NewSeries("B", []any{}),
		"C": tabgo.NewSeries("C", []any{}),
	})
	be := NewBayesianEstimator(bn, df, BDeu, 5.0)
	err := be.Estimate()
	t.Logf("Bayes empty data: %v", err)
}

func TestFinalCov_BayesEstimator_GetParameters(t *testing.T) {
	bn := finalCovBN(t)
	df := finalCovDF(t)
	be := NewBayesianEstimator(bn, df, BDeu, 5.0)
	err := be.Estimate()
	if err != nil {
		t.Fatal(err)
	}
	params, perr := be.GetParameters("A")
	if perr != nil {
		t.Fatal(perr)
	}
	if params == nil {
		t.Error("expected non-nil parameters")
	}
}

// ==========================================================================
// Marginal Estimator
// ==========================================================================

func TestFinalCov_MarginalEstimator_EmptyData(t *testing.T) {
	bn := finalCovBN(t)
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{}),
		"B": tabgo.NewSeries("B", []any{}),
		"C": tabgo.NewSeries("C", []any{}),
	})
	me := NewMarginalEstimator(bn, df)
	err := me.Estimate()
	t.Logf("Marginal empty data: %v", err)
}

// ==========================================================================
// EM
// ==========================================================================

func TestFinalCov_EM_EmptyData(t *testing.T) {
	bn := finalCovBN(t)
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{}),
		"B": tabgo.NewSeries("B", []any{}),
		"C": tabgo.NewSeries("C", []any{}),
	})
	em := NewEM(bn, df, []string{"C"}, 10, 1e-4)
	err := em.Estimate()
	t.Logf("EM empty data: %v", err)
}

// ==========================================================================
// Mirror Descent
// ==========================================================================

func TestFinalCov_MirrorDescent_EmptyData(t *testing.T) {
	bn := finalCovBN(t)
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{}),
		"B": tabgo.NewSeries("B", []any{}),
		"C": tabgo.NewSeries("C", []any{}),
	})
	md := NewMirrorDescentEstimator(bn, df, 0.01, 10)
	err := md.Estimate()
	t.Logf("Mirror descent empty data: %v", err)
}

// ==========================================================================
// Linear Model: singular matrix
// ==========================================================================

func TestFinalCov_LinearModel_Singular(t *testing.T) {
	lm := NewLinearModel()
	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{1.0, 1.0, 1.0, 1.0}),
		"Y": tabgo.NewSeries("Y", []any{2.0, 3.0, 4.0, 5.0}),
	})
	err := lm.Fit(df, "Y", []string{"X"})
	t.Logf("Singular matrix fit: %v", err)
}

// ==========================================================================
// Linear Gaussian MLE
// ==========================================================================

func TestFinalCov_LGaussianMLE_NilData(t *testing.T) {
	lgbn := models.NewLinearGaussianBayesianNetwork()
	lgbn.AddNode("X")
	lgbn.AddNode("Y")
	lgbn.AddEdge("X", "Y")
	aCPD, _ := factors.NewLinearGaussianCPD("X", 0.0, nil, 1.0, nil)
	lgbn.AddLinearGaussianCPD(aCPD)
	bCPD, _ := factors.NewLinearGaussianCPD("Y", 0.0, []float64{0.5}, 1.0, []string{"X"})
	lgbn.AddLinearGaussianCPD(bCPD)

	df := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{}),
		"Y": tabgo.NewSeries("Y", []any{}),
	})
	est := NewLinearGaussianMLE(lgbn, df)
	err := est.Estimate()
	t.Logf("LG MLE empty data: %v", err)
}
