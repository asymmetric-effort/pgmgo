//go:build unit

package inference_test

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// TestCrossval_CausalDoQuery_G validates our CausalInference.Query with
// do(D=0) against pgmpy's CausalInference output for P(G | do(D=0)).
func TestCrossval_CausalDoQuery_G(t *testing.T) {
	ff := testutil.LoadFixtures(t, "causal/fixtures.json")
	tc := ff.FindTestCase(t, "causal_do_query_g")

	var input struct {
		Edges          [][]string     `json:"edges"`
		QueryVariables []string       `json:"query_variables"`
		DoVariables    map[string]int `json:"do_variables"`
		Evidence       map[string]int `json:"evidence"`
		CPDs           map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables []string  `json:"variables"`
		Values    []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBN(t, input.Edges, input.CPDs)

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference failed: %v", err)
	}

	result, err := ci.Query(input.QueryVariables, input.DoVariables, input.Evidence)
	if err != nil {
		t.Fatalf("CausalInference.Query failed: %v", err)
	}

	gotVars := result.Variables()
	if len(gotVars) != len(expected.Variables) {
		t.Fatalf("result variables: expected %v, got %v", expected.Variables, gotVars)
	}
	for i := range expected.Variables {
		if gotVars[i] != expected.Variables[i] {
			t.Errorf("result variables[%d]: expected %q, got %q", i, expected.Variables[i], gotVars[i])
		}
	}

	gotValues := result.Values().Data()
	if len(gotValues) != len(expected.Values) {
		t.Fatalf("result values length: expected %d, got %d", len(expected.Values), len(gotValues))
	}
	for i := range expected.Values {
		if math.Abs(gotValues[i]-expected.Values[i]) > 1e-6 {
			t.Errorf("result values[%d]: expected %f, got %f (diff=%e)",
				i, expected.Values[i], gotValues[i], math.Abs(gotValues[i]-expected.Values[i]))
		}
	}
}

// TestCrossval_CausalDoQuery_L validates P(L | do(D=0)) against pgmpy.
func TestCrossval_CausalDoQuery_L(t *testing.T) {
	ff := testutil.LoadFixtures(t, "causal/fixtures.json")
	tc := ff.FindTestCase(t, "causal_do_query_l")

	var input struct {
		Edges          [][]string     `json:"edges"`
		QueryVariables []string       `json:"query_variables"`
		DoVariables    map[string]int `json:"do_variables"`
		Evidence       map[string]int `json:"evidence"`
		CPDs           map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables []string  `json:"variables"`
		Values    []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBN(t, input.Edges, input.CPDs)

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference failed: %v", err)
	}

	result, err := ci.Query(input.QueryVariables, input.DoVariables, input.Evidence)
	if err != nil {
		t.Fatalf("CausalInference.Query failed: %v", err)
	}

	gotValues := result.Values().Data()
	if len(gotValues) != len(expected.Values) {
		t.Fatalf("result values length: expected %d, got %d", len(expected.Values), len(gotValues))
	}
	for i := range expected.Values {
		if math.Abs(gotValues[i]-expected.Values[i]) > 1e-6 {
			t.Errorf("result values[%d]: expected %f, got %f (diff=%e)",
				i, expected.Values[i], gotValues[i], math.Abs(gotValues[i]-expected.Values[i]))
		}
	}
}

// TestCrossval_ObservationalQuery_G validates that observational P(G | D=0)
// matches pgmpy's VariableElimination output via our CausalInference
// (with empty do-set, it should reduce to standard inference).
func TestCrossval_ObservationalQuery_G(t *testing.T) {
	ff := testutil.LoadFixtures(t, "causal/fixtures.json")
	tc := ff.FindTestCase(t, "observational_query_g")

	var input struct {
		Edges          [][]string     `json:"edges"`
		QueryVariables []string       `json:"query_variables"`
		Evidence       map[string]int `json:"evidence"`
		CPDs           map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables []string  `json:"variables"`
		Values    []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBN(t, input.Edges, input.CPDs)

	// Use CausalInference with empty do-set to get observational result.
	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference failed: %v", err)
	}

	result, err := ci.Query(input.QueryVariables, nil, input.Evidence)
	if err != nil {
		t.Fatalf("CausalInference.Query (observational) failed: %v", err)
	}

	gotValues := result.Values().Data()
	if len(gotValues) != len(expected.Values) {
		t.Fatalf("result values length: expected %d, got %d", len(expected.Values), len(gotValues))
	}
	for i := range expected.Values {
		if math.Abs(gotValues[i]-expected.Values[i]) > 1e-6 {
			t.Errorf("result values[%d]: expected %f, got %f (diff=%e)",
				i, expected.Values[i], gotValues[i], math.Abs(gotValues[i]-expected.Values[i]))
		}
	}
}
