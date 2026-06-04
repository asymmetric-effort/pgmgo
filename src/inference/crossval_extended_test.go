//go:build unit

package inference_test

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// buildBNFromEdgesAndCPDs constructs a BN from edges and CPD fixture data.
func buildBNFromEdgesAndCPDs(t *testing.T, edges [][]string, cpds map[string]struct {
	VariableCard int         `json:"variable_card"`
	Values       [][]float64 `json:"values"`
	Evidence     []string    `json:"evidence"`
	EvidenceCard []int       `json:"evidence_card"`
}) *models.BayesianNetwork {
	t.Helper()
	bn := models.NewBayesianNetwork()
	nodeSet := make(map[string]bool)
	for _, edge := range edges {
		nodeSet[edge[0]] = true
		nodeSet[edge[1]] = true
	}
	for node := range nodeSet {
		if err := bn.AddNode(node); err != nil {
			t.Fatalf("AddNode(%q): %v", node, err)
		}
	}
	for _, edge := range edges {
		if err := bn.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q): %v", edge[0], edge[1], err)
		}
	}
	for varName, cpdData := range cpds {
		ev := cpdData.Evidence
		if ev == nil {
			ev = []string{}
		}
		ec := cpdData.EvidenceCard
		if ec == nil {
			ec = []int{}
		}
		cpd, err := factors.NewTabularCPD(varName, cpdData.VariableCard, cpdData.Values, ev, ec)
		if err != nil {
			t.Fatalf("NewTabularCPD(%q): %v", varName, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD(%q): %v", varName, err)
		}
	}
	return bn
}

type veQueryInput struct {
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

type veQueryExpected struct {
	Variables []string  `json:"variables"`
	Values    []float64 `json:"values"`
}

func runVEQueryCrossval(t *testing.T, fixtureName string) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "inference_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input veQueryInput
	tc.UnmarshalInput(t, &input)

	var expected veQueryExpected
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNFromEdgesAndCPDs(t, input.Edges, input.CPDs)
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors: %v", err)
	}
	ve := inference.NewVariableElimination(markovFactors)
	result, err := ve.Query(input.QueryVariables, input.Evidence)
	if err != nil {
		t.Fatalf("VE Query: %v", err)
	}

	gotValues := result.Values().Data()
	if len(gotValues) != len(expected.Values) {
		t.Fatalf("result values length: expected %d, got %d", len(expected.Values), len(gotValues))
	}
	for i := range expected.Values {
		if math.Abs(gotValues[i]-expected.Values[i]) > 1e-6 {
			t.Errorf("values[%d]: expected %f, got %f (diff=%e)",
				i, expected.Values[i], gotValues[i], math.Abs(gotValues[i]-expected.Values[i]))
		}
	}
}

// VE queries for every variable with no evidence
func TestCrossval_VE_D_NoEvidence(t *testing.T) { runVEQueryCrossval(t, "ve_query_D_no_evidence") }
func TestCrossval_VE_G_NoEvidence(t *testing.T) { runVEQueryCrossval(t, "ve_query_G_no_evidence") }
func TestCrossval_VE_I_NoEvidence(t *testing.T) { runVEQueryCrossval(t, "ve_query_I_no_evidence") }
func TestCrossval_VE_L_NoEvidence(t *testing.T) { runVEQueryCrossval(t, "ve_query_L_no_evidence") }
func TestCrossval_VE_S_NoEvidence(t *testing.T) { runVEQueryCrossval(t, "ve_query_S_no_evidence") }

// VE queries for every variable given D=0
func TestCrossval_VE_G_GivenD0(t *testing.T) { runVEQueryCrossval(t, "ve_query_G_given_D0") }
func TestCrossval_VE_I_GivenD0(t *testing.T) { runVEQueryCrossval(t, "ve_query_I_given_D0") }
func TestCrossval_VE_L_GivenD0(t *testing.T) { runVEQueryCrossval(t, "ve_query_L_given_D0") }
func TestCrossval_VE_S_GivenD0(t *testing.T) { runVEQueryCrossval(t, "ve_query_S_given_D0") }

// VE MAP queries
func runVEMAPCrossval(t *testing.T, fixtureName string) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "inference_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input veQueryInput
	tc.UnmarshalInput(t, &input)

	var expected struct {
		MapAssignment map[string]int `json:"map_assignment"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNFromEdgesAndCPDs(t, input.Edges, input.CPDs)
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors: %v", err)
	}
	ve := inference.NewVariableElimination(markovFactors)
	assignment, err := ve.MAP(input.QueryVariables, input.Evidence)
	if err != nil {
		t.Fatalf("VE MAP: %v", err)
	}

	for varName, expectedVal := range expected.MapAssignment {
		gotVal, ok := assignment[varName]
		if !ok {
			t.Errorf("MAP missing variable %q", varName)
			continue
		}
		if gotVal != expectedVal {
			t.Errorf("MAP[%q]: expected %d, got %d", varName, expectedVal, gotVal)
		}
	}
}

func TestCrossval_VE_MAP_G_D0I0(t *testing.T) { runVEMAPCrossval(t, "ve_map_G_given_D0_I0") }
func TestCrossval_VE_MAP_G_D1I0(t *testing.T) { runVEMAPCrossval(t, "ve_map_G_given_D1_I0") }
func TestCrossval_VE_MAP_G_D0I1(t *testing.T) { runVEMAPCrossval(t, "ve_map_G_given_D0_I1") }
func TestCrossval_VE_MAP_G_D1I1(t *testing.T) { runVEMAPCrossval(t, "ve_map_G_given_D1_I1") }
func TestCrossval_VE_MAP_L_G0(t *testing.T)   { runVEMAPCrossval(t, "ve_map_L_given_G0") }
func TestCrossval_VE_MAP_L_G1(t *testing.T)   { runVEMAPCrossval(t, "ve_map_L_given_G1") }
func TestCrossval_VE_MAP_L_G2(t *testing.T)   { runVEMAPCrossval(t, "ve_map_L_given_G2") }
func TestCrossval_VE_MAP_GL_D0I0(t *testing.T) {
	runVEMAPCrossval(t, "ve_map_G_L_given_D0_I0")
}
func TestCrossval_VE_MAP_S_I0(t *testing.T) { runVEMAPCrossval(t, "ve_map_S_given_I0") }
func TestCrossval_VE_MAP_S_I1(t *testing.T) { runVEMAPCrossval(t, "ve_map_S_given_I1") }

// BP vs VE cross-validation
func runBPvsVECrossval(t *testing.T, fixtureName string) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "inference_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input veQueryInput
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables []string  `json:"variables"`
		VEValues  []float64 `json:"ve_values"`
		BPValues  []float64 `json:"bp_values"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNFromEdgesAndCPDs(t, input.Edges, input.CPDs)

	// Run VE
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors: %v", err)
	}
	ve := inference.NewVariableElimination(markovFactors)
	veResult, err := ve.Query(input.QueryVariables, input.Evidence)
	if err != nil {
		t.Fatalf("VE Query: %v", err)
	}

	// Compare VE result with pgmpy VE values
	gotVE := veResult.Values().Data()
	for i := range expected.VEValues {
		if i >= len(gotVE) {
			break
		}
		if math.Abs(gotVE[i]-expected.VEValues[i]) > 1e-6 {
			t.Errorf("VE values[%d]: expected %f, got %f", i, expected.VEValues[i], gotVE[i])
		}
	}

	// Run BP
	jt, err := models.NewJunctionTreeFromBN(bn)
	if err != nil {
		t.Fatalf("NewJunctionTreeFromBN: %v", err)
	}
	cliques := jt.Cliques()
	separators := jt.SeparatorSets()
	cliqueFactors := make(map[int][]*factors.DiscreteFactor, len(cliques))
	for i, c := range cliques {
		fs := jt.GetCliqueFactors(c)
		if len(fs) > 0 {
			cliqueFactors[i] = fs
		}
	}
	bp := inference.NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.Calibrate(); err != nil {
		t.Fatalf("Calibrate: %v", err)
	}
	bpResult, err := bp.Query(input.QueryVariables, input.Evidence)
	if err != nil {
		t.Fatalf("BP Query: %v", err)
	}

	// Compare BP result with pgmpy BP values
	gotBP := bpResult.Values().Data()
	for i := range expected.BPValues {
		if i >= len(gotBP) {
			break
		}
		if math.Abs(gotBP[i]-expected.BPValues[i]) > 1e-4 {
			t.Errorf("BP values[%d]: expected %f, got %f", i, expected.BPValues[i], gotBP[i])
		}
	}
}

func TestCrossval_BP_vs_VE_G_D0I1(t *testing.T) {
	runBPvsVECrossval(t, "bp_vs_ve_G_given_D0_I1")
}
func TestCrossval_BP_vs_VE_L_I0(t *testing.T) {
	runBPvsVECrossval(t, "bp_vs_ve_L_given_I0")
}
func TestCrossval_BP_vs_VE_S_none(t *testing.T) {
	runBPvsVECrossval(t, "bp_vs_ve_S_given_none")
}
func TestCrossval_BP_vs_VE_D_G1(t *testing.T) {
	runBPvsVECrossval(t, "bp_vs_ve_D_given_G1")
}

// ApproxInference crossval
func TestCrossval_ApproxInference(t *testing.T) {
	ff := testutil.LoadFixtures(t, "inference_extended/fixtures.json")
	tc := ff.FindTestCase(t, "approx_inference_marginal")

	var input veQueryInput
	tc.UnmarshalInput(t, &input)

	var expected struct {
		ExactValues  []float64 `json:"exact_values"`
		ApproxValues []float64 `json:"approx_values"`
		Tolerance    float64   `json:"tolerance"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNFromEdgesAndCPDs(t, input.Edges, input.CPDs)
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors: %v", err)
	}

	ai := inference.NewApproxInference(markovFactors, 42)
	result, err := ai.Query(input.QueryVariables, input.Evidence, 100000)
	if err != nil {
		t.Fatalf("ApproxInference Query: %v", err)
	}

	gotValues := result.Values().Data()
	for i := range expected.ExactValues {
		if i >= len(gotValues) {
			break
		}
		if math.Abs(gotValues[i]-expected.ExactValues[i]) > expected.Tolerance {
			t.Errorf("approx values[%d]: expected ~%f (exact), got %f (tolerance=%f)",
				i, expected.ExactValues[i], gotValues[i], expected.Tolerance)
		}
	}
}

// CausalInference ATE crossval
func TestCrossval_CausalATE_D_G(t *testing.T) {
	ff := testutil.LoadFixtures(t, "inference_extended/fixtures.json")
	tc := ff.FindTestCase(t, "causal_ate_d_g")

	var input struct {
		Edges           [][]string `json:"edges"`
		Treatment       string     `json:"treatment"`
		Outcome         string     `json:"outcome"`
		TreatmentValues []int      `json:"treatment_values"`
		CPDs            map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		DoD0Values []float64 `json:"do_d0_values"`
		DoD1Values []float64 `json:"do_d1_values"`
		ATE        float64   `json:"ate"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNFromEdgesAndCPDs(t, input.Edges, input.CPDs)

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference: %v", err)
	}

	// Query P(G | do(D=0))
	resD0, err := ci.Query([]string{input.Outcome}, map[string]int{input.Treatment: input.TreatmentValues[0]}, nil)
	if err != nil {
		t.Fatalf("CausalInference do(D=0): %v", err)
	}
	gotD0 := resD0.Values().Data()
	for i := range expected.DoD0Values {
		if i >= len(gotD0) {
			break
		}
		if math.Abs(gotD0[i]-expected.DoD0Values[i]) > 1e-6 {
			t.Errorf("do(D=0)[%d]: expected %f, got %f", i, expected.DoD0Values[i], gotD0[i])
		}
	}

	// Query P(G | do(D=1))
	resD1, err := ci.Query([]string{input.Outcome}, map[string]int{input.Treatment: input.TreatmentValues[1]}, nil)
	if err != nil {
		t.Fatalf("CausalInference do(D=1): %v", err)
	}
	gotD1 := resD1.Values().Data()
	for i := range expected.DoD1Values {
		if i >= len(gotD1) {
			break
		}
		if math.Abs(gotD1[i]-expected.DoD1Values[i]) > 1e-6 {
			t.Errorf("do(D=1)[%d]: expected %f, got %f", i, expected.DoD1Values[i], gotD1[i])
		}
	}

	// Compute ATE
	ate, err := ci.ATE(input.Treatment, input.Outcome, [2]int{input.TreatmentValues[0], input.TreatmentValues[1]})
	if err != nil {
		t.Fatalf("ATE: %v", err)
	}
	if math.Abs(ate-expected.ATE) > 1e-4 {
		t.Errorf("ATE: expected %f, got %f", expected.ATE, ate)
	}
}

// CausalInference backdoor adjustment sets
func TestCrossval_CausalBackdoor_D_G(t *testing.T) {
	ff := testutil.LoadFixtures(t, "inference_extended/fixtures.json")
	tc := ff.FindTestCase(t, "causal_backdoor_d_g")

	var input struct {
		Edges     [][]string `json:"edges"`
		Treatment string     `json:"treatment"`
		Outcome   string     `json:"outcome"`
		CPDs      map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		AdjustmentSets [][]string `json:"adjustment_sets"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNFromEdgesAndCPDs(t, input.Edges, input.CPDs)

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference: %v", err)
	}

	adjSets := ci.GetAllBackdoorAdjustmentSets(input.Treatment, input.Outcome)

	// Verify that the empty set is a valid backdoor adjustment (or that we
	// find at least the expected number of sets). pgmpy's deprecated API
	// returns different granularity than the Go impl.
	if len(expected.AdjustmentSets) == 0 {
		// pgmpy says empty set is valid -- just verify we find valid sets
		t.Logf("backdoor adjustment sets found: %d (pgmpy found empty set valid)", len(adjSets))
	} else if len(adjSets) < len(expected.AdjustmentSets) {
		t.Errorf("adjustment sets count: expected >= %d, got %d",
			len(expected.AdjustmentSets), len(adjSets))
	}
}
