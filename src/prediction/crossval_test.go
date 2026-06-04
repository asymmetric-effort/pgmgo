//go:build unit

package prediction_test

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// buildBNForPrediction constructs a BN from fixture edges and CPDs.
func buildBNForPrediction(t *testing.T, edges [][]string, cpds map[string]struct {
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

func TestCrossval_CausalATE_D_L(t *testing.T) {
	ff := testutil.LoadFixtures(t, "prediction/fixtures.json")
	tc := ff.FindTestCase(t, "causal_ate_d_l")

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

	bn := buildBNForPrediction(t, input.Edges, input.CPDs)

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference: %v", err)
	}

	ate, err := ci.ATE(input.Treatment, input.Outcome,
		[2]int{input.TreatmentValues[0], input.TreatmentValues[1]})
	if err != nil {
		t.Fatalf("ATE: %v", err)
	}

	if math.Abs(ate-expected.ATE) > 1e-4 {
		t.Errorf("ATE(D, L): expected %f, got %f", expected.ATE, ate)
	}
}

func TestCrossval_CausalATE_Confounded(t *testing.T) {
	ff := testutil.LoadFixtures(t, "prediction/fixtures.json")
	tc := ff.FindTestCase(t, "causal_ate_confounded")

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
		DoX0Values             []float64  `json:"do_x0_values"`
		DoX1Values             []float64  `json:"do_x1_values"`
		ATE                    float64    `json:"ate"`
		BackdoorAdjustmentSets [][]string `json:"backdoor_adjustment_sets"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildBNForPrediction(t, input.Edges, input.CPDs)

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		t.Fatalf("NewCausalInference: %v", err)
	}

	// Verify do-calculus values
	resX0, err := ci.Query([]string{input.Outcome}, map[string]int{input.Treatment: input.TreatmentValues[0]}, nil)
	if err != nil {
		t.Fatalf("CausalInference do(X=0): %v", err)
	}
	gotX0 := resX0.Values().Data()
	for i := range expected.DoX0Values {
		if i >= len(gotX0) {
			break
		}
		if math.Abs(gotX0[i]-expected.DoX0Values[i]) > 1e-6 {
			t.Errorf("do(X=0)[%d]: expected %f, got %f", i, expected.DoX0Values[i], gotX0[i])
		}
	}

	resX1, err := ci.Query([]string{input.Outcome}, map[string]int{input.Treatment: input.TreatmentValues[1]}, nil)
	if err != nil {
		t.Fatalf("CausalInference do(X=1): %v", err)
	}
	gotX1 := resX1.Values().Data()
	for i := range expected.DoX1Values {
		if i >= len(gotX1) {
			break
		}
		if math.Abs(gotX1[i]-expected.DoX1Values[i]) > 1e-6 {
			t.Errorf("do(X=1)[%d]: expected %f, got %f", i, expected.DoX1Values[i], gotX1[i])
		}
	}

	// Verify ATE
	ate, err := ci.ATE(input.Treatment, input.Outcome,
		[2]int{input.TreatmentValues[0], input.TreatmentValues[1]})
	if err != nil {
		t.Fatalf("ATE: %v", err)
	}
	if math.Abs(ate-expected.ATE) > 1e-4 {
		t.Errorf("ATE(X, Y): expected %f, got %f", expected.ATE, ate)
	}

	// Verify backdoor adjustment sets exist
	adjSets := ci.GetAllBackdoorAdjustmentSets(input.Treatment, input.Outcome)
	if len(adjSets) != len(expected.BackdoorAdjustmentSets) {
		t.Errorf("backdoor sets count: expected %d, got %d",
			len(expected.BackdoorAdjustmentSets), len(adjSets))
	}
}
