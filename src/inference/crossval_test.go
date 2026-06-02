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

// buildStudentBN constructs a Bayesian network from the fixture input data
// (edges and CPDs) and returns the network.
func buildStudentBN(t *testing.T, edges [][]string, cpds map[string]struct {
	VariableCard int         `json:"variable_card"`
	Values       [][]float64 `json:"values"`
	Evidence     []string    `json:"evidence"`
	EvidenceCard []int       `json:"evidence_card"`
}) *models.BayesianNetwork {
	t.Helper()

	bn := models.NewBayesianNetwork()

	// Add all nodes from edges.
	nodeSet := make(map[string]bool)
	for _, edge := range edges {
		nodeSet[edge[0]] = true
		nodeSet[edge[1]] = true
	}
	for node := range nodeSet {
		if err := bn.AddNode(node); err != nil {
			t.Fatalf("AddNode(%q) failed: %v", node, err)
		}
	}

	// Add edges.
	for _, edge := range edges {
		if err := bn.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q) failed: %v", edge[0], edge[1], err)
		}
	}

	// Add CPDs.
	for varName, cpdData := range cpds {
		evidence := cpdData.Evidence
		if evidence == nil {
			evidence = []string{}
		}
		evidenceCard := cpdData.EvidenceCard
		if evidenceCard == nil {
			evidenceCard = []int{}
		}

		cpd, err := factors.NewTabularCPD(varName, cpdData.VariableCard, cpdData.Values,
			evidence, evidenceCard)
		if err != nil {
			t.Fatalf("NewTabularCPD(%q) failed: %v", varName, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD(%q) failed: %v", varName, err)
		}
	}

	return bn
}

func TestCrossval_VariableEliminationQuery(t *testing.T) {
	ff := testutil.LoadFixtures(t, "inference/fixtures.json")
	tc := ff.FindTestCase(t, "variable_elimination_query")

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

	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors failed: %v", err)
	}

	ve := inference.NewVariableElimination(markovFactors)
	result, err := ve.Query(input.QueryVariables, input.Evidence)
	if err != nil {
		t.Fatalf("VE Query failed: %v", err)
	}

	// Verify result variables match expected.
	gotVars := result.Variables()
	if len(gotVars) != len(expected.Variables) {
		t.Fatalf("result variables: expected %v, got %v", expected.Variables, gotVars)
	}
	for i := range expected.Variables {
		if gotVars[i] != expected.Variables[i] {
			t.Errorf("result variables[%d]: expected %q, got %q", i, expected.Variables[i], gotVars[i])
		}
	}

	// Compare result values against pgmpy's expected values with tolerance.
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

func TestCrossval_VariableEliminationMAP(t *testing.T) {
	ff := testutil.LoadFixtures(t, "inference/fixtures.json")
	tc := ff.FindTestCase(t, "variable_elimination_map")

	// The MAP test case does not include CPDs in the fixture input,
	// so we load them from the query test case which uses the same student network.
	queryTC := ff.FindTestCase(t, "variable_elimination_query")

	var queryInput struct {
		CPDs map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	queryTC.UnmarshalInput(t, &queryInput)

	var input struct {
		Edges          [][]string     `json:"edges"`
		QueryVariables []string       `json:"query_variables"`
		Evidence       map[string]int `json:"evidence"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		MapAssignment map[string]int `json:"map_assignment"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBN(t, input.Edges, queryInput.CPDs)

	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors failed: %v", err)
	}

	ve := inference.NewVariableElimination(markovFactors)
	assignment, err := ve.MAP(input.QueryVariables, input.Evidence)
	if err != nil {
		t.Fatalf("VE MAP failed: %v", err)
	}

	// Compare MAP assignment against expected.
	for varName, expectedVal := range expected.MapAssignment {
		gotVal, ok := assignment[varName]
		if !ok {
			t.Errorf("MAP assignment missing variable %q", varName)
			continue
		}
		if gotVal != expectedVal {
			t.Errorf("MAP assignment[%q]: expected %d, got %d", varName, expectedVal, gotVal)
		}
	}

	// Check no extra variables in assignment.
	for varName := range assignment {
		if _, ok := expected.MapAssignment[varName]; !ok {
			t.Errorf("MAP assignment has unexpected variable %q=%d", varName, assignment[varName])
		}
	}
}
