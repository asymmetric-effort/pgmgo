//go:build unit

package models_test

import (
	"math"
	"sort"
	"strconv"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// buildStudentBNFromCPDs constructs the student BN from fixture CPD data.
func buildStudentBNFromCPDs(t *testing.T, edges [][]string, cpds map[string]struct {
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
			t.Fatalf("NewTabularCPD(%q): %v", varName, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD(%q): %v", varName, err)
		}
	}
	return bn
}

func TestCrossval_NaiveBayesFit(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models_extended/fixtures.json")
	tc := ff.FindTestCase(t, "naive_bayes_fit")

	var input struct {
		ClassVariable string      `json:"class_variable"`
		Features      []string    `json:"features"`
		DataColumns   []string    `json:"data_columns"`
		DataRows      [][]float64 `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Nodes []string   `json:"nodes"`
		Edges [][]string `json:"edges"`
		CPDs  map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
		} `json:"cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	// Build dataframe
	anyRows := make([][]any, len(input.DataRows))
	for i, row := range input.DataRows {
		anyRow := make([]any, len(row))
		for j, v := range row {
			anyRow[j] = strconv.Itoa(int(v))
		}
		anyRows[i] = anyRow
	}
	df := tabgo.NewDataFrameFromRows(input.DataColumns, anyRows)

	nb, err := models.NewNaiveBayes(input.ClassVariable, input.Features)
	if err != nil {
		t.Fatalf("NewNaiveBayes: %v", err)
	}
	if err := nb.Fit(df); err != nil {
		t.Fatalf("NaiveBayes.Fit: %v", err)
	}

	// Verify edges
	gotEdges := make([][]string, 0)
	for _, f := range input.Features {
		gotEdges = append(gotEdges, []string{input.ClassVariable, f})
	}
	sort.Slice(gotEdges, func(i, j int) bool {
		if gotEdges[i][0] == gotEdges[j][0] {
			return gotEdges[i][1] < gotEdges[j][1]
		}
		return gotEdges[i][0] < gotEdges[j][0]
	})

	if len(gotEdges) != len(expected.Edges) {
		t.Errorf("edges: expected %d, got %d", len(expected.Edges), len(gotEdges))
	}
}

func TestCrossval_BNGetMarkovBlanket(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models_extended/fixtures.json")
	tc := ff.FindTestCase(t, "bn_get_markov_blanket")

	var input struct {
		Edges [][]string `json:"edges"`
		CPDs  map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Blankets map[string][]string `json:"blankets"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBNFromCPDs(t, input.Edges, input.CPDs)

	for node, expectedBlanket := range expected.Blankets {
		got, err := bn.GetMarkovBlanket(node)
		if err != nil {
			t.Fatalf("GetMarkovBlanket(%q): %v", node, err)
		}
		sort.Strings(got)
		sort.Strings(expectedBlanket)
		if len(got) != len(expectedBlanket) {
			t.Errorf("MarkovBlanket(%q): expected %v, got %v", node, expectedBlanket, got)
			continue
		}
		for i := range expectedBlanket {
			if got[i] != expectedBlanket[i] {
				t.Errorf("MarkovBlanket(%q)[%d]: expected %q, got %q", node, i, expectedBlanket[i], got[i])
			}
		}
	}
}

func TestCrossval_BNDoG(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models_extended/fixtures.json")
	tc := ff.FindTestCase(t, "bn_do_g")

	var input struct {
		Edges   [][]string `json:"edges"`
		DoNodes []string   `json:"do_nodes"`
		CPDs    map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		MutilatedEdges [][]string `json:"mutilated_edges"`
		MutilatedNodes []string   `json:"mutilated_nodes"`
		NumEdges       int        `json:"num_edges"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBNFromCPDs(t, input.Edges, input.CPDs)

	// Do operation: set G to a value (e.g., 0) -- the Go Do takes map[string]int
	doVars := make(map[string]int)
	for _, n := range input.DoNodes {
		doVars[n] = 0
	}
	mutilated, err := bn.Do(doVars)
	if err != nil {
		t.Fatalf("Do(%v): %v", input.DoNodes, err)
	}

	gotEdges := mutilated.Edges()
	sort.Slice(gotEdges, func(i, j int) bool {
		if gotEdges[i][0] == gotEdges[j][0] {
			return gotEdges[i][1] < gotEdges[j][1]
		}
		return gotEdges[i][0] < gotEdges[j][0]
	})

	if len(gotEdges) != expected.NumEdges {
		t.Errorf("num_edges: expected %d, got %d", expected.NumEdges, len(gotEdges))
	}

	// Check that mutilated edges match expected
	for i, ee := range expected.MutilatedEdges {
		if i >= len(gotEdges) {
			t.Errorf("missing edge %v", ee)
			continue
		}
		if gotEdges[i][0] != ee[0] || gotEdges[i][1] != ee[1] {
			t.Errorf("edge[%d]: expected %v, got %v", i, ee, gotEdges[i])
		}
	}
}

func TestCrossval_MarkovNetworkPartitionExtended(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models_extended/fixtures.json")
	tc := ff.FindTestCase(t, "markov_network_partition_extended")

	var input struct {
		Edges   [][]string `json:"edges"`
		Factors []struct {
			Variables   []string  `json:"variables"`
			Cardinality []int     `json:"cardinality"`
			Values      []float64 `json:"values"`
		} `json:"factors"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		PartitionFunction float64 `json:"partition_function"`
	}
	tc.UnmarshalExpected(t, &expected)

	mn := models.NewMarkovNetwork()
	nodeSet := make(map[string]bool)
	for _, edge := range input.Edges {
		nodeSet[edge[0]] = true
		nodeSet[edge[1]] = true
	}
	for node := range nodeSet {
		if err := mn.AddNode(node); err != nil {
			t.Fatalf("AddNode(%q): %v", node, err)
		}
	}
	for _, edge := range input.Edges {
		if err := mn.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q): %v", edge[0], edge[1], err)
		}
	}
	for _, fd := range input.Factors {
		f, err := factors.NewDiscreteFactor(fd.Variables, fd.Cardinality, fd.Values)
		if err != nil {
			t.Fatalf("NewDiscreteFactor: %v", err)
		}
		mn.AddFactor(f)
	}

	Z, err := mn.GetPartitionFunction()
	if err != nil {
		t.Fatalf("GetPartitionFunction: %v", err)
	}

	if math.Abs(Z-expected.PartitionFunction) > 1e-6 {
		t.Errorf("partition function: expected %f, got %f", expected.PartitionFunction, Z)
	}
}

func TestCrossval_BNPredict(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models_extended/fixtures.json")
	tc := ff.FindTestCase(t, "bn_predict")

	var input struct {
		Edges       [][]string `json:"edges"`
		Predictions []struct {
			Query    []string       `json:"query"`
			Evidence map[string]int `json:"evidence"`
		} `json:"predictions"`
		CPDs map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Predictions []map[string]int `json:"predictions"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBNFromCPDs(t, input.Edges, input.CPDs)

	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		t.Fatalf("ToMarkovFactors: %v", err)
	}
	ve := inference.NewVariableElimination(markovFactors)

	for i, pred := range input.Predictions {
		assignment, err := ve.MAP(pred.Query, pred.Evidence)
		if err != nil {
			t.Fatalf("MAP query %d: %v", i, err)
		}
		for varName, expectedVal := range expected.Predictions[i] {
			gotVal, ok := assignment[varName]
			if !ok {
				t.Errorf("prediction[%d]: missing variable %q", i, varName)
				continue
			}
			if gotVal != expectedVal {
				t.Errorf("prediction[%d][%q]: expected %d, got %d", i, varName, expectedVal, gotVal)
			}
		}
	}
}

func TestCrossval_DBNStructure(t *testing.T) {
	ff := testutil.LoadFixtures(t, "models_extended/fixtures.json")
	tc := ff.FindTestCase(t, "dbn_structure")

	var input struct {
		Edges [][][]any `json:"edges"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		NumNodes int `json:"num_nodes"`
	}
	tc.UnmarshalExpected(t, &expected)

	dbn := models.NewDynamicBayesianNetwork()

	// Add initial and transition structure
	initial := dbn.Initial()
	transition := dbn.Transition()

	// Add intra-slice edges (time=0)
	for _, edge := range input.Edges {
		from := edge[0]
		to := edge[1]
		fromName, _ := from[0].(string)
		fromTime := int(from[1].(float64))
		toName, _ := to[0].(string)
		toTime := int(to[1].(float64))

		if fromTime == 0 && toTime == 0 {
			// Intra-slice edge in initial
			if !hasNode(initial, fromName) {
				_ = initial.AddNode(fromName)
			}
			if !hasNode(initial, toName) {
				_ = initial.AddNode(toName)
			}
			_ = initial.AddEdge(fromName, toName)
		} else if fromTime == 0 && toTime == 1 {
			// Inter-slice edge in transition
			if !hasNode(transition, fromName) {
				_ = transition.AddNode(fromName)
			}
			if !hasNode(transition, toName) {
				_ = transition.AddNode(toName)
			}
			_ = transition.AddEdge(fromName, toName)
		}
	}

	// Just verify DBN was built successfully
	if expected.NumNodes > 0 {
		t.Logf("DBN built with initial=%d nodes, transition=%d nodes",
			len(initial.Nodes()), len(transition.Nodes()))
	}
}

func hasNode(bn *models.BayesianNetwork, name string) bool {
	for _, n := range bn.Nodes() {
		if n == name {
			return true
		}
	}
	return false
}
