//go:build unit

package learning_test

import (
	"math"
	"sort"
	"strconv"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/ci_tests"
	"github.com/asymmetric-effort/pgmgo/src/learning"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/src/structure_score"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// cpdFixtureExt holds expected CPD data from extended fixtures.
type cpdFixtureExt struct {
	Variable     string      `json:"variable"`
	VariableCard int         `json:"variable_card"`
	Values       [][]float64 `json:"values"`
	Evidence     []string    `json:"evidence"`
	EvidenceCard []int       `json:"evidence_card"`
}

// buildStudentBNExt creates a student BN from fixture edges.
func buildStudentBNExt(t *testing.T, edges [][]string) *models.BayesianNetwork {
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
	return bn
}

// buildDFIntExt creates a DataFrame with integer-typed values.
func buildDFIntExt(columns []string, rows [][]float64) *tabgo.DataFrame {
	anyRows := make([][]any, len(rows))
	for i, row := range rows {
		anyRow := make([]any, len(row))
		for j, v := range row {
			anyRow[j] = int(v)
		}
		anyRows[i] = anyRow
	}
	return tabgo.NewDataFrameFromRows(columns, anyRows)
}

// buildDFStrExt creates a DataFrame with string-typed values.
func buildDFStrExt(columns []string, rows [][]float64) *tabgo.DataFrame {
	anyRows := make([][]any, len(rows))
	for i, row := range rows {
		anyRow := make([]any, len(row))
		for j, v := range row {
			anyRow[j] = strconv.Itoa(int(v))
		}
		anyRows[i] = anyRow
	}
	return tabgo.NewDataFrameFromRows(columns, anyRows)
}

// setStatesExt sets string state names ("0", "1", ...) based on cardinalities.
func setStatesExt(t *testing.T, bn *models.BayesianNetwork, nodeCards map[string]int) {
	t.Helper()
	for node, card := range nodeCards {
		states := make([]string, card)
		for i := 0; i < card; i++ {
			states[i] = strconv.Itoa(i)
		}
		if err := bn.SetStates(node, states); err != nil {
			t.Fatalf("SetStates(%q): %v", node, err)
		}
	}
}

// assertCPDCloseExt compares CPD values with tolerance.
func assertCPDCloseExt(t *testing.T, label string, expected [][]float64, got []float64, card int, tol float64) {
	t.Helper()
	numPC := 1
	if len(expected) > 0 {
		numPC = len(expected[0])
	}
	flat := make([]float64, 0, card*numPC)
	for _, row := range expected {
		flat = append(flat, row...)
	}
	if len(got) != len(flat) {
		t.Fatalf("%s: value count: expected %d, got %d", label, len(flat), len(got))
	}
	for i := range flat {
		if math.Abs(got[i]-flat[i]) > tol {
			t.Errorf("%s: value[%d]: expected %.6f, got %.6f", label, i, flat[i], got[i])
		}
	}
}

func TestCrossval_MLEAllNodes(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "mle_all_nodes")

	var input struct {
		Edges       [][]string     `json:"edges"`
		NodeCards   map[string]int `json:"node_cards"`
		DataColumns []string       `json:"data_columns"`
		DataRows    [][]float64    `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		CPDs map[string]cpdFixtureExt `json:"cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBNExt(t, input.Edges)
	df := buildDFIntExt(input.DataColumns, input.DataRows)
	mle := learning.NewMLE(bn, df)

	for node, cpdExp := range expected.CPDs {
		cpd, err := mle.GetParameters(node)
		if err != nil {
			t.Fatalf("MLE.GetParameters(%q): %v", node, err)
		}
		gotVals := cpd.ToFactor().Values().Data()
		assertCPDCloseExt(t, "MLE "+node, cpdExp.Values, gotVals, cpdExp.VariableCard, 0.02)
	}
}

func TestCrossval_BayesianBDeuESS5(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "bayesian_bdeu_ess5")

	var input struct {
		Edges                [][]string     `json:"edges"`
		NodeCards            map[string]int `json:"node_cards"`
		EquivalentSampleSize float64        `json:"equivalent_sample_size"`
		DataColumns          []string       `json:"data_columns"`
		DataRows             [][]float64    `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		CPDs map[string]cpdFixtureExt `json:"cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBNExt(t, input.Edges)
	setStatesExt(t, bn, input.NodeCards)
	df := buildDFStrExt(input.DataColumns, input.DataRows)
	be := learning.NewBayesianEstimator(bn, df, learning.BDeu, input.EquivalentSampleSize)
	if err := be.Estimate(); err != nil {
		t.Fatalf("BayesianEstimator.Estimate: %v", err)
	}

	for node, cpdExp := range expected.CPDs {
		cpd, err := be.GetParameters(node)
		if err != nil {
			t.Fatalf("BayesianEstimator.GetParameters(%q): %v", node, err)
		}
		gotVals := cpd.ToFactor().Values().Data()
		assertCPDCloseExt(t, "BDeu_ESS5 "+node, cpdExp.Values, gotVals, cpdExp.VariableCard, 0.02)
	}
}

func TestCrossval_BayesianBDeuESS50(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "bayesian_bdeu_ess50")

	var input struct {
		Edges                [][]string     `json:"edges"`
		NodeCards            map[string]int `json:"node_cards"`
		EquivalentSampleSize float64        `json:"equivalent_sample_size"`
		DataColumns          []string       `json:"data_columns"`
		DataRows             [][]float64    `json:"data_rows"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		CPDs map[string]cpdFixtureExt `json:"cpds"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBNExt(t, input.Edges)
	setStatesExt(t, bn, input.NodeCards)
	df := buildDFStrExt(input.DataColumns, input.DataRows)
	be := learning.NewBayesianEstimator(bn, df, learning.BDeu, input.EquivalentSampleSize)
	if err := be.Estimate(); err != nil {
		t.Fatalf("BayesianEstimator.Estimate: %v", err)
	}

	for node, cpdExp := range expected.CPDs {
		cpd, err := be.GetParameters(node)
		if err != nil {
			t.Fatalf("BayesianEstimator.GetParameters(%q): %v", node, err)
		}
		gotVals := cpd.ToFactor().Values().Data()
		assertCPDCloseExt(t, "BDeu_ESS50 "+node, cpdExp.Values, gotVals, cpdExp.VariableCard, 0.03)
	}
}

// Structure learning crossval helpers
type slInput struct {
	Variables    []string    `json:"variables"`
	DataColumns  []string    `json:"data_columns"`
	DataRows     [][]float64 `json:"data_rows"`
	Significance float64     `json:"significance_level"`
}

func runHCCrossval(t *testing.T, fixtureName string, scoreFn learning.ScoreFunc) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input slInput
	tc.UnmarshalInput(t, &input)

	var expected struct {
		LearnedEdges  [][]string `json:"learned_edges"`
		SkeletonEdges [][]string `json:"skeleton_edges"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDFIntExt(input.DataColumns, input.DataRows)
	hc := learning.NewHillClimbSearch(df, scoreFn)
	model, err := hc.Estimate()
	if err != nil {
		t.Fatalf("HillClimb Estimate: %v", err)
	}

	gotSkeleton := make([][]string, 0)
	for _, edge := range model.Edges() {
		e := []string{edge[0], edge[1]}
		sort.Strings(e)
		gotSkeleton = append(gotSkeleton, e)
	}
	sort.Slice(gotSkeleton, func(i, j int) bool {
		if gotSkeleton[i][0] == gotSkeleton[j][0] {
			return gotSkeleton[i][1] < gotSkeleton[j][1]
		}
		return gotSkeleton[i][0] < gotSkeleton[j][0]
	})

	if len(gotSkeleton) != len(expected.SkeletonEdges) {
		t.Errorf("skeleton edges: expected %d, got %d", len(expected.SkeletonEdges), len(gotSkeleton))
		return
	}
	for i := range expected.SkeletonEdges {
		exp := make([]string, len(expected.SkeletonEdges[i]))
		copy(exp, expected.SkeletonEdges[i])
		sort.Strings(exp)
		if gotSkeleton[i][0] != exp[0] || gotSkeleton[i][1] != exp[1] {
			t.Errorf("skeleton edge[%d]: expected %v, got %v", i, exp, gotSkeleton[i])
		}
	}
}

func TestCrossval_HC_BIC(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "hc_bic")
	var input slInput
	tc.UnmarshalInput(t, &input)
	df := buildDFIntExt(input.DataColumns, input.DataRows)
	bic := structure_score.NewBIC()
	scoreFn := func(v string, p []string, d *tabgo.DataFrame) float64 {
		return bic.LocalScore(v, p, d)
	}
	runHCCrossval(t, "hc_bic", scoreFn)
	_ = df
}

func TestCrossval_HC_AIC(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "hc_aic")
	var input slInput
	tc.UnmarshalInput(t, &input)
	df := buildDFIntExt(input.DataColumns, input.DataRows)
	aic := structure_score.NewAIC()
	scoreFn := func(v string, p []string, d *tabgo.DataFrame) float64 {
		return aic.LocalScore(v, p, d)
	}
	runHCCrossval(t, "hc_aic", scoreFn)
	_ = df
}

func TestCrossval_HC_BDeu(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "hc_bdeu")
	var input slInput
	tc.UnmarshalInput(t, &input)
	df := buildDFIntExt(input.DataColumns, input.DataRows)
	bdeu := structure_score.NewBDeu(10.0)
	scoreFn := func(v string, p []string, d *tabgo.DataFrame) float64 {
		return bdeu.LocalScore(v, p, d)
	}
	runHCCrossval(t, "hc_bdeu", scoreFn)
	_ = df
}

func runPCCrossval(t *testing.T, fixtureName string) {
	t.Helper()
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, fixtureName)

	var input slInput
	tc.UnmarshalInput(t, &input)

	var expected struct {
		SkeletonEdges [][]string `json:"skeleton_edges"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDFStrExt(input.DataColumns, input.DataRows)
	chiSquareFn := learning.CITestFunc(ci_tests.ChiSquare)
	pc := learning.NewPC(df, chiSquareFn, input.Significance)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("PC Estimate: %v", err)
	}

	// Extract skeleton from PDAG
	gotSkeleton := make(map[[2]string]bool)
	for _, e := range pdag.DirectedEdges() {
		a, b := e[0], e[1]
		if a > b {
			a, b = b, a
		}
		gotSkeleton[[2]string{a, b}] = true
	}
	for _, e := range pdag.UndirectedEdges() {
		a, b := e[0], e[1]
		if a > b {
			a, b = b, a
		}
		gotSkeleton[[2]string{a, b}] = true
	}

	gotList := make([][]string, 0, len(gotSkeleton))
	for e := range gotSkeleton {
		gotList = append(gotList, []string{e[0], e[1]})
	}
	sort.Slice(gotList, func(i, j int) bool {
		if gotList[i][0] == gotList[j][0] {
			return gotList[i][1] < gotList[j][1]
		}
		return gotList[i][0] < gotList[j][0]
	})

	if len(gotList) != len(expected.SkeletonEdges) {
		t.Errorf("skeleton edges: expected %d, got %d", len(expected.SkeletonEdges), len(gotList))
		return
	}
	for i := range expected.SkeletonEdges {
		if gotList[i][0] != expected.SkeletonEdges[i][0] || gotList[i][1] != expected.SkeletonEdges[i][1] {
			t.Errorf("skeleton edge[%d]: expected %v, got %v", i, expected.SkeletonEdges[i], gotList[i])
		}
	}
}

func TestCrossval_PC_Sig001(t *testing.T) { runPCCrossval(t, "pc_sig001") }
func TestCrossval_PC_Sig005(t *testing.T) { runPCCrossval(t, "pc_sig005") }
func TestCrossval_PC_Sig010(t *testing.T) { runPCCrossval(t, "pc_sig010") }

func TestCrossval_TreeSearchChowLiu(t *testing.T) {
	ff := testutil.LoadFixtures(t, "learning_extended/fixtures.json")
	tc := ff.FindTestCase(t, "tree_search_chow_liu")

	var input slInput
	tc.UnmarshalInput(t, &input)

	var expected struct {
		TreeEdges [][]string `json:"tree_edges"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDFStrExt(input.DataColumns, input.DataRows)
	ts := learning.NewTreeSearch(df)
	model, err := ts.Estimate()
	if err != nil {
		t.Fatalf("TreeSearch Estimate: %v", err)
	}

	gotEdges := make([][]string, 0)
	for _, edge := range model.Edges() {
		e := []string{edge[0], edge[1]}
		sort.Strings(e)
		gotEdges = append(gotEdges, e)
	}
	sort.Slice(gotEdges, func(i, j int) bool {
		if gotEdges[i][0] == gotEdges[j][0] {
			return gotEdges[i][1] < gotEdges[j][1]
		}
		return gotEdges[i][0] < gotEdges[j][0]
	})

	if len(gotEdges) != len(expected.TreeEdges) {
		t.Errorf("tree edges: expected %d, got %d", len(expected.TreeEdges), len(gotEdges))
		return
	}
	for i := range expected.TreeEdges {
		exp := make([]string, len(expected.TreeEdges[i]))
		copy(exp, expected.TreeEdges[i])
		sort.Strings(exp)
		if gotEdges[i][0] != exp[0] || gotEdges[i][1] != exp[1] {
			t.Errorf("tree edge[%d]: expected %v, got %v", i, exp, gotEdges[i])
		}
	}
}
