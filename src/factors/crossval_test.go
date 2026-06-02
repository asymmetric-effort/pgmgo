//go:build unit

package factors_test

import (
	"encoding/json"
	"math"
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

func sortedFloat64s(s []float64) []float64 {
	c := make([]float64, len(s))
	copy(c, s)
	sort.Float64s(c)
	return c
}

func sortedStrings(s []string) []string {
	c := make([]string, len(s))
	copy(c, s)
	sort.Strings(c)
	return c
}

func assertFloat64sEqual(t *testing.T, label string, expected, actual []float64, tol float64) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Fatalf("%s: length mismatch: expected %d, got %d", label, len(expected), len(actual))
	}
	for i := range expected {
		if math.Abs(expected[i]-actual[i]) > tol {
			t.Errorf("%s[%d]: expected %f, got %f", label, i, expected[i], actual[i])
		}
	}
}

func assertStringsEqual(t *testing.T, label string, expected, actual []string) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Fatalf("%s: length mismatch: expected %d, got %d", label, len(expected), len(actual))
	}
	for i := range expected {
		if expected[i] != actual[i] {
			t.Errorf("%s[%d]: expected %q, got %q", label, i, expected[i], actual[i])
		}
	}
}

func assertIntsEqual(t *testing.T, label string, expected, actual []int) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Fatalf("%s: length mismatch: expected %d, got %d", label, len(expected), len(actual))
	}
	for i := range expected {
		if expected[i] != actual[i] {
			t.Errorf("%s[%d]: expected %d, got %d", label, i, expected[i], actual[i])
		}
	}
}

func TestCrossval_DiscreteFactorCreation(t *testing.T) {
	ff := testutil.LoadFixtures(t, "factors/fixtures.json")
	tc := ff.FindTestCase(t, "discrete_factor_creation")

	var input struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	f, err := factors.NewDiscreteFactor(input.Variables, input.Cardinality, input.Values)
	if err != nil {
		t.Fatalf("NewDiscreteFactor failed: %v", err)
	}

	// Use sorted comparison since fixtures are canonicalized.
	assertStringsEqual(t, "variables", sortedStrings(expected.Variables), sortedStrings(f.Variables()))
	assertIntsEqual(t, "cardinality", expected.Cardinality, f.Cardinality())
	assertFloat64sEqual(t, "values", sortedFloat64s(expected.Values), sortedFloat64s(f.Values().Data()), 1e-9)
}

func TestCrossval_DiscreteFactorMarginalize(t *testing.T) {
	ff := testutil.LoadFixtures(t, "factors/fixtures.json")
	tc := ff.FindTestCase(t, "discrete_factor_marginalize")

	var input struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
		Marginalize []string  `json:"marginalize"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	f, err := factors.NewDiscreteFactor(input.Variables, input.Cardinality, input.Values)
	if err != nil {
		t.Fatalf("NewDiscreteFactor failed: %v", err)
	}

	result, err := f.Marginalize(input.Marginalize)
	if err != nil {
		t.Fatalf("Marginalize failed: %v", err)
	}

	assertStringsEqual(t, "variables", sortedStrings(expected.Variables), sortedStrings(result.Variables()))
	assertIntsEqual(t, "cardinality", expected.Cardinality, result.Cardinality())
	assertFloat64sEqual(t, "values", sortedFloat64s(expected.Values), sortedFloat64s(result.Values().Data()), 1e-9)
}

func TestCrossval_DiscreteFactorReduce(t *testing.T) {
	ff := testutil.LoadFixtures(t, "factors/fixtures.json")
	tc := ff.FindTestCase(t, "discrete_factor_reduce")

	var input struct {
		Variables   []string        `json:"variables"`
		Cardinality []int           `json:"cardinality"`
		Values      []float64       `json:"values"`
		Reduce      json.RawMessage `json:"reduce"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	// Parse reduce: array of [variable, value] pairs.
	var rawReduce []json.RawMessage
	if err := json.Unmarshal(input.Reduce, &rawReduce); err != nil {
		t.Fatalf("failed to parse reduce: %v", err)
	}

	evidence := make(map[string]int)
	for _, item := range rawReduce {
		var pair []json.RawMessage
		if err := json.Unmarshal(item, &pair); err != nil {
			t.Fatalf("failed to parse reduce pair: %v", err)
		}
		var varName string
		var varVal int
		if err := json.Unmarshal(pair[0], &varName); err != nil {
			t.Fatalf("failed to parse reduce variable name: %v", err)
		}
		if err := json.Unmarshal(pair[1], &varVal); err != nil {
			t.Fatalf("failed to parse reduce variable value: %v", err)
		}
		evidence[varName] = varVal
	}

	f, err := factors.NewDiscreteFactor(input.Variables, input.Cardinality, input.Values)
	if err != nil {
		t.Fatalf("NewDiscreteFactor failed: %v", err)
	}

	result, err := f.Reduce(evidence)
	if err != nil {
		t.Fatalf("Reduce failed: %v", err)
	}

	assertStringsEqual(t, "variables", sortedStrings(expected.Variables), sortedStrings(result.Variables()))
	assertIntsEqual(t, "cardinality", expected.Cardinality, result.Cardinality())
	assertFloat64sEqual(t, "values", sortedFloat64s(expected.Values), sortedFloat64s(result.Values().Data()), 1e-9)
}

func TestCrossval_DiscreteFactorProduct(t *testing.T) {
	ff := testutil.LoadFixtures(t, "factors/fixtures.json")
	tc := ff.FindTestCase(t, "discrete_factor_product")

	var input struct {
		Factor1 struct {
			Variables   []string  `json:"variables"`
			Cardinality []int     `json:"cardinality"`
			Values      []float64 `json:"values"`
		} `json:"factor1"`
		Factor2 struct {
			Variables   []string  `json:"variables"`
			Cardinality []int     `json:"cardinality"`
			Values      []float64 `json:"values"`
		} `json:"factor2"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	f1, err := factors.NewDiscreteFactor(input.Factor1.Variables, input.Factor1.Cardinality, input.Factor1.Values)
	if err != nil {
		t.Fatalf("NewDiscreteFactor (factor1) failed: %v", err)
	}

	f2, err := factors.NewDiscreteFactor(input.Factor2.Variables, input.Factor2.Cardinality, input.Factor2.Values)
	if err != nil {
		t.Fatalf("NewDiscreteFactor (factor2) failed: %v", err)
	}

	result, err := factors.FactorProduct(f1, f2)
	if err != nil {
		t.Fatalf("FactorProduct failed: %v", err)
	}

	assertStringsEqual(t, "variables", sortedStrings(expected.Variables), sortedStrings(result.Variables()))
	assertFloat64sEqual(t, "values", sortedFloat64s(expected.Values), sortedFloat64s(result.Values().Data()), 1e-9)
}

func TestCrossval_TabularCPDCreation(t *testing.T) {
	ff := testutil.LoadFixtures(t, "factors/fixtures.json")
	tc := ff.FindTestCase(t, "tabular_cpd_creation")

	var input struct {
		Variable     string      `json:"variable"`
		VariableCard int         `json:"variable_card"`
		Values       [][]float64 `json:"values"`
		Evidence     []string    `json:"evidence"`
		EvidenceCard []int       `json:"evidence_card"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variable    string      `json:"variable"`
		Variables   []string    `json:"variables"`
		Cardinality []int       `json:"cardinality"`
		Values      [][]float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)

	cpd, err := factors.NewTabularCPD(input.Variable, input.VariableCard, input.Values,
		input.Evidence, input.EvidenceCard)
	if err != nil {
		t.Fatalf("NewTabularCPD failed: %v", err)
	}

	if cpd.Variable() != expected.Variable {
		t.Errorf("variable: expected %q, got %q", expected.Variable, cpd.Variable())
	}

	if cpd.VariableCard() != input.VariableCard {
		t.Errorf("variable_card: expected %d, got %d", input.VariableCard, cpd.VariableCard())
	}

	// Verify the underlying factor has the expected variables and cardinality.
	factor := cpd.ToFactor()
	assertStringsEqual(t, "factor variables", sortedStrings(expected.Variables), sortedStrings(factor.Variables()))
	assertIntsEqual(t, "factor cardinality", expected.Cardinality, factor.Cardinality())

	// Verify the CPD values by checking the flat factor data matches the
	// expected 2D values laid out in row-major order.
	var expectedFlat []float64
	for _, row := range expected.Values {
		expectedFlat = append(expectedFlat, row...)
	}
	assertFloat64sEqual(t, "values", sortedFloat64s(expectedFlat), sortedFloat64s(factor.Values().Data()), 1e-9)

	// Validate that the CPD columns sum to 1.
	if err := cpd.Validate(); err != nil {
		t.Errorf("CPD validation failed: %v", err)
	}
}

func TestCrossval_JointProbabilityDistribution(t *testing.T) {
	ff := testutil.LoadFixtures(t, "factors/fixtures.json")
	tc := ff.FindTestCase(t, "joint_probability_distribution")

	var input struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
		MarginalX   struct {
			Variables []string  `json:"variables"`
			Values    []float64 `json:"values"`
		} `json:"marginal_x"`
		MarginalY struct {
			Variables []string  `json:"variables"`
			Values    []float64 `json:"values"`
		} `json:"marginal_y"`
	}
	tc.UnmarshalExpected(t, &expected)

	// Create the JPD.
	jpd, err := factors.NewJointProbabilityDistribution(input.Variables, input.Cardinality, input.Values)
	if err != nil {
		t.Fatalf("NewJointProbabilityDistribution failed: %v", err)
	}

	// Verify basic properties.
	assertStringsEqual(t, "variables", sortedStrings(expected.Variables), sortedStrings(jpd.Variables()))
	assertIntsEqual(t, "cardinality", expected.Cardinality, jpd.Cardinality())
	assertFloat64sEqual(t, "values", expected.Values, jpd.Values().Data(), 1e-9)

	// Compute and verify marginal over X.
	margX, err := jpd.MarginalDistribution(expected.MarginalX.Variables)
	if err != nil {
		t.Fatalf("MarginalDistribution(%v) failed: %v", expected.MarginalX.Variables, err)
	}
	assertStringsEqual(t, "marginal_x variables",
		sortedStrings(expected.MarginalX.Variables), sortedStrings(margX.Variables()))
	assertFloat64sEqual(t, "marginal_x values",
		expected.MarginalX.Values, margX.Values().Data(), 1e-6)

	// Compute and verify marginal over Y.
	margY, err := jpd.MarginalDistribution(expected.MarginalY.Variables)
	if err != nil {
		t.Fatalf("MarginalDistribution(%v) failed: %v", expected.MarginalY.Variables, err)
	}
	assertStringsEqual(t, "marginal_y variables",
		sortedStrings(expected.MarginalY.Variables), sortedStrings(margY.Variables()))
	assertFloat64sEqual(t, "marginal_y values",
		expected.MarginalY.Values, margY.Values().Data(), 1e-6)
}
