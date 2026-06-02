//go:build unit

package sampling_test

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/src/sampling"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// buildStudentBN constructs a Bayesian network from the fixture input data.
func buildStudentBN(t *testing.T, edges [][]string, cpds map[string]struct {
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
			t.Fatalf("AddNode(%q) failed: %v", node, err)
		}
	}

	for _, edge := range edges {
		if err := bn.AddEdge(edge[0], edge[1]); err != nil {
			t.Fatalf("AddEdge(%q, %q) failed: %v", edge[0], edge[1], err)
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
			t.Fatalf("NewTabularCPD(%q) failed: %v", varName, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			t.Fatalf("AddCPD(%q) failed: %v", varName, err)
		}
	}

	return bn
}

// computeEmpiricalMarginals computes empirical marginal distributions from
// sampling results. For each variable, it counts occurrences of each state
// and divides by total count (optionally weighted).
func computeEmpiricalMarginals(
	t *testing.T,
	nodes []string,
	nSamples int,
	getSample func(node string, i int) int,
	getWeight func(i int) float64,
	cardinalities map[string]int,
) map[string][]float64 {
	t.Helper()

	marginals := make(map[string][]float64)
	for _, node := range nodes {
		card := cardinalities[node]
		counts := make([]float64, card)
		totalWeight := 0.0

		for i := 0; i < nSamples; i++ {
			w := getWeight(i)
			val := getSample(node, i)
			counts[val] += w
			totalWeight += w
		}

		dist := make([]float64, card)
		for j := range counts {
			dist[j] = counts[j] / totalWeight
		}
		marginals[node] = dist
	}
	return marginals
}

func TestCrossval_ForwardSampling(t *testing.T) {
	ff := testutil.LoadFixtures(t, "sampling/fixtures.json")
	tc := ff.FindTestCase(t, "forward_sampling")

	var input struct {
		Edges    [][]string `json:"edges"`
		NSamples int        `json:"n_samples"`
		Seed     int64      `json:"seed"`
		CPDs     map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Marginals map[string][]float64 `json:"marginals"`
		Tolerance float64              `json:"tolerance"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBN(t, input.Edges, input.CPDs)

	bms, err := sampling.NewBayesianModelSampling(bn, input.Seed)
	if err != nil {
		t.Fatalf("NewBayesianModelSampling failed: %v", err)
	}

	df, err := bms.ForwardSample(input.NSamples)
	if err != nil {
		t.Fatalf("ForwardSample failed: %v", err)
	}

	// Build cardinality map from CPDs.
	cardinalities := make(map[string]int)
	for varName, cpdData := range input.CPDs {
		cardinalities[varName] = cpdData.VariableCard
	}

	// Compute empirical marginals from the samples.
	nodes := bn.Nodes()
	marginals := computeEmpiricalMarginals(t, nodes, df.Len(),
		func(node string, i int) int {
			col := df.Column(node)
			return col.Int()[i]
		},
		func(i int) float64 { return 1.0 },
		cardinalities,
	)

	// Compare against fixture expected marginals.
	for varName, expectedDist := range expected.Marginals {
		gotDist, ok := marginals[varName]
		if !ok {
			t.Errorf("missing marginal for variable %q", varName)
			continue
		}
		if len(gotDist) != len(expectedDist) {
			t.Errorf("marginal %q: length mismatch: expected %d, got %d", varName, len(expectedDist), len(gotDist))
			continue
		}
		for i := range expectedDist {
			if math.Abs(gotDist[i]-expectedDist[i]) > expected.Tolerance {
				t.Errorf("marginal %q[%d]: expected %f, got %f (diff=%e, tol=%f)",
					varName, i, expectedDist[i], gotDist[i],
					math.Abs(gotDist[i]-expectedDist[i]), expected.Tolerance)
			}
		}
	}
}

func TestCrossval_LikelihoodWeightedSampling(t *testing.T) {
	ff := testutil.LoadFixtures(t, "sampling/fixtures.json")
	tc := ff.FindTestCase(t, "likelihood_weighted_sampling")

	var input struct {
		Edges    [][]string     `json:"edges"`
		NSamples int            `json:"n_samples"`
		Seed     int64          `json:"seed"`
		Evidence map[string]int `json:"evidence"`
		CPDs     map[string]struct {
			VariableCard int         `json:"variable_card"`
			Values       [][]float64 `json:"values"`
			Evidence     []string    `json:"evidence"`
			EvidenceCard []int       `json:"evidence_card"`
		} `json:"cpds"`
	}
	tc.UnmarshalInput(t, &input)

	var expected struct {
		Marginals map[string][]float64 `json:"marginals"`
		Tolerance float64              `json:"tolerance"`
	}
	tc.UnmarshalExpected(t, &expected)

	bn := buildStudentBN(t, input.Edges, input.CPDs)

	bms, err := sampling.NewBayesianModelSampling(bn, input.Seed)
	if err != nil {
		t.Fatalf("NewBayesianModelSampling failed: %v", err)
	}

	df, weights, err := bms.LikelihoodWeightedSample(input.NSamples, input.Evidence)
	if err != nil {
		t.Fatalf("LikelihoodWeightedSample failed: %v", err)
	}

	// Build cardinality map from CPDs.
	cardinalities := make(map[string]int)
	for varName, cpdData := range input.CPDs {
		cardinalities[varName] = cpdData.VariableCard
	}

	// Compute weighted empirical marginals.
	nodes := bn.Nodes()
	marginals := computeEmpiricalMarginals(t, nodes, df.Len(),
		func(node string, i int) int {
			col := df.Column(node)
			return col.Int()[i]
		},
		func(i int) float64 { return weights[i] },
		cardinalities,
	)

	// Compare against fixture expected marginals.
	for varName, expectedDist := range expected.Marginals {
		gotDist, ok := marginals[varName]
		if !ok {
			t.Errorf("missing marginal for variable %q", varName)
			continue
		}
		if len(gotDist) != len(expectedDist) {
			t.Errorf("marginal %q: length mismatch: expected %d, got %d", varName, len(expectedDist), len(gotDist))
			continue
		}
		for i := range expectedDist {
			if math.Abs(gotDist[i]-expectedDist[i]) > expected.Tolerance {
				t.Errorf("marginal %q[%d]: expected %f, got %f (diff=%e, tol=%f)",
					varName, i, expectedDist[i], gotDist[i],
					math.Abs(gotDist[i]-expectedDist[i]), expected.Tolerance)
			}
		}
	}
}
