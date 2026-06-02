//go:build unit

package learning

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

func TestNewMarginalEstimator(t *testing.T) {
	bn := models.NewBayesianNetwork()
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{})
	me := NewMarginalEstimator(bn, data)
	if me == nil {
		t.Fatal("expected non-nil MarginalEstimator")
	}
}

func TestMarginalEstimator_Estimate_NoParents(t *testing.T) {
	// Single node X, values 0,1,2 with known counts.
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")

	vals := make([]any, 100)
	for i := 0; i < 50; i++ {
		vals[i] = 0
	}
	for i := 50; i < 80; i++ {
		vals[i] = 1
	}
	for i := 80; i < 100; i++ {
		vals[i] = 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", vals),
	})

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	cpd := bn.GetCPD("X")
	if cpd == nil {
		t.Fatal("no CPD for X")
	}

	flat := cpd.ToFactor().Values().Data()
	assertClose(t, "P(X=0)", flat[0], 0.5, 1e-9)
	assertClose(t, "P(X=1)", flat[1], 0.3, 1e-9)
	assertClose(t, "P(X=2)", flat[2], 0.2, 1e-9)
}

func TestMarginalEstimator_Estimate_StudentNetwork(t *testing.T) {
	bn := buildStudentBN(t)
	rng := rand.New(rand.NewSource(42))
	data, trueCPDs := generateStudentData(rng, 100000)

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	const tol = 0.02
	for _, node := range []string{"D", "I", "G", "L", "S"} {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			t.Fatalf("no CPD for node %q", node)
		}
		trueVals := trueCPDs[node]
		factor := cpd.ToFactor()
		flatVals := factor.Values().Data()

		nodeCard := cpd.VariableCard()
		evCard := cpd.EvidenceCard()
		numPC := 1
		for _, ec := range evCard {
			numPC *= ec
		}

		for cs := 0; cs < nodeCard; cs++ {
			for pc := 0; pc < numPC; pc++ {
				idx := cs*numPC + pc
				got := flatVals[idx]
				want := trueVals[cs][pc]
				if math.Abs(got-want) > tol {
					t.Errorf("node %q: CPD[%d][%d] = %.4f, want %.4f (tol=%.2f)",
						node, cs, pc, got, want, tol)
				}
			}
		}
	}
}

func TestMarginalEstimator_MarginalLikelihood(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")

	// Deterministic: all X=0.
	vals := make([]any, 100)
	for i := range vals {
		vals[i] = 0
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", vals),
	})

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	ll, err := me.MarginalLikelihood()
	if err != nil {
		t.Fatalf("MarginalLikelihood: %v", err)
	}

	// All data is X=0, P(X=0) = 1.0, so log-likelihood = 100 * log(1) = 0.
	assertClose(t, "log-likelihood", ll, 0.0, 1e-9)
}

func TestMarginalEstimator_MarginalLikelihood_Mixed(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")

	// 50 zeros, 50 ones.
	vals := make([]any, 100)
	for i := 0; i < 50; i++ {
		vals[i] = 0
	}
	for i := 50; i < 100; i++ {
		vals[i] = 1
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", vals),
	})

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	ll, err := me.MarginalLikelihood()
	if err != nil {
		t.Fatalf("MarginalLikelihood: %v", err)
	}

	// P(X=0) = 0.5, P(X=1) = 0.5.
	// LL = 100 * log(0.5) = -69.315...
	expected := 100.0 * math.Log(0.5)
	assertClose(t, "log-likelihood", ll, expected, 1e-9)
}

func TestMarginalEstimator_MarginalLikelihood_WithParents(t *testing.T) {
	// A -> B, deterministic: A=0 => B=0, A=1 => B=1.
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("A")
	_ = bn.AddNode("B")
	_ = bn.AddEdge("A", "B")

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 0, 1, 1}),
		"B": tabgo.NewSeries("B", []any{0, 0, 1, 1}),
	})

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	ll, err := me.MarginalLikelihood()
	if err != nil {
		t.Fatalf("MarginalLikelihood: %v", err)
	}

	// All conditional probabilities are 1.0, and P(A=0)=P(A=1)=0.5.
	// LL = 4 * [log(0.5) + log(1.0)] = 4 * log(0.5)
	expected := 4.0 * math.Log(0.5)
	assertClose(t, "log-likelihood", ll, expected, 1e-9)
}

func TestMarginalEstimator_MarginalLikelihood_NoCPD(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0}),
	})

	me := NewMarginalEstimator(bn, data)
	// Do not call Estimate.
	_, err := me.MarginalLikelihood()
	if err == nil {
		t.Error("expected error when CPDs not estimated")
	}
}

func TestMarginalEstimator_MarginalLikelihood_EmptyData(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{}),
	})

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	ll, err := me.MarginalLikelihood()
	if err != nil {
		t.Fatalf("MarginalLikelihood: %v", err)
	}
	// No data => log-likelihood = 0.
	assertClose(t, "log-likelihood", ll, 0.0, 1e-9)
}

func TestMarginalEstimator_Estimate_NilBN(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0}),
	})
	me := NewMarginalEstimator(nil, data)
	if err := me.Estimate(); err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMarginalEstimator_Estimate_NilData(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	me := NewMarginalEstimator(bn, nil)
	if err := me.Estimate(); err == nil {
		t.Error("expected error for nil data")
	}
}

func TestMarginalEstimator_Estimate_MissingColumn(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	_ = bn.AddNode("Y")
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0}),
	})
	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err == nil {
		t.Error("expected error for missing column")
	}
}

func TestMarginalEstimator_MarginalLikelihood_NilBN(t *testing.T) {
	me := NewMarginalEstimator(nil, nil)
	_, err := me.MarginalLikelihood()
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMarginalEstimator_MarginalLikelihood_NilData(t *testing.T) {
	bn := models.NewBayesianNetwork()
	me := NewMarginalEstimator(bn, nil)
	_, err := me.MarginalLikelihood()
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestMarginalEstimator_Validate(t *testing.T) {
	// Verify all CPD columns sum to 1.
	bn := buildStudentBN(t)
	rng := rand.New(rand.NewSource(99))
	data, _ := generateStudentData(rng, 5000)

	me := NewMarginalEstimator(bn, data)
	if err := me.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	for _, node := range bn.Nodes() {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			t.Fatalf("no CPD for %q", node)
		}
		if err := cpd.Validate(); err != nil {
			t.Errorf("CPD for %q failed validation: %v", node, err)
		}
	}
}
