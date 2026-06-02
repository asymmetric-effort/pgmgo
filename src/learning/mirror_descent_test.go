//go:build unit

package learning

import (
	"math"
	"math/rand"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

func TestNewMirrorDescentEstimator(t *testing.T) {
	bn := models.NewBayesianNetwork()
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{})
	md := NewMirrorDescentEstimator(bn, data, 0.01, 100)
	if md == nil {
		t.Fatal("expected non-nil MirrorDescentEstimator")
	}
}

func TestMirrorDescent_SimpleNode_NoParents(t *testing.T) {
	// Single node X with values 0, 1, 2.
	// True distribution: P(X=0) = 0.5, P(X=1) = 0.3, P(X=2) = 0.2.
	bn := models.NewBayesianNetwork()
	if err := bn.AddNode("X"); err != nil {
		t.Fatal(err)
	}

	vals := make([]any, 1000)
	for i := 0; i < 500; i++ {
		vals[i] = 0
	}
	for i := 500; i < 800; i++ {
		vals[i] = 1
	}
	for i := 800; i < 1000; i++ {
		vals[i] = 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", vals),
	})

	md := NewMirrorDescentEstimator(bn, data, 0.1, 500)
	if err := md.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	cpd, err := md.GetParameters("X")
	if err != nil {
		t.Fatalf("GetParameters: %v", err)
	}

	flat := cpd.ToFactor().Values().Data()
	assertClose(t, "P(X=0)", flat[0], 0.5, 0.02)
	assertClose(t, "P(X=1)", flat[1], 0.3, 0.02)
	assertClose(t, "P(X=2)", flat[2], 0.2, 0.02)
}

func TestMirrorDescent_StudentNetwork(t *testing.T) {
	bn := buildStudentBN(t)
	rng := rand.New(rand.NewSource(42))
	data, trueCPDs := generateStudentData(rng, 50000)

	md := NewMirrorDescentEstimator(bn, data, 0.1, 1000)
	if err := md.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	// Verify estimated CPDs are close to true values.
	const tol = 0.03
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

func TestMirrorDescent_WithParents(t *testing.T) {
	// A -> B, A binary, B binary.
	// Data: A=0,B=0 x3; A=0,B=1 x1; A=1,B=0 x1; A=1,B=1 x3.
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("A")
	_ = bn.AddNode("B")
	_ = bn.AddEdge("A", "B")

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", []any{0, 0, 0, 0, 1, 1, 1, 1}),
		"B": tabgo.NewSeries("B", []any{0, 0, 0, 1, 0, 1, 1, 1}),
	})

	md := NewMirrorDescentEstimator(bn, data, 0.5, 1000)
	if err := md.Estimate(); err != nil {
		t.Fatalf("Estimate: %v", err)
	}

	cpd := bn.GetCPD("B")
	if cpd == nil {
		t.Fatal("no CPD for B")
	}

	flat := cpd.ToFactor().Values().Data()
	// P(B=0|A=0) = 3/4, P(B=0|A=1) = 1/4
	// P(B=1|A=0) = 1/4, P(B=1|A=1) = 3/4
	assertClose(t, "P(B=0|A=0)", flat[0], 0.75, 0.02)
	assertClose(t, "P(B=0|A=1)", flat[1], 0.25, 0.02)
	assertClose(t, "P(B=1|A=0)", flat[2], 0.25, 0.02)
	assertClose(t, "P(B=1|A=1)", flat[3], 0.75, 0.02)
}

func TestMirrorDescent_Estimate_NilBN(t *testing.T) {
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{0}),
	})
	md := NewMirrorDescentEstimator(nil, data, 0.1, 100)
	if err := md.Estimate(); err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMirrorDescent_Estimate_NilData(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	md := NewMirrorDescentEstimator(bn, nil, 0.1, 100)
	if err := md.Estimate(); err == nil {
		t.Error("expected error for nil data")
	}
}

func TestMirrorDescent_Estimate_MissingColumn(t *testing.T) {
	bn := buildStudentBN(t)
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"D": tabgo.NewSeries("D", []any{0}),
		"I": tabgo.NewSeries("I", []any{0}),
	})
	md := NewMirrorDescentEstimator(bn, data, 0.1, 100)
	if err := md.Estimate(); err == nil {
		t.Error("expected error for missing column")
	}
}

func TestMirrorDescent_GetParameters_NilBN(t *testing.T) {
	md := NewMirrorDescentEstimator(nil, nil, 0.1, 100)
	_, err := md.GetParameters("X")
	if err == nil {
		t.Error("expected error for nil BN")
	}
}

func TestMirrorDescent_GetParameters_NoCPD(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	md := NewMirrorDescentEstimator(bn, nil, 0.1, 100)
	_, err := md.GetParameters("X")
	if err == nil {
		t.Error("expected error for missing CPD")
	}
}

func TestMirrorDescent_EmptyData(t *testing.T) {
	bn := models.NewBayesianNetwork()
	_ = bn.AddNode("X")
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"X": tabgo.NewSeries("X", []any{}),
	})

	md := NewMirrorDescentEstimator(bn, data, 0.1, 100)
	err := md.Estimate()
	if err != nil {
		t.Fatalf("Estimate with empty data: %v", err)
	}
}

func TestMirrorDescent_ColumnsNormalize(t *testing.T) {
	// Verify that all CPD columns sum to 1 after estimation.
	bn := buildStudentBN(t)
	rng := rand.New(rand.NewSource(99))
	data, _ := generateStudentData(rng, 5000)

	md := NewMirrorDescentEstimator(bn, data, 0.1, 500)
	if err := md.Estimate(); err != nil {
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
