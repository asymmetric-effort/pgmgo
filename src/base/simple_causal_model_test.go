//go:build unit

package base

import (
	"math"
	"testing"
)

func TestNewSimpleCausalModel(t *testing.T) {
	dag := NewDAG()
	m := NewSimpleCausalModel(dag)
	if m == nil {
		t.Fatal("NewSimpleCausalModel returned nil")
	}
	if m.DAG() != dag {
		t.Error("DAG() should return the underlying DAG")
	}
}

func TestSimpleCausalModelLinearChain(t *testing.T) {
	// X → Y → Z with linear equations: Y = 2*X, Z = Y + 1
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y", "Z")
	_ = dag.AddEdge("X", "Y")
	_ = dag.AddEdge("Y", "Z")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 {
		return 2 * pv["X"]
	})
	m.SetEquation("Z", func(pv map[string]float64) float64 {
		return pv["Y"] + 1
	})

	vals, err := m.Sample(map[string]float64{"X": 3})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}

	if vals["X"] != 3 {
		t.Errorf("X = %f, want 3", vals["X"])
	}
	if vals["Y"] != 6 {
		t.Errorf("Y = %f, want 6 (2*3)", vals["Y"])
	}
	if vals["Z"] != 7 {
		t.Errorf("Z = %f, want 7 (6+1)", vals["Z"])
	}
}

func TestSimpleCausalModelMultipleParents(t *testing.T) {
	// A → C, B → C with C = A + B
	dag := NewDAG()
	_ = dag.AddNodes("A", "B", "C")
	_ = dag.AddEdge("A", "C")
	_ = dag.AddEdge("B", "C")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("C", func(pv map[string]float64) float64 {
		return pv["A"] + pv["B"]
	})

	vals, err := m.Sample(map[string]float64{"A": 2, "B": 5})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if vals["C"] != 7 {
		t.Errorf("C = %f, want 7 (2+5)", vals["C"])
	}
}

func TestSimpleCausalModelNoEquation(t *testing.T) {
	// Root nodes without equations use their exogenous values.
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y")
	_ = dag.AddEdge("X", "Y")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 {
		return pv["X"] * 3
	})

	vals, err := m.Sample(map[string]float64{"X": 4})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if vals["X"] != 4 {
		t.Errorf("X = %f, want 4", vals["X"])
	}
	if vals["Y"] != 12 {
		t.Errorf("Y = %f, want 12", vals["Y"])
	}
}

func TestSimpleCausalModelDefaultZero(t *testing.T) {
	// A variable with no equation and no exogenous value defaults to 0.
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y")
	_ = dag.AddEdge("X", "Y")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 {
		return pv["X"] + 10
	})

	vals, err := m.Sample(map[string]float64{})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if vals["X"] != 0 {
		t.Errorf("X = %f, want 0 (default)", vals["X"])
	}
	if vals["Y"] != 10 {
		t.Errorf("Y = %f, want 10 (0+10)", vals["Y"])
	}
}

func TestSimpleCausalModelIntervene(t *testing.T) {
	// X → Y → Z, Y = 2*X, Z = Y + 1
	// Intervene: do(Y = 10)
	// Expected: Z = 10 + 1 = 11, X is unaffected.
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y", "Z")
	_ = dag.AddEdge("X", "Y")
	_ = dag.AddEdge("Y", "Z")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 {
		return 2 * pv["X"]
	})
	m.SetEquation("Z", func(pv map[string]float64) float64 {
		return pv["Y"] + 1
	})

	mutilated := m.Intervene("Y", 10)

	vals, err := mutilated.Sample(map[string]float64{"X": 3})
	if err != nil {
		t.Fatalf("Sample on mutilated model failed: %v", err)
	}

	if vals["X"] != 3 {
		t.Errorf("X = %f, want 3 (unaffected by intervention)", vals["X"])
	}
	if vals["Y"] != 10 {
		t.Errorf("Y = %f, want 10 (intervened)", vals["Y"])
	}
	if vals["Z"] != 11 {
		t.Errorf("Z = %f, want 11 (10+1)", vals["Z"])
	}

	// The intervened variable should have no parents in the mutilated model.
	if len(mutilated.DAG().Parents("Y")) != 0 {
		t.Error("Y should have no parents in mutilated model")
	}
}

func TestSimpleCausalModelInterveneDoesNotAffectOriginal(t *testing.T) {
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y")
	_ = dag.AddEdge("X", "Y")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 {
		return pv["X"] * 2
	})

	_ = m.Intervene("Y", 99)

	// Original should still have X→Y.
	if !m.DAG().HasEdge("X", "Y") {
		t.Error("intervention should not modify original model's DAG")
	}

	vals, err := m.Sample(map[string]float64{"X": 5})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if vals["Y"] != 10 {
		t.Errorf("Y = %f, want 10 (original equation)", vals["Y"])
	}
}

func TestSimpleCausalModelInterveneNonExistentNode(t *testing.T) {
	dag := NewDAG()
	_ = dag.AddNode("X")
	m := NewSimpleCausalModel(dag)

	// Intervening on a non-existent node should return a copy.
	mutilated := m.Intervene("Z", 42)
	if mutilated == nil {
		t.Fatal("Intervene on non-existent node should return a copy, not nil")
	}
	if !mutilated.DAG().HasNode("X") {
		t.Error("copy should preserve existing nodes")
	}
}

func TestSimpleCausalModelMultipleInterventions(t *testing.T) {
	// A → B, A → C, B → D, C → D
	// D = B + C, B = A*2, C = A*3
	dag := NewDAG()
	_ = dag.AddNodes("A", "B", "C", "D")
	_ = dag.AddEdge("A", "B")
	_ = dag.AddEdge("A", "C")
	_ = dag.AddEdge("B", "D")
	_ = dag.AddEdge("C", "D")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("B", func(pv map[string]float64) float64 { return pv["A"] * 2 })
	m.SetEquation("C", func(pv map[string]float64) float64 { return pv["A"] * 3 })
	m.SetEquation("D", func(pv map[string]float64) float64 { return pv["B"] + pv["C"] })

	// do(B=5), then do(C=7) on the result.
	m1 := m.Intervene("B", 5)
	m2 := m1.Intervene("C", 7)

	vals, err := m2.Sample(map[string]float64{"A": 100})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if vals["B"] != 5 {
		t.Errorf("B = %f, want 5", vals["B"])
	}
	if vals["C"] != 7 {
		t.Errorf("C = %f, want 7", vals["C"])
	}
	if vals["D"] != 12 {
		t.Errorf("D = %f, want 12 (5+7)", vals["D"])
	}
}

func TestSimpleCausalModelCopy(t *testing.T) {
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y")
	_ = dag.AddEdge("X", "Y")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 { return pv["X"] + 1 })

	c := m.Copy()

	// Copy should produce the same results.
	valsOrig, _ := m.Sample(map[string]float64{"X": 5})
	valsCopy, _ := c.Sample(map[string]float64{"X": 5})
	if valsOrig["Y"] != valsCopy["Y"] {
		t.Errorf("copy sample Y=%f, original Y=%f, should be equal", valsCopy["Y"], valsOrig["Y"])
	}

	// Modifying copy's DAG should not affect original.
	_ = c.DAG().AddNode("Z")
	if m.DAG().HasNode("Z") {
		t.Error("modifying copy should not affect original")
	}
}

func TestSimpleCausalModelNonlinearEquation(t *testing.T) {
	// X → Y with Y = X^2.
	dag := NewDAG()
	_ = dag.AddNodes("X", "Y")
	_ = dag.AddEdge("X", "Y")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("Y", func(pv map[string]float64) float64 {
		return pv["X"] * pv["X"]
	})

	vals, err := m.Sample(map[string]float64{"X": 4})
	if err != nil {
		t.Fatalf("Sample failed: %v", err)
	}
	if vals["Y"] != 16 {
		t.Errorf("Y = %f, want 16 (4^2)", vals["Y"])
	}
}

func TestSimpleCausalModelDiamondWithIntervention(t *testing.T) {
	// Classic diamond: X → A, X → B, A → Y, B → Y
	// A = X, B = X, Y = A + B = 2X
	// do(A = 0): Y = 0 + X = X
	dag := NewDAG()
	_ = dag.AddNodes("X", "A", "B", "Y")
	_ = dag.AddEdge("X", "A")
	_ = dag.AddEdge("X", "B")
	_ = dag.AddEdge("A", "Y")
	_ = dag.AddEdge("B", "Y")

	m := NewSimpleCausalModel(dag)
	m.SetEquation("A", func(pv map[string]float64) float64 { return pv["X"] })
	m.SetEquation("B", func(pv map[string]float64) float64 { return pv["X"] })
	m.SetEquation("Y", func(pv map[string]float64) float64 { return pv["A"] + pv["B"] })

	// Observational.
	vals, _ := m.Sample(map[string]float64{"X": 5})
	if vals["Y"] != 10 {
		t.Errorf("observational Y = %f, want 10", vals["Y"])
	}

	// Interventional: do(A=0).
	mutilated := m.Intervene("A", 0)
	vals2, _ := mutilated.Sample(map[string]float64{"X": 5})
	if vals2["A"] != 0 {
		t.Errorf("intervened A = %f, want 0", vals2["A"])
	}
	if vals2["B"] != 5 {
		t.Errorf("B = %f, want 5 (X flows normally)", vals2["B"])
	}
	if vals2["Y"] != 5 {
		t.Errorf("Y = %f, want 5 (0 + 5)", vals2["Y"])
	}
}

func approxEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

func TestSimpleCausalModelSampleEmpty(t *testing.T) {
	dag := NewDAG()
	m := NewSimpleCausalModel(dag)
	vals, err := m.Sample(map[string]float64{})
	if err != nil {
		t.Fatalf("Sample on empty model failed: %v", err)
	}
	if len(vals) != 0 {
		t.Errorf("expected empty result, got %v", vals)
	}
}
