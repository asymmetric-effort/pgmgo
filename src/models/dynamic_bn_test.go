//go:build unit

package models

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

func buildSimpleDynamicBN(t *testing.T) *DynamicBayesianNetwork {
	t.Helper()
	dbn := NewDynamicBayesianNetwork()

	// Initial network (t=0): X -> Y
	if err := dbn.Initial().AddNode("X"); err != nil {
		t.Fatalf("AddNode X: %v", err)
	}
	if err := dbn.Initial().AddNode("Y"); err != nil {
		t.Fatalf("AddNode Y: %v", err)
	}
	if err := dbn.Initial().AddEdge("X", "Y"); err != nil {
		t.Fatalf("AddEdge X->Y: %v", err)
	}

	cpdX0, err := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	if err != nil {
		t.Fatalf("NewTabularCPD X: %v", err)
	}
	cpdY0, err := factors.NewTabularCPD("Y", 2, [][]float64{
		{0.8, 0.2},
		{0.2, 0.8},
	}, []string{"X"}, []int{2})
	if err != nil {
		t.Fatalf("NewTabularCPD Y: %v", err)
	}

	if err := dbn.AddInitialCPD(cpdX0); err != nil {
		t.Fatalf("AddInitialCPD X: %v", err)
	}
	if err := dbn.AddInitialCPD(cpdY0); err != nil {
		t.Fatalf("AddInitialCPD Y: %v", err)
	}

	// Transition network: X -> Y (same structure, shared nodes)
	if err := dbn.Transition().AddNode("X"); err != nil {
		t.Fatalf("AddNode X (transition): %v", err)
	}
	if err := dbn.Transition().AddNode("Y"); err != nil {
		t.Fatalf("AddNode Y (transition): %v", err)
	}
	if err := dbn.Transition().AddEdge("X", "Y"); err != nil {
		t.Fatalf("AddEdge X->Y (transition): %v", err)
	}

	cpdXT, err := factors.NewTabularCPD("X", 2, [][]float64{{0.7}, {0.3}}, nil, nil)
	if err != nil {
		t.Fatalf("NewTabularCPD X (transition): %v", err)
	}
	cpdYT, err := factors.NewTabularCPD("Y", 2, [][]float64{
		{0.9, 0.1},
		{0.1, 0.9},
	}, []string{"X"}, []int{2})
	if err != nil {
		t.Fatalf("NewTabularCPD Y (transition): %v", err)
	}

	if err := dbn.AddTransitionCPD(cpdXT); err != nil {
		t.Fatalf("AddTransitionCPD X: %v", err)
	}
	if err := dbn.AddTransitionCPD(cpdYT); err != nil {
		t.Fatalf("AddTransitionCPD Y: %v", err)
	}

	return dbn
}

func TestNewDynamicBayesianNetwork(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	if dbn == nil {
		t.Fatal("NewDynamicBayesianNetwork returned nil")
	}
	if dbn.Initial() == nil {
		t.Fatal("Initial() returned nil")
	}
	if dbn.Transition() == nil {
		t.Fatal("Transition() returned nil")
	}
}

func TestDynamicBNAddInitialCPD(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	_ = dbn.Initial().AddNode("X")

	cpd, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, nil, nil)
	if err := dbn.AddInitialCPD(cpd); err != nil {
		t.Fatalf("AddInitialCPD: %v", err)
	}

	got := dbn.Initial().GetCPD("X")
	if got == nil {
		t.Fatal("expected CPD in initial network")
	}
}

func TestDynamicBNAddInitialCPDNil(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	if err := dbn.AddInitialCPD(nil); err == nil {
		t.Error("expected error for nil CPD")
	}
}

func TestDynamicBNAddTransitionCPD(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	_ = dbn.Transition().AddNode("X")

	cpd, _ := factors.NewTabularCPD("X", 2, [][]float64{{0.6}, {0.4}}, nil, nil)
	if err := dbn.AddTransitionCPD(cpd); err != nil {
		t.Fatalf("AddTransitionCPD: %v", err)
	}

	got := dbn.Transition().GetCPD("X")
	if got == nil {
		t.Fatal("expected CPD in transition network")
	}
}

func TestDynamicBNAddTransitionCPDNil(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	if err := dbn.AddTransitionCPD(nil); err == nil {
		t.Error("expected error for nil CPD")
	}
}

func TestDynamicBNGetInterfaceNodes(t *testing.T) {
	dbn := buildSimpleDynamicBN(t)

	iface := dbn.GetInterfaceNodes()
	if len(iface) != 2 {
		t.Fatalf("expected 2 interface nodes, got %d: %v", len(iface), iface)
	}
	if iface[0] != "X" || iface[1] != "Y" {
		t.Errorf("expected interface nodes [X Y], got %v", iface)
	}
}

func TestDynamicBNGetInterfaceNodesPartial(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	_ = dbn.Initial().AddNode("A")
	_ = dbn.Initial().AddNode("B")
	_ = dbn.Transition().AddNode("B")
	_ = dbn.Transition().AddNode("C")

	iface := dbn.GetInterfaceNodes()
	if len(iface) != 1 || iface[0] != "B" {
		t.Errorf("expected interface nodes [B], got %v", iface)
	}
}

func TestDynamicBNGetInterfaceNodesEmpty(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	_ = dbn.Initial().AddNode("A")
	_ = dbn.Transition().AddNode("B")

	iface := dbn.GetInterfaceNodes()
	if len(iface) != 0 {
		t.Errorf("expected no interface nodes, got %v", iface)
	}
}

func TestDynamicBNCheckModel(t *testing.T) {
	dbn := buildSimpleDynamicBN(t)
	if err := dbn.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestDynamicBNCheckModelInvalidInitial(t *testing.T) {
	dbn := buildSimpleDynamicBN(t)
	dbn.Initial().RemoveCPD("X")
	if err := dbn.CheckModel(); err == nil {
		t.Error("expected error for invalid initial network")
	}
}

func TestDynamicBNCheckModelInvalidTransition(t *testing.T) {
	dbn := buildSimpleDynamicBN(t)
	dbn.Transition().RemoveCPD("Y")
	if err := dbn.CheckModel(); err == nil {
		t.Error("expected error for invalid transition network")
	}
}

func TestDynamicBNCopy(t *testing.T) {
	dbn := buildSimpleDynamicBN(t)
	cpy := dbn.Copy()

	if err := cpy.CheckModel(); err != nil {
		t.Fatalf("copied model CheckModel: %v", err)
	}

	// Modify copy and ensure original is unaffected.
	cpy.Initial().RemoveCPD("X")
	if dbn.Initial().GetCPD("X") == nil {
		t.Error("original was affected by copy modification")
	}
}

func TestDynamicBNEmptyCheckModel(t *testing.T) {
	dbn := NewDynamicBayesianNetwork()
	// Empty networks are valid.
	if err := dbn.CheckModel(); err != nil {
		t.Fatalf("empty CheckModel: %v", err)
	}
}
