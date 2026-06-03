//go:build unit

package models

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/base"
)

func buildSEMTestData(nRows int) *tabgo.DataFrame {
	// Generate simple data: X ~ N(0, 1), Y = 2*X + 1 + noise
	rows := make([][]any, nRows)
	for i := 0; i < nRows; i++ {
		x := float64(i-nRows/2) / float64(nRows)
		y := 2*x + 1 + 0.01*float64(i%3-1)
		rows[i] = []any{x, y}
	}
	return tabgo.NewDataFrameFromRows([]string{"X", "Y"}, rows)
}

func TestSEMFit(t *testing.T) {
	s := NewSEM()
	// Set up structure.
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.0}, 0.0, 1.0)

	data := buildSEMTestData(100)
	if err := s.Fit(data); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	eq := s.GetEquation("Y")
	if eq == nil {
		t.Fatal("expected equation for Y after Fit")
	}
	// Coefficient for X should be approximately 2.
	if math.Abs(eq.Coefficients[0]-2.0) > 0.5 {
		t.Errorf("expected coefficient ~2.0, got %f", eq.Coefficients[0])
	}
	// Intercept should be approximately 1.
	if math.Abs(eq.Intercept-1.0) > 0.5 {
		t.Errorf("expected intercept ~1.0, got %f", eq.Intercept)
	}
}

func TestSEMFitNilData(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	if err := s.Fit(nil); err == nil {
		t.Error("expected error for nil data")
	}
}

func TestSEMFitEmptyData(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	data := tabgo.NewDataFrameFromRows([]string{"X"}, nil)
	if err := s.Fit(data); err == nil {
		t.Error("expected error for empty data")
	}
}

func TestSEMGenerateSamples(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 5.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{2.0}, 1.0, 0.5)

	df, err := s.GenerateSamples(100)
	if err != nil {
		t.Fatalf("GenerateSamples: %v", err)
	}
	if df.Len() != 100 {
		t.Errorf("expected 100 rows, got %d", df.Len())
	}

	// Check that columns exist.
	cols := df.Columns()
	if len(cols) != 2 {
		t.Errorf("expected 2 columns, got %d", len(cols))
	}
}

func TestSEMGenerateSamplesInvalid(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)
	// X has no equation so model is invalid.
	_, err := s.GenerateSamples(10)
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestSEMGenerateSamplesNonPositive(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_, err := s.GenerateSamples(0)
	if err == nil {
		t.Error("expected error for zero samples")
	}
}

func TestSEMSetParams(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 1.0, 0.25)

	err := s.SetParams("Y", []float64{2.0}, 3.0, 0.5)
	if err != nil {
		t.Fatalf("SetParams: %v", err)
	}

	eq := s.GetEquation("Y")
	if eq.Coefficients[0] != 2.0 {
		t.Errorf("expected coefficient 2.0, got %f", eq.Coefficients[0])
	}
	if eq.Intercept != 3.0 {
		t.Errorf("expected intercept 3.0, got %f", eq.Intercept)
	}
	if eq.Variance != 0.5 {
		t.Errorf("expected variance 0.5, got %f", eq.Variance)
	}
}

func TestSEMSetParamsNonexistent(t *testing.T) {
	s := NewSEM()
	err := s.SetParams("Z", nil, 0, 1)
	if err == nil {
		t.Error("expected error for nonexistent variable")
	}
}

func TestSEMSetParamsWrongLength(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 1.0, 0.25)
	err := s.SetParams("Y", []float64{1.0, 2.0}, 0, 1)
	if err == nil {
		t.Error("expected error for wrong coefficient length")
	}
}

func TestSEMGetScalingIndicators(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 0.5)
	_ = s.AddEquation("Z", []string{"Y"}, []float64{1.0}, 0.0, 0.5)

	indicators := s.GetScalingIndicators()
	if len(indicators) != 1 || indicators[0] != "X" {
		t.Errorf("expected [X], got %v", indicators)
	}
}

func TestSEMActiveTrailNodes(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 0.5)
	_ = s.AddEquation("Z", []string{"Y"}, []float64{1.0}, 0.0, 0.5)

	// From X with no observations: Y and Z should be reachable.
	trail, err := s.ActiveTrailNodes("X", nil)
	if err != nil {
		t.Fatalf("ActiveTrailNodes: %v", err)
	}
	if len(trail) != 2 {
		t.Errorf("expected 2 active trail nodes from X, got %d: %v", len(trail), trail)
	}
}

func TestSEMActiveTrailNodes_Observed(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 0.5)
	_ = s.AddEquation("Z", []string{"Y"}, []float64{1.0}, 0.0, 0.5)

	// From X with Y observed: Z should be blocked.
	observed := map[string]bool{"Y": true}
	trail, err := s.ActiveTrailNodes("X", observed)
	if err != nil {
		t.Fatalf("ActiveTrailNodes: %v", err)
	}
	if len(trail) != 0 {
		t.Errorf("expected 0 active trail nodes from X with Y observed, got %d: %v", len(trail), trail)
	}
}

func TestSEMActiveTrailNodes_InvalidNode(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_, err := s.ActiveTrailNodes("Q", nil)
	if err == nil {
		t.Error("expected error for nonexistent node")
	}
}

func TestSEMMoralize(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Z", []string{"X", "Y"}, []float64{0.5, 0.3}, 0.0, 1.0)

	moral := s.Moralize()
	if moral == nil {
		t.Fatal("Moralize returned nil")
	}
	// X and Y should be married (connected) in the moral graph.
	if !moral.HasEdge("X", "Y") {
		t.Error("expected edge between X and Y in moral graph")
	}
	// All original edges should be present (undirected).
	if !moral.HasEdge("X", "Z") {
		t.Error("expected edge between X and Z in moral graph")
	}
	if !moral.HasEdge("Y", "Z") {
		t.Error("expected edge between Y and Z in moral graph")
	}
}

func TestFromLavaan(t *testing.T) {
	_, err := FromLavaan("")
	if err == nil {
		t.Error("expected error for empty syntax")
	}

	s, err := FromLavaan("y ~ x")
	if err != nil {
		t.Fatalf("expected FromLavaan to succeed for 'y ~ x', got: %v", err)
	}
	vars := s.Variables()
	if len(vars) != 2 {
		t.Errorf("expected 2 variables, got %d", len(vars))
	}
}

func TestFromGraph(t *testing.T) {
	dag := base.NewDAG()
	_ = dag.AddNode("X")
	_ = dag.AddNode("Y")
	_ = dag.AddEdge("X", "Y")

	s, err := FromGraph(dag)
	if err != nil {
		t.Fatalf("FromGraph: %v", err)
	}

	eq := s.GetEquation("Y")
	if eq == nil {
		t.Fatal("expected equation for Y")
	}
	if len(eq.Parents) != 1 || eq.Parents[0] != "X" {
		t.Errorf("expected parents [X], got %v", eq.Parents)
	}
	if err := s.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestFromGraphNil(t *testing.T) {
	_, err := FromGraph(nil)
	if err == nil {
		t.Error("expected error for nil DAG")
	}
}

func TestFromLisrel_EmptySpec(t *testing.T) {
	_, err := FromLisrel("")
	if err == nil {
		t.Error("expected error for empty LISREL spec")
	}
}

func TestFromRAM_EmptySpec(t *testing.T) {
	_, err := FromRAM("")
	if err == nil {
		t.Error("expected error for empty RAM spec")
	}
}

func TestSEMToLisrel(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	result, err := s.ToLisrel()
	if err != nil {
		t.Fatalf("ToLisrel: %v", err)
	}
	if result != s {
		t.Error("expected ToLisrel to return the same SEM")
	}
}

func TestSEMToStandardLisrel(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	result, err := s.ToStandardLisrel()
	if err != nil {
		t.Fatalf("ToStandardLisrel: %v", err)
	}
	if result != s {
		t.Error("expected ToStandardLisrel to return the same SEM")
	}
}

func TestSEMToSEMGraph(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	result := s.ToSEMGraph()
	if result != s {
		t.Error("expected ToSEMGraph to return the same SEM")
	}
}
