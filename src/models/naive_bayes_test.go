//go:build unit

package models

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestNewNaiveBayes(t *testing.T) {
	nb, err := NewNaiveBayes("class", []string{"f1", "f2"})
	if err != nil {
		t.Fatalf("NewNaiveBayes: %v", err)
	}
	if nb.ClassVariable() != "class" {
		t.Errorf("expected class variable 'class', got %q", nb.ClassVariable())
	}
	feats := nb.Features()
	if len(feats) != 2 || feats[0] != "f1" || feats[1] != "f2" {
		t.Errorf("expected features [f1 f2], got %v", feats)
	}
}

func TestNewNaiveBayesTopology(t *testing.T) {
	nb, err := NewNaiveBayes("C", []string{"X", "Y", "Z"})
	if err != nil {
		t.Fatalf("NewNaiveBayes: %v", err)
	}

	// C should be parent of X, Y, Z.
	nodes := nb.Nodes()
	if len(nodes) != 4 {
		t.Fatalf("expected 4 nodes, got %d", len(nodes))
	}

	edges := nb.Edges()
	if len(edges) != 3 {
		t.Fatalf("expected 3 edges, got %d", len(edges))
	}
	for _, e := range edges {
		if e[0] != "C" {
			t.Errorf("expected all edges from C, got %v", e)
		}
	}
}

func TestNewNaiveBayesErrors(t *testing.T) {
	// Empty class variable.
	if _, err := NewNaiveBayes("", []string{"f1"}); err == nil {
		t.Error("expected error for empty class variable")
	}

	// Empty features.
	if _, err := NewNaiveBayes("C", nil); err == nil {
		t.Error("expected error for nil features")
	}
	if _, err := NewNaiveBayes("C", []string{}); err == nil {
		t.Error("expected error for empty features")
	}

	// Feature same as class.
	if _, err := NewNaiveBayes("C", []string{"C"}); err == nil {
		t.Error("expected error for feature same as class")
	}

	// Duplicate feature.
	if _, err := NewNaiveBayes("C", []string{"f1", "f1"}); err == nil {
		t.Error("expected error for duplicate feature")
	}
}

func buildNaiveBayesData() *tabgo.DataFrame {
	// Simple dataset: binary class, two binary features.
	// class=0: f1 tends to be 0, f2 tends to be 0
	// class=1: f1 tends to be 1, f2 tends to be 1
	rows := [][]any{
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 0},
		{1, 1, 1},
		{1, 1, 1},
		{1, 1, 0},
		{1, 0, 1},
	}
	return tabgo.NewDataFrameFromRows([]string{"class", "f1", "f2"}, rows)
}

func TestNaiveBayesFit(t *testing.T) {
	nb, err := NewNaiveBayes("class", []string{"f1", "f2"})
	if err != nil {
		t.Fatalf("NewNaiveBayes: %v", err)
	}

	data := buildNaiveBayesData()
	if err := nb.Fit(data); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// Check class prior: 4 class=0, 4 class=1 -> [0.5, 0.5]
	classCPD := nb.GetCPD("class")
	if classCPD == nil {
		t.Fatal("no CPD for class")
	}
	classVals := classCPD.ToFactor().Values().Data()
	if math.Abs(classVals[0]-0.5) > 1e-6 || math.Abs(classVals[1]-0.5) > 1e-6 {
		t.Errorf("expected class prior [0.5, 0.5], got %v", classVals)
	}

	// Check f1 CPD: given class=0, P(f1=0|class=0) = 3/4, P(f1=1|class=0) = 1/4
	f1CPD := nb.GetCPD("f1")
	if f1CPD == nil {
		t.Fatal("no CPD for f1")
	}
	f1Vals := f1CPD.ToFactor().Values().Data()
	// Layout: [f1=0|class=0, f1=0|class=1, f1=1|class=0, f1=1|class=1]
	if math.Abs(f1Vals[0]-0.75) > 1e-6 {
		t.Errorf("expected P(f1=0|class=0) = 0.75, got %f", f1Vals[0])
	}
	if math.Abs(f1Vals[1]-0.25) > 1e-6 {
		t.Errorf("expected P(f1=0|class=1) = 0.25, got %f", f1Vals[1])
	}

	// Model should be valid after fitting.
	if err := nb.CheckModel(); err != nil {
		t.Fatalf("CheckModel after Fit: %v", err)
	}
}

func TestNaiveBayesFitNilData(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1"})
	if err := nb.Fit(nil); err == nil {
		t.Error("expected error for nil data")
	}
}

func TestNaiveBayesFitEmptyData(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1"})
	data := tabgo.NewDataFrameFromRows([]string{"class", "f1"}, nil)
	if err := nb.Fit(data); err == nil {
		t.Error("expected error for empty data")
	}
}

func TestNaiveBayesPredict(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1", "f2"})
	trainData := buildNaiveBayesData()
	if err := nb.Fit(trainData); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	// Predict on clear-cut examples.
	testRows := [][]any{
		{0, 0}, // should predict class=0
		{1, 1}, // should predict class=1
	}
	testData := tabgo.NewDataFrameFromRows([]string{"f1", "f2"}, testRows)

	predictions, err := nb.Predict(testData)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}

	if len(predictions) != 2 {
		t.Fatalf("expected 2 predictions, got %d", len(predictions))
	}
	if predictions[0] != 0 {
		t.Errorf("expected prediction 0 for [0,0], got %d", predictions[0])
	}
	if predictions[1] != 1 {
		t.Errorf("expected prediction 1 for [1,1], got %d", predictions[1])
	}
}

func TestNaiveBayesPredictProbability(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1", "f2"})
	trainData := buildNaiveBayesData()
	if err := nb.Fit(trainData); err != nil {
		t.Fatalf("Fit: %v", err)
	}

	testRows := [][]any{
		{0, 0},
	}
	testData := tabgo.NewDataFrameFromRows([]string{"f1", "f2"}, testRows)

	probs, err := nb.PredictProbability(testData)
	if err != nil {
		t.Fatalf("PredictProbability: %v", err)
	}

	if len(probs) != 1 {
		t.Fatalf("expected 1 row of probabilities, got %d", len(probs))
	}
	if len(probs[0]) != 2 {
		t.Fatalf("expected 2 class probabilities, got %d", len(probs[0]))
	}

	// Probabilities should sum to 1.
	sum := probs[0][0] + probs[0][1]
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("probabilities sum to %f, expected 1.0", sum)
	}

	// For [f1=0, f2=0], class=0 should be more likely.
	if probs[0][0] <= probs[0][1] {
		t.Errorf("expected P(class=0) > P(class=1) for [0,0], got %v", probs[0])
	}
}

func TestNaiveBayesPredictNilData(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1"})
	_, err := nb.Predict(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestNaiveBayesPredictProbabilityNilData(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1"})
	_, err := nb.PredictProbability(nil)
	if err == nil {
		t.Error("expected error for nil data")
	}
}

func TestNaiveBayesPredictUnfittedModel(t *testing.T) {
	nb, _ := NewNaiveBayes("class", []string{"f1"})
	testData := tabgo.NewDataFrameFromRows([]string{"f1"}, [][]any{{0}})
	_, err := nb.Predict(testData)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}
