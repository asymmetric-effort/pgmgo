//go:build unit

package testutil

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestFixturesRoot(t *testing.T) {
	root := fixturesRoot()
	if root == "" {
		t.Fatal("fixturesRoot returned empty string")
	}
	// Should point to a real directory
	info, err := os.Stat(root)
	if err != nil {
		t.Fatalf("fixturesRoot path does not exist: %v", err)
	}
	if !info.IsDir() {
		t.Fatalf("fixturesRoot is not a directory: %s", root)
	}
}

func TestLoadFixtures_MissingFile(t *testing.T) {
	// LoadFixtures should skip (not fail) when fixture file is missing.
	ff := LoadFixtures(t, "nonexistent/fixtures.json")
	if ff != nil {
		t.Error("expected nil for missing fixture file")
	}
}

func TestLoadFixtures_ModelsFixtures(t *testing.T) {
	ff := LoadFixtures(t, "models/fixtures.json")
	if ff == nil {
		return // skipped
	}

	if ff.Generator != "models" {
		t.Errorf("expected generator=models, got %s", ff.Generator)
	}
	if ff.PgmpyVersion == "" {
		t.Error("expected non-empty pgmpy_version")
	}
	if ff.GeneratedAt == "" {
		t.Error("expected non-empty generated_at")
	}
	if len(ff.TestCases) == 0 {
		t.Fatal("expected at least one test case")
	}

	// Verify we can find a known test case
	tc := ff.FindTestCase(t, "bayesian_network_structure")
	if tc == nil {
		return
	}
	if tc.Description == "" {
		t.Error("expected non-empty description")
	}

	// Verify input can be unmarshaled
	var input struct {
		Edges [][]string `json:"edges"`
	}
	tc.UnmarshalInput(t, &input)
	if len(input.Edges) == 0 {
		t.Error("expected non-empty edges in input")
	}

	// Verify expected output can be unmarshaled
	var expected struct {
		Nodes    []string            `json:"nodes"`
		Edges    [][]string          `json:"edges"`
		NumNodes int                 `json:"num_nodes"`
		NumEdges int                 `json:"num_edges"`
		Parents  map[string][]string `json:"parents"`
		Children map[string][]string `json:"children"`
	}
	tc.UnmarshalExpected(t, &expected)
	if len(expected.Nodes) == 0 {
		t.Error("expected non-empty nodes in expected output")
	}
	if expected.NumNodes != len(expected.Nodes) {
		t.Errorf("num_nodes=%d doesn't match len(nodes)=%d", expected.NumNodes, len(expected.Nodes))
	}
}

func TestLoadFixtures_FactorsFixtures(t *testing.T) {
	ff := LoadFixtures(t, "factors/fixtures.json")
	if ff == nil {
		return
	}

	if ff.Generator != "factors" {
		t.Errorf("expected generator=factors, got %s", ff.Generator)
	}
	if len(ff.TestCases) < 5 {
		t.Errorf("expected at least 5 factor test cases, got %d", len(ff.TestCases))
	}

	// Verify discrete_factor_creation fixture
	tc := ff.FindTestCase(t, "discrete_factor_creation")
	if tc == nil {
		return
	}

	var input struct {
		Variables   []string  `json:"variables"`
		Cardinality []int     `json:"cardinality"`
		Values      []float64 `json:"values"`
	}
	tc.UnmarshalInput(t, &input)
	if len(input.Variables) != 2 {
		t.Errorf("expected 2 variables, got %d", len(input.Variables))
	}
	if len(input.Values) != 6 {
		t.Errorf("expected 6 values, got %d", len(input.Values))
	}

	// Verify factor_product fixture
	tc = ff.FindTestCase(t, "discrete_factor_product")
	if tc == nil {
		return
	}

	var prodInput struct {
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
	tc.UnmarshalInput(t, &prodInput)
	if len(prodInput.Factor1.Variables) == 0 {
		t.Error("expected non-empty factor1 variables")
	}
}

func TestLoadFixtures_InferenceFixtures(t *testing.T) {
	ff := LoadFixtures(t, "inference/fixtures.json")
	if ff == nil {
		return
	}

	if ff.Generator != "inference" {
		t.Errorf("expected generator=inference, got %s", ff.Generator)
	}
	if len(ff.TestCases) < 2 {
		t.Errorf("expected at least 2 inference test cases, got %d", len(ff.TestCases))
	}

	// Verify VE query fixture
	tc := ff.FindTestCase(t, "variable_elimination_query")
	if tc == nil {
		return
	}

	var expected struct {
		Variables []string  `json:"variables"`
		Values    []float64 `json:"values"`
	}
	tc.UnmarshalExpected(t, &expected)
	if len(expected.Variables) == 0 {
		t.Error("expected non-empty variables in VE result")
	}
	if len(expected.Values) == 0 {
		t.Error("expected non-empty values in VE result")
	}

	// Values should sum to ~1.0 (it's a probability distribution)
	sum := 0.0
	for _, v := range expected.Values {
		sum += v
	}
	if sum < 0.99 || sum > 1.01 {
		t.Errorf("VE query result should sum to ~1.0, got %f", sum)
	}
}

func TestFindTestCase_Missing(t *testing.T) {
	ff := LoadFixtures(t, "models/fixtures.json")
	if ff == nil {
		return
	}

	// FindTestCase should skip when name not found
	tc := ff.FindTestCase(t, "nonexistent_test_case")
	if tc != nil {
		t.Error("expected nil for nonexistent test case")
	}
}

func TestUnmarshalTestCase(t *testing.T) {
	tc := TestCase{
		Name:     "test_example",
		Input:    json.RawMessage(`{"key": "value"}`),
		Expected: json.RawMessage(`{"result": 42}`),
	}

	var input map[string]string
	tc.UnmarshalInput(t, &input)
	if input["key"] != "value" {
		t.Errorf("expected key=value, got key=%s", input["key"])
	}

	var expected map[string]int
	tc.UnmarshalExpected(t, &expected)
	if expected["result"] != 42 {
		t.Errorf("expected result=42, got result=%d", expected["result"])
	}
}

func TestFixtureFileRoundTrip(t *testing.T) {
	tmpDir := t.TempDir()
	pkgDir := filepath.Join(tmpDir, "test_pkg")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}

	ff := FixtureFile{
		Generator:    "test_pkg",
		PgmpyVersion: "1.1.2",
		GeneratedAt:  "2026-06-01T00:00:00Z",
		TestCases: []TestCase{
			{
				Name:        "example_a",
				Description: "first test",
				Input:       json.RawMessage(`{"x": 1}`),
				Expected:    json.RawMessage(`{"y": 2}`),
			},
			{
				Name:        "example_b",
				Description: "second test",
				Input:       json.RawMessage(`{"x": 3}`),
				Expected:    json.RawMessage(`{"y": 4}`),
			},
		},
	}

	data, err := json.Marshal(ff)
	if err != nil {
		t.Fatal(err)
	}

	outPath := filepath.Join(pkgDir, "fixtures.json")
	if err := os.WriteFile(outPath, data, 0o644); err != nil {
		t.Fatal(err)
	}

	// Read it back
	readData, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatal(err)
	}

	var loaded FixtureFile
	if err := json.Unmarshal(readData, &loaded); err != nil {
		t.Fatal(err)
	}

	if loaded.Generator != "test_pkg" {
		t.Errorf("expected generator=test_pkg, got %s", loaded.Generator)
	}
	if loaded.PgmpyVersion != "1.1.2" {
		t.Errorf("expected pgmpy_version=1.1.2, got %s", loaded.PgmpyVersion)
	}
	if len(loaded.TestCases) != 2 {
		t.Fatalf("expected 2 test cases, got %d", len(loaded.TestCases))
	}
	if loaded.TestCases[0].Name != "example_a" {
		t.Errorf("expected first test case name=example_a, got %s", loaded.TestCases[0].Name)
	}
	if loaded.TestCases[1].Name != "example_b" {
		t.Errorf("expected second test case name=example_b, got %s", loaded.TestCases[1].Name)
	}
}
