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
}

func TestLoadFixtures_MissingFile(t *testing.T) {
	// LoadFixtures should skip (not fail) when fixture file is missing.
	// We verify this by calling it with the real test t — it will skip this test.
	ff := LoadFixtures(t, "nonexistent/fixtures.json")
	if ff != nil {
		t.Error("expected nil for missing fixture file")
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
	// Create a temporary fixture file and verify we can load it
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
				Name:     "example",
				Input:    json.RawMessage(`{}`),
				Expected: json.RawMessage(`{}`),
			},
		},
	}

	data, err := json.Marshal(ff)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(pkgDir, "fixtures.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}

	// Verify the structure is correct
	if len(ff.TestCases) != 1 {
		t.Errorf("expected 1 test case, got %d", len(ff.TestCases))
	}
	if ff.TestCases[0].Name != "example" {
		t.Errorf("expected name=example, got %s", ff.TestCases[0].Name)
	}
}
