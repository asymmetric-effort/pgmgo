//go:build unit

package testutil

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mockTB implements testing.TB for testing Fatalf/Skipf paths without
// actually terminating the test goroutine.
type mockTB struct {
	testing.TB
	fatalCalled bool
	fatalMsg    string
	skipCalled  bool
	skipMsg     string
}

func (m *mockTB) Helper() {}

func (m *mockTB) Fatalf(format string, args ...any) {
	m.fatalCalled = true
	m.fatalMsg = fmt.Sprintf(format, args...)
}

func (m *mockTB) Skipf(format string, args ...any) {
	m.skipCalled = true
	m.skipMsg = fmt.Sprintf(format, args...)
}

// --- loadFixtureData tests ---

func TestLoadFixtureData_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(p, []byte(`{not json`), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	_, err := loadFixtureData(p)
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestLoadFixtureData_ValidJSON(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "good.json")
	content := `{"generator":"test","pgmpy_version":"0.1","test_cases":[]}`
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}

	ff, err := loadFixtureData(p)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ff.Generator != "test" {
		t.Errorf("expected generator 'test', got %q", ff.Generator)
	}
}

func TestLoadFixtureData_FileNotFound(t *testing.T) {
	_, err := loadFixtureData("/nonexistent/path/file.json")
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

// --- unmarshalRawJSON tests ---

func TestUnmarshalRawJSON_InvalidData(t *testing.T) {
	raw := json.RawMessage(`{not valid json}`)
	var target map[string]string
	err := unmarshalRawJSON(raw, &target)
	if err == nil {
		t.Fatal("expected error for invalid JSON, got nil")
	}
}

func TestUnmarshalRawJSON_TypeMismatch(t *testing.T) {
	raw := json.RawMessage(`"just a string"`)
	var target struct {
		X int `json:"x"`
	}
	err := unmarshalRawJSON(raw, &target)
	if err == nil {
		t.Fatal("expected error for type mismatch, got nil")
	}
}

func TestUnmarshalRawJSON_Valid(t *testing.T) {
	raw := json.RawMessage(`{"key":"value"}`)
	var target map[string]string
	err := unmarshalRawJSON(raw, &target)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if target["key"] != "value" {
		t.Errorf("expected 'value', got %q", target["key"])
	}
}

// --- LoadFixtures tests using mockTB ---

func TestLoadFixtures_InvalidJSON_Fatalf(t *testing.T) {
	// Create a file with invalid JSON under the fixtures root so LoadFixtures
	// can find it and hit the Fatalf path (JSON parse error, not file-not-found).
	root := fixturesRoot()
	if err := os.MkdirAll(root, 0755); err != nil {
		t.Fatalf("setup: %v", err)
	}
	badFile := filepath.Join(root, "_test_bad_fixture.json")
	if err := os.WriteFile(badFile, []byte(`{invalid`), 0644); err != nil {
		t.Fatalf("setup: %v", err)
	}
	defer os.Remove(badFile)

	m := &mockTB{}
	LoadFixtures(m, "_test_bad_fixture.json")
	if !m.fatalCalled {
		t.Fatal("expected Fatalf to be called for invalid JSON fixture file")
	}
	if !strings.Contains(m.fatalMsg, "failed to parse fixture file") {
		t.Errorf("unexpected fatal message: %s", m.fatalMsg)
	}
}

func TestLoadFixtures_FileNotFound_Skipf(t *testing.T) {
	m := &mockTB{}
	// LoadFixtures with a path that doesn't exist under fixturesRoot().
	// fixturesRoot() points to tests/fixtures/ relative to this source file.
	result := LoadFixtures(m, "nonexistent/does_not_exist_12345.json")
	if !m.skipCalled {
		t.Fatal("expected Skipf to be called for missing fixture file")
	}
	if result != nil {
		t.Fatal("expected nil return for missing fixture file")
	}
	if !strings.Contains(m.skipMsg, "fixture file not found") {
		t.Errorf("unexpected skip message: %s", m.skipMsg)
	}
}

// --- UnmarshalInput / UnmarshalExpected tests via mockTB ---

func TestUnmarshalInput_InvalidJSON_CallsFatalf(t *testing.T) {
	tc := &TestCase{
		Name:  "bad-input",
		Input: json.RawMessage(`{not valid json}`),
	}
	m := &mockTB{}
	var target map[string]string
	tc.UnmarshalInput(m, &target)

	if !m.fatalCalled {
		t.Fatal("expected Fatalf to be called for invalid input JSON")
	}
	if !strings.Contains(m.fatalMsg, "failed to unmarshal input") {
		t.Errorf("unexpected fatal message: %s", m.fatalMsg)
	}
}

func TestUnmarshalExpected_InvalidJSON_CallsFatalf(t *testing.T) {
	tc := &TestCase{
		Name:     "bad-expected",
		Expected: json.RawMessage(`{not valid json}`),
	}
	m := &mockTB{}
	var target map[string]string
	tc.UnmarshalExpected(m, &target)

	if !m.fatalCalled {
		t.Fatal("expected Fatalf to be called for invalid expected JSON")
	}
	if !strings.Contains(m.fatalMsg, "failed to unmarshal expected") {
		t.Errorf("unexpected fatal message: %s", m.fatalMsg)
	}
}
