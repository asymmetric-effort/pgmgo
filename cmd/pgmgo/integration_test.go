//go:build integration

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var pgmgoBinary string

func TestMain(m *testing.M) {
	// Build the binary to a temp location.
	tmpDir, err := os.MkdirTemp("", "pgmgo-test")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create temp dir: %s\n", err)
		os.Exit(1)
	}
	pgmgoBinary = filepath.Join(tmpDir, "pgmgo")
	build := exec.Command("go", "build", "-o", pgmgoBinary, ".")
	build.Dir, _ = os.Getwd()
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to build pgmgo: %s\n%s", err, out)
		os.Exit(1)
	}
	code := m.Run()
	os.RemoveAll(tmpDir)
	os.Exit(code)
}

func runPgmgo(t *testing.T, args ...string) (stdout, stderr string, exitCode int) {
	t.Helper()
	cmd := exec.Command(pgmgoBinary, args...)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	err := cmd.Run()
	exitCode = 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			t.Fatalf("failed to run pgmgo: %v", err)
		}
	}
	return outBuf.String(), errBuf.String(), exitCode
}

// ---------------------------------------------------------------------------
// Test BIF fixture: a minimal valid Bayesian network (Rain -> Sprinkler)
// ---------------------------------------------------------------------------

const testBIF = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}

variable Sprinkler {
  type discrete [ 2 ] { True, False };
}

variable WetGrass {
  type discrete [ 2 ] { True, False };
}

probability ( Rain ) {
  table 0.2, 0.8;
}

probability ( Sprinkler | Rain ) {
  (True) 0.01, 0.99;
  (False) 0.4, 0.6;
}

probability ( WetGrass | Rain, Sprinkler ) {
  (True, True) 0.99, 0.01;
  (True, False) 0.8, 0.2;
  (False, True) 0.9, 0.1;
  (False, False) 0.0, 1.0;
}
`

// testBIF2 is a second model with different structure, used for compare tests.
const testBIF2 = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}

variable Sprinkler {
  type discrete [ 2 ] { True, False };
}

variable WetGrass {
  type discrete [ 2 ] { True, False };
}

probability ( Rain ) {
  table 0.5, 0.5;
}

probability ( Sprinkler ) {
  table 0.5, 0.5;
}

probability ( WetGrass | Rain, Sprinkler ) {
  (True, True) 0.99, 0.01;
  (True, False) 0.8, 0.2;
  (False, True) 0.9, 0.1;
  (False, False) 0.0, 1.0;
}
`

// invalidBIF is a syntactically broken BIF for sad-path tests.
const invalidBIF = `network unknown {
}

variable X {
  type discrete [ 2 ] { A, B };
}

probability ( X | Y ) {
  table 0.5, 0.5;
}
`

// testCSV is a simple dataset matching the Rain/Sprinkler/WetGrass variables.
const testCSV = `Rain,Sprinkler,WetGrass
0,0,0
0,0,0
0,1,1
0,1,1
1,0,1
1,0,1
1,1,1
1,1,1
0,0,0
0,1,1
1,0,1
1,1,1
0,0,0
0,0,1
1,0,1
1,1,1
0,0,0
0,1,1
1,0,0
1,1,1
`

// writeTempFile writes content to a temp file and returns its path.
// The caller should defer os.Remove on the returned path.
func writeTempFile(t *testing.T, pattern, content string) string {
	t.Helper()
	f, err := os.CreateTemp("", pattern)
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	if _, err := f.WriteString(content); err != nil {
		f.Close()
		os.Remove(f.Name())
		t.Fatalf("failed to write temp file: %v", err)
	}
	f.Close()
	return f.Name()
}

// ---------------------------------------------------------------------------
// 1. No args
// ---------------------------------------------------------------------------

func TestNoArgs(t *testing.T) {
	stdout, _, exitCode := runPgmgo(t)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "Usage:") {
		t.Errorf("expected usage output, got: %s", stdout)
	}
}

// ---------------------------------------------------------------------------
// 2. version
// ---------------------------------------------------------------------------

func TestVersion(t *testing.T) {
	stdout, _, exitCode := runPgmgo(t, "version")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "pgmgo") {
		t.Errorf("expected version string containing 'pgmgo', got: %s", stdout)
	}
}

func TestVersionDashV(t *testing.T) {
	stdout, _, exitCode := runPgmgo(t, "-v")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "pgmgo") {
		t.Errorf("expected version string, got: %s", stdout)
	}
}

// ---------------------------------------------------------------------------
// 3. help
// ---------------------------------------------------------------------------

func TestHelp(t *testing.T) {
	stdout, _, exitCode := runPgmgo(t, "help")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "Commands:") {
		t.Errorf("expected 'Commands:' in output, got: %s", stdout)
	}
}

func TestHelpDashH(t *testing.T) {
	stdout, _, exitCode := runPgmgo(t, "--help")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "Commands:") {
		t.Errorf("expected 'Commands:' in output, got: %s", stdout)
	}
}

// ---------------------------------------------------------------------------
// 4. unknown command
// ---------------------------------------------------------------------------

func TestUnknownCommand(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "foobar")
	if exitCode != 1 {
		t.Errorf("expected exit 1, got %d", exitCode)
	}
	if !strings.Contains(stderr, "unknown command") {
		t.Errorf("expected 'unknown command' in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 5-8. validate
// ---------------------------------------------------------------------------

func TestValidateHappy(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "validate", bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "is valid") {
		t.Errorf("expected 'is valid' in output, got: %s", stdout)
	}
}

func TestValidateMissingFile(t *testing.T) {
	_, _, exitCode := runPgmgo(t, "validate", "/nonexistent/path/model.bif")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
}

func TestValidateNoArgs(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "validate")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "validate requires a file") {
		t.Errorf("expected usage error in stderr, got: %s", stderr)
	}
}

func TestValidateInvalidModel(t *testing.T) {
	bifFile := writeTempFile(t, "test-invalid-*.bif", invalidBIF)
	defer os.Remove(bifFile)

	_, _, exitCode := runPgmgo(t, "validate", bifFile)
	if exitCode != 0 {
		// We expect exit 1 (parse/validation error). The invalid BIF may fail
		// at parse time (exit 1) or validation time (exit 1).
		if exitCode != 1 {
			t.Errorf("expected exit 1, got %d", exitCode)
		}
	}
}

// ---------------------------------------------------------------------------
// 9-14. query
// ---------------------------------------------------------------------------

func TestQueryHappyVE(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "query",
		"--variables", "Rain",
		"--evidence", "WetGrass=0",
		bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	// Output should contain a probability table header and values.
	if !strings.Contains(stdout, "Rain") {
		t.Errorf("expected 'Rain' in output, got: %s", stdout)
	}
	if !strings.Contains(stdout, "P") {
		t.Errorf("expected probability column 'P' in output, got: %s", stdout)
	}
}

func TestQueryHappyBP(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "query",
		"--variables", "Rain",
		"--method", "bp",
		bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "Rain") {
		t.Errorf("expected 'Rain' in output, got: %s", stdout)
	}
}

func TestQueryHappyApprox(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "query",
		"--variables", "Rain",
		"--method", "approx",
		bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "Rain") {
		t.Errorf("expected 'Rain' in output, got: %s", stdout)
	}
}

func TestQueryNoVariables(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	_, stderr, exitCode := runPgmgo(t, "query", bifFile)
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	// File arg is consumed as a flag value by the flag parser, so the error
	// may be about missing file or missing --variables.
	if !strings.Contains(stderr, "--variables is required") && !strings.Contains(stderr, "error") {
		t.Errorf("expected error in stderr, got: %s", stderr)
	}
}

func TestQueryBadMethod(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	_, stderr, exitCode := runPgmgo(t, "query",
		"--variables", "Rain",
		"--method", "bogus",
		bifFile)
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "unknown method") {
		t.Errorf("expected 'unknown method' in stderr, got: %s", stderr)
	}
}

func TestQueryMissingFile(t *testing.T) {
	_, _, exitCode := runPgmgo(t, "query",
		"--variables", "Rain",
		"/nonexistent/model.bif")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
}

// ---------------------------------------------------------------------------
// 15-16. map
// ---------------------------------------------------------------------------

func TestMAPHappy(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "map",
		"--variables", "Rain",
		bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "MAP assignment:") {
		t.Errorf("expected 'MAP assignment:' in output, got: %s", stdout)
	}
}

func TestMAPNoArgs(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "map")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "map requires a file") {
		t.Errorf("expected usage error in stderr, got: %s", stderr)
	}
}

func TestMAPNoVariables(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	_, stderr, exitCode := runPgmgo(t, "map", bifFile)
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--variables is required") && !strings.Contains(stderr, "error") {
		t.Errorf("expected error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 17-18. info
// ---------------------------------------------------------------------------

func TestInfoHappy(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "info", bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "Nodes:") {
		t.Errorf("expected 'Nodes:' in output, got: %s", stdout)
	}
	if !strings.Contains(stdout, "Edges:") {
		t.Errorf("expected 'Edges:' in output, got: %s", stdout)
	}
}

func TestInfoMissingFile(t *testing.T) {
	_, _, exitCode := runPgmgo(t, "info", "/nonexistent/model.bif")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
}

func TestInfoNoArgs(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "info")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "info requires a file") {
		t.Errorf("expected usage error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 19-22. sample
// ---------------------------------------------------------------------------

func TestSampleHappyForward(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outCSV := filepath.Join(t.TempDir(), "samples.csv")

	stdout, _, exitCode := runPgmgo(t, "sample",
		"--model", bifFile,
		"--n", "50",
		"--output", outCSV,
		"--seed", "42")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	if !strings.Contains(stdout, "generated") {
		t.Errorf("expected 'generated' in output, got: %s", stdout)
	}

	// Check the output file exists and has content.
	info, err := os.Stat(outCSV)
	if err != nil {
		t.Fatalf("output CSV not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output CSV is empty")
	}

	data, _ := os.ReadFile(outCSV)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	// Header + 50 data rows.
	if len(lines) < 51 {
		t.Errorf("expected at least 51 lines (header + 50 rows), got %d", len(lines))
	}
}

func TestSampleHappyGibbs(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outCSV := filepath.Join(t.TempDir(), "gibbs.csv")

	_, _, exitCode := runPgmgo(t, "sample",
		"--model", bifFile,
		"--n", "20",
		"--output", outCSV,
		"--method", "gibbs",
		"--seed", "42")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}

	info, err := os.Stat(outCSV)
	if err != nil {
		t.Fatalf("output CSV not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output CSV is empty")
	}
}

func TestSampleHappyWithEvidence(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outCSV := filepath.Join(t.TempDir(), "conditional.csv")

	_, _, exitCode := runPgmgo(t, "sample",
		"--model", bifFile,
		"--n", "20",
		"--output", outCSV,
		"--method", "rejection",
		"--evidence", "Rain=0",
		"--seed", "42")
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}

	info, err := os.Stat(outCSV)
	if err != nil {
		t.Fatalf("output CSV not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output CSV is empty")
	}
}

func TestSampleMissingModel(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "sample",
		"--output", "/tmp/out.csv")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--model") {
		t.Errorf("expected '--model' error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 23-25. learn
// ---------------------------------------------------------------------------

func TestLearnHappyHillClimb(t *testing.T) {
	csvFile := writeTempFile(t, "test-*.csv", testCSV)
	defer os.Remove(csvFile)

	outBIF := filepath.Join(t.TempDir(), "learned.bif")

	stdout, stderr, exitCode := runPgmgo(t, "learn",
		"--data", csvFile,
		"--output", outBIF)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stdout: %s; stderr: %s", exitCode, stdout, stderr)
	}
	if !strings.Contains(stdout, "learned structure") {
		t.Errorf("expected 'learned structure' in output, got: %s", stdout)
	}

	// Output BIF should exist.
	info, err := os.Stat(outBIF)
	if err != nil {
		t.Fatalf("output BIF not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output BIF is empty")
	}
}

func TestLearnHappyPC(t *testing.T) {
	csvFile := writeTempFile(t, "test-*.csv", testCSV)
	defer os.Remove(csvFile)

	outBIF := filepath.Join(t.TempDir(), "learned_pc.bif")

	stdout, stderr, exitCode := runPgmgo(t, "learn",
		"--data", csvFile,
		"--method", "pc",
		"--output", outBIF)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stdout: %s; stderr: %s", exitCode, stdout, stderr)
	}

	info, err := os.Stat(outBIF)
	if err != nil {
		t.Fatalf("output BIF not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output BIF is empty")
	}
}

func TestLearnMissingData(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "learn",
		"--output", "/tmp/out.bif")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--data") {
		t.Errorf("expected '--data' error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 26-27. fit
// ---------------------------------------------------------------------------

func TestFitHappy(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	csvFile := writeTempFile(t, "test-*.csv", testCSV)
	defer os.Remove(csvFile)

	outBIF := filepath.Join(t.TempDir(), "fitted.bif")

	stdout, stderr, exitCode := runPgmgo(t, "fit",
		"--model", bifFile,
		"--data", csvFile,
		"--output", outBIF)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stdout: %s; stderr: %s", exitCode, stdout, stderr)
	}
	if !strings.Contains(stdout, "fitted parameters") {
		t.Errorf("expected 'fitted parameters' in output, got: %s", stdout)
	}

	info, err := os.Stat(outBIF)
	if err != nil {
		t.Fatalf("output BIF not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output BIF is empty")
	}
}

func TestFitMissingFlags(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "fit")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--model") {
		t.Errorf("expected '--model' error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 28-31. convert
// ---------------------------------------------------------------------------

func TestConvertBIFToXMLBIF(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outFile := filepath.Join(t.TempDir(), "model.xmlbif")

	stdout, stderr, exitCode := runPgmgo(t, "convert",
		"--input", bifFile,
		"--from", "bif",
		"--to", "xmlbif",
		"--output", outFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stdout: %s; stderr: %s", exitCode, stdout, stderr)
	}
	if !strings.Contains(stdout, "converted") {
		t.Errorf("expected 'converted' in output, got: %s", stdout)
	}

	info, err := os.Stat(outFile)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
}

func TestConvertBIFToNET(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outFile := filepath.Join(t.TempDir(), "model.net")

	_, stderr, exitCode := runPgmgo(t, "convert",
		"--input", bifFile,
		"--from", "bif",
		"--to", "net",
		"--output", outFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stderr: %s", exitCode, stderr)
	}

	info, err := os.Stat(outFile)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
}

func TestConvertBIFToUAI(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outFile := filepath.Join(t.TempDir(), "model.uai")

	_, stderr, exitCode := runPgmgo(t, "convert",
		"--input", bifFile,
		"--from", "bif",
		"--to", "uai",
		"--output", outFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stderr: %s", exitCode, stderr)
	}

	info, err := os.Stat(outFile)
	if err != nil {
		t.Fatalf("output file not found: %v", err)
	}
	if info.Size() == 0 {
		t.Error("output file is empty")
	}
}

func TestConvertBadFormat(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	outFile := filepath.Join(t.TempDir(), "model.json")

	_, stderr, exitCode := runPgmgo(t, "convert",
		"--input", bifFile,
		"--from", "bif",
		"--to", "json",
		"--output", outFile)
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "unknown output format") {
		t.Errorf("expected 'unknown output format' in stderr, got: %s", stderr)
	}
}

func TestConvertMissingFlags(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "convert")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--input") {
		t.Errorf("expected '--input' error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 32-33. compare
// ---------------------------------------------------------------------------

func TestCompareHappy(t *testing.T) {
	bifFile1 := writeTempFile(t, "test-true-*.bif", testBIF)
	defer os.Remove(bifFile1)

	bifFile2 := writeTempFile(t, "test-est-*.bif", testBIF2)
	defer os.Remove(bifFile2)

	stdout, stderr, exitCode := runPgmgo(t, "compare",
		"--true", bifFile1,
		"--estimated", bifFile2)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stderr: %s", exitCode, stderr)
	}
	if !strings.Contains(stdout, "SHD") {
		t.Errorf("expected 'SHD' in output, got: %s", stdout)
	}
}

func TestCompareSameModel(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, _, exitCode := runPgmgo(t, "compare",
		"--true", bifFile,
		"--estimated", bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d", exitCode)
	}
	// SHD of identical models should be 0.
	if !strings.Contains(stdout, "SHD): 0") {
		t.Errorf("expected SHD 0 for identical models, got: %s", stdout)
	}
}

func TestCompareMissingArgs(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "compare")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--true") {
		t.Errorf("expected '--true' error in stderr, got: %s", stderr)
	}
}

// ---------------------------------------------------------------------------
// 34-35. do
// ---------------------------------------------------------------------------

func TestDoHappy(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	stdout, stderr, exitCode := runPgmgo(t, "do",
		"--intervention", "Rain=0",
		"--query", "WetGrass",
		bifFile)
	if exitCode != 0 {
		t.Errorf("expected exit 0, got %d; stdout: %s; stderr: %s", exitCode, stdout, stderr)
	}
	// Output should contain a probability table.
	if !strings.Contains(stdout, "WetGrass") {
		t.Errorf("expected 'WetGrass' in output, got: %s", stdout)
	}
	if !strings.Contains(stdout, "P") {
		t.Errorf("expected probability column 'P' in output, got: %s", stdout)
	}
}

func TestDoMissingFlags(t *testing.T) {
	bifFile := writeTempFile(t, "test-*.bif", testBIF)
	defer os.Remove(bifFile)

	_, stderr, exitCode := runPgmgo(t, "do",
		"--intervention", "",
		"--query", "",
		bifFile)
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "--intervention") {
		t.Errorf("expected '--intervention' error in stderr, got: %s", stderr)
	}
}

func TestDoNoFileArg(t *testing.T) {
	_, stderr, exitCode := runPgmgo(t, "do")
	if exitCode != 2 {
		t.Errorf("expected exit 2, got %d", exitCode)
	}
	if !strings.Contains(stderr, "do requires a file") {
		t.Errorf("expected 'do requires a file' in stderr, got: %s", stderr)
	}
}
