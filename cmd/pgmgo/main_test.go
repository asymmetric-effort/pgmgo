//go:build unit

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Shared BIF content
// ---------------------------------------------------------------------------

const validBIF = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}

variable Sprinkler {
  type discrete [ 2 ] { True, False };
}

probability ( Rain ) {
  table 0.2, 0.8;
}

probability ( Sprinkler | Rain ) {
  (True) 0.01, 0.99;
  (False) 0.4, 0.6;
}
`

const invalidBIF = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}
`

const malformedBIF = `this is not a valid BIF file {{{
  broken syntax !!!
`

// validBIF2 has the same structure as validBIF but different parameters.
const validBIF2 = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}

variable Sprinkler {
  type discrete [ 2 ] { True, False };
}

probability ( Rain ) {
  table 0.5, 0.5;
}

probability ( Sprinkler | Rain ) {
  (True) 0.3, 0.7;
  (False) 0.6, 0.4;
}
`

// validBIF3 has reversed edge direction for compare tests.
const validBIF3 = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}

variable Sprinkler {
  type discrete [ 2 ] { True, False };
}

probability ( Sprinkler ) {
  table 0.4, 0.6;
}

probability ( Rain | Sprinkler ) {
  (True) 0.3, 0.7;
  (False) 0.6, 0.4;
}
`

func writeTempBIF(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bif")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp BIF file: %v", err)
	}
	return path
}

func writeTempFile(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp file: %v", err)
	}
	return path
}

// ---------------------------------------------------------------------------
// validate tests
// ---------------------------------------------------------------------------

func TestValidateBIFFile_ValidModel(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := validateBIFFile(path)
	if code != 0 {
		t.Errorf("expected exit code 0 for valid model, got %d", code)
	}
}

func TestValidateBIFFile_MissingCPD(t *testing.T) {
	path := writeTempBIF(t, invalidBIF)
	code := validateBIFFile(path)
	if code != 1 {
		t.Errorf("expected exit code 1 for model with missing CPD, got %d", code)
	}
}

func TestValidateBIFFile_FileNotFound(t *testing.T) {
	code := validateBIFFile("/nonexistent/path/to/file.bif")
	if code != 2 {
		t.Errorf("expected exit code 2 for missing file, got %d", code)
	}
}

func TestRunValidate_NoArgs(t *testing.T) {
	code := runValidate([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2 for missing argument, got %d", code)
	}
}

func TestRunValidate_ValidFile(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := runValidate([]string{path})
	if code != 0 {
		t.Errorf("expected exit code 0 for valid file, got %d", code)
	}
}

func TestVersion(t *testing.T) {
	if version != "0.0.45" {
		t.Errorf("expected version 0.0.45, got %s", version)
	}
}

// ---------------------------------------------------------------------------
// query tests
// ---------------------------------------------------------------------------

func TestQueryBIF_VE(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := queryBIF(path, []string{"Rain"}, nil, "ve")
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestQueryBIF_VE_WithEvidence(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := queryBIF(path, []string{"Rain"}, map[string]int{"Sprinkler": 0}, "ve")
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestQueryBIF_BP(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := queryBIF(path, []string{"Rain"}, nil, "bp")
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestQueryBIF_Approx(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := queryBIF(path, []string{"Rain"}, nil, "approx")
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestQueryBIF_BadMethod(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := queryBIF(path, []string{"Rain"}, nil, "badmethod")
	if code != 2 {
		t.Errorf("expected exit code 2 for bad method, got %d", code)
	}
}

func TestQueryBIF_BadFile(t *testing.T) {
	code := queryBIF("/nonexistent", []string{"Rain"}, nil, "ve")
	if code != 2 {
		t.Errorf("expected exit code 2 for bad file, got %d", code)
	}
}

func TestRunQuery_NoArgs(t *testing.T) {
	code := runQuery([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunQuery_NoVariables(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := runQuery([]string{path})
	if code != 2 {
		t.Errorf("expected exit code 2 for missing variables, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// map tests
// ---------------------------------------------------------------------------

func TestMapBIF_Success(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := mapBIF(path, []string{"Rain"}, nil)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestMapBIF_WithEvidence(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := mapBIF(path, []string{"Rain"}, map[string]int{"Sprinkler": 0})
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestMapBIF_BadFile(t *testing.T) {
	code := mapBIF("/nonexistent", []string{"Rain"}, nil)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunMAP_NoArgs(t *testing.T) {
	code := runMAP([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// info tests
// ---------------------------------------------------------------------------

func TestInfoBIF_Success(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := infoBIF(path)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestInfoBIF_BadFile(t *testing.T) {
	code := infoBIF("/nonexistent")
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunInfo_NoArgs(t *testing.T) {
	code := runInfo([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// convert tests
// ---------------------------------------------------------------------------

func TestConvertModel_BIFToXMLBIF(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "out.xmlbif")
	code := convertModel(path, "bif", "xmlbif", outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Errorf("output file not created: %v", err)
	}
}

func TestConvertModel_BIFToNET(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "out.net")
	code := convertModel(path, "bif", "net", outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestConvertModel_BIFToUAI(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "out.uai")
	code := convertModel(path, "bif", "uai", outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestConvertModel_BIFToXDSL(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "out.xdsl")
	code := convertModel(path, "bif", "xdsl", outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestConvertModel_BadInputFormat(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "out.bif")
	code := convertModel(path, "badformat", "bif", outPath)
	if code != 2 {
		t.Errorf("expected exit code 2 for bad format, got %d", code)
	}
}

func TestConvertModel_BadOutputFormat(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "out.bif")
	code := convertModel(path, "bif", "badformat", outPath)
	if code != 2 {
		t.Errorf("expected exit code 2 for bad output format, got %d", code)
	}
}

func TestConvertModel_BadInputFile(t *testing.T) {
	outPath := filepath.Join(t.TempDir(), "out.bif")
	code := convertModel("/nonexistent", "bif", "xmlbif", outPath)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunConvert_MissingFlags(t *testing.T) {
	code := runConvert([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// compare tests
// ---------------------------------------------------------------------------

func TestCompareModels_SameStructure(t *testing.T) {
	path1 := writeTempBIF(t, validBIF)
	path2 := writeTempBIF(t, validBIF2)
	code := compareModels(path1, path2)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestCompareModels_DifferentStructure(t *testing.T) {
	path1 := writeTempBIF(t, validBIF)
	path2 := writeTempBIF(t, validBIF3)
	code := compareModels(path1, path2)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestCompareModels_BadFile(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := compareModels("/nonexistent", path)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunCompare_MissingFlags(t *testing.T) {
	code := runCompare([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// do tests
// ---------------------------------------------------------------------------

func TestDoCausalQuery_Success(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := doCausalQuery(path, map[string]int{"Rain": 0}, "Sprinkler", nil)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestDoCausalQuery_BadFile(t *testing.T) {
	code := doCausalQuery("/nonexistent", map[string]int{"Rain": 0}, "Sprinkler", nil)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunDo_NoArgs(t *testing.T) {
	code := runDo([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// sample tests
// ---------------------------------------------------------------------------

func TestSampleModel_Forward(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "samples.csv")
	code := sampleModel(path, 10, outPath, "forward", nil, 42)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("cannot read output: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) < 11 {
		t.Errorf("expected at least 11 lines (header + 10 samples), got %d", len(lines))
	}
}

func TestSampleModel_Rejection(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "samples.csv")
	code := sampleModel(path, 5, outPath, "rejection", map[string]int{"Rain": 0}, 42)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestSampleModel_Gibbs(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "samples.csv")
	code := sampleModel(path, 5, outPath, "gibbs", map[string]int{"Rain": 0}, 42)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestSampleModel_BadMethod(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "samples.csv")
	code := sampleModel(path, 5, outPath, "badmethod", nil, 42)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestSampleModel_BadFile(t *testing.T) {
	outPath := filepath.Join(t.TempDir(), "samples.csv")
	code := sampleModel("/nonexistent", 5, outPath, "forward", nil, 42)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunSample_MissingFlags(t *testing.T) {
	code := runSample([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// fit tests
// ---------------------------------------------------------------------------

func TestFitParameters_MLE(t *testing.T) {
	modelPath := writeTempBIF(t, validBIF)
	csvContent := "Rain,Sprinkler\n0,0\n0,1\n1,0\n1,1\n0,0\n1,0\n0,1\n1,1\n0,0\n1,0\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "fitted.bif")
	code := fitParameters(modelPath, csvPath, "mle", outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Errorf("output file not created: %v", err)
	}
}

func TestFitParameters_Bayesian(t *testing.T) {
	modelPath := writeTempBIF(t, validBIF)
	csvContent := "Rain,Sprinkler\n0,0\n0,1\n1,0\n1,1\n0,0\n1,0\n0,1\n1,1\n0,0\n1,0\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "fitted.bif")
	code := fitParameters(modelPath, csvPath, "bayesian", outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestFitParameters_BadMethod(t *testing.T) {
	modelPath := writeTempBIF(t, validBIF)
	csvContent := "Rain,Sprinkler\n0,0\n0,1\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "fitted.bif")
	code := fitParameters(modelPath, csvPath, "badmethod", outPath)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestFitParameters_BadModelFile(t *testing.T) {
	csvContent := "Rain,Sprinkler\n0,0\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "fitted.bif")
	code := fitParameters("/nonexistent", csvPath, "mle", outPath)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestFitParameters_BadCSVFile(t *testing.T) {
	modelPath := writeTempBIF(t, validBIF)
	outPath := filepath.Join(t.TempDir(), "fitted.bif")
	code := fitParameters(modelPath, "/nonexistent.csv", "mle", outPath)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunFit_MissingFlags(t *testing.T) {
	code := runFit([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// learn tests
// ---------------------------------------------------------------------------

func TestLearnStructure_HillClimb(t *testing.T) {
	csvContent := "A,B,C\n0,0,0\n0,0,1\n0,1,0\n0,1,1\n1,0,0\n1,0,1\n1,1,0\n1,1,1\n0,0,0\n1,1,1\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "learned.bif")
	code := learnStructure(csvPath, "hillclimb", "bic", 0.05, outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
	if _, err := os.Stat(outPath); err != nil {
		t.Errorf("output file not created: %v", err)
	}
}

func TestLearnStructure_Tree(t *testing.T) {
	csvContent := "A,B,C\n0,0,0\n0,0,1\n0,1,0\n0,1,1\n1,0,0\n1,0,1\n1,1,0\n1,1,1\n0,0,0\n1,1,1\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "learned.bif")
	code := learnStructure(csvPath, "tree", "bic", 0.05, outPath)
	if code != 0 {
		t.Errorf("expected exit code 0, got %d", code)
	}
}

func TestLearnStructure_BadMethod(t *testing.T) {
	csvContent := "A,B\n0,0\n1,1\n"
	csvPath := writeTempFile(t, "data.csv", csvContent)
	outPath := filepath.Join(t.TempDir(), "learned.bif")
	code := learnStructure(csvPath, "badmethod", "bic", 0.05, outPath)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestLearnStructure_BadCSV(t *testing.T) {
	outPath := filepath.Join(t.TempDir(), "learned.bif")
	code := learnStructure("/nonexistent.csv", "hillclimb", "bic", 0.05, outPath)
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

func TestRunLearn_MissingFlags(t *testing.T) {
	code := runLearn([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2, got %d", code)
	}
}

// ---------------------------------------------------------------------------
// helper function tests
// ---------------------------------------------------------------------------

func TestParseCSVList(t *testing.T) {
	tests := []struct {
		input    string
		expected []string
	}{
		{"", nil},
		{"A,B,C", []string{"A", "B", "C"}},
		{" A , B ", []string{"A", "B"}},
		{"single", []string{"single"}},
	}
	for _, tc := range tests {
		got := parseCSVList(tc.input)
		if len(got) != len(tc.expected) {
			t.Errorf("parseCSVList(%q) = %v, want %v", tc.input, got, tc.expected)
			continue
		}
		for i := range got {
			if got[i] != tc.expected[i] {
				t.Errorf("parseCSVList(%q)[%d] = %q, want %q", tc.input, i, got[i], tc.expected[i])
			}
		}
	}
}

func TestParseEvidenceMap(t *testing.T) {
	m, err := parseEvidenceMap("A=1,B=0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m["A"] != 1 || m["B"] != 0 {
		t.Errorf("got %v, want A=1,B=0", m)
	}

	m, err = parseEvidenceMap("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if m != nil {
		t.Errorf("expected nil for empty string, got %v", m)
	}

	_, err = parseEvidenceMap("badformat")
	if err == nil {
		t.Error("expected error for bad format")
	}

	_, err = parseEvidenceMap("A=notanumber")
	if err == nil {
		t.Error("expected error for non-integer value")
	}
}

func TestGetScoreFunc(t *testing.T) {
	_ = getScoreFunc("bic")
	_ = getScoreFunc("bdeu")
	_ = getScoreFunc("k2")
	_ = getScoreFunc("unknown") // defaults to bic
}
