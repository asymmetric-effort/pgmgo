//go:build unit

package factors

import (
	"math"
	"os"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// GetValues
// ---------------------------------------------------------------------------

func TestTabularCPD_GetValues_NoEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("A", 3,
		[][]float64{{0.2}, {0.3}, {0.5}},
		nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	vals := cpd.GetValues()
	if len(vals) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(vals))
	}
	if len(vals[0]) != 1 {
		t.Fatalf("expected 1 column, got %d", len(vals[0]))
	}
	if !floatEq(vals[0][0], 0.2) || !floatEq(vals[1][0], 0.3) || !floatEq(vals[2][0], 0.5) {
		t.Errorf("unexpected values: %v", vals)
	}
}

func TestTabularCPD_GetValues_WithEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.4, 0.9},
			{0.6, 0.1},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	vals := cpd.GetValues()
	if len(vals) != 2 || len(vals[0]) != 2 {
		t.Fatalf("unexpected shape: %d x %d", len(vals), len(vals[0]))
	}
	if !floatEq(vals[0][0], 0.4) || !floatEq(vals[0][1], 0.9) {
		t.Errorf("row 0 = %v", vals[0])
	}
	if !floatEq(vals[1][0], 0.6) || !floatEq(vals[1][1], 0.1) {
		t.Errorf("row 1 = %v", vals[1])
	}
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------

func TestTabularCPD_Normalize(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{2, 3},
			{8, 7},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	norm := cpd.Normalize()
	vals := norm.GetValues()

	// Column 0: 2/10=0.2, 8/10=0.8
	if !floatEq(vals[0][0], 0.2) || !floatEq(vals[1][0], 0.8) {
		t.Errorf("column 0: got %f, %f", vals[0][0], vals[1][0])
	}
	// Column 1: 3/10=0.3, 7/10=0.7
	if !floatEq(vals[0][1], 0.3) || !floatEq(vals[1][1], 0.7) {
		t.Errorf("column 1: got %f, %f", vals[0][1], vals[1][1])
	}
	if err := norm.Validate(); err != nil {
		t.Errorf("normalized CPD failed validation: %v", err)
	}
}

func TestTabularCPD_Normalize_ZeroColumn(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0, 3},
			{0, 7},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	norm := cpd.Normalize()
	vals := norm.GetValues()
	// Zero column should stay zero.
	if vals[0][0] != 0 || vals[1][0] != 0 {
		t.Errorf("zero column changed: %f, %f", vals[0][0], vals[1][0])
	}
}

// ---------------------------------------------------------------------------
// Marginalize
// ---------------------------------------------------------------------------

func TestTabularCPD_Marginalize(t *testing.T) {
	// P(X | Y, Z) with X:2, Y:2, Z:2 => 4 parent configs
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.1, 0.2, 0.3, 0.4},
			{0.9, 0.8, 0.7, 0.6},
		},
		[]string{"Y", "Z"}, []int{2, 2},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Marginalize out Z.
	result, err := cpd.Marginalize([]string{"Z"})
	if err != nil {
		t.Fatal(err)
	}
	if result.Variable() != "X" {
		t.Errorf("variable = %q", result.Variable())
	}
	ev := result.Evidence()
	if len(ev) != 1 || ev[0] != "Y" {
		t.Errorf("evidence = %v", ev)
	}
	vals := result.GetValues()
	// Column 0 (Y=0): 0.1+0.2=0.3, 0.9+0.8=1.7
	if !floatEq(vals[0][0], 0.3) {
		t.Errorf("vals[0][0] = %f, want 0.3", vals[0][0])
	}
	if !floatEq(vals[1][0], 1.7) {
		t.Errorf("vals[1][0] = %f, want 1.7", vals[1][0])
	}
	// Column 1 (Y=1): 0.3+0.4=0.7, 0.7+0.6=1.3
	if !floatEq(vals[0][1], 0.7) {
		t.Errorf("vals[0][1] = %f, want 0.7", vals[0][1])
	}
	if !floatEq(vals[1][1], 1.3) {
		t.Errorf("vals[1][1] = %f, want 1.3", vals[1][1])
	}
}

func TestTabularCPD_Marginalize_ChildError(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{{0.4, 0.9}, {0.6, 0.1}},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = cpd.Marginalize([]string{"X"})
	if err == nil {
		t.Error("expected error marginalizing child variable")
	}
}

func TestTabularCPD_Marginalize_UnknownVariable(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{{0.4, 0.9}, {0.6, 0.1}},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = cpd.Marginalize([]string{"Z"})
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

// ---------------------------------------------------------------------------
// Reduce
// ---------------------------------------------------------------------------

func TestTabularCPD_Reduce(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.1, 0.2, 0.3, 0.4},
			{0.9, 0.8, 0.7, 0.6},
		},
		[]string{"Y", "Z"}, []int{2, 2},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Reduce Z=1.
	result, err := cpd.Reduce(map[string]int{"Z": 1})
	if err != nil {
		t.Fatal(err)
	}
	ev := result.Evidence()
	if len(ev) != 1 || ev[0] != "Y" {
		t.Errorf("evidence = %v", ev)
	}
	vals := result.GetValues()
	// With Z=1: parent configs (Y=0,Z=1) and (Y=1,Z=1) => columns 1 and 3
	if !floatEq(vals[0][0], 0.2) {
		t.Errorf("vals[0][0] = %f, want 0.2", vals[0][0])
	}
	if !floatEq(vals[0][1], 0.4) {
		t.Errorf("vals[0][1] = %f, want 0.4", vals[0][1])
	}
}

func TestTabularCPD_Reduce_ChildError(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{{0.4, 0.9}, {0.6, 0.1}},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = cpd.Reduce(map[string]int{"X": 0})
	if err == nil {
		t.Error("expected error reducing child variable")
	}
}

// ---------------------------------------------------------------------------
// ReorderParents
// ---------------------------------------------------------------------------

func TestTabularCPD_ReorderParents(t *testing.T) {
	// P(X | Y, Z) with Y:2, Z:3 => 6 parent configs.
	// Parent configs in order: (Y=0,Z=0),(Y=0,Z=1),(Y=0,Z=2),(Y=1,Z=0),(Y=1,Z=1),(Y=1,Z=2)
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
			{0.9, 0.8, 0.7, 0.6, 0.5, 0.4},
		},
		[]string{"Y", "Z"}, []int{2, 3},
	)
	if err != nil {
		t.Fatal(err)
	}

	reordered, err := cpd.ReorderParents([]string{"Z", "Y"})
	if err != nil {
		t.Fatal(err)
	}

	ev := reordered.Evidence()
	if len(ev) != 2 || ev[0] != "Z" || ev[1] != "Y" {
		t.Errorf("evidence = %v", ev)
	}
	ec := reordered.EvidenceCard()
	if ec[0] != 3 || ec[1] != 2 {
		t.Errorf("evidenceCard = %v", ec)
	}

	vals := reordered.GetValues()
	// New parent config order: (Z=0,Y=0),(Z=0,Y=1),(Z=1,Y=0),(Z=1,Y=1),(Z=2,Y=0),(Z=2,Y=1)
	// Original: (Y=0,Z=0)=col0, (Y=0,Z=1)=col1, (Y=0,Z=2)=col2, (Y=1,Z=0)=col3, (Y=1,Z=1)=col4, (Y=1,Z=2)=col5
	// So new col0 = (Z=0,Y=0) = old col0: 0.1,0.9
	//    new col1 = (Z=0,Y=1) = old col3: 0.4,0.6
	//    new col2 = (Z=1,Y=0) = old col1: 0.2,0.8
	//    new col3 = (Z=1,Y=1) = old col4: 0.5,0.5
	//    new col4 = (Z=2,Y=0) = old col2: 0.3,0.7
	//    new col5 = (Z=2,Y=1) = old col5: 0.6,0.4
	expected := [][]float64{
		{0.1, 0.4, 0.2, 0.5, 0.3, 0.6},
		{0.9, 0.6, 0.8, 0.5, 0.7, 0.4},
	}
	for i := range expected {
		for j := range expected[i] {
			if !floatEq(vals[i][j], expected[i][j]) {
				t.Errorf("vals[%d][%d] = %f, want %f", i, j, vals[i][j], expected[i][j])
			}
		}
	}
}

func TestTabularCPD_ReorderParents_Errors(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{{0.4, 0.9}, {0.6, 0.1}},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}

	// Wrong length.
	_, err = cpd.ReorderParents([]string{"Y", "Z"})
	if err == nil {
		t.Error("expected error for wrong length")
	}

	// Unknown variable.
	_, err = cpd.ReorderParents([]string{"Z"})
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

// ---------------------------------------------------------------------------
// String
// ---------------------------------------------------------------------------

func TestTabularCPD_String(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.4, 0.9},
			{0.6, 0.1},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	s := cpd.String()
	if !strings.Contains(s, "TabularCPD(X)") {
		t.Errorf("String() missing header: %s", s)
	}
	if !strings.Contains(s, "X=0") || !strings.Contains(s, "X=1") {
		t.Errorf("String() missing variable states: %s", s)
	}
	if !strings.Contains(s, "Y") {
		t.Errorf("String() missing evidence name: %s", s)
	}
}

func TestTabularCPD_String_NoEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("A", 2,
		[][]float64{{0.3}, {0.7}},
		nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	s := cpd.String()
	if !strings.Contains(s, "TabularCPD(A)") {
		t.Errorf("String() = %s", s)
	}
}

// ---------------------------------------------------------------------------
// ToCSV
// ---------------------------------------------------------------------------

func TestTabularCPD_ToCSV(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.4, 0.9},
			{0.6, 0.1},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}

	tmpFile := t.TempDir() + "/cpd_test.csv"
	if err := cpd.ToCSV(tmpFile); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatal(err)
	}
	content := string(data)
	if !strings.Contains(content, "Y") {
		t.Errorf("CSV missing evidence header: %s", content)
	}
	if !strings.Contains(content, "X=0") || !strings.Contains(content, "X=1") {
		t.Errorf("CSV missing variable labels: %s", content)
	}
	if !strings.Contains(content, "0.4") || !strings.Contains(content, "0.9") {
		t.Errorf("CSV missing values: %s", content)
	}
}

// ---------------------------------------------------------------------------
// GetRandom
// ---------------------------------------------------------------------------

func TestGetRandom(t *testing.T) {
	cpd, err := GetRandom("X", 3, []string{"Y"}, []int{2}, 42)
	if err != nil {
		t.Fatal(err)
	}
	if cpd.Variable() != "X" {
		t.Errorf("Variable() = %q", cpd.Variable())
	}
	if cpd.VariableCard() != 3 {
		t.Errorf("VariableCard() = %d", cpd.VariableCard())
	}
	if err := cpd.Validate(); err != nil {
		t.Errorf("random CPD failed validation: %v", err)
	}
}

func TestGetRandom_NoEvidence(t *testing.T) {
	cpd, err := GetRandom("X", 2, nil, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := cpd.Validate(); err != nil {
		t.Errorf("validation failed: %v", err)
	}
}

func TestGetRandom_Errors(t *testing.T) {
	_, err := GetRandom("X", 0, nil, nil, 1)
	if err == nil {
		t.Error("expected error for zero variableCard")
	}
	_, err = GetRandom("X", 2, []string{"Y"}, nil, 1)
	if err == nil {
		t.Error("expected error for mismatched evidence/evidenceCard")
	}
}

// ---------------------------------------------------------------------------
// GetUniform
// ---------------------------------------------------------------------------

func TestGetUniform(t *testing.T) {
	cpd, err := GetUniform("X", 4, []string{"Y"}, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	if err := cpd.Validate(); err != nil {
		t.Errorf("uniform CPD failed validation: %v", err)
	}
	vals := cpd.GetValues()
	expected := 0.25
	for i := range vals {
		for j := range vals[i] {
			if !floatEq(vals[i][j], expected) {
				t.Errorf("vals[%d][%d] = %f, want %f", i, j, vals[i][j], expected)
			}
		}
	}
}

func TestGetUniform_NoEvidence(t *testing.T) {
	cpd, err := GetUniform("X", 2, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	vals := cpd.GetValues()
	if !floatEq(vals[0][0], 0.5) || !floatEq(vals[1][0], 0.5) {
		t.Errorf("unexpected values: %v", vals)
	}
}

func TestGetUniform_Errors(t *testing.T) {
	_, err := GetUniform("X", 0, nil, nil)
	if err == nil {
		t.Error("expected error for zero variableCard")
	}
	_, err = GetUniform("X", 2, []string{"Y"}, nil)
	if err == nil {
		t.Error("expected error for mismatched evidence/evidenceCard")
	}
}

// ---------------------------------------------------------------------------
// Determinism: GetRandom with same seed produces same results
// ---------------------------------------------------------------------------

func TestGetRandom_Deterministic(t *testing.T) {
	cpd1, _ := GetRandom("X", 3, []string{"Y", "Z"}, []int{2, 2}, 99)
	cpd2, _ := GetRandom("X", 3, []string{"Y", "Z"}, []int{2, 2}, 99)
	v1 := cpd1.GetValues()
	v2 := cpd2.GetValues()
	for i := range v1 {
		for j := range v1[i] {
			if v1[i][j] != v2[i][j] {
				t.Errorf("same seed produced different values at [%d][%d]: %f vs %f",
					i, j, v1[i][j], v2[i][j])
			}
		}
	}
}

// ---------------------------------------------------------------------------
// ToDataFrame
// ---------------------------------------------------------------------------

func TestTabularCPD_ToDataFrame_NoEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("A", 2,
		[][]float64{{0.3}, {0.7}},
		nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	df := cpd.ToDataFrame()
	if df == nil {
		t.Fatal("ToDataFrame returned nil")
	}
	// Should have 1 column and 2 rows.
	cols := df.Columns()
	if len(cols) != 1 {
		t.Errorf("expected 1 column, got %d", len(cols))
	}
	if df.Len() != 2 {
		t.Errorf("expected 2 rows, got %d", df.Len())
	}
}

func TestTabularCPD_ToDataFrame_WithEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.4, 0.9},
			{0.6, 0.1},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	df := cpd.ToDataFrame()
	if df == nil {
		t.Fatal("ToDataFrame returned nil")
	}
	// Should have 2 columns (Y=0, Y=1) and 2 rows.
	cols := df.Columns()
	if len(cols) != 2 {
		t.Errorf("expected 2 columns, got %d: %v", len(cols), cols)
	}
	if df.Len() != 2 {
		t.Errorf("expected 2 rows, got %d", df.Len())
	}
}

func TestTabularCPD_ToDataFrame_MultiEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.1, 0.2, 0.3, 0.4},
			{0.9, 0.8, 0.7, 0.6},
		},
		[]string{"Y", "Z"}, []int{2, 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	df := cpd.ToDataFrame()
	cols := df.Columns()
	if len(cols) != 4 {
		t.Errorf("expected 4 columns, got %d", len(cols))
	}
}

// ---------------------------------------------------------------------------
// Repr
// ---------------------------------------------------------------------------

func TestTabularCPD_Repr_NoEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("A", 2,
		[][]float64{{0.3}, {0.7}},
		nil, nil,
	)
	if err != nil {
		t.Fatal(err)
	}
	r := cpd.Repr()
	if !strings.Contains(r, "TabularCPD(variable=\"A\"") {
		t.Errorf("Repr missing header: %s", r)
	}
	if !strings.Contains(r, "variableCard=2") {
		t.Errorf("Repr missing cardinality: %s", r)
	}
}

func TestTabularCPD_Repr_WithEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.4, 0.9},
			{0.6, 0.1},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	r := cpd.Repr()
	if !strings.Contains(r, "evidence=") {
		t.Errorf("Repr missing evidence: %s", r)
	}
	if !strings.Contains(r, "evidenceCard=") {
		t.Errorf("Repr missing evidenceCard: %s", r)
	}
	// Should also contain the table from String()
	if !strings.Contains(r, "X=0") {
		t.Errorf("Repr missing table: %s", r)
	}
}

// ---------------------------------------------------------------------------
// Marginalize then Normalize round-trip
// ---------------------------------------------------------------------------

func TestTabularCPD_Marginalize_AllEvidence(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2,
		[][]float64{
			{0.4, 0.9},
			{0.6, 0.1},
		},
		[]string{"Y"}, []int{2},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := cpd.Marginalize([]string{"Y"})
	if err != nil {
		t.Fatal(err)
	}
	vals := result.GetValues()
	// X=0: 0.4+0.9=1.3, X=1: 0.6+0.1=0.7
	if !floatEq(vals[0][0], 1.3) {
		t.Errorf("vals[0][0] = %f, want 1.3", vals[0][0])
	}
	if !floatEq(vals[1][0], 0.7) {
		t.Errorf("vals[1][0] = %f, want 0.7", vals[1][0])
	}

	// Now normalize to get a proper marginal.
	norm := result.Normalize()
	if math.Abs(norm.GetValues()[0][0]-0.65) > 1e-9 {
		t.Errorf("normalized vals[0][0] = %f, want 0.65", norm.GetValues()[0][0])
	}
}
