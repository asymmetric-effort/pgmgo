//go:build unit

package factors

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
)

// ---------------------------------------------------------------------------
// Parity methods: DiscreteFactor
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Equals_NilOther(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	if f.Equals(nil, 0.01) {
		t.Error("expected false for nil other")
	}
}

func TestDiscreteFactor_Equals_DiffVarCount(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	if f1.Equals(f2, 0.01) {
		t.Error("expected false for different variable counts")
	}
}

func TestDiscreteFactor_Equals_DiffVarNames(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"B"}, []int{2}, []float64{0.3, 0.7})
	if f1.Equals(f2, 0.01) {
		t.Error("expected false for different variable names")
	}
}

func TestDiscreteFactor_Equals_DiffCard(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A"}, []int{3}, []float64{0.2, 0.3, 0.5})
	if f1.Equals(f2, 0.01) {
		t.Error("expected false for different cardinalities")
	}
}

func TestDiscreteFactor_Equals_DiffValues(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.5, 0.5})
	if f1.Equals(f2, 0.01) {
		t.Error("expected false for different values beyond tolerance")
	}
}

func TestDiscreteFactor_Equals_True(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	if !f1.Equals(f2, 0.01) {
		t.Error("expected true for equal factors")
	}
}

func TestDiscreteFactor_ToDataFrame(t *testing.T) {
	f := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	df := f.ToDataFrame()
	if df == nil {
		t.Fatal("expected non-nil DataFrame")
	}
}

func TestDiscreteFactor_Divide(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.6, 0.4})
	f2 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.2})
	result, err := f1.Divide(f2)
	if err != nil {
		t.Fatalf("Divide failed: %v", err)
	}
	data := result.Values().Data()
	if math.Abs(data[0]-2.0) > 1e-9 {
		t.Errorf("expected 2.0, got %f", data[0])
	}
}

func TestDiscreteFactor_Product(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"B"}, []int{2}, []float64{0.4, 0.6})
	result, err := f1.Product(f2)
	if err != nil {
		t.Fatalf("Product failed: %v", err)
	}
	if len(result.Variables()) != 2 {
		t.Errorf("expected 2 variables, got %d", len(result.Variables()))
	}
}

func TestDiscreteFactor_Scope(t *testing.T) {
	f := mustFactor(t, []string{"A", "B"}, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	scope := f.Scope()
	if len(scope) != 2 || scope[0] != "A" || scope[1] != "B" {
		t.Errorf("expected [A B], got %v", scope)
	}
}

func TestDiscreteFactor_StateNames(t *testing.T) {
	f := mustFactor(t, []string{"A", "B"}, []int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	sn := f.StateNames()
	if len(sn["A"]) != 2 || sn["A"][0] != "0" {
		t.Errorf("expected A states [0 1], got %v", sn["A"])
	}
	if len(sn["B"]) != 3 {
		t.Errorf("expected B states [0 1 2], got %v", sn["B"])
	}
}

// ---------------------------------------------------------------------------
// Parity methods: TabularCPD
// ---------------------------------------------------------------------------

func TestTabularCPD_StateNames(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	sn := cpd.StateNames()
	if len(sn["X"]) != 2 || sn["X"][0] != "0" {
		t.Errorf("expected X states [0 1], got %v", sn["X"])
	}
	if len(sn["A"]) != 2 {
		t.Errorf("expected A states [0 1], got %v", sn["A"])
	}
}

func TestTabularCPD_Scope(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	scope := cpd.Scope()
	if len(scope) != 2 || scope[0] != "X" || scope[1] != "A" {
		t.Errorf("expected [X A], got %v", scope)
	}
}

// ---------------------------------------------------------------------------
// Parity methods: FactorSet
// ---------------------------------------------------------------------------

func TestFactorSet_Factors(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"B"}, []int{2}, []float64{0.4, 0.6})
	fs := NewFactorSet(f1, f2)
	factors := fs.Factors()
	if len(factors) != 2 {
		t.Errorf("expected 2 factors, got %d", len(factors))
	}
}

func TestFactorSet_Marginalize(t *testing.T) {
	f := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	fs := NewFactorSet(f)
	result, err := fs.Marginalize("B")
	if err != nil {
		t.Fatalf("Marginalize failed: %v", err)
	}
	if result.Len() != 1 {
		t.Errorf("expected 1 factor, got %d", result.Len())
	}
}

func TestFactorSet_Marginalize_OnlyVar(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs := NewFactorSet(f)
	result, err := fs.Marginalize("A")
	if err != nil {
		t.Fatalf("Marginalize failed: %v", err)
	}
	// Factor with only variable A should be skipped
	if result.Len() != 0 {
		t.Errorf("expected 0 factors, got %d", result.Len())
	}
}

func TestFactorSet_Marginalize_NotPresent(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs := NewFactorSet(f)
	result, err := fs.Marginalize("Z")
	if err != nil {
		t.Fatalf("Marginalize failed: %v", err)
	}
	if result.Len() != 1 {
		t.Errorf("expected 1 factor (copy), got %d", result.Len())
	}
}

func TestFactorSet_Union(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"B"}, []int{2}, []float64{0.4, 0.6})
	fs1 := NewFactorSet(f1)
	fs2 := NewFactorSet(f2)
	result := fs1.Union(fs2)
	if result.Len() != 2 {
		t.Errorf("expected 2 factors, got %d", result.Len())
	}
}

func TestFactorSet_Union_NilOther(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs1 := NewFactorSet(f1)
	result := fs1.Union(nil)
	if result.Len() != 1 {
		t.Errorf("expected 1 factor, got %d", result.Len())
	}
}

func TestFactorSet_String(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs := NewFactorSet(f)
	s := fs.String()
	if !strings.Contains(s, "FactorSet(1 factors)") {
		t.Errorf("expected FactorSet header, got %q", s)
	}
}

// ---------------------------------------------------------------------------
// Parity methods: FactorDict
// ---------------------------------------------------------------------------

func TestFactorDict_HasDeleteValues(t *testing.T) {
	fd := NewFactorDict()
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	fd.Set("A", f)
	if !fd.Has("A") {
		t.Error("expected Has=true")
	}
	if fd.Has("Z") {
		t.Error("expected Has=false for missing key")
	}
	vals := fd.Values()
	if len(vals) != 1 {
		t.Errorf("expected 1 value, got %d", len(vals))
	}
	fd.Delete("A")
	if fd.Has("A") {
		t.Error("expected Has=false after Delete")
	}
}

// ---------------------------------------------------------------------------
// FactorProduct error paths
// ---------------------------------------------------------------------------

func TestFactorProduct_Empty_Boost(t *testing.T) {
	_, err := FactorProduct()
	if err == nil {
		t.Error("expected error for empty FactorProduct")
	}
}

func TestFactorProduct_Single(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	result, err := FactorProduct(f)
	if err != nil {
		t.Fatalf("FactorProduct(single) failed: %v", err)
	}
	if !result.Equals(f, 1e-9) {
		t.Error("expected copy of single factor")
	}
}

func TestFactorProduct_MismatchedCard(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A"}, []int{3}, []float64{0.2, 0.3, 0.5})
	_, err := FactorProduct(f1, f2)
	if err == nil {
		t.Error("expected error for mismatched cardinalities")
	}
}

// ---------------------------------------------------------------------------
// FactorDivide error paths
// ---------------------------------------------------------------------------

func TestFactorDivide_Nil(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	_, err := FactorDivide(f, nil)
	if err == nil {
		t.Error("expected error for nil f2")
	}
	_, err = FactorDivide(nil, f)
	if err == nil {
		t.Error("expected error for nil f1")
	}
}

func TestFactorDivide_VarNotInF1(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"B"}, []int{2}, []float64{0.5, 0.5})
	_, err := FactorDivide(f1, f2)
	if err == nil {
		t.Error("expected error for variable in f2 not in f1")
	}
}

func TestFactorDivide_CardMismatch(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A"}, []int{3}, []float64{0.2, 0.3, 0.5})
	_, err := FactorDivide(f1, f2)
	if err == nil {
		t.Error("expected error for cardinality mismatch")
	}
}

func TestFactorDivide_DivisionByZero_Boost(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.6, 0.4})
	f2 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.0, 0.2})
	result, err := FactorDivide(f1, f2)
	if err != nil {
		t.Fatalf("FactorDivide failed: %v", err)
	}
	data := result.Values().Data()
	if data[0] != 0.0 {
		t.Errorf("expected 0 for division by zero, got %f", data[0])
	}
}

// ---------------------------------------------------------------------------
// FactorSumProduct edge cases
// ---------------------------------------------------------------------------

func TestFactorSumProduct_VarNotInAny(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	result, err := FactorSumProduct([]*DiscreteFactor{f}, []string{"Z"})
	if err != nil {
		t.Fatalf("FactorSumProduct failed: %v", err)
	}
	// Z not in any factor, so result should be same as input
	data := result.Values().Data()
	if math.Abs(data[0]-0.3) > 1e-9 {
		t.Errorf("expected 0.3, got %f", data[0])
	}
}

func TestFactorSumProduct_Basic(t *testing.T) {
	f1 := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	f2 := mustFactor(t, []string{"B", "C"}, []int{2, 2}, []float64{0.5, 0.5, 0.5, 0.5})
	result, err := FactorSumProduct([]*DiscreteFactor{f1, f2}, []string{"B"})
	if err != nil {
		t.Fatalf("FactorSumProduct failed: %v", err)
	}
	if len(result.Variables()) != 2 {
		t.Errorf("expected 2 variables, got %d", len(result.Variables()))
	}
}

// ---------------------------------------------------------------------------
// DiscreteFactor: Reduce edge cases
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Reduce_OutOfRange(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	_, err := f.Reduce(map[string]int{"A": 5})
	if err == nil {
		t.Error("expected error for out-of-range evidence value")
	}
}

func TestDiscreteFactor_Reduce_AllVars(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	result, err := f.Reduce(map[string]int{"A": 1})
	if err != nil {
		t.Fatalf("Reduce failed: %v", err)
	}
	data := result.Values().Data()
	if math.Abs(data[0]-0.7) > 1e-9 {
		t.Errorf("expected 0.7, got %f", data[0])
	}
}

// ---------------------------------------------------------------------------
// DiscreteFactor: Sum error paths
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Sum_NilOther(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	_, err := f.Sum(nil)
	if err == nil {
		t.Error("expected error for nil other")
	}
}

func TestDiscreteFactor_Sum_DiffVarCount(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	_, err := f1.Sum(f2)
	if err == nil {
		t.Error("expected error for different variable counts")
	}
}

func TestDiscreteFactor_Sum_DiffVarNames(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"B"}, []int{2}, []float64{0.4, 0.6})
	_, err := f1.Sum(f2)
	if err == nil {
		t.Error("expected error for different variable names")
	}
}

func TestDiscreteFactor_Sum_DiffCard(t *testing.T) {
	f1 := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2 := mustFactor(t, []string{"A"}, []int{3}, []float64{0.2, 0.3, 0.5})
	_, err := f1.Sum(f2)
	if err == nil {
		t.Error("expected error for different cardinalities")
	}
}

// ---------------------------------------------------------------------------
// DiscreteFactor: Sample error paths
// ---------------------------------------------------------------------------

func TestDiscreteFactor_Sample_NegativeValues(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{-0.3, 0.7})
	_, err := f.Sample(1, 42)
	if err == nil {
		t.Error("expected error for negative values")
	}
}

func TestDiscreteFactor_Sample_AllZero(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.0, 0.0})
	_, err := f.Sample(1, 42)
	if err == nil {
		t.Error("expected error for all-zero values")
	}
}

func TestDiscreteFactor_Sample_BadN(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	_, err := f.Sample(0, 42)
	if err == nil {
		t.Error("expected error for n <= 0")
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: Normalize with zero column
// ---------------------------------------------------------------------------

func TestTabularCPD_Normalize_ZeroColumn_Boost(t *testing.T) {
	cpd, err := NewTabularCPD("X", 2, [][]float64{{0.0, 0.3}, {0.0, 0.7}}, []string{"A"}, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	result := cpd.Normalize()
	vals := result.GetValues()
	// Column 0 sums to 0, should stay 0
	if vals[0][0] != 0 || vals[1][0] != 0 {
		t.Errorf("expected zero column to stay zero, got %v %v", vals[0][0], vals[1][0])
	}
	// Column 1 should be normalized
	if math.Abs(vals[0][1]-0.3) > 1e-9 || math.Abs(vals[1][1]-0.7) > 1e-9 {
		t.Errorf("expected normalized values, got %v %v", vals[0][1], vals[1][1])
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: Marginalize child variable error
// ---------------------------------------------------------------------------

func TestTabularCPD_Marginalize_ChildVar(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	_, err := cpd.Marginalize([]string{"X"})
	if err == nil {
		t.Error("expected error for marginalizing child variable")
	}
}

func TestTabularCPD_Marginalize_Empty(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	result, err := cpd.Marginalize(nil)
	if err != nil {
		t.Fatalf("Marginalize(nil) failed: %v", err)
	}
	if result.Variable() != "X" {
		t.Errorf("expected variable X, got %s", result.Variable())
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: Reduce edge cases
// ---------------------------------------------------------------------------

func TestTabularCPD_Reduce_ChildVar(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	_, err := cpd.Reduce(map[string]int{"X": 0})
	if err == nil {
		t.Error("expected error for reducing child variable")
	}
}

func TestTabularCPD_Reduce_UnknownEvidence(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	_, err := cpd.Reduce(map[string]int{"Z": 0})
	if err == nil {
		t.Error("expected error for unknown evidence variable")
	}
}

func TestTabularCPD_Reduce_Empty(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	result, err := cpd.Reduce(nil)
	if err != nil {
		t.Fatalf("Reduce(nil) failed: %v", err)
	}
	if result.Variable() != "X" {
		t.Errorf("expected variable X, got %s", result.Variable())
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: ReorderParents error paths
// ---------------------------------------------------------------------------

func TestTabularCPD_ReorderParents_WrongLength(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.1, 0.2, 0.3, 0.4}, {0.9, 0.8, 0.7, 0.6}},
		[]string{"A", "B"}, []int{2, 2})
	_, err := cpd.ReorderParents([]string{"A"})
	if err == nil {
		t.Error("expected error for wrong newOrder length")
	}
}

func TestTabularCPD_ReorderParents_UnknownVar(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.1, 0.2, 0.3, 0.4}, {0.9, 0.8, 0.7, 0.6}},
		[]string{"A", "B"}, []int{2, 2})
	_, err := cpd.ReorderParents([]string{"A", "Z"})
	if err == nil {
		t.Error("expected error for unknown variable in newOrder")
	}
}

func TestTabularCPD_ReorderParents_Duplicate(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.1, 0.2, 0.3, 0.4}, {0.9, 0.8, 0.7, 0.6}},
		[]string{"A", "B"}, []int{2, 2})
	_, err := cpd.ReorderParents([]string{"A", "A"})
	if err == nil {
		t.Error("expected error for duplicate variable in newOrder")
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: String with parents (covers evidence strides path)
// ---------------------------------------------------------------------------

func TestTabularCPD_String_WithParents(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.1, 0.2, 0.3, 0.4}, {0.9, 0.8, 0.7, 0.6}},
		[]string{"A", "B"}, []int{2, 2})
	s := cpd.String()
	if !strings.Contains(s, "TabularCPD(X)") {
		t.Errorf("expected TabularCPD header, got %q", s)
	}
	if !strings.Contains(s, "A") {
		t.Errorf("expected parent A in output")
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: ToCSV
// ---------------------------------------------------------------------------

func TestTabularCPD_ToCSV_Boost(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.1, 0.2}, {0.9, 0.8}},
		[]string{"A"}, []int{2})
	tmpFile := "/tmp/test_cpd.csv"
	defer os.Remove(tmpFile)
	err := cpd.ToCSV(tmpFile)
	if err != nil {
		t.Fatalf("ToCSV failed: %v", err)
	}
}

func TestTabularCPD_ToCSV_NoParents(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3}, {0.7}}, nil, nil)
	tmpFile := "/tmp/test_cpd_noparents.csv"
	defer os.Remove(tmpFile)
	err := cpd.ToCSV(tmpFile)
	if err != nil {
		t.Fatalf("ToCSV failed: %v", err)
	}
}

func TestTabularCPD_ToCSV_BadPath(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3}, {0.7}}, nil, nil)
	err := cpd.ToCSV("/nonexistent/dir/file.csv")
	if err == nil {
		t.Error("expected error for bad file path")
	}
}

// ---------------------------------------------------------------------------
// GetRandom, GetUniform with evidence
// ---------------------------------------------------------------------------

func TestGetRandom_WithEvidence(t *testing.T) {
	cpd, err := GetRandom("X", 2, []string{"A"}, []int{2}, 42)
	if err != nil {
		t.Fatalf("GetRandom failed: %v", err)
	}
	if cpd.Variable() != "X" {
		t.Errorf("expected variable X, got %s", cpd.Variable())
	}
}

func TestGetUniform_WithEvidence(t *testing.T) {
	cpd, err := GetUniform("X", 3, []string{"A"}, []int{2})
	if err != nil {
		t.Fatalf("GetUniform failed: %v", err)
	}
	vals := cpd.GetValues()
	expected := 1.0 / 3.0
	for i := range vals {
		for j := range vals[i] {
			if math.Abs(vals[i][j]-expected) > 1e-9 {
				t.Errorf("expected %f, got %f", expected, vals[i][j])
			}
		}
	}
}

func TestGetUniform_BadEvidenceCard(t *testing.T) {
	_, err := GetUniform("X", 2, []string{"A"}, []int{0})
	if err == nil {
		t.Error("expected error for evidence cardinality <= 0")
	}
}

// ---------------------------------------------------------------------------
// NewTabularCPD: more error paths
// ---------------------------------------------------------------------------

func TestNewTabularCPD_WrongRowCount(t *testing.T) {
	_, err := NewTabularCPD("X", 2, [][]float64{{0.5}}, nil, nil)
	if err == nil {
		t.Error("expected error for wrong row count")
	}
}

func TestNewTabularCPD_WrongColCount(t *testing.T) {
	_, err := NewTabularCPD("X", 2, [][]float64{{0.3, 0.5}, {0.7}}, []string{"A"}, []int{2})
	if err == nil {
		t.Error("expected error for wrong column count")
	}
}

func TestNewTabularCPD_BadEvidenceCard(t *testing.T) {
	_, err := NewTabularCPD("X", 2, [][]float64{{0.5}, {0.5}}, []string{"A"}, []int{0})
	if err == nil {
		t.Error("expected error for evidence cardinality <= 0")
	}
}

// ---------------------------------------------------------------------------
// NoisyOR: ToTabularCPD with invalid validate
// ---------------------------------------------------------------------------

func TestNoisyOR_ToTabularCPD_InvalidValidate(t *testing.T) {
	n := &NoisyOR{
		variable:        "Y",
		variableCard:    3, // invalid
		parents:         []string{"X"},
		inhibitionProbs: []float64{0.5},
		leakProb:        0.1,
	}
	_, err := n.ToTabularCPD()
	if err == nil {
		t.Error("expected error for invalid NoisyOR in ToTabularCPD")
	}
}

func TestNoisyOR_ToTabularCPD_NoParents_Boost(t *testing.T) {
	n, err := NewNoisyOR("Y", 2, nil, nil, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatalf("ToTabularCPD failed: %v", err)
	}
	vals := cpd.GetValues()
	if math.Abs(vals[0][0]-0.9) > 1e-9 {
		t.Errorf("expected P(Y=0) = 0.9, got %f", vals[0][0])
	}
}

// ---------------------------------------------------------------------------
// FunctionalCPD: Sample actually called with RNG
// ---------------------------------------------------------------------------

func TestFunctionalCPD_Sample_WithRNG(t *testing.T) {
	cpd, err := NewFunctionalCPD("X", nil, func(parents map[string]float64) []float64 {
		return []float64{0.0, 0.0, 1.0} // always returns state 2
	})
	if err != nil {
		t.Fatal(err)
	}
	rng := numgo.NewRNG(42)
	idx := cpd.Sample(nil, rng)
	if idx != 2 {
		t.Errorf("expected state 2, got %d", idx)
	}
}

func TestFunctionalCPD_Sample_FallbackLastIndex(t *testing.T) {
	// Test edge case: cumulative exactly == u (floating point)
	cpd, err := NewFunctionalCPD("X", []string{"A"}, func(parents map[string]float64) []float64 {
		return []float64{1.0} // single state, always sampled
	})
	if err != nil {
		t.Fatal(err)
	}
	rng := numgo.NewRNG(42)
	idx := cpd.Sample(map[string]float64{"A": 1.0}, rng)
	if idx != 0 {
		t.Errorf("expected state 0, got %d", idx)
	}
}

// ---------------------------------------------------------------------------
// LinearGaussianCPD: String with no evidence (covers branch)
// ---------------------------------------------------------------------------

func TestLinearGaussianCPD_String_NoEvidence(t *testing.T) {
	cpd, err := NewLinearGaussianCPD("Y", 1.0, nil, 0.5, nil)
	if err != nil {
		t.Fatal(err)
	}
	s := cpd.String()
	if !strings.Contains(s, "LinearGaussianCPD(Y") {
		t.Errorf("expected header, got %q", s)
	}
	if strings.Contains(s, "betas") {
		t.Errorf("should not contain betas for no-evidence CPD")
	}
}

func TestLinearGaussianCPD_Validate_BadVariance(t *testing.T) {
	cpd := &LinearGaussianCPD{variable: "Y", variance: -1.0}
	if err := cpd.Validate(); err == nil {
		t.Error("expected error for negative variance")
	}
}

func TestLinearGaussianCPD_Validate_MismatchedBetas(t *testing.T) {
	cpd := &LinearGaussianCPD{
		variable: "Y",
		betas:    []float64{1.0, 2.0},
		evidence: []string{"X"},
		variance: 1.0,
	}
	if err := cpd.Validate(); err == nil {
		t.Error("expected error for mismatched betas/evidence")
	}
}

func TestGetRandomLinearGaussianCPD_EmptyVar(t *testing.T) {
	_, err := GetRandomLinearGaussianCPD("", nil, 42)
	if err == nil {
		t.Error("expected error for empty variable name")
	}
}

func TestGetRandomLinearGaussianCPD_Valid(t *testing.T) {
	cpd, err := GetRandomLinearGaussianCPD("Y", []string{"X"}, 42)
	if err != nil {
		t.Fatalf("GetRandomLinearGaussianCPD failed: %v", err)
	}
	if cpd.Variable() != "Y" {
		t.Errorf("expected variable Y, got %s", cpd.Variable())
	}
}

// ---------------------------------------------------------------------------
// JointProbabilityDistribution: MarginalDistribution edge cases
// ---------------------------------------------------------------------------

func TestJPD_MarginalDistribution_Empty(t *testing.T) {
	jpd := mustJPD(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	_, err := jpd.MarginalDistribution(nil)
	if err == nil {
		t.Error("expected error for empty variables")
	}
}

func TestJPD_MarginalDistribution_UnknownVar(t *testing.T) {
	jpd := mustJPD(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	_, err := jpd.MarginalDistribution([]string{"Z"})
	if err == nil {
		t.Error("expected error for unknown variable")
	}
}

func TestJPD_MarginalDistribution_AllVars(t *testing.T) {
	jpd := mustJPD(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	result, err := jpd.MarginalDistribution([]string{"A", "B"})
	if err != nil {
		t.Fatalf("MarginalDistribution failed: %v", err)
	}
	// Keeping all variables = copy
	if len(result.Variables()) != 2 {
		t.Errorf("expected 2 variables, got %d", len(result.Variables()))
	}
}

// ---------------------------------------------------------------------------
// JointProbabilityDistribution: ConditionalDistribution with no marginalize
// ---------------------------------------------------------------------------

func TestJPD_ConditionalDistribution_NoMarg(t *testing.T) {
	// All variables are either query or evidence, no marginalization needed
	jpd := mustJPD(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	result, err := jpd.ConditionalDistribution([]string{"A"}, map[string]int{"B": 0})
	if err != nil {
		t.Fatalf("ConditionalDistribution failed: %v", err)
	}
	data := result.Values().Data()
	if len(data) != 2 {
		t.Errorf("expected 2 values, got %d", len(data))
	}
}

// ---------------------------------------------------------------------------
// JointProbabilityDistribution: GetIndependencies with 3 vars
// ---------------------------------------------------------------------------

func TestJPD_GetIndependencies_ThreeVars(t *testing.T) {
	// Independent: A,B,C all independent
	jpd := mustJPD(t, []string{"A", "B", "C"}, []int{2, 2, 2},
		[]float64{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125})
	indeps := jpd.GetIndependencies(0.01)
	// Should find many independencies for uniform distribution
	if len(indeps) == 0 {
		t.Error("expected independencies for uniform distribution")
	}
}

// ---------------------------------------------------------------------------
// JointProbabilityDistribution: MinimalIMap
// ---------------------------------------------------------------------------

func TestJPD_MinimalIMap(t *testing.T) {
	// Independent distribution
	jpd := mustJPD(t, []string{"A", "B"}, []int{2, 2}, []float64{0.15, 0.15, 0.35, 0.35})
	edges := jpd.MinimalIMap([]string{"A", "B"}, 0.01)
	// Should have no edges since A and B are independent
	if len(edges) != 0 {
		t.Errorf("expected 0 edges, got %d: %v", len(edges), edges)
	}
}

func TestJPD_MinimalIMap_Dependent(t *testing.T) {
	jpd := mustJPD(t, []string{"A", "B"}, []int{2, 2}, []float64{0.5, 0.0, 0.0, 0.5})
	edges := jpd.MinimalIMap([]string{"A", "B"}, 0.01)
	if len(edges) != 1 {
		t.Errorf("expected 1 edge, got %d: %v", len(edges), edges)
	}
}

// ---------------------------------------------------------------------------
// JointProbabilityDistribution: jpdHasDirectedPath
// ---------------------------------------------------------------------------

func TestJpdHasDirectedPath_NotFound(t *testing.T) {
	edges := [][2]string{{"A", "B"}}
	if jpdHasDirectedPath(edges, "B", "A") {
		t.Error("expected no path from B to A")
	}
}

func TestJpdHasDirectedPath_Found(t *testing.T) {
	edges := [][2]string{{"A", "B"}, {"B", "C"}}
	if !jpdHasDirectedPath(edges, "A", "C") {
		t.Error("expected path from A to C")
	}
}

// ---------------------------------------------------------------------------
// JointProbabilityDistribution: ToFactor, Copy, Validate
// ---------------------------------------------------------------------------

func TestJPD_ToFactor(t *testing.T) {
	jpd := mustJPD(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	f := jpd.ToFactor()
	if len(f.Variables()) != 1 || f.Variables()[0] != "A" {
		t.Errorf("unexpected variables: %v", f.Variables())
	}
}

func TestJPD_Copy(t *testing.T) {
	jpd := mustJPD(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	cp := jpd.Copy()
	if !cp.DiscreteFactor.Equals(jpd.DiscreteFactor, 1e-9) {
		t.Error("copy should equal original")
	}
}

func TestJPD_Validate_NegativeProb(t *testing.T) {
	_, err := NewJointProbabilityDistribution([]string{"A"}, []int{2}, []float64{-0.3, 1.3})
	if err == nil {
		t.Error("expected error for negative probability")
	}
}

func TestJPD_Validate_BadSum(t *testing.T) {
	_, err := NewJointProbabilityDistribution([]string{"A"}, []int{2}, []float64{0.3, 0.3})
	if err == nil {
		t.Error("expected error for probabilities not summing to 1")
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: Repr
// ---------------------------------------------------------------------------

func TestTabularCPD_Repr(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	r := cpd.Repr()
	if !strings.Contains(r, "TabularCPD(variable=") {
		t.Errorf("expected Repr header, got %q", r)
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: ToDataFrame
// ---------------------------------------------------------------------------

func TestTabularCPD_ToDataFrame(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.3, 0.5}, {0.7, 0.5}}, []string{"A"}, []int{2})
	df := cpd.ToDataFrame()
	if df == nil {
		t.Fatal("expected non-nil DataFrame")
	}
}

func TestTabularCPD_ToDataFrame_NoParents(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3}, {0.7}}, nil, nil)
	df := cpd.ToDataFrame()
	if df == nil {
		t.Fatal("expected non-nil DataFrame")
	}
}

// ---------------------------------------------------------------------------
// IsValidCPD
// ---------------------------------------------------------------------------

func TestDiscreteFactor_IsValidCPD_NoVars(t *testing.T) {
	f, _ := NewDiscreteFactor(nil, nil, []float64{1.0})
	if f.IsValidCPD() {
		t.Error("expected false for factor with no variables")
	}
}

func TestDiscreteFactor_IsValidCPD_Valid(t *testing.T) {
	f := mustFactor(t, []string{"X", "A"}, []int{2, 2}, []float64{0.3, 0.5, 0.7, 0.5})
	if !f.IsValidCPD() {
		t.Error("expected valid CPD")
	}
}

func TestDiscreteFactor_IsValidCPD_Invalid(t *testing.T) {
	f := mustFactor(t, []string{"X", "A"}, []int{2, 2}, []float64{0.3, 0.5, 0.3, 0.5})
	if f.IsValidCPD() {
		t.Error("expected invalid CPD (columns don't sum to 1)")
	}
}

// ---------------------------------------------------------------------------
// NoisyOR: Copy
// ---------------------------------------------------------------------------

func TestNoisyOR_Copy_Boost(t *testing.T) {
	n, _ := NewNoisyOR("Y", 2, []string{"X"}, []float64{0.5}, 0.9)
	cp := n.Copy()
	if cp.Variable() != "Y" || cp.LeakProb() != 0.9 {
		t.Error("copy mismatch")
	}
}

// ---------------------------------------------------------------------------
// FunctionalCPD: Copy, Variable, Evidence, GetDistribution
// ---------------------------------------------------------------------------

func TestFunctionalCPD_Copy_Boost(t *testing.T) {
	cpd, _ := NewFunctionalCPD("X", []string{"A"}, func(p map[string]float64) []float64 {
		return []float64{0.5, 0.5}
	})
	cp := cpd.Copy()
	if cp.Variable() != "X" {
		t.Errorf("expected variable X, got %s", cp.Variable())
	}
	ev := cp.Evidence()
	if len(ev) != 1 || ev[0] != "A" {
		t.Errorf("expected evidence [A], got %v", ev)
	}
}

func TestFunctionalCPD_GetDistribution_Boost(t *testing.T) {
	cpd, _ := NewFunctionalCPD("X", nil, func(p map[string]float64) []float64 {
		return []float64{0.3, 0.7}
	})
	dist := cpd.GetDistribution(nil)
	if len(dist) != 2 || math.Abs(dist[0]-0.3) > 1e-9 {
		t.Errorf("unexpected distribution: %v", dist)
	}
}

// ---------------------------------------------------------------------------
// LinearGaussianCPD: Copy, accessors
// ---------------------------------------------------------------------------

func TestLinearGaussianCPD_Copy(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("Y", 1.0, []float64{2.0}, 0.5, []string{"X"})
	cp := cpd.Copy()
	if cp.Variable() != "Y" {
		t.Errorf("expected variable Y, got %s", cp.Variable())
	}
	if cp.Mean() != 1.0 {
		t.Errorf("expected mean 1.0, got %f", cp.Mean())
	}
	if cp.Variance() != 0.5 {
		t.Errorf("expected variance 0.5, got %f", cp.Variance())
	}
	betas := cp.Betas()
	if len(betas) != 1 || betas[0] != 2.0 {
		t.Errorf("expected betas [2.0], got %v", betas)
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: Validate
// ---------------------------------------------------------------------------

func TestTabularCPD_Validate_Valid(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3}, {0.7}}, nil, nil)
	if err := cpd.Validate(); err != nil {
		t.Errorf("expected valid CPD: %v", err)
	}
}

// ---------------------------------------------------------------------------
// FactorSumProduct: remaining edge cases
// ---------------------------------------------------------------------------

func TestFactorSumProduct_AllEliminated(t *testing.T) {
	f := mustFactor(t, []string{"A"}, []int{2}, []float64{0.3, 0.7})
	// Eliminating the only variable from a single factor causes error
	_, err := FactorSumProduct([]*DiscreteFactor{f}, []string{"A"})
	if err == nil {
		t.Error("expected error when eliminating all variables")
	}
}

func TestFactorSumProduct_MultipleFactors(t *testing.T) {
	f1 := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.5, 0.8, 0.1, 0.3})
	f2 := mustFactor(t, []string{"B", "C"}, []int{2, 2}, []float64{0.5, 0.5, 0.5, 0.5})
	f3 := mustFactor(t, []string{"C"}, []int{2}, []float64{0.4, 0.6})
	result, err := FactorSumProduct([]*DiscreteFactor{f1, f2, f3}, []string{"B", "C"})
	if err != nil {
		t.Fatalf("FactorSumProduct failed: %v", err)
	}
	if len(result.Variables()) != 1 {
		t.Errorf("expected 1 variable, got %d", len(result.Variables()))
	}
}

// ---------------------------------------------------------------------------
// ConditionalDistribution: with 3 variables (marginalizes middle variable)
// ---------------------------------------------------------------------------

func TestJPD_ConditionalDistribution_ThreeVars(t *testing.T) {
	jpd := mustJPD(t, []string{"A", "B", "C"}, []int{2, 2, 2},
		[]float64{0.1, 0.05, 0.15, 0.1, 0.05, 0.15, 0.1, 0.3})
	// Query A given C=0, B gets marginalized out
	result, err := jpd.ConditionalDistribution([]string{"A"}, map[string]int{"C": 0})
	if err != nil {
		t.Fatalf("ConditionalDistribution failed: %v", err)
	}
	data := result.Values().Data()
	if len(data) != 2 {
		t.Errorf("expected 2 values, got %d", len(data))
	}
}

// ---------------------------------------------------------------------------
// FactorSet Marginalize error path
// ---------------------------------------------------------------------------

func TestFactorSet_Marginalize_Error(t *testing.T) {
	// Factor that when marginalized would leave 0 variables
	f := mustFactor(t, []string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	// Create a factor set with a factor where marginalizing B works
	fs := NewFactorSet(f)
	result, err := fs.Marginalize("A")
	if err != nil {
		t.Fatalf("Marginalize failed: %v", err)
	}
	if result.Len() != 1 {
		t.Errorf("expected 1 factor, got %d", result.Len())
	}
}

// ---------------------------------------------------------------------------
// LinearGaussianCPD: String with multiple betas
// ---------------------------------------------------------------------------

func TestLinearGaussianCPD_String_MultipleBetas(t *testing.T) {
	cpd, err := NewLinearGaussianCPD("Y", 1.0, []float64{2.0, 3.0}, 0.5, []string{"X1", "X2"})
	if err != nil {
		t.Fatal(err)
	}
	s := cpd.String()
	if !strings.Contains(s, "2.0000") || !strings.Contains(s, "3.0000") {
		t.Errorf("expected both betas in string, got %q", s)
	}
}

// ---------------------------------------------------------------------------
// TabularCPD: ToCSV with multiple evidence variables
// ---------------------------------------------------------------------------

func TestTabularCPD_ToCSV_MultiEvidence(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2,
		[][]float64{{0.1, 0.2, 0.3, 0.4}, {0.9, 0.8, 0.7, 0.6}},
		[]string{"A", "B"}, []int{2, 2})
	tmpFile := "/tmp/test_cpd_multi_ev.csv"
	defer os.Remove(tmpFile)
	err := cpd.ToCSV(tmpFile)
	if err != nil {
		t.Fatalf("ToCSV failed: %v", err)
	}
}

func TestTabularCPD_Validate_Invalid(t *testing.T) {
	cpd, _ := NewTabularCPD("X", 2, [][]float64{{0.3}, {0.3}}, nil, nil)
	if err := cpd.Validate(); err == nil {
		t.Error("expected error for CPD column not summing to 1")
	}
}
