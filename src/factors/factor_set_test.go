//go:build unit

package factors

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// FactorSet — NewFactorSet
// ---------------------------------------------------------------------------

func TestNewFactorSet_Empty(t *testing.T) {
	fs := NewFactorSet()
	if fs.Len() != 0 {
		t.Errorf("Len() = %d, want 0", fs.Len())
	}
}

func TestNewFactorSet_WithFactors(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.4, 0.6})
	fs := NewFactorSet(f1, f2)
	if fs.Len() != 2 {
		t.Errorf("Len() = %d, want 2", fs.Len())
	}
}

func TestNewFactorSet_NilFactorsIgnored(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs := NewFactorSet(f1, nil, nil)
	if fs.Len() != 1 {
		t.Errorf("Len() = %d, want 1", fs.Len())
	}
}

// ---------------------------------------------------------------------------
// FactorSet — Add / Remove / Contains
// ---------------------------------------------------------------------------

func TestFactorSet_Add(t *testing.T) {
	fs := NewFactorSet()
	f, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs.Add(f)
	if fs.Len() != 1 {
		t.Errorf("Len() = %d, want 1", fs.Len())
	}
}

func TestFactorSet_AddNil(t *testing.T) {
	fs := NewFactorSet()
	fs.Add(nil)
	if fs.Len() != 0 {
		t.Errorf("Len() = %d, want 0 after adding nil", fs.Len())
	}
}

func TestFactorSet_Contains(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.4, 0.6})
	fs := NewFactorSet(f1)
	if !fs.Contains(f1) {
		t.Error("Contains(f1) = false, want true")
	}
	if fs.Contains(f2) {
		t.Error("Contains(f2) = true, want false")
	}
}

func TestFactorSet_Remove(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.4, 0.6})
	fs := NewFactorSet(f1, f2)

	ok := fs.Remove(f1)
	if !ok {
		t.Error("Remove(f1) returned false")
	}
	if fs.Len() != 1 {
		t.Errorf("Len() = %d, want 1 after removal", fs.Len())
	}
	if fs.Contains(f1) {
		t.Error("Contains(f1) = true after removal")
	}
}

func TestFactorSet_Remove_NotFound(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.4, 0.6})
	fs := NewFactorSet(f1)

	ok := fs.Remove(f2)
	if ok {
		t.Error("Remove(f2) returned true for non-member")
	}
	if fs.Len() != 1 {
		t.Errorf("Len() = %d, want 1", fs.Len())
	}
}

// ---------------------------------------------------------------------------
// FactorSet — Product
// ---------------------------------------------------------------------------

func TestFactorSet_Product_Single(t *testing.T) {
	f, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fs := NewFactorSet(f)
	prod, err := fs.Product()
	if err != nil {
		t.Fatal(err)
	}
	data := prod.Values().Data()
	if !floatEq(data[0], 0.3) || !floatEq(data[1], 0.7) {
		t.Errorf("Product values = %v, want [0.3, 0.7]", data)
	}
}

func TestFactorSet_Product_Multiple(t *testing.T) {
	// f1(A): [0.3, 0.7], f2(A,B): [0.1, 0.2, 0.3, 0.4]
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	fs := NewFactorSet(f1, f2)

	prod, err := fs.Product()
	if err != nil {
		t.Fatal(err)
	}

	// Product should have variables [A, B], card [2, 2].
	vars := prod.Variables()
	if len(vars) != 2 || vars[0] != "A" || vars[1] != "B" {
		t.Errorf("Variables() = %v, want [A B]", vars)
	}

	// Values: f1(A=0)*f2(A=0,B=0)=0.3*0.1=0.03, etc.
	expected := map[string]float64{
		"00": 0.3 * 0.1,
		"01": 0.3 * 0.2,
		"10": 0.7 * 0.3,
		"11": 0.7 * 0.4,
	}
	for key, want := range expected {
		a := int(key[0] - '0')
		b := int(key[1] - '0')
		got := prod.GetValue(map[string]int{"A": a, "B": b})
		if !floatEq(got, want) {
			t.Errorf("Product(A=%d,B=%d) = %f, want %f", a, b, got, want)
		}
	}
}

func TestFactorSet_Product_Empty(t *testing.T) {
	fs := NewFactorSet()
	_, err := fs.Product()
	if err == nil {
		t.Error("expected error for empty FactorSet product")
	}
}

// ---------------------------------------------------------------------------
// FactorSet — GetFactorsOf
// ---------------------------------------------------------------------------

func TestFactorSet_GetFactorsOf(t *testing.T) {
	f1, _ := NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
	f2, _ := NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.5, 0.5, 0.5, 0.5})
	f3, _ := NewDiscreteFactor([]string{"D"}, []int{2}, []float64{0.6, 0.4})
	fs := NewFactorSet(f1, f2, f3)

	bFactors := fs.GetFactorsOf("B")
	if len(bFactors) != 2 {
		t.Errorf("GetFactorsOf(B) returned %d factors, want 2", len(bFactors))
	}

	aFactors := fs.GetFactorsOf("A")
	if len(aFactors) != 1 {
		t.Errorf("GetFactorsOf(A) returned %d factors, want 1", len(aFactors))
	}

	dFactors := fs.GetFactorsOf("D")
	if len(dFactors) != 1 {
		t.Errorf("GetFactorsOf(D) returned %d factors, want 1", len(dFactors))
	}

	xFactors := fs.GetFactorsOf("X")
	if len(xFactors) != 0 {
		t.Errorf("GetFactorsOf(X) returned %d factors, want 0", len(xFactors))
	}
}

// ---------------------------------------------------------------------------
// FactorDict — basic operations
// ---------------------------------------------------------------------------

func TestNewFactorDict_Empty(t *testing.T) {
	fd := NewFactorDict()
	if fd.Len() != 0 {
		t.Errorf("Len() = %d, want 0", fd.Len())
	}
}

func TestFactorDict_SetGet(t *testing.T) {
	fd := NewFactorDict()
	f, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	fd.Set("A", f)

	got := fd.Get("A")
	if got != f {
		t.Error("Get(A) did not return the same factor")
	}
	if fd.Len() != 1 {
		t.Errorf("Len() = %d, want 1", fd.Len())
	}
}

func TestFactorDict_GetMissing(t *testing.T) {
	fd := NewFactorDict()
	if fd.Get("missing") != nil {
		t.Error("Get(missing) should return nil")
	}
}

func TestFactorDict_Overwrite(t *testing.T) {
	fd := NewFactorDict()
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.5, 0.5})
	fd.Set("A", f1)
	fd.Set("A", f2)

	if fd.Len() != 1 {
		t.Errorf("Len() = %d, want 1 after overwrite", fd.Len())
	}
	if fd.Get("A") != f2 {
		t.Error("Get(A) should return f2 after overwrite")
	}
}

func TestFactorDict_Keys(t *testing.T) {
	fd := NewFactorDict()
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{2}, []float64{0.4, 0.6})
	f3, _ := NewDiscreteFactor([]string{"C"}, []int{2}, []float64{0.5, 0.5})
	fd.Set("C", f3)
	fd.Set("A", f1)
	fd.Set("B", f2)

	keys := fd.Keys()
	if len(keys) != 3 {
		t.Fatalf("Keys() has length %d, want 3", len(keys))
	}
	// Keys should be sorted.
	if keys[0] != "A" || keys[1] != "B" || keys[2] != "C" {
		t.Errorf("Keys() = %v, want [A B C]", keys)
	}
}

func TestFactorDict_MultipleEntries(t *testing.T) {
	fd := NewFactorDict()
	f1, _ := NewDiscreteFactor([]string{"A"}, []int{2}, []float64{0.3, 0.7})
	f2, _ := NewDiscreteFactor([]string{"B"}, []int{3}, []float64{0.2, 0.3, 0.5})
	fd.Set("alpha", f1)
	fd.Set("beta", f2)

	if fd.Len() != 2 {
		t.Errorf("Len() = %d, want 2", fd.Len())
	}
	if fd.Get("alpha") != f1 {
		t.Error("Get(alpha) != f1")
	}
	if fd.Get("beta") != f2 {
		t.Error("Get(beta) != f2")
	}
}

// Suppress unused import warning for math.
var _ = math.Abs
