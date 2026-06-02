//go:build unit

package independencies

import (
	"testing"
)

// ---------------------------------------------------------------------------
// IndependenceAssertion tests
// ---------------------------------------------------------------------------

func TestNewIndependenceAssertion(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	if a == nil {
		t.Fatal("expected non-nil assertion")
	}
	assertSliceEqual(t, a.Event1(), []string{"X"})
	assertSliceEqual(t, a.Event2(), []string{"Y"})
	assertSliceEqual(t, a.Given(), []string{"Z"})
}

func TestNewIndependenceAssertionSortsInputs(t *testing.T) {
	a := NewIndependenceAssertion([]string{"B", "A"}, []string{"D", "C"}, []string{"F", "E"})
	assertSliceEqual(t, a.Event1(), []string{"A", "B"})
	assertSliceEqual(t, a.Event2(), []string{"C", "D"})
	assertSliceEqual(t, a.Given(), []string{"E", "F"})
}

func TestNewIndependenceAssertionEmptyGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	if a.Given() != nil {
		t.Error("expected nil given for nil input")
	}
	b := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{})
	if len(b.Given()) != 0 {
		t.Error("expected empty given")
	}
}

func TestIndependenceAssertionImmutability(t *testing.T) {
	orig := []string{"X", "Y"}
	a := NewIndependenceAssertion(orig, []string{"Z"}, nil)
	orig[0] = "MUTATED"
	if a.Event1()[0] == "MUTATED" {
		t.Error("internal state should not be affected by external mutation")
	}
	e := a.Event1()
	e[0] = "MUTATED"
	if a.Event1()[0] == "MUTATED" {
		t.Error("returned slice should be a copy")
	}
}

func TestEquals_Identical(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	if !a.Equals(b) {
		t.Error("identical assertions should be equal")
	}
}

func TestEquals_OrderIndependentEvents(t *testing.T) {
	a := NewIndependenceAssertion([]string{"B", "A"}, []string{"D", "C"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"A", "B"}, []string{"C", "D"}, []string{"Z"})
	if !a.Equals(b) {
		t.Error("assertions with reordered event sets should be equal")
	}
}

func TestEquals_SymmetryOfIndependence(t *testing.T) {
	// X ⊥ Y | Z should equal Y ⊥ X | Z
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"Y"}, []string{"X"}, []string{"Z"})
	if !a.Equals(b) {
		t.Error("X ⊥ Y | Z should equal Y ⊥ X | Z")
	}
}

func TestEquals_SymmetryMultiVars(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A", "B"}, []string{"C", "D"}, []string{"E"})
	b := NewIndependenceAssertion([]string{"C", "D"}, []string{"B", "A"}, []string{"E"})
	if !a.Equals(b) {
		t.Error("{A,B} ⊥ {C,D} | E should equal {C,D} ⊥ {A,B} | E")
	}
}

func TestEquals_DifferentGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"W"})
	if a.Equals(b) {
		t.Error("assertions with different given sets should not be equal")
	}
}

func TestEquals_DifferentEvents(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"X"}, []string{"W"}, []string{"Z"})
	if a.Equals(b) {
		t.Error("assertions with different event sets should not be equal")
	}
}

func TestEquals_NilOther(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	if a.Equals(nil) {
		t.Error("should not be equal to nil")
	}
}

func TestEquals_EmptyVsNilGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	b := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{})
	// Both represent unconditional independence; nil and empty should match
	if !a.Equals(b) {
		t.Error("nil and empty given should be equal (both unconditional)")
	}
}

func TestContains_ExactMatch(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	if !a.Contains(b) {
		t.Error("assertion should contain an equal assertion")
	}
}

func TestContains_SubsetEvents(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A", "B", "C"}, []string{"D", "E"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"A", "B"}, []string{"D"}, []string{"Z"})
	if !a.Contains(b) {
		t.Error("larger assertion should contain smaller subset assertion")
	}
}

func TestContains_SubsetSymmetric(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A", "B", "C"}, []string{"D", "E"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"D"}, []string{"A", "B"}, []string{"Z"})
	if !a.Contains(b) {
		t.Error("containment should work with symmetric orientation")
	}
}

func TestContains_DifferentGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A", "B"}, []string{"C"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"A"}, []string{"C"}, []string{"W"})
	if a.Contains(b) {
		t.Error("should not contain assertion with different given")
	}
}

func TestContains_NotSubset(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A"}, []string{"C"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"A", "B"}, []string{"C"}, []string{"Z"})
	if a.Contains(b) {
		t.Error("smaller assertion should not contain larger one")
	}
}

func TestContains_NilOther(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	if !a.Contains(nil) {
		t.Error("any assertion should contain nil")
	}
}

func TestString_Simple(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	expected := "X ⊥ Y | Z"
	if a.String() != expected {
		t.Errorf("expected %q, got %q", expected, a.String())
	}
}

func TestString_MultipleVars(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A", "B"}, []string{"C", "D"}, []string{"E", "F"})
	expected := "{A, B} ⊥ {C, D} | {E, F}"
	if a.String() != expected {
		t.Errorf("expected %q, got %q", expected, a.String())
	}
}

func TestString_NoGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	expected := "X ⊥ Y"
	if a.String() != expected {
		t.Errorf("expected %q, got %q", expected, a.String())
	}
}

func TestString_EmptyGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{})
	expected := "X ⊥ Y"
	if a.String() != expected {
		t.Errorf("expected %q, got %q", expected, a.String())
	}
}

func TestLatexString_Simple(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	expected := "X \\perp Y \\mid Z"
	if a.LatexString() != expected {
		t.Errorf("expected %q, got %q", expected, a.LatexString())
	}
}

func TestLatexString_MultipleVars(t *testing.T) {
	a := NewIndependenceAssertion([]string{"A", "B"}, []string{"C", "D"}, []string{"E"})
	expected := "\\{A, B\\} \\perp \\{C, D\\} \\mid E"
	if a.LatexString() != expected {
		t.Errorf("expected %q, got %q", expected, a.LatexString())
	}
}

func TestLatexString_NoGiven(t *testing.T) {
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	expected := "X \\perp Y"
	if a.LatexString() != expected {
		t.Errorf("expected %q, got %q", expected, a.LatexString())
	}
}

// ---------------------------------------------------------------------------
// Independencies tests
// ---------------------------------------------------------------------------

func TestNewIndependencies(t *testing.T) {
	ind := NewIndependencies()
	if ind == nil {
		t.Fatal("expected non-nil")
	}
	if ind.Len() != 0 {
		t.Error("new collection should be empty")
	}
}

func TestIndependenciesAdd(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	ind.Add(a)
	if ind.Len() != 1 {
		t.Errorf("expected 1 assertion, got %d", ind.Len())
	}
}

func TestIndependenciesAddMultiple(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	b := NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"})
	ind.Add(a, b)
	if ind.Len() != 2 {
		t.Errorf("expected 2 assertions, got %d", ind.Len())
	}
}

func TestIndependenciesAddSkipsDuplicates(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	b := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	ind.Add(a)
	ind.Add(b)
	if ind.Len() != 1 {
		t.Errorf("expected 1 assertion (duplicate skipped), got %d", ind.Len())
	}
}

func TestIndependenciesAddSkipsSymmetricDuplicate(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	b := NewIndependenceAssertion([]string{"Y"}, []string{"X"}, []string{"Z"})
	ind.Add(a, b)
	if ind.Len() != 1 {
		t.Errorf("expected 1 (symmetric duplicate skipped), got %d", ind.Len())
	}
}

func TestIndependenciesAddNil(t *testing.T) {
	ind := NewIndependencies()
	ind.Add(nil)
	if ind.Len() != 0 {
		t.Error("nil assertion should not be added")
	}
}

func TestIndependenciesRemove(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	b := NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"})
	ind.Add(a, b)
	ind.Remove(a)
	if ind.Len() != 1 {
		t.Errorf("expected 1 after remove, got %d", ind.Len())
	}
	if ind.Contains(a) {
		t.Error("should not contain removed assertion")
	}
	if !ind.Contains(b) {
		t.Error("should still contain other assertion")
	}
}

func TestIndependenciesRemoveByEquality(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	ind.Add(a)
	// Remove using symmetric equivalent
	sym := NewIndependenceAssertion([]string{"Y"}, []string{"X"}, []string{"Z"})
	ind.Remove(sym)
	if ind.Len() != 0 {
		t.Error("should remove by equality (symmetric)")
	}
}

func TestIndependenciesRemoveNonExistent(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	ind.Add(a)
	b := NewIndependenceAssertion([]string{"A"}, []string{"B"}, nil)
	ind.Remove(b)
	if ind.Len() != 1 {
		t.Error("removing non-existent should not change collection")
	}
}

func TestIndependenciesRemoveNil(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	ind.Add(a)
	ind.Remove(nil)
	if ind.Len() != 1 {
		t.Error("removing nil should not change collection")
	}
}

func TestIndependenciesContains(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"})
	ind.Add(a)
	if !ind.Contains(a) {
		t.Error("should contain added assertion")
	}
	b := NewIndependenceAssertion([]string{"A"}, []string{"B"}, nil)
	if ind.Contains(b) {
		t.Error("should not contain assertion not added")
	}
}

func TestIndependenciesContainsNil(t *testing.T) {
	ind := NewIndependencies()
	if ind.Contains(nil) {
		t.Error("should not contain nil")
	}
}

func TestIndependenciesGetAssertions(t *testing.T) {
	ind := NewIndependencies()
	a := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	b := NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"})
	ind.Add(a, b)
	got := ind.GetAssertions()
	if len(got) != 2 {
		t.Fatalf("expected 2, got %d", len(got))
	}
	// Verify it's a copy
	got[0] = nil
	if ind.GetAssertions()[0] == nil {
		t.Error("GetAssertions should return a copy")
	}
}

func TestIndependenciesIsEquivalent_Same(t *testing.T) {
	a1 := NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil)
	a2 := NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"})

	ind1 := NewIndependencies()
	ind1.Add(a1, a2)

	ind2 := NewIndependencies()
	ind2.Add(
		NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"}),
		NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil),
	)

	if !ind1.IsEquivalent(ind2) {
		t.Error("collections with same assertions (different order) should be equivalent")
	}
}

func TestIndependenciesIsEquivalent_DifferentSize(t *testing.T) {
	ind1 := NewIndependencies()
	ind1.Add(NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil))

	ind2 := NewIndependencies()
	ind2.Add(
		NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil),
		NewIndependenceAssertion([]string{"A"}, []string{"B"}, nil),
	)

	if ind1.IsEquivalent(ind2) {
		t.Error("collections with different sizes should not be equivalent")
	}
}

func TestIndependenciesIsEquivalent_DifferentContent(t *testing.T) {
	ind1 := NewIndependencies()
	ind1.Add(NewIndependenceAssertion([]string{"X"}, []string{"Y"}, nil))

	ind2 := NewIndependencies()
	ind2.Add(NewIndependenceAssertion([]string{"A"}, []string{"B"}, nil))

	if ind1.IsEquivalent(ind2) {
		t.Error("collections with different content should not be equivalent")
	}
}

func TestIndependenciesIsEquivalent_Nil(t *testing.T) {
	ind := NewIndependencies()
	if ind.IsEquivalent(nil) {
		t.Error("should not be equivalent to nil")
	}
}

func TestIndependenciesIsEquivalent_BothEmpty(t *testing.T) {
	ind1 := NewIndependencies()
	ind2 := NewIndependencies()
	if !ind1.IsEquivalent(ind2) {
		t.Error("two empty collections should be equivalent")
	}
}

func TestIndependenciesIsEquivalent_SymmetricAssertions(t *testing.T) {
	ind1 := NewIndependencies()
	ind1.Add(NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"}))

	ind2 := NewIndependencies()
	ind2.Add(NewIndependenceAssertion([]string{"Y"}, []string{"X"}, []string{"Z"}))

	if !ind1.IsEquivalent(ind2) {
		t.Error("collections with symmetric assertions should be equivalent")
	}
}

func TestIndependenciesString_Empty(t *testing.T) {
	ind := NewIndependencies()
	if ind.String() != "{}" {
		t.Errorf("expected '{}', got %q", ind.String())
	}
}

func TestIndependenciesString_NonEmpty(t *testing.T) {
	ind := NewIndependencies()
	ind.Add(NewIndependenceAssertion([]string{"X"}, []string{"Y"}, []string{"Z"}))
	ind.Add(NewIndependenceAssertion([]string{"A"}, []string{"B"}, nil))
	s := ind.String()
	if len(s) == 0 {
		t.Error("string should not be empty")
	}
	// Verify it contains the assertion strings
	if !containsSubstring(s, "X ⊥ Y | Z") {
		t.Errorf("string should contain assertion, got: %s", s)
	}
	if !containsSubstring(s, "A ⊥ B") {
		t.Errorf("string should contain assertion, got: %s", s)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func assertSliceEqual(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("slice length mismatch: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("slice mismatch at %d: got %v, want %v", i, got, want)
		}
	}
}

func containsSubstring(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && stringContains(s, sub))
}

func stringContains(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
