//go:build unit

package identification

import (
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// buildConfoundedGraph creates:
//
//	U -> X, U -> Y, X -> Y
//
// U is a confounder. Adjusting for U satisfies the back-door criterion.
func buildConfoundedGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "U")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Y")
	g.AddEdge("X", "Y")
	return g
}

// buildMultipleConfounderGraph creates:
//
//	U1 -> X, U1 -> Y, U2 -> X, U2 -> Y, X -> Y
func buildMultipleConfounderGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "U1", "U2")
	g.AddEdge("U1", "X")
	g.AddEdge("U1", "Y")
	g.AddEdge("U2", "X")
	g.AddEdge("U2", "Y")
	g.AddEdge("X", "Y")
	return g
}

// buildMediatorGraph creates:
//
//	X -> M -> Y (no confounders)
func buildMediatorGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "M", "Y")
	g.AddEdge("X", "M")
	g.AddEdge("M", "Y")
	return g
}

// buildColliderGraph creates:
//
//	X -> Y, X -> C <- Y
//
// C is a collider — adjusting for C opens a spurious path.
func buildColliderGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "C")
	g.AddEdge("X", "Y")
	g.AddEdge("X", "C")
	g.AddEdge("Y", "C")
	return g
}

func TestIsValidAdjustmentSet_SimpleConfounder(t *testing.T) {
	g := buildConfoundedGraph()

	// Adjusting for U blocks the back-door path X <- U -> Y.
	if !IsValidAdjustmentSet(g, "X", "Y", []string{"U"}) {
		t.Error("adjusting for U should be valid (blocks back-door path)")
	}
}

func TestIsValidAdjustmentSet_EmptySetWithConfounder(t *testing.T) {
	g := buildConfoundedGraph()

	// Empty set does NOT block the back-door path X <- U -> Y.
	if IsValidAdjustmentSet(g, "X", "Y", []string{}) {
		t.Error("empty adjustment set should be invalid when confounder exists")
	}
}

func TestIsValidAdjustmentSet_NoConfounder(t *testing.T) {
	g := buildMediatorGraph()

	// X -> M -> Y has no back-door paths, so empty set is valid.
	if !IsValidAdjustmentSet(g, "X", "Y", []string{}) {
		t.Error("empty set should be valid when there are no back-door paths")
	}
}

func TestIsValidAdjustmentSet_DescendantBlocked(t *testing.T) {
	g := buildConfoundedGraph()

	// Y is a descendant of X — cannot adjust for it.
	if IsValidAdjustmentSet(g, "X", "Y", []string{"Y"}) {
		t.Error("adjusting for a descendant of treatment should be invalid")
	}
}

func TestIsValidAdjustmentSet_MediatorIsDescendant(t *testing.T) {
	// X -> M -> Y with confounder U -> X, U -> Y.
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "M", "Y", "U")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Y")
	g.AddEdge("X", "M")
	g.AddEdge("M", "Y")

	// M is a descendant of X — cannot use it in adjustment set.
	if IsValidAdjustmentSet(g, "X", "Y", []string{"M"}) {
		t.Error("adjusting for mediator M (descendant of X) should be invalid")
	}

	// U is valid.
	if !IsValidAdjustmentSet(g, "X", "Y", []string{"U"}) {
		t.Error("adjusting for U should be valid")
	}
}

func TestIsValidAdjustmentSet_ColliderNotAdjusted(t *testing.T) {
	g := buildColliderGraph()

	// No back-door paths exist (X -> Y is causal, X -> C <- Y has collider C).
	// Empty set is valid.
	if !IsValidAdjustmentSet(g, "X", "Y", []string{}) {
		t.Error("empty set should be valid when only causal path and collider exist")
	}
}

func TestIsValidAdjustmentSet_ColliderAdjustedOpensPath(t *testing.T) {
	g := buildColliderGraph()

	// C is a descendant of X, so adjusting for it violates condition 1.
	if IsValidAdjustmentSet(g, "X", "Y", []string{"C"}) {
		t.Error("adjusting for collider C (descendant of X) should be invalid")
	}
}

func TestIsValidAdjustmentSet_MultipleConfounders(t *testing.T) {
	g := buildMultipleConfounderGraph()

	// Need to adjust for both confounders.
	if !IsValidAdjustmentSet(g, "X", "Y", []string{"U1", "U2"}) {
		t.Error("adjusting for both U1 and U2 should be valid")
	}

	// Adjusting for only one may not block all paths.
	if IsValidAdjustmentSet(g, "X", "Y", []string{"U1"}) {
		t.Error("adjusting for only U1 should be invalid (U2 path remains open)")
	}

	if IsValidAdjustmentSet(g, "X", "Y", []string{"U2"}) {
		t.Error("adjusting for only U2 should be invalid (U1 path remains open)")
	}
}

func TestGetMinimalAdjustmentSet_SimpleConfounder(t *testing.T) {
	g := buildConfoundedGraph()

	set, err := GetMinimalAdjustmentSet(g, "X", "Y")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Parents of X are {U}, which is the minimal set.
	if len(set) != 1 || set[0] != "U" {
		t.Errorf("expected minimal adjustment set {U}, got %v", set)
	}
}

func TestGetMinimalAdjustmentSet_MultipleConfounders(t *testing.T) {
	g := buildMultipleConfounderGraph()

	set, err := GetMinimalAdjustmentSet(g, "X", "Y")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	sort.Strings(set)
	expected := []string{"U1", "U2"}
	if len(set) != 2 || set[0] != "U1" || set[1] != "U2" {
		t.Errorf("expected minimal adjustment set %v, got %v", expected, set)
	}
}

func TestGetMinimalAdjustmentSet_NoConfounders(t *testing.T) {
	g := buildMediatorGraph()

	set, err := GetMinimalAdjustmentSet(g, "X", "Y")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// No confounders, empty set is valid (parents of X = empty).
	if len(set) != 0 {
		t.Errorf("expected empty minimal adjustment set, got %v", set)
	}
}

func TestGetMinimalAdjustmentSet_RedundantParent(t *testing.T) {
	// X has two parents, but only one is a confounder.
	// U -> X, U -> Y, W -> X, X -> Y
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "U", "W")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Y")
	g.AddEdge("W", "X")
	g.AddEdge("X", "Y")

	set, err := GetMinimalAdjustmentSet(g, "X", "Y")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// U is necessary, W is redundant. Minimal set should be {U}.
	if len(set) != 1 || set[0] != "U" {
		t.Errorf("expected minimal set {U}, got %v", set)
	}
}

func TestGetMinimalAdjustmentSet_ErrorWhenParentsInvalid(t *testing.T) {
	// Contrived case: treatment has no parents but there's a hidden back-door.
	// Actually, if treatment has no parents, there are no back-door paths,
	// so the empty set is always valid. Let's test that.
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y")
	g.AddEdge("X", "Y")

	set, err := GetMinimalAdjustmentSet(g, "X", "Y")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(set) != 0 {
		t.Errorf("expected empty set for graph with no confounders, got %v", set)
	}
}

func TestGetAllAdjustmentSets_SimpleConfounder(t *testing.T) {
	g := buildConfoundedGraph()

	sets := GetAllAdjustmentSets(g, "X", "Y")

	// Only candidate variable is U (Y is outcome, X is treatment).
	// Empty set is invalid, {U} is valid.
	if len(sets) != 1 {
		t.Fatalf("expected 1 valid adjustment set, got %d: %v", len(sets), sets)
	}

	if len(sets[0]) != 1 || sets[0][0] != "U" {
		t.Errorf("expected {U}, got %v", sets[0])
	}
}

func TestGetAllAdjustmentSets_NoConfounders(t *testing.T) {
	g := buildMediatorGraph()

	sets := GetAllAdjustmentSets(g, "X", "Y")

	// Empty set should be valid. No non-descendant candidates exist (M is
	// a descendant of X), so empty set is the only candidate.
	if len(sets) != 1 {
		t.Fatalf("expected 1 valid adjustment set, got %d: %v", len(sets), sets)
	}

	if len(sets[0]) != 0 {
		t.Errorf("expected empty set, got %v", sets[0])
	}
}

func TestGetAllAdjustmentSets_MultipleConfounders(t *testing.T) {
	g := buildMultipleConfounderGraph()

	sets := GetAllAdjustmentSets(g, "X", "Y")

	// Candidates are U1 and U2. Subsets: {}, {U1}, {U2}, {U1, U2}.
	// Only {U1, U2} should be valid.
	if len(sets) != 1 {
		t.Fatalf("expected 1 valid adjustment set, got %d: %v", len(sets), sets)
	}

	sort.Strings(sets[0])
	if len(sets[0]) != 2 || sets[0][0] != "U1" || sets[0][1] != "U2" {
		t.Errorf("expected {U1, U2}, got %v", sets[0])
	}
}

func TestGetAllAdjustmentSets_MultipleValid(t *testing.T) {
	// Graph: U1 -> X, U1 -> U2, U2 -> Y, X -> Y
	// Back-door path: X <- U1 -> U2 -> Y.
	// Adjusting for U1 blocks it. Adjusting for U2 also blocks it.
	// Adjusting for {U1, U2} also works.
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "U1", "U2")
	g.AddEdge("U1", "X")
	g.AddEdge("U1", "U2")
	g.AddEdge("U2", "Y")
	g.AddEdge("X", "Y")

	sets := GetAllAdjustmentSets(g, "X", "Y")

	// We expect at least {U1}, {U2}, and {U1, U2} to be valid.
	if len(sets) < 2 {
		t.Errorf("expected multiple valid adjustment sets, got %d: %v", len(sets), sets)
	}

	// Check that {U1} is among them.
	foundU1 := false
	foundU1U2 := false
	for _, s := range sets {
		sort.Strings(s)
		if len(s) == 1 && s[0] == "U1" {
			foundU1 = true
		}
		if len(s) == 2 && s[0] == "U1" && s[1] == "U2" {
			foundU1U2 = true
		}
	}
	if !foundU1 {
		t.Errorf("expected {U1} to be a valid adjustment set, sets: %v", sets)
	}
	if !foundU1U2 {
		t.Errorf("expected {U1, U2} to be a valid adjustment set, sets: %v", sets)
	}
}

func TestIsValidAdjustmentSet_TreatmentNotInGraph(t *testing.T) {
	g := graphgo.NewDiGraph()
	g.AddNodes("A", "B")
	g.AddEdge("A", "B")

	// Treatment "X" doesn't exist; Descendants returns empty, no back-door paths.
	// The function should handle this gracefully — the manipulated graph won't
	// have "X" either. This is an edge case; we just ensure no panic.
	_ = IsValidAdjustmentSet(g, "X", "B", []string{})
}

func TestGetAllAdjustmentSets_LargerGraph(t *testing.T) {
	// Diamond graph: U -> X, U -> Z, Z -> Y, X -> Y
	// Back-door: X <- U -> Z -> Y
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "U", "Z")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Z")
	g.AddEdge("Z", "Y")
	g.AddEdge("X", "Y")

	sets := GetAllAdjustmentSets(g, "X", "Y")

	// Candidates: U and Z (neither is a descendant of X).
	// {U} blocks at U; {Z} blocks at Z; {U, Z} blocks at both.
	// {} does not block.
	foundEmpty := false
	for _, s := range sets {
		if len(s) == 0 {
			foundEmpty = true
		}
	}
	if foundEmpty {
		t.Error("empty set should not be valid with confounder U")
	}

	if len(sets) < 3 {
		t.Errorf("expected at least 3 valid sets ({U}, {Z}, {U,Z}), got %d: %v", len(sets), sets)
	}
}
