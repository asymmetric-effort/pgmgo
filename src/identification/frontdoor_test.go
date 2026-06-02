//go:build unit

package identification

import (
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// buildSmokingGraph creates the classic smoking -> tar -> cancer graph
// with an unobserved confounder U:
//
//	U -> Smoking, U -> Cancer, Smoking -> Tar, Tar -> Cancer
//
// The front-door criterion applies with {Tar} as the front-door set.
func buildSmokingGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("Smoking", "Tar", "Cancer", "U")
	g.AddEdge("U", "Smoking")
	g.AddEdge("U", "Cancer")
	g.AddEdge("Smoking", "Tar")
	g.AddEdge("Tar", "Cancer")
	return g
}

// buildDirectEffectGraph creates:
//
//	X -> Y (direct edge, no mediators)
func buildDirectEffectGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y")
	g.AddEdge("X", "Y")
	return g
}

// buildTwoMediatorGraph creates:
//
//	U -> X, U -> Y, X -> M1 -> M2 -> Y
func buildTwoMediatorGraph() *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "M1", "M2", "U")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Y")
	g.AddEdge("X", "M1")
	g.AddEdge("M1", "M2")
	g.AddEdge("M2", "Y")
	return g
}

func TestIsValidFrontdoorSet_SmokingExample(t *testing.T) {
	g := buildSmokingGraph()

	// {Tar} satisfies the front-door criterion for Smoking -> Cancer.
	if !IsValidFrontdoorSet(g, "Smoking", "Cancer", []string{"Tar"}) {
		t.Error("{Tar} should be a valid front-door set for Smoking -> Cancer")
	}
}

func TestIsValidFrontdoorSet_EmptySet(t *testing.T) {
	g := buildSmokingGraph()

	// Empty front-door set is never valid.
	if IsValidFrontdoorSet(g, "Smoking", "Cancer", []string{}) {
		t.Error("empty front-door set should be invalid")
	}
}

func TestIsValidFrontdoorSet_InvalidVariable(t *testing.T) {
	g := buildSmokingGraph()

	// U is a confounder, not a mediator.
	if IsValidFrontdoorSet(g, "Smoking", "Cancer", []string{"U"}) {
		t.Error("{U} should not be a valid front-door set")
	}
}

func TestIsValidFrontdoorSet_DirectEffect(t *testing.T) {
	g := buildDirectEffectGraph()

	// No mediator exists for a direct X -> Y edge.
	if IsValidFrontdoorSet(g, "X", "Y", []string{}) {
		t.Error("empty set is not a valid front-door set")
	}
}

func TestIsValidFrontdoorSet_TwoMediators(t *testing.T) {
	g := buildTwoMediatorGraph()

	// {M1, M2} should intercept all directed paths.
	if !IsValidFrontdoorSet(g, "X", "Y", []string{"M1", "M2"}) {
		t.Error("{M1, M2} should be a valid front-door set")
	}

	// {M1} alone: does it intercept all paths? Path is X -> M1 -> M2 -> Y.
	// M1 intercepts it.
	if !IsValidFrontdoorSet(g, "X", "Y", []string{"M1"}) {
		t.Error("{M1} should be a valid front-door set (intercepts all paths)")
	}

	// {M2} alone: intercepts the path X -> M1 -> M2 -> Y at M2.
	if !IsValidFrontdoorSet(g, "X", "Y", []string{"M2"}) {
		t.Error("{M2} should be a valid front-door set (intercepts all paths)")
	}
}

func TestIsValidFrontdoorSet_BypassPath(t *testing.T) {
	// X -> M -> Y and X -> Y (direct path that bypasses M).
	// U -> X, U -> Y (confounder).
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "M", "U")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Y")
	g.AddEdge("X", "M")
	g.AddEdge("M", "Y")
	g.AddEdge("X", "Y") // Direct path bypasses M.

	// {M} does NOT intercept all directed paths (X -> Y bypasses M).
	if IsValidFrontdoorSet(g, "X", "Y", []string{"M"}) {
		t.Error("{M} should not be valid when direct path X -> Y exists")
	}
}

func TestIsValidFrontdoorSet_BackdoorToMediatorOpen(t *testing.T) {
	// X -> M -> Y, V -> X, V -> M (V confounds X and M).
	// Condition 2 fails: back-door from X to M via V.
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "M", "V")
	g.AddEdge("X", "M")
	g.AddEdge("M", "Y")
	g.AddEdge("V", "X")
	g.AddEdge("V", "M")

	if IsValidFrontdoorSet(g, "X", "Y", []string{"M"}) {
		t.Error("{M} should be invalid: back-door from X to M via V")
	}
}

func TestIsValidFrontdoorSet_BackdoorMediatorToOutcomeOpen(t *testing.T) {
	// X -> M -> Y, W -> M, W -> Y (W confounds M and Y, not blocked by X).
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "M", "W")
	g.AddEdge("X", "M")
	g.AddEdge("M", "Y")
	g.AddEdge("W", "M")
	g.AddEdge("W", "Y")

	// Back-door path from M to Y via W exists, and X does not block it.
	if IsValidFrontdoorSet(g, "X", "Y", []string{"M"}) {
		t.Error("{M} should be invalid: back-door from M to Y via W not blocked by X")
	}
}

func TestGetFrontdoorSet_SmokingExample(t *testing.T) {
	g := buildSmokingGraph()

	set, err := GetFrontdoorSet(g, "Smoking", "Cancer")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(set) != 1 || set[0] != "Tar" {
		t.Errorf("expected {Tar}, got %v", set)
	}
}

func TestGetFrontdoorSet_TwoMediators(t *testing.T) {
	g := buildTwoMediatorGraph()

	set, err := GetFrontdoorSet(g, "X", "Y")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should find a valid (possibly minimal) front-door set.
	sort.Strings(set)
	if len(set) == 0 {
		t.Error("expected non-empty front-door set")
	}

	// Verify the returned set is actually valid.
	if !IsValidFrontdoorSet(g, "X", "Y", set) {
		t.Errorf("returned set %v is not a valid front-door set", set)
	}
}

func TestGetFrontdoorSet_NoMediators(t *testing.T) {
	g := buildDirectEffectGraph()

	_, err := GetFrontdoorSet(g, "X", "Y")
	if err == nil {
		t.Error("expected error when no front-door set exists (direct edge only)")
	}
}

func TestGetFrontdoorSet_ErrorWhenInvalid(t *testing.T) {
	// X -> M -> Y and X -> Y (bypass), U -> X, U -> Y.
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "M", "U")
	g.AddEdge("U", "X")
	g.AddEdge("U", "Y")
	g.AddEdge("X", "M")
	g.AddEdge("M", "Y")
	g.AddEdge("X", "Y")

	_, err := GetFrontdoorSet(g, "X", "Y")
	if err == nil {
		t.Error("expected error when no valid front-door set exists (direct path bypasses mediators)")
	}
}

func TestInterceptsAllDirectedPaths(t *testing.T) {
	// X -> A -> Y, X -> B -> Y
	g := graphgo.NewDiGraph()
	g.AddNodes("X", "Y", "A", "B")
	g.AddEdge("X", "A")
	g.AddEdge("A", "Y")
	g.AddEdge("X", "B")
	g.AddEdge("B", "Y")

	// {A} alone does not intercept all paths (X -> B -> Y bypasses A).
	if interceptsAllDirectedPaths(g, "X", "Y", map[string]bool{"A": true}) {
		t.Error("{A} should not intercept all paths (X -> B -> Y exists)")
	}

	// {A, B} intercepts all paths.
	if !interceptsAllDirectedPaths(g, "X", "Y", map[string]bool{"A": true, "B": true}) {
		t.Error("{A, B} should intercept all directed paths")
	}
}

func TestFindMediators(t *testing.T) {
	g := buildSmokingGraph()

	mediators := findMediators(g, "Smoking", "Cancer")
	if len(mediators) != 1 || mediators[0] != "Tar" {
		t.Errorf("expected [Tar], got %v", mediators)
	}
}

func TestFindMediators_TwoMediators(t *testing.T) {
	g := buildTwoMediatorGraph()

	mediators := findMediators(g, "X", "Y")
	sort.Strings(mediators)

	expected := []string{"M1", "M2"}
	if len(mediators) != 2 || mediators[0] != "M1" || mediators[1] != "M2" {
		t.Errorf("expected %v, got %v", expected, mediators)
	}
}

func TestCombinations(t *testing.T) {
	items := []string{"A", "B", "C"}

	c1 := combinations(items, 1)
	if len(c1) != 3 {
		t.Errorf("expected 3 combinations of size 1, got %d", len(c1))
	}

	c2 := combinations(items, 2)
	if len(c2) != 3 {
		t.Errorf("expected 3 combinations of size 2, got %d", len(c2))
	}

	c3 := combinations(items, 3)
	if len(c3) != 1 {
		t.Errorf("expected 1 combination of size 3, got %d", len(c3))
	}
}
