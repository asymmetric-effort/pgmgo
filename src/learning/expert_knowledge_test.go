//go:build unit

package learning

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestExpertKnowledge_RequiredEdge(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddRequiredEdge("A", "B")

	if !ek.IsRequired("A", "B") {
		t.Error("expected A->B to be required")
	}
	if ek.IsRequired("B", "A") {
		t.Error("expected B->A to not be required")
	}
	if ek.IsRequired("A", "C") {
		t.Error("expected A->C to not be required")
	}
}

func TestExpertKnowledge_ForbiddenEdge(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddForbiddenEdge("A", "B")

	if ek.IsAllowed("A", "B") {
		t.Error("expected A->B to be forbidden")
	}
	if !ek.IsAllowed("B", "A") {
		t.Error("expected B->A to be allowed")
	}
	if !ek.IsAllowed("A", "C") {
		t.Error("expected A->C to be allowed")
	}
}

func TestExpertKnowledge_TierOrdering(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddTierOrdering([][]string{
		{"A", "B"}, // tier 0 (earliest)
		{"C"},      // tier 1
		{"D", "E"}, // tier 2 (latest)
	})

	// Edges from earlier to later tier: allowed.
	if !ek.IsAllowed("A", "C") {
		t.Error("expected A->C (tier 0->1) to be allowed")
	}
	if !ek.IsAllowed("A", "D") {
		t.Error("expected A->D (tier 0->2) to be allowed")
	}
	if !ek.IsAllowed("C", "D") {
		t.Error("expected C->D (tier 1->2) to be allowed")
	}

	// Edges within same tier: allowed.
	if !ek.IsAllowed("A", "B") {
		t.Error("expected A->B (same tier 0) to be allowed")
	}
	if !ek.IsAllowed("D", "E") {
		t.Error("expected D->E (same tier 2) to be allowed")
	}

	// Edges from later to earlier tier: forbidden.
	if ek.IsAllowed("C", "A") {
		t.Error("expected C->A (tier 1->0) to be forbidden")
	}
	if ek.IsAllowed("D", "A") {
		t.Error("expected D->A (tier 2->0) to be forbidden")
	}
	if ek.IsAllowed("D", "C") {
		t.Error("expected D->C (tier 2->1) to be forbidden")
	}
}

func TestExpertKnowledge_TierWithUnknownVars(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddTierOrdering([][]string{
		{"A"}, // tier 0
		{"B"}, // tier 1
	})

	// Unknown variables should be allowed (no tier constraint).
	if !ek.IsAllowed("X", "A") {
		t.Error("expected X->A to be allowed (X has no tier)")
	}
	if !ek.IsAllowed("A", "X") {
		t.Error("expected A->X to be allowed (X has no tier)")
	}
}

func TestExpertKnowledge_CombinedConstraints(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddRequiredEdge("A", "B")
	ek.AddForbiddenEdge("B", "C")
	ek.AddTierOrdering([][]string{
		{"A"},
		{"B"},
		{"C"},
	})

	// A->B: required and allowed.
	if !ek.IsRequired("A", "B") {
		t.Error("expected A->B to be required")
	}
	if !ek.IsAllowed("A", "B") {
		t.Error("expected A->B to be allowed")
	}

	// B->C: forbidden explicitly.
	if ek.IsAllowed("B", "C") {
		t.Error("expected B->C to be forbidden")
	}

	// C->A: forbidden by tier ordering.
	if ek.IsAllowed("C", "A") {
		t.Error("expected C->A to be forbidden by tier ordering")
	}

	// A->C: allowed (tier 0->2, not forbidden).
	if !ek.IsAllowed("A", "C") {
		t.Error("expected A->C to be allowed")
	}
}

func TestExpertKnowledge_ApplyToHillClimb(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddRequiredEdge("A", "B")
	ek.AddForbiddenEdge("C", "A")
	ek.AddTierOrdering([][]string{
		{"A", "B"},
		{"C"},
	})

	data := syntheticData()
	hc := NewHillClimbSearch(data, countScore)

	ek.ApplyToHillClimb(hc)

	// Check whitelist contains required edge.
	if !hc.whiteList[[2]string{"A", "B"}] {
		t.Error("expected whitelist to contain A->B")
	}

	// Check blacklist contains forbidden edge.
	if !hc.blackList[[2]string{"C", "A"}] {
		t.Error("expected blacklist to contain C->A")
	}

	// Check blacklist contains tier-violating edge C->B (tier 1->0).
	if !hc.blackList[[2]string{"C", "A"}] {
		t.Error("expected blacklist to contain C->A (tier violation)")
	}
	if !hc.blackList[[2]string{"C", "B"}] {
		t.Error("expected blacklist to contain C->B (tier violation)")
	}

	// A->C should NOT be blacklisted (tier 0->1 is allowed).
	if hc.blackList[[2]string{"A", "C"}] {
		t.Error("expected A->C to NOT be blacklisted (tier 0->1)")
	}
}

func TestExpertKnowledge_ApplyAndEstimate(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddRequiredEdge("A", "B")
	ek.AddForbiddenEdge("C", "A")

	data := syntheticData()
	hc := NewHillClimbSearch(data, countScore)
	ek.ApplyToHillClimb(hc)

	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	es := edgeSet(edges)
	t.Logf("Edges with expert knowledge: %v", edges)

	// Required edge A->B should be present.
	if !es["A->B"] {
		t.Error("expected required edge A->B to be present")
	}

	// Forbidden edge C->A should not be present.
	if es["C->A"] {
		t.Error("expected forbidden edge C->A to not be present")
	}
}

func TestExpertKnowledge_EmptyConstraints(t *testing.T) {
	ek := NewExpertKnowledge()

	// Everything should be allowed and nothing required.
	if !ek.IsAllowed("A", "B") {
		t.Error("expected A->B to be allowed with no constraints")
	}
	if ek.IsRequired("A", "B") {
		t.Error("expected A->B to not be required with no constraints")
	}
}

func TestExpertKnowledge_ReplaceTierOrdering(t *testing.T) {
	ek := NewExpertKnowledge()
	ek.AddTierOrdering([][]string{{"A"}, {"B"}})

	// B->A should be forbidden.
	if ek.IsAllowed("B", "A") {
		t.Error("expected B->A to be forbidden")
	}

	// Replace tier ordering.
	ek.AddTierOrdering([][]string{{"B"}, {"A"}})

	// Now A->B should be forbidden, B->A should be allowed.
	if ek.IsAllowed("A", "B") {
		t.Error("expected A->B to be forbidden after reordering")
	}
	if !ek.IsAllowed("B", "A") {
		t.Error("expected B->A to be allowed after reordering")
	}
}

func TestExpertKnowledge_HillClimbWithTiers(t *testing.T) {
	// Create data where all vars are perfectly correlated.
	n := 100
	vals := make([]any, n)
	for i := range vals {
		vals[i] = i % 2
	}
	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", vals),
		"B": tabgo.NewSeries("B", vals),
		"C": tabgo.NewSeries("C", vals),
	})

	ek := NewExpertKnowledge()
	ek.AddTierOrdering([][]string{
		{"A"},
		{"B"},
		{"C"},
	})

	hc := NewHillClimbSearch(data, countScore)
	ek.ApplyToHillClimb(hc)

	bn, err := hc.Estimate()
	if err != nil {
		t.Fatalf("Estimate() error: %v", err)
	}

	edges := bn.Edges()
	es := edgeSet(edges)
	t.Logf("Edges with tier ordering: %v", edges)

	// No edge from B->A, C->A, or C->B should exist.
	if es["B->A"] {
		t.Error("tier violation: B->A should not exist")
	}
	if es["C->A"] {
		t.Error("tier violation: C->A should not exist")
	}
	if es["C->B"] {
		t.Error("tier violation: C->B should not exist")
	}
}
