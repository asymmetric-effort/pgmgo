//go:build unit

package factors

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
)

// ---------------------------------------------------------------------------
// NewFunctionalCPD — construction and validation
// ---------------------------------------------------------------------------

func TestNewFunctionalCPD_Basic(t *testing.T) {
	fn := func(pv map[string]float64) []float64 {
		return []float64{0.3, 0.7}
	}
	cpd, err := NewFunctionalCPD("Y", []string{"A"}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if cpd.Variable() != "Y" {
		t.Errorf("Variable() = %q, want Y", cpd.Variable())
	}
	ev := cpd.Evidence()
	if len(ev) != 1 || ev[0] != "A" {
		t.Errorf("Evidence() = %v, want [A]", ev)
	}
}

func TestNewFunctionalCPD_NilFn(t *testing.T) {
	_, err := NewFunctionalCPD("Y", []string{"A"}, nil)
	if err == nil {
		t.Fatal("expected error for nil function")
	}
}

func TestNewFunctionalCPD_NoEvidence(t *testing.T) {
	fn := func(pv map[string]float64) []float64 {
		return []float64{0.5, 0.5}
	}
	cpd, err := NewFunctionalCPD("Y", nil, fn)
	if err != nil {
		t.Fatal(err)
	}
	if len(cpd.Evidence()) != 0 {
		t.Errorf("Evidence() should be empty, got %v", cpd.Evidence())
	}
}

// ---------------------------------------------------------------------------
// GetDistribution
// ---------------------------------------------------------------------------

func TestFunctionalCPD_GetDistribution(t *testing.T) {
	// Function that returns different distributions based on parent value.
	fn := func(pv map[string]float64) []float64 {
		if pv["A"] > 0.5 {
			return []float64{0.2, 0.8}
		}
		return []float64{0.9, 0.1}
	}
	cpd, err := NewFunctionalCPD("Y", []string{"A"}, fn)
	if err != nil {
		t.Fatal(err)
	}

	dist1 := cpd.GetDistribution(map[string]float64{"A": 0.0})
	if !floatEq(dist1[0], 0.9) || !floatEq(dist1[1], 0.1) {
		t.Errorf("GetDistribution(A=0) = %v, want [0.9, 0.1]", dist1)
	}

	dist2 := cpd.GetDistribution(map[string]float64{"A": 1.0})
	if !floatEq(dist2[0], 0.2) || !floatEq(dist2[1], 0.8) {
		t.Errorf("GetDistribution(A=1) = %v, want [0.2, 0.8]", dist2)
	}
}

func TestFunctionalCPD_GetDistribution_MultiParent(t *testing.T) {
	fn := func(pv map[string]float64) []float64 {
		sum := pv["A"] + pv["B"]
		if sum > 1.0 {
			return []float64{0.1, 0.3, 0.6}
		}
		return []float64{0.5, 0.3, 0.2}
	}
	cpd, err := NewFunctionalCPD("Y", []string{"A", "B"}, fn)
	if err != nil {
		t.Fatal(err)
	}

	dist := cpd.GetDistribution(map[string]float64{"A": 0.8, "B": 0.5})
	if !floatEq(dist[0], 0.1) || !floatEq(dist[1], 0.3) || !floatEq(dist[2], 0.6) {
		t.Errorf("unexpected distribution: %v", dist)
	}
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

func TestFunctionalCPD_Sample_Deterministic(t *testing.T) {
	// Distribution that is deterministic: [0, 0, 1]
	fn := func(pv map[string]float64) []float64 {
		return []float64{0.0, 0.0, 1.0}
	}
	cpd, err := NewFunctionalCPD("Y", nil, fn)
	if err != nil {
		t.Fatal(err)
	}
	rng := numgo.NewRNG(42)
	for i := 0; i < 100; i++ {
		s := cpd.Sample(nil, rng)
		if s != 2 {
			t.Fatalf("Sample() = %d, want 2 for deterministic [0,0,1]", s)
		}
	}
}

func TestFunctionalCPD_Sample_Distribution(t *testing.T) {
	// Verify sampling from [0.25, 0.75] produces roughly correct frequencies.
	fn := func(pv map[string]float64) []float64 {
		return []float64{0.25, 0.75}
	}
	cpd, err := NewFunctionalCPD("Y", nil, fn)
	if err != nil {
		t.Fatal(err)
	}
	rng := numgo.NewRNG(123)

	counts := make([]int, 2)
	n := 10000
	for i := 0; i < n; i++ {
		s := cpd.Sample(nil, rng)
		counts[s]++
	}

	// Expected: ~25% state 0, ~75% state 1.
	ratio0 := float64(counts[0]) / float64(n)
	ratio1 := float64(counts[1]) / float64(n)
	if math.Abs(ratio0-0.25) > 0.05 {
		t.Errorf("state 0 frequency = %f, expected ~0.25", ratio0)
	}
	if math.Abs(ratio1-0.75) > 0.05 {
		t.Errorf("state 1 frequency = %f, expected ~0.75", ratio1)
	}
}

func TestFunctionalCPD_Sample_ParentDependent(t *testing.T) {
	// Distribution depends on parent value.
	fn := func(pv map[string]float64) []float64 {
		if pv["A"] == 1.0 {
			return []float64{0.0, 1.0}
		}
		return []float64{1.0, 0.0}
	}
	cpd, err := NewFunctionalCPD("Y", []string{"A"}, fn)
	if err != nil {
		t.Fatal(err)
	}
	rng := numgo.NewRNG(99)

	for i := 0; i < 50; i++ {
		s := cpd.Sample(map[string]float64{"A": 1.0}, rng)
		if s != 1 {
			t.Fatalf("Sample(A=1) = %d, want 1", s)
		}
	}
	for i := 0; i < 50; i++ {
		s := cpd.Sample(map[string]float64{"A": 0.0}, rng)
		if s != 0 {
			t.Fatalf("Sample(A=0) = %d, want 0", s)
		}
	}
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

func TestFunctionalCPD_Validate(t *testing.T) {
	fn := func(pv map[string]float64) []float64 { return []float64{1.0} }
	cpd, err := NewFunctionalCPD("Y", nil, fn)
	if err != nil {
		t.Fatal(err)
	}
	if err := cpd.Validate(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestFunctionalCPD_Validate_NilFn(t *testing.T) {
	cpd := &FunctionalCPD{variable: "Y", fn: nil}
	if err := cpd.Validate(); err == nil {
		t.Error("expected validation error for nil fn")
	}
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

func TestFunctionalCPD_Copy(t *testing.T) {
	fn := func(pv map[string]float64) []float64 { return []float64{0.5, 0.5} }
	cpd, err := NewFunctionalCPD("Y", []string{"A", "B"}, fn)
	if err != nil {
		t.Fatal(err)
	}
	c := cpd.Copy()
	if c.Variable() != cpd.Variable() {
		t.Errorf("copy Variable() = %q, want %q", c.Variable(), cpd.Variable())
	}
	if len(c.Evidence()) != len(cpd.Evidence()) {
		t.Errorf("copy Evidence() length mismatch")
	}
	// Mutate original evidence and verify copy is independent.
	cpd.evidence[0] = "Z"
	if c.Evidence()[0] == "Z" {
		t.Error("copy shares evidence slice with original")
	}
	// Function reference should be the same.
	dist := c.GetDistribution(nil)
	if !floatEq(dist[0], 0.5) || !floatEq(dist[1], 0.5) {
		t.Errorf("copy function returned unexpected distribution: %v", dist)
	}
}
