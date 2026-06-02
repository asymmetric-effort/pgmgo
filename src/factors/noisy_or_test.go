//go:build unit

package factors

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// NewNoisyOR — construction and validation
// ---------------------------------------------------------------------------

func TestNewNoisyOR_Basic(t *testing.T) {
	n, err := NewNoisyOR("Y", 2, []string{"A", "B"}, []float64{0.3, 0.5}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	if n.Variable() != "Y" {
		t.Errorf("Variable() = %q, want Y", n.Variable())
	}
	if n.VariableCard() != 2 {
		t.Errorf("VariableCard() = %d, want 2", n.VariableCard())
	}
	parents := n.Parents()
	if len(parents) != 2 || parents[0] != "A" || parents[1] != "B" {
		t.Errorf("Parents() = %v, want [A B]", parents)
	}
	inh := n.InhibitionProbs()
	if len(inh) != 2 || inh[0] != 0.3 || inh[1] != 0.5 {
		t.Errorf("InhibitionProbs() = %v, want [0.3 0.5]", inh)
	}
	if n.LeakProb() != 0.9 {
		t.Errorf("LeakProb() = %f, want 0.9", n.LeakProb())
	}
}

func TestNewNoisyOR_MismatchedLengths(t *testing.T) {
	_, err := NewNoisyOR("Y", 2, []string{"A", "B"}, []float64{0.3}, 0.9)
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

func TestNewNoisyOR_InvalidVariableCard(t *testing.T) {
	_, err := NewNoisyOR("Y", 3, []string{"A"}, []float64{0.3}, 0.9)
	if err == nil {
		t.Fatal("expected error for non-binary variableCard")
	}
}

func TestNewNoisyOR_InvalidLeakProb(t *testing.T) {
	_, err := NewNoisyOR("Y", 2, []string{"A"}, []float64{0.3}, 1.5)
	if err == nil {
		t.Fatal("expected error for leakProb > 1")
	}
	_, err = NewNoisyOR("Y", 2, []string{"A"}, []float64{0.3}, -0.1)
	if err == nil {
		t.Fatal("expected error for leakProb < 0")
	}
}

func TestNewNoisyOR_InvalidInhibitionProb(t *testing.T) {
	_, err := NewNoisyOR("Y", 2, []string{"A"}, []float64{1.5}, 0.9)
	if err == nil {
		t.Fatal("expected error for inhibition prob > 1")
	}
	_, err = NewNoisyOR("Y", 2, []string{"A"}, []float64{-0.1}, 0.9)
	if err == nil {
		t.Fatal("expected error for inhibition prob < 0")
	}
}

func TestNewNoisyOR_NoParents(t *testing.T) {
	n, err := NewNoisyOR("Y", 2, nil, nil, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	if len(n.Parents()) != 0 {
		t.Errorf("expected no parents, got %v", n.Parents())
	}
}

// ---------------------------------------------------------------------------
// ToTabularCPD — hand-computed values
// ---------------------------------------------------------------------------

func TestNoisyOR_ToTabularCPD_NoParents(t *testing.T) {
	// No parents: P(Y=0) = leakProb, P(Y=1) = 1 - leakProb
	n, err := NewNoisyOR("Y", 2, nil, nil, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()
	// Only one parent config (no parents).
	pY0 := factor.GetValue(map[string]int{"Y": 0})
	pY1 := factor.GetValue(map[string]int{"Y": 1})
	if !floatEq(pY0, 0.8) {
		t.Errorf("P(Y=0) = %f, want 0.8", pY0)
	}
	if !floatEq(pY1, 0.2) {
		t.Errorf("P(Y=1) = %f, want 0.2", pY1)
	}
}

func TestNoisyOR_ToTabularCPD_OneParent(t *testing.T) {
	// Y with one parent A, inhibition=0.4, leak=0.9
	// P(Y=0|A=0) = 0.9 * 1     = 0.9
	// P(Y=0|A=1) = 0.9 * 0.4   = 0.36
	// P(Y=1|A=0) = 1 - 0.9     = 0.1
	// P(Y=1|A=1) = 1 - 0.36    = 0.64
	n, err := NewNoisyOR("Y", 2, []string{"A"}, []float64{0.4}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	tests := []struct {
		y, a int
		want float64
	}{
		{0, 0, 0.9},
		{0, 1, 0.36},
		{1, 0, 0.1},
		{1, 1, 0.64},
	}
	for _, tc := range tests {
		got := factor.GetValue(map[string]int{"Y": tc.y, "A": tc.a})
		if !floatEq(got, tc.want) {
			t.Errorf("P(Y=%d|A=%d) = %f, want %f", tc.y, tc.a, got, tc.want)
		}
	}
}

func TestNoisyOR_ToTabularCPD_TwoParents(t *testing.T) {
	// Y with parents A, B; inhibition=[0.3, 0.5]; leak=0.9
	// Parent configs (row-major, A varies slowest):
	//   config 0: A=0, B=0 -> P(Y=0) = 0.9 * 1   * 1   = 0.9
	//   config 1: A=0, B=1 -> P(Y=0) = 0.9 * 1   * 0.5 = 0.45
	//   config 2: A=1, B=0 -> P(Y=0) = 0.9 * 0.3 * 1   = 0.27
	//   config 3: A=1, B=1 -> P(Y=0) = 0.9 * 0.3 * 0.5 = 0.135
	n, err := NewNoisyOR("Y", 2, []string{"A", "B"}, []float64{0.3, 0.5}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	tests := []struct {
		y, a, b int
		want    float64
	}{
		{0, 0, 0, 0.9},
		{0, 0, 1, 0.45},
		{0, 1, 0, 0.27},
		{0, 1, 1, 0.135},
		{1, 0, 0, 0.1},
		{1, 0, 1, 0.55},
		{1, 1, 0, 0.73},
		{1, 1, 1, 0.865},
	}
	for _, tc := range tests {
		got := factor.GetValue(map[string]int{"Y": tc.y, "A": tc.a, "B": tc.b})
		if !floatEq(got, tc.want) {
			t.Errorf("P(Y=%d|A=%d,B=%d) = %f, want %f", tc.y, tc.a, tc.b, got, tc.want)
		}
	}
}

func TestNoisyOR_ToTabularCPD_ColumnsSumToOne(t *testing.T) {
	n, err := NewNoisyOR("Y", 2, []string{"A", "B"}, []float64{0.3, 0.5}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	if err := cpd.Validate(); err != nil {
		t.Errorf("TabularCPD validation failed: %v", err)
	}
}

func TestNoisyOR_ToTabularCPD_ThreeParents(t *testing.T) {
	// Y with parents A, B, C; inhibition=[0.2, 0.4, 0.6]; leak=1.0
	// P(Y=0|A=1,B=1,C=1) = 1.0 * 0.2 * 0.4 * 0.6 = 0.048
	n, err := NewNoisyOR("Y", 2, []string{"A", "B", "C"}, []float64{0.2, 0.4, 0.6}, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	pY0 := factor.GetValue(map[string]int{"Y": 0, "A": 1, "B": 1, "C": 1})
	if !floatEq(pY0, 0.048) {
		t.Errorf("P(Y=0|A=1,B=1,C=1) = %f, want 0.048", pY0)
	}
	pY1 := factor.GetValue(map[string]int{"Y": 1, "A": 1, "B": 1, "C": 1})
	if !floatEq(pY1, 0.952) {
		t.Errorf("P(Y=1|A=1,B=1,C=1) = %f, want 0.952", pY1)
	}

	// All parents off: P(Y=0|A=0,B=0,C=0) = 1.0
	pY0all := factor.GetValue(map[string]int{"Y": 0, "A": 0, "B": 0, "C": 0})
	if !floatEq(pY0all, 1.0) {
		t.Errorf("P(Y=0|A=0,B=0,C=0) = %f, want 1.0", pY0all)
	}
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

func TestNoisyOR_Validate(t *testing.T) {
	n, err := NewNoisyOR("Y", 2, []string{"A"}, []float64{0.3}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	if err := n.Validate(); err != nil {
		t.Errorf("unexpected validation error: %v", err)
	}
}

func TestNoisyOR_Validate_Invalid(t *testing.T) {
	n := &NoisyOR{
		variable:        "Y",
		variableCard:    2,
		parents:         []string{"A"},
		inhibitionProbs: []float64{0.3, 0.5}, // mismatched
		leakProb:        0.9,
	}
	if err := n.Validate(); err == nil {
		t.Error("expected validation error for mismatched lengths")
	}
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

func TestNoisyOR_Copy(t *testing.T) {
	n, err := NewNoisyOR("Y", 2, []string{"A", "B"}, []float64{0.3, 0.5}, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	c := n.Copy()
	if c.Variable() != n.Variable() {
		t.Errorf("copy Variable() = %q, want %q", c.Variable(), n.Variable())
	}
	if c.LeakProb() != n.LeakProb() {
		t.Errorf("copy LeakProb() = %f, want %f", c.LeakProb(), n.LeakProb())
	}
	// Mutate original and verify copy is independent.
	n.inhibitionProbs[0] = 0.99
	if c.InhibitionProbs()[0] == 0.99 {
		t.Error("copy shares inhibitionProbs slice with original")
	}
	n.parents[0] = "Z"
	if c.Parents()[0] == "Z" {
		t.Error("copy shares parents slice with original")
	}
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

func TestNoisyOR_ZeroLeak(t *testing.T) {
	// Leak=0 means without any parent active, P(Y=0)=0 and P(Y=1)=0... wait,
	// P(Y=0|all parents=0) = 0 * prod(1) = 0, so P(Y=1|all=0) = 1.
	n, err := NewNoisyOR("Y", 2, []string{"A"}, []float64{0.5}, 0.0)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	pY0a0 := factor.GetValue(map[string]int{"Y": 0, "A": 0})
	if !floatEq(pY0a0, 0.0) {
		t.Errorf("P(Y=0|A=0) = %f, want 0.0", pY0a0)
	}
	pY1a0 := factor.GetValue(map[string]int{"Y": 1, "A": 0})
	if !floatEq(pY1a0, 1.0) {
		t.Errorf("P(Y=1|A=0) = %f, want 1.0", pY1a0)
	}
}

func TestNoisyOR_ZeroInhibition(t *testing.T) {
	// If inhibition is 0 for a parent, any activation of that parent forces
	// P(Y=0) to be 0 (since we multiply by 0).
	n, err := NewNoisyOR("Y", 2, []string{"A"}, []float64{0.0}, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	// A=0: P(Y=0) = 1.0 * 1 = 1.0
	if got := factor.GetValue(map[string]int{"Y": 0, "A": 0}); !floatEq(got, 1.0) {
		t.Errorf("P(Y=0|A=0) = %f, want 1.0", got)
	}
	// A=1: P(Y=0) = 1.0 * 0.0 = 0.0
	if got := factor.GetValue(map[string]int{"Y": 0, "A": 1}); !floatEq(got, 0.0) {
		t.Errorf("P(Y=0|A=1) = %f, want 0.0", got)
	}
}

func TestNoisyOR_DeterministicOR(t *testing.T) {
	// leak=1.0, all inhibitions=0 -> behaves like a deterministic OR.
	// P(Y=1) = 1 if any parent is 1, P(Y=0) = 1 only if all parents are 0.
	n, err := NewNoisyOR("Y", 2, []string{"A", "B"}, []float64{0.0, 0.0}, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	// A=0, B=0: P(Y=0)=1
	if got := factor.GetValue(map[string]int{"Y": 0, "A": 0, "B": 0}); !floatEq(got, 1.0) {
		t.Errorf("P(Y=0|A=0,B=0) = %f, want 1.0", got)
	}
	// A=1, B=0: P(Y=1)=1
	if got := factor.GetValue(map[string]int{"Y": 1, "A": 1, "B": 0}); !floatEq(got, 1.0) {
		t.Errorf("P(Y=1|A=1,B=0) = %f, want 1.0", got)
	}
	// A=0, B=1: P(Y=1)=1
	if got := factor.GetValue(map[string]int{"Y": 1, "A": 0, "B": 1}); !floatEq(got, 1.0) {
		t.Errorf("P(Y=1|A=0,B=1) = %f, want 1.0", got)
	}
	// A=1, B=1: P(Y=1)=1
	if got := factor.GetValue(map[string]int{"Y": 1, "A": 1, "B": 1}); !floatEq(got, 1.0) {
		t.Errorf("P(Y=1|A=1,B=1) = %f, want 1.0", got)
	}
}

func TestNoisyOR_MonotonicProperty(t *testing.T) {
	// P(Y=1) should be monotonically non-decreasing as more parents are active.
	n, err := NewNoisyOR("Y", 2, []string{"A", "B", "C"}, []float64{0.3, 0.4, 0.5}, 0.8)
	if err != nil {
		t.Fatal(err)
	}
	cpd, err := n.ToTabularCPD()
	if err != nil {
		t.Fatal(err)
	}
	factor := cpd.ToFactor()

	pNone := factor.GetValue(map[string]int{"Y": 1, "A": 0, "B": 0, "C": 0})
	pA := factor.GetValue(map[string]int{"Y": 1, "A": 1, "B": 0, "C": 0})
	pAB := factor.GetValue(map[string]int{"Y": 1, "A": 1, "B": 1, "C": 0})
	pABC := factor.GetValue(map[string]int{"Y": 1, "A": 1, "B": 1, "C": 1})

	if pA < pNone-epsilon {
		t.Errorf("P(Y=1|A=1) = %f < P(Y=1|none) = %f", pA, pNone)
	}
	if pAB < pA-epsilon {
		t.Errorf("P(Y=1|A=1,B=1) = %f < P(Y=1|A=1) = %f", pAB, pA)
	}
	if pABC < pAB-epsilon {
		t.Errorf("P(Y=1|A=1,B=1,C=1) = %f < P(Y=1|A=1,B=1) = %f", pABC, pAB)
	}
}

// Suppress unused import warning for math.
var _ = math.Abs
