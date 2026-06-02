//go:build unit

package models

import (
	"math"
	"testing"
)

func TestNewMarkovChain(t *testing.T) {
	tm := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc, err := NewMarkovChain(tm, []string{"sunny", "rainy"})
	if err != nil {
		t.Fatalf("NewMarkovChain: %v", err)
	}
	if mc.NumStates() != 2 {
		t.Errorf("expected 2 states, got %d", mc.NumStates())
	}
	names := mc.StateNames()
	if names[0] != "sunny" || names[1] != "rainy" {
		t.Errorf("expected [sunny rainy], got %v", names)
	}
}

func TestNewMarkovChainNilNames(t *testing.T) {
	tm := [][]float64{
		{0.5, 0.5},
		{0.5, 0.5},
	}
	mc, err := NewMarkovChain(tm, nil)
	if err != nil {
		t.Fatalf("NewMarkovChain: %v", err)
	}
	if mc.StateNames() != nil {
		t.Errorf("expected nil state names, got %v", mc.StateNames())
	}
}

func TestNewMarkovChainErrors(t *testing.T) {
	// Empty matrix.
	if _, err := NewMarkovChain(nil, nil); err == nil {
		t.Error("expected error for empty matrix")
	}

	// Non-square matrix.
	if _, err := NewMarkovChain([][]float64{{0.5, 0.5, 0.0}}, nil); err == nil {
		t.Error("expected error for non-square matrix")
	}

	// Row doesn't sum to 1.
	if _, err := NewMarkovChain([][]float64{{0.3, 0.3}}, nil); err == nil {
		t.Error("expected error for row not summing to 1")
	}

	// Negative value.
	if _, err := NewMarkovChain([][]float64{{1.5, -0.5}}, nil); err == nil {
		t.Error("expected error for negative value")
	}

	// Wrong number of state names.
	if _, err := NewMarkovChain([][]float64{{0.5, 0.5}, {0.5, 0.5}}, []string{"a"}); err == nil {
		t.Error("expected error for wrong number of state names")
	}
}

func TestMarkovChainTransitionMatrix(t *testing.T) {
	tm := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc, _ := NewMarkovChain(tm, nil)

	got := mc.TransitionMatrix()
	for i := range tm {
		for j := range tm[i] {
			if got[i][j] != tm[i][j] {
				t.Errorf("T[%d][%d]: expected %f, got %f", i, j, tm[i][j], got[i][j])
			}
		}
	}

	// Ensure it's a deep copy.
	got[0][0] = 999
	got2 := mc.TransitionMatrix()
	if got2[0][0] == 999 {
		t.Error("TransitionMatrix did not return a deep copy")
	}
}

func TestMarkovChainStationaryDistribution(t *testing.T) {
	// Known stationary: for T = [[0.7, 0.3], [0.4, 0.6]],
	// pi = [4/7, 3/7] approximately [0.5714, 0.4286]
	tm := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc, _ := NewMarkovChain(tm, nil)

	pi, err := mc.StationaryDistribution()
	if err != nil {
		t.Fatalf("StationaryDistribution: %v", err)
	}

	expected := []float64{4.0 / 7.0, 3.0 / 7.0}
	for i := range expected {
		if math.Abs(pi[i]-expected[i]) > 1e-6 {
			t.Errorf("pi[%d]: expected %f, got %f", i, expected[i], pi[i])
		}
	}
}

func TestMarkovChainStationaryDistributionUniform(t *testing.T) {
	// Doubly stochastic matrix -> uniform stationary distribution.
	tm := [][]float64{
		{0.5, 0.5},
		{0.5, 0.5},
	}
	mc, _ := NewMarkovChain(tm, nil)

	pi, err := mc.StationaryDistribution()
	if err != nil {
		t.Fatalf("StationaryDistribution: %v", err)
	}

	for i := range pi {
		if math.Abs(pi[i]-0.5) > 1e-6 {
			t.Errorf("pi[%d]: expected 0.5, got %f", i, pi[i])
		}
	}
}

func TestMarkovChainSample(t *testing.T) {
	tm := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc, _ := NewMarkovChain(tm, nil)

	samples, err := mc.Sample(100, 0, 42)
	if err != nil {
		t.Fatalf("Sample: %v", err)
	}

	if len(samples) != 100 {
		t.Fatalf("expected 100 samples, got %d", len(samples))
	}
	if samples[0] != 0 {
		t.Errorf("first sample should be start state 0, got %d", samples[0])
	}

	// All samples should be valid states.
	for i, s := range samples {
		if s < 0 || s >= 2 {
			t.Errorf("sample[%d] = %d, out of range [0, 2)", i, s)
		}
	}
}

func TestMarkovChainSampleDeterministic(t *testing.T) {
	tm := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc, _ := NewMarkovChain(tm, nil)

	s1, _ := mc.Sample(50, 0, 42)
	s2, _ := mc.Sample(50, 0, 42)

	for i := range s1 {
		if s1[i] != s2[i] {
			t.Errorf("sample[%d] differs: %d vs %d", i, s1[i], s2[i])
		}
	}
}

func TestMarkovChainSampleErrors(t *testing.T) {
	tm := [][]float64{{0.5, 0.5}, {0.5, 0.5}}
	mc, _ := NewMarkovChain(tm, nil)

	if _, err := mc.Sample(0, 0, 42); err == nil {
		t.Error("expected error for n=0")
	}
	if _, err := mc.Sample(-1, 0, 42); err == nil {
		t.Error("expected error for negative n")
	}
	if _, err := mc.Sample(10, -1, 42); err == nil {
		t.Error("expected error for negative start state")
	}
	if _, err := mc.Sample(10, 5, 42); err == nil {
		t.Error("expected error for start state out of range")
	}
}

func TestMarkovChainIsAbsorbing(t *testing.T) {
	// Absorbing chain: state 1 is absorbing.
	tm := [][]float64{
		{0.5, 0.5},
		{0.0, 1.0},
	}
	mc, _ := NewMarkovChain(tm, nil)
	if !mc.IsAbsorbing() {
		t.Error("expected IsAbsorbing=true")
	}

	// Non-absorbing chain.
	tm2 := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc2, _ := NewMarkovChain(tm2, nil)
	if mc2.IsAbsorbing() {
		t.Error("expected IsAbsorbing=false")
	}
}

func TestMarkovChainIsErgodic(t *testing.T) {
	// Ergodic chain (irreducible and aperiodic).
	tm := [][]float64{
		{0.7, 0.3},
		{0.4, 0.6},
	}
	mc, _ := NewMarkovChain(tm, nil)
	if !mc.IsErgodic() {
		t.Error("expected IsErgodic=true")
	}

	// Not irreducible: state 1 is absorbing, so state 0 can't be reached from state 1.
	tm2 := [][]float64{
		{0.5, 0.5},
		{0.0, 1.0},
	}
	mc2, _ := NewMarkovChain(tm2, nil)
	if mc2.IsErgodic() {
		t.Error("expected IsErgodic=false for absorbing chain")
	}
}

func TestMarkovChainIsErgodicPeriodic(t *testing.T) {
	// Periodic chain: deterministic alternation between states.
	tm := [][]float64{
		{0.0, 1.0},
		{1.0, 0.0},
	}
	mc, _ := NewMarkovChain(tm, nil)
	if mc.IsErgodic() {
		t.Error("expected IsErgodic=false for periodic chain")
	}
}

func TestMarkovChainThreeStates(t *testing.T) {
	tm := [][]float64{
		{0.1, 0.6, 0.3},
		{0.4, 0.2, 0.4},
		{0.3, 0.3, 0.4},
	}
	mc, err := NewMarkovChain(tm, []string{"A", "B", "C"})
	if err != nil {
		t.Fatalf("NewMarkovChain: %v", err)
	}

	if mc.NumStates() != 3 {
		t.Errorf("expected 3 states, got %d", mc.NumStates())
	}

	pi, err := mc.StationaryDistribution()
	if err != nil {
		t.Fatalf("StationaryDistribution: %v", err)
	}

	// Stationary distribution should sum to 1.
	sum := 0.0
	for _, p := range pi {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("stationary distribution sums to %f, expected 1.0", sum)
	}

	if !mc.IsErgodic() {
		t.Error("expected 3-state chain to be ergodic")
	}
}
