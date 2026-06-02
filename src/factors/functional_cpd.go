package factors

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
)

// FunctionalCPD represents a conditional probability distribution defined by
// an arbitrary function. The function takes parent variable values and returns
// a probability distribution over the child variable's states.
type FunctionalCPD struct {
	variable string
	evidence []string
	fn       func(parentValues map[string]float64) []float64
}

// NewFunctionalCPD creates a new FunctionalCPD.
// fn returns a probability distribution over the variable's states given
// parent values. fn must not be nil.
func NewFunctionalCPD(variable string, evidence []string, fn func(parentValues map[string]float64) []float64) (*FunctionalCPD, error) {
	if fn == nil {
		return nil, fmt.Errorf("factors: FunctionalCPD function must not be nil")
	}

	ev := make([]string, len(evidence))
	copy(ev, evidence)

	return &FunctionalCPD{
		variable: variable,
		evidence: ev,
		fn:       fn,
	}, nil
}

// Variable returns the child variable name.
func (cpd *FunctionalCPD) Variable() string {
	return cpd.variable
}

// Evidence returns a copy of the evidence variable names.
func (cpd *FunctionalCPD) Evidence() []string {
	ev := make([]string, len(cpd.evidence))
	copy(ev, cpd.evidence)
	return ev
}

// GetDistribution returns the probability distribution over the variable's
// states given the parent values.
func (cpd *FunctionalCPD) GetDistribution(parentValues map[string]float64) []float64 {
	return cpd.fn(parentValues)
}

// Sample draws a sample from the conditional distribution given parent values.
// It returns the index of the sampled state. The RNG is used to generate the
// random number for sampling.
func (cpd *FunctionalCPD) Sample(parentValues map[string]float64, rng *numgo.RNG) int {
	dist := cpd.fn(parentValues)
	// Draw a uniform random number.
	u := rng.Rand(1).Data()[0]
	cumulative := 0.0
	for i, p := range dist {
		cumulative += p
		if u < cumulative {
			return i
		}
	}
	// Return the last index in case of floating-point rounding.
	return len(dist) - 1
}

// Validate checks that the FunctionalCPD is properly configured.
func (cpd *FunctionalCPD) Validate() error {
	if cpd.fn == nil {
		return fmt.Errorf("factors: FunctionalCPD function must not be nil")
	}
	return nil
}

// Copy returns a deep copy of the FunctionalCPD.
// Note: the function reference is shared since functions cannot be deep-copied.
func (cpd *FunctionalCPD) Copy() *FunctionalCPD {
	ev := make([]string, len(cpd.evidence))
	copy(ev, cpd.evidence)
	return &FunctionalCPD{
		variable: cpd.variable,
		evidence: ev,
		fn:       cpd.fn,
	}
}
