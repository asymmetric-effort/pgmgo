package factors

import (
	"fmt"
	"math"
)

// NoisyOR represents a Noisy-OR conditional probability distribution.
//
// The Noisy-OR model is a compact representation for binary variables where
// each parent independently has a chance of causing the child to be true.
//
// P(Y=0 | parents) = leakProb * prod(inhibitionProbs[i]^parent_i)
// P(Y=1 | parents) = 1 - P(Y=0 | parents)
//
// The variable must be binary (cardinality 2). Parents must also be binary.
type NoisyOR struct {
	variable        string
	variableCard    int
	parents         []string
	inhibitionProbs []float64
	leakProb        float64
}

// NewNoisyOR creates a new NoisyOR CPD.
// variableCard must be 2 (binary). Each parent is assumed binary (card 2).
// inhibitionProbs must have the same length as parents.
// All probabilities must be in [0, 1].
func NewNoisyOR(variable string, variableCard int, parents []string, inhibitionProbs []float64, leakProb float64) (*NoisyOR, error) {
	if variableCard != 2 {
		return nil, fmt.Errorf("factors: NoisyOR requires variableCard == 2, got %d", variableCard)
	}
	if len(inhibitionProbs) != len(parents) {
		return nil, fmt.Errorf("factors: inhibitionProbs length %d != parents length %d",
			len(inhibitionProbs), len(parents))
	}
	if leakProb < 0 || leakProb > 1 {
		return nil, fmt.Errorf("factors: leakProb must be in [0,1], got %f", leakProb)
	}
	for i, p := range inhibitionProbs {
		if p < 0 || p > 1 {
			return nil, fmt.Errorf("factors: inhibitionProbs[%d] must be in [0,1], got %f", i, p)
		}
	}

	par := make([]string, len(parents))
	copy(par, parents)
	inh := make([]float64, len(inhibitionProbs))
	copy(inh, inhibitionProbs)

	return &NoisyOR{
		variable:        variable,
		variableCard:    variableCard,
		parents:         par,
		inhibitionProbs: inh,
		leakProb:        leakProb,
	}, nil
}

// Variable returns the child variable name.
func (n *NoisyOR) Variable() string {
	return n.variable
}

// VariableCard returns the cardinality of the child variable.
func (n *NoisyOR) VariableCard() int {
	return n.variableCard
}

// Parents returns a copy of the parent variable names.
func (n *NoisyOR) Parents() []string {
	p := make([]string, len(n.parents))
	copy(p, n.parents)
	return p
}

// InhibitionProbs returns a copy of the inhibition probabilities.
func (n *NoisyOR) InhibitionProbs() []float64 {
	p := make([]float64, len(n.inhibitionProbs))
	copy(p, n.inhibitionProbs)
	return p
}

// LeakProb returns the leak probability.
func (n *NoisyOR) LeakProb() float64 {
	return n.leakProb
}

// ToTabularCPD expands the NoisyOR into a full TabularCPD.
//
// The resulting CPD has the child variable (binary) and all parents (each binary).
// For each parent configuration:
//
//	P(Y=0 | parents) = leakProb * prod(inhibitionProbs[i]^parent_i)
//	P(Y=1 | parents) = 1 - P(Y=0 | parents)
func (n *NoisyOR) ToTabularCPD() (*TabularCPD, error) {
	if err := n.Validate(); err != nil {
		return nil, err
	}

	numParents := len(n.parents)
	numParentConfigs := 1
	for i := 0; i < numParents; i++ {
		numParentConfigs *= 2
	}

	// values[childState][parentConfig]
	values := make([][]float64, 2)
	values[0] = make([]float64, numParentConfigs)
	values[1] = make([]float64, numParentConfigs)

	for parentConfig := 0; parentConfig < numParentConfigs; parentConfig++ {
		// Decompose parentConfig into individual parent states.
		// Row-major order: parent 0 varies slowest.
		pY0 := n.leakProb
		rem := parentConfig
		for i := numParents - 1; i >= 0; i-- {
			parentState := rem % 2
			rem /= 2
			if parentState == 1 {
				pY0 *= n.inhibitionProbs[i]
			}
		}
		values[0][parentConfig] = pY0
		values[1][parentConfig] = 1 - pY0
	}

	evidenceCard := make([]int, numParents)
	for i := range evidenceCard {
		evidenceCard[i] = 2
	}

	return NewTabularCPD(n.variable, 2, values, n.parents, evidenceCard)
}

// Validate checks that the NoisyOR parameters are consistent.
func (n *NoisyOR) Validate() error {
	if n.variableCard != 2 {
		return fmt.Errorf("factors: NoisyOR requires variableCard == 2, got %d", n.variableCard)
	}
	if len(n.inhibitionProbs) != len(n.parents) {
		return fmt.Errorf("factors: inhibitionProbs length %d != parents length %d",
			len(n.inhibitionProbs), len(n.parents))
	}
	if n.leakProb < 0 || n.leakProb > 1 {
		return fmt.Errorf("factors: leakProb must be in [0,1], got %f", n.leakProb)
	}
	if math.IsNaN(n.leakProb) {
		return fmt.Errorf("factors: leakProb must not be NaN")
	}
	for i, p := range n.inhibitionProbs {
		if p < 0 || p > 1 {
			return fmt.Errorf("factors: inhibitionProbs[%d] must be in [0,1], got %f", i, p)
		}
		if math.IsNaN(p) {
			return fmt.Errorf("factors: inhibitionProbs[%d] must not be NaN", i)
		}
	}
	return nil
}

// Copy returns a deep copy of the NoisyOR.
func (n *NoisyOR) Copy() *NoisyOR {
	par := make([]string, len(n.parents))
	copy(par, n.parents)
	inh := make([]float64, len(n.inhibitionProbs))
	copy(inh, n.inhibitionProbs)
	return &NoisyOR{
		variable:        n.variable,
		variableCard:    n.variableCard,
		parents:         par,
		inhibitionProbs: inh,
		leakProb:        n.leakProb,
	}
}
