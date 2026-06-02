package factors

import (
	"fmt"
	"math"
)

const jpdTolerance = 1e-9

// JointProbabilityDistribution represents a joint probability distribution
// over discrete random variables. It wraps a DiscreteFactor and enforces
// the constraint that values must be non-negative and sum to 1.
type JointProbabilityDistribution struct {
	*DiscreteFactor
}

// NewJointProbabilityDistribution creates a new JointProbabilityDistribution.
// It validates that all values are non-negative and sum to 1.0 (within tolerance).
func NewJointProbabilityDistribution(variables []string, cardinality []int, values []float64) (*JointProbabilityDistribution, error) {
	df, err := NewDiscreteFactor(variables, cardinality, values)
	if err != nil {
		return nil, err
	}
	jpd := &JointProbabilityDistribution{DiscreteFactor: df}
	if err := jpd.Validate(); err != nil {
		return nil, err
	}
	return jpd, nil
}

// Validate checks that the distribution values are non-negative and sum to 1.0.
func (j *JointProbabilityDistribution) Validate() error {
	data := j.Values().Data()
	sum := 0.0
	for i, v := range data {
		if v < 0 {
			return fmt.Errorf("factors: negative probability %f at index %d", v, i)
		}
		sum += v
	}
	if math.Abs(sum-1.0) > jpdTolerance {
		return fmt.Errorf("factors: probabilities sum to %f, expected 1.0", sum)
	}
	return nil
}

// MarginalDistribution marginalizes out all variables not in the given list,
// returning a new JointProbabilityDistribution over the specified variables.
func (j *JointProbabilityDistribution) MarginalDistribution(variables []string) (*JointProbabilityDistribution, error) {
	if len(variables) == 0 {
		return nil, fmt.Errorf("factors: must specify at least one variable to keep")
	}

	// Determine which variables to marginalize out.
	keepSet := make(map[string]bool, len(variables))
	for _, v := range variables {
		if j.varIndex(v) == -1 {
			return nil, fmt.Errorf("factors: variable %q not in distribution", v)
		}
		keepSet[v] = true
	}

	var margVars []string
	for _, v := range j.variables {
		if !keepSet[v] {
			margVars = append(margVars, v)
		}
	}

	if len(margVars) == 0 {
		return j.Copy(), nil
	}

	df, err := j.DiscreteFactor.Marginalize(margVars)
	if err != nil {
		return nil, err
	}

	return &JointProbabilityDistribution{DiscreteFactor: df}, nil
}

// ConditionalDistribution computes P(variables | evidence) by reducing
// on the evidence, then normalizing the result.
func (j *JointProbabilityDistribution) ConditionalDistribution(variables []string, evidence map[string]int) (*DiscreteFactor, error) {
	if len(variables) == 0 {
		return nil, fmt.Errorf("factors: must specify at least one query variable")
	}
	if len(evidence) == 0 {
		return nil, fmt.Errorf("factors: must specify at least one evidence variable")
	}

	// Validate query variables exist and are not in evidence.
	for _, v := range variables {
		if j.varIndex(v) == -1 {
			return nil, fmt.Errorf("factors: query variable %q not in distribution", v)
		}
		if _, ok := evidence[v]; ok {
			return nil, fmt.Errorf("factors: variable %q cannot be both query and evidence", v)
		}
	}

	// First marginalize to keep only query variables + evidence variables.
	keepSet := make(map[string]bool)
	for _, v := range variables {
		keepSet[v] = true
	}
	for v := range evidence {
		keepSet[v] = true
	}
	var margVars []string
	for _, v := range j.variables {
		if !keepSet[v] {
			margVars = append(margVars, v)
		}
	}

	var working *DiscreteFactor
	if len(margVars) > 0 {
		var err error
		working, err = j.DiscreteFactor.Marginalize(margVars)
		if err != nil {
			return nil, err
		}
	} else {
		working = j.DiscreteFactor.Copy()
	}

	// Reduce on evidence.
	reduced, err := working.Reduce(evidence)
	if err != nil {
		return nil, err
	}

	// Normalize so the result sums to 1.
	reduced.Normalize()
	return reduced, nil
}

// CheckIndependence tests whether var1 is conditionally independent of var2
// given the variables in 'given', i.e. var1 _|_ var2 | given.
//
// It checks this by comparing P(var1, var2 | given) with
// P(var1 | given) * P(var2 | given) for all value combinations, using the
// supplied absolute tolerance atol.
func (j *JointProbabilityDistribution) CheckIndependence(var1, var2 string, given []string, atol float64) bool {
	if j.varIndex(var1) == -1 || j.varIndex(var2) == -1 {
		return false
	}
	for _, v := range given {
		if j.varIndex(v) == -1 {
			return false
		}
	}

	// Get cardinalities for the given variables.
	givenCards := make([]int, len(given))
	for i, v := range given {
		givenCards[i] = j.cardinality[j.varIndex(v)]
	}

	// Get cardinalities for var1 and var2.
	card1 := j.cardinality[j.varIndex(var1)]
	card2 := j.cardinality[j.varIndex(var2)]

	// Iterate over all assignments of given variables.
	givenSize := 1
	for _, c := range givenCards {
		givenSize *= c
	}

	for gFlat := 0; gFlat < givenSize; gFlat++ {
		evidence := make(map[string]int)
		rem := gFlat
		for i := len(given) - 1; i >= 0; i-- {
			evidence[given[i]] = rem % givenCards[i]
			rem /= givenCards[i]
		}

		// Get the joint factor over var1, var2 (conditioned on given if any).
		var jointFactor *DiscreteFactor
		if len(evidence) == 0 {
			// Unconditional case: marginalize to get P(var1, var2).
			mJoint, err := j.MarginalDistribution([]string{var1, var2})
			if err != nil {
				return false
			}
			jointFactor = mJoint.DiscreteFactor
		} else {
			var err error
			jointFactor, err = j.ConditionalDistribution([]string{var1, var2}, evidence)
			if err != nil {
				return false
			}
		}

		// Get P(var1 | given) and P(var2 | given) by marginalizing the joint.
		marg1, err := jointFactor.Marginalize([]string{var2})
		if err != nil {
			return false
		}
		marg2, err := jointFactor.Marginalize([]string{var1})
		if err != nil {
			return false
		}

		// Compare P(var1, var2 | given) with P(var1 | given) * P(var2 | given)
		for v1 := 0; v1 < card1; v1++ {
			for v2 := 0; v2 < card2; v2++ {
				jointAssign := map[string]int{var1: v1, var2: v2}
				pJoint := jointFactor.GetValue(jointAssign)

				p1 := marg1.GetValue(map[string]int{var1: v1})
				p2 := marg2.GetValue(map[string]int{var2: v2})

				if math.Abs(pJoint-p1*p2) > atol {
					return false
				}
			}
		}
	}

	return true
}

// Copy returns a deep copy of the JointProbabilityDistribution.
func (j *JointProbabilityDistribution) Copy() *JointProbabilityDistribution {
	return &JointProbabilityDistribution{
		DiscreteFactor: j.DiscreteFactor.Copy(),
	}
}
