package factors

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
)

// LinearGaussianCPD represents a linear Gaussian conditional probability
// distribution of the form:
//
//	X = mean + sum(beta_i * Parent_i) + N(0, variance)
//
// where the conditional distribution of X given its parents is:
//
//	X | Parents ~ N(mean + sum(beta_i * Parent_i), variance)
type LinearGaussianCPD struct {
	variable string
	mean     float64
	betas    []float64
	variance float64
	evidence []string
}

// NewLinearGaussianCPD creates a new LinearGaussianCPD.
// It validates that len(betas) == len(evidence) and variance > 0.
func NewLinearGaussianCPD(variable string, mean float64, betas []float64, variance float64, evidence []string) (*LinearGaussianCPD, error) {
	if len(betas) != len(evidence) {
		return nil, fmt.Errorf("factors: betas length %d != evidence length %d", len(betas), len(evidence))
	}
	if variance <= 0 {
		return nil, fmt.Errorf("factors: variance must be positive, got %f", variance)
	}

	b := make([]float64, len(betas))
	copy(b, betas)
	ev := make([]string, len(evidence))
	copy(ev, evidence)

	return &LinearGaussianCPD{
		variable: variable,
		mean:     mean,
		betas:    b,
		variance: variance,
		evidence: ev,
	}, nil
}

// Variable returns the child variable name.
func (cpd *LinearGaussianCPD) Variable() string {
	return cpd.variable
}

// Mean returns the base mean (intercept).
func (cpd *LinearGaussianCPD) Mean() float64 {
	return cpd.mean
}

// Betas returns a copy of the regression coefficients.
func (cpd *LinearGaussianCPD) Betas() []float64 {
	b := make([]float64, len(cpd.betas))
	copy(b, cpd.betas)
	return b
}

// Variance returns the conditional variance.
func (cpd *LinearGaussianCPD) Variance() float64 {
	return cpd.variance
}

// Evidence returns a copy of the parent variable names.
func (cpd *LinearGaussianCPD) Evidence() []string {
	ev := make([]string, len(cpd.evidence))
	copy(ev, cpd.evidence)
	return ev
}

// ConditionalMean computes the mean of the conditional distribution given
// parent values: mean + sum(beta_i * parentValues[evidence_i]).
func (cpd *LinearGaussianCPD) ConditionalMean(parentValues map[string]float64) float64 {
	mu := cpd.mean
	for i, name := range cpd.evidence {
		mu += cpd.betas[i] * parentValues[name]
	}
	return mu
}

// ConditionalVariance returns the conditional variance, which is constant
// in the linear Gaussian model.
func (cpd *LinearGaussianCPD) ConditionalVariance() float64 {
	return cpd.variance
}

// PDF returns the value of the Gaussian probability density function at the
// given value, conditioned on the parent values.
func (cpd *LinearGaussianCPD) PDF(value float64, parentValues map[string]float64) float64 {
	mu := cpd.ConditionalMean(parentValues)
	diff := value - mu
	return math.Exp(-diff*diff/(2*cpd.variance)) / math.Sqrt(2*math.Pi*cpd.variance)
}

// LogPDF returns the log of the Gaussian probability density function at the
// given value, conditioned on the parent values.
func (cpd *LinearGaussianCPD) LogPDF(value float64, parentValues map[string]float64) float64 {
	mu := cpd.ConditionalMean(parentValues)
	diff := value - mu
	return -0.5*math.Log(2*math.Pi*cpd.variance) - diff*diff/(2*cpd.variance)
}

// Sample draws a random sample from N(conditionalMean, variance) given the
// parent values, using the provided RNG.
func (cpd *LinearGaussianCPD) Sample(parentValues map[string]float64, rng *numgo.RNG) float64 {
	mu := cpd.ConditionalMean(parentValues)
	std := math.Sqrt(cpd.variance)
	// Normal returns an NDArray; we draw a single value.
	arr := rng.Normal(mu, std, 1)
	return arr.Data()[0]
}

// Validate checks that the CPD parameters are consistent:
// variance > 0 and len(betas) == len(evidence).
func (cpd *LinearGaussianCPD) Validate() error {
	if cpd.variance <= 0 {
		return fmt.Errorf("factors: variance must be positive, got %f", cpd.variance)
	}
	if len(cpd.betas) != len(cpd.evidence) {
		return fmt.Errorf("factors: betas length %d != evidence length %d", len(cpd.betas), len(cpd.evidence))
	}
	return nil
}

// Copy returns a deep copy of the LinearGaussianCPD.
func (cpd *LinearGaussianCPD) Copy() *LinearGaussianCPD {
	b := make([]float64, len(cpd.betas))
	copy(b, cpd.betas)
	ev := make([]string, len(cpd.evidence))
	copy(ev, cpd.evidence)
	return &LinearGaussianCPD{
		variable: cpd.variable,
		mean:     cpd.mean,
		betas:    b,
		variance: cpd.variance,
		evidence: ev,
	}
}

// GetRandomLinearGaussianCPD generates a random LinearGaussianCPD with a
// random mean in [-5, 5], random betas in [-2, 2], and random variance in
// (0, 5]. The seed controls the RNG.
func GetRandomLinearGaussianCPD(variable string, evidence []string, seed int64) (*LinearGaussianCPD, error) {
	if variable == "" {
		return nil, fmt.Errorf("factors: variable name must not be empty")
	}
	rng := rand.New(rand.NewSource(seed))

	mean := rng.Float64()*10 - 5 // [-5, 5]

	betas := make([]float64, len(evidence))
	for i := range betas {
		betas[i] = rng.Float64()*4 - 2 // [-2, 2]
	}

	variance := rng.Float64()*4.9 + 0.1 // [0.1, 5.0]

	return NewLinearGaussianCPD(variable, mean, betas, variance, evidence)
}

// String returns a human-readable representation of the LinearGaussianCPD.
func (cpd *LinearGaussianCPD) String() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("LinearGaussianCPD(%s", cpd.variable))
	b.WriteString(fmt.Sprintf(" | mean=%.4f", cpd.mean))
	if len(cpd.evidence) > 0 {
		b.WriteString(", betas=[")
		for i, beta := range cpd.betas {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(fmt.Sprintf("%.4f", beta))
		}
		b.WriteString("]")
		b.WriteString(fmt.Sprintf(", evidence=%v", cpd.evidence))
	}
	b.WriteString(fmt.Sprintf(", variance=%.4f)", cpd.variance))
	return b.String()
}
