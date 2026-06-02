package scigo

import (
	"math"
	"math/rand"
)

// Distribution defines the interface for probability distributions.
type Distribution interface {
	// PDF returns the probability density function evaluated at x.
	PDF(x float64) float64
	// CDF returns the cumulative distribution function evaluated at x.
	CDF(x float64) float64
	// PPF returns the percent point function (inverse CDF) for probability p in [0,1].
	PPF(p float64) float64
	// LogPDF returns the natural logarithm of the PDF evaluated at x.
	LogPDF(x float64) float64
	// Mean returns the mean of the distribution.
	Mean() float64
	// Var returns the variance of the distribution.
	Var() float64
}

// ---------------------------------------------------------------------------
// Normal (Gaussian) Distribution
// ---------------------------------------------------------------------------

// Normal represents a Gaussian distribution with parameters mu (mean) and sigma (standard deviation).
type Normal struct {
	mu    float64
	sigma float64
}

// NewNormal creates a Normal distribution with the given mean and standard deviation.
// Panics if sigma <= 0.
func NewNormal(mu, sigma float64) *Normal {
	if sigma <= 0 {
		panic("scigo: Normal sigma must be positive")
	}
	return &Normal{mu: mu, sigma: sigma}
}

// PDF returns the probability density of the Normal distribution at x.
func (n *Normal) PDF(x float64) float64 {
	z := (x - n.mu) / n.sigma
	return math.Exp(-0.5*z*z) / (n.sigma * math.Sqrt(2*math.Pi))
}

// CDF returns the cumulative distribution function of the Normal distribution at x.
func (n *Normal) CDF(x float64) float64 {
	z := (x - n.mu) / (n.sigma * math.Sqrt2)
	return 0.5 * (1 + math.Erf(z))
}

// PPF returns the percent point function (inverse CDF) for probability p.
func (n *Normal) PPF(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	// Phi^{-1}(p) = mu + sigma * sqrt(2) * erfinv(2p - 1)
	return n.mu + n.sigma*math.Sqrt2*Erfinv(2*p-1)
}

// LogPDF returns the log of the probability density at x.
func (n *Normal) LogPDF(x float64) float64 {
	z := (x - n.mu) / n.sigma
	return -0.5*z*z - math.Log(n.sigma) - 0.5*math.Log(2*math.Pi)
}

// Mean returns the mean of the Normal distribution.
func (n *Normal) Mean() float64 {
	return n.mu
}

// Var returns the variance of the Normal distribution.
func (n *Normal) Var() float64 {
	return n.sigma * n.sigma
}

// Sample generates n random samples from the Normal distribution.
func (n *Normal) Sample(rng *rand.Rand, count int) []float64 {
	samples := make([]float64, count)
	for i := range samples {
		samples[i] = rng.NormFloat64()*n.sigma + n.mu
	}
	return samples
}

// ---------------------------------------------------------------------------
// Chi-Squared Distribution
// ---------------------------------------------------------------------------

// ChiSquared represents a chi-squared distribution with df degrees of freedom.
type ChiSquared struct {
	df float64
}

// NewChiSquared creates a ChiSquared distribution with the given degrees of freedom.
// Panics if df <= 0.
func NewChiSquared(df float64) *ChiSquared {
	if df <= 0 {
		panic("scigo: ChiSquared df must be positive")
	}
	return &ChiSquared{df: df}
}

// PDF returns the probability density of the chi-squared distribution at x.
func (c *ChiSquared) PDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		if c.df < 2 {
			return math.Inf(1)
		}
		if c.df == 2 {
			return 0.5
		}
		return 0
	}
	k2 := c.df / 2
	return math.Exp((k2-1)*math.Log(x) - x/2 - k2*math.Ln2 - Gammaln(k2))
}

// CDF returns the cumulative distribution function of the chi-squared distribution at x.
// CDF(x) = P(df/2, x/2) where P is the regularized incomplete gamma function.
func (c *ChiSquared) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return RegularizedIncompleteGamma(c.df/2, x/2)
}

// PPF returns the percent point function (inverse CDF) using Newton's method.
func (c *ChiSquared) PPF(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return math.Inf(1)
	}

	// Initial guess using Wilson-Hilferty approximation
	x := c.df * math.Pow(1-2/(9*c.df)+math.Sqrt(2/(9*c.df))*Erfinv(2*p-1)*math.Sqrt2, 3)
	if x <= 0 {
		x = 0.01
	}

	// Newton's method
	for i := 0; i < 50; i++ {
		cdfVal := c.CDF(x)
		pdfVal := c.PDF(x)
		if pdfVal == 0 {
			break
		}
		dx := (cdfVal - p) / pdfVal
		x -= dx
		if x <= 0 {
			x = 1e-10
		}
		if math.Abs(dx) < 1e-12*x {
			break
		}
	}
	return x
}

// LogPDF returns the log of the probability density at x.
func (c *ChiSquared) LogPDF(x float64) float64 {
	if x <= 0 {
		if x == 0 && c.df >= 2 {
			if c.df == 2 {
				return math.Log(0.5)
			}
			return math.Inf(-1)
		}
		return math.Inf(-1)
	}
	k2 := c.df / 2
	return (k2-1)*math.Log(x) - x/2 - k2*math.Ln2 - Gammaln(k2)
}

// Mean returns the mean of the chi-squared distribution.
func (c *ChiSquared) Mean() float64 {
	return c.df
}

// Var returns the variance of the chi-squared distribution.
func (c *ChiSquared) Var() float64 {
	return 2 * c.df
}

// SurvivalFunction returns the survival function (1 - CDF) at x.
// This is the p-value for a chi-squared test statistic.
func (c *ChiSquared) SurvivalFunction(x float64) float64 {
	return 1 - c.CDF(x)
}

// ---------------------------------------------------------------------------
// Student's t-Distribution
// ---------------------------------------------------------------------------

// TDistribution represents a Student's t-distribution with df degrees of freedom.
type TDistribution struct {
	df float64
}

// NewTDistribution creates a TDistribution with the given degrees of freedom.
// Panics if df <= 0.
func NewTDistribution(df float64) *TDistribution {
	if df <= 0 {
		panic("scigo: TDistribution df must be positive")
	}
	return &TDistribution{df: df}
}

// PDF returns the probability density of the t-distribution at x.
func (td *TDistribution) PDF(x float64) float64 {
	v := td.df
	coeff := math.Exp(Gammaln((v+1)/2) - Gammaln(v/2))
	coeff /= math.Sqrt(v * math.Pi)
	return coeff * math.Pow(1+x*x/v, -(v+1)/2)
}

// CDF returns the cumulative distribution function of the t-distribution at x.
// Uses the regularized incomplete beta function: CDF(x) = 1 - 0.5*I(v/(v+x^2); v/2, 1/2) for x >= 0.
func (td *TDistribution) CDF(x float64) float64 {
	v := td.df
	if x == 0 {
		return 0.5
	}
	// Use the relationship: CDF(t) = 1 - 0.5 * I_x(a, b)
	// where x = v/(v+t^2), a = v/2, b = 1/2
	bt := v / (v + x*x)
	ib := RegularizedIncompleteBeta(bt, v/2, 0.5)
	if x >= 0 {
		return 1 - 0.5*ib
	}
	return 0.5 * ib
}

// SurvivalFunction returns the survival function (1 - CDF) at x.
// This is the two-tailed p-value when called as 2*SF(|t|).
func (td *TDistribution) SurvivalFunction(x float64) float64 {
	return 1 - td.CDF(x)
}

// LogPDF returns the natural log of the PDF at x.
func (td *TDistribution) LogPDF(x float64) float64 {
	v := td.df
	return Gammaln((v+1)/2) - Gammaln(v/2) - 0.5*math.Log(v*math.Pi) - (v+1)/2*math.Log(1+x*x/v)
}

// Mean returns the mean of the t-distribution (0 for df > 1, NaN otherwise).
func (td *TDistribution) Mean() float64 {
	if td.df > 1 {
		return 0
	}
	return math.NaN()
}

// Var returns the variance of the t-distribution.
func (td *TDistribution) Var() float64 {
	if td.df > 2 {
		return td.df / (td.df - 2)
	}
	if td.df > 1 {
		return math.Inf(1)
	}
	return math.NaN()
}

// PPF returns the percent point function (inverse CDF) for probability p.
// Uses Newton's method.
func (td *TDistribution) PPF(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	if p == 0.5 {
		return 0
	}

	// Initial guess from normal approximation
	n := NewNormal(0, 1)
	x := n.PPF(p)

	// Newton's method
	for i := 0; i < 50; i++ {
		cdfVal := td.CDF(x)
		pdfVal := td.PDF(x)
		if pdfVal == 0 {
			break
		}
		dx := (cdfVal - p) / pdfVal
		x -= dx
		if math.Abs(dx) < 1e-12*(1+math.Abs(x)) {
			break
		}
	}
	return x
}
