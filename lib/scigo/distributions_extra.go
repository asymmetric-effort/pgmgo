package scigo

import "math"

// ---------------------------------------------------------------------------
// Beta Distribution
// ---------------------------------------------------------------------------

// Beta represents a Beta distribution with shape parameters alpha and beta.
type Beta struct {
	alpha float64
	beta  float64
}

// NewBeta creates a Beta distribution with the given alpha and beta parameters.
// Panics if alpha <= 0 or beta <= 0.
func NewBeta(alpha, beta float64) *Beta {
	if alpha <= 0 {
		panic("scigo: Beta alpha must be positive")
	}
	if beta <= 0 {
		panic("scigo: Beta beta must be positive")
	}
	return &Beta{alpha: alpha, beta: beta}
}

// PDF returns the probability density of the Beta distribution at x.
func (b *Beta) PDF(x float64) float64 {
	if x < 0 || x > 1 {
		return 0
	}
	if x == 0 {
		if b.alpha < 1 {
			return math.Inf(1)
		}
		if b.alpha == 1 {
			return b.alpha // will simplify via log form
		}
		return 0
	}
	if x == 1 {
		if b.beta < 1 {
			return math.Inf(1)
		}
		if b.beta == 1 {
			return b.beta
		}
		return 0
	}
	lbeta := Gammaln(b.alpha) + Gammaln(b.beta) - Gammaln(b.alpha+b.beta)
	return math.Exp((b.alpha-1)*math.Log(x) + (b.beta-1)*math.Log(1-x) - lbeta)
}

// CDF returns the cumulative distribution function of the Beta distribution at x.
func (b *Beta) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	return RegularizedIncompleteBeta(x, b.alpha, b.beta)
}

// PPF returns the percent point function (inverse CDF) for probability p.
// Uses Newton's method.
func (b *Beta) PPF(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return 1
	}

	// Initial guess: use mean as starting point
	x := b.alpha / (b.alpha + b.beta)

	for i := 0; i < 100; i++ {
		cdfVal := b.CDF(x)
		pdfVal := b.PDF(x)
		if pdfVal == 0 {
			break
		}
		dx := (cdfVal - p) / pdfVal
		x -= dx
		if x <= 0 {
			x = 1e-10
		}
		if x >= 1 {
			x = 1 - 1e-10
		}
		if math.Abs(dx) < 1e-12 {
			break
		}
	}
	return x
}

// LogPDF returns the log of the probability density at x.
func (b *Beta) LogPDF(x float64) float64 {
	if x <= 0 || x >= 1 {
		return math.Inf(-1)
	}
	lbeta := Gammaln(b.alpha) + Gammaln(b.beta) - Gammaln(b.alpha+b.beta)
	return (b.alpha-1)*math.Log(x) + (b.beta-1)*math.Log(1-x) - lbeta
}

// Mean returns the mean of the Beta distribution.
func (b *Beta) Mean() float64 {
	return b.alpha / (b.alpha + b.beta)
}

// Var returns the variance of the Beta distribution.
func (b *Beta) Var() float64 {
	ab := b.alpha + b.beta
	return (b.alpha * b.beta) / (ab * ab * (ab + 1))
}

// ---------------------------------------------------------------------------
// Gamma Distribution
// ---------------------------------------------------------------------------

// Gamma represents a Gamma distribution with shape (k) and scale (theta) parameters.
type Gamma struct {
	shape float64
	scale float64
}

// NewGamma creates a Gamma distribution with the given shape and scale parameters.
// Panics if shape <= 0 or scale <= 0.
func NewGamma(shape, scale float64) *Gamma {
	if shape <= 0 {
		panic("scigo: Gamma shape must be positive")
	}
	if scale <= 0 {
		panic("scigo: Gamma scale must be positive")
	}
	return &Gamma{shape: shape, scale: scale}
}

// PDF returns the probability density of the Gamma distribution at x.
func (g *Gamma) PDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		if g.shape < 1 {
			return math.Inf(1)
		}
		if g.shape == 1 {
			return 1.0 / g.scale
		}
		return 0
	}
	return math.Exp((g.shape-1)*math.Log(x) - x/g.scale - g.shape*math.Log(g.scale) - Gammaln(g.shape))
}

// CDF returns the cumulative distribution function of the Gamma distribution at x.
func (g *Gamma) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return RegularizedIncompleteGamma(g.shape, x/g.scale)
}

// LogPDF returns the log of the probability density at x.
func (g *Gamma) LogPDF(x float64) float64 {
	if x <= 0 {
		return math.Inf(-1)
	}
	return (g.shape-1)*math.Log(x) - x/g.scale - g.shape*math.Log(g.scale) - Gammaln(g.shape)
}

// Mean returns the mean of the Gamma distribution.
func (g *Gamma) Mean() float64 {
	return g.shape * g.scale
}

// Var returns the variance of the Gamma distribution.
func (g *Gamma) Var() float64 {
	return g.shape * g.scale * g.scale
}

// ---------------------------------------------------------------------------
// Exponential Distribution
// ---------------------------------------------------------------------------

// Exponential represents an exponential distribution with rate parameter lambda.
type Exponential struct {
	rate float64
}

// NewExponential creates an Exponential distribution with the given rate.
// Panics if rate <= 0.
func NewExponential(rate float64) *Exponential {
	if rate <= 0 {
		panic("scigo: Exponential rate must be positive")
	}
	return &Exponential{rate: rate}
}

// PDF returns the probability density of the Exponential distribution at x.
func (e *Exponential) PDF(x float64) float64 {
	if x < 0 {
		return 0
	}
	return e.rate * math.Exp(-e.rate*x)
}

// CDF returns the cumulative distribution function of the Exponential distribution at x.
func (e *Exponential) CDF(x float64) float64 {
	if x <= 0 {
		return 0
	}
	return 1 - math.Exp(-e.rate*x)
}

// PPF returns the percent point function (inverse CDF) for probability p.
func (e *Exponential) PPF(p float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return math.Inf(1)
	}
	return -math.Log(1-p) / e.rate
}

// LogPDF returns the log of the probability density at x.
func (e *Exponential) LogPDF(x float64) float64 {
	if x < 0 {
		return math.Inf(-1)
	}
	return math.Log(e.rate) - e.rate*x
}

// Mean returns the mean of the Exponential distribution.
func (e *Exponential) Mean() float64 {
	return 1.0 / e.rate
}

// Var returns the variance of the Exponential distribution.
func (e *Exponential) Var() float64 {
	return 1.0 / (e.rate * e.rate)
}

// ---------------------------------------------------------------------------
// Uniform (Continuous) Distribution
// ---------------------------------------------------------------------------

// Uniform represents a continuous uniform distribution on [low, high].
type Uniform struct {
	low  float64
	high float64
}

// NewUniform creates a Uniform distribution on [low, high].
// Panics if low >= high.
func NewUniform(low, high float64) *Uniform {
	if low >= high {
		panic("scigo: Uniform requires low < high")
	}
	return &Uniform{low: low, high: high}
}

// PDF returns the probability density of the Uniform distribution at x.
func (u *Uniform) PDF(x float64) float64 {
	if x < u.low || x > u.high {
		return 0
	}
	return 1.0 / (u.high - u.low)
}

// CDF returns the cumulative distribution function of the Uniform distribution at x.
func (u *Uniform) CDF(x float64) float64 {
	if x <= u.low {
		return 0
	}
	if x >= u.high {
		return 1
	}
	return (x - u.low) / (u.high - u.low)
}

// PPF returns the percent point function (inverse CDF) for probability p.
func (u *Uniform) PPF(p float64) float64 {
	if p <= 0 {
		return u.low
	}
	if p >= 1 {
		return u.high
	}
	return u.low + p*(u.high-u.low)
}

// LogPDF returns the log of the probability density at x.
func (u *Uniform) LogPDF(x float64) float64 {
	if x < u.low || x > u.high {
		return math.Inf(-1)
	}
	return -math.Log(u.high - u.low)
}

// Mean returns the mean of the Uniform distribution.
func (u *Uniform) Mean() float64 {
	return (u.low + u.high) / 2
}

// Var returns the variance of the Uniform distribution.
func (u *Uniform) Var() float64 {
	d := u.high - u.low
	return d * d / 12
}

// ---------------------------------------------------------------------------
// Poisson Distribution (discrete)
// ---------------------------------------------------------------------------

// Poisson represents a Poisson distribution with rate parameter lambda.
type Poisson struct {
	lambda float64
}

// NewPoisson creates a Poisson distribution with the given lambda.
// Panics if lambda <= 0.
func NewPoisson(lambda float64) *Poisson {
	if lambda <= 0 {
		panic("scigo: Poisson lambda must be positive")
	}
	return &Poisson{lambda: lambda}
}

// PMF returns the probability mass function of the Poisson distribution at k.
// Returns 0 for negative or non-integer values.
func (po *Poisson) PMF(k int) float64 {
	if k < 0 {
		return 0
	}
	return math.Exp(float64(k)*math.Log(po.lambda) - po.lambda - Gammaln(float64(k)+1))
}

// CDF returns the cumulative distribution function of the Poisson distribution at k.
// CDF(k) = P(X <= k) = sum_{i=0}^{floor(k)} PMF(i).
func (po *Poisson) CDF(k int) float64 {
	if k < 0 {
		return 0
	}
	// Use the regularized incomplete gamma: P(X <= k) = 1 - P(k+1, lambda)
	// which is the upper incomplete gamma Q(k+1, lambda) = 1 - P(k+1, lambda)
	return 1 - RegularizedIncompleteGamma(float64(k)+1, po.lambda)
}

// Mean returns the mean of the Poisson distribution.
func (po *Poisson) Mean() float64 {
	return po.lambda
}

// Var returns the variance of the Poisson distribution.
func (po *Poisson) Var() float64 {
	return po.lambda
}

// ---------------------------------------------------------------------------
// Binomial Distribution (discrete)
// ---------------------------------------------------------------------------

// Binomial represents a binomial distribution with n trials and success probability p.
type Binomial struct {
	n int
	p float64
}

// NewBinomial creates a Binomial distribution with n trials and probability p.
// Panics if n < 0 or p is not in [0, 1].
func NewBinomial(n int, p float64) *Binomial {
	if n < 0 {
		panic("scigo: Binomial n must be non-negative")
	}
	if p < 0 || p > 1 {
		panic("scigo: Binomial p must be in [0, 1]")
	}
	return &Binomial{n: n, p: p}
}

// PMF returns the probability mass function of the Binomial distribution at k.
func (bi *Binomial) PMF(k int) float64 {
	if k < 0 || k > bi.n {
		return 0
	}
	if bi.p == 0 {
		if k == 0 {
			return 1
		}
		return 0
	}
	if bi.p == 1 {
		if k == bi.n {
			return 1
		}
		return 0
	}
	// Use log-space to avoid overflow: C(n,k) * p^k * (1-p)^(n-k)
	logC := Gammaln(float64(bi.n)+1) - Gammaln(float64(k)+1) - Gammaln(float64(bi.n-k)+1)
	return math.Exp(logC + float64(k)*math.Log(bi.p) + float64(bi.n-k)*math.Log(1-bi.p))
}

// CDF returns the cumulative distribution function of the Binomial distribution at k.
// CDF(k) = P(X <= k) = sum_{i=0}^{k} PMF(i).
func (bi *Binomial) CDF(k int) float64 {
	if k < 0 {
		return 0
	}
	if k >= bi.n {
		return 1
	}
	// Use the regularized incomplete beta: P(X <= k) = I_{1-p}(n-k, k+1)
	return RegularizedIncompleteBeta(1-bi.p, float64(bi.n-k), float64(k)+1)
}

// Mean returns the mean of the Binomial distribution.
func (bi *Binomial) Mean() float64 {
	return float64(bi.n) * bi.p
}

// Var returns the variance of the Binomial distribution.
func (bi *Binomial) Var() float64 {
	return float64(bi.n) * bi.p * (1 - bi.p)
}
