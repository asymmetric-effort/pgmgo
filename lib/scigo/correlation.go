package scigo

import "math"

// PearsonCorrelation computes the Pearson product-moment correlation coefficient
// between x and y, and its two-tailed p-value using the t-distribution approximation:
// t = r * sqrt((n-2) / (1 - r^2)), with df = n-2.
// Panics if x and y have different lengths or fewer than 3 elements.
func PearsonCorrelation(x, y []float64) (r, pvalue float64) {
	n := len(x)
	if n != len(y) {
		panic("scigo: PearsonCorrelation: x and y must have the same length")
	}
	if n < 3 {
		panic("scigo: PearsonCorrelation: need at least 3 data points")
	}

	// Compute means
	mx, my := 0.0, 0.0
	for i := range x {
		mx += x[i]
		my += y[i]
	}
	mx /= float64(n)
	my /= float64(n)

	// Compute correlation
	var sxy, sxx, syy float64
	for i := range x {
		dx := x[i] - mx
		dy := y[i] - my
		sxy += dx * dy
		sxx += dx * dx
		syy += dy * dy
	}

	if sxx == 0 || syy == 0 {
		// Constant variable: correlation undefined
		return 0, 1
	}

	r = sxy / math.Sqrt(sxx*syy)

	// Clamp r to [-1, 1] for numerical safety
	if r > 1 {
		r = 1
	} else if r < -1 {
		r = -1
	}

	// Compute p-value via t-distribution
	if math.Abs(r) >= 1 {
		pvalue = 0
		return
	}
	df := float64(n - 2)
	tstat := r * math.Sqrt(df/(1-r*r))
	tdist := NewTDistribution(df)
	// Two-tailed p-value
	pvalue = 2 * tdist.SurvivalFunction(math.Abs(tstat))
	return
}

// PartialCorrelation computes the partial correlation between columns x and y
// of the data matrix, controlling for the columns listed in z.
// data is row-major: data[i] is the i-th observation (row), each of equal length.
// Uses the recursive formula: r(x,y|z) = (r(x,y|z\last) - r(x,last|z\last)*r(y,last|z\last)) /
//
//	sqrt((1 - r(x,last|z\last)^2) * (1 - r(y,last|z\last)^2))
//
// The p-value uses the t-distribution with df = n - 2 - len(z).
// Panics if data has inconsistent row lengths or invalid column indices.
func PartialCorrelation(data [][]float64, x, y int, z []int) (r, pvalue float64) {
	n := len(data)
	if n < 3 {
		panic("scigo: PartialCorrelation: need at least 3 observations")
	}
	cols := len(data[0])
	for i := range data {
		if len(data[i]) != cols {
			panic("scigo: PartialCorrelation: inconsistent row lengths")
		}
	}
	if x < 0 || x >= cols || y < 0 || y >= cols {
		panic("scigo: PartialCorrelation: column index out of range")
	}
	for _, zi := range z {
		if zi < 0 || zi >= cols {
			panic("scigo: PartialCorrelation: conditioning column index out of range")
		}
	}

	// Extract column helper
	col := func(j int) []float64 {
		c := make([]float64, n)
		for i := range data {
			c[i] = data[i][j]
		}
		return c
	}

	if len(z) == 0 {
		// Base case: just Pearson correlation
		r, pvalue = PearsonCorrelation(col(x), col(y))
		return
	}

	// Recursive case: remove last conditioning variable
	last := z[len(z)-1]
	zRest := z[:len(z)-1]

	rxy, _ := PartialCorrelation(data, x, y, zRest)
	rxz, _ := PartialCorrelation(data, x, last, zRest)
	ryz, _ := PartialCorrelation(data, y, last, zRest)

	denom := math.Sqrt((1 - rxz*rxz) * (1 - ryz*ryz))
	if denom == 0 {
		return 0, 1
	}
	r = (rxy - rxz*ryz) / denom

	// Clamp
	if r > 1 {
		r = 1
	} else if r < -1 {
		r = -1
	}

	// P-value with df = n - 2 - len(z)
	df := float64(n - 2 - len(z))
	if df < 1 {
		pvalue = math.NaN()
		return
	}
	if math.Abs(r) >= 1 {
		pvalue = 0
		return
	}
	tstat := r * math.Sqrt(df/(1-r*r))
	tdist := NewTDistribution(df)
	pvalue = 2 * tdist.SurvivalFunction(math.Abs(tstat))
	return
}

// FisherZTransform computes Fisher's Z transformation of a correlation coefficient r.
// Z = 0.5 * ln((1+r)/(1-r))
// This transforms the correlation to an approximately normal distribution with
// standard error 1/sqrt(n-3), where n is the sample size.
// The n parameter is not used in the transformation itself but is included
// for completeness; use it externally as SE = 1/sqrt(n-3).
func FisherZTransform(r float64, n int) float64 {
	_ = n // available for the caller's use in computing SE = 1/sqrt(n-3)
	return 0.5 * math.Log((1+r)/(1-r))
}

// PowerDivergenceTest computes the power divergence statistic for goodness of fit.
// The statistic is:
//
//	(2 / (lambda*(lambda+1))) * sum_i( observed_i * ((observed_i/expected_i)^lambda - 1) )
//
// Special cases:
//   - lambda = 1: Pearson chi-squared statistic
//   - lambda = 0: G-test (log-likelihood ratio)
//   - lambda = -1: modified log-likelihood ratio
//   - lambda = -0.5: Freeman-Tukey statistic
//   - lambda = 2/3: Cressie-Read statistic
//
// The p-value is computed using the chi-squared distribution with df = len(observed) - 1.
// Panics if slices have different lengths or fewer than 2 elements.
func PowerDivergenceTest(observed, expected []float64, lambda float64) (statistic, pvalue float64) {
	if len(observed) != len(expected) {
		panic("scigo: PowerDivergenceTest: observed and expected must have the same length")
	}
	if len(observed) < 2 {
		panic("scigo: PowerDivergenceTest: need at least 2 categories")
	}

	const eps = 1e-12

	if math.Abs(lambda) < eps {
		// lambda -> 0: G-test (log-likelihood ratio)
		// statistic = 2 * sum(observed * ln(observed/expected))
		for i := range observed {
			if observed[i] > 0 {
				statistic += observed[i] * math.Log(observed[i]/expected[i])
			}
		}
		statistic *= 2
	} else if math.Abs(lambda+1) < eps {
		// lambda -> -1: modified log-likelihood ratio
		// statistic = 2 * sum(expected * ln(expected/observed))
		for i := range observed {
			if expected[i] > 0 {
				statistic += expected[i] * math.Log(expected[i]/observed[i])
			}
		}
		statistic *= 2
	} else {
		// General case
		for i := range observed {
			if observed[i] > 0 {
				ratio := observed[i] / expected[i]
				statistic += observed[i] * (math.Pow(ratio, lambda) - 1)
			}
		}
		statistic *= 2.0 / (lambda * (lambda + 1))
	}

	df := float64(len(observed) - 1)
	chi2 := NewChiSquared(df)
	pvalue = chi2.SurvivalFunction(statistic)
	return
}
