package scigo

import "math"

// GammaFunc returns the gamma function of x.
// Uses math.Gamma from the standard library.
func GammaFunc(x float64) float64 {
	return math.Gamma(x)
}

// BetaFunc returns the beta function B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b).
func BetaFunc(a, b float64) float64 {
	return math.Exp(Betaln(a, b))
}

// Psi is an alias for the digamma function.
// psi(x) = d/dx ln(Gamma(x)) = Gamma'(x)/Gamma(x).
func Psi(x float64) float64 {
	return Digamma(x)
}

// Polygamma computes the nth derivative of the digamma function.
// polygamma(0, x) = digamma(x), polygamma(n, x) = d^n/dx^n digamma(x).
// For n >= 1, uses the series representation:
// psi^(n)(x) = (-1)^(n+1) * n! * sum_{k=0}^{inf} 1/(x+k)^(n+1)
func Polygamma(n int, x float64) float64 {
	if n < 0 {
		return math.NaN()
	}
	if n == 0 {
		return Digamma(x)
	}
	if x <= 0 && x == math.Floor(x) {
		return math.NaN() // poles at non-positive integers
	}

	// Use recurrence to shift x to a large value for convergence
	result := 0.0
	sign := 1.0
	if n%2 == 0 {
		sign = -1.0
	}
	// Shift x up for better series convergence
	for x < 10 {
		// psi^(n)(x) = psi^(n)(x+1) + (-1)^(n+1) * n! / x^(n+1)
		result += sign * Factorial(n) / math.Pow(x, float64(n+1))
		x++
	}

	// Asymptotic expansion for large x:
	// psi^(n)(x) ~ (-1)^(n+1) * [(n-1)!/x^n + n!/(2*x^(n+1)) + sum B_{2k} * prod_{j=0}^{2k-1}(n+j) / ((2k)! * x^(n+2k))]
	// Simplified: use the first few terms
	fn := float64(n)
	xn := math.Pow(x, fn)
	xn1 := xn * x

	// Leading terms of the asymptotic expansion
	val := sign * (Factorial(n-1)/xn + Factorial(n)/(2*xn1))

	// Bernoulli number terms
	// B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730
	bernoulli := []float64{1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730}
	for k := 1; k <= 6; k++ {
		// coefficient: prod_{j=0}^{2k-1}(n+j) / (2k)!
		prod := 1.0
		for j := 0; j < 2*k; j++ {
			prod *= fn + float64(j)
		}
		prod /= Factorial(2 * k)
		xpow := math.Pow(x, fn+float64(2*k))
		val += sign * bernoulli[k-1] * prod / xpow
	}

	result += val
	return result
}

// bernoulliNumber returns the nth Bernoulli number for small n.
// B_0=1, B_1=-1/2, B_2=1/6, B_3=0, B_4=-1/30, ...
func bernoulliNumber(n int) float64 {
	// Precomputed Bernoulli numbers B_0 through B_20
	bn := map[int]float64{
		0: 1, 1: -0.5, 2: 1.0 / 6, 3: 0, 4: -1.0 / 30,
		5: 0, 6: 1.0 / 42, 7: 0, 8: -1.0 / 30,
		9: 0, 10: 5.0 / 66, 11: 0, 12: -691.0 / 2730,
		13: 0, 14: 7.0 / 6, 15: 0, 16: -3617.0 / 510,
		17: 0, 18: 43867.0 / 798, 19: 0, 20: -174611.0 / 330,
	}
	if v, ok := bn[n]; ok {
		return v
	}
	return 0
}

// zetaBorwein computes zeta(s) using Borwein's method for the Dirichlet eta function.
// Works for all s != 1. Particularly useful for 0 < s < 1 where Euler-Maclaurin needs s > 1.
func zetaBorwein(s float64) float64 {
	const n = 30
	// Compute Borwein d_k coefficients:
	// d_k = n * sum_{i=0}^{k} (n+i-1)! * 4^i / ((n-i)! * (2i)!)
	d := make([]float64, n+1)
	for k := 0; k <= n; k++ {
		sum := 0.0
		for i := 0; i <= k; i++ {
			// (n+i-1)! * 4^i / ((n-i)! * (2i)!)
			// Use lgamma for numerical stability
			lnum, _ := math.Lgamma(float64(n + i))      // (n+i-1)!
			lden1, _ := math.Lgamma(float64(n - i + 1)) // (n-i)!
			lden2, _ := math.Lgamma(float64(2*i + 1))   // (2i)!
			term := math.Exp(lnum - lden1 - lden2 + float64(i)*math.Ln2*2)
			sum += term
		}
		d[k] = float64(n) * sum
	}
	dn := d[n]

	// eta(s) = -1/d_n * sum_{k=0}^{n-1} (-1)^k * (d_k - d_n) / (k+1)^s
	eta := 0.0
	for k := 0; k < n; k++ {
		sign := 1.0
		if k%2 != 0 {
			sign = -1.0
		}
		eta += sign * (d[k] - dn) / math.Pow(float64(k+1), s)
	}
	eta = -eta / dn

	// zeta(s) = eta(s) / (1 - 2^(1-s))
	return eta / (1 - math.Pow(2, 1-s))
}

// Zeta computes the Riemann zeta function for real s.
// zeta(s) = sum_{k=1}^{inf} 1/k^s for s > 1.
// For s <= 1, uses analytic continuation via the reflection formula.
// Uses the Euler-Maclaurin formula for efficient computation when s > 1.
func Zeta(s float64) float64 {
	if s == 1 {
		return math.Inf(1)
	}

	// Handle s = 0: known value zeta(0) = -1/2
	if s == 0 {
		return -0.5
	}

	// Handle negative integers: zeta(-n) = (-1)^n * B_{n+1} / (n+1)
	if s < 0 && s == math.Floor(s) {
		n := int(-s)
		if n <= 19 { // we have Bernoulli numbers up to B_20
			sign := 1.0
			if n%2 != 0 {
				sign = -1.0
			}
			return sign * bernoulliNumber(n+1) / float64(n+1)
		}
	}

	// For s < 0 (non-integer), use the reflection formula:
	// zeta(s) = 2^s * pi^(s-1) * sin(pi*s/2) * Gamma(1-s) * zeta(1-s)
	// This maps s < 0 to 1-s > 1, which the Euler-Maclaurin code handles.
	if s < 0 {
		return math.Pow(2, s) * math.Pow(math.Pi, s-1) *
			math.Sin(math.Pi*s/2) * math.Gamma(1-s) * Zeta(1-s)
	}

	// For 0 < s < 1, use the Dirichlet eta function with Borwein acceleration:
	// zeta(s) = eta(s) / (1 - 2^(1-s))
	// where eta(s) = sum_{k=0}^{n-1} (-1)^k * d_k / (k+1)^s / d_n
	// with Borwein's d_k coefficients.
	if s < 1 {
		return zetaBorwein(s)
	}

	// Direct summation up to N, then Euler-Maclaurin correction for the tail.
	// The sum_{k=1}^{inf} k^{-s} = sum_{k=1}^{N-1} k^{-s} + sum_{k=N}^{inf} k^{-s}.
	// Euler-Maclaurin for sum_{k=N}^{inf} k^{-s}:
	//   = integral_N^inf x^{-s} dx + 1/2*N^{-s} + sum B_{2k}/(2k)! * prod_{j=0}^{2k-2}(s+j) * N^{-(s+2k-1)}
	const N = 200
	sum := 0.0
	for k := 1; k < N; k++ {
		sum += math.Pow(float64(k), -s)
	}

	fN := float64(N)
	// Integral from N to infinity of x^{-s} dx = N^{1-s}/(s-1)
	sum += math.Pow(fN, 1-s) / (s - 1)
	// 1/2 * N^{-s}
	sum += 0.5 * math.Pow(fN, -s)

	// Bernoulli correction terms: B_{2k}/(2k)! * falling_factorial(s, 2k-1) * N^{-(s+2k-1)}
	// B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66
	type bterm struct {
		b float64
		k int
	}
	bterms := []bterm{
		{1.0 / 6, 1},
		{-1.0 / 30, 2},
		{1.0 / 42, 3},
		{-1.0 / 30, 4},
		{5.0 / 66, 5},
		{-691.0 / 2730, 6},
	}
	for _, bt := range bterms {
		k2 := 2 * bt.k
		// falling factorial of s for 2k-1 terms: s*(s+1)*...*(s+2k-2)
		ff := 1.0
		for j := 0; j < k2-1; j++ {
			ff *= s + float64(j)
		}
		fac := 1.0
		for j := 2; j <= k2; j++ {
			fac *= float64(j)
		}
		sum += bt.b / fac * ff * math.Pow(fN, -(s+float64(k2-1)))
	}

	return sum
}

// I0 computes the modified Bessel function of the first kind, order 0.
// This is an exported wrapper around the internal besselI0 implementation.
func I0(x float64) float64 {
	return besselI0(x)
}

// I1 computes the modified Bessel function of the first kind, order 1.
// This is an exported wrapper around the internal besselI1 implementation.
func I1(x float64) float64 {
	return besselI1(x)
}

// Logit computes the logit function: logit(p) = log(p / (1-p)).
// Defined for p in (0, 1). Returns -Inf for p=0, +Inf for p=1, NaN outside [0,1].
func Logit(p float64) float64 {
	if p < 0 || p > 1 {
		return math.NaN()
	}
	if p == 0 {
		return math.Inf(-1)
	}
	if p == 1 {
		return math.Inf(1)
	}
	return math.Log(p / (1 - p))
}

// Expit computes the expit (logistic sigmoid) function: expit(x) = 1 / (1 + exp(-x)).
// This is the inverse of the logit function.
func Expit(x float64) float64 {
	if x >= 0 {
		return 1.0 / (1.0 + math.Exp(-x))
	}
	// For numerical stability with large negative x
	ex := math.Exp(x)
	return ex / (1.0 + ex)
}
