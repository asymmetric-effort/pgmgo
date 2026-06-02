package scigo

import "math"

// Gammaln returns the natural logarithm of the absolute value of the gamma function.
func Gammaln(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// Digamma computes the digamma (psi) function, the logarithmic derivative of the gamma function.
// Uses the asymptotic series expansion after shifting the argument to a large value.
func Digamma(x float64) float64 {
	if math.IsNaN(x) {
		return math.NaN()
	}
	if math.IsInf(x, 1) {
		return math.Inf(1)
	}
	if x <= 0 && x == math.Floor(x) {
		return math.NaN() // poles at non-positive integers
	}

	// Reflection formula for negative x: psi(1-x) - pi*cot(pi*x)
	if x < 0 {
		return Digamma(1-x) - math.Pi/math.Tan(math.Pi*x)
	}

	// Shift x to large value using recurrence: psi(x+1) = psi(x) + 1/x
	result := 0.0
	for x < 7 {
		result -= 1.0 / x
		x++
	}

	// Asymptotic expansion for large x
	// psi(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{} B_{2k}/(2k * x^{2k})
	x2 := 1.0 / (x * x)
	result += math.Log(x) - 0.5/x
	// Bernoulli number coefficients: B2/(2), B4/(4), B6/(6), B8/(8), B10/(10), B12/(12)
	result -= x2 * (1.0/12.0 - x2*(1.0/120.0-x2*(1.0/252.0-x2*(1.0/240.0-x2*(1.0/132.0-x2*691.0/32760.0)))))
	return result
}

// RegularizedIncompleteGamma computes the regularized lower incomplete gamma function P(a, x).
// P(a, x) = gamma(a, x) / Gamma(a) where gamma(a, x) = integral from 0 to x of t^(a-1)*e^(-t) dt.
// Uses series expansion for x < a+1, continued fraction otherwise.
func RegularizedIncompleteGamma(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return math.NaN()
	}
	if x == 0 {
		return 0
	}
	if math.IsInf(x, 1) {
		return 1
	}

	if x < a+1 {
		return regularizedGammaSeries(a, x)
	}
	return 1 - regularizedGammaCF(a, x)
}

// regularizedGammaSeries computes P(a,x) using the series expansion.
func regularizedGammaSeries(a, x float64) float64 {
	lna := Gammaln(a)
	sum := 1.0 / a
	term := 1.0 / a
	for n := 1; n < 200; n++ {
		term *= x / (a + float64(n))
		sum += term
		if math.Abs(term) < math.Abs(sum)*1e-14 {
			break
		}
	}
	return sum * math.Exp(-x+a*math.Log(x)-lna)
}

// regularizedGammaCF computes Q(a,x) = 1-P(a,x) using the continued fraction (Lentz's method).
func regularizedGammaCF(a, x float64) float64 {
	lna := Gammaln(a)
	const tiny = 1e-30
	// Modified Lentz's algorithm
	f := tiny
	c := f
	d := 0.0

	for i := 1; i < 200; i++ {
		var an, bn float64
		if i == 1 {
			an = 1.0
			bn = x + 1 - a
		} else {
			n := float64(i - 1)
			if i%2 == 0 {
				an = n / 2.0 * (a - n/2.0) // even: k*(a-k) where k=n/2... recompute
				k := float64(i/2 - 1)
				an = (k + 1) * (a - k - 1)
				// Actually, let me use the standard CF for Q(a,x):
				// b0 = 0, a1=1, b1=x+1-a
				// for n>=1: a_{2n} = n(a-n), b_{2n} = x+2n+1-a (nope)
				// Simpler: use the CF from Numerical Recipes
			} else {
				_ = n
			}
			_ = an
			_ = bn
		}
		_ = an
		_ = bn
		_ = c
		_ = d
		_ = f
		break
	}

	// Restart with cleaner implementation using standard CF
	// Q(a,x) = e^{-x} x^a / Gamma(a) * CF
	// CF = 1/(x+1-a+ 1*(1-a)/(x+3-a+ 2*(2-a)/(x+5-a+ ...)))
	// Using Lentz's method on the CF: b0=x+1-a, a_i and b_i alternate
	b := x + 1 - a
	c = 1e30
	d = 1.0 / b
	f = d

	for i := 1; i <= 200; i++ {
		an := -float64(i) * (float64(i) - a)
		bn := b + 2*float64(i)
		d = an*d + bn
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = bn + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		delta := c * d
		f *= delta
		if math.Abs(delta-1) < 1e-14 {
			break
		}
	}

	return math.Exp(-x+a*math.Log(x)-lna) * f
}

// Erf returns the error function of x.
func Erf(x float64) float64 {
	return math.Erf(x)
}

// Erfinv returns the inverse of the error function.
// For y in (-1, 1), returns x such that erf(x) = y.
// Uses a rational approximation method.
func Erfinv(y float64) float64 {
	if y <= -1 {
		return math.Inf(-1)
	}
	if y >= 1 {
		return math.Inf(1)
	}
	if y == 0 {
		return 0
	}

	// Use the relationship erfinv(y) = sign(y) * ndtri((1+|y|)/2) / sqrt(2)
	// But implement directly with rational approximation.
	// Based on J.M. Blair's rational approximation approach.

	a := math.Abs(y)

	var x float64
	if a <= 0.7 {
		// Central region: rational approximation in y^2
		z := a * a
		// Coefficients for the rational approximation
		x = a * ((((-0.140543331*z+0.914624893)*z-1.645349621)*z + 0.886226899) /
			((((0.012229801*z-0.329097515)*z+1.442710462)*z-2.118377725)*z + 1))
	} else {
		// Tail region
		z := math.Sqrt(-math.Log((1 - a) / 2))
		x = (((0.012229801*z+0.886226899)*z + 1) /
			((z+0.914624893)*z + 1))
		// Better approximation for the tail
		x = z - math.Log(z)/(2*z) // rough start
	}

	// Polish with Newton-Raphson iterations: erfinv via erf
	// f(x) = erf(x) - y, f'(x) = 2/sqrt(pi) * exp(-x^2)
	twoOverSqrtPi := 2.0 / math.Sqrt(math.Pi)
	sign := 1.0
	if y < 0 {
		sign = -1.0
	}
	x = math.Abs(x)

	for i := 0; i < 10; i++ {
		err := math.Erf(x) - a
		deriv := twoOverSqrtPi * math.Exp(-x*x)
		if deriv == 0 {
			break
		}
		x -= err / deriv
		if math.Abs(err) < 1e-15 {
			break
		}
	}

	return sign * x
}

// Logsumexp computes log(sum(exp(values))) in a numerically stable way.
// Returns -Inf for an empty slice.
func Logsumexp(values []float64) float64 {
	if len(values) == 0 {
		return math.Inf(-1)
	}

	// Find max value for numerical stability
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	if math.IsInf(maxVal, -1) {
		return math.Inf(-1)
	}

	sum := 0.0
	for _, v := range values {
		sum += math.Exp(v - maxVal)
	}

	return maxVal + math.Log(sum)
}

// Betaln returns the natural logarithm of the Beta function: ln(B(a, b)).
// B(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b).
func Betaln(a, b float64) float64 {
	return Gammaln(a) + Gammaln(b) - Gammaln(a+b)
}

// Comb returns the number of combinations C(n, k) = n! / (k! * (n-k)!).
// Uses Gammaln for numerical stability with large values.
// Returns 0 if k < 0 or k > n.
func Comb(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	return math.Exp(Gammaln(float64(n)+1) - Gammaln(float64(k)+1) - Gammaln(float64(n-k)+1))
}

// Factorial returns n! as a float64.
// Uses Gammaln for numerical stability: n! = exp(Gammaln(n+1)).
// Returns NaN for negative n.
func Factorial(n int) float64 {
	if n < 0 {
		return math.NaN()
	}
	if n == 0 || n == 1 {
		return 1
	}
	return math.Exp(Gammaln(float64(n) + 1))
}

// Softmax computes the softmax function in a numerically stable way.
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max)).
// Panics if values is empty.
func Softmax(values []float64) []float64 {
	if len(values) == 0 {
		panic("scigo: Softmax: values must not be empty")
	}

	// Find max for numerical stability.
	maxVal := values[0]
	for _, v := range values[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	result := make([]float64, len(values))
	sum := 0.0
	for i, v := range values {
		result[i] = math.Exp(v - maxVal)
		sum += result[i]
	}
	for i := range result {
		result[i] /= sum
	}
	return result
}

// RegularizedIncompleteBeta computes the regularized incomplete beta function I_x(a, b).
// I_x(a, b) = B(x; a, b) / B(a, b), where B(x; a, b) is the incomplete beta function.
// Uses the continued fraction representation from Numerical Recipes.
func RegularizedIncompleteBeta(x, a, b float64) float64 {
	if x < 0 || x > 1 {
		return math.NaN()
	}
	if x == 0 {
		return 0
	}
	if x == 1 {
		return 1
	}

	// Use the symmetry relation when x > (a+1)/(a+b+2) for better convergence
	if x > (a+1)/(a+b+2) {
		return 1 - RegularizedIncompleteBeta(1-x, b, a)
	}

	lbeta := Gammaln(a) + Gammaln(b) - Gammaln(a+b)
	bt := math.Exp(a*math.Log(x) + b*math.Log(1-x) - lbeta)

	return bt * betacf(a, b, x) / a
}

// betacf evaluates the continued fraction for the incomplete beta function
// using the modified Lentz's method (Numerical Recipes algorithm).
func betacf(a, b, x float64) float64 {
	const maxIter = 200
	const eps = 1e-14
	const fpmin = 1e-30

	qab := a + b
	qap := a + 1
	qam := a - 1
	c := 1.0
	d := 1 - qab*x/qap
	if math.Abs(d) < fpmin {
		d = fpmin
	}
	d = 1.0 / d
	h := d

	for m := 1; m <= maxIter; m++ {
		fm := float64(m)
		// Even step
		aa := fm * (b - fm) * x / ((qam + 2*fm) * (a + 2*fm))
		d = 1 + aa*d
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = 1 + aa/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1.0 / d
		h *= d * c

		// Odd step
		aa = -(a + fm) * (qab + fm) * x / ((a + 2*fm) * (qap + 2*fm))
		d = 1 + aa*d
		if math.Abs(d) < fpmin {
			d = fpmin
		}
		c = 1 + aa/c
		if math.Abs(c) < fpmin {
			c = fpmin
		}
		d = 1.0 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < eps {
			break
		}
	}
	return h
}
