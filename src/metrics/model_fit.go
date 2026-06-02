package metrics

import (
	"math"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// pearson computes the Pearson correlation coefficient between two float64 slices.
func pearson(x, y []float64) float64 {
	n := len(x)
	if n == 0 || n != len(y) {
		return 0
	}
	var sumX, sumY, sumXY, sumX2, sumY2 float64
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}
	nf := float64(n)
	num := nf*sumXY - sumX*sumY
	den := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))
	if den == 0 {
		return 0
	}
	return num / den
}

// CorrelationScore returns the average absolute Pearson correlation between
// pairs of variables connected by the given edges. Each edge is [src, dst].
// Column values are extracted from data using Float64 conversion.
func CorrelationScore(edges [][2]string, data *tabgo.DataFrame) float64 {
	if len(edges) == 0 {
		return 0
	}
	var total float64
	for _, e := range edges {
		x := data.Column(e[0]).Float64()
		y := data.Column(e[1]).Float64()
		total += math.Abs(pearson(x, y))
	}
	return total / float64(len(edges))
}

// regularizedIncompleteBeta computes the regularized incomplete beta function
// I_x(a, b) using a continued fraction expansion. This is used for computing
// chi-square CDF p-values.
func regularizedIncompleteBeta(x, a, b float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}

	// Use the symmetry relation when x > (a+1)/(a+b+2) for better convergence.
	if x > (a+1)/(a+b+2) {
		return 1 - regularizedIncompleteBeta(1-x, b, a)
	}

	// Compute ln(Beta(a,b)) = lnGamma(a) + lnGamma(b) - lnGamma(a+b)
	lnBeta := lgamma(a) + lgamma(b) - lgamma(a+b)

	// Front factor: x^a * (1-x)^b / (a * Beta(a,b))
	front := math.Exp(a*math.Log(x) + b*math.Log(1-x) - lnBeta - math.Log(a))

	// Lentz's continued fraction method.
	const maxIter = 200
	const eps = 1e-14
	const tiny = 1e-30

	// The continued fraction for I_x(a,b) / front is:
	// 1 / (1+ d1/(1+ d2/(1+ ...)))
	// where d_m are the CF coefficients.
	cf := 1.0
	c := 1.0
	d := 1.0 - (a+b)*x/(a+1)
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1 / d
	cf = d

	for m := 1; m <= maxIter; m++ {
		mf := float64(m)

		// Even step: d_{2m}
		num := mf * (b - mf) * x / ((a + 2*mf - 1) * (a + 2*mf))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		cf *= c * d

		// Odd step: d_{2m+1}
		num = -((a + mf) * (a + b + mf) * x) / ((a + 2*mf) * (a + 2*mf + 1))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		delta := c * d
		cf *= delta

		if math.Abs(delta-1) < eps {
			break
		}
	}

	return front * cf
}

// lgamma returns the natural log of Gamma(x).
func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// chiSquareCDF returns the CDF of the chi-square distribution with df degrees
// of freedom evaluated at x. Uses the regularized incomplete gamma function:
// P(df/2, x/2).
func chiSquareCDF(x float64, df int) float64 {
	if x <= 0 || df <= 0 {
		return 0
	}
	a := float64(df) / 2
	z := x / 2
	// P(a, z) = regularized lower incomplete gamma = I_z(a, ...) can be
	// expressed via the regularized incomplete beta function:
	// P(a, z) = 1 - I_{e^{-z} ...} but it's easier to use a series/CF directly.
	return lowerIncompleteGammaReg(a, z)
}

// lowerIncompleteGammaReg computes the regularized lower incomplete gamma
// function P(a, x) = gamma(a, x) / Gamma(a) using a series expansion.
func lowerIncompleteGammaReg(a, x float64) float64 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}

	// For x < a+1, use the series expansion; otherwise use continued fraction.
	if x < a+1 {
		return lowerGammaSeries(a, x)
	}
	return 1 - upperGammaCF(a, x)
}

// lowerGammaSeries computes P(a,x) via the series expansion.
func lowerGammaSeries(a, x float64) float64 {
	const maxIter = 200
	const eps = 1e-14

	term := 1.0 / a
	sum := term
	for n := 1; n <= maxIter; n++ {
		term *= x / (a + float64(n))
		sum += term
		if math.Abs(term) < math.Abs(sum)*eps {
			break
		}
	}
	return sum * math.Exp(-x+a*math.Log(x)-lgamma(a))
}

// upperGammaCF computes Q(a,x) = 1 - P(a,x) via continued fraction (Lentz).
func upperGammaCF(a, x float64) float64 {
	const maxIter = 200
	const eps = 1e-14
	const tiny = 1e-30

	b0 := x + 1 - a
	c := 1.0 / tiny
	d := 1.0 / b0
	if math.Abs(b0) < tiny {
		d = 1.0 / tiny
	}
	f := d

	for i := 1; i <= maxIter; i++ {
		an := -float64(i) * (float64(i) - a)
		bn := x + float64(2*i) + 1 - a
		d = bn + an*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = bn + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		delta := d * c
		f *= delta
		if math.Abs(delta-1) < eps {
			break
		}
	}

	return f * math.Exp(-x+a*math.Log(x)-lgamma(a))
}

// ImpliedCIs enumerates conditional independencies implied by the graph
// structure. For each pair of non-adjacent variables (not connected by any
// edge), they are conditionally independent given some separating set. This
// function returns a simple enumeration: for each non-adjacent pair {X, Y},
// it returns [X_vars, Y_vars, conditioning_set] where X_vars=[X], Y_vars=[Y],
// and conditioning_set is the set of neighbors of X that are also neighbors of Y
// (a heuristic separating set).
//
// Each element is [3][]string: [0]=first variable(s), [1]=second variable(s),
// [2]=conditioning variables.
func ImpliedCIs(edges [][2]string, allVars []string) [][3][]string {
	// Build adjacency set.
	adj := make(map[string]map[string]bool)
	for _, v := range allVars {
		adj[v] = make(map[string]bool)
	}
	for _, e := range edges {
		if adj[e[0]] == nil {
			adj[e[0]] = make(map[string]bool)
		}
		if adj[e[1]] == nil {
			adj[e[1]] = make(map[string]bool)
		}
		adj[e[0]][e[1]] = true
		adj[e[1]][e[0]] = true
	}

	// For each non-adjacent pair, compute a conditioning set.
	var result [][3][]string
	for i := 0; i < len(allVars); i++ {
		for j := i + 1; j < len(allVars); j++ {
			x, y := allVars[i], allVars[j]
			if adj[x][y] {
				continue // adjacent, skip
			}
			// Conditioning set: neighbors of X that are also neighbors of Y.
			var condSet []string
			for nb := range adj[x] {
				if adj[y][nb] {
					condSet = append(condSet, nb)
				}
			}
			sort.Strings(condSet)
			result = append(result, [3][]string{
				{x},
				{y},
				condSet,
			})
		}
	}
	return result
}

// FisherC computes Fisher's C statistic for testing overall model fit.
// It uses the implied conditional independencies from the edge structure:
//
//	C = -2 * sum(log(p_i))
//
// where p_i is the p-value from testing each implied CI using partial
// correlation and a t-test. Under the null (model is correct), C follows
// a chi-square distribution with 2k degrees of freedom, where k is the
// number of independence tests.
//
// Returns the C statistic and its p-value.
func FisherC(edges [][2]string, data *tabgo.DataFrame) (statistic, pvalue float64) {
	allVars := data.Columns()
	cis := ImpliedCIs(edges, allVars)

	if len(cis) == 0 {
		// No implied CIs: model is saturated; perfect fit.
		return 0, 1
	}

	// Precompute float64 columns.
	colData := make(map[string][]float64)
	for _, name := range allVars {
		colData[name] = data.Column(name).Float64()
	}

	n := data.Len()
	var sumLogP float64

	for _, ci := range cis {
		x := ci[0][0]
		y := ci[1][0]
		condVars := ci[2]

		// Compute partial correlation between x and y given condVars.
		pCorr := partialCorrelation(colData, x, y, condVars, n)

		// Convert to p-value using t-test.
		// t = r * sqrt((n - |Z| - 2) / (1 - r^2))
		dfT := n - len(condVars) - 2
		if dfT < 1 {
			dfT = 1
		}

		r2 := pCorr * pCorr
		if r2 >= 1 {
			r2 = 0.9999
		}
		tStat := math.Abs(pCorr) * math.Sqrt(float64(dfT)/(1-r2))
		p := tTestPValue(tStat, dfT)
		if p < 1e-300 {
			p = 1e-300 // avoid log(0)
		}
		sumLogP += math.Log(p)
	}

	k := len(cis)
	statistic = -2 * sumLogP
	df := 2 * k
	pvalue = 1 - chiSquareCDF(statistic, df)
	return
}

// partialCorrelation computes the partial correlation between x and y given z
// using recursive formula. For efficiency with multiple conditioning variables,
// it uses residual-based computation.
func partialCorrelation(colData map[string][]float64, x, y string, z []string, n int) float64 {
	if len(z) == 0 {
		return pearson(colData[x], colData[y])
	}

	// Compute residuals of x and y after regressing on z variables.
	xResid := residuals(colData, x, z, n)
	yResid := residuals(colData, y, z, n)
	return pearson(xResid, yResid)
}

// residuals computes the residuals of target after linear regression on predictors.
// Uses normal equations via simple iterative approach for small predictor sets.
func residuals(colData map[string][]float64, target string, predictors []string, n int) []float64 {
	y := colData[target]
	if len(predictors) == 0 {
		cp := make([]float64, n)
		copy(cp, y)
		return cp
	}

	// For simplicity, use sequential residualization (Frisch-Waugh-Lovell).
	// Residualize y on each predictor one at a time.
	resid := make([]float64, n)
	copy(resid, y)

	for _, pred := range predictors {
		xp := colData[pred]
		// Simple linear regression of resid on xp.
		var sumX, sumY, sumXY, sumX2 float64
		for i := 0; i < n; i++ {
			sumX += xp[i]
			sumY += resid[i]
			sumXY += xp[i] * resid[i]
			sumX2 += xp[i] * xp[i]
		}
		nf := float64(n)
		denom := nf*sumX2 - sumX*sumX
		if math.Abs(denom) < 1e-15 {
			continue
		}
		slope := (nf*sumXY - sumX*sumY) / denom
		intercept := (sumY - slope*sumX) / nf
		for i := 0; i < n; i++ {
			resid[i] = resid[i] - intercept - slope*xp[i]
		}
	}
	return resid
}

// tTestPValue computes the two-tailed p-value for a t-statistic with given df.
// Uses the relationship: p = 1 - I_{df/(df+t^2)}(df/2, 1/2) ... actually:
// p = I_{df/(df+t^2)}(df/2, 1/2)   (two-tailed via the beta function).
func tTestPValue(t float64, df int) float64 {
	if df <= 0 {
		return 1
	}
	x := float64(df) / (float64(df) + t*t)
	p := regularizedIncompleteBeta(x, float64(df)/2, 0.5)
	return p
}
