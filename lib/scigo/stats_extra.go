package scigo

import "math"

// Chi2Contingency performs a chi-square test of independence on a contingency table.
// It computes expected frequencies, the chi-square statistic, degrees of freedom,
// and the p-value.
// Panics if the table has fewer than 2 rows or 2 columns.
func Chi2Contingency(observed [][]float64) (statistic, pvalue float64, dof int, expected [][]float64) {
	rows := len(observed)
	if rows < 2 {
		panic("scigo: Chi2Contingency: need at least 2 rows")
	}
	cols := len(observed[0])
	if cols < 2 {
		panic("scigo: Chi2Contingency: need at least 2 columns")
	}
	for _, row := range observed {
		if len(row) != cols {
			panic("scigo: Chi2Contingency: all rows must have the same length")
		}
	}

	// Compute row sums, column sums, and grand total.
	rowSums := make([]float64, rows)
	colSums := make([]float64, cols)
	total := 0.0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			rowSums[i] += observed[i][j]
			colSums[j] += observed[i][j]
			total += observed[i][j]
		}
	}

	// Compute expected frequencies: E[i][j] = rowSum[i] * colSum[j] / total.
	expected = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		expected[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			expected[i][j] = rowSums[i] * colSums[j] / total
		}
	}

	// Compute chi-square statistic.
	statistic = 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if expected[i][j] > 0 {
				diff := observed[i][j] - expected[i][j]
				statistic += diff * diff / expected[i][j]
			}
		}
	}

	dof = (rows - 1) * (cols - 1)
	chi2 := NewChiSquared(float64(dof))
	pvalue = chi2.SurvivalFunction(statistic)
	return
}

// FisherExact performs Fisher's exact test for a 2x2 contingency table.
// It returns the odds ratio and a two-sided p-value computed using the
// hypergeometric distribution.
func FisherExact(table [2][2]int) (oddsRatio, pvalue float64) {
	a, b := table[0][0], table[0][1]
	c, d := table[1][0], table[1][1]

	// Odds ratio: (a*d) / (b*c).
	if b*c == 0 {
		oddsRatio = math.Inf(1)
	} else {
		oddsRatio = float64(a*d) / float64(b*c)
	}

	// Fisher's exact test uses the hypergeometric distribution.
	// Under the null hypothesis, the distribution of a is hypergeometric
	// with parameters N = a+b+c+d, K = a+c, n = a+b.
	N := a + b + c + d
	K := a + c // successes in population
	n := a + b // draws

	hyper := NewHypergeometric(N, K, n)

	// Probability of observing exactly the given table.
	pObserved := hyper.PMF(a)

	// Two-sided p-value: sum probabilities of all tables at least as extreme
	// (i.e., whose probability is <= pObserved).
	lo := 0
	if n-(N-K) > 0 {
		lo = n - (N - K)
	}
	hi := K
	if n < hi {
		hi = n
	}

	pvalue = 0
	for k := lo; k <= hi; k++ {
		p := hyper.PMF(k)
		if p <= pObserved+1e-15 { // tolerance for floating point
			pvalue += p
		}
	}
	if pvalue > 1 {
		pvalue = 1
	}
	return
}

// PointBiserialR computes the point-biserial correlation between a continuous
// variable x and a binary variable y. This is equivalent to the Pearson
// correlation with y coded as 0/1.
// Panics if x and y have different lengths or fewer than 3 elements.
func PointBiserialR(x []float64, y []bool) (r, pvalue float64) {
	n := len(x)
	if n != len(y) {
		panic("scigo: PointBiserialR: x and y must have the same length")
	}
	if n < 3 {
		panic("scigo: PointBiserialR: need at least 3 data points")
	}

	// Convert y to float64 (0/1) and compute Pearson correlation.
	yf := make([]float64, n)
	for i, v := range y {
		if v {
			yf[i] = 1
		}
	}

	r, pvalue = PearsonCorrelation(x, yf)
	return
}
