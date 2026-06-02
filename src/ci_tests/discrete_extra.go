package ci_tests

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// ModifiedLogLikelihood is a CITest that uses the G-test with Williams' correction
// for small samples. The corrected statistic is:
//
//	G_mod = G / q
//
// where G = 2 * sum(O * ln(O/E)) is the standard G-test statistic and
// q = 1 + (k - 1) / (6 * N) is the Williams' correction factor, with k being
// the number of cells in the contingency table and N the total count in the stratum.
//
// This correction reduces the tendency of the G-test to reject too liberally
// when sample sizes are small.
var ModifiedLogLikelihood CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	tables, xLevels, yLevels, dfs := buildContingencyTables(x, y, z, data)
	nX := len(xLevels)
	nY := len(yLevels)

	totalStat := 0.0
	totalDF := 0
	for i, table := range tables {
		expected := computeExpected(table, nX, nY)
		obs, exp := flattenForTest(table, expected)
		if len(obs) < 2 {
			continue
		}

		// Compute the raw G statistic: 2 * sum(O * ln(O/E))
		g := 0.0
		for j := range obs {
			if obs[j] > 0 {
				g += obs[j] * math.Log(obs[j]/exp[j])
			}
		}
		g *= 2

		// Williams' correction factor: q = 1 + (k - 1) / (6 * N)
		// where k is the number of cells and N is the total count.
		k := float64(nX * nY)
		totalN := 0.0
		for _, v := range table {
			totalN += v
		}
		q := 1.0
		if totalN > 0 {
			q = 1.0 + (k-1)/(6.0*totalN)
		}

		// Corrected statistic.
		if q > 0 {
			g /= q
		}

		totalStat += g
		totalDF += dfs[i]
	}

	if totalDF <= 0 {
		return 0, 1, true
	}

	chi2 := scigo.NewChiSquared(float64(totalDF))
	pvalue := chi2.SurvivalFunction(totalStat)
	return totalStat, pvalue, pvalue > significance
}

// IndependenceMatch is a CITest that uses a match-based independence test for
// discrete data.
//
// For each stratum (unique combination of z-values), it computes the observed
// proportion of matching pairs: for each pair of observations (i, j) within the
// stratum, it checks whether x[i]==x[j] and y[i]==y[j] simultaneously. Under
// independence, the expected proportion of such matches is:
//
//	E[match] = (sum_a p_a^2) * (sum_b q_b^2)
//
// where p_a is the proportion of x-value a and q_b is the proportion of y-value b.
//
// The test statistic aggregates across strata using a chi-square formulation:
//
//	chi2 = sum_strata n_pairs * (obs_match - exp_match)^2 / (exp_match * (1 - exp_match))
//
// and is compared to a chi-squared distribution with degrees of freedom equal
// to the number of strata.
var IndependenceMatch CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	if n < 4 {
		return 0, 1, true
	}

	xVals := toStringSlice(data.Column(x).Values())
	yVals := toStringSlice(data.Column(y).Values())

	// Group rows by z-stratum.
	type stratum struct {
		rows []int
	}
	var strata []stratum

	if len(z) == 0 {
		rows := make([]int, n)
		for i := range rows {
			rows[i] = i
		}
		strata = append(strata, stratum{rows: rows})
	} else {
		zCols := make([][]string, len(z))
		for i, name := range z {
			zCols[i] = toStringSlice(data.Column(name).Values())
		}

		groups := make(map[string][]int)
		for i := 0; i < n; i++ {
			key := zKey(zCols, i)
			groups[key] = append(groups[key], i)
		}

		// Collect in sorted order for determinism.
		keys := make([]string, 0, len(groups))
		for k := range groups {
			keys = append(keys, k)
		}
		sortStrings(keys)
		for _, k := range keys {
			strata = append(strata, stratum{rows: groups[k]})
		}
	}

	totalStat := 0.0
	numStrata := 0

	for _, s := range strata {
		rows := s.rows
		m := len(rows)
		if m < 2 {
			continue
		}

		nPairs := float64(m * (m - 1) / 2)

		// Count x-value frequencies and y-value frequencies within this stratum.
		xFreq := make(map[string]int)
		yFreq := make(map[string]int)
		for _, r := range rows {
			xFreq[xVals[r]]++
			yFreq[yVals[r]]++
		}

		// Expected match proportion under independence:
		// P(x-match) = sum_a (n_a choose 2) / (m choose 2)
		// P(y-match) = sum_b (n_b choose 2) / (m choose 2)
		// P(both match) = P(x-match) * P(y-match) under independence.
		xMatchPairs := 0.0
		for _, cnt := range xFreq {
			xMatchPairs += float64(cnt) * float64(cnt-1) / 2
		}
		yMatchPairs := 0.0
		for _, cnt := range yFreq {
			yMatchPairs += float64(cnt) * float64(cnt-1) / 2
		}

		pXMatch := xMatchPairs / nPairs
		pYMatch := yMatchPairs / nPairs
		expMatch := pXMatch * pYMatch

		// Count observed joint matching pairs.
		// A matching pair (i,j) has x[i]==x[j] AND y[i]==y[j].
		// Group by (x,y) pair to count efficiently.
		xyFreq := make(map[string]int)
		for _, r := range rows {
			key := xVals[r] + "\x00" + yVals[r]
			xyFreq[key]++
		}
		obsMatchPairs := 0.0
		for _, cnt := range xyFreq {
			obsMatchPairs += float64(cnt) * float64(cnt-1) / 2
		}
		obsMatch := obsMatchPairs / nPairs

		// Compute chi-square contribution for this stratum.
		// Avoid division by zero when expected match proportion is 0 or 1.
		denom := expMatch * (1 - expMatch)
		if denom < 1e-15 {
			continue
		}

		diff := obsMatch - expMatch
		totalStat += nPairs * diff * diff / denom
		numStrata++
	}

	if numStrata == 0 {
		return 0, 1, true
	}

	chi2 := scigo.NewChiSquared(float64(numStrata))
	pvalue := chi2.SurvivalFunction(totalStat)
	return totalStat, pvalue, pvalue > significance
}

// sortStrings sorts a string slice in place. (Avoids importing sort in this file
// since discrete.go already has it; but we keep this file self-contained.)
func sortStrings(s []string) {
	// Simple insertion sort for small slices (strata counts are typically small).
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j] > key {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}

// Compile-time interface checks.
var _ CITest = ModifiedLogLikelihood
var _ CITest = IndependenceMatch
