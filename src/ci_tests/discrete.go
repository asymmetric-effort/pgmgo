package ci_tests

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// buildContingencyTables constructs contingency tables for (x, y) conditioned on each
// unique combination of z-variable values. Returns:
//   - tables: for each z-stratum, a 2D count matrix [xLevel][yLevel]
//   - xLevels, yLevels: the unique values for x and y (sorted for determinism)
func buildContingencyTables(x, y string, z []string, data *tabgo.DataFrame) (tables [][]float64, xLevels, yLevels []string, dfs []int) {
	n := data.Len()
	if n == 0 {
		return nil, nil, nil, nil
	}

	// Extract column values as strings for categorical handling.
	xVals := toStringSlice(data.Column(x).Values())
	yVals := toStringSlice(data.Column(y).Values())

	// Get sorted unique levels for x and y.
	xLevels = sortedUnique(xVals)
	yLevels = sortedUnique(yVals)
	nX := len(xLevels)
	nY := len(yLevels)

	// Build index maps.
	xIdx := indexMap(xLevels)
	yIdx := indexMap(yLevels)

	if len(z) == 0 {
		// No conditioning: single stratum.
		table := make([]float64, nX*nY)
		for i := 0; i < n; i++ {
			xi := xIdx[xVals[i]]
			yi := yIdx[yVals[i]]
			table[xi*nY+yi]++
		}
		return [][]float64{table}, xLevels, yLevels, []int{(nX - 1) * (nY - 1)}
	}

	// Extract z-column values.
	zCols := make([][]string, len(z))
	for i, name := range z {
		zCols[i] = toStringSlice(data.Column(name).Values())
	}

	// Group rows by z-combination.
	strata := make(map[string][]int) // z-key -> row indices
	for i := 0; i < n; i++ {
		key := zKey(zCols, i)
		strata[key] = append(strata[key], i)
	}

	// Sort strata keys for determinism.
	strataKeys := make([]string, 0, len(strata))
	for k := range strata {
		strataKeys = append(strataKeys, k)
	}
	sort.Strings(strataKeys)

	for _, key := range strataKeys {
		rows := strata[key]
		table := make([]float64, nX*nY)
		for _, r := range rows {
			xi := xIdx[xVals[r]]
			yi := yIdx[yVals[r]]
			table[xi*nY+yi]++
		}
		tables = append(tables, table)
		dfs = append(dfs, (nX-1)*(nY-1))
	}

	return tables, xLevels, yLevels, dfs
}

// computeExpected computes expected counts under independence for a single stratum table.
// table is nX*nY stored row-major.
func computeExpected(table []float64, nX, nY int) []float64 {
	rowSums := make([]float64, nX)
	colSums := make([]float64, nY)
	total := 0.0
	for xi := 0; xi < nX; xi++ {
		for yi := 0; yi < nY; yi++ {
			v := table[xi*nY+yi]
			rowSums[xi] += v
			colSums[yi] += v
			total += v
		}
	}

	expected := make([]float64, nX*nY)
	if total == 0 {
		return expected
	}
	for xi := 0; xi < nX; xi++ {
		for yi := 0; yi < nY; yi++ {
			expected[xi*nY+yi] = rowSums[xi] * colSums[yi] / total
		}
	}
	return expected
}

// flattenForTest flattens observed and expected into parallel slices, skipping cells
// where expected is zero (to avoid division by zero).
func flattenForTest(observed, expected []float64) (obs, exp []float64) {
	for i := range observed {
		if expected[i] > 0 {
			obs = append(obs, observed[i])
			exp = append(exp, expected[i])
		}
	}
	return
}

// ChiSquare is a CITest that uses the Pearson chi-squared statistic for discrete data.
var ChiSquare CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
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
		stat, _ := scigo.ChiSquareTest(obs, exp)
		totalStat += stat
		totalDF += dfs[i]
	}

	if totalDF <= 0 {
		return 0, 1, true
	}

	chi2 := scigo.NewChiSquared(float64(totalDF))
	pvalue := chi2.SurvivalFunction(totalStat)
	return totalStat, pvalue, pvalue > significance
}

// GSq is a CITest that uses the G-test (log-likelihood ratio) for discrete data.
var GSq CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
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
		stat, _ := scigo.GTest(obs, exp)
		totalStat += stat
		totalDF += dfs[i]
	}

	if totalDF <= 0 {
		return 0, 1, true
	}

	chi2 := scigo.NewChiSquared(float64(totalDF))
	pvalue := chi2.SurvivalFunction(totalStat)
	return totalStat, pvalue, pvalue > significance
}

// LogLikelihood is the same as GSq (the G-test is the log-likelihood ratio test).
var LogLikelihood CITest = GSq

// PowerDivergence returns a CITest using the generalized power divergence statistic
// with the given lambda parameter.
//
// Special cases: lambda=1 gives Pearson chi-squared, lambda=0 gives the G-test.
func PowerDivergence(lambda float64) CITest {
	return func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
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
			stat, _ := scigo.PowerDivergenceTest(obs, exp, lambda)
			totalStat += stat
			totalDF += dfs[i]
		}

		if totalDF <= 0 {
			return 0, 1, true
		}

		chi2 := scigo.NewChiSquared(float64(totalDF))
		pvalue := chi2.SurvivalFunction(totalStat)
		return totalStat, pvalue, pvalue > significance
	}
}

// --- helpers ---

func toStringSlice(vals []any) []string {
	out := make([]string, len(vals))
	for i, v := range vals {
		out[i] = fmt.Sprintf("%v", v)
	}
	return out
}

func sortedUnique(vals []string) []string {
	seen := make(map[string]bool)
	var out []string
	for _, v := range vals {
		if !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	sort.Strings(out)
	return out
}

func indexMap(levels []string) map[string]int {
	m := make(map[string]int, len(levels))
	for i, l := range levels {
		m[l] = i
	}
	return m
}

func zKey(zCols [][]string, row int) string {
	if len(zCols) == 1 {
		return zCols[0][row]
	}
	key := ""
	for i, col := range zCols {
		if i > 0 {
			key += "|"
		}
		key += col[row]
	}
	return key
}

// Ensure variables satisfy the CITest interface at compile time.
var _ CITest = ChiSquare
var _ CITest = GSq
var _ CITest = LogLikelihood

// Verify PowerDivergence returns a CITest.
var _ CITest = PowerDivergence(1.0)
