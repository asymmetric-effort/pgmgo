package ci_tests

import "github.com/asymmetric-effort/pgmgo/lib/tabgo"

// CITest is a function that tests conditional independence of x and y given z.
// Returns test statistic, p-value, and whether the variables are independent at the given significance level.
type CITest func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (statistic float64, pvalue float64, independent bool)
