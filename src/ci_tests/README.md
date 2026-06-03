# ci_tests

Package `ci_tests` provides conditional independence tests for discrete, continuous, and multivariate data, used primarily in constraint-based structure learning.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/ci_tests`

## CITest Function Type

All tests conform to the `CITest` function signature:

```go
type CITest func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (statistic float64, pvalue float64, independent bool)
```

Parameters:
- `x`, `y`: variable names to test for independence
- `z`: conditioning set (may be empty)
- `data`: DataFrame with columns for all referenced variables
- `significance`: significance level (e.g., 0.05)

Returns: test statistic, p-value, and whether the variables are independent at the given significance level.

## Discrete Tests

| Test | Description |
|------|-------------|
| `ChiSquare` | Pearson's chi-squared test on contingency tables |
| `GSq` | G-test (log-likelihood ratio test) |
| `LogLikelihood` | Alias for GSq |
| `ModifiedLogLikelihood` | G-test with Williams' correction for small samples |
| `IndependenceMatch` | Tests if empirical distribution matches the product of marginals |
| `PowerDivergence(lambda)` | Generalized power-divergence family (lambda=1 gives chi-sq, lambda=0 gives G-test) |

## Continuous Tests

| Test | Description |
|------|-------------|
| `FisherZ` | Partial correlation with Fisher's Z transform |
| `Pearsonr` | Partial correlation with F-test |
| `PearsonrEquivalence(epsilon)` | Equivalence-based partial correlation test |
| `GCM` | Generalized Covariance Measure (OLS residual correlation) |
| `GeneralizedCov` | Direct covariance t-test on residuals |

## Multivariate Tests

| Test | Description |
|------|-------------|
| `HotellingLawley` | Hotelling-Lawley trace statistic |
| `PillaiTrace` | Pillai's trace statistic |
| `RoysLargestRoot` | Roy's largest root statistic |
| `WilksLambda` | Wilks' lambda statistic |

## Tree-Based Tests

| Test | Description |
|------|-------------|
| `TreeBasedCI` | Non-parametric CI test using CART regression tree feature importance |

## Usage Examples

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/ci_tests"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// Chi-squared test for discrete data
stat, pval, indep := ci_tests.ChiSquare("X", "Y", []string{"Z"}, data, 0.05)

// Fisher's Z test for continuous data
stat, pval, indep = ci_tests.FisherZ("X", "Y", []string{"Z"}, data, 0.05)

// Power divergence with custom lambda
test := ci_tests.PowerDivergence(0.5)
stat, pval, indep = test("X", "Y", nil, data, 0.05)

// Equivalence-based correlation test
eqTest := ci_tests.PearsonrEquivalence(0.1)
stat, pval, indep = eqTest("X", "Y", []string{"Z"}, data, 0.05)

// Tree-based nonparametric test
stat, pval, indep = ci_tests.TreeBasedCI("X", "Y", []string{"Z"}, data, 0.05)

// Using with structure learning
import "github.com/asymmetric-effort/pgmgo/src/learning"
pc := learning.NewPC(data, ci_tests.ChiSquare, 0.05)
```
