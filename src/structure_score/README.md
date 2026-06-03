# structure_score

Package `structure_score` provides scoring functions for structure learning, including BIC, AIC, BDeu, BDs, K2, log-likelihood, and variants for Gaussian and conditional Gaussian data.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/structure_score`

## Interface

```go
type StructureScore interface {
    LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64
    Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64
}
```

## Discrete Scoring Functions

| Type | Description |
|------|-------------|
| `BIC` | Bayesian Information Criterion: LL - 0.5 * k * ln(N) |
| `AIC` | Akaike Information Criterion: LL - k |
| `BDeu` | Bayesian Dirichlet equivalent uniform score with equivalent sample size |
| `BDs` | Bayesian Dirichlet sparse score with structure prior penalty |
| `K2` | K2 score (Cooper & Herskovits, 1992) |
| `LogLikelihood` | Pure log-likelihood with no penalty term |

## Gaussian Scoring Functions

| Type | Description |
|------|-------------|
| `GaussianBIC` | BIC for continuous Gaussian data using OLS regression |
| `GaussianAIC` | AIC for continuous Gaussian data |
| `GaussianLogLikelihood` | Log-likelihood for Gaussian data |

## Conditional Gaussian Scoring Functions

| Type | Description |
|------|-------------|
| `ConditionalGaussianBIC` | BIC for mixed discrete/continuous parents |
| `ConditionalGaussianAIC` | AIC for mixed discrete/continuous parents |

## Usage

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/structure_score"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// BIC score
bic := structure_score.NewBIC()
score := bic.LocalScore("Grade", []string{"Difficulty", "Intelligence"}, data)

// BDeu with equivalent sample size = 10
bdeu := structure_score.NewBDeu(10.0)
score = bdeu.LocalScore("Grade", []string{"Difficulty"}, data)

// BDs with ESS=10 and structure prior=1.0
bds := structure_score.NewBDs(10.0, 1.0)
score = bds.LocalScore("Y", []string{"X"}, data)

// K2 score
k2 := structure_score.NewK2()
score = k2.LocalScore("Y", nil, data)

// Total score for an entire DAG structure
parentMap := map[string][]string{
    "X": {},
    "Y": {"X"},
    "Z": {"X", "Y"},
}
totalScore := bic.Score([]string{"X", "Y", "Z"}, parentMap, data)

// Gaussian BIC for continuous data
gbic := structure_score.NewGaussianBIC()
score = gbic.LocalScore("Y", []string{"X1", "X2"}, continuousData)

// Conditional Gaussian for mixed data
cgbic := structure_score.NewConditionalGaussianBIC(
    []string{"Discrete1"},     // discrete parents
    []string{"Continuous1"},   // continuous parents
)
score = cgbic.LocalScore("Y", []string{"Discrete1", "Continuous1"}, mixedData)
```
