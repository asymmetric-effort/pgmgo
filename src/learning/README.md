# learning

Package `learning` provides parameter estimation and structure learning algorithms for probabilistic graphical models.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/learning`

## Parameter Estimation

| Type | Description |
|------|-------------|
| `MaximumLikelihoodEstimator` | MLE parameter estimation for discrete BNs from data |
| `BayesianEstimator` | Bayesian parameter estimation with Dirichlet priors |
| `ExpectationMaximization` | EM algorithm for learning with latent variables |
| `MirrorDescentEstimator` | Mirror descent optimization for CPD parameters |
| `MarginalEstimator` | Marginal likelihood estimation |
| `LinearGaussianMLE` | MLE for LinearGaussianBayesianNetworks |
| `SEMEstimator` | OLS-based parameter estimation for SEMs |
| `IVEstimator` | Instrumental variable (2SLS) causal effect estimator |
| `LinearModel` | Simple OLS linear regression |
| `BaseEstimator` | Base type for parameter estimators |

## Structure Learning

| Type | Description |
|------|-------------|
| `PCAlgorithm` | PC algorithm (constraint-based structure learning) |
| `HillClimbSearch` | Hill-climbing score-based structure search with tabu list |
| `GES` | Greedy Equivalence Search (score-based, learns CPDAGs) |
| `MMHC` | Max-Min Hill-Climbing (hybrid constraint + score) |
| `ExhaustiveSearch` | Exhaustive DAG enumeration (small networks only) |
| `TreeSearch` | Chow-Liu tree structure learning via mutual information |
| `ConstraintBasedEstimator` | Wrapper around PC for constraint-based learning |
| `ExpertInLoop` | Structure learning with LLM-assisted edge orientation |

## Supporting Types

| Type | Description |
|------|-------------|
| `ExpertKnowledge` | Required/forbidden edges and tier orderings for constrained search |
| `LLMClient` | Interface for LLM integration (with `HTTPLLMClient` implementation) |
| `CausalPromptTemplate` | Templates for causal direction and independence queries to LLMs |
| `ScoreFunc` | Function type for local structure scores |
| `CITestFunc` | Function type for conditional independence tests |

## Parameter Estimation Examples

### MLE

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/learning"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

bn := models.NewBayesianNetwork()
// ... add nodes and edges ...

mle := learning.NewMLE(bn, data)
err := mle.Estimate() // fits CPDs for all nodes
```

### Bayesian Estimation

```go
be := learning.NewBayesianEstimator(bn, data, 1.0) // ESS = 1.0
err := be.Estimate()
```

### EM with Latent Variables

```go
em := learning.NewEM(bn, data, []string{"Latent"}, 100, 1e-4)
err := em.Estimate()
iters := em.Iterations()
converged := em.Converged()
params, _ := em.GetParameters()
```

### SEM Estimation

```go
se := learning.NewSEMEstimator(sem, data)
err := se.Estimate()
coeffs, intercept, variance, _ := se.GetCoefficients("Y")
```

## Structure Learning Examples

### PC Algorithm

```go
import "github.com/asymmetric-effort/pgmgo/src/ci_tests"

pc := learning.NewPC(data, ci_tests.ChiSquare, 0.05,
    learning.WithMaxCondSetSize(3),
)
pdag, _ := pc.Estimate()    // returns CPDAG
bn, _ := pc.EstimateBN()    // returns oriented BN
```

### Hill-Climbing

```go
hc := learning.NewHillClimbSearch(data, learning.BICScore(),
    learning.WithMaxIndegree(3),
    learning.WithTabuSize(10),
    learning.WithWhiteList([][2]string{{"X", "Y"}}),
    learning.WithBlackList([][2]string{{"Y", "X"}}),
)
bn, _ := hc.Estimate()
```

### GES

```go
ges := learning.NewGES(data, learning.BICScore())
pdag, _ := ges.Estimate() // returns CPDAG
```

### MMHC (Hybrid)

```go
mmhc := learning.NewMMHC(data, learning.BICScore(), ci_tests.ChiSquare, 0.05)
bn, _ := mmhc.Estimate()
```

### Tree Search (Chow-Liu)

```go
ts := learning.NewTreeSearch(data,
    learning.WithRoot("X0"),
    learning.WithClassVariable("Class"),
)
bn, _ := ts.Estimate()
```

### Expert Knowledge Constraints

```go
ek := learning.NewExpertKnowledge()
ek.AddRequiredEdge("Gene", "Protein")
ek.AddForbiddenEdge("Protein", "Gene")
ek.AddTierOrdering([][]string{{"Gene"}, {"Protein"}, {"Phenotype"}})

hc := learning.NewHillClimbSearch(data, learning.BICScore())
ek.ApplyToHillClimb(hc)
bn, _ := hc.Estimate()
```

## Score Functions

```go
bic := learning.BICScore()
aic := learning.AICScore()
k2 := learning.K2Score()
bdeu := learning.BDeuScore(10.0) // ESS = 10.0
```
