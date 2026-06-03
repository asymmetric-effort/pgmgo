# sampling

Package `sampling` provides MCMC and other sampling-based methods for approximate inference in Bayesian networks.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/sampling`

## Types

| Type | Description |
|------|-------------|
| `BayesianModelSampling` | Forward sampling and rejection sampling for Bayesian networks |
| `GibbsSampling` | Gibbs sampling (MCMC) for approximate inference with burn-in and thinning |

## BayesianModelSampling

Generates samples by forward-sampling the Bayesian network in topological order.

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

bn := models.NewBayesianNetwork()
// ... build and add CPDs ...

sampler, _ := sampling.NewBayesianModelSampling(bn, 42) // seed=42

// Forward sampling: generate 1000 unconditional samples
samples, _ := sampler.ForwardSample(1000)
// samples is a *tabgo.DataFrame with one column per variable

// Rejection sampling: generate samples conditioned on evidence
condSamples, _ := sampler.RejectionSample(
    1000,
    map[string]int{"Rain": 1}, // evidence
)

// Likelihood-weighted sampling
weightedSamples, weights, _ := sampler.LikelihoodWeightedSample(
    1000,
    map[string]int{"Rain": 1},
)
```

## GibbsSampling

Generates samples using Gibbs sampling (single-site MCMC), which iteratively resamples each variable conditioned on the current values of all other variables.

```go
gibbs, _ := sampling.NewGibbsSampling(bn, 42)

// Sample with burn-in and thinning
samples, _ := gibbs.Sample(
    1000,                        // number of samples to collect
    500,                         // burn-in iterations to discard
    2,                           // thinning: collect every 2nd sample
    map[string]int{"Rain": 1},  // evidence (fixed variables)
)
// samples is a *tabgo.DataFrame
```
