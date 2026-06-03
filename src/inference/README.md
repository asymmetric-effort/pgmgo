# inference

Package `inference` provides exact and approximate inference algorithms including variable elimination, belief propagation, MPLP, causal inference, approximate inference, and DBN inference.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/inference`

## Types

| Type | Description |
|------|-------------|
| `VariableElimination` | Exact inference via variable elimination with configurable elimination-order heuristics |
| `BeliefPropagation` | Exact inference on junction trees using Hugin-style message passing |
| `MPLP` | MAP inference via Max-Product Linear Programming (coordinate descent on LP dual) |
| `CausalInference` | Causal reasoning using do-calculus, ATE estimation, and backdoor adjustment |
| `ApproxInference` | Approximate marginal inference via likelihood-weighted sampling |
| `DBNInference` | Forward filtering inference for Dynamic Bayesian Networks |

## VariableElimination

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

// Get factors from a Bayesian network
markovFactors, _ := bn.ToMarkovFactors()

// Create VE engine (optional heuristic: "min_neighbors", "min_fill", "min_weight")
ve := inference.NewVariableElimination(markovFactors, "min_neighbors")

// Conditional query: P(Grade | Intelligence=1)
result, _ := ve.Query(
    []string{"Grade"},
    map[string]int{"Intelligence": 1},
)

// MAP query: most probable assignment
mapAssignment, _ := ve.MAP(
    []string{"Grade", "Difficulty"},
    map[string]int{"Intelligence": 1},
)
```

## BeliefPropagation

```go
// Build a junction tree from a BN
jt, _ := bn.ToJunctionTree()

// Create BP engine from junction tree components
bp, _ := inference.NewBeliefPropagation(
    jt.Cliques(),
    jt.SeparatorSets(),
    cliqueFactors, // map[int][]*factors.DiscreteFactor
)

// Run calibration (collect + distribute messages)
bp.Calibrate()

// Query marginals
result, _ := bp.Query(
    []string{"Grade"},
    map[string]int{"Intelligence": 1},
)
```

## MPLP (MAP Inference)

```go
mplp := inference.NewMPLP(markovFactors)

// Find MAP assignment
assignment, _ := mplp.MAP(
    []string{"Grade", "Difficulty"},
    map[string]int{"Intelligence": 1},
    100,  // max iterations
    1e-6, // tolerance
)
```

## CausalInference

```go
ci, _ := inference.NewCausalInference(bn)

// Interventional query: P(Y | do(X=1))
result, _ := ci.Query(
    []string{"Y"},
    map[string]int{"X": 1}, // do-variables
    nil,                      // evidence
)

// Average Treatment Effect
ate, _ := ci.ATE("X", "Y", 0, 1) // E[Y|do(X=1)] - E[Y|do(X=0)]

// Backdoor adjustment
adjustedResult, _ := ci.BackdoorAdjustment(
    "X", "Y",
    []string{"Z"},          // adjustment set
    map[string]int{"X": 1}, // intervention
)
```

## ApproxInference

```go
ai := inference.NewApproxInference(markovFactors, 42) // seed=42

result, _ := ai.Query(
    []string{"Grade"},
    map[string]int{"Intelligence": 1},
    10000, // number of samples
)
```

## DBNInference

```go
dbnInf := inference.NewDBNInference(
    initialFactors,    // factors for t=0
    transitionFactors, // factors for t -> t+1 (use "_prev" suffix)
    []string{"X"},     // interface nodes
)

// Forward filtering: query at time step T
result, _ := dbnInf.Forward(
    []string{"X"},
    map[string]int{"Obs": 1}, // evidence at each step
    5, // number of time steps
)
```
