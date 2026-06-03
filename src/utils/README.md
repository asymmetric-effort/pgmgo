# utils

Package `utils` provides shared utilities for pgmgo including combinatorics, convergence checking, state name helpers, and validation functions.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/utils`

## Combinatorics

| Function | Description |
|----------|-------------|
| `Combinations(items, k)` | Generate all k-element combinations of items |
| `Permutations(items)` | Generate all permutations using Heap's algorithm |

```go
import "github.com/asymmetric-effort/pgmgo/src/utils"

combos := utils.Combinations([]string{"A", "B", "C"}, 2)
// [["A","B"], ["A","C"], ["B","C"]]

perms := utils.Permutations([]string{"A", "B"})
// [["A","B"], ["B","A"]]
```

## Convergence Checking

| Type/Function | Description |
|---------------|-------------|
| `ConvergenceChecker` | Tracks a scalar across iterations, reports convergence when change < tolerance |
| `NewConvergenceChecker(tolerance, maxIter)` | Create a checker with given tolerance and max iterations |

```go
cc := utils.NewConvergenceChecker(1e-6, 1000)

for {
    value := computeObjective()
    if cc.Update(value) {
        break // converged or max iterations reached
    }
}

fmt.Println(cc.Converged())  // true if change < tolerance
fmt.Println(cc.Iteration())  // number of iterations run
```

## State Name Helpers

| Function | Description |
|----------|-------------|
| `GenerateStateNames(cardinality)` | Produce default names: ["s0", "s1", ..., "s{n-1}"] |
| `ValidateStateNames(names, cardinality)` | Check length matches and no duplicates |
| `StateIndex(names, name)` | Find the index of a state name |

```go
names := utils.GenerateStateNames(3) // ["s0", "s1", "s2"]

err := utils.ValidateStateNames(names, 3) // nil

idx, err := utils.StateIndex(names, "s1") // 1, nil
```

## Validation Helpers

| Function | Description |
|----------|-------------|
| `ValidatePositiveInt(name, value)` | Error if value <= 0 |
| `ValidateNonNegativeFloat(name, value)` | Error if value < 0 |
| `ValidateProbability(name, value)` | Error if value not in [0, 1] |
| `ValidateProbabilityDistribution(name, values, tolerance)` | Error if values don't sum to 1 |

```go
err := utils.ValidatePositiveInt("nSamples", 100) // nil
err = utils.ValidateProbability("p", 1.5)          // error

probs := []float64{0.3, 0.3, 0.4}
err = utils.ValidateProbabilityDistribution("prior", probs, 1e-9) // nil
```
