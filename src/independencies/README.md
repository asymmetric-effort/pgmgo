# independencies

Package `independencies` provides representations for conditional independence assertions and independence relations, including graphoid axiom closure.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/independencies`

## Types

| Type | Description |
|------|-------------|
| `IndependenceAssertion` | A single conditional independence statement X _\|_ Y \| Z |
| `Independencies` | A collection of independence assertions with set operations |

## IndependenceAssertion

Represents X _|_ Y | Z where X and Y are variable sets and Z is the conditioning set.

```go
import "github.com/asymmetric-effort/pgmgo/src/independencies"

// Create: A _|_ B | {C, D}
a := independencies.NewIndependenceAssertion(
    []string{"A"},
    []string{"B"},
    []string{"C", "D"},
)

// Access components
x := a.Event1() // ["A"]
y := a.Event2() // ["B"]
z := a.Given()  // ["C", "D"]

// Equality (symmetric: X _|_ Y == Y _|_ X)
b := independencies.NewIndependenceAssertion([]string{"B"}, []string{"A"}, []string{"C", "D"})
a.Equals(b) // true

// Containment (X _|_ {Y,W} | Z implies X _|_ Y | Z)
broad := independencies.NewIndependenceAssertion([]string{"A"}, []string{"B", "E"}, []string{"C"})
narrow := independencies.NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"})
broad.Contains(narrow) // true

// Display
fmt.Println(a.String())      // "A _|_ B | {C, D}"
fmt.Println(a.LatexString()) // "A \\perp B \\mid \\{C, D\\}"
```

## Independencies Collection

```go
ind := independencies.NewIndependencies()

// Add assertions
ind.Add(
    independencies.NewIndependenceAssertion([]string{"A"}, []string{"B"}, []string{"C"}),
    independencies.NewIndependenceAssertion([]string{"D"}, []string{"E"}, nil),
)

// Query
ind.Len()                      // 2
ind.Contains(someAssertion)    // true/false
ind.Entails(someAssertion)     // true if implied by containment

// Set operations
ind.Remove(someAssertion)

// Compare two independence sets
ind.IsEquivalent(other)

// Get all referenced variables
vars := ind.GetAllVariables() // sorted union of all variables

// Closure under graphoid axioms (symmetry, decomposition, weak union, contraction)
closed := ind.Closure()

// Remove redundant assertions
reduced := ind.Reduce()

// Display
fmt.Println(ind.String())
fmt.Println(ind.LatexString())
fmt.Println(ind.GetFactorizedProduct()) // e.g., "P(A | C) * P(D)"
```
