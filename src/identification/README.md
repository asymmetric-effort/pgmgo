# identification

Package `identification` provides causal effect identification algorithms including back-door adjustment and front-door criterion.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/identification`

## Functions

| Function | Description |
|----------|-------------|
| `IsValidAdjustmentSet` | Check if a set satisfies the back-door criterion for causal effect estimation |
| `GetMinimalAdjustmentSet` | Find a minimal valid back-door adjustment set |
| `GetAllAdjustmentSets` | Enumerate all valid adjustment sets (up to a size limit) |
| `IsValidFrontdoorSet` | Check if a set satisfies the front-door criterion |
| `GetFrontdoorSet` | Find a valid front-door set |

## Back-Door Criterion

The back-door criterion requires:
1. No variable in the adjustment set is a descendant of treatment.
2. The adjustment set blocks all back-door paths from treatment to outcome.

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/identification"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

g := graphgo.NewDiGraph()
g.AddNode("X")
g.AddNode("Y")
g.AddNode("Z")
g.AddEdge("Z", "X")
g.AddEdge("Z", "Y")
g.AddEdge("X", "Y")

// Check if {Z} is a valid adjustment set for X -> Y
valid := identification.IsValidAdjustmentSet(g, "X", "Y", []string{"Z"})
// true: Z blocks the back-door path X <- Z -> Y

// Find minimal adjustment set
minSet, err := identification.GetMinimalAdjustmentSet(g, "X", "Y")
// ["Z"]

// Enumerate all valid adjustment sets
allSets := identification.GetAllAdjustmentSets(g, "X", "Y", 3)
```

## Front-Door Criterion

The front-door criterion requires:
1. The front-door set intercepts all directed paths from treatment to outcome.
2. No unblocked back-door path from treatment to any variable in the set.
3. All back-door paths from front-door variables to outcome are blocked by treatment.

```go
// X -> M -> Y with latent confounder U -> X, U -> Y
g := graphgo.NewDiGraph()
g.AddNode("X")
g.AddNode("M")
g.AddNode("Y")
g.AddEdge("X", "M")
g.AddEdge("M", "Y")

valid := identification.IsValidFrontdoorSet(g, "X", "Y", []string{"M"})

fdSet, err := identification.GetFrontdoorSet(g, "X", "Y")
```
