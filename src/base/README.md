# base

Package `base` provides the foundational graph types used by all pgmgo models.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/base`

## Types

| Type | Description |
|------|-------------|
| `DAG` | Directed acyclic graph with acyclicity enforcement on every edge addition |
| `PDAG` | Partially directed acyclic graph supporting both directed and undirected edges (CPDAGs) |
| `UndirectedGraph` | Undirected graph with error-checked operations |
| `ADMG` | Acyclic Directed Mixed Graph with directed and bidirected edges for latent common causes |
| `MAG` | Maximal Ancestral Graph extending ADMG with m-separation |
| `AncestralBase` | Composable helper providing ancestor/descendant traversal for directed graphs |
| `SimpleCausalModel` | Structural causal model (SCM) with a DAG and functional equations |
| `DAGStats` | Summary statistics struct (NumNodes, NumEdges, NumRoots, NumLeaves) |

## Key Functions

| Function | Description |
|----------|-------------|
| `NewDAG()` | Create an empty DAG |
| `NewPDAG()` | Create an empty PDAG |
| `FromDAG(dag)` | Convert a DAG to its CPDAG representation |
| `NewUndirectedGraph()` | Create an empty undirected graph |
| `NewADMG()` | Create an empty ADMG |
| `NewMAG()` | Create an empty MAG |
| `FromADMG(admg)` | Convert an ADMG to a MAG |
| `NewSimpleCausalModel(dag)` | Create an SCM backed by a DAG |
| `FromLavaan(syntax)` | Parse a DAG from lavaan-style syntax |
| `FromDagitty(syntax)` | Parse a DAG from dagitty-style syntax |
| `GetRandomDAG(nNodes, nEdges, seed)` | Generate a random DAG |
| `GetRandom(nodes, edgeProb, seed)` | Generate a random DAG with given node names |

## DAG Usage

```go
import "github.com/asymmetric-effort/pgmgo/src/base"

// Create a DAG and add structure
dag := base.NewDAG()
dag.AddNodes("X", "Y", "Z")
dag.AddEdge("X", "Y")
dag.AddEdge("Y", "Z")

// Query structure
parents := dag.Parents("Z")    // ["Y"]
children := dag.Children("X")  // ["Y"]
roots := dag.GetRoots()        // ["X"]
leaves := dag.GetLeaves()      // ["Z"]

// Topological ordering
order, _ := dag.TopologicalOrder() // ["X", "Y", "Z"]

// D-separation and independence
connected := dag.IsDConnected("X", "Z", nil) // true
sep, ok := dag.MinimalDSeparator("X", "Z")   // (["Y"], true)
blanket := dag.GetMarkovBlanket("Y")          // ["X", "Z"]

// Graph analysis
immoralities := dag.GetImmoralities()
indeps := dag.GetIndependencies()
ancestors := dag.GetAncestors("Z")   // ["X", "Y"]
descendants := dag.GetDescendants("X") // ["Y", "Z"]

// Interventional graph (do-calculus)
mutilated := dag.Do("Y") // removes incoming edges to Y

// Export formats
dot := dag.ToGraphviz()
lavaan := dag.ToLavaan()
dagitty := dag.ToDagitty()

// Parse from external syntax
dag2, _ := base.FromLavaan("Y ~ X\nZ ~ Y")
dag3, _ := base.FromDagitty("dag { X -> Y; Y -> Z }")
```

## PDAG Usage

```go
// Convert DAG to CPDAG (Markov equivalence class)
pdag := base.FromDAG(dag)

// Or build manually
pdag := base.NewPDAG()
pdag.AddNode("A")
pdag.AddNode("B")
pdag.AddDirectedEdge("A", "B")
pdag.AddUndirectedEdge("B", "C")

// Orient undirected edges using Meek rules
pdag.ApplyMeekRules()

// Convert back to a DAG
dag, err := pdag.ToDAG()
```

## ADMG / MAG Usage

```go
// ADMG with directed and bidirected edges
admg := base.NewADMG()
admg.AddNode("X")
admg.AddNode("Y")
admg.AddNode("Z")
admg.AddDirectedEdge("X", "Y")
admg.AddBidirectedEdge("X", "Z") // latent common cause

siblings := admg.Siblings("X")   // ["Z"]
districts := admg.Districts()

// Convert ADMG to MAG
mag, _ := base.FromADMG(admg)
separated := mag.MSeparation(
    map[string]bool{"X": true},
    map[string]bool{"Y": true},
    map[string]bool{"Z": true},
)
```

## SimpleCausalModel Usage

```go
dag := base.NewDAG()
dag.AddNodes("X", "Y")
dag.AddEdge("X", "Y")

scm := base.NewSimpleCausalModel(dag)
scm.SetEquation("Y", func(p map[string]float64) float64 {
    return 2.0*p["X"] + 1.0
})

// Forward simulation
values, _ := scm.Sample(map[string]float64{"X": 3.0})
// values["Y"] == 7.0

// Interventional model (do-calculus)
mutilated := scm.Intervene("Y", 5.0)
```
