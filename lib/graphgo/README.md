# graphgo

Package `graphgo` provides graph data structures and algorithms including directed and undirected graphs, topological sort, d-separation, clique finding, moralization, and junction tree construction. It serves as the NetworkX equivalent for pgmgo.

```
import "github.com/asymmetric-effort/pgmgo/lib/graphgo"
```

## DiGraph (Directed Graph)

```go
g := graphgo.NewDiGraph()

// Nodes
g.AddNode("A")
g.AddNodes("B", "C", "D")
g.HasNode("A")          // true
g.Nodes()               // all node names
g.NumberOfNodes()        // 4
g.RemoveNode("D")

// Edges
g.AddEdge("A", "B")     // creates nodes if needed
g.HasEdge("A", "B")     // true
g.Edges()               // []Edge{{Src: "A", Dst: "B"}, ...}
g.NumberOfEdges()
g.RemoveEdge("A", "B")

// Neighbors
g.Predecessors("B")     // nodes with edges to B
g.Successors("A")       // nodes A has edges to
g.Parents("B")          // alias for Predecessors
g.Children("A")         // alias for Successors
g.InDegree("B")
g.OutDegree("A")

// Attributes
g.NodeAttr("A")["label"] = "Node A"
g.EdgeAttr("A", "B")["weight"] = 1.5

// Copy and subgraph
g2 := g.Copy()
sub := g.Subgraph([]string{"A", "B"})
```

## Graph (Undirected)

```go
g := graphgo.NewGraph()

g.AddNode("A")
g.AddEdge("A", "B")     // creates nodes if needed
g.HasEdge("A", "B")     // true
g.HasEdge("B", "A")     // true (undirected)
g.Edges()               // []UndirectedEdge (each edge once)
g.Neighbors("A")        // all adjacent nodes
g.Degree("A")

g.RemoveEdge("A", "B")
g.RemoveNode("A")

g2 := g.Copy()
```

## PDAG (Partially Directed Acyclic Graph)

PDAGs represent Markov equivalence classes (CPDAGs) in causal inference, supporting both directed and undirected edges.

```go
p := graphgo.NewPDAG()

p.AddNodes("A", "B", "C")
p.AddDirectedEdge("A", "C")    // A -> C
p.AddUndirectedEdge("A", "B")  // A -- B

p.HasDirectedEdge("A", "C")    // true
p.HasUndirectedEdge("A", "B")  // true
p.HasEdge("A", "B")            // true (any edge type)
p.Adjacent("A", "B")           // true

p.DirectedEdges()               // [][2]string directed edges
p.UndirectedEdges()             // [][2]string undirected edges (each once)
p.Neighbors("A")                // all adjacent nodes (any edge type)
p.Parents("C")                  // directed predecessors
p.Children("A")                 // directed successors
p.Skeleton()                    // undirected Graph ignoring directions

p.RemoveDirectedEdge("A", "C")
p.RemoveUndirectedEdge("A", "B")
p.RemoveNode("A")

p2 := p.Copy()
```

## Meek Rules and DAG-to-CPDAG Conversion

Convert a DAG to its CPDAG (Completed PDAG) representing the Markov equivalence class.

```go
dag := graphgo.NewDiGraph()
dag.AddEdge("A", "C")
dag.AddEdge("B", "C")
dag.AddEdge("C", "D")

// Convert DAG to CPDAG: v-structures stay directed, other edges become undirected
cpdag := graphgo.DAGToPDAG(dag)

// Apply Meek rules to orient additional undirected edges
// (already called internally by DAGToPDAG, but can be applied independently)
changed := graphgo.ApplyMeekRules(cpdag)
```

Meek rules orient undirected edges while preserving acyclicity and the equivalence class:
- **R1**: Orient u--v into u->v if there exists w->u and w is not adjacent to v.
- **R2**: Orient u--v into u->v if there exists a chain u->w->v.
- **R3**: Orient u--v into u->v if there exist w1--u, w2--u, w1->v, w2->v, and w1 is not adjacent to w2.
- **R4**: Orient u--v into u->v if there exist w--u, w->x->v.

## DAG Algorithms

```go
dag := graphgo.NewDiGraph()
dag.AddEdge("A", "B")
dag.AddEdge("B", "C")
dag.AddEdge("A", "C")

// Check if the graph is a DAG
graphgo.IsDAG(dag)  // true

// Topological sort (Kahn's algorithm)
order, err := graphgo.TopologicalSort(dag)
// order = ["A", "B", "C"] (one valid ordering)
// err != nil if graph has a cycle

// Ancestors: all nodes with a directed path to the target
anc := graphgo.Ancestors(dag, "C")  // {"A": true, "B": true}

// Descendants: all nodes reachable from the source
desc := graphgo.Descendants(dag, "A")  // {"B": true, "C": true}
```

## D-Separation and Markov Blanket

```go
// D-separation: test conditional independence in a Bayesian network
dag := graphgo.NewDiGraph()
dag.AddEdge("A", "C")
dag.AddEdge("B", "C")
dag.AddEdge("C", "D")

x := map[string]bool{"A": true}
y := map[string]bool{"B": true}
z := map[string]bool{}  // no conditioning

// A and B are d-separated (independent) given empty set
graphgo.DSeparation(dag, x, y, z)  // true

// A and B are NOT d-separated given {C} (explaining away)
z = map[string]bool{"C": true}
graphgo.DSeparation(dag, x, y, z)  // false

// Markov blanket: parents + children + co-parents
blanket := graphgo.MarkovBlanket(dag, "C")
// {"A": true, "B": true, "D": true}
```

## Moralization and Triangulation

```go
dag := graphgo.NewDiGraph()
dag.AddEdge("A", "C")
dag.AddEdge("B", "C")

// Moralize: marry co-parents and drop edge directions
moral := graphgo.Moralize(dag)
// moral has edges: A--C, B--C, A--B (co-parents married)

// Triangulate: make a graph chordal using elimination ordering
order := []string{"A", "B", "C"}
tri := graphgo.Triangulate(moral, order)

// Check if a graph is chordal (uses maximum cardinality search)
graphgo.IsChordal(tri)  // true
```

## Clique Finding and Junction Trees

```go
g := graphgo.NewGraph()
g.AddEdge("A", "B")
g.AddEdge("B", "C")
g.AddEdge("A", "C")
g.AddEdge("C", "D")

// Find all maximal cliques (Bron-Kerbosch with pivoting)
cliques := graphgo.MaxCliques(g)
// [["A", "B", "C"], ["C", "D"]]

// Build a junction tree (clique tree) from cliques
// Uses maximum spanning tree (Kruskal's) on clique intersection weights
tree, separators := graphgo.BuildJunctionTree(cliques)
// tree: undirected Graph with nodes "0", "1", ...
// separators: map from edge key to separator set (shared variables)
```

## API Summary

| Category | Types / Functions |
|---|---|
| **Directed Graph** | `DiGraph`, `NewDiGraph`, `Edge` |
| **DiGraph Methods** | `AddNode`, `AddNodes`, `AddEdge`, `RemoveNode`, `RemoveEdge`, `HasNode`, `HasEdge`, `Nodes`, `Edges`, `Predecessors`, `Successors`, `Parents`, `Children`, `InDegree`, `OutDegree`, `NumberOfNodes`, `NumberOfEdges`, `NodeAttr`, `EdgeAttr`, `Copy`, `Subgraph` |
| **Undirected Graph** | `Graph`, `NewGraph`, `UndirectedEdge` |
| **Graph Methods** | `AddNode`, `AddEdge`, `RemoveNode`, `RemoveEdge`, `HasNode`, `HasEdge`, `Nodes`, `Edges`, `Neighbors`, `Degree`, `Copy` |
| **PDAG** | `PDAG`, `NewPDAG` |
| **PDAG Methods** | `AddNode`, `AddNodes`, `AddDirectedEdge`, `AddUndirectedEdge`, `RemoveNode`, `RemoveDirectedEdge`, `RemoveUndirectedEdge`, `HasNode`, `HasDirectedEdge`, `HasUndirectedEdge`, `HasEdge`, `Adjacent`, `Nodes`, `DirectedEdges`, `UndirectedEdges`, `Neighbors`, `Parents`, `Children`, `Skeleton`, `Copy` |
| **Meek Rules** | `DAGToPDAG`, `ApplyMeekRules` |
| **DAG Algorithms** | `IsDAG`, `TopologicalSort`, `Ancestors`, `Descendants` |
| **D-Separation** | `DSeparation`, `MarkovBlanket` |
| **Moralization** | `Moralize`, `Triangulate`, `IsChordal` |
| **Cliques** | `MaxCliques`, `BuildJunctionTree` |
