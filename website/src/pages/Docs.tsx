import { createElement, useHead, Link } from "@asymmetric-effort/specifyjs";

export function Docs() {
  useHead({
    title: "Documentation — pgmgo",
    description: "Comprehensive documentation for pgmgo, a zero-dependency Go library for probabilistic graphical models. Installation, architecture, packages, CLI, file formats, and more.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/docs",
  });

  return (
    <div class="page">
      <h1>Documentation</h1>

      <nav class="page-toc">
        <strong>On this page:</strong>{" "}
        <a href="#overview">Overview</a> | <a href="#installation">Installation</a> | <a href="#getting-started">Getting Started</a> | <a href="#architecture">Architecture</a> | <a href="#library-packages">Library Packages</a> | <a href="#core-packages">Core Packages</a> | <a href="#cli-reference">CLI Reference</a> | <a href="#file-formats">File Formats</a> | <a href="#example-models">Example Models</a> | <a href="#datasets">Datasets</a> | <a href="#testing">Testing</a> | <a href="#configuration">Configuration</a> | <a href="#contributing">Contributing</a>
      </nav>

      {/* ============================================================ */}
      {/* OVERVIEW */}
      {/* ============================================================ */}
      <section class="section" id="overview">
        <h2>Overview</h2>
        <p>
          <strong>pgmgo</strong> is a zero-dependency Go library for probabilistic graphical models (PGMs).
          It aims for feature parity with <a href="https://pgmpy.org" target="_blank" rel="noopener noreferrer">pgmpy</a>,
          the popular Python library, providing tools for creating, parameterizing, learning, and performing inference on
          Bayesian networks, Markov networks, and related structures.
        </p>
        <p>
          All numerical primitives -- linear algebra, statistical distributions, graph algorithms, and tabular data
          processing -- are implemented from scratch in pure Go. There are no C bindings, no cgo, no third-party
          module dependencies. The result is a library that compiles to a single static binary with <code>go build</code>,
          cross-compiles trivially, and deploys without runtime dependencies.
        </p>
        <p>
          The current release is <strong>v0.0.37</strong> with approximately 5,000 tests and 392 cross-validation
          fixtures covering inference, learning, sampling, serialization, and cross-validation across 24 packages.
        </p>

        <h3>Key Capabilities</h3>
        <ul>
          <li><strong>13 model types</strong>: Bayesian networks, Markov networks, dynamic BNs, naive Bayes, SEMs, factor graphs, junction trees, and more</li>
          <li><strong>7 inference algorithms</strong>: Variable Elimination, Belief Propagation, MPLP, approximate inference, causal do-calculus, DBN inference, MAP queries</li>
          <li><strong>15+ learning algorithms</strong>: MLE, Bayesian estimation, EM, hill-climbing, PC, GES, exhaustive search, tree search, MMHC, expert-in-the-loop, and more</li>
          <li><strong>16 conditional independence tests</strong>: Chi-squared, G-squared, Fisher-Z, Pearson, GCM, Hotelling-Lawley, and more</li>
          <li><strong>13 structure scoring functions</strong>: BIC, AIC, BDeu, BDs, K2, log-likelihood, Gaussian, conditional Gaussian</li>
          <li><strong>10 file formats</strong>: BIF, XMLBIF, NET, UAI, XDSL, PomdpX, XBN, CSV, JSON, XML</li>
          <li><strong>25 built-in example models</strong> and <strong>40 built-in datasets</strong></li>
          <li><strong>GPU acceleration</strong> for compute-intensive operations</li>
          <li><strong>LLM integration</strong> for expert-in-the-loop structure learning</li>
          <li><strong>Full CLI</strong> with 10 commands for model validation, inference, learning, sampling, and more</li>
        </ul>

        <h3>Relationship to pgmpy</h3>
        <p>
          pgmgo is inspired by <a href="https://pgmpy.org" target="_blank" rel="noopener noreferrer">pgmpy</a> and
          follows similar API patterns where possible. If you have used pgmpy in Python, the concepts and workflows
          in pgmgo will feel familiar. However, pgmgo is not a direct port -- it is a ground-up reimplementation
          in Go with its own design decisions, particularly around the zero-dependency philosophy.
        </p>
        <p>
          Where pgmpy relies on numpy, scipy, networkx, and pandas, pgmgo provides its own equivalents:
          <code>numgo</code>, <code>scigo</code>, <code>graphgo</code>, and <code>tabgo</code>. These are general-purpose
          libraries that can be used independently of the PGM layer.
        </p>
      </section>

      {/* ============================================================ */}
      {/* INSTALLATION */}
      {/* ============================================================ */}
      <section class="section" id="installation">
        <h2>Installation</h2>

        <h3>Go Module</h3>
        <pre><code>go get github.com/asymmetric-effort/pgmgo</code></pre>
        <p>
          Requires <strong>Go 1.21</strong> or later. No C compiler needed, no system libraries needed.
          Works on Linux, macOS, and Windows. Cross-compilation works out of the box.
        </p>

        <h3>CLI Binary</h3>
        <pre><code>{`# Install the CLI tool
go install github.com/asymmetric-effort/pgmgo/cmd/pgmgo@latest

# Verify installation
pgmgo --help`}</code></pre>

        <h3>From Source</h3>
        <pre><code>{`git clone https://github.com/asymmetric-effort/pgmgo.git
cd pgmgo
go build ./cmd/pgmgo
./pgmgo --help`}</code></pre>

        <h3>Verify</h3>
        <pre><code>{`# Run all tests to verify the installation
go test ./...

# Run a quick sanity check
go test ./src/models/... -run TestBayesianNetwork -v`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* GETTING STARTED */}
      {/* ============================================================ */}
      <section class="section" id="getting-started">
        <h2>Getting Started</h2>

        <h3>Your First Bayesian Network</h3>
        <p>
          A Bayesian network is a directed acyclic graph (DAG) where nodes represent random variables and
          edges represent conditional dependencies. Here is a complete, runnable program that creates the
          classic "wet grass" network:
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

func main() {
    // Create an empty Bayesian network
    bn := models.NewBayesianNetwork()

    // Add nodes (random variables)
    bn.AddNode("Cloudy")
    bn.AddNode("Rain")
    bn.AddNode("Sprinkler")
    bn.AddNode("WetGrass")

    // Add directed edges (causal relationships)
    bn.AddEdge("Cloudy", "Rain")
    bn.AddEdge("Cloudy", "Sprinkler")
    bn.AddEdge("Rain", "WetGrass")
    bn.AddEdge("Sprinkler", "WetGrass")

    // Define states for each variable
    bn.SetStates("Cloudy", []string{"clear", "cloudy"})
    bn.SetStates("Rain", []string{"no", "yes"})
    bn.SetStates("Sprinkler", []string{"off", "on"})
    bn.SetStates("WetGrass", []string{"dry", "wet"})

    // P(Cloudy) -- root node, marginal distribution
    bn.SetCPD("Cloudy", factors.NewTabularCPD(
        "Cloudy", 2,
        []float64{0.5, 0.5},
        nil, nil,
    ))

    // P(Rain | Cloudy) -- conditional distribution
    bn.SetCPD("Rain", factors.NewTabularCPD(
        "Rain", 2,
        []float64{0.8, 0.2, 0.2, 0.8},
        []string{"Cloudy"}, []int{2},
    ))

    // P(Sprinkler | Cloudy)
    bn.SetCPD("Sprinkler", factors.NewTabularCPD(
        "Sprinkler", 2,
        []float64{0.5, 0.9, 0.5, 0.1},
        []string{"Cloudy"}, []int{2},
    ))

    // P(WetGrass | Sprinkler, Rain)
    bn.SetCPD("WetGrass", factors.NewTabularCPD(
        "WetGrass", 2,
        []float64{
            1.0, 0.1, 0.1, 0.01,
            0.0, 0.9, 0.9, 0.99,
        },
        []string{"Sprinkler", "Rain"}, []int{2, 2},
    ))

    // Validate: checks DAG, CPD dimensions, probability sums
    if err := bn.CheckModel(); err != nil {
        log.Fatal("Model error:", err)
    }
    fmt.Println("Model is valid!")
    fmt.Printf("Nodes: %d, Edges: %d\\n", len(bn.Nodes()), len(bn.Edges()))
}`}</code></pre>

        <h3>Your First Query</h3>
        <p>
          Once you have a valid model, convert its CPDs to Markov factors and use Variable Elimination
          to compute posterior probabilities:
        </p>
        <pre><code>{`import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/inference"
)

func main() {
    // Load a built-in model (13 models have full CPDs)
    bn, _ := example_models.Get("asia")

    // Convert CPDs to Markov factors
    facs, err := bn.ToMarkovFactors()
    if err != nil {
        log.Fatal(err)
    }

    // Create a Variable Elimination engine
    ve := inference.NewVariableElimination(facs)

    // Posterior query: P(Dyspnea | Smoker=yes)
    result, err := ve.Query(
        []string{"Dyspnea"},           // query variables
        map[string]int{"Smoker": 1},   // evidence (1 = "yes")
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Dyspnea | Smoker=yes):", result.Values().Data())

    // MAP query: most likely assignment
    assignment, err := ve.MAP(
        []string{"Lung", "Bronc"},
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("MAP(Lung, Bronc | Smoker=yes):", assignment)
}`}</code></pre>

        <h3>Your First Structure Learning</h3>
        <p>
          When you have observational data but no known structure, use structure learning to discover the DAG:
        </p>
        <pre><code>{`import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/learning"
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/src/structure_score"
)

func main() {
    // Generate training data from a known model
    bn, _ := example_models.Get("asia")
    bms, _ := sampling.NewBayesianModelSampling(bn, 42)
    data, _ := bms.ForwardSample(5000)

    // Learn structure using hill-climbing with BIC scoring
    scorer := structure_score.NewBIC()
    hc := learning.NewHillClimbSearch(data, scorer.LocalScore)
    learnedBN, err := hc.Estimate()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Learned: %d nodes, %d edges\\n",
        len(learnedBN.Nodes()), len(learnedBN.Edges()))

    // Fit parameters to the learned structure
    mle := learning.NewMLE(learnedBN, data)
    if err := mle.Estimate(); err != nil {
        log.Fatal(err)
    }
    fmt.Println("Parameters fitted successfully")
}`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* ARCHITECTURE */}
      {/* ============================================================ */}
      <section class="section" id="architecture">
        <h2>Architecture</h2>

        <h3>Project Layout</h3>
        <pre><code>{`pgmgo/
  cmd/
    pgmgo/                 # CLI entry point (10 commands)
  lib/                     # Internal primitive modules (zero-dep foundations)
    numgo/                 # numpy equivalent: NDArray, Matrix, Vector
    scigo/                 # scipy equivalent: distributions, optimization, special functions
    graphgo/               # networkx equivalent: DiGraph, PDAG, graph algorithms
    tabgo/                 # pandas equivalent: DataFrame, Series, CSV/Parquet I/O
    gpu/                   # GPU compute backend for large-scale operations
  src/                     # Core PGM library
    base/                  # DAG, PDAG, UndirectedGraph, ADMG, MAG, SimpleCausalModel
    models/                # 13 probabilistic model types
    factors/               # Factor representations: TabularCPD, DiscreteFactor, etc.
    inference/             # 7 inference algorithms (VE, BP, MPLP, Causal, etc.)
    sampling/              # Forward, rejection, likelihood-weighted, Gibbs sampling
    learning/              # 15+ learning algorithms (parameter + structure)
    ci_tests/              # 16 conditional independence tests
    structure_score/       # 13 scoring functions for structure learning
    identification/        # Causal effect identification (back-door, front-door)
    prediction/            # DoubleML, naive adjustment, IV regression
    metrics/               # SHD, confusion matrices, correlation, Fisher's C
    independencies/        # Independence assertion representations
    readwrite/             # 10 file format readers and writers
    config/                # Global configuration
    utils/                 # Shared parsing, optimization, compatibility utilities
  example_models/          # 25 built-in Bayesian networks
  examples/
    datasets/              # 40 built-in CSV datasets
  website/                 # Project website (this site)
  docs/                    # Additional documentation`}</code></pre>

        <h3>Dependency Flow</h3>
        <p>
          The dependency flow is strictly layered. Each layer depends only on layers below it:
        </p>
        <pre><code>{`Layer 4: cmd/pgmgo          (CLI application)
         |
Layer 3: src/*              (PGM domain: models, inference, learning, etc.)
         |
Layer 2: lib/*              (primitives: numgo, scigo, graphgo, tabgo, gpu)
         |
Layer 1: Go standard library (only dependency)`}</code></pre>
        <p>
          The <code>lib/</code> packages are general-purpose and can be used independently. For example,
          you could use <code>numgo</code> for matrix math or <code>graphgo</code> for graph algorithms
          without importing any PGM-specific code.
        </p>
        <p>
          The <code>src/</code> packages build on <code>lib/</code> to implement PGM-specific functionality.
          They also depend on each other -- for example, <code>inference</code> depends on <code>factors</code>,
          and <code>learning</code> depends on <code>structure_score</code> and <code>ci_tests</code>.
        </p>
        <p>
          The <code>cmd/pgmgo</code> CLI is a thin wrapper that wires the <code>src/</code> packages into
          a command-line interface. It depends on everything but nothing depends on it.
        </p>

        <h3>Design Principles</h3>
        <ul>
          <li><strong>Zero dependencies:</strong> The entire library compiles with only the Go standard library. No cgo, no system libraries, no third-party modules.</li>
          <li><strong>pgmpy compatibility:</strong> API patterns follow pgmpy where practical, making it easier for Python PGM practitioners to transition to Go.</li>
          <li><strong>Cross-validation:</strong> 392 test fixtures validate results against known-correct outputs, ensuring numerical accuracy across inference, learning, and sampling.</li>
          <li><strong>Layered architecture:</strong> Primitive libraries (numgo, scigo, graphgo, tabgo) are reusable beyond PGMs. The PGM layer builds on these without polluting them.</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* LIBRARY PACKAGES */}
      {/* ============================================================ */}
      <section class="section" id="library-packages">
        <h2>Library Packages (lib/)</h2>
        <p>
          These packages replace common Python scientific computing libraries with pure Go implementations.
          They are general-purpose and can be imported independently of the PGM layer.
        </p>

        <h3>numgo -- numpy equivalent</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/lib/numgo</code>
        </p>
        <p>
          N-dimensional arrays, linear algebra, matrix operations, broadcasting, and element-wise arithmetic.
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>NDArray</code></td><td>N-dimensional array with shape, stride, and element-wise operations (add, multiply, etc.)</td></tr>
            <tr><td><code>Matrix</code></td><td>2D matrix with multiply, transpose, inverse, determinant, eigenvalues</td></tr>
            <tr><td><code>Vector</code></td><td>1D vector with dot product, norm, element-wise operations</td></tr>
          </tbody>
        </table>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/lib/numgo"

// Create a matrix
m := numgo.NewMatrix(3, 3)
m.Set(0, 0, 1.0)
m.Set(1, 1, 2.0)
m.Set(2, 2, 3.0)

// Matrix operations
det := m.Det()
inv := m.Inverse()
transposed := m.Transpose()
product := m.Multiply(inv) // should be identity

// NDArray operations
arr := numgo.NewNDArray([]int{2, 3, 4}) // 2x3x4 array
arr.Fill(1.0)
sum := arr.Sum()

// Vector operations
v1 := numgo.NewVector([]float64{1, 2, 3})
v2 := numgo.NewVector([]float64{4, 5, 6})
dot := v1.Dot(v2)
norm := v1.Norm()`}</code></pre>

        <h3>scigo -- scipy equivalent</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/lib/scigo</code>
        </p>
        <p>
          Statistical distributions, optimization routines, special functions, and hypothesis tests.
        </p>
        <table>
          <thead>
            <tr><th>Category</th><th>Key Types / Functions</th></tr>
          </thead>
          <tbody>
            <tr><td>Distributions</td><td><code>Normal</code>, <code>ChiSquared</code>, <code>Beta</code>, <code>Gamma</code>, <code>StudentT</code>, <code>Uniform</code>, <code>Exponential</code></td></tr>
            <tr><td>Optimization</td><td><code>Minimize</code>, <code>GradientDescent</code>, <code>NewtonMethod</code></td></tr>
            <tr><td>Statistics</td><td>Hypothesis tests, p-value computation, quantile functions, CDF/PDF/PPF</td></tr>
            <tr><td>Special Functions</td><td>Gamma function, beta function, incomplete gamma, digamma</td></tr>
          </tbody>
        </table>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/lib/scigo"

// Normal distribution
n := scigo.NewNormal(0, 1) // mean=0, std=1
pdf := n.PDF(1.96)
cdf := n.CDF(1.96)       // ~0.975
ppf := n.PPF(0.975)      // ~1.96

// Chi-squared distribution (used in CI tests)
chi2 := scigo.NewChiSquared(5) // 5 degrees of freedom
pValue := 1.0 - chi2.CDF(11.07)

// Optimization
result := scigo.Minimize(func(x float64) float64 {
    return (x - 3) * (x - 3)
}, 0.0, 10.0)`}</code></pre>

        <h3>graphgo -- networkx equivalent</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/lib/graphgo</code>
        </p>
        <p>
          Directed and undirected graphs with a full suite of graph algorithms.
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DiGraph</code></td><td>Directed graph with adjacency operations, successors, predecessors</td></tr>
            <tr><td><code>Graph</code></td><td>Undirected graph with neighbors, degree, connected components</td></tr>
            <tr><td><code>PDAG</code></td><td>Partially directed acyclic graph (for equivalence classes)</td></tr>
          </tbody>
        </table>
        <p>
          Algorithms: topological sort, d-separation, moral graph, triangulation, maximum cardinality search,
          clique finding, connected components, shortest paths, cycle detection.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/lib/graphgo"

// Create a directed graph
g := graphgo.NewDiGraph()
g.AddNode("A")
g.AddNode("B")
g.AddNode("C")
g.AddEdge("A", "B")
g.AddEdge("B", "C")

// Graph queries
parents := g.Predecessors("C")  // ["B"]
children := g.Successors("A")   // ["B"]
sorted := g.TopologicalSort()   // ["A", "B", "C"]

// Undirected graph
ug := graphgo.NewGraph()
ug.AddEdge("X", "Y")
ug.AddEdge("Y", "Z")
neighbors := ug.Neighbors("Y")  // ["X", "Z"]`}</code></pre>

        <h3>tabgo -- pandas equivalent</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/lib/tabgo</code>
        </p>
        <p>
          Tabular data with named columns, row filtering, groupby, and file I/O.
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DataFrame</code></td><td>Tabular data with named columns, row filtering, groupby, merge</td></tr>
            <tr><td><code>Series</code></td><td>Single column with value counts, unique values, statistical summaries</td></tr>
          </tbody>
        </table>
        <p>
          I/O: <code>ReadCSV</code>, <code>WriteCSV</code>, <code>ReadParquet</code>, <code>WriteParquet</code>, <code>ReadXLSX</code>.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/lib/tabgo"

// Read CSV data
df, err := tabgo.ReadCSV("observations.csv")
fmt.Printf("Rows: %d, Columns: %d\\n", df.NRows(), len(df.Columns()))

// Access a column as a Series
col := df.Column("Temperature")
fmt.Println("Unique values:", col.Unique())
fmt.Println("Value counts:", col.ValueCounts())

// Filter rows
filtered := df.Filter(func(row map[string]interface{}) bool {
    return row["Temperature"].(int) > 70
})

// Write CSV
tabgo.WriteCSV(filtered, "warm_days.csv")`}</code></pre>

        <h3>gpu -- GPU Compute Backend</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/lib/gpu</code>
        </p>
        <p>
          Optional GPU acceleration for compute-intensive operations on large networks. Provides GPU-backed
          matrix operations and factor computations that can significantly speed up inference and learning
          on networks with many nodes.
        </p>
      </section>

      {/* ============================================================ */}
      {/* CORE PACKAGES */}
      {/* ============================================================ */}
      <section class="section" id="core-packages">
        <h2>Core Packages (src/)</h2>

        <h3>base -- Foundational Graph Types</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/base</code>
        </p>
        <p>
          Provides the underlying graph structures that all model types are built on.
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DAG</code></td><td>Directed acyclic graph with cycle detection, topological sort, d-separation</td></tr>
            <tr><td><code>PDAG</code></td><td>Partially directed acyclic graph for Markov equivalence classes</td></tr>
            <tr><td><code>UndirectedGraph</code></td><td>Undirected graph for Markov networks</td></tr>
            <tr><td><code>ADMG</code></td><td>Acyclic directed mixed graph (with bidirected edges for latent confounders)</td></tr>
            <tr><td><code>MAG</code></td><td>Maximal ancestral graph</td></tr>
            <tr><td><code>SimpleCausalModel</code></td><td>Basic causal model with intervention semantics</td></tr>
          </tbody>
        </table>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/base"

dag := base.NewDAG()
dag.AddNode("X")
dag.AddNode("Y")
dag.AddNode("Z")
dag.AddEdge("X", "Y")
dag.AddEdge("Y", "Z")

// Check d-separation: X _||_ Z | Y?
separated := dag.DSeparation([]string{"X"}, []string{"Z"}, []string{"Y"})
fmt.Println("X _||_ Z | Y:", separated) // true`}</code></pre>

        <h3>models -- Probabilistic Model Types</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/models</code>
        </p>
        <p>13 model types for different PGM use cases:</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th><th>Constructor</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BayesianNetwork</code></td><td>DAG with TabularCPDs. The primary model type for most use cases.</td><td><code>NewBayesianNetwork()</code></td></tr>
            <tr><td><code>MarkovNetwork</code></td><td>Undirected graphical model with factor potentials.</td><td><code>NewMarkovNetwork()</code></td></tr>
            <tr><td><code>DynamicBayesianNetwork</code></td><td>BN over time slices for temporal modeling.</td><td><code>NewDynamicBayesianNetwork()</code></td></tr>
            <tr><td><code>NaiveBayes</code></td><td>Naive Bayes classifier (BN with single class parent).</td><td><code>NewNaiveBayes()</code></td></tr>
            <tr><td><code>SEM</code></td><td>Structural Equation Model with linear/nonlinear equations.</td><td><code>NewSEM()</code></td></tr>
            <tr><td><code>FactorGraph</code></td><td>Bipartite graph of variable nodes and factor nodes.</td><td><code>NewFactorGraph()</code></td></tr>
            <tr><td><code>JunctionTree</code></td><td>Clique tree for exact inference via message passing.</td><td><code>NewJunctionTree()</code></td></tr>
            <tr><td><code>ClusterGraph</code></td><td>Generalized cluster graph (superset of junction tree).</td><td><code>NewClusterGraph()</code></td></tr>
            <tr><td><code>LinearGaussianBN</code></td><td>BN with continuous, linearly-related Gaussian variables.</td><td><code>NewLinearGaussianBN()</code></td></tr>
            <tr><td><code>FunctionalBN</code></td><td>BN where CPDs are defined by arbitrary functions.</td><td><code>NewFunctionalBN()</code></td></tr>
            <tr><td><code>MarkovChain</code></td><td>First-order Markov chain for sequential data.</td><td><code>NewMarkovChain()</code></td></tr>
            <tr><td><code>DiscreteBayesianNetwork</code></td><td>Specialized discrete-only BN with optimized operations.</td><td><code>NewDiscreteBayesianNetwork()</code></td></tr>
            <tr><td><code>DiscreteMarkovNetwork</code></td><td>Specialized discrete-only Markov network.</td><td><code>NewDiscreteMarkovNetwork()</code></td></tr>
          </tbody>
        </table>
        <p>
          Key methods shared by most model types: <code>AddNode</code>, <code>AddEdge</code>, <code>Nodes()</code>,
          <code>Edges()</code>, <code>SetStates</code>, <code>SetCPD</code>, <code>CheckModel()</code>,
          <code>ToMarkovFactors()</code>, <code>ToJunctionTree()</code>.
        </p>

        <h3>factors -- Factor Representations</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/factors</code>
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th><th>Constructor</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DiscreteFactor</code></td><td>General discrete factor with product, marginalize, reduce, normalize operations.</td><td><code>NewDiscreteFactor()</code></td></tr>
            <tr><td><code>TabularCPD</code></td><td>Conditional probability distribution table. The standard CPD type for BayesianNetwork.</td><td><code>NewTabularCPD()</code></td></tr>
            <tr><td><code>JointProbabilityDistribution</code></td><td>Full joint distribution over a set of variables.</td><td><code>NewJPD()</code></td></tr>
            <tr><td><code>LinearGaussianCPD</code></td><td>Linear Gaussian conditional: child = sum(beta_i * parent_i) + noise.</td><td><code>NewLinearGaussianCPD()</code></td></tr>
            <tr><td><code>NoisyOR</code></td><td>Noisy-OR parameterization. Compact CPD for nodes with many binary parents.</td><td><code>NewNoisyOR()</code></td></tr>
          </tbody>
        </table>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/factors"

// Create a TabularCPD: P(B | A)
// A has 2 states, B has 2 states
cpd := factors.NewTabularCPD(
    "B", 2,
    []float64{0.3, 0.7, 0.8, 0.2},  // P(b0|a0)=0.3, P(b1|a0)=0.7, P(b0|a1)=0.8, P(b1|a1)=0.2
    []string{"A"}, []int{2},
)

// Factor operations
f1 := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
f2 := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.5, 0.6, 0.7, 0.8})
product := f1.Product(f2)           // factor product
marginal := product.Marginalize([]string{"B"})  // sum out B
reduced := product.Reduce(map[string]int{"A": 0})  // fix A=0
normalized := marginal.Normalize()  // normalize to sum to 1`}</code></pre>

        <h3>inference -- Inference Algorithms</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/inference</code>
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th><th>Constructor</th></tr>
          </thead>
          <tbody>
            <tr><td><code>VariableElimination</code></td><td>Exact inference via factor elimination. Supports <code>Query()</code> for posteriors and <code>MAP()</code> for most-probable assignments.</td><td><code>NewVariableElimination(factors)</code></td></tr>
            <tr><td><code>BeliefPropagation</code></td><td>Message-passing on junction trees. Calibrate once, query multiple times.</td><td><code>NewBeliefPropagation(cliques, separators, factors)</code></td></tr>
            <tr><td><code>MPLP</code></td><td>Max-Product Linear Programming for MAP inference.</td><td><code>NewMPLP(factors)</code></td></tr>
            <tr><td><code>ApproxInference</code></td><td>Sampling-based approximate inference. Uses likelihood-weighted sampling.</td><td><code>NewApproxInference(bn, nSamples)</code></td></tr>
            <tr><td><code>CausalInference</code></td><td>Do-calculus interventional queries. Computes P(Y | do(X=x)).</td><td><code>NewCausalInference(bn)</code></td></tr>
            <tr><td><code>DBNInference</code></td><td>Inference over dynamic Bayesian networks across time slices.</td><td><code>NewDBNInference(dbn)</code></td></tr>
          </tbody>
        </table>

        <h3>sampling -- Sampling Methods</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/sampling</code>
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Methods</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BayesianModelSampling</code></td><td><code>ForwardSample</code>, <code>RejectionSample</code>, <code>LikelihoodWeightedSample</code></td><td>Exact and weighted sampling from BN joint distribution.</td></tr>
            <tr><td><code>GibbsSampling</code></td><td><code>Sample</code></td><td>MCMC Gibbs sampler with configurable burn-in and thinning.</td></tr>
          </tbody>
        </table>

        <h3>learning -- Learning Algorithms</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/learning</code>
        </p>
        <p><strong>Parameter Learning:</strong></p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>MLE</code></td><td>Maximum Likelihood Estimation. Computes CPD parameters from complete data by counting.</td></tr>
            <tr><td><code>BayesianEstimator</code></td><td>Bayesian parameter estimation with Dirichlet priors (BDeu). Handles small sample sizes gracefully.</td></tr>
            <tr><td><code>EM</code></td><td>Expectation-Maximization for incomplete data (missing values).</td></tr>
            <tr><td><code>LinearGaussianMLE</code></td><td>MLE for linear Gaussian BNs (continuous variables).</td></tr>
            <tr><td><code>MarginalEstimator</code></td><td>Marginal likelihood estimation.</td></tr>
            <tr><td><code>MirrorDescent</code></td><td>Mirror descent optimization for parameter learning.</td></tr>
          </tbody>
        </table>
        <p><strong>Structure Learning:</strong></p>
        <table>
          <thead>
            <tr><th>Type</th><th>Category</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>HillClimbSearch</code></td><td>Score-based</td><td>Greedy hill-climbing. Iteratively adds, removes, or reverses edges to maximize a score.</td></tr>
            <tr><td><code>PC</code></td><td>Constraint-based</td><td>PC algorithm. Uses CI tests to discover skeleton, then orients edges.</td></tr>
            <tr><td><code>GES</code></td><td>Score-based</td><td>Greedy Equivalence Search. Searches equivalence classes of DAGs.</td></tr>
            <tr><td><code>ExhaustiveSearch</code></td><td>Score-based</td><td>Exhaustive DAG enumeration. Optimal but only feasible for small networks (up to ~5 nodes).</td></tr>
            <tr><td><code>TreeSearch</code></td><td>Score-based</td><td>Chow-Liu tree search. Learns tree-structured BNs.</td></tr>
            <tr><td><code>MMHC</code></td><td>Hybrid</td><td>Max-Min Hill-Climbing. Combines CI tests (for skeleton) with scoring (for orientation).</td></tr>
            <tr><td><code>ExpertInLoop</code></td><td>Interactive</td><td>Interactive structure learning with expert or LLM guidance.</td></tr>
          </tbody>
        </table>
        <p><strong>Causal Estimation:</strong></p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>IVEstimator</code></td><td>Instrumental variable estimation for causal effects.</td></tr>
            <tr><td><code>SEMEstimator</code></td><td>Structural equation model estimation.</td></tr>
            <tr><td><code>LLMClient</code></td><td>LLM client for AI-assisted structure elicitation.</td></tr>
          </tbody>
        </table>

        <h3>ci_tests -- Conditional Independence Tests</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/ci_tests</code>
        </p>
        <p>Used by constraint-based structure learning (PC, MMHC) to test whether two variables are conditionally independent given a set of conditioning variables.</p>
        <table>
          <thead>
            <tr><th>Category</th><th>Tests</th><th>Data Type</th></tr>
          </thead>
          <tbody>
            <tr><td>Discrete</td><td><code>ChiSquare</code>, <code>GSq</code>, <code>ModifiedChiSquare</code>, <code>PowerDivergence</code></td><td>Categorical data</td></tr>
            <tr><td>Continuous</td><td><code>FisherZ</code>, <code>Pearsonr</code>, <code>PartialCorrelation</code></td><td>Continuous data</td></tr>
            <tr><td>Multivariate</td><td><code>GCM</code>, <code>HotellingLawley</code>, <code>PillaiBartlett</code></td><td>Multivariate data</td></tr>
            <tr><td>Tree-based</td><td><code>TreeBasedCI</code></td><td>Any data</td></tr>
          </tbody>
        </table>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/ci_tests"

// Test X _||_ Y | Z using chi-square
statistic, pValue, independent := ci_tests.ChiSquare("X", "Y", []string{"Z"}, data, 0.05)
fmt.Printf("Chi-square=%.3f, p=%.4f, independent=%v\\n", statistic, pValue, independent)`}</code></pre>

        <h3>structure_score -- Scoring Functions</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/structure_score</code>
        </p>
        <p>Used by score-based structure learning (HillClimb, GES, ExhaustiveSearch) to evaluate candidate structures.</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th><th>When to Use</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BIC</code></td><td>Bayesian Information Criterion. Penalizes complexity by log(N).</td><td>Default choice. Good balance of fit and parsimony.</td></tr>
            <tr><td><code>AIC</code></td><td>Akaike Information Criterion. Less penalty than BIC.</td><td>When you prefer less penalization of complexity.</td></tr>
            <tr><td><code>BDeu</code></td><td>Bayesian Dirichlet equivalent uniform. Bayesian score with equivalent sample size.</td><td>When you want a Bayesian prior over parameters.</td></tr>
            <tr><td><code>BDs</code></td><td>Bayesian Dirichlet sparse. Favors sparse networks.</td><td>When you expect few edges.</td></tr>
            <tr><td><code>K2</code></td><td>K2 score. Fast, assumes uniform prior.</td><td>Quick evaluation, large networks.</td></tr>
            <tr><td><code>LogLikelihood</code></td><td>Raw log-likelihood (no penalty).</td><td>Comparing models of the same complexity.</td></tr>
            <tr><td><code>Gaussian</code></td><td>Score for continuous (Gaussian) data.</td><td>Continuous variables.</td></tr>
            <tr><td><code>ConditionalGaussian</code></td><td>Score for mixed discrete/continuous data.</td><td>Mixed data types.</td></tr>
          </tbody>
        </table>

        <h3>identification -- Causal Effect Identification</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/identification</code>
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Adjustment</code></td><td>Back-door criterion. Finds adjustment sets that block all back-door paths.</td></tr>
            <tr><td><code>Frontdoor</code></td><td>Front-door criterion. Identifies causal effects when back-door is blocked by unobserved confounders.</td></tr>
          </tbody>
        </table>

        <h3>prediction -- Causal Prediction</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/prediction</code>
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DoubleMLRegressor</code></td><td>Double/Debiased Machine Learning for causal effect estimation. Handles high-dimensional confounders.</td></tr>
            <tr><td><code>NaiveAdjustmentRegressor</code></td><td>Naive back-door adjustment regression. Simple but requires correct adjustment set.</td></tr>
            <tr><td><code>NaiveIVRegressor</code></td><td>Instrumental variable regression. Uses instruments to estimate causal effects with unmeasured confounders.</td></tr>
          </tbody>
        </table>

        <h3>metrics -- Model Evaluation</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/metrics</code>
        </p>
        <table>
          <thead>
            <tr><th>Function</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>SHD</code></td><td>Structural Hamming Distance. Counts edge additions, deletions, and reversals between two DAGs.</td></tr>
            <tr><td><code>AdjacencyConfusionMatrix</code></td><td>TP, FP, TN, FN for edge presence (ignoring direction).</td></tr>
            <tr><td><code>OrientationConfusionMatrix</code></td><td>TP, FP, TN, FN for edge orientation (given correct adjacency).</td></tr>
            <tr><td><code>CorrelationScore</code></td><td>Correlation-based model scoring.</td></tr>
            <tr><td><code>FisherC</code></td><td>Fisher's C statistic for overall model fit.</td></tr>
            <tr><td><code>LogLikelihoodScore</code></td><td>Log-likelihood of data given model.</td></tr>
            <tr><td><code>NormalizedSHD</code></td><td>SHD normalized by the maximum possible SHD.</td></tr>
          </tbody>
        </table>

        <h3>independencies -- Independence Assertions</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/independencies</code>
        </p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>IndependenceAssertion</code></td><td>A single assertion: X _||_ Y | Z.</td></tr>
            <tr><td><code>Independencies</code></td><td>A collection of independence assertions. Can be derived from a BN or specified manually.</td></tr>
          </tbody>
        </table>

        <h3>readwrite -- File Format I/O</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/readwrite</code>
        </p>
        <p>See the <a href="#file-formats">File Formats</a> section below for details on each format.</p>
        <table>
          <thead>
            <tr><th>Functions</th><th>Format</th></tr>
          </thead>
          <tbody>
            <tr><td><code>ReadBIF</code> / <code>WriteBIF</code></td><td>Bayesian Interchange Format</td></tr>
            <tr><td><code>ReadXMLBIF</code> / <code>WriteXMLBIF</code></td><td>XML-based BIF</td></tr>
            <tr><td><code>ReadNET</code> / <code>WriteNET</code></td><td>Hugin NET</td></tr>
            <tr><td><code>ReadUAI</code> / <code>WriteUAI</code></td><td>UAI format</td></tr>
            <tr><td><code>ReadXDSL</code> / <code>WriteXDSL</code></td><td>GeNIe XDSL</td></tr>
            <tr><td><code>ReadPomdpX</code></td><td>POMDP XML (read only)</td></tr>
            <tr><td><code>ReadXBN</code></td><td>Microsoft XBN (read only)</td></tr>
            <tr><td><code>ReadCSVModel</code> / <code>WriteCSVModel</code></td><td>CSV model serialization</td></tr>
            <tr><td><code>ReadJSONModel</code> / <code>WriteJSONModel</code></td><td>JSON model serialization</td></tr>
            <tr><td><code>ReadXMLModel</code> / <code>WriteXMLModel</code></td><td>XML model serialization</td></tr>
          </tbody>
        </table>

        <h3>config -- Configuration</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/config</code>
        </p>
        <p>See the <a href="#configuration">Configuration</a> section below.</p>

        <h3>utils -- Shared Utilities</h3>
        <p>
          Import: <code>github.com/asymmetric-effort/pgmgo/src/utils</code>
        </p>
        <p>
          Shared parsing, optimization, and compatibility utilities used across packages. Includes
          helper functions for common operations like set manipulation, string parsing, and
          numerical utilities.
        </p>
      </section>

      {/* ============================================================ */}
      {/* CLI REFERENCE */}
      {/* ============================================================ */}
      <section class="section" id="cli-reference">
        <h2>CLI Reference</h2>
        <p>
          The <code>pgmgo</code> CLI provides 10 commands for working with probabilistic graphical models
          from the command line. Install it with:
        </p>
        <pre><code>go install github.com/asymmetric-effort/pgmgo/cmd/pgmgo@latest</code></pre>
        <pre><code>pgmgo [command] [options]</code></pre>

        <h3>Commands Summary</h3>
        <table>
          <thead>
            <tr><th>Command</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>validate</code></td><td>Validate a BIF model file</td></tr>
            <tr><td><code>query</code></td><td>Run probabilistic inference query</td></tr>
            <tr><td><code>map</code></td><td>Find MAP (most likely) assignment</td></tr>
            <tr><td><code>learn</code></td><td>Learn network structure from data</td></tr>
            <tr><td><code>fit</code></td><td>Fit parameters to data given a structure</td></tr>
            <tr><td><code>sample</code></td><td>Generate samples from a model</td></tr>
            <tr><td><code>info</code></td><td>Print model summary information</td></tr>
            <tr><td><code>convert</code></td><td>Convert between model file formats</td></tr>
            <tr><td><code>compare</code></td><td>Compare two network structures</td></tr>
            <tr><td><code>do</code></td><td>Causal do-calculus query</td></tr>
          </tbody>
        </table>

        <h3>validate</h3>
        <p>Validate a BIF model file. Parses the file and checks that the model is well-formed (all CPDs are consistent with the graph structure).</p>
        <pre><code>pgmgo validate &lt;file&gt;</code></pre>
        <pre><code>{`$ pgmgo validate asia.bif
model asia.bif is valid (8 nodes, 8 edges)

$ pgmgo validate broken.bif
error: model validation failed: CPD for "Lung" has inconsistent cardinality`}</code></pre>

        <h3>query</h3>
        <p>Run a probabilistic inference query. Computes posterior probabilities for query variables given observed evidence.</p>
        <pre><code>{`pgmgo query <file> --variables V1,V2 [--evidence E1=v1,E2=v2] [--method ve|bp|approx]`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--variables</code></td><td>(required)</td><td>Comma-separated query variables</td></tr>
            <tr><td><code>--evidence</code></td><td>(none)</td><td>Comma-separated evidence as KEY=VALUE pairs</td></tr>
            <tr><td><code>--method</code></td><td><code>ve</code></td><td>Inference method: <code>ve</code>, <code>bp</code>, <code>approx</code> (10,000 samples)</td></tr>
          </tbody>
        </table>
        <pre><code>{`# P(Dyspnea) using Variable Elimination
$ pgmgo query asia.bif --variables Dyspnea

# P(Lung, Bronc | Smoker=1) using Belief Propagation
$ pgmgo query asia.bif --variables Lung,Bronc --evidence Smoker=1 --method bp

# Approximate inference
$ pgmgo query asia.bif --variables Dyspnea --evidence Smoker=1 --method approx`}</code></pre>

        <h3>map</h3>
        <p>Find the MAP (Maximum A Posteriori) assignment -- the most likely values for query variables given evidence.</p>
        <pre><code>{`pgmgo map <file> --variables V1,V2 [--evidence E1=v1,E2=v2]`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--variables</code></td><td>(required)</td><td>Comma-separated query variables</td></tr>
            <tr><td><code>--evidence</code></td><td>(none)</td><td>Comma-separated evidence as KEY=VALUE pairs</td></tr>
          </tbody>
        </table>
        <pre><code>{`$ pgmgo map asia.bif --variables Lung,Bronc --evidence Smoker=1
MAP assignment:
  Bronc = 1
  Lung = 0`}</code></pre>

        <h3>learn</h3>
        <p>Learn a Bayesian network structure from CSV data. Automatically fits MLE parameters after learning.</p>
        <pre><code>{`pgmgo learn --data <csv> --method <method> --output <bif> [--score bic|bdeu|k2] [--significance 0.05]`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--data</code></td><td>(required)</td><td>Path to CSV data file</td></tr>
            <tr><td><code>--method</code></td><td><code>hillclimb</code></td><td>Learning method: <code>hillclimb</code>, <code>pc</code>, <code>ges</code>, <code>exhaustive</code>, <code>tree</code></td></tr>
            <tr><td><code>--score</code></td><td><code>bic</code></td><td>Scoring function: <code>bic</code>, <code>bdeu</code>, <code>k2</code></td></tr>
            <tr><td><code>--significance</code></td><td><code>0.05</code></td><td>Significance level for constraint-based methods (PC)</td></tr>
            <tr><td><code>--output</code></td><td>(required)</td><td>Output BIF file path</td></tr>
          </tbody>
        </table>
        <pre><code>{`# Hill-climb with BIC scoring (default)
$ pgmgo learn --data observations.csv --output learned.bif

# PC algorithm with chi-square test
$ pgmgo learn --data observations.csv --method pc --significance 0.01 --output learned_pc.bif

# GES with BDeu scoring
$ pgmgo learn --data observations.csv --method ges --score bdeu --output learned_ges.bif

# Exhaustive search (small networks only)
$ pgmgo learn --data small_data.csv --method exhaustive --output optimal.bif

# Tree search (Chow-Liu)
$ pgmgo learn --data observations.csv --method tree --output tree_model.bif`}</code></pre>

        <h3>fit</h3>
        <p>Fit parameters (CPDs) to an existing network structure using observed data.</p>
        <pre><code>{`pgmgo fit --model <bif> --data <csv> --method <method> --output <bif>`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--model</code></td><td>(required)</td><td>Input BIF model file (provides structure)</td></tr>
            <tr><td><code>--data</code></td><td>(required)</td><td>Path to CSV data file</td></tr>
            <tr><td><code>--method</code></td><td><code>mle</code></td><td>Parameter learning method: <code>mle</code>, <code>bayesian</code>, <code>em</code></td></tr>
            <tr><td><code>--output</code></td><td>(required)</td><td>Output BIF file path</td></tr>
          </tbody>
        </table>
        <pre><code>{`# Maximum Likelihood Estimation
$ pgmgo fit --model structure.bif --data train.csv --output fitted_mle.bif

# Bayesian estimation with BDeu prior
$ pgmgo fit --model structure.bif --data train.csv --method bayesian --output fitted_bayes.bif

# EM for incomplete data
$ pgmgo fit --model structure.bif --data incomplete.csv --method em --output fitted_em.bif`}</code></pre>

        <h3>sample</h3>
        <p>Generate samples from a Bayesian network model.</p>
        <pre><code>{`pgmgo sample --model <bif> --output <csv> [--n 100] [--method forward|rejection|gibbs] [--evidence E1=v1] [--seed 42]`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--model</code></td><td>(required)</td><td>Input BIF model file</td></tr>
            <tr><td><code>--n</code></td><td><code>100</code></td><td>Number of samples to generate</td></tr>
            <tr><td><code>--output</code></td><td>(required)</td><td>Output CSV file path</td></tr>
            <tr><td><code>--method</code></td><td><code>forward</code></td><td>Sampling method: <code>forward</code>, <code>rejection</code>, <code>gibbs</code></td></tr>
            <tr><td><code>--evidence</code></td><td>(none)</td><td>Evidence for rejection/gibbs sampling</td></tr>
            <tr><td><code>--seed</code></td><td><code>0</code></td><td>Random seed (0 = non-deterministic)</td></tr>
          </tbody>
        </table>
        <pre><code>{`# Forward sampling (100 samples)
$ pgmgo sample --model asia.bif --output samples.csv

# 5000 samples with fixed seed
$ pgmgo sample --model asia.bif --n 5000 --seed 42 --output samples.csv

# Rejection sampling with evidence
$ pgmgo sample --model asia.bif --n 500 --method rejection --evidence Smoker=1 --output smoker_samples.csv

# Gibbs sampling
$ pgmgo sample --model asia.bif --n 1000 --method gibbs --evidence Smoker=1 --output gibbs_samples.csv`}</code></pre>

        <h3>info</h3>
        <p>Print a summary of a BIF model.</p>
        <pre><code>pgmgo info &lt;file&gt;</code></pre>
        <pre><code>{`$ pgmgo info asia.bif
Model: asia.bif
Nodes: 8
Edges: 8

Node list:
  VisitAsia (states: no, yes)
  Tuberculosis (states: no, yes)
  Smoker (states: no, yes)
  ...

Edge list:
  VisitAsia -> Tuberculosis
  Smoker -> Lung
  ...

CPD summary:
  VisitAsia: 2 states, no parents
  Tuberculosis: 2 states, parents: VisitAsia
  ...`}</code></pre>

        <h3>convert</h3>
        <p>Convert a model between supported file formats.</p>
        <pre><code>{`pgmgo convert --input <file> --from <format> --to <format> --output <file>`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--input</code></td><td>Input file path</td></tr>
            <tr><td><code>--from</code></td><td>Input format: <code>bif</code>, <code>xmlbif</code>, <code>net</code>, <code>uai</code>, <code>xdsl</code></td></tr>
            <tr><td><code>--to</code></td><td>Output format: <code>bif</code>, <code>xmlbif</code>, <code>net</code>, <code>uai</code>, <code>xdsl</code></td></tr>
            <tr><td><code>--output</code></td><td>Output file path</td></tr>
          </tbody>
        </table>
        <pre><code>{`# BIF to XMLBIF
$ pgmgo convert --input asia.bif --from bif --to xmlbif --output asia.xmlbif

# NET to UAI
$ pgmgo convert --input model.net --from net --to uai --output model.uai

# XDSL to BIF
$ pgmgo convert --input model.xdsl --from xdsl --to bif --output model.bif`}</code></pre>

        <h3>compare</h3>
        <p>Compare two network structures using standard metrics.</p>
        <pre><code>{`pgmgo compare --true <bif> --estimated <bif>`}</code></pre>
        <pre><code>{`$ pgmgo compare --true ground_truth.bif --estimated learned.bif
Structural Hamming Distance (SHD): 3

Adjacency confusion matrix:
  TP=6  FP=1  TN=15  FN=2

Orientation confusion matrix:
  TP=5  FP=2  TN=14  FN=3`}</code></pre>

        <h3>do</h3>
        <p>Perform a causal do-calculus query. Computes P(Y | do(X=x)).</p>
        <pre><code>{`pgmgo do <file> --intervention X=v --query Y [--evidence E1=v1]`}</code></pre>
        <table>
          <thead>
            <tr><th>Flag</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--intervention</code></td><td>Intervention variable(s) as KEY=VALUE (comma-separated)</td></tr>
            <tr><td><code>--query</code></td><td>Query variable</td></tr>
            <tr><td><code>--evidence</code></td><td>Optional observational evidence</td></tr>
          </tbody>
        </table>
        <pre><code>{`# P(Dyspnea | do(Smoker=1))
$ pgmgo do asia.bif --intervention Smoker=1 --query Dyspnea
Dyspnea	P
0	0.304000
1	0.696000

# With additional evidence
$ pgmgo do asia.bif --intervention Smoker=1 --query Lung --evidence VisitAsia=0`}</code></pre>

        <h3>Exit Codes</h3>
        <table>
          <thead>
            <tr><th>Code</th><th>Meaning</th></tr>
          </thead>
          <tbody>
            <tr><td><code>0</code></td><td>Success</td></tr>
            <tr><td><code>1</code></td><td>Runtime error (inference failure, invalid model, etc.)</td></tr>
            <tr><td><code>2</code></td><td>Invalid input (missing arguments, file not found, bad flags)</td></tr>
          </tbody>
        </table>
      </section>

      {/* ============================================================ */}
      {/* FILE FORMATS */}
      {/* ============================================================ */}
      <section class="section" id="file-formats">
        <h2>File Formats</h2>
        <p>
          pgmgo supports 10 file formats for reading and writing probabilistic graphical models.
          All formats are accessed through the <code>readwrite</code> package.
        </p>
        <table>
          <thead>
            <tr><th>Format</th><th>Extension</th><th>Read</th><th>Write</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>BIF</strong></td><td>.bif</td><td>Yes</td><td>Yes</td><td>Bayesian Interchange Format. The standard PGM format. Stores structure, states, and CPD tables in a human-readable text format.</td></tr>
            <tr><td><strong>XMLBIF</strong></td><td>.xmlbif</td><td>Yes</td><td>Yes</td><td>XML-based BIF. Same information as BIF but in XML for easier parsing by other tools.</td></tr>
            <tr><td><strong>NET</strong></td><td>.net</td><td>Yes</td><td>Yes</td><td>Hugin NET format. Used by the Hugin BN software.</td></tr>
            <tr><td><strong>UAI</strong></td><td>.uai</td><td>Yes</td><td>Yes</td><td>UAI format. Used by the UAI inference competition. Compact numeric format.</td></tr>
            <tr><td><strong>XDSL</strong></td><td>.xdsl</td><td>Yes</td><td>Yes</td><td>GeNIe XDSL format. Used by the GeNIe/SMILE BN software.</td></tr>
            <tr><td><strong>PomdpX</strong></td><td>.pomdpx</td><td>Yes</td><td>--</td><td>POMDP XML format. Read-only. Used in POMDP planning literature.</td></tr>
            <tr><td><strong>XBN</strong></td><td>.xbn</td><td>Yes</td><td>--</td><td>Microsoft XBN format. Read-only. Legacy Microsoft Research format.</td></tr>
            <tr><td><strong>CSV</strong></td><td>.csv</td><td>Yes</td><td>Yes</td><td>CSV model serialization. Stores structure and parameters in CSV tables.</td></tr>
            <tr><td><strong>JSON</strong></td><td>.json</td><td>Yes</td><td>Yes</td><td>JSON model serialization. Ideal for web applications and REST APIs.</td></tr>
            <tr><td><strong>XML</strong></td><td>.xml</td><td>Yes</td><td>Yes</td><td>XML model serialization. General-purpose XML format.</td></tr>
          </tbody>
        </table>

        <h3>BIF Format Example</h3>
        <pre><code>{`network asia {
}

variable VisitAsia {
  type discrete [ 2 ] { no, yes };
}

variable Tuberculosis {
  type discrete [ 2 ] { no, yes };
}

probability ( VisitAsia ) {
  table 0.99, 0.01;
}

probability ( Tuberculosis | VisitAsia ) {
  (no) 0.99, 0.01;
  (yes) 0.95, 0.05;
}`}</code></pre>

        <h3>JSON Format Example</h3>
        <pre><code>{`{
  "nodes": ["A", "B", "C"],
  "edges": [["A", "B"], ["B", "C"]],
  "states": {
    "A": ["a0", "a1"],
    "B": ["b0", "b1"],
    "C": ["c0", "c1"]
  },
  "cpds": {
    "A": {
      "variable": "A",
      "cardinality": 2,
      "values": [0.4, 0.6],
      "parents": [],
      "parent_cardinalities": []
    }
  }
}`}</code></pre>

        <h3>Reading and Writing</h3>
        <pre><code>{`import (
    "os"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
)

// Read BIF
f, _ := os.Open("model.bif")
bn, _ := readwrite.ReadBIF(f)
f.Close()

// Write as JSON (for a web API)
out, _ := os.Create("model.json")
readwrite.WriteJSONModel(out, bn)
out.Close()

// Convert: read NET, write XMLBIF
netFile, _ := os.Open("model.net")
bn2, _ := readwrite.ReadNET(netFile)
netFile.Close()

xmlbifFile, _ := os.Create("model.xmlbif")
readwrite.WriteXMLBIF(xmlbifFile, bn2)
xmlbifFile.Close()`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* EXAMPLE MODELS */}
      {/* ============================================================ */}
      <section class="section" id="example-models">
        <h2>Example Models</h2>
        <p>
          pgmgo ships with 25 built-in Bayesian networks accessible via the <code>example_models</code> package.
          13 models include full CPDs (conditional probability distributions); 12 are structure-only for use in
          learning and benchmarking.
        </p>

        <h3>Models with Full CPDs</h3>
        <table>
          <thead>
            <tr><th>Name</th><th>Description</th><th>Nodes</th></tr>
          </thead>
          <tbody>
            <tr><td><code>student</code></td><td>Classic Student network (difficulty, intelligence, grade, SAT, letter)</td><td>5</td></tr>
            <tr><td><code>asia</code></td><td>Lung disease diagnosis</td><td>8</td></tr>
            <tr><td><code>alarm</code></td><td>Monitoring system with alarm, burglary, earthquake</td><td>5</td></tr>
            <tr><td><code>cancer</code></td><td>Cancer diagnosis network</td><td>5</td></tr>
            <tr><td><code>watersprinkler</code></td><td>Classic sprinkler/rain/wet grass example</td><td>4</td></tr>
            <tr><td><code>survey</code></td><td>Survey response model</td><td>6</td></tr>
            <tr><td><code>montyhall</code></td><td>Monty Hall problem as a Bayesian network</td><td>3</td></tr>
            <tr><td><code>dogproblem</code></td><td>Dog behavior inference</td><td>5</td></tr>
            <tr><td><code>frauddetection</code></td><td>Financial fraud detection model</td><td>5</td></tr>
            <tr><td><code>medicaldiagnosis</code></td><td>Medical symptom/disease model</td><td>8</td></tr>
            <tr><td><code>earthquake</code></td><td>Earthquake alert network</td><td>5</td></tr>
            <tr><td><code>visitasia</code></td><td>Visit to Asia variant</td><td>8</td></tr>
            <tr><td><code>cointoss</code></td><td>Simple coin toss model</td><td>2</td></tr>
          </tbody>
        </table>

        <h3>Structure-Only Models (Large Networks)</h3>
        <table>
          <thead>
            <tr><th>Name</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>sachs</code></td><td>Protein signaling network (11 nodes)</td></tr>
            <tr><td><code>child</code></td><td>Child health assessment network</td></tr>
            <tr><td><code>insurance</code></td><td>Insurance risk assessment</td></tr>
            <tr><td><code>alarmfull</code></td><td>Full ALARM monitoring network (37 nodes)</td></tr>
            <tr><td><code>water</code></td><td>Water treatment network</td></tr>
            <tr><td><code>mildew</code></td><td>Crop disease model</td></tr>
            <tr><td><code>barley</code></td><td>Barley crop yield model</td></tr>
            <tr><td><code>hailfinder</code></td><td>Severe weather prediction</td></tr>
            <tr><td><code>hepar2</code></td><td>Liver disorder diagnosis</td></tr>
            <tr><td><code>win95pts</code></td><td>Windows 95 printer troubleshooting</td></tr>
            <tr><td><code>pathfinder</code></td><td>Pathology diagnosis</td></tr>
            <tr><td><code>pigs</code></td><td>Pig breeding network</td></tr>
          </tbody>
        </table>

        <h3>Usage</h3>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/example_models"

// List all available models
names := example_models.List()
for _, name := range names {
    fmt.Println(name)
}

// Load a specific model
bn, err := example_models.Get("asia")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Nodes: %d, Edges: %d\\n", len(bn.Nodes()), len(bn.Edges()))

// Use the model for inference
facs, _ := bn.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)
result, _ := ve.Query([]string{"Dyspnea"}, map[string]int{"Smoker": 1})
fmt.Println(result.Values().Data())`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* DATASETS */}
      {/* ============================================================ */}
      <section class="section" id="datasets">
        <h2>Datasets</h2>
        <p>
          pgmgo includes 40 built-in datasets accessible via the <code>examples/datasets</code> package.
          These datasets are embedded in the binary using Go's <code>embed</code> package, so they are
          always available without external file dependencies.
        </p>

        <h3>BN-Specific Datasets</h3>
        <table>
          <thead>
            <tr><th>Name</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>asia</code></td><td>Sampled data from the Asia (lung disease) network</td></tr>
            <tr><td><code>alarm</code></td><td>Sampled data from the ALARM monitoring network</td></tr>
            <tr><td><code>sachs</code></td><td>Protein signaling data (Sachs et al.)</td></tr>
            <tr><td><code>cancer</code></td><td>Cancer diagnosis observations</td></tr>
            <tr><td><code>student</code></td><td>Student performance data</td></tr>
            <tr><td><code>sprinkler</code></td><td>Sprinkler/rain/wet grass observations</td></tr>
            <tr><td><code>survey</code></td><td>Survey response data</td></tr>
            <tr><td><code>earthquake</code></td><td>Earthquake alert observations</td></tr>
            <tr><td><code>child</code></td><td>Child health assessment data</td></tr>
            <tr><td><code>insurance</code></td><td>Insurance risk data</td></tr>
            <tr><td><code>water</code></td><td>Water treatment data</td></tr>
            <tr><td><code>mildew</code></td><td>Crop disease data</td></tr>
            <tr><td><code>hailfinder</code></td><td>Severe weather observations</td></tr>
            <tr><td><code>hepar2</code></td><td>Liver disorder data</td></tr>
            <tr><td><code>barley</code></td><td>Barley crop data</td></tr>
            <tr><td><code>win95pts</code></td><td>Windows 95 troubleshooting data</td></tr>
            <tr><td><code>andes</code></td><td>ANDES intelligent tutoring system data</td></tr>
            <tr><td><code>munin</code></td><td>MUNIN neural network data</td></tr>
            <tr><td><code>lucas</code></td><td>LUCAS causal discovery benchmark</td></tr>
          </tbody>
        </table>

        <h3>Classic ML Datasets</h3>
        <table>
          <thead>
            <tr><th>Name</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>titanic</code></td><td>Titanic survival data</td></tr>
            <tr><td><code>iris</code></td><td>Fisher's Iris flower dataset</td></tr>
            <tr><td><code>heart</code></td><td>Heart disease prediction</td></tr>
            <tr><td><code>wine</code></td><td>Wine quality classification</td></tr>
            <tr><td><code>boston</code></td><td>Boston housing prices</td></tr>
            <tr><td><code>pima_diabetes</code></td><td>Pima Indians diabetes</td></tr>
            <tr><td><code>adult</code></td><td>Adult income prediction (Census)</td></tr>
            <tr><td><code>breast_cancer</code></td><td>Wisconsin breast cancer diagnosis</td></tr>
          </tbody>
        </table>

        <h3>UCI Repository Datasets</h3>
        <table>
          <thead>
            <tr><th>Name</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>zoo</code></td><td>Zoo animal classification</td></tr>
            <tr><td><code>glass</code></td><td>Glass identification</td></tr>
            <tr><td><code>ecoli</code></td><td>E. coli protein localization</td></tr>
            <tr><td><code>monks</code></td><td>MONKS problem</td></tr>
            <tr><td><code>nursery</code></td><td>Nursery school evaluation</td></tr>
            <tr><td><code>credit_approval</code></td><td>Credit card approval</td></tr>
            <tr><td><code>balance_scale</code></td><td>Balance scale weight/distance</td></tr>
            <tr><td><code>automobile</code></td><td>Automobile price prediction</td></tr>
            <tr><td><code>mushroom</code></td><td>Mushroom edibility classification</td></tr>
            <tr><td><code>car_evaluation</code></td><td>Car evaluation</td></tr>
            <tr><td><code>hepatitis</code></td><td>Hepatitis prognosis</td></tr>
            <tr><td><code>vote</code></td><td>Congressional voting records</td></tr>
            <tr><td><code>tic_tac_toe</code></td><td>Tic-tac-toe endgame</td></tr>
          </tbody>
        </table>

        <h3>Usage</h3>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/examples/datasets"

// List all available datasets
names := datasets.List()

// Load a dataset as a DataFrame
df, err := datasets.Load("asia")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Rows: %d, Columns: %d\\n", df.NRows(), len(df.Columns()))

// Use the dataset for structure learning
scorer := structure_score.NewBIC()
hc := learning.NewHillClimbSearch(df, scorer.LocalScore)
bn, _ := hc.Estimate()`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* TESTING */}
      {/* ============================================================ */}
      <section class="section" id="testing">
        <h2>Testing</h2>
        <p>
          pgmgo has approximately 5,000 tests and 392 cross-validation fixtures spanning unit tests,
          integration tests, and cross-validation tests across 24 packages.
        </p>

        <h3>Running Tests</h3>
        <pre><code>{`# Run all tests
go test ./...

# Run tests for a specific package
go test ./src/inference/...

# Run with verbose output
go test -v ./src/models/...

# Run with race detector
go test -race ./...

# Run a specific test function
go test -run TestVariableElimination ./src/inference/...

# Run cross-validation tests only
go test -run CrossVal ./...

# Run benchmarks
go test -bench=. ./src/inference/...`}</code></pre>

        <h3>Cross-Validation System</h3>
        <p>
          Many packages include <code>crossval_*_test.go</code> files that validate algorithms against
          known-correct results. These tests load built-in example models, run computations, and compare
          outputs against pre-computed reference values. Examples:
        </p>
        <ul>
          <li><code>src/models/crossval_dsep_test.go</code> -- validates d-separation queries</li>
          <li><code>src/inference/crossval_causal_test.go</code> -- validates causal inference results</li>
          <li><code>src/inference/crossval_ve_test.go</code> -- validates Variable Elimination posteriors</li>
          <li><code>src/learning/crossval_hillclimb_test.go</code> -- validates structure learning output</li>
          <li><code>src/sampling/crossval_forward_test.go</code> -- validates sampling distributions</li>
        </ul>
        <p>
          Cross-validation fixtures are generated by running the equivalent pgmpy code in Python and
          storing the results. This ensures pgmgo produces the same numerical outputs as the reference
          implementation.
        </p>

        <h3>Test Fixture Generation</h3>
        <p>
          Test fixtures use the built-in example models from the <code>example_models</code> package.
          This ensures reproducible test data without external file dependencies. When adding new tests,
          use existing models or create new ones in the <code>example_models</code> package.
        </p>

        <h3>Writing Tests</h3>
        <pre><code>{`func TestMyFeature(t *testing.T) {
    // Load a known model
    bn, err := example_models.Get("asia")
    if err != nil {
        t.Fatal(err)
    }

    // Perform computation
    facs, _ := bn.ToMarkovFactors()
    ve := inference.NewVariableElimination(facs)
    result, err := ve.Query([]string{"Dyspnea"}, map[string]int{"Smoker": 1})
    if err != nil {
        t.Fatal(err)
    }

    // Compare against expected values
    values := result.Values().Data()
    if math.Abs(values[0] - 0.304) > 0.01 {
        t.Errorf("expected P(Dyspnea=0|Smoker=1) ~ 0.304, got %f", values[0])
    }
}`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* CONFIGURATION */}
      {/* ============================================================ */}
      <section class="section" id="configuration">
        <h2>Configuration</h2>
        <p>
          The <code>config</code> package provides global configuration options that control
          default behavior across pgmgo.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/config"

// Get the global config
cfg := config.Global()

// Configuration is used internally by various packages
// to control default inference methods, scoring functions,
// numerical tolerances, and other global settings.`}</code></pre>
        <p>
          Configuration options include numerical tolerances for probability comparisons,
          default inference methods, default scoring functions for structure learning,
          convergence thresholds for iterative algorithms (EM, Belief Propagation),
          and logging verbosity.
        </p>
      </section>

      {/* ============================================================ */}
      {/* CONTRIBUTING */}
      {/* ============================================================ */}
      <section class="section" id="contributing">
        <h2>Contributing</h2>
        <p>
          Contributions are welcome. See the full{" "}
          <a href="https://github.com/asymmetric-effort/pgmgo/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener noreferrer">
            CONTRIBUTING.md
          </a>{" "}
          for details.
        </p>

        <h3>Development Workflow</h3>
        <ol>
          <li>Fork the repository and clone your fork</li>
          <li>Create a feature branch: <code>git checkout -b feature/my-feature</code></li>
          <li>Make changes and add tests</li>
          <li>Run <code>go test ./...</code> to verify all tests pass</li>
          <li>Run <code>go vet ./...</code> for static analysis</li>
          <li>Commit with a clear message and submit a pull request</li>
        </ol>

        <h3>Guidelines</h3>
        <ul>
          <li><strong>Zero dependencies:</strong> Do not add third-party modules. All functionality must be implemented in pure Go using only the standard library.</li>
          <li><strong>Tests required:</strong> All new functionality must include unit tests. Cross-validation tests against pgmpy are strongly encouraged.</li>
          <li><strong>Backward compatibility:</strong> Public API changes require discussion in an issue before implementation.</li>
          <li><strong>Documentation:</strong> Exported types and functions must have godoc comments.</li>
        </ul>
      </section>
    </div>
  );
}
