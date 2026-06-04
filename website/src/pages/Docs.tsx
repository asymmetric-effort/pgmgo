import { createElement, useHead } from "@asymmetric-effort/specifyjs";

export function Docs() {
  useHead({
    title: "Documentation — pgmgo",
    description: "Comprehensive documentation for pgmgo, a Go library for probabilistic graphical models.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/docs",
  });

  return (
    <div class="page">
      <h1>Documentation</h1>

      <section class="section">
        <h2>Overview</h2>
        <p>
          pgmgo is a zero-dependency Go library for probabilistic graphical models (PGMs).
          It aims for feature parity with <a href="https://pgmpy.org" target="_blank" rel="noopener noreferrer">pgmpy</a>,
          providing tools for creating, parameterizing, learning, and performing inference on
          Bayesian networks, Markov networks, and related structures. All numerical primitives
          (linear algebra, statistics, graph algorithms, tabular data) are implemented from scratch in Go.
        </p>
        <p>
          The current release is <strong>v0.0.35</strong> with approximately 5,000 tests and 392 cross-validation
          fixtures covering inference, learning, sampling, serialization, and cross-validation across 24 packages.
        </p>
      </section>

      <section class="section">
        <h2>Getting Started</h2>

        <h3>Installation</h3>
        <pre><code>go get github.com/asymmetric-effort/pgmgo</code></pre>

        <h3>Your First Bayesian Network</h3>
        <pre><code>{`package main

import (
    "fmt"
    "github.com/asymmetric-effort/pgmgo/src/models"
    "github.com/asymmetric-effort/pgmgo/src/factors"
)

func main() {
    bn := models.NewBayesianNetwork()

    // Add nodes
    bn.AddNode("Cloudy")
    bn.AddNode("Rain")
    bn.AddNode("Sprinkler")
    bn.AddNode("WetGrass")

    // Add directed edges
    bn.AddEdge("Cloudy", "Rain")
    bn.AddEdge("Cloudy", "Sprinkler")
    bn.AddEdge("Rain", "WetGrass")
    bn.AddEdge("Sprinkler", "WetGrass")

    // Define states
    bn.SetStates("Cloudy", []string{"clear", "cloudy"})
    bn.SetStates("Rain", []string{"no", "yes"})
    bn.SetStates("Sprinkler", []string{"off", "on"})
    bn.SetStates("WetGrass", []string{"dry", "wet"})

    // Add CPDs
    bn.SetCPD("Cloudy", factors.NewTabularCPD(
        "Cloudy", 2, []float64{0.5, 0.5}, nil, nil,
    ))
    bn.SetCPD("Rain", factors.NewTabularCPD(
        "Rain", 2,
        []float64{0.8, 0.2, 0.2, 0.8},
        []string{"Cloudy"}, []int{2},
    ))

    // Validate the model
    if err := bn.CheckModel(); err != nil {
        fmt.Println("Model error:", err)
    } else {
        fmt.Println("Model is valid!")
    }
}`}</code></pre>

        <h3>Your First Query</h3>
        <pre><code>{`import (
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/example_models"
)

// Load a built-in model
bn, _ := example_models.Get("asia")

// Convert to factors for Variable Elimination
facs, _ := bn.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)

// P(Dyspnea | Smoker=yes)
result, _ := ve.Query(
    []string{"Dyspnea"},
    map[string]int{"Smoker": 1},
)
fmt.Println(result)`}</code></pre>
      </section>

      <section class="section">
        <h2>Library Packages (lib/)</h2>
        <p>
          These packages replace common Python scientific computing libraries with pure Go implementations.
          They have no third-party dependencies.
        </p>
        <table>
          <thead>
            <tr><th>Package</th><th>Replaces</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>numgo</code></td><td>numpy</td><td>N-dimensional arrays, linear algebra, matrix operations, broadcasting, element-wise arithmetic</td></tr>
            <tr><td><code>scigo</code></td><td>scipy</td><td>Statistical distributions (normal, chi-squared, beta, gamma), optimization, special functions, hypothesis tests</td></tr>
            <tr><td><code>graphgo</code></td><td>networkx</td><td>Directed and undirected graphs, DiGraph, PDAG, d-separation, topological sort, clique finding, moral graph</td></tr>
            <tr><td><code>tabgo</code></td><td>pandas</td><td>DataFrame and Series, CSV/Parquet I/O, groupby, filtering, column operations, value counts</td></tr>
            <tr><td><code>gpu</code></td><td>--</td><td>GPU compute backend for accelerating matrix operations and inference on large networks</td></tr>
          </tbody>
        </table>
      </section>

      <section class="section">
        <h2>Core Packages (src/)</h2>
        <table>
          <thead>
            <tr><th>Package</th><th>Description</th><th>Key Types</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><code>base</code></td>
              <td>Foundational graph types for all model classes</td>
              <td>DAG, PDAG, UndirectedGraph, ADMG, MAG, SimpleCausalModel</td>
            </tr>
            <tr>
              <td><code>models</code></td>
              <td>13 probabilistic model implementations</td>
              <td>BayesianNetwork, MarkovNetwork, DynamicBN, NaiveBayes, SEM, FactorGraph, JunctionTree, ClusterGraph, LinearGaussianBN, FunctionalBN, MarkovChain, DiscreteBN, DiscreteMarkovNetwork</td>
            </tr>
            <tr>
              <td><code>factors</code></td>
              <td>Factor representations and operations (product, marginalization, reduction)</td>
              <td>DiscreteFactor, TabularCPD, JointProbabilityDistribution, LinearGaussianCPD, NoisyOR</td>
            </tr>
            <tr>
              <td><code>inference</code></td>
              <td>Exact and approximate inference algorithms</td>
              <td>VariableElimination, BeliefPropagation, MPLP, ApproxInference, CausalInference, DBNInference</td>
            </tr>
            <tr>
              <td><code>sampling</code></td>
              <td>Forward, rejection, likelihood-weighted, and Gibbs sampling</td>
              <td>BayesianModelSampling, GibbsSampling</td>
            </tr>
            <tr>
              <td><code>learning</code></td>
              <td>Parameter estimation and structure learning</td>
              <td>MLE, BayesianEstimator, EM, HillClimbSearch, PC, GES, ExhaustiveSearch, TreeSearch, MMHC, ExpertInLoop, IVEstimator, SEMEstimator, MirrorDescent, LLMClient</td>
            </tr>
            <tr>
              <td><code>ci_tests</code></td>
              <td>Conditional independence tests for constraint-based learning</td>
              <td>ChiSquare, GSq, FisherZ, Pearsonr, GCM, HotellingLawley (discrete, continuous, multivariate, tree-based)</td>
            </tr>
            <tr>
              <td><code>structure_score</code></td>
              <td>Scoring functions for score-based structure learning</td>
              <td>BIC, AIC, BDeu, BDs, K2, LogLikelihood, Gaussian, ConditionalGaussian</td>
            </tr>
            <tr>
              <td><code>identification</code></td>
              <td>Causal effect identification algorithms</td>
              <td>Adjustment (back-door), Frontdoor</td>
            </tr>
            <tr>
              <td><code>prediction</code></td>
              <td>Causal prediction and treatment effect estimation</td>
              <td>DoubleMLRegressor, NaiveAdjustmentRegressor, NaiveIVRegressor</td>
            </tr>
            <tr>
              <td><code>metrics</code></td>
              <td>Model evaluation and comparison metrics</td>
              <td>SHD, AdjacencyConfusionMatrix, OrientationConfusionMatrix, CorrelationScore, FisherC</td>
            </tr>
            <tr>
              <td><code>independencies</code></td>
              <td>Independence assertion representations</td>
              <td>IndependenceAssertion, Independencies</td>
            </tr>
            <tr>
              <td><code>readwrite</code></td>
              <td>10 file format readers and writers</td>
              <td>BIF, XMLBIF, NET, UAI, XDSL, PomdpX, XBN, CSV, JSON, XML</td>
            </tr>
            <tr>
              <td><code>config</code></td>
              <td>Configuration and global settings</td>
              <td>Config</td>
            </tr>
            <tr>
              <td><code>utils</code></td>
              <td>Shared parsing, optimization, and compatibility utilities</td>
              <td>Utility functions</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section class="section">
        <h2>Project Structure</h2>
        <pre><code>{`pgmgo/
  cmd/
    pgmgo/             # CLI entry point (10 commands)
  lib/                 # Internal primitive modules
    numgo/             # numpy equivalent (NDArray, Matrix, Vector)
    scigo/             # scipy equivalent (distributions, optimization)
    graphgo/           # networkx equivalent (DiGraph, PDAG, algorithms)
    tabgo/             # pandas equivalent (DataFrame, Series, CSV)
    gpu/               # GPU compute backend
  src/                 # Core pgmgo library
    base/              # DAG, PDAG, UndirectedGraph, ADMG, MAG
    models/            # 13 probabilistic model types
    factors/           # Factor representations and operations
    inference/         # 7 inference algorithms
    sampling/          # Forward, rejection, Gibbs sampling
    learning/          # 11+ learning algorithms + LLM integration
    ci_tests/          # Conditional independence tests
    structure_score/   # 13 scoring functions
    identification/    # Causal effect identification
    prediction/        # DoubleML, naive adjustment, IV regression
    metrics/           # SHD, confusion matrices, correlation
    independencies/    # Independence assertion representations
    readwrite/         # 10 I/O formats
    config/            # Configuration
    utils/             # Shared utilities
  example_models/      # 25 built-in Bayesian networks
  website/             # Project website (this site)
  docs/                # Additional documentation`}</code></pre>
      </section>

      <section class="section">
        <h2>Example Models</h2>
        <p>
          pgmgo ships with 25 built-in Bayesian networks accessible via the <code>example_models</code> package.
          13 models include full CPDs (conditional probability distributions); 12 are structure-only for use in
          learning and benchmarking.
        </p>

        <h3>Models with Full CPDs</h3>
        <table>
          <thead>
            <tr><th>Model</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>student</code></td><td>Classic Student network (difficulty, intelligence, grade, SAT, letter)</td></tr>
            <tr><td><code>asia</code></td><td>Lung disease diagnosis (8 nodes)</td></tr>
            <tr><td><code>alarm</code></td><td>Monitoring system with alarm, burglary, earthquake</td></tr>
            <tr><td><code>cancer</code></td><td>Cancer diagnosis network</td></tr>
            <tr><td><code>watersprinkler</code></td><td>Classic sprinkler/rain/wet grass example</td></tr>
            <tr><td><code>survey</code></td><td>Survey response model</td></tr>
            <tr><td><code>montyhall</code></td><td>Monty Hall problem as a Bayesian network</td></tr>
            <tr><td><code>dogproblem</code></td><td>Dog behavior inference</td></tr>
            <tr><td><code>frauddetection</code></td><td>Financial fraud detection model</td></tr>
            <tr><td><code>medicaldiagnosis</code></td><td>Medical symptom/disease model</td></tr>
            <tr><td><code>earthquake</code></td><td>Earthquake alert network</td></tr>
            <tr><td><code>visitasia</code></td><td>Visit to Asia variant</td></tr>
            <tr><td><code>cointoss</code></td><td>Simple coin toss model</td></tr>
          </tbody>
        </table>

        <h3>Structure-Only Models (Large Networks)</h3>
        <table>
          <thead>
            <tr><th>Model</th><th>Description</th></tr>
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

// Load a specific model
bn, err := example_models.Get("asia")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Nodes: %d, Edges: %d\\n", len(bn.Nodes()), len(bn.Edges()))`}</code></pre>
      </section>

      <section class="section">
        <h2>Testing</h2>
        <p>
          pgmgo has approximately 5,000 tests and 392 cross-validation fixtures spanning unit tests, integration tests, and cross-validation tests across 24 packages.
        </p>
        <h3>Running Tests</h3>
        <pre><code>{`# Run all tests
go test ./...

# Run tests for a specific package
go test ./src/inference/...

# Run with verbose output
go test -v ./src/models/...

# Run cross-validation tests
go test -run CrossVal ./src/models/...`}</code></pre>

        <h3>Cross-Validation Tests</h3>
        <p>
          Many packages include <code>crossval_*_test.go</code> files that verify algorithms
          against known results. For example, <code>src/models/crossval_dsep_test.go</code> validates
          d-separation queries, and <code>src/inference/crossval_causal_test.go</code> validates
          causal inference results.
        </p>

        <h3>Test Fixtures</h3>
        <p>
          Tests use the built-in example models from the <code>example_models</code> package as fixtures.
          This ensures reproducible test data without external file dependencies.
        </p>
      </section>

      <section class="section">
        <h2>I/O Formats</h2>
        <table>
          <thead>
            <tr><th>Format</th><th>Read</th><th>Write</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BIF</code></td><td>Yes</td><td>Yes</td><td>Bayesian Interchange Format (standard PGM format)</td></tr>
            <tr><td><code>XMLBIF</code></td><td>Yes</td><td>Yes</td><td>XML-based Bayesian Interchange Format</td></tr>
            <tr><td><code>NET</code></td><td>Yes</td><td>Yes</td><td>Hugin NET format</td></tr>
            <tr><td><code>UAI</code></td><td>Yes</td><td>Yes</td><td>UAI inference competition format</td></tr>
            <tr><td><code>XDSL</code></td><td>Yes</td><td>Yes</td><td>GeNIe XDSL format</td></tr>
            <tr><td><code>PomdpX</code></td><td>Yes</td><td>--</td><td>POMDP XML format</td></tr>
            <tr><td><code>XBN</code></td><td>Yes</td><td>--</td><td>Microsoft XBN format</td></tr>
            <tr><td><code>CSV</code></td><td>Yes</td><td>Yes</td><td>CSV model serialization</td></tr>
            <tr><td><code>JSON</code></td><td>Yes</td><td>Yes</td><td>JSON model serialization</td></tr>
            <tr><td><code>XML</code></td><td>Yes</td><td>Yes</td><td>XML model serialization</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
