import { createElement, useHead, Link } from "@asymmetric-effort/specifyjs";
import { ScrollLink } from "../components/ScrollLink";

export function Tutorials() {
  useHead({
    title: "Tutorials — pgmgo",
    description: "Step-by-step tutorials for building Bayesian networks, inference, structure learning, causal inference, file formats, sampling, advanced models, and internal libraries with pgmgo.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/tutorials",
  });

  return (
    <div class="page">
      <h1>Tutorials</h1>

      <nav class="page-toc">
        <strong>Tutorials:</strong>{" "}
        <ScrollLink to="tutorial-1">1. First Bayesian Network</ScrollLink> | <ScrollLink to="tutorial-2">2. Probabilistic Inference</ScrollLink> | <ScrollLink to="tutorial-3">3. Learning from Data</ScrollLink> | <ScrollLink to="tutorial-4">4. Causal Inference</ScrollLink> | <ScrollLink to="tutorial-5">5. File Formats</ScrollLink> | <ScrollLink to="tutorial-6">6. Sampling</ScrollLink> | <ScrollLink to="tutorial-7">7. Advanced Models</ScrollLink> | <ScrollLink to="tutorial-8">8. Internal Libraries</ScrollLink>
      </nav>

      {/* ============================================================ */}
      {/* TUTORIAL 1 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-1">
        <h2>Tutorial 1: Building Your First Bayesian Network</h2>
        <p>
          This tutorial walks through creating a Bayesian network from scratch. You will create
          the classic "wet grass" model, add conditional probability distributions, validate the
          model, run an inference query, and save it to a file. By the end you will understand
          the core workflow for using pgmgo.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Go 1.21+ installed</li>
          <li>pgmgo installed: <code>go get github.com/asymmetric-effort/pgmgo</code></li>
        </ul>

        <h3>Introduction</h3>
        <p>
          A Bayesian network (BN) is a directed acyclic graph (DAG) where each node represents
          a random variable and each directed edge represents a conditional dependency. The key
          idea: each variable's probability distribution depends only on its parents in the graph.
          This allows compact representation of joint probability distributions.
        </p>
        <p>
          We will model the classic "wet grass" scenario. On a given day, it might be cloudy.
          Cloudy weather makes rain more likely and the sprinkler less likely. Both rain and the
          sprinkler can cause the grass to be wet.
        </p>

        <h3>Step 1: Create the Network Structure</h3>
        <p>
          Start by creating an empty BayesianNetwork, then add nodes (random variables) and
          directed edges (causal relationships).
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/models"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
    "os"
)

func main() {
    // Create an empty Bayesian network
    bn := models.NewBayesianNetwork()

    // Add four random variables as nodes
    bn.AddNode("Cloudy")
    bn.AddNode("Sprinkler")
    bn.AddNode("Rain")
    bn.AddNode("WetGrass")

    // Add directed edges representing causal influence:
    // Cloudy weather affects both the sprinkler and rain
    bn.AddEdge("Cloudy", "Sprinkler")
    bn.AddEdge("Cloudy", "Rain")
    // Both sprinkler and rain can cause wet grass
    bn.AddEdge("Sprinkler", "WetGrass")
    bn.AddEdge("Rain", "WetGrass")

    fmt.Println("Nodes:", bn.Nodes())
    // Output: Nodes: [Cloudy Sprinkler Rain WetGrass]

    fmt.Println("Edges:", bn.Edges())
    // Output: Edges: [[Cloudy Sprinkler] [Cloudy Rain] [Sprinkler WetGrass] [Rain WetGrass]]`}</code></pre>
        <p>
          At this point we have the graph structure but no probability information. The
          network knows the causal relationships but cannot answer probabilistic queries yet.
        </p>

        <h3>Step 2: Define States</h3>
        <p>
          Each node needs a list of possible states (discrete values it can take). States are
          represented as string slices. The order matters: state 0 is the first string, state 1
          is the second, and so on.
        </p>
        <pre><code>{`    // Define the possible values for each variable
    bn.SetStates("Cloudy", []string{"clear", "cloudy"})     // 0=clear, 1=cloudy
    bn.SetStates("Sprinkler", []string{"off", "on"})        // 0=off, 1=on
    bn.SetStates("Rain", []string{"no", "yes"})             // 0=no, 1=yes
    bn.SetStates("WetGrass", []string{"dry", "wet"})        // 0=dry, 1=wet`}</code></pre>

        <h3>Step 3: Add Conditional Probability Distributions (CPDs)</h3>
        <p>
          Each node needs a CPD that specifies P(node | parents). Root nodes (no parents) have
          marginal distributions. Child nodes have conditional distributions.
        </p>
        <p>
          <code>NewTabularCPD(variable, cardinality, values, parents, parentCardinalities)</code> creates
          a CPD. The <code>values</code> array is in column-major order: iterate over parent combinations
          first, then over the variable's states.
        </p>
        <pre><code>{`    // P(Cloudy) -- root node, no parents
    // 50% chance of clear, 50% chance of cloudy
    bn.SetCPD("Cloudy", factors.NewTabularCPD(
        "Cloudy", 2,                    // variable name, number of states
        []float64{0.5, 0.5},            // P(clear)=0.5, P(cloudy)=0.5
        nil, nil,                        // no parents
    ))

    // P(Sprinkler | Cloudy)
    // When clear: sprinkler on 50% of the time
    // When cloudy: sprinkler on only 10% of the time
    //
    // Values layout (column-major):
    //                  Cloudy=clear  Cloudy=cloudy
    // Sprinkler=off:     0.5           0.9
    // Sprinkler=on:      0.5           0.1
    bn.SetCPD("Sprinkler", factors.NewTabularCPD(
        "Sprinkler", 2,
        []float64{
            0.5, 0.9,   // P(off|clear)=0.5, P(off|cloudy)=0.9
            0.5, 0.1,   // P(on|clear)=0.5,  P(on|cloudy)=0.1
        },
        []string{"Cloudy"}, []int{2},
    ))

    // P(Rain | Cloudy)
    // When clear: 80% no rain. When cloudy: 80% rain.
    bn.SetCPD("Rain", factors.NewTabularCPD(
        "Rain", 2,
        []float64{
            0.8, 0.2,   // P(no|clear)=0.8, P(no|cloudy)=0.2
            0.2, 0.8,   // P(yes|clear)=0.2, P(yes|cloudy)=0.8
        },
        []string{"Cloudy"}, []int{2},
    ))

    // P(WetGrass | Sprinkler, Rain)
    // Two parents, each with 2 states = 4 columns
    // Column order: (S=off,R=no), (S=off,R=yes), (S=on,R=no), (S=on,R=yes)
    bn.SetCPD("WetGrass", factors.NewTabularCPD(
        "WetGrass", 2,
        []float64{
            // S=off,R=no  S=off,R=yes  S=on,R=no  S=on,R=yes
            1.0,          0.1,         0.1,       0.01,   // P(dry|...)
            0.0,          0.9,         0.9,       0.99,   // P(wet|...)
        },
        []string{"Sprinkler", "Rain"}, []int{2, 2},
    ))`}</code></pre>

        <h3>Step 4: Validate the Model</h3>
        <p>
          <code>CheckModel()</code> verifies that the graph is a valid DAG, every node has a CPD,
          CPD dimensions match the graph structure, and all probability distributions sum to 1.
        </p>
        <pre><code>{`    // Validate the complete model
    if err := bn.CheckModel(); err != nil {
        log.Fatal("Model validation failed:", err)
    }
    fmt.Println("Model is valid!")
    // Output: Model is valid!`}</code></pre>

        <h3>Step 5: Run an Inference Query</h3>
        <p>
          Convert the CPDs to Markov factors, create a Variable Elimination engine, and query
          posterior probabilities.
        </p>
        <pre><code>{`    // Convert CPDs to Markov factors for Variable Elimination
    facs, err := bn.ToMarkovFactors()
    if err != nil {
        log.Fatal(err)
    }
    ve := inference.NewVariableElimination(facs)

    // Query: P(Rain | WetGrass=wet)
    // "The grass is wet. Did it rain?"
    result, err := ve.Query(
        []string{"Rain"},                    // query variable
        map[string]int{"WetGrass": 1},       // evidence: WetGrass=1 means "wet"
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Rain | WetGrass=wet):", result.Values().Data())
    // Rain is more likely when grass is wet

    // MAP query: most likely assignment for Sprinkler given it rained
    assignment, err := ve.MAP(
        []string{"Sprinkler"},
        map[string]int{"Rain": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("MAP(Sprinkler | Rain=yes):", assignment)
    // Sprinkler is likely off when it's raining (both caused by cloudy weather)`}</code></pre>

        <h3>Step 6: Save the Model</h3>
        <pre><code>{`    // Save the model as a BIF file
    f, err := os.Create("wetgrass.bif")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    if err := readwrite.WriteBIF(f, bn); err != nil {
        log.Fatal(err)
    }
    fmt.Println("Model saved to wetgrass.bif")
}`}</code></pre>

        <h3>Expected Output</h3>
        <pre><code>{`Nodes: [Cloudy Sprinkler Rain WetGrass]
Edges: [[Cloudy Sprinkler] [Cloudy Rain] [Sprinkler WetGrass] [Rain WetGrass]]
Model is valid!
P(Rain | WetGrass=wet): [0.2921... 0.7079...]
MAP(Sprinkler | Rain=yes): map[Sprinkler:0]
Model saved to wetgrass.bif`}</code></pre>

        <h3>What's Next</h3>
        <p>
          Now that you can build and query a Bayesian network, explore:
        </p>
        <ul>
          <li><ScrollLink to="tutorial-2">Tutorial 2</ScrollLink>: Different inference methods (VE, BP, approximate)</li>
          <li><ScrollLink to="tutorial-3">Tutorial 3</ScrollLink>: Learning structure and parameters from data</li>
          <li><ScrollLink to="tutorial-4">Tutorial 4</ScrollLink>: Causal inference with do-calculus</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 2 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-2">
        <h2>Tutorial 2: Probabilistic Inference</h2>
        <p>
          This tutorial covers the inference algorithms in pgmgo: Variable Elimination (exact),
          Belief Propagation (exact), and Approximate Inference (sampling-based). You will learn
          how to compute posterior marginals, MAP assignments, and when to use each method.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Completed <ScrollLink to="tutorial-1">Tutorial 1</ScrollLink></li>
          <li>Understanding of conditional probability and Bayes' theorem</li>
        </ul>

        <h3>Introduction</h3>
        <p>
          Inference is the process of computing posterior probabilities given a model and evidence.
          There are two main types:
        </p>
        <ul>
          <li><strong>Marginal inference</strong>: P(query | evidence) -- what is the distribution of the query variables given what we observed?</li>
          <li><strong>MAP inference</strong>: argmax P(query | evidence) -- what is the most likely assignment of the query variables?</li>
        </ul>

        <h3>Step 1: Load a Model</h3>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/inference"
)

func main() {
    // Load the Asia (lung disease) model
    // 8 nodes: VisitAsia, Tuberculosis, Smoker, Lung, Bronc, TbOrCa, XRay, Dyspnea
    bn, err := example_models.Get("asia")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Loaded: %d nodes, %d edges\\n", len(bn.Nodes()), len(bn.Edges()))`}</code></pre>

        <h3>Step 2: Variable Elimination</h3>
        <p>
          Variable Elimination (VE) is an exact inference algorithm. It works by systematically
          eliminating (summing out) variables from the joint distribution. It is the default
          method and works well for small to medium networks.
        </p>
        <pre><code>{`    // Convert to factors
    facs, err := bn.ToMarkovFactors()
    if err != nil {
        log.Fatal(err)
    }
    ve := inference.NewVariableElimination(facs)

    // Marginal query: P(Dyspnea)
    // No evidence -- just the prior distribution
    result, err := ve.Query([]string{"Dyspnea"}, nil)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Dyspnea):", result.Values().Data())

    // Posterior query: P(Dyspnea | Smoker=yes)
    result, err = ve.Query(
        []string{"Dyspnea"},
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Dyspnea | Smoker=yes):", result.Values().Data())

    // Multiple query variables: P(Lung, Bronc | Smoker=yes)
    result, err = ve.Query(
        []string{"Lung", "Bronc"},
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Lung, Bronc | Smoker=yes):", result.Values().Data())

    // Multiple evidence: P(Dyspnea | Smoker=yes, VisitAsia=yes)
    result, err = ve.Query(
        []string{"Dyspnea"},
        map[string]int{"Smoker": 1, "VisitAsia": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Dyspnea | Smoker=yes, VisitAsia=yes):", result.Values().Data())`}</code></pre>

        <h3>Step 3: MAP Queries</h3>
        <p>
          MAP (Maximum A Posteriori) queries find the most likely assignment of query variables
          given evidence. This answers "what is the single best explanation?"
        </p>
        <pre><code>{`    // MAP: most likely values of Lung and Bronc given smoking
    assignment, err := ve.MAP(
        []string{"Lung", "Bronc"},
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("MAP(Lung, Bronc | Smoker=yes):", assignment)
    // Returns a map like: map[Bronc:1 Lung:0]
    // Meaning: most likely Bronc=yes, Lung=no`}</code></pre>

        <h3>Step 4: Belief Propagation</h3>
        <p>
          Belief Propagation (BP) is another exact inference algorithm that works on junction trees.
          It is more efficient than VE when you need to answer many queries on the same model,
          because calibration is done once and subsequent queries are fast.
        </p>
        <pre><code>{`    // Build junction tree
    jt, err := bn.ToJunctionTree()
    if err != nil {
        log.Fatal(err)
    }

    // Set up Belief Propagation
    cliques := jt.Cliques()
    separators := jt.SeparatorSets()
    cliqueFactors := make(map[int][]*factors.DiscreteFactor)
    for i, c := range cliques {
        fs := jt.GetCliqueFactors(c)
        if len(fs) > 0 {
            cliqueFactors[i] = fs
        }
    }
    bp := inference.NewBeliefPropagation(cliques, separators, cliqueFactors)

    // Calibrate (run message passing) -- do this once
    if err := bp.Calibrate(); err != nil {
        log.Fatal(err)
    }

    // Query (fast after calibration)
    bpResult, err := bp.Query(
        []string{"Dyspnea"},
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("BP P(Dyspnea | Smoker=yes):", bpResult.Values().Data())
    // Should match VE result`}</code></pre>

        <h3>Step 5: Approximate Inference</h3>
        <p>
          For large networks where exact inference is too slow, use sampling-based approximate
          inference. This generates many samples from the model and estimates posteriors from
          the empirical distribution.
        </p>
        <pre><code>{`    // Approximate inference using likelihood-weighted sampling
    approx, err := inference.NewApproxInference(bn, 10000) // 10,000 samples
    if err != nil {
        log.Fatal(err)
    }

    approxResult, err := approx.Query(
        []string{"Dyspnea"},
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Approx P(Dyspnea | Smoker=yes):", approxResult.Values().Data())
    // Close to but not exactly equal to exact result`}</code></pre>

        <h3>Comparing Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Type</th><th>Speed</th><th>Accuracy</th><th>When to Use</th></tr>
          </thead>
          <tbody>
            <tr><td>Variable Elimination</td><td>Exact</td><td>Fast for small networks</td><td>Exact</td><td>Default choice. Networks with up to ~50 nodes.</td></tr>
            <tr><td>Belief Propagation</td><td>Exact</td><td>Fast for many queries</td><td>Exact</td><td>When answering many queries on the same model.</td></tr>
            <tr><td>MPLP</td><td>Exact (MAP)</td><td>Fast</td><td>Exact for MAP</td><td>MAP inference on large networks.</td></tr>
            <tr><td>Approximate</td><td>Approximate</td><td>Scales to large networks</td><td>Approximate</td><td>Networks too large for exact inference.</td></tr>
          </tbody>
        </table>

        <h3>Expected Output</h3>
        <pre><code>{`Loaded: 8 nodes, 8 edges
P(Dyspnea): [0.560... 0.440...]
P(Dyspnea | Smoker=yes): [0.304... 0.696...]
MAP(Lung, Bronc | Smoker=yes): map[Bronc:1 Lung:0]
BP P(Dyspnea | Smoker=yes): [0.304... 0.696...]
Approx P(Dyspnea | Smoker=yes): [~0.30 ~0.70]`}</code></pre>

        <h3>What's Next</h3>
        <ul>
          <li><ScrollLink to="tutorial-3">Tutorial 3</ScrollLink>: Learn models from data instead of specifying them manually</li>
          <li><ScrollLink to="tutorial-4">Tutorial 4</ScrollLink>: Causal inference -- observation vs. intervention</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 3 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-3">
        <h2>Tutorial 3: Learning from Data</h2>
        <p>
          This tutorial covers the complete learning pipeline: loading data, learning network
          structure using score-based (HillClimb) and constraint-based (PC) methods, fitting
          parameters with MLE and Bayesian estimation, and evaluating the result.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Completed <ScrollLink to="tutorial-1">Tutorial 1</ScrollLink> and <ScrollLink to="tutorial-2">Tutorial 2</ScrollLink></li>
          <li>CSV data file with discrete observations</li>
        </ul>

        <h3>Introduction</h3>
        <p>
          In practice, you often have data but no known model structure. pgmgo provides two
          families of structure learning algorithms:
        </p>
        <ul>
          <li><strong>Score-based</strong>: Search the space of DAGs to maximize a scoring function (BIC, BDeu, K2). Includes HillClimb, GES, ExhaustiveSearch, TreeSearch.</li>
          <li><strong>Constraint-based</strong>: Use conditional independence tests to discover the structure. Includes PC.</li>
          <li><strong>Hybrid</strong>: Combine both approaches. Includes MMHC.</li>
        </ul>

        <h3>Step 1: Generate or Load Training Data</h3>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
    "github.com/asymmetric-effort/pgmgo/src/ci_tests"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/learning"
    "github.com/asymmetric-effort/pgmgo/src/metrics"
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/src/structure_score"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

func main() {
    // Load the ground-truth Asia model
    trueBN, _ := example_models.Get("asia")

    // Generate 5000 training samples
    bms, _ := sampling.NewBayesianModelSampling(trueBN, 42)
    data, _ := bms.ForwardSample(5000)
    fmt.Printf("Training data: %d rows, %d columns\\n", data.NRows(), len(data.Columns()))

    // Alternatively, load from CSV:
    // data, _ := tabgo.ReadCSV("observations.csv")`}</code></pre>

        <h3>Step 2: Score-Based Learning (Hill-Climb with BIC)</h3>
        <p>
          Hill-climbing starts with an empty graph and iteratively adds, removes, or reverses
          edges to maximize the BIC score. It is the most commonly used structure learning method.
        </p>
        <pre><code>{`    // Choose a scoring function
    scorer := structure_score.NewBIC()

    // Run hill-climbing
    hc := learning.NewHillClimbSearch(data, scorer.LocalScore)
    hcBN, err := hc.Estimate()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Hill-Climb: %d nodes, %d edges\\n",
        len(hcBN.Nodes()), len(hcBN.Edges()))`}</code></pre>

        <h3>Step 3: Constraint-Based Learning (PC Algorithm)</h3>
        <p>
          The PC algorithm uses conditional independence (CI) tests to discover the graph skeleton
          (undirected edges), then orients edges based on v-structures and orientation rules.
        </p>
        <pre><code>{`    // Wrap the CI test function for the learning API
    // Signature: (x, y string, z []string, data *DataFrame, significance float64) -> (stat, pValue, independent)
    ciTest := func(x, y string, z []string, d *tabgo.DataFrame, sig float64) (float64, float64, bool) {
        return ci_tests.ChiSquare(x, y, z, d, sig)
    }

    // Run PC with significance level 0.05
    pc := learning.NewPC(data, ciTest, 0.05)
    pcBN, err := pc.EstimateBN()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("PC: %d nodes, %d edges\\n",
        len(pcBN.Nodes()), len(pcBN.Edges()))`}</code></pre>

        <h3>Step 4: Other Structure Learning Methods</h3>
        <pre><code>{`    // GES (Greedy Equivalence Search)
    ges := learning.NewGES(data, scorer.LocalScore)
    gesPDAG, err := ges.Estimate()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("GES: learned PDAG")

    // Tree search (Chow-Liu) -- learns tree-structured BN
    ts := learning.NewTreeSearch(data)
    treeBN, err := ts.Estimate()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Tree: %d nodes, %d edges\\n",
        len(treeBN.Nodes()), len(treeBN.Edges()))

    // Exhaustive search (small networks only, up to ~5 nodes)
    // es := learning.NewExhaustiveSearch(smallData, scorer.LocalScore)
    // optBN, _ := es.Estimate()`}</code></pre>

        <h3>Step 5: Fit Parameters (MLE)</h3>
        <p>
          After learning the structure, fit conditional probability distributions using Maximum
          Likelihood Estimation.
        </p>
        <pre><code>{`    // MLE: compute CPD parameters by counting occurrences in data
    mle := learning.NewMLE(hcBN, data)
    if err := mle.Estimate(); err != nil {
        log.Fatal(err)
    }
    fmt.Println("MLE parameters fitted")`}</code></pre>

        <h3>Step 6: Fit Parameters (Bayesian Estimation)</h3>
        <p>
          Bayesian estimation adds a Dirichlet prior, which helps with small datasets or
          rare state combinations.
        </p>
        <pre><code>{`    // Bayesian estimation with BDeu prior, equivalent sample size = 5.0
    be := learning.NewBayesianEstimator(hcBN, data, learning.BDeu, 5.0)
    if err := be.Estimate(); err != nil {
        log.Fatal(err)
    }
    fmt.Println("Bayesian parameters fitted")`}</code></pre>

        <h3>Step 7: EM for Incomplete Data</h3>
        <pre><code>{`    // If data has missing values, use Expectation-Maximization
    // em := learning.NewEM(hcBN, incompleteData, nil, 100, 1e-6)
    // if err := em.Estimate(); err != nil {
    //     log.Fatal(err)
    // }`}</code></pre>

        <h3>Step 8: Evaluate the Learned Structure</h3>
        <pre><code>{`    // Convert BNs to DiGraphs for comparison
    toDigraph := func(bn interface{ Nodes() []string; Edges() [][2]string }) *graphgo.DiGraph {
        g := graphgo.NewDiGraph()
        for _, n := range bn.Nodes() { g.AddNode(n) }
        for _, e := range bn.Edges() { g.AddEdge(e[0], e[1]) }
        return g
    }

    trueG := toDigraph(trueBN)
    hcG := toDigraph(hcBN)
    pcG := toDigraph(pcBN)

    // Structural Hamming Distance (lower is better)
    fmt.Println("Hill-Climb SHD:", metrics.SHD(trueG, hcG))
    fmt.Println("PC SHD:", metrics.SHD(trueG, pcG))

    // Adjacency confusion matrix
    aTP, aFP, aTN, aFN := metrics.AdjacencyConfusionMatrix(trueG, hcG)
    fmt.Printf("HC Adjacency: TP=%d FP=%d TN=%d FN=%d\\n", aTP, aFP, aTN, aFN)

    // Orientation confusion matrix
    oTP, oFP, oTN, oFN := metrics.OrientationConfusionMatrix(trueG, hcG)
    fmt.Printf("HC Orientation: TP=%d FP=%d TN=%d FN=%d\\n", oTP, oFP, oTN, oFN)`}</code></pre>

        <h3>Step 9: Run Inference on the Learned Model</h3>
        <pre><code>{`    // The learned model is now fully parameterized -- use it for inference
    facs, _ := hcBN.ToMarkovFactors()
    ve := inference.NewVariableElimination(facs)
    result, _ := ve.Query(
        []string{"Dyspnea"},
        map[string]int{"Smoker": 1},
    )
    fmt.Println("Learned model P(Dyspnea | Smoker=yes):", result.Values().Data())
}`}</code></pre>

        <h3>Choosing a Scoring Function</h3>
        <table>
          <thead>
            <tr><th>Score</th><th>Best For</th><th>Notes</th></tr>
          </thead>
          <tbody>
            <tr><td>BIC</td><td>General use</td><td>Good balance of fit and complexity. Consistent (recovers true structure with enough data).</td></tr>
            <tr><td>BDeu</td><td>Small datasets</td><td>Bayesian prior helps when data is scarce. Set equivalent sample size carefully.</td></tr>
            <tr><td>K2</td><td>Fast evaluation</td><td>Simple uniform prior. Fast to compute.</td></tr>
            <tr><td>AIC</td><td>Less penalization</td><td>Tends to learn denser networks than BIC.</td></tr>
          </tbody>
        </table>

        <h3>What's Next</h3>
        <ul>
          <li><ScrollLink to="tutorial-4">Tutorial 4</ScrollLink>: Causal inference on the learned model</li>
          <li><ScrollLink to="tutorial-6">Tutorial 6</ScrollLink>: Generating data via sampling</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 4 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-4">
        <h2>Tutorial 4: Causal Inference</h2>
        <p>
          This tutorial covers causal reasoning with pgmgo. You will learn the difference between
          observational and interventional queries, use do-calculus, compute average treatment effects,
          and use back-door adjustment.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Completed <ScrollLink to="tutorial-1">Tutorial 1</ScrollLink> and <ScrollLink to="tutorial-2">Tutorial 2</ScrollLink></li>
          <li>Basic understanding of causation vs. correlation</li>
        </ul>

        <h3>Introduction</h3>
        <p>
          Bayesian networks encode causal relationships through directed edges. This enables
          two types of reasoning:
        </p>
        <ul>
          <li><strong>Observational</strong>: P(Y | X=x) -- "If I observe X=x, what do I expect for Y?" This reflects correlation and may be confounded.</li>
          <li><strong>Interventional</strong>: P(Y | do(X=x)) -- "If I force X to be x (intervene), what happens to Y?" This reflects causation. The do() operator "cuts" incoming edges to X, removing confounding.</li>
        </ul>

        <h3>Step 1: Build a Causal Model</h3>
        <p>
          Consider a model of smoking, tar deposits, and cancer. Smoking causes tar deposits,
          and both smoking and tar cause cancer.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/identification"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

func main() {
    bn := models.NewBayesianNetwork()
    bn.AddNode("Smoking")
    bn.AddNode("Tar")
    bn.AddNode("Cancer")

    // Causal structure:
    // Smoking -> Tar -> Cancer
    // Smoking -> Cancer (direct effect)
    bn.AddEdge("Smoking", "Tar")
    bn.AddEdge("Smoking", "Cancer")
    bn.AddEdge("Tar", "Cancer")

    bn.SetStates("Smoking", []string{"no", "yes"})
    bn.SetStates("Tar", []string{"low", "high"})
    bn.SetStates("Cancer", []string{"no", "yes"})

    bn.SetCPD("Smoking", factors.NewTabularCPD(
        "Smoking", 2, []float64{0.7, 0.3}, nil, nil,
    ))
    bn.SetCPD("Tar", factors.NewTabularCPD(
        "Tar", 2,
        []float64{0.95, 0.1, 0.05, 0.9},
        []string{"Smoking"}, []int{2},
    ))
    bn.SetCPD("Cancer", factors.NewTabularCPD(
        "Cancer", 2,
        []float64{
            0.98, 0.9, 0.7, 0.05,    // P(no cancer | ...)
            0.02, 0.1, 0.3, 0.95,    // P(cancer | ...)
        },
        []string{"Smoking", "Tar"}, []int{2, 2},
    ))
    bn.CheckModel()`}</code></pre>

        <h3>Step 2: Observational vs. Interventional Query</h3>
        <pre><code>{`    // OBSERVATIONAL: P(Cancer | Smoking=yes)
    // "Among people who smoke, what is the cancer rate?"
    facs, _ := bn.ToMarkovFactors()
    ve := inference.NewVariableElimination(facs)
    obsResult, _ := ve.Query(
        []string{"Cancer"},
        map[string]int{"Smoking": 1},
    )
    fmt.Println("Observational P(Cancer | Smoking=yes):", obsResult.Values().Data())

    // INTERVENTIONAL: P(Cancer | do(Smoking=yes))
    // "If we forced everyone to smoke, what would the cancer rate be?"
    // The do() operator removes incoming edges to Smoking,
    // isolating the causal effect from confounders.
    ci, err := inference.NewCausalInference(bn)
    if err != nil {
        log.Fatal(err)
    }
    doResult, _ := ci.Query(
        []string{"Cancer"},
        map[string]int{"Smoking": 1},   // do(Smoking=yes)
        nil,                              // no additional evidence
    )
    fmt.Println("Interventional P(Cancer | do(Smoking=yes)):", doResult.Values().Data())
    // In this model, the two may differ because Smoking has
    // both direct and indirect effects on Cancer.`}</code></pre>

        <h3>Step 3: Computing Average Treatment Effect (ATE)</h3>
        <p>
          The ATE measures the causal effect of a treatment (Smoking) on an outcome (Cancer):
          ATE = P(Cancer=yes | do(Smoking=yes)) - P(Cancer=yes | do(Smoking=no))
        </p>
        <pre><code>{`    // P(Cancer | do(Smoking=yes))
    doSmokeYes, _ := ci.Query(
        []string{"Cancer"},
        map[string]int{"Smoking": 1},
        nil,
    )

    // P(Cancer | do(Smoking=no))
    doSmokeNo, _ := ci.Query(
        []string{"Cancer"},
        map[string]int{"Smoking": 0},
        nil,
    )

    // ATE = P(Cancer=yes | do(Smoke=yes)) - P(Cancer=yes | do(Smoke=no))
    pCancerYesTreated := doSmokeYes.Values().Data()[1]
    pCancerYesControl := doSmokeNo.Values().Data()[1]
    ate := pCancerYesTreated - pCancerYesControl
    fmt.Printf("ATE of Smoking on Cancer: %.4f\\n", ate)
    // Positive ATE means smoking causes increased cancer risk`}</code></pre>

        <h3>Step 4: Interventional Query with Evidence</h3>
        <pre><code>{`    // P(Cancer | do(Tar=high), Smoking=no)
    // "If we forced tar deposits to be high but the person doesn't smoke,
    //  what is the cancer risk?"
    result, _ := ci.Query(
        []string{"Cancer"},
        map[string]int{"Tar": 1},         // intervene on Tar
        map[string]int{"Smoking": 0},      // observe Smoking=no
    )
    fmt.Println("P(Cancer | do(Tar=high), Smoking=no):", result.Values().Data())`}</code></pre>

        <h3>Step 5: Causal Identification</h3>
        <p>
          Before computing a causal effect, check whether it is identifiable (computable from
          observational data) using the back-door or front-door criterion.
        </p>
        <pre><code>{`    // Back-door criterion: find an adjustment set
    adj := identification.NewAdjustment(bn)
    adjustmentSet, err := adj.GetAdjustmentSet("Smoking", "Cancer")
    if err == nil {
        fmt.Println("Back-door adjustment set:", adjustmentSet)
        // You can use this set to compute causal effects from
        // observational data by adjusting (conditioning) on these variables.
    } else {
        fmt.Println("No valid back-door adjustment set found")
    }

    // Front-door criterion
    fd := identification.NewFrontdoor(bn)
    frontdoorSet, err := fd.GetFrontdoorSet("Smoking", "Cancer")
    if err == nil {
        fmt.Println("Front-door set:", frontdoorSet)
        // Tar mediates the effect and can be used for front-door adjustment
    } else {
        fmt.Println("No valid front-door set found")
    }
}`}</code></pre>

        <h3>The Intuition</h3>
        <p>
          <strong>Why do observation and intervention differ?</strong> When you observe X=x,
          you are conditioning on X. This selects a subpopulation where X happens to be x,
          which may carry information about confounders. When you intervene (do(X=x)), you
          force X to be x for everyone, breaking the connection between X and its causes.
          This isolates the downstream causal effect.
        </p>
        <p>
          In a simple chain A -&gt; B -&gt; C, observing B gives information about A (through
          the chain), while do(B) does not -- the intervention breaks the A-&gt;B link.
        </p>

        <h3>Using the CLI</h3>
        <pre><code>{`# Same queries from the command line
$ pgmgo do model.bif --intervention Smoking=1 --query Cancer
Cancer	P
0	0.350000
1	0.650000

$ pgmgo do model.bif --intervention Tar=1 --query Cancer --evidence Smoking=0`}</code></pre>

        <h3>What's Next</h3>
        <ul>
          <li><ScrollLink to="tutorial-5">Tutorial 5</ScrollLink>: Working with different file formats</li>
          <li><ScrollLink to="tutorial-7">Tutorial 7</ScrollLink>: Advanced model types (SEMs, Markov networks)</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 5 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-5">
        <h2>Tutorial 5: Working with File Formats</h2>
        <p>
          pgmgo supports 10 file formats for model serialization. This tutorial covers reading,
          writing, converting between formats, and using the CLI for format conversion.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Completed <ScrollLink to="tutorial-1">Tutorial 1</ScrollLink></li>
          <li>A BIF file (or create one in Tutorial 1)</li>
        </ul>

        <h3>Step 1: Reading Models from Files</h3>
        <pre><code>{`package main

import (
    "fmt"
    "log"
    "os"

    "github.com/asymmetric-effort/pgmgo/src/readwrite"
)

func main() {
    // Read a BIF file
    f, err := os.Open("asia.bif")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    bn, err := readwrite.ReadBIF(f)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Loaded from BIF: %d nodes, %d edges\\n",
        len(bn.Nodes()), len(bn.Edges()))

    // Read other formats -- same pattern
    // f2, _ := os.Open("model.xmlbif")
    // bn2, _ := readwrite.ReadXMLBIF(f2)
    //
    // f3, _ := os.Open("model.net")
    // bn3, _ := readwrite.ReadNET(f3)
    //
    // f4, _ := os.Open("model.uai")
    // bn4, _ := readwrite.ReadUAI(f4)
    //
    // f5, _ := os.Open("model.xdsl")
    // bn5, _ := readwrite.ReadXDSL(f5)`}</code></pre>

        <h3>Step 2: Writing Models to Files</h3>
        <pre><code>{`    // Write as BIF
    bifOut, _ := os.Create("output.bif")
    readwrite.WriteBIF(bifOut, bn)
    bifOut.Close()

    // Write as XMLBIF
    xmlbifOut, _ := os.Create("output.xmlbif")
    readwrite.WriteXMLBIF(xmlbifOut, bn)
    xmlbifOut.Close()

    // Write as NET (Hugin format)
    netOut, _ := os.Create("output.net")
    readwrite.WriteNET(netOut, bn)
    netOut.Close()

    // Write as UAI (competition format)
    uaiOut, _ := os.Create("output.uai")
    readwrite.WriteUAI(uaiOut, bn)
    uaiOut.Close()

    // Write as XDSL (GeNIe format)
    xdslOut, _ := os.Create("output.xdsl")
    readwrite.WriteXDSL(xdslOut, bn)
    xdslOut.Close()`}</code></pre>

        <h3>Step 3: JSON for Web Applications</h3>
        <p>
          JSON is ideal for REST APIs, web dashboards, and JavaScript interop.
        </p>
        <pre><code>{`    // Write model as JSON
    jsonOut, _ := os.Create("model.json")
    readwrite.WriteJSONModel(jsonOut, bn)
    jsonOut.Close()
    fmt.Println("Wrote model.json")

    // Read model from JSON
    jsonIn, _ := os.Open("model.json")
    bnFromJSON, _ := readwrite.ReadJSONModel(jsonIn)
    jsonIn.Close()
    fmt.Printf("Read from JSON: %d nodes\\n", len(bnFromJSON.Nodes()))`}</code></pre>

        <h3>Step 4: CSV for Data Exchange</h3>
        <pre><code>{`    // CSV model serialization
    csvOut, _ := os.Create("model.csv")
    readwrite.WriteCSVModel(csvOut, bn)
    csvOut.Close()

    // XML model serialization
    xmlOut, _ := os.Create("model.xml")
    readwrite.WriteXMLModel(xmlOut, bn)
    xmlOut.Close()`}</code></pre>

        <h3>Step 5: Format Conversion Pipeline</h3>
        <pre><code>{`    // Read NET, convert to BIF and JSON
    netFile, _ := os.Open("model.net")
    bnFromNET, _ := readwrite.ReadNET(netFile)
    netFile.Close()

    bifFile, _ := os.Create("converted.bif")
    readwrite.WriteBIF(bifFile, bnFromNET)
    bifFile.Close()

    jsonFile, _ := os.Create("converted.json")
    readwrite.WriteJSONModel(jsonFile, bnFromNET)
    jsonFile.Close()
    fmt.Println("Converted NET -> BIF and JSON")
}`}</code></pre>

        <h3>Step 6: CLI Commands for Format Conversion</h3>
        <pre><code>{`# Convert BIF to XMLBIF
$ pgmgo convert --input asia.bif --from bif --to xmlbif --output asia.xmlbif

# Convert NET to BIF
$ pgmgo convert --input model.net --from net --to bif --output model.bif

# Convert XDSL to UAI (for competition submission)
$ pgmgo convert --input model.xdsl --from xdsl --to uai --output model.uai

# Validate a model after conversion
$ pgmgo validate output.bif

# Inspect a model
$ pgmgo info output.bif`}</code></pre>

        <h3>Step 7: Working with Built-in Example Models</h3>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/example_models"

// List all 25 available models
for _, name := range example_models.List() {
    fmt.Println(name)
}

// Load and export a model to BIF
bn, _ := example_models.Get("alarm")
f, _ := os.Create("alarm.bif")
readwrite.WriteBIF(f, bn)
f.Close()

// Load and export to JSON for a web API
bn, _ = example_models.Get("asia")
jsonOut, _ := os.Create("asia.json")
readwrite.WriteJSONModel(jsonOut, bn)
jsonOut.Close()`}</code></pre>

        <h3>Format Comparison</h3>
        <table>
          <thead>
            <tr><th>Format</th><th>Human Readable</th><th>Tool Support</th><th>Best For</th></tr>
          </thead>
          <tbody>
            <tr><td>BIF</td><td>Yes</td><td>Broad</td><td>General exchange, version control</td></tr>
            <tr><td>XMLBIF</td><td>Somewhat</td><td>Broad</td><td>XML-based toolchains</td></tr>
            <tr><td>NET</td><td>Yes</td><td>Hugin</td><td>Hugin software users</td></tr>
            <tr><td>UAI</td><td>Minimal</td><td>Competitions</td><td>UAI inference competitions</td></tr>
            <tr><td>XDSL</td><td>Somewhat</td><td>GeNIe/SMILE</td><td>GeNIe software users</td></tr>
            <tr><td>JSON</td><td>Yes</td><td>Universal</td><td>Web APIs, JavaScript interop</td></tr>
            <tr><td>CSV</td><td>Yes</td><td>Universal</td><td>Spreadsheets, simple tools</td></tr>
            <tr><td>XML</td><td>Somewhat</td><td>Universal</td><td>Enterprise systems</td></tr>
          </tbody>
        </table>

        <h3>What's Next</h3>
        <ul>
          <li><ScrollLink to="tutorial-6">Tutorial 6</ScrollLink>: Sampling data from models</li>
          <li><ScrollLink to="tutorial-8">Tutorial 8</ScrollLink>: Using internal libraries for data processing</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 6 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-6">
        <h2>Tutorial 6: Sampling and Monte Carlo Methods</h2>
        <p>
          This tutorial covers all sampling methods in pgmgo: forward sampling, rejection sampling,
          likelihood-weighted sampling, and Gibbs sampling. You will learn when to use each method
          and how to compare empirical distributions against exact marginals.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Completed <ScrollLink to="tutorial-1">Tutorial 1</ScrollLink></li>
          <li>Basic understanding of Monte Carlo methods</li>
        </ul>

        <h3>Introduction</h3>
        <p>
          Sampling generates data points (samples) from the joint distribution defined by a
          Bayesian network. Uses include:
        </p>
        <ul>
          <li>Generating synthetic training data</li>
          <li>Approximate inference when exact methods are too expensive</li>
          <li>Validating models by comparing sampled distributions to expected distributions</li>
          <li>Monte Carlo estimation of probabilities and expectations</li>
        </ul>

        <h3>Step 1: Forward Sampling</h3>
        <p>
          Forward sampling generates samples by traversing the DAG in topological order. Each
          variable is sampled from its CPD given its parents' values. This is the simplest and
          most efficient method but cannot incorporate evidence.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func main() {
    bn, _ := example_models.Get("asia")

    // Create sampler with seed=42 for reproducibility
    bms, err := sampling.NewBayesianModelSampling(bn, 42)
    if err != nil {
        log.Fatal(err)
    }

    // Generate 10,000 forward samples
    samples, err := bms.ForwardSample(10000)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Generated %d samples with %d columns\\n",
        samples.NRows(), len(samples.Columns()))

    // Estimate P(Smoker=yes) from samples
    smokerCol := samples.Column("Smoker")
    counts := smokerCol.ValueCounts()
    fmt.Println("Smoker value counts:", counts)
    // Should be approximately 50/50 (depends on the model)

    // Save to CSV
    tabgo.WriteCSV(samples, "asia_samples.csv")
    fmt.Println("Saved to asia_samples.csv")`}</code></pre>

        <h3>Step 2: Rejection Sampling</h3>
        <p>
          Rejection sampling incorporates evidence by generating forward samples and keeping
          only those consistent with the evidence. Simple but inefficient when evidence is rare.
        </p>
        <pre><code>{`    // Rejection sampling: generate samples consistent with Smoker=yes
    rejSamples, err := bms.RejectionSample(1000, map[string]int{"Smoker": 1})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Rejection samples: %d (all have Smoker=1)\\n", rejSamples.NRows())

    // Estimate P(Dyspnea=yes | Smoker=yes) from rejection samples
    dyspneaCol := rejSamples.Column("Dyspnea")
    dyspneaCounts := dyspneaCol.ValueCounts()
    fmt.Println("Dyspnea | Smoker=yes:", dyspneaCounts)`}</code></pre>

        <h3>Step 3: Likelihood-Weighted Sampling</h3>
        <p>
          Likelihood-weighted sampling is more efficient than rejection sampling. Instead of
          discarding samples, it assigns weights to each sample based on the likelihood of the
          evidence. All samples are used, but weighted.
        </p>
        <pre><code>{`    // Likelihood-weighted sampling with evidence
    lwSamples, weights, err := bms.LikelihoodWeightedSample(
        5000,
        map[string]int{"Smoker": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("LW samples: %d with weights\\n", lwSamples.NRows())

    // The weights account for the evidence
    // Use weighted counts to estimate posteriors
    _ = weights`}</code></pre>

        <h3>Step 4: Gibbs Sampling (MCMC)</h3>
        <p>
          Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method. It starts with an initial
          assignment and iteratively resamples each variable from its full conditional distribution.
          It can handle evidence and is effective for large networks.
        </p>
        <pre><code>{`    // Create Gibbs sampler
    gs, err := sampling.NewGibbsSampling(bn, 42)
    if err != nil {
        log.Fatal(err)
    }

    // Parameters:
    //   nSamples=2000 -- number of samples to collect
    //   burnIn=500    -- discard first 500 samples (let chain converge)
    //   thinning=2    -- keep every 2nd sample (reduce autocorrelation)
    //   evidence      -- fix these variables
    gibbsSamples, err := gs.Sample(
        2000,                            // number of samples
        500,                             // burn-in
        2,                               // thinning interval
        map[string]int{"Smoker": 1},     // evidence
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Gibbs samples: %d\\n", gibbsSamples.NRows())

    // Estimate P(Dyspnea | Smoker=yes) from Gibbs samples
    gibbsDyspnea := gibbsSamples.Column("Dyspnea")
    fmt.Println("Gibbs Dyspnea | Smoker=yes:", gibbsDyspnea.ValueCounts())`}</code></pre>

        <h3>Step 5: Compare Empirical vs. Exact Marginals</h3>
        <pre><code>{`    import "github.com/asymmetric-effort/pgmgo/src/inference"

    // Exact posterior: P(Dyspnea | Smoker=yes)
    facs, _ := bn.ToMarkovFactors()
    ve := inference.NewVariableElimination(facs)
    exact, _ := ve.Query([]string{"Dyspnea"}, map[string]int{"Smoker": 1})
    fmt.Println("Exact P(Dyspnea | Smoker=yes):", exact.Values().Data())

    // The sampling-based estimates should converge to the exact values
    // as the number of samples increases.
    fmt.Println("\\nComparison:")
    fmt.Println("  Exact:     ", exact.Values().Data())
    fmt.Println("  Rejection: (from counts above)")
    fmt.Println("  Gibbs:     (from counts above)")
}`}</code></pre>

        <h3>When to Use Each Method</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Evidence</th><th>Efficiency</th><th>Convergence</th><th>Best For</th></tr>
          </thead>
          <tbody>
            <tr><td>Forward</td><td>No</td><td>High</td><td>Immediate</td><td>Generating training data, no-evidence estimates</td></tr>
            <tr><td>Rejection</td><td>Yes</td><td>Low (if evidence is rare)</td><td>Immediate</td><td>Simple evidence, high-probability evidence</td></tr>
            <tr><td>Likelihood-Weighted</td><td>Yes</td><td>Medium</td><td>Immediate</td><td>General evidence, moderate networks</td></tr>
            <tr><td>Gibbs</td><td>Yes</td><td>High</td><td>Requires burn-in</td><td>Large networks, complex evidence</td></tr>
          </tbody>
        </table>

        <h3>CLI Sampling</h3>
        <pre><code>{`# Forward sampling
$ pgmgo sample --model asia.bif --n 5000 --seed 42 --output samples.csv

# Rejection sampling with evidence
$ pgmgo sample --model asia.bif --n 500 --method rejection --evidence Smoker=1 --output rej.csv

# Gibbs sampling with evidence
$ pgmgo sample --model asia.bif --n 2000 --method gibbs --evidence Smoker=1 --output gibbs.csv`}</code></pre>

        <h3>What's Next</h3>
        <ul>
          <li><ScrollLink to="tutorial-3">Tutorial 3</ScrollLink>: Use sampled data for structure learning</li>
          <li><ScrollLink to="tutorial-7">Tutorial 7</ScrollLink>: Advanced model types</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 7 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-7">
        <h2>Tutorial 7: Advanced Models</h2>
        <p>
          Beyond the standard BayesianNetwork, pgmgo provides several specialized model types
          for different use cases. This tutorial covers Dynamic Bayesian Networks, Markov Networks,
          Naive Bayes classifiers, and Structural Equation Models.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>Completed Tutorials 1-3</li>
          <li>Understanding of the BayesianNetwork model type</li>
        </ul>

        <h3>Dynamic Bayesian Networks (DBN)</h3>
        <p>
          A Dynamic Bayesian Network models temporal processes. It defines relationships between
          variables at time t and time t+1, allowing inference about how state evolves over time.
        </p>
        <pre><code>{`import (
    "github.com/asymmetric-effort/pgmgo/src/models"
    "github.com/asymmetric-effort/pgmgo/src/inference"
)

// Create a 2-time-slice DBN
dbn := models.NewDynamicBayesianNetwork()

// Add nodes at time 0 and time 1
dbn.AddNode("Weather_0")
dbn.AddNode("Weather_1")
dbn.AddNode("Mood_0")
dbn.AddNode("Mood_1")

// Intra-slice edges (within same time step)
dbn.AddEdge("Weather_0", "Mood_0")
dbn.AddEdge("Weather_1", "Mood_1")

// Inter-slice edges (across time steps)
dbn.AddEdge("Weather_0", "Weather_1")  // weather persists
dbn.AddEdge("Mood_0", "Mood_1")        // mood persists

// Set states and CPDs (similar to regular BN)
// ... (states and CPDs for each node)

// DBN-specific inference
dbnInf, _ := inference.NewDBNInference(dbn)
// Query variables at future time steps
// result, _ := dbnInf.Query(...)
_ = dbnInf`}</code></pre>

        <h3>Markov Networks (Undirected Graphical Models)</h3>
        <p>
          Markov Networks use undirected edges and factor potentials instead of directed edges
          and CPDs. They are useful when the direction of influence is unknown or symmetric.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/models"

// Create a Markov Network
mn := models.NewMarkovNetwork()

// Add nodes
mn.AddNode("A")
mn.AddNode("B")
mn.AddNode("C")

// Add undirected edges
mn.AddEdge("A", "B")
mn.AddEdge("B", "C")

// Set states
mn.SetStates("A", []string{"a0", "a1"})
mn.SetStates("B", []string{"b0", "b1"})
mn.SetStates("C", []string{"c0", "c1"})

// Add factor potentials (not CPDs -- these don't need to sum to 1)
// Factor over {A, B}
mn.AddFactor(factors.NewDiscreteFactor(
    []string{"A", "B"},
    []int{2, 2},
    []float64{30, 5, 1, 10},  // compatibility scores
))

// Factor over {B, C}
mn.AddFactor(factors.NewDiscreteFactor(
    []string{"B", "C"},
    []int{2, 2},
    []float64{100, 1, 1, 100},
))

// Run inference (Belief Propagation works on Markov Networks)
// The partition function normalizes the potentials to probabilities`}</code></pre>

        <h3>Naive Bayes Classifier</h3>
        <p>
          A Naive Bayes classifier is a Bayesian network where a single class variable is the
          parent of all feature variables, which are assumed conditionally independent given
          the class.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/models"

// Create a Naive Bayes classifier
nb := models.NewNaiveBayes()

// Set the class variable and features
nb.AddNode("Spam")           // class variable
nb.AddNode("HasLink")        // feature 1
nb.AddNode("HasAttachment")  // feature 2
nb.AddNode("LongSubject")    // feature 3

// Edges from class to all features (automatic in NaiveBayes)
nb.AddEdge("Spam", "HasLink")
nb.AddEdge("Spam", "HasAttachment")
nb.AddEdge("Spam", "LongSubject")

// Set states and CPDs
nb.SetStates("Spam", []string{"no", "yes"})
nb.SetStates("HasLink", []string{"no", "yes"})
nb.SetStates("HasAttachment", []string{"no", "yes"})
nb.SetStates("LongSubject", []string{"no", "yes"})

nb.SetCPD("Spam", factors.NewTabularCPD(
    "Spam", 2, []float64{0.7, 0.3}, nil, nil,
))
nb.SetCPD("HasLink", factors.NewTabularCPD(
    "HasLink", 2,
    []float64{0.9, 0.2, 0.1, 0.8},  // links much more common in spam
    []string{"Spam"}, []int{2},
))
// ... CPDs for HasAttachment, LongSubject

// Classify: P(Spam | HasLink=yes, HasAttachment=no, LongSubject=yes)
facs, _ := nb.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)
result, _ := ve.Query(
    []string{"Spam"},
    map[string]int{"HasLink": 1, "HasAttachment": 0, "LongSubject": 1},
)
fmt.Println("P(Spam | features):", result.Values().Data())`}</code></pre>

        <h3>Structural Equation Models (SEM)</h3>
        <p>
          SEMs define variables as explicit equations of their parents plus noise terms. They
          are widely used in causal inference and econometrics.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/models"

// Create a Structural Equation Model
sem := models.NewSEM()

// Add variables and structural equations
sem.AddNode("Education")
sem.AddNode("Income")
sem.AddNode("Health")

// Causal structure
sem.AddEdge("Education", "Income")
sem.AddEdge("Education", "Health")
sem.AddEdge("Income", "Health")

// Define equations:
// Income = beta1 * Education + noise1
// Health = beta2 * Education + beta3 * Income + noise2
// ...

// SEM-specific estimation
// semEst := learning.NewSEMEstimator(sem, data)
// semEst.Estimate()`}</code></pre>

        <h3>Linear Gaussian BN</h3>
        <p>
          For continuous variables with linear relationships and Gaussian noise, use
          LinearGaussianBN. Each variable is defined as a linear combination of its parents
          plus Gaussian noise.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/models"

lgbn := models.NewLinearGaussianBN()
lgbn.AddNode("X")
lgbn.AddNode("Y")
lgbn.AddNode("Z")
lgbn.AddEdge("X", "Y")
lgbn.AddEdge("Y", "Z")

// Set linear Gaussian CPDs
// Y = 0.5 * X + N(0, 1)
lgbn.SetCPD("X", factors.NewLinearGaussianCPD(
    "X", 0.0, 1.0, nil, nil,  // mean=0, variance=1, no parents
))
lgbn.SetCPD("Y", factors.NewLinearGaussianCPD(
    "Y", 0.0, 1.0,
    []string{"X"}, []float64{0.5},  // beta=0.5 for parent X
))
lgbn.SetCPD("Z", factors.NewLinearGaussianCPD(
    "Z", 0.0, 1.0,
    []string{"Y"}, []float64{0.8},  // beta=0.8 for parent Y
))`}</code></pre>

        <h3>What's Next</h3>
        <ul>
          <li><ScrollLink to="tutorial-8">Tutorial 8</ScrollLink>: Using the internal libraries (numgo, scigo, graphgo, tabgo) directly</li>
          <li><Link to="/api">API Reference</Link>: Full type and method documentation</li>
        </ul>
      </section>

      {/* ============================================================ */}
      {/* TUTORIAL 8 */}
      {/* ============================================================ */}
      <section class="section" id="tutorial-8">
        <h2>Tutorial 8: Using the Internal Libraries</h2>
        <p>
          pgmgo's internal libraries -- numgo, scigo, graphgo, and tabgo -- are general-purpose
          and can be used independently of the PGM layer. This tutorial shows how to use each
          for common tasks.
        </p>

        <h3>Prerequisites</h3>
        <ul>
          <li>pgmgo installed: <code>go get github.com/asymmetric-effort/pgmgo</code></li>
        </ul>

        <h3>numgo: Matrix and Array Operations</h3>
        <p>
          numgo provides N-dimensional arrays, matrices, and vectors -- similar to numpy in Python.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "github.com/asymmetric-effort/pgmgo/lib/numgo"
)

func main() {
    // === Vectors ===
    v1 := numgo.NewVector([]float64{1, 2, 3, 4, 5})
    v2 := numgo.NewVector([]float64{5, 4, 3, 2, 1})

    dot := v1.Dot(v2)
    fmt.Println("Dot product:", dot)  // 35

    norm := v1.Norm()
    fmt.Println("L2 norm:", norm)     // sqrt(55) ~ 7.416

    // Element-wise operations
    sum := v1.Add(v2)
    fmt.Println("v1 + v2:", sum.Data())  // [6 6 6 6 6]

    // === Matrices ===
    m1 := numgo.NewMatrixFromData(2, 3, []float64{
        1, 2, 3,
        4, 5, 6,
    })
    m2 := numgo.NewMatrixFromData(3, 2, []float64{
        7, 8,
        9, 10,
        11, 12,
    })

    product := m1.Multiply(m2)
    fmt.Println("Matrix product shape:", product.Rows(), "x", product.Cols())
    fmt.Println("Matrix product:", product.Data())

    // Transpose
    transposed := m1.Transpose()
    fmt.Println("Transposed shape:", transposed.Rows(), "x", transposed.Cols())

    // Square matrix operations
    sq := numgo.NewMatrixFromData(3, 3, []float64{
        1, 2, 3,
        0, 1, 4,
        5, 6, 0,
    })
    det := sq.Det()
    fmt.Println("Determinant:", det)  // 1

    inv := sq.Inverse()
    fmt.Println("Inverse:", inv.Data())

    // === NDArrays ===
    arr := numgo.NewNDArray([]int{2, 3, 4})  // 2x3x4 array
    arr.Fill(1.0)
    total := arr.Sum()
    fmt.Println("Sum of 2x3x4 ones:", total)  // 24
}`}</code></pre>

        <h3>scigo: Statistical Distributions and Optimization</h3>
        <p>
          scigo provides statistical distributions, optimization, and hypothesis testing -- similar
          to scipy in Python.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "github.com/asymmetric-effort/pgmgo/lib/scigo"
)

func main() {
    // === Normal Distribution ===
    normal := scigo.NewNormal(0, 1)  // standard normal
    fmt.Println("PDF at 0:", normal.PDF(0))      // ~0.3989
    fmt.Println("CDF at 1.96:", normal.CDF(1.96)) // ~0.975
    fmt.Println("PPF at 0.975:", normal.PPF(0.975)) // ~1.96

    // Sample from the distribution
    sample := normal.Sample(1000)
    fmt.Println("Sample mean:", mean(sample))  // ~0

    // === Chi-Squared Distribution ===
    // Used in chi-squared independence tests
    chi2 := scigo.NewChiSquared(5)  // 5 degrees of freedom
    fmt.Println("Chi2 CDF at 11.07:", chi2.CDF(11.07))  // ~0.95
    pValue := 1.0 - chi2.CDF(11.07)
    fmt.Println("p-value:", pValue)

    // === Other Distributions ===
    beta := scigo.NewBeta(2, 5)
    fmt.Println("Beta mean:", beta.Mean())  // 2/7 ~ 0.286

    gamma := scigo.NewGamma(2, 1)
    fmt.Println("Gamma mean:", gamma.Mean())  // 2.0

    studentT := scigo.NewStudentT(10)  // 10 degrees of freedom
    fmt.Println("t CDF at 2.228:", studentT.CDF(2.228))

    // === Optimization ===
    // Minimize f(x) = (x-3)^2 on [0, 10]
    result := scigo.Minimize(func(x float64) float64 {
        return (x - 3) * (x - 3)
    }, 0.0, 10.0)
    fmt.Println("Minimum at x =", result)  // ~3.0
}`}</code></pre>

        <h3>graphgo: Graph Algorithms</h3>
        <p>
          graphgo provides directed and undirected graphs with a full suite of algorithms -- similar
          to networkx in Python.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

func main() {
    // === Directed Graph ===
    dg := graphgo.NewDiGraph()
    dg.AddNode("A")
    dg.AddNode("B")
    dg.AddNode("C")
    dg.AddNode("D")
    dg.AddEdge("A", "B")
    dg.AddEdge("A", "C")
    dg.AddEdge("B", "D")
    dg.AddEdge("C", "D")

    // Basic queries
    fmt.Println("Nodes:", dg.Nodes())
    fmt.Println("Successors of A:", dg.Successors("A"))      // [B, C]
    fmt.Println("Predecessors of D:", dg.Predecessors("D"))   // [B, C]

    // Topological sort
    sorted := dg.TopologicalSort()
    fmt.Println("Topological order:", sorted)  // [A, B, C, D] or [A, C, B, D]

    // Cycle detection
    fmt.Println("Has cycle:", dg.HasCycle())  // false

    // === Undirected Graph ===
    ug := graphgo.NewGraph()
    ug.AddEdge("X", "Y")
    ug.AddEdge("Y", "Z")
    ug.AddEdge("Z", "X")
    ug.AddEdge("W", "V")

    fmt.Println("Neighbors of Y:", ug.Neighbors("Y"))  // [X, Z]
    fmt.Println("Degree of Y:", ug.Degree("Y"))         // 2

    // Connected components
    components := ug.ConnectedComponents()
    fmt.Println("Components:", components)  // [[X Y Z], [W V]]

    // === PDAG (Partially Directed) ===
    pdag := graphgo.NewPDAG()
    pdag.AddDirectedEdge("A", "B")
    pdag.AddUndirectedEdge("B", "C")
    // PDAGs represent Markov equivalence classes

    // === Graph Algorithms ===
    // Moral graph (for converting BN to undirected)
    moral := dg.MoralGraph()
    fmt.Println("Moral graph edges:", moral.Edges())

    // D-separation (for BN independence queries)
    // separated := dg.DSeparation([]string{"A"}, []string{"D"}, []string{"B"})
}`}</code></pre>

        <h3>tabgo: Data Manipulation</h3>
        <p>
          tabgo provides DataFrames and Series for tabular data -- similar to pandas in Python.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func main() {
    // === Read CSV ===
    df, err := tabgo.ReadCSV("data.csv")
    if err != nil {
        panic(err)
    }
    fmt.Printf("DataFrame: %d rows, %d columns\\n", df.NRows(), len(df.Columns()))
    fmt.Println("Column names:", df.Columns())

    // === Access Columns ===
    col := df.Column("Age")
    fmt.Println("Unique values:", col.Unique())
    fmt.Println("Value counts:", col.ValueCounts())

    // === Filter Rows ===
    adults := df.Filter(func(row map[string]interface{}) bool {
        age, ok := row["Age"].(int)
        return ok && age >= 18
    })
    fmt.Printf("Adults: %d rows\\n", adults.NRows())

    // === GroupBy ===
    // Group by a column and compute aggregates
    // grouped := df.GroupBy("Category")

    // === Create DataFrame from scratch ===
    newDF := tabgo.NewDataFrame(
        map[string][]interface{}{
            "Name": {"Alice", "Bob", "Charlie"},
            "Score": {95, 87, 92},
        },
    )
    fmt.Println("New DataFrame columns:", newDF.Columns())

    // === Write CSV ===
    tabgo.WriteCSV(newDF, "output.csv")
    fmt.Println("Wrote output.csv")

    // === Other I/O ===
    // tabgo.ReadParquet("data.parquet")
    // tabgo.WriteParquet(df, "output.parquet")
    // tabgo.ReadXLSX("data.xlsx")
}`}</code></pre>

        <h3>Combining Libraries</h3>
        <p>
          The libraries work together naturally. Here is an example that loads data with tabgo,
          computes statistics with scigo, and uses numgo for matrix operations:
        </p>
        <pre><code>{`// Load data
df, _ := tabgo.ReadCSV("measurements.csv")

// Extract a column as a float slice
values := df.Column("Temperature").Float64()

// Compute statistics with scigo
mean := scigo.Mean(values)
std := scigo.Std(values)
fmt.Printf("Temperature: mean=%.2f, std=%.2f\\n", mean, std)

// Build a correlation matrix with numgo
cols := []string{"Temperature", "Humidity", "Pressure"}
n := len(cols)
corrMatrix := numgo.NewMatrix(n, n)
for i := 0; i < n; i++ {
    for j := 0; j < n; j++ {
        vi := df.Column(cols[i]).Float64()
        vj := df.Column(cols[j]).Float64()
        corrMatrix.Set(i, j, scigo.PearsonCorrelation(vi, vj))
    }
}
fmt.Println("Correlation matrix:", corrMatrix.Data())`}</code></pre>

        <h3>What's Next</h3>
        <ul>
          <li><Link to="/docs">Documentation</Link>: Full package reference for all types and methods</li>
          <li><Link to="/api">API Reference</Link>: Detailed API documentation with signatures and examples</li>
        </ul>
      </section>
    </div>
  );
}
