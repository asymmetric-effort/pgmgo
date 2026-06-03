import { createElement, useHead } from "@asymmetric-effort/specifyjs";

export function Api() {
  useHead({
    title: "API Reference — pgmgo",
    description: "Complete Go API reference for pgmgo probabilistic graphical models library.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/api",
  });

  return (
    <div class="page">
      <h1>API Reference</h1>

      <section class="section">
        <h2>Import Paths</h2>
        <pre><code>{`import (
    // Core packages
    "github.com/asymmetric-effort/pgmgo/src/models"
    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/src/learning"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
    "github.com/asymmetric-effort/pgmgo/src/ci_tests"
    "github.com/asymmetric-effort/pgmgo/src/structure_score"
    "github.com/asymmetric-effort/pgmgo/src/identification"
    "github.com/asymmetric-effort/pgmgo/src/prediction"
    "github.com/asymmetric-effort/pgmgo/src/metrics"
    "github.com/asymmetric-effort/pgmgo/src/independencies"
    "github.com/asymmetric-effort/pgmgo/src/base"
    "github.com/asymmetric-effort/pgmgo/src/config"
    "github.com/asymmetric-effort/pgmgo/src/utils"

    // Built-in example models
    "github.com/asymmetric-effort/pgmgo/example_models"

    // Primitive library modules
    "github.com/asymmetric-effort/pgmgo/lib/numgo"
    "github.com/asymmetric-effort/pgmgo/lib/scigo"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
    "github.com/asymmetric-effort/pgmgo/lib/gpu"
)`}</code></pre>
      </section>

      <section class="section">
        <h2>Core Packages</h2>

        <h3>models</h3>
        <p>All probabilistic graphical model types.</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BayesianNetwork</code></td><td>Directed acyclic graphical model with TabularCPDs</td></tr>
            <tr><td><code>MarkovNetwork</code></td><td>Undirected graphical model with factor potentials</td></tr>
            <tr><td><code>DynamicBayesianNetwork</code></td><td>Bayesian network over time slices</td></tr>
            <tr><td><code>NaiveBayes</code></td><td>Naive Bayes classifier (BN with single parent)</td></tr>
            <tr><td><code>SEM</code></td><td>Structural Equation Model</td></tr>
            <tr><td><code>FactorGraph</code></td><td>Bipartite graph of variables and factors</td></tr>
            <tr><td><code>JunctionTree</code></td><td>Clique tree for exact inference</td></tr>
            <tr><td><code>ClusterGraph</code></td><td>Generalized cluster graph</td></tr>
            <tr><td><code>LinearGaussianBN</code></td><td>Bayesian network with linear Gaussian CPDs</td></tr>
            <tr><td><code>FunctionalBN</code></td><td>Bayesian network with functional relationships</td></tr>
            <tr><td><code>MarkovChain</code></td><td>First-order Markov chain</td></tr>
            <tr><td><code>DiscreteBayesianNetwork</code></td><td>Specialized discrete-only BN</td></tr>
            <tr><td><code>DiscreteMarkovNetwork</code></td><td>Specialized discrete-only Markov network</td></tr>
          </tbody>
        </table>

        <h3>factors</h3>
        <p>Factor representations and operations.</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DiscreteFactor</code></td><td>General discrete factor with product, marginalize, reduce, normalize</td></tr>
            <tr><td><code>TabularCPD</code></td><td>Conditional probability distribution table</td></tr>
            <tr><td><code>JointProbabilityDistribution</code></td><td>Full joint distribution over a set of variables</td></tr>
            <tr><td><code>LinearGaussianCPD</code></td><td>Linear Gaussian conditional distribution</td></tr>
            <tr><td><code>NoisyOR</code></td><td>Noisy-OR parameterization for CPDs</td></tr>
          </tbody>
        </table>

        <h3>inference</h3>
        <p>Exact and approximate inference algorithms.</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>VariableElimination</code></td><td>Exact inference via factor elimination; supports Query and MAP</td></tr>
            <tr><td><code>BeliefPropagation</code></td><td>Message-passing on junction trees</td></tr>
            <tr><td><code>MPLP</code></td><td>Max-Product Linear Programming for MAP inference</td></tr>
            <tr><td><code>ApproxInference</code></td><td>Sampling-based approximate inference</td></tr>
            <tr><td><code>CausalInference</code></td><td>Do-calculus interventional queries</td></tr>
            <tr><td><code>DBNInference</code></td><td>Inference over dynamic Bayesian networks</td></tr>
          </tbody>
        </table>

        <h3>learning</h3>
        <p>Parameter estimation and structure learning algorithms.</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>MLE</code></td><td>Maximum Likelihood Estimation for CPD parameters</td></tr>
            <tr><td><code>BayesianEstimator</code></td><td>Bayesian parameter estimation with Dirichlet priors (BDeu)</td></tr>
            <tr><td><code>EM</code></td><td>Expectation-Maximization for incomplete data</td></tr>
            <tr><td><code>HillClimbSearch</code></td><td>Greedy hill-climbing structure learning</td></tr>
            <tr><td><code>PC</code></td><td>PC algorithm (constraint-based structure learning)</td></tr>
            <tr><td><code>GES</code></td><td>Greedy Equivalence Search</td></tr>
            <tr><td><code>ExhaustiveSearch</code></td><td>Exhaustive DAG enumeration (small networks)</td></tr>
            <tr><td><code>TreeSearch</code></td><td>Tree-structured network learning (Chow-Liu)</td></tr>
            <tr><td><code>MMHC</code></td><td>Max-Min Hill-Climbing (hybrid method)</td></tr>
            <tr><td><code>ExpertInLoop</code></td><td>Interactive structure learning with expert/LLM guidance</td></tr>
            <tr><td><code>IVEstimator</code></td><td>Instrumental variable estimation</td></tr>
            <tr><td><code>SEMEstimator</code></td><td>Structural equation model estimation</td></tr>
            <tr><td><code>MirrorDescent</code></td><td>Mirror descent optimization for parameter learning</td></tr>
            <tr><td><code>LinearGaussianMLE</code></td><td>MLE for linear Gaussian BNs</td></tr>
            <tr><td><code>MarginalEstimator</code></td><td>Marginal likelihood estimation</td></tr>
          </tbody>
        </table>

        <h3>sampling</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BayesianModelSampling</code></td><td>Forward sampling, rejection sampling, likelihood-weighted sampling</td></tr>
            <tr><td><code>GibbsSampling</code></td><td>MCMC Gibbs sampler with burn-in and thinning</td></tr>
          </tbody>
        </table>

        <h3>ci_tests</h3>
        <p>Conditional independence tests used by constraint-based structure learning.</p>
        <table>
          <thead>
            <tr><th>Category</th><th>Tests</th></tr>
          </thead>
          <tbody>
            <tr><td>Discrete</td><td><code>ChiSquare</code>, <code>GSq</code>, <code>ModifiedChiSquare</code>, <code>PowerDivergence</code></td></tr>
            <tr><td>Continuous</td><td><code>FisherZ</code>, <code>Pearsonr</code>, <code>PartialCorrelation</code></td></tr>
            <tr><td>Multivariate</td><td><code>GCM</code>, <code>HotellingLawley</code>, <code>PillaiBartlett</code></td></tr>
            <tr><td>Tree-based</td><td><code>TreeBasedCI</code></td></tr>
          </tbody>
        </table>

        <h3>structure_score</h3>
        <p>Scoring functions for score-based structure learning.</p>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BIC</code></td><td>Bayesian Information Criterion</td></tr>
            <tr><td><code>AIC</code></td><td>Akaike Information Criterion</td></tr>
            <tr><td><code>BDeu</code></td><td>Bayesian Dirichlet equivalent uniform</td></tr>
            <tr><td><code>BDs</code></td><td>Bayesian Dirichlet sparse</td></tr>
            <tr><td><code>K2</code></td><td>K2 score</td></tr>
            <tr><td><code>LogLikelihood</code></td><td>Log-likelihood score</td></tr>
            <tr><td><code>Gaussian</code></td><td>Gaussian score for continuous data</td></tr>
            <tr><td><code>ConditionalGaussian</code></td><td>Conditional Gaussian score for mixed data</td></tr>
          </tbody>
        </table>

        <h3>readwrite</h3>
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

        <h3>identification</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Adjustment</code></td><td>Back-door criterion adjustment set identification</td></tr>
            <tr><td><code>Frontdoor</code></td><td>Front-door criterion identification</td></tr>
          </tbody>
        </table>

        <h3>prediction</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DoubleMLRegressor</code></td><td>Double/Debiased Machine Learning for causal effect estimation</td></tr>
            <tr><td><code>NaiveAdjustmentRegressor</code></td><td>Naive back-door adjustment regression</td></tr>
            <tr><td><code>NaiveIVRegressor</code></td><td>Instrumental variable regression</td></tr>
          </tbody>
        </table>

        <h3>metrics</h3>
        <table>
          <thead>
            <tr><th>Function</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>SHD</code></td><td>Structural Hamming Distance between two DAGs</td></tr>
            <tr><td><code>AdjacencyConfusionMatrix</code></td><td>TP, FP, TN, FN for edge presence</td></tr>
            <tr><td><code>OrientationConfusionMatrix</code></td><td>TP, FP, TN, FN for edge orientation</td></tr>
            <tr><td><code>CorrelationScore</code></td><td>Correlation-based scoring</td></tr>
            <tr><td><code>FisherC</code></td><td>Fisher's C statistic for model fit</td></tr>
          </tbody>
        </table>
      </section>

      <section class="section">
        <h2>Primitive Library Modules</h2>

        <h3>numgo (numpy equivalent)</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>NDArray</code></td><td>N-dimensional array with shape, stride, and element-wise operations</td></tr>
            <tr><td><code>Matrix</code></td><td>2D matrix with multiply, transpose, inverse, determinant</td></tr>
            <tr><td><code>Vector</code></td><td>1D vector with dot product, norm, element-wise ops</td></tr>
          </tbody>
        </table>

        <h3>scigo (scipy equivalent)</h3>
        <table>
          <thead>
            <tr><th>Category</th><th>Key Types</th></tr>
          </thead>
          <tbody>
            <tr><td>Distributions</td><td>Normal, ChiSquared, Beta, Gamma, StudentT, Uniform, Exponential</td></tr>
            <tr><td>Optimization</td><td>Minimize, GradientDescent, NewtonMethod</td></tr>
            <tr><td>Statistics</td><td>Hypothesis tests, p-value computation, quantile functions</td></tr>
          </tbody>
        </table>

        <h3>graphgo (networkx equivalent)</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DiGraph</code></td><td>Directed graph with adjacency operations</td></tr>
            <tr><td><code>Graph</code></td><td>Undirected graph</td></tr>
            <tr><td><code>PDAG</code></td><td>Partially directed acyclic graph</td></tr>
          </tbody>
        </table>
        <p>Algorithms: topological sort, d-separation, moral graph, triangulation, maximum cardinality search, clique finding, connected components.</p>

        <h3>tabgo (pandas equivalent)</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DataFrame</code></td><td>Tabular data with named columns, row filtering, groupby</td></tr>
            <tr><td><code>Series</code></td><td>Single column with value counts, unique, statistical summaries</td></tr>
          </tbody>
        </table>
        <p>I/O: <code>ReadCSV</code>, <code>WriteCSV</code>, <code>ReadParquet</code>, <code>WriteParquet</code>, <code>ReadXLSX</code>.</p>
      </section>

      <section class="section">
        <h2>Example: Creating a BayesianNetwork</h2>
        <pre><code>{`bn := models.NewBayesianNetwork()

// Add nodes and edges
bn.AddNode("A")
bn.AddNode("B")
bn.AddNode("C")
bn.AddEdge("A", "B")
bn.AddEdge("B", "C")

// Define states for each node
bn.SetStates("A", []string{"a0", "a1"})
bn.SetStates("B", []string{"b0", "b1"})
bn.SetStates("C", []string{"c0", "c1"})

// Check the structure is a valid DAG
err := bn.CheckModel()
fmt.Println("Nodes:", bn.Nodes())
fmt.Println("Edges:", bn.Edges())`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Adding CPDs</h2>
        <pre><code>{`// Root node CPD: P(A)
bn.SetCPD("A", factors.NewTabularCPD(
    "A", 2,
    []float64{0.4, 0.6},   // P(a0)=0.4, P(a1)=0.6
    nil, nil,               // no parents
))

// Child node CPD: P(B | A)
// Values in column-major order: P(b0|a0), P(b1|a0), P(b0|a1), P(b1|a1)
bn.SetCPD("B", factors.NewTabularCPD(
    "B", 2,
    []float64{0.2, 0.8, 0.9, 0.1},
    []string{"A"}, []int{2},
))

// P(C | B)
bn.SetCPD("C", factors.NewTabularCPD(
    "C", 2,
    []float64{0.3, 0.7, 0.6, 0.4},
    []string{"B"}, []int{2},
))

// Validate model (checks CPD dimensions match graph structure)
if err := bn.CheckModel(); err != nil {
    log.Fatal(err)
}`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Variable Elimination Query</h2>
        <pre><code>{`// Convert BN CPDs to Markov factors
facs, err := bn.ToMarkovFactors()
if err != nil {
    log.Fatal(err)
}

// Create Variable Elimination engine
ve := inference.NewVariableElimination(facs)

// Posterior query: P(C | A=1)
result, err := ve.Query(
    []string{"C"},               // query variables
    map[string]int{"A": 1},      // evidence
)
if err != nil {
    log.Fatal(err)
}
fmt.Println("P(C | A=1):", result.Values().Data())

// MAP query: most likely assignment for B given A=0
assignment, err := ve.MAP(
    []string{"B"},
    map[string]int{"A": 0},
)
fmt.Println("MAP(B | A=0):", assignment)`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Belief Propagation</h2>
        <pre><code>{`// Build junction tree from Bayesian network
jt, err := bn.ToJunctionTree()
if err != nil {
    log.Fatal(err)
}

// Set up belief propagation
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

// Calibrate (run message passing)
if err := bp.Calibrate(); err != nil {
    log.Fatal(err)
}

// Query
result, err := bp.Query([]string{"C"}, map[string]int{"A": 1})`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Structure Learning from Data</h2>
        <pre><code>{`// Load data
data, err := tabgo.ReadCSV("observations.csv")
if err != nil {
    log.Fatal(err)
}

// Score-based: Hill-Climb with BIC
scorer := structure_score.NewBIC()
hc := learning.NewHillClimbSearch(data, scorer.LocalScore)
bn, err := hc.Estimate()
if err != nil {
    log.Fatal(err)
}
fmt.Println("Hill-climb:", len(bn.Nodes()), "nodes,", len(bn.Edges()), "edges")

// Constraint-based: PC algorithm with chi-square test
ciTest := func(x, y string, z []string, data *tabgo.DataFrame, sig float64) (float64, float64, bool) {
    return ci_tests.ChiSquare(x, y, z, data, sig)
}
pc := learning.NewPC(data, ciTest, 0.05)
pcBN, err := pc.EstimateBN()

// GES (Greedy Equivalence Search)
ges := learning.NewGES(data, scorer.LocalScore)
pdag, err := ges.Estimate()

// Tree search (Chow-Liu)
ts := learning.NewTreeSearch(data)
treeBN, err := ts.Estimate()`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Parameter Learning</h2>
        <pre><code>{`// Maximum Likelihood Estimation
mle := learning.NewMLE(bn, data)
err := mle.Estimate() // populates CPDs on bn

// Bayesian Estimation with BDeu prior
be := learning.NewBayesianEstimator(bn, data, learning.BDeu, 5.0)
err = be.Estimate()

// EM for incomplete data
em := learning.NewEM(bn, data, nil, 100, 1e-6)
err = em.Estimate()`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Causal Inference with do-calculus</h2>
        <pre><code>{`// Create a causal inference engine from a Bayesian network
ci, err := inference.NewCausalInference(bn)
if err != nil {
    log.Fatal(err)
}

// Interventional query: P(Y | do(X=1))
result, err := ci.Query(
    []string{"Y"},             // query variable
    map[string]int{"X": 1},    // do() intervention
    nil,                       // no additional evidence
)
fmt.Println("P(Y | do(X=1)):", result.Values().Data())

// With additional observational evidence: P(Y | do(X=1), Z=0)
result, err = ci.Query(
    []string{"Y"},
    map[string]int{"X": 1},
    map[string]int{"Z": 0},
)`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Sampling</h2>
        <pre><code>{`// Forward sampling
bms, err := sampling.NewBayesianModelSampling(bn, 42) // seed=42
if err != nil {
    log.Fatal(err)
}
samples, err := bms.ForwardSample(1000)
// samples is a *tabgo.DataFrame with 1000 rows

// Rejection sampling with evidence
evidenceSamples, err := bms.RejectionSample(500, map[string]int{"A": 1})

// Gibbs sampling with burn-in and thinning
gs, err := sampling.NewGibbsSampling(bn, 42)
if err != nil {
    log.Fatal(err)
}
gibbsSamples, err := gs.Sample(
    1000,                        // number of samples
    100,                         // burn-in
    1,                           // thinning interval
    map[string]int{"A": 1},     // evidence
)`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Reading and Writing BIF Files</h2>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/readwrite"

// Read a BIF file
f, _ := os.Open("model.bif")
defer f.Close()
bn, err := readwrite.ReadBIF(f)

// Write a BIF file
out, _ := os.Create("output.bif")
defer out.Close()
err = readwrite.WriteBIF(out, bn)

// Convert between formats
f2, _ := os.Open("model.net")
bn2, _ := readwrite.ReadNET(f2)
f2.Close()

out2, _ := os.Create("model.xmlbif")
readwrite.WriteXMLBIF(out2, bn2)
out2.Close()

// JSON serialization
jsonOut, _ := os.Create("model.json")
readwrite.WriteJSONModel(jsonOut, bn)
jsonOut.Close()`}</code></pre>
      </section>

      <section class="section">
        <h2>Example: Model Comparison</h2>
        <pre><code>{`import (
    "github.com/asymmetric-effort/pgmgo/src/metrics"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// Convert BNs to DiGraphs for comparison
trueGraph := graphgo.NewDiGraph()
for _, n := range trueBN.Nodes() { trueGraph.AddNode(n) }
for _, e := range trueBN.Edges() { trueGraph.AddEdge(e[0], e[1]) }

estGraph := graphgo.NewDiGraph()
for _, n := range estBN.Nodes() { estGraph.AddNode(n) }
for _, e := range estBN.Edges() { estGraph.AddEdge(e[0], e[1]) }

// Structural Hamming Distance
shd := metrics.SHD(trueGraph, estGraph)
fmt.Println("SHD:", shd)

// Confusion matrices
aTP, aFP, aTN, aFN := metrics.AdjacencyConfusionMatrix(trueGraph, estGraph)
oTP, oFP, oTN, oFN := metrics.OrientationConfusionMatrix(trueGraph, estGraph)`}</code></pre>
      </section>
    </div>
  );
}
