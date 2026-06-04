import { createElement, useHead, Link } from "@asymmetric-effort/specifyjs";

export function Api() {
  useHead({
    title: "API Reference — pgmgo",
    description: "Complete Go API reference for pgmgo: models, factors, inference, sampling, learning, CI tests, scoring, identification, prediction, metrics, readwrite, base, and library packages.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/api",
  });

  return (
    <div class="page">
      <h1>API Reference</h1>

      <nav class="page-toc">
        <strong>Packages:</strong>{" "}
        <a href="#import-paths">Import Paths</a> | <a href="#api-models">Models</a> | <a href="#api-factors">Factors</a> | <a href="#api-inference">Inference</a> | <a href="#api-sampling">Sampling</a> | <a href="#api-learning">Learning</a> | <a href="#api-ci-tests">CI Tests</a> | <a href="#api-structure-score">Structure Scores</a> | <a href="#api-identification">Identification</a> | <a href="#api-prediction">Prediction</a> | <a href="#api-metrics">Metrics</a> | <a href="#api-readwrite">Readwrite</a> | <a href="#api-base">Base</a> | <a href="#api-independencies">Independencies</a> | <a href="#api-config">Config</a> | <a href="#api-utils">Utils</a> | <a href="#api-numgo">numgo</a> | <a href="#api-scigo">scigo</a> | <a href="#api-graphgo">graphgo</a> | <a href="#api-tabgo">tabgo</a> | <a href="#api-gpu">gpu</a> | <a href="#api-example-models">example_models</a>
      </nav>

      {/* ============================================================ */}
      {/* IMPORT PATHS */}
      {/* ============================================================ */}
      <section class="section" id="import-paths">
        <h2>Import Paths</h2>
        <pre><code>{`import (
    // Core packages (src/)
    "github.com/asymmetric-effort/pgmgo/src/base"
    "github.com/asymmetric-effort/pgmgo/src/models"
    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/src/learning"
    "github.com/asymmetric-effort/pgmgo/src/ci_tests"
    "github.com/asymmetric-effort/pgmgo/src/structure_score"
    "github.com/asymmetric-effort/pgmgo/src/identification"
    "github.com/asymmetric-effort/pgmgo/src/prediction"
    "github.com/asymmetric-effort/pgmgo/src/metrics"
    "github.com/asymmetric-effort/pgmgo/src/independencies"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
    "github.com/asymmetric-effort/pgmgo/src/config"
    "github.com/asymmetric-effort/pgmgo/src/utils"

    // Library packages (lib/)
    "github.com/asymmetric-effort/pgmgo/lib/numgo"
    "github.com/asymmetric-effort/pgmgo/lib/scigo"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
    "github.com/asymmetric-effort/pgmgo/lib/gpu"

    // Built-in example models and datasets
    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/examples/datasets"
)`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* MODELS */}
      {/* ============================================================ */}
      <section class="section" id="api-models">
        <h2>models -- Probabilistic Model Types</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/models</code></p>
        <p>Provides 13 probabilistic graphical model types. All model types share a common interface for node/edge management, state definition, and CPD assignment.</p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BayesianNetwork</code></td><td>Directed acyclic graphical model with TabularCPDs. The primary model type.</td></tr>
            <tr><td><code>MarkovNetwork</code></td><td>Undirected graphical model with factor potentials.</td></tr>
            <tr><td><code>DynamicBayesianNetwork</code></td><td>BN over time slices for temporal modeling.</td></tr>
            <tr><td><code>NaiveBayes</code></td><td>Naive Bayes classifier (BN with single class parent).</td></tr>
            <tr><td><code>SEM</code></td><td>Structural Equation Model with linear/nonlinear equations.</td></tr>
            <tr><td><code>FactorGraph</code></td><td>Bipartite graph of variable nodes and factor nodes.</td></tr>
            <tr><td><code>JunctionTree</code></td><td>Clique tree for exact inference via message passing.</td></tr>
            <tr><td><code>ClusterGraph</code></td><td>Generalized cluster graph (superset of junction tree).</td></tr>
            <tr><td><code>LinearGaussianBN</code></td><td>BN with continuous linearly-related Gaussian variables.</td></tr>
            <tr><td><code>FunctionalBN</code></td><td>BN where CPDs are defined by arbitrary Go functions.</td></tr>
            <tr><td><code>MarkovChain</code></td><td>First-order Markov chain for sequential data.</td></tr>
            <tr><td><code>DiscreteBayesianNetwork</code></td><td>Specialized discrete-only BN with optimized operations.</td></tr>
            <tr><td><code>DiscreteMarkovNetwork</code></td><td>Specialized discrete-only Markov network.</td></tr>
          </tbody>
        </table>

        <h3>Constructors</h3>
        <pre><code>{`bn := models.NewBayesianNetwork()
mn := models.NewMarkovNetwork()
dbn := models.NewDynamicBayesianNetwork()
nb := models.NewNaiveBayes()
sem := models.NewSEM()
fg := models.NewFactorGraph()
jt := models.NewJunctionTree()
cg := models.NewClusterGraph()
lgbn := models.NewLinearGaussianBN()
fbn := models.NewFunctionalBN()
mc := models.NewMarkovChain()
dbn2 := models.NewDiscreteBayesianNetwork()
dmn := models.NewDiscreteMarkovNetwork()`}</code></pre>

        <h3>BayesianNetwork Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>AddNode</code></td><td><code>(name string)</code></td><td>Add a node to the network</td></tr>
            <tr><td><code>AddEdge</code></td><td><code>(from, to string)</code></td><td>Add a directed edge</td></tr>
            <tr><td><code>RemoveNode</code></td><td><code>(name string)</code></td><td>Remove a node and its edges</td></tr>
            <tr><td><code>RemoveEdge</code></td><td><code>(from, to string)</code></td><td>Remove a directed edge</td></tr>
            <tr><td><code>Nodes</code></td><td><code>() []string</code></td><td>List all node names</td></tr>
            <tr><td><code>Edges</code></td><td><code>() [][2]string</code></td><td>List all edges as [from, to] pairs</td></tr>
            <tr><td><code>Parents</code></td><td><code>(name string) []string</code></td><td>Get parents of a node</td></tr>
            <tr><td><code>Children</code></td><td><code>(name string) []string</code></td><td>Get children of a node</td></tr>
            <tr><td><code>SetStates</code></td><td><code>(name string, states []string)</code></td><td>Define possible values for a node</td></tr>
            <tr><td><code>GetStates</code></td><td><code>(name string) []string</code></td><td>Get states of a node</td></tr>
            <tr><td><code>SetCPD</code></td><td><code>(name string, cpd *factors.TabularCPD)</code></td><td>Assign a CPD to a node</td></tr>
            <tr><td><code>GetCPD</code></td><td><code>(name string) *factors.TabularCPD</code></td><td>Get the CPD of a node</td></tr>
            <tr><td><code>CheckModel</code></td><td><code>() error</code></td><td>Validate DAG, CPDs, dimensions, probability sums</td></tr>
            <tr><td><code>ToMarkovFactors</code></td><td><code>() ([]*factors.DiscreteFactor, error)</code></td><td>Convert CPDs to factors for inference</td></tr>
            <tr><td><code>ToJunctionTree</code></td><td><code>() (*JunctionTree, error)</code></td><td>Build junction tree for BP inference</td></tr>
            <tr><td><code>DSeparation</code></td><td><code>(x, y, z []string) bool</code></td><td>Test d-separation: X _||_ Y | Z</td></tr>
            <tr><td><code>GetIndependencies</code></td><td><code>() *independencies.Independencies</code></td><td>List all conditional independencies</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`bn := models.NewBayesianNetwork()
bn.AddNode("A")
bn.AddNode("B")
bn.AddNode("C")
bn.AddEdge("A", "B")
bn.AddEdge("B", "C")

bn.SetStates("A", []string{"a0", "a1"})
bn.SetStates("B", []string{"b0", "b1"})
bn.SetStates("C", []string{"c0", "c1"})

bn.SetCPD("A", factors.NewTabularCPD("A", 2, []float64{0.4, 0.6}, nil, nil))
bn.SetCPD("B", factors.NewTabularCPD("B", 2, []float64{0.2, 0.8, 0.9, 0.1}, []string{"A"}, []int{2}))
bn.SetCPD("C", factors.NewTabularCPD("C", 2, []float64{0.3, 0.7, 0.6, 0.4}, []string{"B"}, []int{2}))

err := bn.CheckModel()
fmt.Println("Valid:", err == nil)`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* FACTORS */}
      {/* ============================================================ */}
      <section class="section" id="api-factors">
        <h2>factors -- Factor Representations</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/factors</code></p>
        <p>Factor types and operations for probabilistic models. Factors are the building blocks of inference.</p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DiscreteFactor</code></td><td>General discrete factor over a set of variables. Supports product, marginalize, reduce, normalize.</td></tr>
            <tr><td><code>TabularCPD</code></td><td>Conditional probability distribution table. CPD values are stored in column-major order.</td></tr>
            <tr><td><code>JointProbabilityDistribution</code></td><td>Full joint distribution over a set of variables.</td></tr>
            <tr><td><code>LinearGaussianCPD</code></td><td>Linear Gaussian conditional: child = sum(beta_i * parent_i) + N(mean, variance).</td></tr>
            <tr><td><code>NoisyOR</code></td><td>Noisy-OR parameterization for compact CPD representation with many binary parents.</td></tr>
          </tbody>
        </table>

        <h3>Constructors</h3>
        <pre><code>{`// DiscreteFactor: general factor over named variables
f := factors.NewDiscreteFactor(
    []string{"A", "B"},     // variable names
    []int{2, 3},            // cardinalities
    []float64{...},         // values (product of cardinalities)
)

// TabularCPD: conditional probability distribution
cpd := factors.NewTabularCPD(
    "B",                    // variable name
    2,                      // number of states
    []float64{0.3, 0.7, 0.8, 0.2},  // values in column-major order
    []string{"A"},          // parent names (nil for root nodes)
    []int{2},               // parent cardinalities (nil for root nodes)
)

// JointProbabilityDistribution
jpd := factors.NewJPD(
    []string{"A", "B"},
    []int{2, 2},
    []float64{0.1, 0.2, 0.3, 0.4},
)

// LinearGaussianCPD
lgcpd := factors.NewLinearGaussianCPD(
    "Y",                    // variable name
    0.0,                    // mean of noise term
    1.0,                    // variance of noise term
    []string{"X"},          // parent names
    []float64{0.5},         // regression coefficients (betas)
)

// NoisyOR
nor := factors.NewNoisyOR(
    "Effect",
    []string{"Cause1", "Cause2"},
    []float64{0.1, 0.3},   // inhibition probabilities
    0.01,                   // leak probability
)`}</code></pre>

        <h3>DiscreteFactor Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Product</code></td><td><code>(other *DiscreteFactor) *DiscreteFactor</code></td><td>Factor product (join)</td></tr>
            <tr><td><code>Marginalize</code></td><td><code>(vars []string) *DiscreteFactor</code></td><td>Sum out variables</td></tr>
            <tr><td><code>Reduce</code></td><td><code>(evidence map[string]int) *DiscreteFactor</code></td><td>Fix variables to observed values</td></tr>
            <tr><td><code>Normalize</code></td><td><code>() *DiscreteFactor</code></td><td>Normalize values to sum to 1</td></tr>
            <tr><td><code>Variables</code></td><td><code>() []string</code></td><td>Get variable names</td></tr>
            <tr><td><code>Cardinality</code></td><td><code>() []int</code></td><td>Get cardinalities</td></tr>
            <tr><td><code>Values</code></td><td><code>() *numgo.NDArray</code></td><td>Get values as NDArray</td></tr>
          </tbody>
        </table>

        <h3>Example: Factor Operations</h3>
        <pre><code>{`// Create two factors
f1 := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2}, []float64{0.1, 0.2, 0.3, 0.4})
f2 := factors.NewDiscreteFactor([]string{"B", "C"}, []int{2, 2}, []float64{0.5, 0.6, 0.7, 0.8})

// Factor product: f1 * f2 -> factor over {A, B, C}
product := f1.Product(f2)
fmt.Println("Product variables:", product.Variables())  // [A, B, C]

// Marginalize out B: sum_B (f1 * f2) -> factor over {A, C}
marginal := product.Marginalize([]string{"B"})
fmt.Println("Marginalized variables:", marginal.Variables())  // [A, C]

// Reduce: fix A=0 -> factor over {B, C}
reduced := product.Reduce(map[string]int{"A": 0})
fmt.Println("Reduced values:", reduced.Values().Data())

// Normalize
normalized := marginal.Normalize()
fmt.Println("Sum after normalize:", normalized.Values().Sum())  // 1.0`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* INFERENCE */}
      {/* ============================================================ */}
      <section class="section" id="api-inference">
        <h2>inference -- Inference Algorithms</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/inference</code></p>
        <p>7 inference algorithms for computing posterior probabilities and MAP assignments.</p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>VariableElimination</code></td><td>Exact inference via factor elimination. Default method.</td></tr>
            <tr><td><code>BeliefPropagation</code></td><td>Exact inference via message passing on junction trees.</td></tr>
            <tr><td><code>MPLP</code></td><td>Max-Product Linear Programming for MAP inference.</td></tr>
            <tr><td><code>ApproxInference</code></td><td>Sampling-based approximate inference.</td></tr>
            <tr><td><code>CausalInference</code></td><td>Do-calculus interventional queries.</td></tr>
            <tr><td><code>DBNInference</code></td><td>Inference over dynamic Bayesian networks.</td></tr>
          </tbody>
        </table>

        <h3>Constructors</h3>
        <pre><code>{`// Variable Elimination (from Markov factors)
facs, _ := bn.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)

// Belief Propagation (from junction tree)
jt, _ := bn.ToJunctionTree()
cliques := jt.Cliques()
separators := jt.SeparatorSets()
cliqueFactors := make(map[int][]*factors.DiscreteFactor)
for i, c := range cliques {
    fs := jt.GetCliqueFactors(c)
    if len(fs) > 0 { cliqueFactors[i] = fs }
}
bp := inference.NewBeliefPropagation(cliques, separators, cliqueFactors)

// MPLP
mplp := inference.NewMPLP(facs)

// Approximate Inference
approx, _ := inference.NewApproxInference(bn, 10000)

// Causal Inference
ci, _ := inference.NewCausalInference(bn)

// DBN Inference
dbnInf, _ := inference.NewDBNInference(dbn)`}</code></pre>

        <h3>VariableElimination Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Query</code></td><td><code>(variables []string, evidence map[string]int) (*factors.DiscreteFactor, error)</code></td><td>Compute posterior P(variables | evidence)</td></tr>
            <tr><td><code>MAP</code></td><td><code>(variables []string, evidence map[string]int) (map[string]int, error)</code></td><td>Find most likely assignment</td></tr>
          </tbody>
        </table>

        <h3>CausalInference Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Query</code></td><td><code>(variables []string, doEvidence map[string]int, obsEvidence map[string]int) (*factors.DiscreteFactor, error)</code></td><td>Compute P(variables | do(interventions), observations)</td></tr>
          </tbody>
        </table>

        <h3>Example: VE Query and MAP</h3>
        <pre><code>{`facs, _ := bn.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)

// Posterior query
result, _ := ve.Query([]string{"C"}, map[string]int{"A": 1})
fmt.Println("P(C | A=1):", result.Values().Data())

// MAP query
assignment, _ := ve.MAP([]string{"B"}, map[string]int{"A": 0})
fmt.Println("MAP(B | A=0):", assignment)`}</code></pre>

        <h3>Example: Causal Inference</h3>
        <pre><code>{`ci, _ := inference.NewCausalInference(bn)

// P(Y | do(X=1))
result, _ := ci.Query([]string{"Y"}, map[string]int{"X": 1}, nil)
fmt.Println("P(Y | do(X=1)):", result.Values().Data())

// P(Y | do(X=1), Z=0)
result, _ = ci.Query([]string{"Y"}, map[string]int{"X": 1}, map[string]int{"Z": 0})`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* SAMPLING */}
      {/* ============================================================ */}
      <section class="section" id="api-sampling">
        <h2>sampling -- Sampling Methods</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/sampling</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BayesianModelSampling</code></td><td>Forward, rejection, and likelihood-weighted sampling from a BN.</td></tr>
            <tr><td><code>GibbsSampling</code></td><td>MCMC Gibbs sampler with burn-in and thinning.</td></tr>
          </tbody>
        </table>

        <h3>Constructors</h3>
        <pre><code>{`bms, err := sampling.NewBayesianModelSampling(bn, 42)  // seed=42
gs, err := sampling.NewGibbsSampling(bn, 42)`}</code></pre>

        <h3>BayesianModelSampling Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>ForwardSample</code></td><td><code>(n int) (*tabgo.DataFrame, error)</code></td><td>Generate n forward samples</td></tr>
            <tr><td><code>RejectionSample</code></td><td><code>(n int, evidence map[string]int) (*tabgo.DataFrame, error)</code></td><td>Generate n samples consistent with evidence</td></tr>
            <tr><td><code>LikelihoodWeightedSample</code></td><td><code>(n int, evidence map[string]int) (*tabgo.DataFrame, []float64, error)</code></td><td>Generate n weighted samples</td></tr>
          </tbody>
        </table>

        <h3>GibbsSampling Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Sample</code></td><td><code>(nSamples, burnIn, thinning int, evidence map[string]int) (*tabgo.DataFrame, error)</code></td><td>Run Gibbs sampler</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`bms, _ := sampling.NewBayesianModelSampling(bn, 42)

// Forward sampling
samples, _ := bms.ForwardSample(1000)
fmt.Printf("Forward: %d samples\\n", samples.NRows())

// Rejection sampling with evidence
rejSamples, _ := bms.RejectionSample(500, map[string]int{"A": 1})

// Gibbs sampling
gs, _ := sampling.NewGibbsSampling(bn, 42)
gibbsSamples, _ := gs.Sample(1000, 100, 2, map[string]int{"A": 1})`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* LEARNING */}
      {/* ============================================================ */}
      <section class="section" id="api-learning">
        <h2>learning -- Learning Algorithms</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/learning</code></p>
        <p>15+ algorithms for parameter estimation and structure learning.</p>

        <h3>Parameter Learning</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>MLE</code></td><td><code>NewMLE(bn, data)</code></td><td>Maximum Likelihood Estimation from complete data</td></tr>
            <tr><td><code>BayesianEstimator</code></td><td><code>NewBayesianEstimator(bn, data, prior, ess)</code></td><td>Bayesian estimation with Dirichlet prior (BDeu)</td></tr>
            <tr><td><code>EM</code></td><td><code>NewEM(bn, data, latent, maxIter, tol)</code></td><td>Expectation-Maximization for incomplete data</td></tr>
            <tr><td><code>LinearGaussianMLE</code></td><td><code>NewLinearGaussianMLE(lgbn, data)</code></td><td>MLE for linear Gaussian BNs</td></tr>
            <tr><td><code>MarginalEstimator</code></td><td><code>NewMarginalEstimator(bn, data)</code></td><td>Marginal likelihood estimation</td></tr>
            <tr><td><code>MirrorDescent</code></td><td><code>NewMirrorDescent(bn, data, lr, maxIter)</code></td><td>Mirror descent optimization</td></tr>
          </tbody>
        </table>

        <h3>Structure Learning</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Category</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>HillClimbSearch</code></td><td><code>NewHillClimbSearch(data, scoreFn)</code></td><td>Score-based</td><td>Greedy hill-climbing with add/remove/reverse edge operations</td></tr>
            <tr><td><code>PC</code></td><td><code>NewPC(data, ciTest, significance)</code></td><td>Constraint-based</td><td>PC algorithm using CI tests</td></tr>
            <tr><td><code>GES</code></td><td><code>NewGES(data, scoreFn)</code></td><td>Score-based</td><td>Greedy Equivalence Search over DAG equivalence classes</td></tr>
            <tr><td><code>ExhaustiveSearch</code></td><td><code>NewExhaustiveSearch(data, scoreFn)</code></td><td>Score-based</td><td>Exhaustive enumeration (small networks only)</td></tr>
            <tr><td><code>TreeSearch</code></td><td><code>NewTreeSearch(data)</code></td><td>Score-based</td><td>Chow-Liu tree-structured network learning</td></tr>
            <tr><td><code>MMHC</code></td><td><code>NewMMHC(data, ciTest, scoreFn, significance)</code></td><td>Hybrid</td><td>Max-Min Hill-Climbing</td></tr>
            <tr><td><code>ExpertInLoop</code></td><td><code>NewExpertInLoop(data, scoreFn, client)</code></td><td>Interactive</td><td>Expert/LLM-guided structure learning</td></tr>
          </tbody>
        </table>

        <h3>Causal Estimation</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>IVEstimator</code></td><td><code>NewIVEstimator(data, instrument, treatment, outcome)</code></td><td>Instrumental variable estimation</td></tr>
            <tr><td><code>SEMEstimator</code></td><td><code>NewSEMEstimator(sem, data)</code></td><td>SEM parameter estimation</td></tr>
            <tr><td><code>LLMClient</code></td><td><code>NewLLMClient(endpoint, apiKey)</code></td><td>LLM client for expert-in-the-loop</td></tr>
          </tbody>
        </table>

        <h3>Common Methods</h3>
        <p>All parameter learners have <code>Estimate() error</code> which populates CPDs on the model.</p>
        <p>All structure learners have <code>Estimate() (*models.BayesianNetwork, error)</code> which returns a learned structure.</p>
        <p>PC also provides <code>EstimateBN() (*models.BayesianNetwork, error)</code> and <code>EstimatePDAG() (*graphgo.PDAG, error)</code>.</p>

        <h3>Example: Complete Learning Pipeline</h3>
        <pre><code>{`// Load data
data, _ := tabgo.ReadCSV("observations.csv")

// Learn structure
scorer := structure_score.NewBIC()
hc := learning.NewHillClimbSearch(data, scorer.LocalScore)
bn, _ := hc.Estimate()

// Fit parameters
mle := learning.NewMLE(bn, data)
mle.Estimate()

// Use the learned model
facs, _ := bn.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)
result, _ := ve.Query([]string{"Y"}, map[string]int{"X": 1})`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* CI TESTS */}
      {/* ============================================================ */}
      <section class="section" id="api-ci-tests">
        <h2>ci_tests -- Conditional Independence Tests</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/ci_tests</code></p>
        <p>16 conditional independence tests used by constraint-based structure learning algorithms.</p>

        <h3>Tests by Category</h3>
        <table>
          <thead>
            <tr><th>Category</th><th>Function</th><th>Data Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td>Discrete</td><td><code>ChiSquare</code></td><td>Categorical</td><td>Pearson's chi-squared test</td></tr>
            <tr><td>Discrete</td><td><code>GSq</code></td><td>Categorical</td><td>G-squared (log-likelihood ratio) test</td></tr>
            <tr><td>Discrete</td><td><code>ModifiedChiSquare</code></td><td>Categorical</td><td>Modified chi-squared with correction</td></tr>
            <tr><td>Discrete</td><td><code>PowerDivergence</code></td><td>Categorical</td><td>Power divergence family (generalizes chi-sq and G-sq)</td></tr>
            <tr><td>Continuous</td><td><code>FisherZ</code></td><td>Continuous</td><td>Fisher's Z-transform of partial correlation</td></tr>
            <tr><td>Continuous</td><td><code>Pearsonr</code></td><td>Continuous</td><td>Pearson correlation test</td></tr>
            <tr><td>Continuous</td><td><code>PartialCorrelation</code></td><td>Continuous</td><td>Partial correlation test</td></tr>
            <tr><td>Multivariate</td><td><code>GCM</code></td><td>Any</td><td>Generalized Covariance Measure</td></tr>
            <tr><td>Multivariate</td><td><code>HotellingLawley</code></td><td>Multivariate</td><td>Hotelling-Lawley trace test</td></tr>
            <tr><td>Multivariate</td><td><code>PillaiBartlett</code></td><td>Multivariate</td><td>Pillai-Bartlett trace test</td></tr>
            <tr><td>Tree-based</td><td><code>TreeBasedCI</code></td><td>Any</td><td>Decision-tree-based CI test</td></tr>
          </tbody>
        </table>

        <h3>Common Signature</h3>
        <pre><code>{`func ChiSquare(x, y string, z []string, data *tabgo.DataFrame, significance float64) (statistic float64, pValue float64, independent bool)`}</code></pre>
        <p>Parameters: <code>x</code> and <code>y</code> are variables to test, <code>z</code> is the conditioning set, <code>data</code> is the dataset, <code>significance</code> is the alpha level.</p>
        <p>Returns: test statistic, p-value, and whether X is independent of Y given Z at the given significance level.</p>

        <h3>Example</h3>
        <pre><code>{`stat, pValue, indep := ci_tests.ChiSquare("X", "Y", []string{"Z"}, data, 0.05)
fmt.Printf("Chi-square=%.3f, p=%.4f, independent=%v\\n", stat, pValue, indep)

stat2, pValue2, indep2 := ci_tests.FisherZ("X", "Y", []string{"Z"}, data, 0.05)
fmt.Printf("FisherZ=%.3f, p=%.4f, independent=%v\\n", stat2, pValue2, indep2)`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* STRUCTURE SCORE */}
      {/* ============================================================ */}
      <section class="section" id="api-structure-score">
        <h2>structure_score -- Scoring Functions</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/structure_score</code></p>
        <p>13 scoring function variants for score-based structure learning.</p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>BIC</code></td><td><code>NewBIC()</code></td><td>Bayesian Information Criterion. Penalizes by log(N)/2.</td></tr>
            <tr><td><code>AIC</code></td><td><code>NewAIC()</code></td><td>Akaike Information Criterion. Less penalty than BIC.</td></tr>
            <tr><td><code>BDeu</code></td><td><code>NewBDeu(equivalentSampleSize)</code></td><td>Bayesian Dirichlet equivalent uniform.</td></tr>
            <tr><td><code>BDs</code></td><td><code>NewBDs()</code></td><td>Bayesian Dirichlet sparse.</td></tr>
            <tr><td><code>K2</code></td><td><code>NewK2()</code></td><td>K2 score with uniform prior.</td></tr>
            <tr><td><code>LogLikelihood</code></td><td><code>NewLogLikelihood()</code></td><td>Raw log-likelihood (no penalty).</td></tr>
            <tr><td><code>Gaussian</code></td><td><code>NewGaussian()</code></td><td>Score for continuous Gaussian data.</td></tr>
            <tr><td><code>ConditionalGaussian</code></td><td><code>NewConditionalGaussian()</code></td><td>Score for mixed discrete/continuous.</td></tr>
          </tbody>
        </table>

        <h3>Common Method</h3>
        <pre><code>{`// All scorers provide LocalScore for use with structure learning
scorer := structure_score.NewBIC()
score := scorer.LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64`}</code></pre>

        <h3>Example</h3>
        <pre><code>{`bic := structure_score.NewBIC()
bdeu := structure_score.NewBDeu(5.0)  // equivalent sample size = 5

// Use with hill-climbing
hc := learning.NewHillClimbSearch(data, bic.LocalScore)
bn, _ := hc.Estimate()`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* IDENTIFICATION */}
      {/* ============================================================ */}
      <section class="section" id="api-identification">
        <h2>identification -- Causal Effect Identification</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/identification</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Adjustment</code></td><td><code>NewAdjustment(bn)</code></td><td>Back-door criterion. Finds valid adjustment sets to identify causal effects.</td></tr>
            <tr><td><code>Frontdoor</code></td><td><code>NewFrontdoor(bn)</code></td><td>Front-door criterion. Identifies effects via mediating variables.</td></tr>
          </tbody>
        </table>

        <h3>Methods</h3>
        <pre><code>{`adj := identification.NewAdjustment(bn)
adjustmentSet, err := adj.GetAdjustmentSet("Treatment", "Outcome")
// Returns the set of variables to condition on for back-door adjustment

fd := identification.NewFrontdoor(bn)
frontdoorSet, err := fd.GetFrontdoorSet("Treatment", "Outcome")
// Returns the set of mediating variables for front-door adjustment`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* PREDICTION */}
      {/* ============================================================ */}
      <section class="section" id="api-prediction">
        <h2>prediction -- Causal Prediction</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/prediction</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DoubleMLRegressor</code></td><td><code>NewDoubleMLRegressor(data, treatment, outcome, confounders)</code></td><td>Double/Debiased Machine Learning for ATE estimation. Handles high-dimensional confounders.</td></tr>
            <tr><td><code>NaiveAdjustmentRegressor</code></td><td><code>NewNaiveAdjustmentRegressor(data, treatment, outcome, adjustmentSet)</code></td><td>Naive back-door adjustment regression.</td></tr>
            <tr><td><code>NaiveIVRegressor</code></td><td><code>NewNaiveIVRegressor(data, instrument, treatment, outcome)</code></td><td>Instrumental variable regression for causal effects with unmeasured confounding.</td></tr>
          </tbody>
        </table>

        <h3>Common Methods</h3>
        <pre><code>{`regressor := prediction.NewDoubleMLRegressor(data, "Treatment", "Outcome", []string{"X1", "X2"})
ate, err := regressor.EstimateATE()
fmt.Println("Average Treatment Effect:", ate)`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* METRICS */}
      {/* ============================================================ */}
      <section class="section" id="api-metrics">
        <h2>metrics -- Model Evaluation</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/metrics</code></p>

        <h3>Functions</h3>
        <table>
          <thead>
            <tr><th>Function</th><th>Signature</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>SHD</code></td><td><code>(true, est *graphgo.DiGraph) int</code></td><td>Structural Hamming Distance (edge additions + deletions + reversals)</td></tr>
            <tr><td><code>AdjacencyConfusionMatrix</code></td><td><code>(true, est *graphgo.DiGraph) (tp, fp, tn, fn int)</code></td><td>Edge presence TP/FP/TN/FN (ignoring direction)</td></tr>
            <tr><td><code>OrientationConfusionMatrix</code></td><td><code>(true, est *graphgo.DiGraph) (tp, fp, tn, fn int)</code></td><td>Edge orientation TP/FP/TN/FN</td></tr>
            <tr><td><code>CorrelationScore</code></td><td><code>(true, est *graphgo.DiGraph) float64</code></td><td>Correlation-based structure similarity</td></tr>
            <tr><td><code>FisherC</code></td><td><code>(bn *models.BayesianNetwork, data *tabgo.DataFrame) float64</code></td><td>Fisher's C statistic for model fit</td></tr>
            <tr><td><code>LogLikelihoodScore</code></td><td><code>(bn *models.BayesianNetwork, data *tabgo.DataFrame) float64</code></td><td>Log-likelihood of data given model</td></tr>
            <tr><td><code>NormalizedSHD</code></td><td><code>(true, est *graphgo.DiGraph) float64</code></td><td>SHD normalized by max possible SHD</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`shd := metrics.SHD(trueGraph, learnedGraph)
fmt.Println("SHD:", shd)

tp, fp, tn, fn := metrics.AdjacencyConfusionMatrix(trueGraph, learnedGraph)
precision := float64(tp) / float64(tp + fp)
recall := float64(tp) / float64(tp + fn)
fmt.Printf("Precision=%.3f, Recall=%.3f\\n", precision, recall)`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* READWRITE */}
      {/* ============================================================ */}
      <section class="section" id="api-readwrite">
        <h2>readwrite -- File Format I/O</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/readwrite</code></p>
        <p>10 file format readers and writers. All read functions take <code>io.Reader</code>, all write functions take <code>io.Writer</code>.</p>

        <h3>Functions</h3>
        <table>
          <thead>
            <tr><th>Read</th><th>Write</th><th>Format</th></tr>
          </thead>
          <tbody>
            <tr><td><code>ReadBIF(r io.Reader) (*models.BayesianNetwork, error)</code></td><td><code>WriteBIF(w io.Writer, bn *models.BayesianNetwork) error</code></td><td>BIF</td></tr>
            <tr><td><code>ReadXMLBIF(r) (*BN, error)</code></td><td><code>WriteXMLBIF(w, bn) error</code></td><td>XMLBIF</td></tr>
            <tr><td><code>ReadNET(r) (*BN, error)</code></td><td><code>WriteNET(w, bn) error</code></td><td>NET</td></tr>
            <tr><td><code>ReadUAI(r) (*BN, error)</code></td><td><code>WriteUAI(w, bn) error</code></td><td>UAI</td></tr>
            <tr><td><code>ReadXDSL(r) (*BN, error)</code></td><td><code>WriteXDSL(w, bn) error</code></td><td>XDSL</td></tr>
            <tr><td><code>ReadPomdpX(r) (*BN, error)</code></td><td>--</td><td>PomdpX (read only)</td></tr>
            <tr><td><code>ReadXBN(r) (*BN, error)</code></td><td>--</td><td>XBN (read only)</td></tr>
            <tr><td><code>ReadCSVModel(r) (*BN, error)</code></td><td><code>WriteCSVModel(w, bn) error</code></td><td>CSV</td></tr>
            <tr><td><code>ReadJSONModel(r) (*BN, error)</code></td><td><code>WriteJSONModel(w, bn) error</code></td><td>JSON</td></tr>
            <tr><td><code>ReadXMLModel(r) (*BN, error)</code></td><td><code>WriteXMLModel(w, bn) error</code></td><td>XML</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`// Read BIF
f, _ := os.Open("model.bif")
bn, _ := readwrite.ReadBIF(f)
f.Close()

// Write JSON
out, _ := os.Create("model.json")
readwrite.WriteJSONModel(out, bn)
out.Close()`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* BASE */}
      {/* ============================================================ */}
      <section class="section" id="api-base">
        <h2>base -- Foundational Graph Types</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/base</code></p>
        <p>7 graph types that serve as the structural foundation for all model types.</p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DAG</code></td><td><code>NewDAG()</code></td><td>Directed acyclic graph with cycle detection, topological sort, d-separation</td></tr>
            <tr><td><code>PDAG</code></td><td><code>NewPDAG()</code></td><td>Partially directed acyclic graph (for equivalence classes)</td></tr>
            <tr><td><code>UndirectedGraph</code></td><td><code>NewUndirectedGraph()</code></td><td>Undirected graph for Markov networks</td></tr>
            <tr><td><code>ADMG</code></td><td><code>NewADMG()</code></td><td>Acyclic directed mixed graph (directed + bidirected edges)</td></tr>
            <tr><td><code>MAG</code></td><td><code>NewMAG()</code></td><td>Maximal ancestral graph</td></tr>
            <tr><td><code>SimpleCausalModel</code></td><td><code>NewSimpleCausalModel()</code></td><td>Basic causal model with intervention semantics</td></tr>
          </tbody>
        </table>

        <h3>DAG Methods</h3>
        <pre><code>{`dag := base.NewDAG()
dag.AddNode("X")
dag.AddNode("Y")
dag.AddNode("Z")
dag.AddEdge("X", "Y")
dag.AddEdge("Y", "Z")

sorted := dag.TopologicalSort()
hasCycle := dag.HasCycle()
separated := dag.DSeparation([]string{"X"}, []string{"Z"}, []string{"Y"})
ancestors := dag.Ancestors("Z")
descendants := dag.Descendants("X")`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* INDEPENDENCIES */}
      {/* ============================================================ */}
      <section class="section" id="api-independencies">
        <h2>independencies -- Independence Assertions</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/independencies</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>IndependenceAssertion</code></td><td>A single assertion: X _||_ Y | Z. Fields: Event1, Event2, Event3 (conditioning).</td></tr>
            <tr><td><code>Independencies</code></td><td>A collection of IndependenceAssertions.</td></tr>
          </tbody>
        </table>

        <pre><code>{`// Get all independencies from a BN
indeps := bn.GetIndependencies()
for _, assertion := range indeps.Assertions() {
    fmt.Printf("%v _||_ %v | %v\\n", assertion.Event1, assertion.Event2, assertion.Event3)
}`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* CONFIG */}
      {/* ============================================================ */}
      <section class="section" id="api-config">
        <h2>config -- Configuration</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/config</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Config</code></td><td>Global configuration for tolerances, defaults, and behavior tuning.</td></tr>
          </tbody>
        </table>

        <pre><code>{`cfg := config.Global()
// Configuration controls numerical tolerances, default methods,
// convergence thresholds, and logging verbosity.`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* UTILS */}
      {/* ============================================================ */}
      <section class="section" id="api-utils">
        <h2>utils -- Shared Utilities</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/src/utils</code></p>
        <p>
          Shared parsing, optimization, and compatibility utilities. Includes helper functions
          for set operations, string parsing, numerical utilities, and other common operations
          used across packages.
        </p>
      </section>

      {/* ============================================================ */}
      {/* NUMGO */}
      {/* ============================================================ */}
      <section class="section" id="api-numgo">
        <h2>numgo -- N-Dimensional Arrays (numpy equivalent)</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/lib/numgo</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>NDArray</code></td><td><code>NewNDArray(shape []int)</code></td><td>N-dimensional array with shape, stride, element-wise ops</td></tr>
            <tr><td><code>Matrix</code></td><td><code>NewMatrix(rows, cols int)</code> / <code>NewMatrixFromData(rows, cols, data)</code></td><td>2D matrix with linear algebra operations</td></tr>
            <tr><td><code>Vector</code></td><td><code>NewVector(data []float64)</code></td><td>1D vector with dot product, norm, element-wise ops</td></tr>
          </tbody>
        </table>

        <h3>NDArray Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Shape() []int</code></td><td>Get array dimensions</td></tr>
            <tr><td><code>Fill(value float64)</code></td><td>Set all elements to value</td></tr>
            <tr><td><code>Sum() float64</code></td><td>Sum all elements</td></tr>
            <tr><td><code>Data() []float64</code></td><td>Get underlying flat data</td></tr>
            <tr><td><code>Get(indices ...int) float64</code></td><td>Get element at indices</td></tr>
            <tr><td><code>Set(value float64, indices ...int)</code></td><td>Set element at indices</td></tr>
          </tbody>
        </table>

        <h3>Matrix Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Multiply(other *Matrix) *Matrix</code></td><td>Matrix multiplication</td></tr>
            <tr><td><code>Transpose() *Matrix</code></td><td>Matrix transpose</td></tr>
            <tr><td><code>Inverse() *Matrix</code></td><td>Matrix inverse</td></tr>
            <tr><td><code>Det() float64</code></td><td>Determinant</td></tr>
            <tr><td><code>Rows() int</code></td><td>Number of rows</td></tr>
            <tr><td><code>Cols() int</code></td><td>Number of columns</td></tr>
            <tr><td><code>Get(i, j int) float64</code></td><td>Get element</td></tr>
            <tr><td><code>Set(i, j int, v float64)</code></td><td>Set element</td></tr>
            <tr><td><code>Data() []float64</code></td><td>Get flat data</td></tr>
          </tbody>
        </table>

        <h3>Vector Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Dot(other *Vector) float64</code></td><td>Dot product</td></tr>
            <tr><td><code>Norm() float64</code></td><td>L2 norm</td></tr>
            <tr><td><code>Add(other *Vector) *Vector</code></td><td>Element-wise addition</td></tr>
            <tr><td><code>Scale(s float64) *Vector</code></td><td>Scalar multiplication</td></tr>
            <tr><td><code>Data() []float64</code></td><td>Get underlying data</td></tr>
            <tr><td><code>Len() int</code></td><td>Vector length</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`m := numgo.NewMatrixFromData(2, 2, []float64{1, 2, 3, 4})
det := m.Det()     // -2
inv := m.Inverse()
identity := m.Multiply(inv)  // should be [[1,0],[0,1]]

v := numgo.NewVector([]float64{1, 2, 3})
norm := v.Norm()   // sqrt(14)`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* SCIGO */}
      {/* ============================================================ */}
      <section class="section" id="api-scigo">
        <h2>scigo -- Statistical Computing (scipy equivalent)</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/lib/scigo</code></p>

        <h3>Distributions</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Methods</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Normal</code></td><td><code>NewNormal(mean, std)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code>, <code>Sample</code>, <code>Mean</code>, <code>Variance</code></td></tr>
            <tr><td><code>ChiSquared</code></td><td><code>NewChiSquared(df)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code>, <code>Sample</code></td></tr>
            <tr><td><code>Beta</code></td><td><code>NewBeta(alpha, beta)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code>, <code>Mean</code></td></tr>
            <tr><td><code>Gamma</code></td><td><code>NewGamma(shape, rate)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code>, <code>Mean</code></td></tr>
            <tr><td><code>StudentT</code></td><td><code>NewStudentT(df)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code></td></tr>
            <tr><td><code>Uniform</code></td><td><code>NewUniform(low, high)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code>, <code>Sample</code></td></tr>
            <tr><td><code>Exponential</code></td><td><code>NewExponential(rate)</code></td><td><code>PDF</code>, <code>CDF</code>, <code>PPF</code></td></tr>
          </tbody>
        </table>

        <h3>Optimization</h3>
        <table>
          <thead>
            <tr><th>Function</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Minimize(f func(float64) float64, low, high float64) float64</code></td><td>Minimize a univariate function on [low, high]</td></tr>
            <tr><td><code>GradientDescent(f, grad func([]float64) float64, x0 []float64, lr float64, maxIter int) []float64</code></td><td>Gradient descent optimization</td></tr>
            <tr><td><code>NewtonMethod(f, fprime func(float64) float64, x0 float64, tol float64, maxIter int) float64</code></td><td>Newton's method for root finding</td></tr>
          </tbody>
        </table>

        <h3>Statistics</h3>
        <table>
          <thead>
            <tr><th>Function</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Mean(data []float64) float64</code></td><td>Arithmetic mean</td></tr>
            <tr><td><code>Std(data []float64) float64</code></td><td>Standard deviation</td></tr>
            <tr><td><code>PearsonCorrelation(x, y []float64) float64</code></td><td>Pearson correlation coefficient</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`n := scigo.NewNormal(0, 1)
fmt.Println("CDF(1.96):", n.CDF(1.96))   // ~0.975
fmt.Println("PPF(0.975):", n.PPF(0.975))  // ~1.96

chi2 := scigo.NewChiSquared(5)
pValue := 1.0 - chi2.CDF(11.07)  // ~0.05`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* GRAPHGO */}
      {/* ============================================================ */}
      <section class="section" id="api-graphgo">
        <h2>graphgo -- Graph Algorithms (networkx equivalent)</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/lib/graphgo</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Constructor</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DiGraph</code></td><td><code>NewDiGraph()</code></td><td>Directed graph</td></tr>
            <tr><td><code>Graph</code></td><td><code>NewGraph()</code></td><td>Undirected graph</td></tr>
            <tr><td><code>PDAG</code></td><td><code>NewPDAG()</code></td><td>Partially directed acyclic graph</td></tr>
          </tbody>
        </table>

        <h3>DiGraph Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>AddNode(name string)</code></td><td>Add a node</td></tr>
            <tr><td><code>AddEdge(from, to string)</code></td><td>Add a directed edge</td></tr>
            <tr><td><code>RemoveNode(name string)</code></td><td>Remove a node and edges</td></tr>
            <tr><td><code>RemoveEdge(from, to string)</code></td><td>Remove an edge</td></tr>
            <tr><td><code>Nodes() []string</code></td><td>List nodes</td></tr>
            <tr><td><code>Edges() [][2]string</code></td><td>List edges</td></tr>
            <tr><td><code>Successors(name string) []string</code></td><td>Children of a node</td></tr>
            <tr><td><code>Predecessors(name string) []string</code></td><td>Parents of a node</td></tr>
            <tr><td><code>HasCycle() bool</code></td><td>Detect cycles</td></tr>
            <tr><td><code>TopologicalSort() []string</code></td><td>Topological ordering</td></tr>
            <tr><td><code>DSeparation(x, y, z []string) bool</code></td><td>Test d-separation</td></tr>
            <tr><td><code>MoralGraph() *Graph</code></td><td>Convert to moral graph</td></tr>
          </tbody>
        </table>

        <h3>Graph Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>AddEdge(a, b string)</code></td><td>Add undirected edge</td></tr>
            <tr><td><code>Neighbors(name string) []string</code></td><td>Get neighbors</td></tr>
            <tr><td><code>Degree(name string) int</code></td><td>Get degree</td></tr>
            <tr><td><code>ConnectedComponents() [][]string</code></td><td>Find connected components</td></tr>
          </tbody>
        </table>

        <h3>Algorithms</h3>
        <p>
          Topological sort, d-separation, moral graph, triangulation, maximum cardinality search,
          clique finding, connected components, shortest paths, cycle detection, ancestors, descendants.
        </p>

        <h3>Example</h3>
        <pre><code>{`g := graphgo.NewDiGraph()
g.AddNode("A")
g.AddNode("B")
g.AddNode("C")
g.AddEdge("A", "B")
g.AddEdge("B", "C")

fmt.Println("Topo sort:", g.TopologicalSort())   // [A, B, C]
fmt.Println("Children of A:", g.Successors("A"))  // [B]
fmt.Println("Has cycle:", g.HasCycle())           // false`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* TABGO */}
      {/* ============================================================ */}
      <section class="section" id="api-tabgo">
        <h2>tabgo -- Tabular Data (pandas equivalent)</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/lib/tabgo</code></p>

        <h3>Types</h3>
        <table>
          <thead>
            <tr><th>Type</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>DataFrame</code></td><td>Tabular data with named columns, filtering, groupby</td></tr>
            <tr><td><code>Series</code></td><td>Single column with value counts, unique, statistics</td></tr>
          </tbody>
        </table>

        <h3>DataFrame Constructors</h3>
        <pre><code>{`// From CSV file
df, err := tabgo.ReadCSV("data.csv")

// From map
df := tabgo.NewDataFrame(map[string][]interface{}{
    "Name":  {"Alice", "Bob"},
    "Score": {95, 87},
})`}</code></pre>

        <h3>DataFrame Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>NRows() int</code></td><td>Number of rows</td></tr>
            <tr><td><code>Columns() []string</code></td><td>Column names</td></tr>
            <tr><td><code>Column(name string) *Series</code></td><td>Get a column as Series</td></tr>
            <tr><td><code>Filter(fn func(map[string]interface{}) bool) *DataFrame</code></td><td>Filter rows by predicate</td></tr>
            <tr><td><code>GroupBy(col string) *GroupedDataFrame</code></td><td>Group by column</td></tr>
          </tbody>
        </table>

        <h3>Series Methods</h3>
        <table>
          <thead>
            <tr><th>Method</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>Unique() []interface{}</code></td><td>Unique values</td></tr>
            <tr><td><code>ValueCounts() map[interface{}]int</code></td><td>Count of each value</td></tr>
            <tr><td><code>Float64() []float64</code></td><td>Convert to float64 slice</td></tr>
          </tbody>
        </table>

        <h3>I/O Functions</h3>
        <table>
          <thead>
            <tr><th>Function</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>ReadCSV(path string) (*DataFrame, error)</code></td><td>Read CSV file</td></tr>
            <tr><td><code>WriteCSV(df *DataFrame, path string) error</code></td><td>Write CSV file</td></tr>
            <tr><td><code>ReadParquet(path string) (*DataFrame, error)</code></td><td>Read Parquet file</td></tr>
            <tr><td><code>WriteParquet(df *DataFrame, path string) error</code></td><td>Write Parquet file</td></tr>
            <tr><td><code>ReadXLSX(path string) (*DataFrame, error)</code></td><td>Read Excel file</td></tr>
          </tbody>
        </table>

        <h3>Example</h3>
        <pre><code>{`df, _ := tabgo.ReadCSV("data.csv")
fmt.Println("Rows:", df.NRows())
fmt.Println("Columns:", df.Columns())

col := df.Column("Score")
fmt.Println("Unique:", col.Unique())
fmt.Println("Counts:", col.ValueCounts())

tabgo.WriteCSV(df, "output.csv")`}</code></pre>
      </section>

      {/* ============================================================ */}
      {/* GPU */}
      {/* ============================================================ */}
      <section class="section" id="api-gpu">
        <h2>gpu -- GPU Compute Backend</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/lib/gpu</code></p>
        <p>
          Optional GPU acceleration for compute-intensive operations. Provides GPU-backed
          matrix operations and factor computations for large-scale inference and learning.
        </p>
      </section>

      {/* ============================================================ */}
      {/* EXAMPLE MODELS */}
      {/* ============================================================ */}
      <section class="section" id="api-example-models">
        <h2>example_models -- Built-in Models</h2>
        <p>Import: <code>github.com/asymmetric-effort/pgmgo/example_models</code></p>

        <h3>Functions</h3>
        <table>
          <thead>
            <tr><th>Function</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>List() []string</code></td><td>List all 25 available model names</td></tr>
            <tr><td><code>Get(name string) (*models.BayesianNetwork, error)</code></td><td>Load a model by name</td></tr>
          </tbody>
        </table>

        <h3>Available Models</h3>
        <p>
          <strong>With CPDs (13):</strong> student, asia, alarm, cancer, watersprinkler, survey, montyhall,
          dogproblem, frauddetection, medicaldiagnosis, earthquake, visitasia, cointoss
        </p>
        <p>
          <strong>Structure-only (12):</strong> sachs, child, insurance, alarmfull, water, mildew,
          barley, hailfinder, hepar2, win95pts, pathfinder, pigs
        </p>

        <h3>Example</h3>
        <pre><code>{`names := example_models.List()
fmt.Println("Available:", len(names), "models")

bn, err := example_models.Get("asia")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("%d nodes, %d edges\\n", len(bn.Nodes()), len(bn.Edges()))`}</code></pre>
      </section>
    </div>
  );
}
