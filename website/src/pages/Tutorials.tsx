import { createElement, useHead } from "@asymmetric-effort/specifyjs";

export function Tutorials() {
  useHead({
    title: "Tutorials — pgmgo",
    description: "Step-by-step tutorials for building Bayesian networks, learning structure, causal inference, and file formats with pgmgo.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/tutorials",
  });

  return (
    <div class="page">
      <h1>Tutorials</h1>

      <section class="section">
        <h2>Tutorial 1: Building Your First Bayesian Network</h2>
        <p>
          This tutorial walks through creating the classic "wet grass" Bayesian network from
          scratch, adding conditional probability distributions, validating the model, and
          running your first inference query.
        </p>

        <h3>Step 1: Create the Network Structure</h3>
        <p>
          A Bayesian network is a directed acyclic graph (DAG) where nodes represent random
          variables and edges represent conditional dependencies. We will model the relationship
          between cloudy weather, rain, a sprinkler, and wet grass.
        </p>
        <pre><code>{`package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

func main() {
    // Create an empty Bayesian network
    bn := models.NewBayesianNetwork()

    // Add nodes (random variables)
    bn.AddNode("Cloudy")
    bn.AddNode("Sprinkler")
    bn.AddNode("Rain")
    bn.AddNode("WetGrass")

    // Add directed edges (causal relationships)
    // Cloudy weather influences both the sprinkler and rain
    bn.AddEdge("Cloudy", "Sprinkler")
    bn.AddEdge("Cloudy", "Rain")
    // Both sprinkler and rain cause wet grass
    bn.AddEdge("Sprinkler", "WetGrass")
    bn.AddEdge("Rain", "WetGrass")

    fmt.Println("Nodes:", bn.Nodes())
    fmt.Println("Edges:", bn.Edges())
}`}</code></pre>

        <h3>Step 2: Define States and CPDs</h3>
        <p>
          Each node needs defined states (possible values) and a conditional probability
          distribution (CPD). Root nodes have marginal distributions; child nodes have
          distributions conditioned on their parents.
        </p>
        <pre><code>{`    // Define states
    bn.SetStates("Cloudy", []string{"clear", "cloudy"})
    bn.SetStates("Sprinkler", []string{"off", "on"})
    bn.SetStates("Rain", []string{"no", "yes"})
    bn.SetStates("WetGrass", []string{"dry", "wet"})

    // P(Cloudy) - root node, no parents
    bn.SetCPD("Cloudy", factors.NewTabularCPD(
        "Cloudy", 2,
        []float64{0.5, 0.5},
        nil, nil,
    ))

    // P(Sprinkler | Cloudy)
    // Columns: Cloudy=clear, Cloudy=cloudy
    // Rows: Sprinkler=off, Sprinkler=on
    bn.SetCPD("Sprinkler", factors.NewTabularCPD(
        "Sprinkler", 2,
        []float64{
            0.5, 0.9,  // P(off | clear)=0.5, P(off | cloudy)=0.9
            0.5, 0.1,  // P(on  | clear)=0.5, P(on  | cloudy)=0.1
        },
        []string{"Cloudy"}, []int{2},
    ))

    // P(Rain | Cloudy)
    bn.SetCPD("Rain", factors.NewTabularCPD(
        "Rain", 2,
        []float64{
            0.8, 0.2,  // P(no  | clear)=0.8, P(no  | cloudy)=0.2
            0.2, 0.8,  // P(yes | clear)=0.2, P(yes | cloudy)=0.8
        },
        []string{"Cloudy"}, []int{2},
    ))

    // P(WetGrass | Sprinkler, Rain)
    // Parent order: Sprinkler, Rain
    // Columns iterate over all parent combinations
    bn.SetCPD("WetGrass", factors.NewTabularCPD(
        "WetGrass", 2,
        []float64{
            //  S=off,R=no  S=off,R=yes  S=on,R=no  S=on,R=yes
            1.0,         0.1,         0.1,       0.01,   // P(dry | ...)
            0.0,         0.9,         0.9,       0.99,   // P(wet | ...)
        },
        []string{"Sprinkler", "Rain"}, []int{2, 2},
    ))`}</code></pre>

        <h3>Step 3: Validate the Model</h3>
        <pre><code>{`    // CheckModel verifies:
    // - The graph is a valid DAG (no cycles)
    // - Every node has a CPD
    // - CPD dimensions match the graph structure
    // - All probability distributions sum to 1
    if err := bn.CheckModel(); err != nil {
        log.Fatal("Model validation failed:", err)
    }
    fmt.Println("Model is valid!")`}</code></pre>

        <h3>Step 4: Run an Inference Query</h3>
        <pre><code>{`    // Convert CPDs to Markov factors for Variable Elimination
    facs, err := bn.ToMarkovFactors()
    if err != nil {
        log.Fatal(err)
    }
    ve := inference.NewVariableElimination(facs)

    // Query: What is P(Rain | WetGrass=wet)?
    // "It's wet outside — did it rain?"
    result, err := ve.Query(
        []string{"Rain"},
        map[string]int{"WetGrass": 1},  // 1 = "wet"
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("P(Rain | WetGrass=wet):", result.Values().Data())
    // Expected: rain is more likely when grass is wet

    // MAP query: most likely state of Sprinkler given it rained
    assignment, err := ve.MAP(
        []string{"Sprinkler"},
        map[string]int{"Rain": 1},
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Most likely Sprinkler state given Rain:", assignment)
}`}</code></pre>

        <h3>Step 5: Save the Model</h3>
        <pre><code>{`import (
    "os"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
)

// Save as BIF
f, _ := os.Create("wetgrass.bif")
defer f.Close()
readwrite.WriteBIF(f, bn)`}</code></pre>
      </section>

      <section class="section">
        <h2>Tutorial 2: Learning Structure from Data</h2>
        <p>
          When you have observational data but do not know the causal structure, pgmgo can
          learn the network structure automatically. This tutorial demonstrates score-based
          and constraint-based approaches.
        </p>

        <h3>Step 1: Prepare Your Data</h3>
        <p>
          Data should be in CSV format with column headers matching variable names.
          Each row is one observation. Values should be integers representing discrete states.
        </p>
        <pre><code>{`// Example CSV format:
// Cloudy,Sprinkler,Rain,WetGrass
// 0,1,0,1
// 1,0,1,1
// 0,0,0,0
// ...`}</code></pre>

        <h3>Step 2: Generate Training Data by Sampling</h3>
        <p>
          If you have an existing model, you can generate training data by sampling.
          This is useful for testing learning algorithms.
        </p>
        <pre><code>{`import (
    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/sampling"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// Load a known model
bn, _ := example_models.Get("asia")

// Generate 5000 samples
bms, _ := sampling.NewBayesianModelSampling(bn, 42)
data, _ := bms.ForwardSample(5000)

// Save to CSV for later use
tabgo.WriteCSV(data, "asia_samples.csv")`}</code></pre>

        <h3>Step 3: Score-Based Learning (Hill-Climb)</h3>
        <p>
          Hill-climbing searches the space of DAGs by iteratively adding, removing, or
          reversing edges to maximize a scoring function.
        </p>
        <pre><code>{`import (
    "github.com/asymmetric-effort/pgmgo/src/learning"
    "github.com/asymmetric-effort/pgmgo/src/structure_score"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// Load data
data, _ := tabgo.ReadCSV("asia_samples.csv")

// Choose a scoring function
scorer := structure_score.NewBIC()

// Run hill-climbing
hc := learning.NewHillClimbSearch(data, scorer.LocalScore)
learnedBN, err := hc.Estimate()
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Learned: %d nodes, %d edges\\n",
    len(learnedBN.Nodes()), len(learnedBN.Edges()))`}</code></pre>

        <h3>Step 4: Constraint-Based Learning (PC Algorithm)</h3>
        <p>
          The PC algorithm uses conditional independence tests to discover the network
          structure. It first finds the skeleton (undirected edges), then orients edges.
        </p>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/ci_tests"

// Wrap the CI test for the learning API
ciTest := func(x, y string, z []string, data *tabgo.DataFrame, sig float64) (float64, float64, bool) {
    return ci_tests.ChiSquare(x, y, z, data, sig)
}

// Run PC algorithm
pc := learning.NewPC(data, ciTest, 0.05) // significance level = 0.05
pcBN, err := pc.EstimateBN()
if err != nil {
    log.Fatal(err)
}
fmt.Printf("PC: %d nodes, %d edges\\n",
    len(pcBN.Nodes()), len(pcBN.Edges()))`}</code></pre>

        <h3>Step 5: Evaluate the Learned Structure</h3>
        <pre><code>{`import (
    "github.com/asymmetric-effort/pgmgo/src/metrics"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// Convert to DiGraphs for comparison
toDigraph := func(bn *models.BayesianNetwork) *graphgo.DiGraph {
    g := graphgo.NewDiGraph()
    for _, n := range bn.Nodes() { g.AddNode(n) }
    for _, e := range bn.Edges() { g.AddEdge(e[0], e[1]) }
    return g
}

trueG := toDigraph(bn)       // original Asia model
hcG := toDigraph(learnedBN)  // hill-climb result
pcG := toDigraph(pcBN)       // PC result

fmt.Println("Hill-Climb SHD:", metrics.SHD(trueG, hcG))
fmt.Println("PC SHD:", metrics.SHD(trueG, pcG))

// Detailed adjacency analysis
aTP, aFP, aTN, aFN := metrics.AdjacencyConfusionMatrix(trueG, hcG)
fmt.Printf("HC Adjacency: TP=%d FP=%d TN=%d FN=%d\\n", aTP, aFP, aTN, aFN)`}</code></pre>

        <h3>Step 6: Fit Parameters to the Learned Structure</h3>
        <pre><code>{`// MLE parameter fitting
mle := learning.NewMLE(learnedBN, data)
if err := mle.Estimate(); err != nil {
    log.Fatal(err)
}

// Now the learned BN has both structure and parameters
// You can run inference on it
facs, _ := learnedBN.ToMarkovFactors()
ve := inference.NewVariableElimination(facs)
result, _ := ve.Query([]string{"Dyspnea"}, map[string]int{"Smoker": 1})
fmt.Println("P(Dyspnea | Smoker=1):", result.Values().Data())`}</code></pre>
      </section>

      <section class="section">
        <h2>Tutorial 3: Causal Inference</h2>
        <p>
          Bayesian networks encode causal relationships. pgmgo supports causal reasoning
          via do-calculus, allowing you to answer interventional questions like
          "What would happen if we forced X to a specific value?"
        </p>

        <h3>Observation vs. Intervention</h3>
        <p>
          <strong>Observing</strong> (conditioning): P(Y | X=x) -- what is the probability
          of Y given that we see X=x? This reflects correlation.
        </p>
        <p>
          <strong>Intervening</strong> (do-calculus): P(Y | do(X=x)) -- what is the probability
          of Y if we force X to be x? This reflects causation. The do() operator "cuts" incoming
          edges to X, removing confounding effects.
        </p>

        <h3>Step 1: Build a Causal Model</h3>
        <pre><code>{`import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

func main() {
    // Smoking -> Tar -> Cancer
    // Smoking -> Cancer (direct effect)
    bn := models.NewBayesianNetwork()
    bn.AddNode("Smoking")
    bn.AddNode("Tar")
    bn.AddNode("Cancer")
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
            0.98, 0.9, 0.7, 0.05,
            0.02, 0.1, 0.3, 0.95,
        },
        []string{"Smoking", "Tar"}, []int{2, 2},
    ))
    bn.CheckModel()`}</code></pre>

        <h3>Step 2: Compare Observation vs. Intervention</h3>
        <pre><code>{`    // Observational query: P(Cancer | Smoking=yes)
    facs, _ := bn.ToMarkovFactors()
    ve := inference.NewVariableElimination(facs)
    obsResult, _ := ve.Query(
        []string{"Cancer"},
        map[string]int{"Smoking": 1},
    )
    fmt.Println("P(Cancer | Smoking=yes):", obsResult.Values().Data())

    // Interventional query: P(Cancer | do(Smoking=yes))
    ci, err := inference.NewCausalInference(bn)
    if err != nil {
        log.Fatal(err)
    }
    doResult, _ := ci.Query(
        []string{"Cancer"},
        map[string]int{"Smoking": 1},
        nil,
    )
    fmt.Println("P(Cancer | do(Smoking=yes)):", doResult.Values().Data())
    // Note: the interventional probability may differ from the
    // observational probability when there are confounders.`}</code></pre>

        <h3>Step 3: Interventional Queries with Evidence</h3>
        <pre><code>{`    // P(Cancer | do(Tar=high), Smoking=no)
    // "If we force tar to be high but the person doesn't smoke,
    //  what is the cancer risk?"
    result, _ := ci.Query(
        []string{"Cancer"},
        map[string]int{"Tar": 1},        // intervene on Tar
        map[string]int{"Smoking": 0},     // observe Smoking
    )
    fmt.Println("P(Cancer | do(Tar=high), Smoking=no):", result.Values().Data())
}`}</code></pre>

        <h3>Using the CLI for Causal Queries</h3>
        <pre><code>{`# Same query from the command line
$ pgmgo do model.bif --intervention Smoking=1 --query Cancer
Cancer	P
0	0.350000
1	0.650000

# With observational evidence
$ pgmgo do model.bif --intervention Tar=1 --query Cancer --evidence Smoking=0`}</code></pre>

        <h3>Causal Identification</h3>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/src/identification"

// Check if causal effect is identifiable via back-door
adj := identification.NewAdjustment(bn)
adjustmentSet, err := adj.GetAdjustmentSet("Smoking", "Cancer")
if err == nil {
    fmt.Println("Back-door adjustment set:", adjustmentSet)
}

// Front-door criterion
fd := identification.NewFrontdoor(bn)
frontdoorSet, err := fd.GetFrontdoorSet("Smoking", "Cancer")
if err == nil {
    fmt.Println("Front-door set:", frontdoorSet)
}`}</code></pre>
      </section>

      <section class="section">
        <h2>Tutorial 4: Working with Different File Formats</h2>
        <p>
          pgmgo supports 10 file formats for model serialization. This tutorial covers
          reading, writing, and converting between formats.
        </p>

        <h3>Supported Formats</h3>
        <table>
          <thead>
            <tr><th>Format</th><th>Extension</th><th>Read</th><th>Write</th><th>Common Use</th></tr>
          </thead>
          <tbody>
            <tr><td>BIF</td><td>.bif</td><td>Yes</td><td>Yes</td><td>Standard PGM interchange format</td></tr>
            <tr><td>XMLBIF</td><td>.xmlbif</td><td>Yes</td><td>Yes</td><td>XML-based interchange</td></tr>
            <tr><td>NET</td><td>.net</td><td>Yes</td><td>Yes</td><td>Hugin software format</td></tr>
            <tr><td>UAI</td><td>.uai</td><td>Yes</td><td>Yes</td><td>UAI inference competitions</td></tr>
            <tr><td>XDSL</td><td>.xdsl</td><td>Yes</td><td>Yes</td><td>GeNIe/SMILE software</td></tr>
            <tr><td>PomdpX</td><td>.pomdpx</td><td>Yes</td><td>--</td><td>POMDP planning</td></tr>
            <tr><td>XBN</td><td>.xbn</td><td>Yes</td><td>--</td><td>Microsoft Research format</td></tr>
            <tr><td>CSV</td><td>.csv</td><td>Yes</td><td>Yes</td><td>Simple tabular model storage</td></tr>
            <tr><td>JSON</td><td>.json</td><td>Yes</td><td>Yes</td><td>Web APIs and JavaScript interop</td></tr>
            <tr><td>XML</td><td>.xml</td><td>Yes</td><td>Yes</td><td>General XML model storage</td></tr>
          </tbody>
        </table>

        <h3>Step 1: Reading Models</h3>
        <pre><code>{`import (
    "os"
    "github.com/asymmetric-effort/pgmgo/src/readwrite"
)

// Read a BIF file
func loadBIF(path string) *models.BayesianNetwork {
    f, err := os.Open(path)
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    bn, err := readwrite.ReadBIF(f)
    if err != nil {
        log.Fatal(err)
    }
    return bn
}

// Read other formats — same pattern
f, _ := os.Open("model.xmlbif")
bn, _ := readwrite.ReadXMLBIF(f)

f, _ = os.Open("model.net")
bn, _ = readwrite.ReadNET(f)

f, _ = os.Open("model.uai")
bn, _ = readwrite.ReadUAI(f)

f, _ = os.Open("model.xdsl")
bn, _ = readwrite.ReadXDSL(f)`}</code></pre>

        <h3>Step 2: Writing Models</h3>
        <pre><code>{`// Write to BIF
func saveBIF(bn *models.BayesianNetwork, path string) {
    f, err := os.Create(path)
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()
    if err := readwrite.WriteBIF(f, bn); err != nil {
        log.Fatal(err)
    }
}

// Write to other formats
out, _ := os.Create("model.xmlbif")
readwrite.WriteXMLBIF(out, bn)
out.Close()

out, _ = os.Create("model.net")
readwrite.WriteNET(out, bn)
out.Close()

out, _ = os.Create("model.uai")
readwrite.WriteUAI(out, bn)
out.Close()

out, _ = os.Create("model.xdsl")
readwrite.WriteXDSL(out, bn)
out.Close()`}</code></pre>

        <h3>Step 3: JSON Serialization (for Web Applications)</h3>
        <pre><code>{`// Write model as JSON
jsonFile, _ := os.Create("model.json")
readwrite.WriteJSONModel(jsonFile, bn)
jsonFile.Close()

// Read model from JSON
jsonIn, _ := os.Open("model.json")
bn, _ = readwrite.ReadJSONModel(jsonIn)
jsonIn.Close()`}</code></pre>

        <h3>Step 4: Format Conversion via CLI</h3>
        <pre><code>{`# Convert BIF to XMLBIF
$ pgmgo convert --input asia.bif --from bif --to xmlbif --output asia.xmlbif

# Convert NET to BIF
$ pgmgo convert --input model.net --from net --to bif --output model.bif

# Convert XDSL to UAI (for competition submission)
$ pgmgo convert --input model.xdsl --from xdsl --to uai --output model.uai`}</code></pre>

        <h3>Step 5: Working with Tabular Data</h3>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/lib/tabgo"

// Read CSV data
data, err := tabgo.ReadCSV("observations.csv")
if err != nil {
    log.Fatal(err)
}
fmt.Printf("Rows: %d, Columns: %d\\n", data.NRows(), len(data.Columns()))

// Access columns
col := data.Column("Temperature")
fmt.Println("Unique values:", col.Unique())

// Write CSV
tabgo.WriteCSV(data, "output.csv")`}</code></pre>

        <h3>Step 6: Using Built-in Example Models</h3>
        <pre><code>{`import "github.com/asymmetric-effort/pgmgo/example_models"

// List all 25 available models
for _, name := range example_models.List() {
    fmt.Println(name)
}

// Load and export a model
bn, _ := example_models.Get("alarm")
f, _ := os.Create("alarm.bif")
readwrite.WriteBIF(f, bn)
f.Close()

// Load, convert to JSON, and serve via web API
bn, _ = example_models.Get("asia")
jsonOut, _ := os.Create("asia.json")
readwrite.WriteJSONModel(jsonOut, bn)
jsonOut.Close()`}</code></pre>
      </section>
    </div>
  );
}
