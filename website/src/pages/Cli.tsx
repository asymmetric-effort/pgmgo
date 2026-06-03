import { createElement, useHead } from "@asymmetric-effort/specifyjs";

export function Cli() {
  useHead({
    title: "CLI Reference — pgmgo",
    description: "Complete command-line reference for all 10 pgmgo commands.",
    canonical: "https://pgmgo.asymmetric-effort.com/#/cli",
  });

  return (
    <div class="page">
      <h1>CLI Reference</h1>

      <section class="section">
        <h2>Overview</h2>
        <p>
          The <code>pgmgo</code> CLI provides 10 commands for working with probabilistic graphical
          models from the command line. It supports model validation, inference queries, structure
          learning, parameter fitting, sampling, format conversion, model comparison, and causal
          do-calculus queries.
        </p>
        <pre><code>pgmgo [command] [options]</code></pre>
      </section>

      <section class="section">
        <h2>Commands Summary</h2>
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
      </section>

      <section class="section">
        <h2>validate</h2>
        <p>Validate a BIF model file. Parses the file and checks that the model is well-formed (all CPDs are consistent with the graph structure).</p>
        <pre><code>pgmgo validate &lt;file&gt;</code></pre>
        <h3>Example</h3>
        <pre><code>{`$ pgmgo validate asia.bif
model asia.bif is valid (8 nodes, 8 edges)

$ pgmgo validate broken.bif
error: model validation failed: CPD for "Lung" has inconsistent cardinality`}</code></pre>
      </section>

      <section class="section">
        <h2>query</h2>
        <p>Run a probabilistic inference query on a BIF model. Computes posterior probabilities for query variables given observed evidence.</p>
        <pre><code>{`pgmgo query <file> --variables V1,V2 [--evidence E1=v1,E2=v2] [--method ve|bp|approx]`}</code></pre>
        <h3>Flags</h3>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--variables</code></td><td>(required)</td><td>Comma-separated query variables</td></tr>
            <tr><td><code>--evidence</code></td><td>(none)</td><td>Comma-separated evidence as KEY=VALUE pairs</td></tr>
            <tr><td><code>--method</code></td><td><code>ve</code></td><td>Inference method: <code>ve</code> (Variable Elimination), <code>bp</code> (Belief Propagation), <code>approx</code> (approximate sampling, 10,000 samples)</td></tr>
          </tbody>
        </table>
        <h3>Examples</h3>
        <pre><code>{`# P(Dyspnea) using Variable Elimination
$ pgmgo query asia.bif --variables Dyspnea

# P(Lung, Bronc | Smoker=1) using Belief Propagation
$ pgmgo query asia.bif --variables Lung,Bronc --evidence Smoker=1 --method bp

# Approximate inference
$ pgmgo query asia.bif --variables Dyspnea --evidence Smoker=1 --method approx`}</code></pre>
      </section>

      <section class="section">
        <h2>map</h2>
        <p>Find the MAP (Maximum A Posteriori) assignment -- the most likely values for query variables given evidence.</p>
        <pre><code>{`pgmgo map <file> --variables V1,V2 [--evidence E1=v1,E2=v2]`}</code></pre>
        <h3>Flags</h3>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--variables</code></td><td>(required)</td><td>Comma-separated query variables</td></tr>
            <tr><td><code>--evidence</code></td><td>(none)</td><td>Comma-separated evidence as KEY=VALUE pairs</td></tr>
          </tbody>
        </table>
        <h3>Example</h3>
        <pre><code>{`$ pgmgo map asia.bif --variables Lung,Bronc --evidence Smoker=1
MAP assignment:
  Bronc = 1
  Lung = 0`}</code></pre>
      </section>

      <section class="section">
        <h2>learn</h2>
        <p>Learn a Bayesian network structure from CSV data. After learning the structure, MLE parameters are automatically fitted. The result is written as a BIF file.</p>
        <pre><code>{`pgmgo learn --data <csv> --method <method> --output <bif> [--score bic|bdeu|k2] [--significance 0.05]`}</code></pre>
        <h3>Flags</h3>
        <table>
          <thead>
            <tr><th>Flag</th><th>Default</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--data</code></td><td>(required)</td><td>Path to CSV data file</td></tr>
            <tr><td><code>--method</code></td><td><code>hillclimb</code></td><td>Learning method: <code>hillclimb</code>, <code>pc</code>, <code>ges</code>, <code>exhaustive</code>, <code>tree</code></td></tr>
            <tr><td><code>--score</code></td><td><code>bic</code></td><td>Scoring function for score-based methods: <code>bic</code>, <code>bdeu</code>, <code>k2</code></td></tr>
            <tr><td><code>--significance</code></td><td><code>0.05</code></td><td>Significance level for constraint-based methods (PC)</td></tr>
            <tr><td><code>--output</code></td><td>(required)</td><td>Output BIF file path</td></tr>
          </tbody>
        </table>
        <h3>Examples</h3>
        <pre><code>{`# Hill-climb with BIC scoring (default)
$ pgmgo learn --data observations.csv --output learned.bif
learned structure with 5 nodes, 6 edges -> learned.bif

# PC algorithm with chi-square test
$ pgmgo learn --data observations.csv --method pc --significance 0.01 --output learned_pc.bif

# GES with BDeu scoring
$ pgmgo learn --data observations.csv --method ges --score bdeu --output learned_ges.bif

# Exhaustive search (small networks only)
$ pgmgo learn --data small_data.csv --method exhaustive --output optimal.bif

# Tree search
$ pgmgo learn --data observations.csv --method tree --output tree_model.bif`}</code></pre>
      </section>

      <section class="section">
        <h2>fit</h2>
        <p>Fit parameters (CPDs) to an existing network structure using observed data. Takes an existing BIF file (structure) and a CSV data file, and outputs a new BIF file with learned parameters.</p>
        <pre><code>{`pgmgo fit --model <bif> --data <csv> --method <method> --output <bif>`}</code></pre>
        <h3>Flags</h3>
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
        <h3>Examples</h3>
        <pre><code>{`# Maximum Likelihood Estimation
$ pgmgo fit --model structure.bif --data train.csv --output fitted_mle.bif
fitted parameters for 8 nodes -> fitted_mle.bif

# Bayesian estimation with BDeu prior
$ pgmgo fit --model structure.bif --data train.csv --method bayesian --output fitted_bayes.bif

# EM for incomplete data
$ pgmgo fit --model structure.bif --data incomplete.csv --method em --output fitted_em.bif`}</code></pre>
      </section>

      <section class="section">
        <h2>sample</h2>
        <p>Generate samples from a Bayesian network model. Supports forward sampling, rejection sampling (with evidence), and Gibbs sampling.</p>
        <pre><code>{`pgmgo sample --model <bif> --output <csv> [--n 100] [--method forward|rejection|gibbs] [--evidence E1=v1] [--seed 42]`}</code></pre>
        <h3>Flags</h3>
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
        <h3>Examples</h3>
        <pre><code>{`# Forward sampling (100 samples)
$ pgmgo sample --model asia.bif --output samples.csv
generated 100 samples -> samples.csv

# 5000 samples with fixed seed for reproducibility
$ pgmgo sample --model asia.bif --n 5000 --seed 42 --output samples.csv

# Rejection sampling with evidence
$ pgmgo sample --model asia.bif --n 500 --method rejection --evidence Smoker=1 --output smoker_samples.csv

# Gibbs sampling with evidence
$ pgmgo sample --model asia.bif --n 1000 --method gibbs --evidence Smoker=1 --output gibbs_samples.csv`}</code></pre>
      </section>

      <section class="section">
        <h2>info</h2>
        <p>Print a summary of a BIF model: node list with states, edge list, and CPD summary.</p>
        <pre><code>pgmgo info &lt;file&gt;</code></pre>
        <h3>Example</h3>
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
      </section>

      <section class="section">
        <h2>convert</h2>
        <p>Convert a model between supported file formats.</p>
        <pre><code>{`pgmgo convert --input <file> --from <format> --to <format> --output <file>`}</code></pre>
        <h3>Flags</h3>
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
        <h3>Examples</h3>
        <pre><code>{`# BIF to XMLBIF
$ pgmgo convert --input asia.bif --from bif --to xmlbif --output asia.xmlbif
converted asia.bif (bif) -> asia.xmlbif (xmlbif)

# NET to UAI
$ pgmgo convert --input model.net --from net --to uai --output model.uai

# XDSL to BIF
$ pgmgo convert --input model.xdsl --from xdsl --to bif --output model.bif`}</code></pre>
      </section>

      <section class="section">
        <h2>compare</h2>
        <p>Compare two network structures using standard metrics: Structural Hamming Distance (SHD), adjacency confusion matrix, and orientation confusion matrix.</p>
        <pre><code>{`pgmgo compare --true <bif> --estimated <bif>`}</code></pre>
        <h3>Flags</h3>
        <table>
          <thead>
            <tr><th>Flag</th><th>Description</th></tr>
          </thead>
          <tbody>
            <tr><td><code>--true</code></td><td>Ground truth model BIF file</td></tr>
            <tr><td><code>--estimated</code></td><td>Estimated/learned model BIF file</td></tr>
          </tbody>
        </table>
        <h3>Example</h3>
        <pre><code>{`$ pgmgo compare --true ground_truth.bif --estimated learned.bif
Structural Hamming Distance (SHD): 3

Adjacency confusion matrix:
  TP=6  FP=1  TN=15  FN=2

Orientation confusion matrix:
  TP=5  FP=2  TN=14  FN=3`}</code></pre>
      </section>

      <section class="section">
        <h2>do</h2>
        <p>Perform a causal do-calculus query. Computes the interventional distribution P(Y | do(X=x)) using CausalInference.</p>
        <pre><code>{`pgmgo do <file> --intervention X=v --query Y [--evidence E1=v1]`}</code></pre>
        <h3>Flags</h3>
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
        <h3>Examples</h3>
        <pre><code>{`# P(Dyspnea | do(Smoker=1))
$ pgmgo do asia.bif --intervention Smoker=1 --query Dyspnea
Dyspnea	P
0	0.304000
1	0.696000

# With additional evidence
$ pgmgo do asia.bif --intervention Smoker=1 --query Lung --evidence VisitAsia=0`}</code></pre>
      </section>

      <section class="section">
        <h2>Exit Codes</h2>
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
    </div>
  );
}
