import { createElement, useHead } from "@asymmetric-effort/specifyjs";

export function Home() {
  useHead({
    title: "pgmgo — Probabilistic Graphical Models in Go",
    description: "A zero-dependency Go library for probabilistic graphical models, similar to pgmpy.",
    canonical: "https://pgmgo.asymmetric-effort.com/",
    og: {
      title: "pgmgo — Probabilistic Graphical Models in Go",
      description: "A zero-dependency Go library for probabilistic graphical models, similar to pgmpy.",
      url: "https://pgmgo.asymmetric-effort.com/",
    },
  });

  return (
    <div class="page">
      <section class="hero">
        <img src="/docs/img/logo.png" alt="pgmgo logo" class="hero-logo" />
        <h1>pgmgo</h1>
        <p class="hero-subtitle">Probabilistic Graphical Models in Go</p>
        <div class="badges">
          <span class="badge">v0.0.28</span>
          <span class="badge">Zero Dependencies</span>
          <span class="badge">Go</span>
          <span class="badge">MIT License</span>
          <span class="badge">4,700+ Tests</span>
          <span class="badge">10 I/O Formats</span>
          <span class="badge">pgmpy-Inspired</span>
        </div>
      </section>

      <section class="section">
        <h2>Installation</h2>
        <pre><code>go get github.com/asymmetric-effort/pgmgo</code></pre>
      </section>

      <section class="section">
        <h2>Quick Start</h2>
        <pre><code>{`package main

import (
    "fmt"
    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/factors"
    "github.com/asymmetric-effort/pgmgo/src/inference"
    "github.com/asymmetric-effort/pgmgo/src/models"
)

func main() {
    // Option 1: Load a built-in example model
    asia, _ := example_models.Get("asia")
    fmt.Println("Asia model:", len(asia.Nodes()), "nodes")

    // Option 2: Build a Bayesian Network from scratch
    bn := models.NewBayesianNetwork()
    bn.AddNode("Rain")
    bn.AddNode("Sprinkler")
    bn.AddNode("WetGrass")
    bn.AddEdge("Rain", "WetGrass")
    bn.AddEdge("Sprinkler", "WetGrass")

    // Add CPDs
    bn.SetStates("Rain", []string{"no", "yes"})
    bn.SetStates("Sprinkler", []string{"off", "on"})
    bn.SetStates("WetGrass", []string{"dry", "wet"})

    bn.SetCPD("Rain", factors.NewTabularCPD(
        "Rain", 2, []float64{0.8, 0.2}, nil, nil,
    ))

    // Run inference with Variable Elimination
    facs, _ := bn.ToMarkovFactors()
    ve := inference.NewVariableElimination(facs)
    result, _ := ve.Query(
        []string{"WetGrass"},
        map[string]int{"Rain": 1},
    )
    fmt.Println(result)
}`}</code></pre>
      </section>

      <section class="section">
        <h2>Features</h2>
        <div class="features-grid">
          <div class="feature-card">
            <h3>13 Model Types</h3>
            <p>BayesianNetwork, MarkovNetwork, DynamicBN, NaiveBayes, SEM, FactorGraph, JunctionTree, ClusterGraph, LinearGaussianBN, FunctionalBN, MarkovChain, DiscreteBN, DiscreteMarkovNetwork.</p>
          </div>
          <div class="feature-card">
            <h3>7 Inference Algorithms</h3>
            <p>Variable Elimination, Belief Propagation, MPLP, Approximate Inference, Causal Inference (do-calculus), DBN Inference, and MAP queries.</p>
          </div>
          <div class="feature-card">
            <h3>11+ Learning Algorithms</h3>
            <p>MLE, BayesianEstimator, EM, HillClimb, PC, GES, ExhaustiveSearch, TreeSearch, MMHC, Expert-in-the-Loop, IV Estimator, SEM Estimator, Mirror Descent, and LLM-assisted learning.</p>
          </div>
          <div class="feature-card">
            <h3>16+ CI Tests</h3>
            <p>ChiSquare, G-squared, FisherZ, Pearsonr, GCM, Hotelling-Lawley, and more across discrete, continuous, multivariate, and tree-based categories.</p>
          </div>
          <div class="feature-card">
            <h3>13 Scoring Functions</h3>
            <p>BIC, AIC, BDeu, BDs, K2, Log-Likelihood, Gaussian scores, and Conditional Gaussian scores for structure learning.</p>
          </div>
          <div class="feature-card">
            <h3>10 I/O Formats</h3>
            <p>BIF, XMLBIF, NET, UAI, XDSL, PomdpX, XBN, CSV, JSON, XML model serialization with full read/write support.</p>
          </div>
          <div class="feature-card">
            <h3>25 Example Models</h3>
            <p>Built-in models including Asia, Alarm, Cancer, Student, Sachs, Insurance, Hailfinder, Hepar2, Pigs, and more. 13 with full CPDs, 12 structure-only.</p>
          </div>
          <div class="feature-card">
            <h3>GPU Compute Backend</h3>
            <p>Optional GPU acceleration via the <code>lib/gpu</code> package for compute-intensive operations on large networks.</p>
          </div>
          <div class="feature-card">
            <h3>LLM Integration</h3>
            <p>Expert-in-the-loop structure learning with LLM client support for AI-assisted model construction and knowledge elicitation.</p>
          </div>
          <div class="feature-card">
            <h3>Zero Dependencies</h3>
            <p>Built entirely in Go with custom implementations of numpy (numgo), scipy (scigo), networkx (graphgo), and pandas (tabgo).</p>
          </div>
        </div>
      </section>
    </div>
  );
}
