<p align="center">
  <img src="docs/img/logo.png" alt="pgmgo logo" width="150">
</p>

<h1 align="center">pgmgo</h1>

<p align="center">
  <strong>Probabilistic Graphical Models in Go</strong><br>
  A zero-dependency Go library with 100% pgmpy feature parity
</p>

<p align="center">
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="https://go.dev"><img src="https://img.shields.io/badge/Go-1.26+-00ADD8.svg" alt="Go Version"></a>
  <img src="https://img.shields.io/badge/dependencies-zero-brightgreen.svg" alt="Zero Dependencies">
  <a href="https://github.com/asymmetric-effort/pgmgo/actions"><img src="https://github.com/asymmetric-effort/pgmgo/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
</p>

---

## Overview

pgmgo is a pure Go library for probabilistic graphical models. It provides a
complete implementation of the algorithms and data structures found in
[pgmpy](https://pgmpy.org), the popular Python library, ported to idiomatic Go
with full feature parity.

pgmgo has **zero third-party dependencies**. Every numerical, graph-theoretic,
and tabular operation is implemented from scratch in Go within the project's own
internal libraries. This eliminates supply-chain risk and simplifies deployment
to a single static binary.

## Installation

```bash
go get github.com/asymmetric-effort/pgmgo
```

Requires Go 1.26 or later.

## Quick Start

Build a Bayesian network, add CPDs, and run a variable elimination query:

```go
package main

import (
    "fmt"
    "log"

    "github.com/asymmetric-effort/pgmgo/example_models"
    "github.com/asymmetric-effort/pgmgo/src/inference"
)

func main() {
    // Build the classic Student network (D, I, G, L, S).
    bn := example_models.Student()

    if err := bn.CheckModel(); err != nil {
        log.Fatalf("Model validation failed: %v", err)
    }

    // Convert to Markov factors for inference.
    markovFactors, _ := bn.ToMarkovFactors()

    // Query: P(G | D=Easy, I=High)
    ve := inference.NewVariableElimination(markovFactors)
    evidence := map[string]int{"D": 0, "I": 1}
    result, _ := ve.Query([]string{"G"}, evidence)

    for i, state := range bn.GetStates("G") {
        prob := result.GetValue(map[string]int{"G": i})
        fmt.Printf("P(G=%s | D=Easy, I=High) = %.4f\n", state, prob)
    }
}
```

## Features

### Models (13 types)

- Bayesian Network
- Discrete Bayesian Network
- Markov Network
- Discrete Markov Network
- Dynamic Bayesian Network
- Factor Graph
- Cluster Graph
- Junction Tree
- Naive Bayes
- Markov Chain
- Linear Gaussian Bayesian Network
- Functional Bayesian Network
- Structural Equation Model (SEM)

### Inference (7 algorithms)

- Variable Elimination
- Belief Propagation
- Max-Product Linear Programming (MPLP)
- Approximate Inference
- Causal Inference (do-calculus, backdoor/frontdoor adjustment)
- Dynamic Bayesian Network Inference
- MAP and MPE queries

### Learning (11 algorithms)

- Maximum Likelihood Estimation (MLE)
- Bayesian Estimation
- Expectation-Maximization (EM)
- Linear Gaussian MLE
- SEM Estimation
- Hill Climb Search
- Exhaustive Search
- PC Algorithm
- GES (Greedy Equivalence Search)
- MMHC (Max-Min Hill Climbing)
- Tree Search

### Additional Learning

- Expert Knowledge integration
- Expert-in-the-Loop learning
- LLM-assisted structure learning
- Instrumental Variable (IV) estimation
- Mirror Descent optimization
- Marginal Estimation

### Sampling

- Forward / Bayesian Sampling
- Gibbs Sampling

### Causal Inference

- do-calculus interventions
- Average Treatment Effect (ATE)
- Backdoor and frontdoor adjustment

### Structure Scoring

- BIC, BDeu, K2, and other scoring functions
- Constraint-based and score-based structure learning

### File I/O (7 formats)

- BIF (Bayesian Interchange Format)
- XMLBIF
- UAI
- NET
- XBN
- XDSL
- POMDPX

## Internal Libraries

pgmgo includes five internal libraries that replace common third-party
dependencies:

| Library    | Purpose                                                        |
|------------|----------------------------------------------------------------|
| **numgo**  | N-dimensional array operations, broadcasting, linear algebra   |
| **scigo**  | Scientific computing: statistics, probability distributions    |
| **graphgo**| Graph data structures, algorithms, d-separation, moralization  |
| **tabgo**  | Tabular data (DataFrames), CSV I/O, filtering, aggregation     |
| **gpu**    | Compute backend abstraction (CPU fallback included)            |

## Project Structure

```
pgmgo/
  cmd/pgmgo/         CLI entry point
  src/
    models/           Graphical model types
    inference/        Inference algorithms
    learning/         Parameter and structure learning
    sampling/         Sampling methods
    readwrite/        File format readers and writers
    factors/          Factor (CPD/JPD) representations
    metrics/          Scoring and evaluation metrics
    structure_score/  Structure learning scores
    identification/   Causal identification
    independencies/   Independence testing
    utils/            Shared utilities
  lib/
    numgo/            Numerical arrays
    scigo/            Scientific computing
    graphgo/          Graph algorithms
    tabgo/            Tabular data
    gpu/              Compute backend
  examples/           Runnable example programs
    datasets/         Built-in datasets
  example_models/     Pre-built canonical networks
  tests/              Cross-validation fixtures
  website/            Project website source
  docs/               Documentation assets
```

## Examples

The [`examples/`](examples/) directory contains runnable programs demonstrating
core features:

| Example                | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `basic_bn`             | Build a Bayesian network and run variable elimination         |
| `structure_learning`   | Learn network structure from data with HillClimbSearch        |
| `causal_inference`     | Observational vs. interventional queries and ATE              |
| `sampling`             | Forward sampling and likelihood-weighted sampling             |
| `bif_io`              | Round-trip BIF file serialization and deserialization          |

Run any example with:

```bash
go run ./examples/basic_bn
```

## Datasets

Built-in datasets for experimentation are available in
[`examples/datasets/`](examples/datasets/).

## Documentation

- [Project Website](website/) (local build with `npm run dev` in `website/`)
- [GoDoc](https://pkg.go.dev/github.com/asymmetric-effort/pgmgo) (once published)
- [Examples](examples/)

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on development workflow, testing, commit conventions, and the
zero-dependency policy.

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md). Do not open
public issues for security concerns.

## License

pgmgo is released under the [MIT License](LICENSE.txt).

Copyright (c) 2026 Asymmetric Effort, LLC.
