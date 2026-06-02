# pgmgo Examples

This directory contains runnable Go examples demonstrating the core features of pgmgo.

## Examples

### basic_bn

Builds the classic Student Bayesian network (D, I, G, L, S), adds CPDs, validates
the model, and runs a variable elimination query to compute P(G | D=0, I=1).

```
go run ./examples/basic_bn
```

### structure_learning

Generates synthetic data from a known Bayesian network using forward sampling,
then runs HillClimbSearch with BIC scoring to learn the structure from data.
Compares learned edges against the true structure.

```
go run ./examples/structure_learning
```

### causal_inference

Builds a Bayesian network with confounding and demonstrates the difference
between observational conditioning P(Y | X=1) and interventional queries
P(Y | do(X=1)). Computes the Average Treatment Effect (ATE).

```
go run ./examples/causal_inference
```

### sampling

Builds the Student network and draws forward samples and likelihood-weighted
samples with evidence. Compares empirical marginals against exact inference
results.

```
go run ./examples/sampling
```

### bif_io

Creates a Bayesian network, writes it to a BIF (Bayesian Interchange Format)
file, reads it back, and verifies that the round-trip preserves the model
structure and parameters.

```
go run ./examples/bif_io
```

## Example Models

The `example_models/` package provides factory functions that return fully
parameterized, well-known Bayesian networks from the literature. See
`example_models/models.go` for details.
