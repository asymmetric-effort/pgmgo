# models

Package `models` provides graphical model structures including Bayesian networks, Markov networks, factor graphs, junction trees, and structural equation models.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/models`

## Types

| Type | Description |
|------|-------------|
| `BayesianNetwork` | DAG with TabularCPDs; supports inference, prediction, do-calculus |
| `DiscreteBayesianNetwork` | BayesianNetwork with discrete-specific validation and forward sampling |
| `LinearGaussianBayesianNetwork` | BN with LinearGaussianCPDs for continuous variables |
| `FunctionalBayesianNetwork` | BN with arbitrary FunctionalCPDs |
| `DynamicBayesianNetwork` | Two-time-slice BN (2TBN) for temporal modeling |
| `NaiveBayes` | Star-topology classifier (class -> features) |
| `MarkovNetwork` | Undirected graphical model (MRF) with discrete factor potentials |
| `DiscreteMarkovNetwork` | MarkovNetwork with additional discrete validation |
| `FactorGraph` | Bipartite graph of variable and factor nodes |
| `ClusterGraph` | Graph of variable clusters connected by separation sets |
| `JunctionTree` | Clique tree for exact inference, built from BN or MN |
| `MarkovChain` | Discrete-time finite-state Markov chain |
| `SEM` | Linear Structural Equation Model with OLS fitting |

## BayesianNetwork

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/models"
    "github.com/asymmetric-effort/pgmgo/src/factors"
)

bn := models.NewBayesianNetwork()
bn.AddNode("Rain")
bn.AddNode("Sprinkler")
bn.AddNode("Wet")
bn.AddEdge("Rain", "Wet")
bn.AddEdge("Sprinkler", "Wet")

// Add CPDs
rainCPD, _ := factors.NewTabularCPD("Rain", 2,
    [][]float64{{0.8}, {0.2}}, nil, nil)
bn.AddCPD(rainCPD)

// Validate
err := bn.CheckModel()

// Do-calculus intervention
mutilated, _ := bn.Do(map[string]int{"Rain": 1})

// Get probability of a state assignment
prob, _ := bn.GetStateProbability(map[string]int{"Rain": 1, "Wet": 1})

// Save/Load BIF format
bn.Save("model.bif")
loaded, _ := models.LoadBayesianNetwork("model.bif")

// Generate random network
randBN, _ := models.GetRandomBayesianNetwork(5, 4, 2)
```

## LinearGaussianBayesianNetwork

```go
lgbn := models.NewLinearGaussianBayesianNetwork()
lgbn.AddNode("X")
lgbn.AddNode("Y")
lgbn.AddEdge("X", "Y")

cpd, _ := factors.NewLinearGaussianCPD("Y", 1.0, []float64{0.5}, 0.1, []string{"X"})
lgbn.AddLinearGaussianCPD(cpd)

// Fit from data
lgbn.Fit(data)

// Simulate samples
samples, _ := lgbn.Simulate(1000)

// Joint Gaussian parameters
mu, sigma, _ := lgbn.ToJointGaussian()
```

## NaiveBayes

```go
nb, _ := models.NewNaiveBayes("Class", []string{"F1", "F2", "F3"})
nb.Fit(trainingData)
predictions, _ := nb.Predict(testData)
probabilities, _ := nb.PredictProbability(testData)
```

## MarkovNetwork

```go
mn := models.NewMarkovNetwork()
mn.AddNode("A")
mn.AddNode("B")
mn.AddEdge("A", "B")

factor, _ := factors.NewDiscreteFactor([]string{"A", "B"}, []int{2, 2},
    []float64{0.5, 0.8, 0.1, 0.0})
mn.AddFactor(factor)

z, _ := mn.GetPartitionFunction()
jt, _ := mn.ToJunctionTree()
```

## SEM

```go
sem := models.NewSEM()
sem.AddEquation("Y", []string{"X"}, []float64{0.5}, 1.0, 0.1)

// Or parse from lavaan syntax
sem2, _ := models.FromLavaan("Y ~ X1 + X2\nZ ~ Y")

// Fit from data
sem.Fit(data)

// Implied covariance matrix
sigma, _ := sem.ImpliedCovarianceMatrix()

// Generate samples
samples, _ := sem.GenerateSamples(1000)

// LISREL matrices
lisrel, _ := sem.ToLisrel()
```

## DynamicBayesianNetwork

```go
dbn := models.NewDynamicBayesianNetwork()
dbn.Initial().AddNode("X")
dbn.Transition().AddNode("X")
dbn.Transition().AddNode("X_prev")

ifaceNodes := dbn.GetInterfaceNodes()
```

## MarkovChain

```go
mc, _ := models.NewMarkovChain(
    [][]float64{{0.7, 0.3}, {0.4, 0.6}},
    []string{"Sunny", "Rainy"},
)

pi, _ := mc.StationaryDistribution()
samples, _ := mc.Sample(100, 0, 42)
isErg := mc.IsErgodic()
```

## JunctionTree

```go
jt, _ := bn.ToJunctionTree()
cliques := jt.Cliques()
seps := jt.SeparatorSets()
err := jt.CheckModel() // running intersection property
```
