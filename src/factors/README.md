# factors

Package `factors` provides discrete and continuous factor representations for probabilistic graphical models.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/factors`

## Types

| Type | Description |
|------|-------------|
| `DiscreteFactor` | Factor (potential) over discrete variables with operations like product, marginalization, reduction |
| `TabularCPD` | Tabular conditional probability distribution wrapping a DiscreteFactor |
| `LinearGaussianCPD` | Linear Gaussian CPD: X | Parents ~ N(mean + sum(beta_i * Parent_i), variance) |
| `NoisyOR` | Compact binary CPD where each parent independently causes the child |
| `FunctionalCPD` | CPD defined by an arbitrary function over parent values |
| `JointProbabilityDistribution` | Joint distribution (non-negative, sums to 1) wrapping a DiscreteFactor |
| `FactorSet` | Unordered collection of discrete factors with product and lookup operations |
| `FactorDict` | Named collection of discrete factors |

## Key Functions

| Function | Description |
|----------|-------------|
| `FactorProduct(factors...)` | Multiply factors together, aligning on shared variables |
| `FactorDivide(f1, f2)` | Divide f1 by f2 (f2's variables must be a subset of f1's) |

## DiscreteFactor

```go
import "github.com/asymmetric-effort/pgmgo/src/factors"

// Create a factor over two binary variables
f, _ := factors.NewDiscreteFactor(
    []string{"A", "B"},
    []int{2, 2},
    []float64{0.5, 0.8, 0.1, 0.0},
)

// Query
vars := f.Variables()       // ["A", "B"]
card := f.Cardinality()     // [2, 2]
val := f.GetValue(map[string]int{"A": 0, "B": 1}) // 0.8

// Operations
marginalized, _ := f.Marginalize([]string{"B"}) // sum out B
reduced, _ := f.Reduce(map[string]int{"A": 0})  // fix A=0
f.Normalize()                                     // normalize to sum to 1

// Factor arithmetic
product, _ := factors.FactorProduct(f1, f2)
quotient, _ := factors.FactorDivide(f1, f2)
```

## TabularCPD

```go
// P(Grade | Difficulty, Intelligence)
// Grade has 3 states, Difficulty and Intelligence each have 2
cpd, _ := factors.NewTabularCPD(
    "Grade", 3,
    [][]float64{
        {0.3, 0.05, 0.9, 0.5},  // P(Grade=0 | parent configs)
        {0.4, 0.25, 0.08, 0.3}, // P(Grade=1 | parent configs)
        {0.3, 0.7, 0.02, 0.2},  // P(Grade=2 | parent configs)
    },
    []string{"Difficulty", "Intelligence"},
    []int{2, 2},
)

err := cpd.Validate()           // check columns sum to 1
f := cpd.ToFactor()             // convert to DiscreteFactor
v := cpd.Variable()             // "Grade"
ev := cpd.Evidence()            // ["Difficulty", "Intelligence"]
```

## LinearGaussianCPD

```go
// Y | X ~ N(1.0 + 0.5*X, 0.1)
cpd, _ := factors.NewLinearGaussianCPD(
    "Y",           // variable
    1.0,           // intercept (mean)
    []float64{0.5}, // betas
    0.1,           // variance
    []string{"X"}, // evidence (parents)
)

mu := cpd.ConditionalMean(map[string]float64{"X": 2.0}) // 2.0
logP := cpd.LogPDF(2.5, map[string]float64{"X": 2.0})
```

## NoisyOR

```go
// Binary child with two binary parents
nor, _ := factors.NewNoisyOR(
    "Alarm", 2,
    []string{"Fire", "Tamper"},
    []float64{0.01, 0.05}, // inhibition probabilities
    0.99,                    // leak probability
)

cpd := nor.ToTabularCPD()
```

## FunctionalCPD

```go
cpd, _ := factors.NewFunctionalCPD("Y", []string{"X"},
    func(parents map[string]float64) []float64 {
        if parents["X"] > 0.5 {
            return []float64{0.2, 0.8}
        }
        return []float64{0.9, 0.1}
    },
)

dist := cpd.GetDistribution(map[string]float64{"X": 0.7}) // [0.2, 0.8]
```

## JointProbabilityDistribution

```go
jpd, _ := factors.NewJointProbabilityDistribution(
    []string{"X", "Y"},
    []int{2, 2},
    []float64{0.2, 0.1, 0.3, 0.4},
)

marginal, _ := jpd.MarginalDistribution([]string{"X"})
indep := jpd.CheckIndependence("X", "Y", nil, 1e-4)
```

## FactorSet

```go
fs := factors.NewFactorSet(f1, f2, f3)
fs.Add(f4)
product, _ := fs.Product()
related := fs.GetFactorsOf("X") // factors containing variable X
```
