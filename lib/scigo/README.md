# scigo

Package `scigo` provides scientific computing primitives including statistical distributions, hypothesis testing, optimization, sparse matrices, and special functions. It serves as the SciPy equivalent for pgmgo.

```
import "github.com/asymmetric-effort/pgmgo/lib/scigo"
```

## Probability Distributions

All continuous distributions implement the `Distribution` interface:

```go
type Distribution interface {
    PDF(x float64) float64    // probability density function
    CDF(x float64) float64    // cumulative distribution function
    PPF(p float64) float64    // percent point function (inverse CDF)
    LogPDF(x float64) float64 // log of the PDF
    Mean() float64
    Var() float64
}
```

### Continuous Distributions

```go
// Normal (Gaussian)
n := scigo.NewNormal(0, 1)    // mu=0, sigma=1
n.PDF(0.5)
n.CDF(1.96)                   // ~0.975
n.PPF(0.975)                  // ~1.96
n.LogPDF(0.5)
n.Mean()                      // 0
n.Var()                       // 1
n.Sample(rng, 1000)           // generate 1000 samples

// Chi-Squared
chi2 := scigo.NewChiSquared(5.0)  // df=5
chi2.PDF(3.0)
chi2.CDF(11.07)
chi2.PPF(0.95)
chi2.SurvivalFunction(11.07)      // 1 - CDF (p-value)

// Student's t
t := scigo.NewTDistribution(10.0) // df=10
t.PDF(0)
t.CDF(2.228)
t.PPF(0.975)
t.SurvivalFunction(2.228)

// Beta
beta := scigo.NewBeta(2, 5)       // alpha=2, beta=5
beta.PDF(0.3)
beta.CDF(0.5)
beta.PPF(0.5)

// Gamma
gamma := scigo.NewGamma(2.0, 1.0) // shape=2, scale=1
gamma.PDF(1.0)
gamma.CDF(3.0)

// Exponential
exp := scigo.NewExponential(0.5)   // rate=0.5
exp.PDF(1.0)
exp.CDF(2.0)
exp.PPF(0.5)

// Uniform
u := scigo.NewUniform(0, 1)       // [low, high]
u.PDF(0.5)
u.CDF(0.75)
u.PPF(0.5)
```

### Discrete Distributions

```go
// Poisson
pois := scigo.NewPoisson(3.0)     // lambda=3
pois.PMF(2)
pois.CDF(4)
pois.Mean()  // 3
pois.Var()   // 3

// Binomial
binom := scigo.NewBinomial(10, 0.5) // n=10, p=0.5
binom.PMF(5)
binom.CDF(7)
binom.Mean()  // 5
binom.Var()   // 2.5
```

## Hypothesis Testing

```go
// Pearson chi-squared goodness-of-fit test
observed := []float64{16, 18, 16, 14, 12, 12}
expected := []float64{16, 16, 16, 16, 8, 16}
stat, pval := scigo.ChiSquareTest(observed, expected)

// G-test (log-likelihood ratio test)
stat, pval := scigo.GTest(observed, expected)

// Power divergence test (generalizes chi-squared and G-test)
// lambda=1: Pearson chi-squared
// lambda=0: G-test (log-likelihood)
// lambda=-1: modified log-likelihood
// lambda=-0.5: Freeman-Tukey
// lambda=2/3: Cressie-Read
stat, pval := scigo.PowerDivergenceTest(observed, expected, 1.0)
```

## Correlation

```go
x := []float64{1, 2, 3, 4, 5}
y := []float64{2, 4, 5, 4, 5}

// Pearson correlation with two-tailed p-value
r, pval := scigo.PearsonCorrelation(x, y)

// Partial correlation: correlation between columns x and y controlling for z
data := [][]float64{
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
    {10, 11, 12},
}
r, pval := scigo.PartialCorrelation(data, 0, 1, []int{2})

// Fisher Z-transform of a correlation coefficient
z := scigo.FisherZTransform(0.8, 50)
// Standard error: SE = 1/sqrt(n-3)
```

## Optimization

### Multivariate Minimization

```go
// Nelder-Mead simplex algorithm
rosenbrock := func(x []float64) float64 {
    return (1-x[0])*(1-x[0]) + 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0])
}
result, err := scigo.Minimize(rosenbrock, []float64{0, 0}, "nelder-mead")
// result.X       - solution vector
// result.Fun     - objective value at solution
// result.Success - whether optimizer converged
// result.Nit     - number of iterations

// Gradient descent with numerical gradients and backtracking line search
result, err := scigo.Minimize(rosenbrock, []float64{0, 0}, "gradient-descent")
```

### Scalar Minimization (Brent's Method)

```go
f := func(x float64) float64 { return (x - 3) * (x - 3) }
result, err := scigo.MinimizeScalar(f, [2]float64{0, 10})
// result.X   ~= 3.0
// result.Fun ~= 0.0
```

### Root Finding (Brent's Method)

```go
f := func(x float64) float64 { return x*x - 4 }
root, err := scigo.RootScalar(f, [2]float64{0, 5})
// root ~= 2.0
```

## Sparse Matrices

Three sparse matrix formats are supported: COO (coordinate/triplet), CSR (compressed sparse row), and CSC (compressed sparse column).

### COO (Coordinate Format)

```go
rows := []int{0, 1, 2}
cols := []int{0, 1, 2}
vals := []float64{1, 2, 3}
coo, err := scigo.NewCOO(rows, cols, vals, [2]int{3, 3})

coo.Get(1, 1)    // 2.0
coo.Set(0, 2, 5) // add entry
coo.Shape()      // [3, 3]
coo.NNZ()        // number of stored entries

csr := coo.ToCSR()       // convert to CSR
dense := coo.ToDense()   // convert to dense 2D slice
```

### CSR (Compressed Sparse Row)

```go
indptr := []int{0, 2, 3, 4}
indices := []int{0, 2, 1, 2}
data := []float64{1, 3, 2, 4}
csr, err := scigo.NewCSR(indptr, indices, data, [2]int{3, 3})

csr.Get(0, 2)          // 3.0
csr.Row(0)             // column indices and values of row 0
csr.Shape()            // [3, 3]
csr.NNZ()              // 4

y := csr.MulVec(x)     // sparse matrix-vector product
d := csr.MulDense(mat) // sparse-dense matrix product
t := csr.Transpose()   // transpose (returns CSR)
coo := csr.ToCOO()     // convert to COO
dense := csr.ToDense() // convert to dense
```

### CSC (Compressed Sparse Column)

```go
indptr := []int{0, 1, 2, 4}
indices := []int{0, 1, 0, 2}
data := []float64{1, 2, 3, 4}
csc, err := scigo.NewCSC(indptr, indices, data, [2]int{3, 3})

csc.Get(0, 2)          // 3.0
csc.Col(2)             // row indices and values of column 2
csr := csc.ToCSR()     // convert to CSR
dense := csc.ToDense() // convert to dense
```

### Dense-to-Sparse Conversion

```go
dense := [][]float64{{1, 0, 3}, {0, 2, 0}, {0, 0, 4}}
csr := scigo.DenseToCSR(dense)
coo := scigo.DenseToCOO(dense)
```

## Special Functions

```go
scigo.Gammaln(5.0)       // ln(Gamma(x))
scigo.Digamma(5.0)       // psi function (logarithmic derivative of Gamma)

scigo.Erf(1.0)           // error function
scigo.Erfinv(0.5)        // inverse error function

scigo.Logsumexp([]float64{1, 2, 3}) // numerically stable log(sum(exp(x)))

// Regularized incomplete gamma function P(a, x)
scigo.RegularizedIncompleteGamma(2.0, 3.0)

// Regularized incomplete beta function I_x(a, b)
scigo.RegularizedIncompleteBeta(0.5, 2.0, 3.0)
```

## API Summary

| Category | Types / Functions |
|---|---|
| **Distribution Interface** | `Distribution` (PDF, CDF, PPF, LogPDF, Mean, Var) |
| **Continuous Distributions** | `Normal`, `ChiSquared`, `TDistribution`, `Beta`, `Gamma`, `Exponential`, `Uniform` |
| **Discrete Distributions** | `Poisson` (PMF, CDF), `Binomial` (PMF, CDF) |
| **Hypothesis Testing** | `ChiSquareTest`, `GTest`, `PowerDivergenceTest` |
| **Correlation** | `PearsonCorrelation`, `PartialCorrelation`, `FisherZTransform` |
| **Optimization** | `Minimize` (nelder-mead, gradient-descent), `MinimizeScalar`, `RootScalar` |
| **Optimization Results** | `OptResult`, `ScalarResult` |
| **Sparse Matrices** | `COO`, `CSR`, `CSC`, `NewCOO`, `NewCSR`, `NewCSC`, `DenseToCSR`, `DenseToCOO` |
| **Special Functions** | `Gammaln`, `Digamma`, `Erf`, `Erfinv`, `Logsumexp`, `RegularizedIncompleteGamma`, `RegularizedIncompleteBeta` |
