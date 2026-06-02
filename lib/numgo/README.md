# numgo

Package `numgo` provides n-dimensional array operations, linear algebra, and numerical primitives. It serves as the NumPy equivalent for pgmgo.

```
import "github.com/asymmetric-effort/pgmgo/lib/numgo"
```

## NDArray Creation

```go
// Zero-initialized array
a := numgo.Zeros(3, 4)

// All ones
b := numgo.Ones(2, 3)

// Fill with a constant
c := numgo.Full(7.0, 2, 2)

// Identity matrix
eye := numgo.Eye(3)

// From flat slice (1-D)
v := numgo.FromSlice([]float64{1, 2, 3, 4})

// From 2-D slice
m := numgo.FromSlice2D([][]float64{
    {1, 2, 3},
    {4, 5, 6},
})

// General constructor with explicit shape and data
arr := numgo.NewNDArray([]int{2, 3}, []float64{1, 2, 3, 4, 5, 6})
```

## NDArray Properties and Manipulation

```go
a := numgo.Ones(3, 4)

a.Shape()   // []int{3, 4}
a.Ndim()    // 2
a.Size()    // 12
a.Data()    // []float64 copy of underlying data

a.Get(1, 2)         // element at row 1, col 2
a.Set(9.0, 1, 2)    // set element

a.Reshape(4, 3)     // new array with different shape, same data
a.Flatten()          // 1-D copy
a.Copy()             // deep copy
a.T()                // transpose (reverses axes)
```

## Arithmetic Operations

All element-wise operations support broadcasting between compatible shapes.

```go
a := numgo.FromSlice2D([][]float64{{1, 2}, {3, 4}})
b := numgo.FromSlice2D([][]float64{{5, 6}, {7, 8}})

numgo.Add(a, b)          // element-wise addition (broadcasts)
numgo.Sub(a, b)          // element-wise subtraction
numgo.Mul(a, b)          // element-wise multiplication
numgo.Div(a, b)          // element-wise division

numgo.AddScalar(a, 10)   // add scalar to every element
numgo.SubScalar(a, 1)
numgo.MulScalar(a, 2)
numgo.DivScalar(a, 3)
```

## Reductions

```go
a := numgo.FromSlice2D([][]float64{{1, 2}, {3, 4}})

numgo.Sum(a)         // sum all elements -> [10]
numgo.Sum(a, 0)      // sum along axis 0 -> [4, 6]
numgo.Sum(a, 1)      // sum along axis 1 -> [3, 7]

numgo.Prod(a)        // product of all elements
numgo.Max(a)         // maximum value
numgo.ArgMax(a, 1)   // index of max along axis
```

## Broadcasting

NumPy-style broadcasting rules: dimensions are compared right-to-left; size-1 dimensions are stretched to match.

```go
shape, err := numgo.BroadcastShapes([]int{3, 1}, []int{1, 4})
// shape = []int{3, 4}, err = nil

a := numgo.Ones(3, 1)
b, err := numgo.BroadcastTo(a, []int{3, 4})
```

## Linear Algebra (Einsum)

Einstein summation supports arbitrary subscript notation with optimized paths for common patterns.

```go
a := numgo.FromSlice2D([][]float64{{1, 2}, {3, 4}})
b := numgo.FromSlice2D([][]float64{{5, 6}, {7, 8}})

// Matrix multiply: C[i,k] = sum_j A[i,j] * B[j,k]
c, _ := numgo.Einsum("ij,jk->ik", a, b)

// Dot product
u := numgo.FromSlice([]float64{1, 2, 3})
v := numgo.FromSlice([]float64{4, 5, 6})
dot, _ := numgo.Einsum("i,i->", u, v)

// Outer product
outer, _ := numgo.Einsum("i,j->ij", u, v)

// Trace
trace, _ := numgo.Einsum("ii->", a)

// Row sums, column sums
rowSums, _ := numgo.Einsum("ij->i", a)
colSums, _ := numgo.Einsum("ij->j", a)

// Batch matrix multiply
// batchC, _ := numgo.Einsum("bij,bjk->bik", batchA, batchB)
```

## Random Number Generation

All RNG methods use a seeded source for reproducibility.

```go
rng := numgo.NewRNG(42)

rng.Rand(3, 3)             // uniform [0, 1)
rng.Randn(3, 3)            // standard normal
rng.Normal(5.0, 2.0, 100)  // normal with mean=5, std=2
rng.Uniform(0, 10, 50)     // uniform [0, 10)
rng.RandInt(0, 10, 3, 3)   // integer values in [0, 10)

rng.Choice(100, 5, false)  // 5 unique indices from [0, 100)
rng.Choice(10, 20, true)   // 20 indices with replacement

rng.Shuffle(arr)            // in-place shuffle along axis 0

rng.Dirichlet([]float64{1, 1, 1})      // Dirichlet sample
rng.Multinomial(100, []float64{0.2, 0.3, 0.5}) // multinomial sample
```

## Sorting, Searching, and Set Operations

```go
a := numgo.FromSlice([]float64{3, 1, 4, 1, 5})

numgo.Sort(a, 0)         // sorted copy along axis
numgo.ArgSort(a, 0)      // indices that would sort the array
numgo.Unique(a)          // sorted unique values -> [1, 3, 4, 5]

// Conditional selection
cond := numgo.FromSlice([]float64{1, 0, 1, 0, 1})
x := numgo.FromSlice([]float64{10, 20, 30, 40, 50})
y := numgo.Zeros(5)
numgo.Where(cond, x, y)  // [10, 0, 30, 0, 50]

numgo.Nonzero(a)          // indices of nonzero elements

sorted := numgo.FromSlice([]float64{1, 3, 5, 7, 9})
vals := numgo.FromSlice([]float64{2, 6})
numgo.SearchSorted(sorted, vals) // insertion points -> [1, 3]
```

### Set Operations

```go
a := numgo.FromSlice([]float64{1, 2, 3, 4})
b := numgo.FromSlice([]float64{3, 4, 5, 6})

numgo.Intersect1D(a, b)  // [3, 4]
numgo.Union1D(a, b)      // [1, 2, 3, 4, 5, 6]
numgo.SetDiff1D(a, b)    // [1, 2]
```

## Comparison

```go
numgo.AllClose(a, b, 1e-8, 1e-5) // true if all elements are close
```

## API Summary

| Category | Functions / Methods |
|---|---|
| **Creation** | `NewNDArray`, `Zeros`, `Ones`, `Full`, `Eye`, `FromSlice`, `FromSlice2D` |
| **Properties** | `Shape`, `Ndim`, `Size`, `Data`, `String` |
| **Indexing** | `Get`, `Set` |
| **Manipulation** | `Reshape`, `Flatten`, `Copy`, `T` |
| **Arithmetic** | `Add`, `Sub`, `Mul`, `Div`, `AddScalar`, `SubScalar`, `MulScalar`, `DivScalar` |
| **Reductions** | `Sum`, `Prod`, `Max`, `ArgMax` |
| **Broadcasting** | `BroadcastShapes`, `BroadcastTo` |
| **Linear Algebra** | `Einsum` |
| **Random** | `NewRNG`, `Rand`, `Randn`, `Normal`, `Uniform`, `RandInt`, `Choice`, `Shuffle`, `Dirichlet`, `Multinomial` |
| **Sorting** | `Sort`, `ArgSort`, `Unique`, `Where`, `Nonzero`, `SearchSorted` |
| **Set Ops** | `Intersect1D`, `Union1D`, `SetDiff1D` |
| **Comparison** | `AllClose` |
