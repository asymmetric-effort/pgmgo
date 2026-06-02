# gpu

Package `gpu` provides a compute backend abstraction for accelerated factor operations. The default implementation uses pure Go on the CPU. Future CGO backends (CUDA, OpenCL) can be plugged in via the `Backend` interface.

```
import "github.com/asymmetric-effort/pgmgo/lib/gpu"
```

## Backend Interface

All compute backends implement the `Backend` interface:

```go
type Backend interface {
    Name() string
    IsAvailable() bool
    MatMul(a, b []float64, m, k, n int) []float64
    ElementWiseMul(a, b []float64) []float64
    Sum(a []float64) float64
    Normalize(a []float64) []float64
    FactorProduct(aValues []float64, aShape []int, bValues []float64, bShape []int, resultShape []int) []float64
    Marginalize(values []float64, shape []int, axis int) ([]float64, []int)
    Close() error
}
```

| Method | Description |
|---|---|
| `Name` | Backend identifier (e.g. "cpu", "cuda") |
| `IsAvailable` | Whether the backend can run on the current system |
| `MatMul` | Matrix multiplication of (m x k) and (k x n) matrices in row-major order |
| `ElementWiseMul` | Element-wise product of two equal-length slices |
| `Sum` | Sum of all elements |
| `Normalize` | Divide each element by the total sum (produces a distribution summing to 1) |
| `FactorProduct` | Outer product of two discrete factors given their shapes |
| `Marginalize` | Sum out one axis from a multi-dimensional tensor, returning reduced values and new shape |
| `Close` | Release resources held by the backend |

## CPUBackend (Default)

The `CPUBackend` is a pure-Go implementation that is always available.

```go
backend := gpu.NewCPUBackend()

backend.Name()        // "cpu"
backend.IsAvailable() // true

// Matrix multiplication: (2x3) * (3x2) -> (2x2)
a := []float64{1, 2, 3, 4, 5, 6}
b := []float64{7, 8, 9, 10, 11, 12}
c := backend.MatMul(a, b, 2, 3, 2) // row-major result

// Element-wise multiplication
result := backend.ElementWiseMul([]float64{1, 2, 3}, []float64{4, 5, 6})

// Sum
total := backend.Sum([]float64{1, 2, 3}) // 6.0

// Normalize to a probability distribution
dist := backend.Normalize([]float64{2, 3, 5}) // [0.2, 0.3, 0.5]

// Factor product (outer product of two discrete factors)
aVals := []float64{0.3, 0.7}
bVals := []float64{0.4, 0.6}
product := backend.FactorProduct(aVals, []int{2}, bVals, []int{2}, []int{2, 2})
// [0.12, 0.18, 0.28, 0.42]

// Marginalize: sum out axis 1 from a 2x3 tensor
vals := []float64{1, 2, 3, 4, 5, 6}
reduced, newShape := backend.Marginalize(vals, []int{2, 3}, 1)
// reduced = [6, 15], newShape = [2]

backend.Close() // no-op for CPU
```

## Global Backend Management

The package maintains a global default backend used by other pgmgo components.

```go
// Get the current backend (CPUBackend by default)
b := gpu.GetBackend()

// Set a custom backend
gpu.SetBackend(myCustomBackend)

// Reset to the default CPUBackend
gpu.ResetBackend()
```

## Implementing a Custom Backend

To add GPU acceleration, implement the `Backend` interface and register it:

```go
type CUDABackend struct {
    // device handle, context, etc.
}

func (c *CUDABackend) Name() string        { return "cuda" }
func (c *CUDABackend) IsAvailable() bool   { /* check for CUDA device */ }
func (c *CUDABackend) MatMul(a, b []float64, m, k, n int) []float64 { /* CUDA kernel */ }
func (c *CUDABackend) ElementWiseMul(a, b []float64) []float64      { /* CUDA kernel */ }
func (c *CUDABackend) Sum(a []float64) float64                      { /* CUDA reduction */ }
func (c *CUDABackend) Normalize(a []float64) []float64              { /* CUDA kernel */ }
func (c *CUDABackend) FactorProduct(aValues []float64, aShape []int, bValues []float64, bShape []int, resultShape []int) []float64 { /* ... */ }
func (c *CUDABackend) Marginalize(values []float64, shape []int, axis int) ([]float64, []int) { /* ... */ }
func (c *CUDABackend) Close() error { /* release CUDA resources */ }

// Register the backend
gpu.SetBackend(&CUDABackend{})
```

## API Summary

| Category | Types / Functions |
|---|---|
| **Interface** | `Backend` |
| **CPU Backend** | `CPUBackend`, `NewCPUBackend` |
| **Global Management** | `GetBackend`, `SetBackend`, `ResetBackend` |
