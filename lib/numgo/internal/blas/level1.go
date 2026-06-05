package blas

import "math"

// Ddot computes the dot product of two vectors x and y with increments incx and incy.
//
//	result = sum_{i=0}^{n-1} x[i*incx] * y[i*incy]
//
// Uses 4x loop unrolling for better ILP.
func Ddot(n int, x []float64, incx int, y []float64, incy int) float64 {
	if n <= 0 {
		return 0
	}

	// Fast path: unit stride.
	if incx == 1 && incy == 1 {
		return ddotUnit(n, x, y)
	}

	// General stride path.
	var sum float64
	ix, iy := 0, 0
	for i := 0; i < n; i++ {
		sum += x[ix] * y[iy]
		ix += incx
		iy += incy
	}
	return sum
}

// ddotUnit is the unit-stride fast path with 4x unrolling.
func ddotUnit(n int, x, y []float64) float64 {
	var s0, s1, s2, s3 float64
	m := n - n%4
	for i := 0; i < m; i += 4 {
		s0 += x[i] * y[i]
		s1 += x[i+1] * y[i+1]
		s2 += x[i+2] * y[i+2]
		s3 += x[i+3] * y[i+3]
	}
	sum := s0 + s1 + s2 + s3
	for i := m; i < n; i++ {
		sum += x[i] * y[i]
	}
	return sum
}

// Daxpy computes y = alpha*x + y.
//
//	y[i*incy] += alpha * x[i*incx]  for i = 0..n-1
func Daxpy(n int, alpha float64, x []float64, incx int, y []float64, incy int) {
	if n <= 0 || alpha == 0 {
		return
	}

	// Fast path: unit stride with 4x unrolling.
	if incx == 1 && incy == 1 {
		m := n - n%4
		for i := 0; i < m; i += 4 {
			y[i] += alpha * x[i]
			y[i+1] += alpha * x[i+1]
			y[i+2] += alpha * x[i+2]
			y[i+3] += alpha * x[i+3]
		}
		for i := m; i < n; i++ {
			y[i] += alpha * x[i]
		}
		return
	}

	// General stride path.
	ix, iy := 0, 0
	for i := 0; i < n; i++ {
		y[iy] += alpha * x[ix]
		ix += incx
		iy += incy
	}
}

// Dscal scales a vector by a constant: x = alpha*x.
//
//	x[i*incx] *= alpha  for i = 0..n-1
func Dscal(n int, alpha float64, x []float64, incx int) {
	if n <= 0 {
		return
	}

	// Fast path: unit stride with 4x unrolling.
	if incx == 1 {
		m := n - n%4
		for i := 0; i < m; i += 4 {
			x[i] *= alpha
			x[i+1] *= alpha
			x[i+2] *= alpha
			x[i+3] *= alpha
		}
		for i := m; i < n; i++ {
			x[i] *= alpha
		}
		return
	}

	// General stride path.
	ix := 0
	for i := 0; i < n; i++ {
		x[ix] *= alpha
		ix += incx
	}
}

// Dnrm2 computes the Euclidean norm of a vector using compensated (Kahan) summation
// to reduce floating-point error.
//
//	result = sqrt(sum_{i=0}^{n-1} x[i*incx]^2)
func Dnrm2(n int, x []float64, incx int) float64 {
	if n <= 0 {
		return 0
	}
	if n == 1 {
		return math.Abs(x[0])
	}

	// Compensated summation for better accuracy.
	var sum, comp float64
	ix := 0
	for i := 0; i < n; i++ {
		v := x[ix]
		prod := v * v
		y := prod - comp
		t := sum + y
		comp = (t - sum) - y
		sum = t
		ix += incx
	}
	return math.Sqrt(sum)
}

// Dasum computes the sum of absolute values of a vector.
//
//	result = sum_{i=0}^{n-1} |x[i*incx]|
func Dasum(n int, x []float64, incx int) float64 {
	if n <= 0 {
		return 0
	}

	// Fast path: unit stride with 4x unrolling.
	if incx == 1 {
		var s0, s1, s2, s3 float64
		m := n - n%4
		for i := 0; i < m; i += 4 {
			s0 += math.Abs(x[i])
			s1 += math.Abs(x[i+1])
			s2 += math.Abs(x[i+2])
			s3 += math.Abs(x[i+3])
		}
		sum := s0 + s1 + s2 + s3
		for i := m; i < n; i++ {
			sum += math.Abs(x[i])
		}
		return sum
	}

	// General stride path.
	var sum float64
	ix := 0
	for i := 0; i < n; i++ {
		sum += math.Abs(x[ix])
		ix += incx
	}
	return sum
}

// Idamax returns the index of the element with the largest absolute value.
// Returns -1 if n <= 0.
func Idamax(n int, x []float64, incx int) int {
	if n <= 0 {
		return -1
	}
	if n == 1 {
		return 0
	}

	maxIdx := 0
	maxVal := math.Abs(x[0])
	ix := incx
	for i := 1; i < n; i++ {
		v := math.Abs(x[ix])
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
		ix += incx
	}
	return maxIdx
}
