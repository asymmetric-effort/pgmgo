//go:build unit

package blas

import "testing"

func benchDgemm(b *testing.B, n int) {
	a := make([]float64, n*n)
	bm := make([]float64, n*n)
	c := make([]float64, n*n)
	for i := range a {
		a[i] = float64(i%7) + 1
		bm[i] = float64(i%5) + 1
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Zero c each iteration to keep consistent work.
		for j := range c {
			c[j] = 0
		}
		Dgemm(false, false, n, n, n, 1.0, a, n, bm, n, 0.0, c, n)
	}
}

func BenchmarkDgemm_100(b *testing.B)  { benchDgemm(b, 100) }
func BenchmarkDgemm_500(b *testing.B)  { benchDgemm(b, 500) }
func BenchmarkDgemm_1000(b *testing.B) { benchDgemm(b, 1000) }

func benchDdot(b *testing.B, n int) {
	x := make([]float64, n)
	y := make([]float64, n)
	for i := range x {
		x[i] = float64(i + 1)
		y[i] = float64(n - i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Ddot(n, x, 1, y, 1)
	}
}

func BenchmarkDdot_100(b *testing.B)  { benchDdot(b, 100) }
func BenchmarkDdot_1000(b *testing.B) { benchDdot(b, 1000) }

func benchDgemv(b *testing.B, n int) {
	a := make([]float64, n*n)
	x := make([]float64, n)
	y := make([]float64, n)
	for i := range a {
		a[i] = float64(i%7) + 1
	}
	for i := range x {
		x[i] = float64(i + 1)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dgemv(false, n, n, 1.0, a, n, x, 1, 0.0, y, 1)
	}
}

func BenchmarkDgemv_100(b *testing.B)  { benchDgemv(b, 100) }
func BenchmarkDgemv_500(b *testing.B)  { benchDgemv(b, 500) }
func BenchmarkDgemv_1000(b *testing.B) { benchDgemv(b, 1000) }
