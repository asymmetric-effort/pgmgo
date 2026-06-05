//go:build unit

package blas

import (
	"math"
	"testing"
)

const tol = 1e-12

func approxEq(a, b, eps float64) bool {
	return math.Abs(a-b) < eps
}

// --- Ddot ---

func TestDdot_Basic(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{5, 4, 3, 2, 1}
	got := Ddot(5, x, 1, y, 1)
	want := 35.0 // 5+8+9+8+5
	if !approxEq(got, want, tol) {
		t.Errorf("Ddot = %v, want %v", got, want)
	}
}

func TestDdot_Stride(t *testing.T) {
	x := []float64{1, 0, 2, 0, 3}
	y := []float64{4, 0, 5, 0, 6}
	got := Ddot(3, x, 2, y, 2)
	want := 1*4 + 2*5 + 3*6.0
	if !approxEq(got, want, tol) {
		t.Errorf("Ddot stride = %v, want %v", got, want)
	}
}

func TestDdot_ZeroN(t *testing.T) {
	got := Ddot(0, nil, 1, nil, 1)
	if got != 0 {
		t.Errorf("Ddot(0) = %v, want 0", got)
	}
}

func TestDdot_NegativeN(t *testing.T) {
	got := Ddot(-1, nil, 1, nil, 1)
	if got != 0 {
		t.Errorf("Ddot(-1) = %v, want 0", got)
	}
}

func TestDdot_SingleElement(t *testing.T) {
	x := []float64{3}
	y := []float64{7}
	got := Ddot(1, x, 1, y, 1)
	if !approxEq(got, 21, tol) {
		t.Errorf("Ddot single = %v, want 21", got)
	}
}

func TestDdot_LargeUnrolled(t *testing.T) {
	n := 100
	x := make([]float64, n)
	y := make([]float64, n)
	var want float64
	for i := 0; i < n; i++ {
		x[i] = float64(i + 1)
		y[i] = float64(n - i)
		want += x[i] * y[i]
	}
	got := Ddot(n, x, 1, y, 1)
	if !approxEq(got, want, tol) {
		t.Errorf("Ddot large = %v, want %v", got, want)
	}
}

func TestDdot_Zeros(t *testing.T) {
	x := make([]float64, 8)
	y := make([]float64, 8)
	got := Ddot(8, x, 1, y, 1)
	if got != 0 {
		t.Errorf("Ddot zeros = %v, want 0", got)
	}
}

// --- Daxpy ---

func TestDaxpy_Basic(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	y := []float64{10, 20, 30, 40}
	Daxpy(4, 2.0, x, 1, y, 1)
	want := []float64{12, 24, 36, 48}
	for i := range want {
		if !approxEq(y[i], want[i], tol) {
			t.Errorf("Daxpy[%d] = %v, want %v", i, y[i], want[i])
		}
	}
}

func TestDaxpy_AlphaZero(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{10, 20, 30}
	Daxpy(3, 0, x, 1, y, 1)
	want := []float64{10, 20, 30}
	for i := range want {
		if y[i] != want[i] {
			t.Errorf("Daxpy alpha=0 [%d] = %v, want %v", i, y[i], want[i])
		}
	}
}

func TestDaxpy_Stride(t *testing.T) {
	x := []float64{1, 0, 2, 0, 3}
	y := []float64{10, 0, 20, 0, 30}
	Daxpy(3, 3.0, x, 2, y, 2)
	if !approxEq(y[0], 13, tol) || !approxEq(y[2], 26, tol) || !approxEq(y[4], 39, tol) {
		t.Errorf("Daxpy stride = %v", y)
	}
}

func TestDaxpy_ZeroN(t *testing.T) {
	y := []float64{1, 2}
	Daxpy(0, 5, []float64{1, 2}, 1, y, 1)
	if y[0] != 1 || y[1] != 2 {
		t.Errorf("Daxpy n=0 modified y: %v", y)
	}
}

func TestDaxpy_NegativeN(t *testing.T) {
	y := []float64{1}
	Daxpy(-1, 5, []float64{1}, 1, y, 1)
	if y[0] != 1 {
		t.Errorf("Daxpy n=-1 modified y: %v", y)
	}
}

func TestDaxpy_LargeUnrolled(t *testing.T) {
	n := 100
	x := make([]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = float64(i)
		y[i] = float64(i * 2)
	}
	alpha := 3.0
	Daxpy(n, alpha, x, 1, y, 1)
	for i := 0; i < n; i++ {
		want := float64(i*2) + alpha*float64(i)
		if !approxEq(y[i], want, tol) {
			t.Errorf("Daxpy large[%d] = %v, want %v", i, y[i], want)
		}
	}
}

// --- Dscal ---

func TestDscal_Basic(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	Dscal(5, 3.0, x, 1)
	want := []float64{3, 6, 9, 12, 15}
	for i := range want {
		if !approxEq(x[i], want[i], tol) {
			t.Errorf("Dscal[%d] = %v, want %v", i, x[i], want[i])
		}
	}
}

func TestDscal_Stride(t *testing.T) {
	x := []float64{1, 100, 2, 100, 3}
	Dscal(3, 5.0, x, 2)
	if !approxEq(x[0], 5, tol) || x[1] != 100 || !approxEq(x[2], 10, tol) || x[3] != 100 || !approxEq(x[4], 15, tol) {
		t.Errorf("Dscal stride = %v", x)
	}
}

func TestDscal_ZeroN(t *testing.T) {
	x := []float64{1}
	Dscal(0, 5, x, 1)
	if x[0] != 1 {
		t.Errorf("Dscal n=0 modified x: %v", x)
	}
}

func TestDscal_NegativeN(t *testing.T) {
	x := []float64{1}
	Dscal(-1, 5, x, 1)
	if x[0] != 1 {
		t.Errorf("Dscal n=-1 modified x: %v", x)
	}
}

func TestDscal_ZeroAlpha(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	Dscal(4, 0, x, 1)
	for i, v := range x {
		if v != 0 {
			t.Errorf("Dscal zero alpha [%d] = %v, want 0", i, v)
		}
	}
}

func TestDscal_LargeUnrolled(t *testing.T) {
	n := 100
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i + 1)
	}
	Dscal(n, 2.5, x, 1)
	for i := 0; i < n; i++ {
		want := float64(i+1) * 2.5
		if !approxEq(x[i], want, tol) {
			t.Errorf("Dscal large[%d] = %v, want %v", i, x[i], want)
		}
	}
}

// --- Dnrm2 ---

func TestDnrm2_Basic(t *testing.T) {
	x := []float64{3, 4}
	got := Dnrm2(2, x, 1)
	if !approxEq(got, 5, tol) {
		t.Errorf("Dnrm2 = %v, want 5", got)
	}
}

func TestDnrm2_SingleElement(t *testing.T) {
	x := []float64{-7}
	got := Dnrm2(1, x, 1)
	if !approxEq(got, 7, tol) {
		t.Errorf("Dnrm2 single = %v, want 7", got)
	}
}

func TestDnrm2_ZeroN(t *testing.T) {
	got := Dnrm2(0, nil, 1)
	if got != 0 {
		t.Errorf("Dnrm2 n=0 = %v, want 0", got)
	}
}

func TestDnrm2_NegativeN(t *testing.T) {
	got := Dnrm2(-1, nil, 1)
	if got != 0 {
		t.Errorf("Dnrm2 n=-1 = %v, want 0", got)
	}
}

func TestDnrm2_Stride(t *testing.T) {
	x := []float64{3, 0, 4}
	got := Dnrm2(2, x, 2)
	if !approxEq(got, 5, tol) {
		t.Errorf("Dnrm2 stride = %v, want 5", got)
	}
}

func TestDnrm2_Large(t *testing.T) {
	n := 100
	x := make([]float64, n)
	var want float64
	for i := 0; i < n; i++ {
		x[i] = float64(i + 1)
		want += x[i] * x[i]
	}
	want = math.Sqrt(want)
	got := Dnrm2(n, x, 1)
	if !approxEq(got, want, 1e-8) {
		t.Errorf("Dnrm2 large = %v, want %v", got, want)
	}
}

func TestDnrm2_AllZeros(t *testing.T) {
	x := make([]float64, 10)
	got := Dnrm2(10, x, 1)
	if got != 0 {
		t.Errorf("Dnrm2 zeros = %v, want 0", got)
	}
}

// --- Dasum ---

func TestDasum_Basic(t *testing.T) {
	x := []float64{1, -2, 3, -4, 5}
	got := Dasum(5, x, 1)
	if !approxEq(got, 15, tol) {
		t.Errorf("Dasum = %v, want 15", got)
	}
}

func TestDasum_ZeroN(t *testing.T) {
	got := Dasum(0, nil, 1)
	if got != 0 {
		t.Errorf("Dasum n=0 = %v, want 0", got)
	}
}

func TestDasum_NegativeN(t *testing.T) {
	got := Dasum(-1, nil, 1)
	if got != 0 {
		t.Errorf("Dasum n=-1 = %v, want 0", got)
	}
}

func TestDasum_Stride(t *testing.T) {
	x := []float64{1, 100, -2, 100, 3}
	got := Dasum(3, x, 2)
	want := 1.0 + 2.0 + 3.0
	if !approxEq(got, want, tol) {
		t.Errorf("Dasum stride = %v, want %v", got, want)
	}
}

func TestDasum_AllNegative(t *testing.T) {
	x := []float64{-1, -2, -3, -4}
	got := Dasum(4, x, 1)
	if !approxEq(got, 10, tol) {
		t.Errorf("Dasum neg = %v, want 10", got)
	}
}

func TestDasum_LargeUnrolled(t *testing.T) {
	n := 100
	x := make([]float64, n)
	var want float64
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			x[i] = float64(i + 1)
		} else {
			x[i] = -float64(i + 1)
		}
		want += float64(i + 1)
	}
	got := Dasum(n, x, 1)
	if !approxEq(got, want, tol) {
		t.Errorf("Dasum large = %v, want %v", got, want)
	}
}

// --- Idamax ---

func TestIdamax_Basic(t *testing.T) {
	x := []float64{1, -5, 3, -2}
	got := Idamax(4, x, 1)
	if got != 1 {
		t.Errorf("Idamax = %v, want 1", got)
	}
}

func TestIdamax_ZeroN(t *testing.T) {
	got := Idamax(0, nil, 1)
	if got != -1 {
		t.Errorf("Idamax n=0 = %v, want -1", got)
	}
}

func TestIdamax_NegativeN(t *testing.T) {
	got := Idamax(-1, nil, 1)
	if got != -1 {
		t.Errorf("Idamax n=-1 = %v, want -1", got)
	}
}

func TestIdamax_SingleElement(t *testing.T) {
	x := []float64{42}
	got := Idamax(1, x, 1)
	if got != 0 {
		t.Errorf("Idamax single = %v, want 0", got)
	}
}

func TestIdamax_Stride(t *testing.T) {
	x := []float64{1, 100, 2, -100, 3}
	got := Idamax(3, x, 2) // looks at indices 0, 2, 4 -> values 1, 2, 3
	if got != 2 {
		t.Errorf("Idamax stride = %v, want 2", got)
	}
}

func TestIdamax_FirstMaxWins(t *testing.T) {
	x := []float64{5, -5, 5}
	got := Idamax(3, x, 1)
	if got != 0 {
		t.Errorf("Idamax tie = %v, want 0 (first occurrence)", got)
	}
}

func TestIdamax_AllNegative(t *testing.T) {
	x := []float64{-1, -10, -3}
	got := Idamax(3, x, 1)
	if got != 1 {
		t.Errorf("Idamax all neg = %v, want 1", got)
	}
}

func TestIdamax_AllZeros(t *testing.T) {
	x := make([]float64, 5)
	got := Idamax(5, x, 1)
	if got != 0 {
		t.Errorf("Idamax zeros = %v, want 0", got)
	}
}
