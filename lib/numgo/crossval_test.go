//go:build unit

package numgo_test

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// tolerances
const (
	tolExact     = 1e-10 // exact arithmetic (creation, basic ops)
	tolIterative = 1e-6  // iterative algorithms (SVD, Eig, QR)
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// toNDArray1D builds a 1-D NDArray from a JSON float slice.
func toNDArray1D(data []float64) *numgo.NDArray {
	return numgo.FromSlice(data)
}

// toNDArray2D builds a 2-D NDArray from a JSON [][]float64.
func toNDArray2D(data [][]float64) *numgo.NDArray {
	return numgo.FromSlice2D(data)
}

// flatResult returns the flat data of an NDArray as []float64 for comparison.
func flatResult(a *numgo.NDArray) []float64 {
	return a.Data()
}

// assertClose checks that got and want slices are element-wise within tol.
func assertClose(t *testing.T, name string, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("[%s] length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(want[i]) && math.IsNaN(got[i]) {
			continue
		}
		if math.IsInf(want[i], 0) && math.IsInf(got[i], 0) {
			if math.Signbit(want[i]) == math.Signbit(got[i]) {
				continue
			}
		}
		diff := math.Abs(got[i] - want[i])
		if diff > tol+tol*math.Abs(want[i]) {
			t.Errorf("[%s] element %d: got %g, want %g (diff %g, tol %g)",
				name, i, got[i], want[i], diff, tol)
		}
	}
}

// assertScalarClose checks a single float value.
func assertScalarClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	diff := math.Abs(got - want)
	if diff > tol+tol*math.Abs(want) {
		t.Errorf("[%s] got %g, want %g (diff %g, tol %g)", name, got, want, diff, tol)
	}
}

// flatten2D converts [][]float64 to []float64 in row-major order.
func flatten2D(data [][]float64) []float64 {
	var flat []float64
	for _, row := range data {
		flat = append(flat, row...)
	}
	return flat
}

// flatten3D converts [][][]float64 to []float64 in row-major order.
func flatten3D(data [][][]float64) []float64 {
	var flat []float64
	for _, matrix := range data {
		for _, row := range matrix {
			flat = append(flat, row...)
		}
	}
	return flat
}

// ndArrayFromNestedJSON builds an NDArray from arbitrary-depth nested JSON arrays.
// Supports 1D, 2D, and 3D.
func ndArrayFromJSON(raw json.RawMessage) *numgo.NDArray {
	// Try 1D first
	var arr1 []float64
	if err := json.Unmarshal(raw, &arr1); err == nil {
		return numgo.FromSlice(arr1)
	}
	// Try 2D
	var arr2 [][]float64
	if err := json.Unmarshal(raw, &arr2); err == nil {
		return numgo.FromSlice2D(arr2)
	}
	// Try 3D
	var arr3 [][][]float64
	if err := json.Unmarshal(raw, &arr3); err == nil {
		if len(arr3) == 0 {
			return numgo.NewNDArray([]int{0, 0, 0}, nil)
		}
		d0 := len(arr3)
		d1 := len(arr3[0])
		d2 := len(arr3[0][0])
		flat := flatten3D(arr3)
		return numgo.NewNDArray([]int{d0, d1, d2}, flat)
	}
	panic("ndArrayFromJSON: unsupported nesting depth")
}

// ---------------------------------------------------------------------------
// test runner
// ---------------------------------------------------------------------------

func TestCrossval_Numgo(t *testing.T) {
	ff := testutil.LoadFixtures(t, "numgo/fixtures.json")
	if ff == nil {
		return
	}

	for _, tc := range ff.TestCases {
		tc := tc
		t.Run(tc.Name, func(t *testing.T) {
			var input map[string]json.RawMessage
			tc.UnmarshalInput(t, &input)

			var expected map[string]json.RawMessage
			tc.UnmarshalExpected(t, &expected)

			// Determine category from the test case name prefix or explicit field.
			// The fixture includes a "category" field at the test case level,
			// but since testutil only gives us Input/Expected raw messages, we
			// route based on the test name instead.
			switch tc.Name {

			// ----- CREATION -----
			case "zeros_2x3":
				testCreationShape(t, tc.Name, expected, func(shape []int) *numgo.NDArray {
					return numgo.Zeros(shape...)
				}, input, tolExact)
			case "ones_3x2":
				testCreationShape(t, tc.Name, expected, func(shape []int) *numgo.NDArray {
					return numgo.Ones(shape...)
				}, input, tolExact)
			case "eye_3":
				var n int
				unmarshalField(t, input, "n", &n)
				got := numgo.Eye(n)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "arange_0_10_2":
				var start, stop, step float64
				unmarshalField(t, input, "start", &start)
				unmarshalField(t, input, "stop", &stop)
				unmarshalField(t, input, "step", &step)
				got := numgo.Arange(start, stop, step)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "linspace_0_1_5":
				var start, stop float64
				var num int
				unmarshalField(t, input, "start", &start)
				unmarshalField(t, input, "stop", &stop)
				unmarshalField(t, input, "num", &num)
				got := numgo.Linspace(start, stop, num)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "logspace_0_2_5":
				var start, stop float64
				var num int
				unmarshalField(t, input, "start", &start)
				unmarshalField(t, input, "stop", &stop)
				unmarshalField(t, input, "num", &num)
				got := numgo.Logspace(start, stop, num)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "diag_1d":
				var a []float64
				var k int
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "k", &k)
				got := numgo.Diag(toNDArray1D(a), k)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "diag_2d_extract":
				var a [][]float64
				var k int
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "k", &k)
				got := numgo.Diag(toNDArray2D(a), k)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "tri_3x3_k0":
				var n, m, k int
				unmarshalField(t, input, "n", &n)
				unmarshalField(t, input, "m", &m)
				unmarshalField(t, input, "k", &k)
				got := numgo.Tri(n, m, k)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "tril_3x3":
				var a [][]float64
				var k int
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "k", &k)
				got := numgo.Tril(toNDArray2D(a), k)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "triu_3x3":
				var a [][]float64
				var k int
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "k", &k)
				got := numgo.Triu(toNDArray2D(a), k)
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- ARITHMETIC -----
			case "add_2x2":
				testBinaryOp2D(t, tc.Name, input, expected, numgo.Add, tolExact)
			case "sub_2x2":
				testBinaryOp2D(t, tc.Name, input, expected, numgo.Sub, tolExact)
			case "mul_2x2":
				testBinaryOp2D(t, tc.Name, input, expected, numgo.Mul, tolExact)
			case "div_2x2":
				testBinaryOp2D(t, tc.Name, input, expected, numgo.Div, tolExact)
			case "add_broadcast_2x3_1x3", "add_broadcast_3x1_1x3":
				var a, b [][]float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.Add(toNDArray2D(a), toNDArray2D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "sum_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sum(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "prod_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Prod(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "sum_axis0":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sum(toNDArray2D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "sum_axis1":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sum(toNDArray2D(a), 1)
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- MATH -----
			case "sin":
				testUnaryOp(t, tc.Name, input, expected, numgo.Sin, tolExact)
			case "cos":
				testUnaryOp(t, tc.Name, input, expected, numgo.Cos, tolExact)
			case "exp":
				testUnaryOp(t, tc.Name, input, expected, numgo.Exp, tolExact)
			case "log":
				testUnaryOp(t, tc.Name, input, expected, numgo.Log, tolExact)
			case "sqrt":
				testUnaryOp(t, tc.Name, input, expected, numgo.Sqrt, tolExact)
			case "abs":
				testUnaryOp(t, tc.Name, input, expected, numgo.Absolute, tolExact)
			case "sign":
				testUnaryOp(t, tc.Name, input, expected, numgo.Sign, tolExact)
			case "clip":
				var a []float64
				var mn, mx float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "min", &mn)
				unmarshalField(t, input, "max", &mx)
				got := numgo.Clip(toNDArray1D(a), mn, mx)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "floor":
				testUnaryOp(t, tc.Name, input, expected, numgo.Floor, tolExact)
			case "ceil":
				testUnaryOp(t, tc.Name, input, expected, numgo.Ceil, tolExact)
			case "round":
				var a []float64
				var decimals int
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "decimals", &decimals)
				got := numgo.Around(toNDArray1D(a), decimals)
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- LINEAR ALGEBRA -----
			case "matmul_2x2", "matmul_3x2_2x4":
				var a, b [][]float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got, err := numgo.Matmul(toNDArray2D(a), toNDArray2D(b))
				if err != nil {
					t.Fatalf("[%s] Matmul error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "dot_1d":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got, err := numgo.Dot(toNDArray1D(a), toNDArray1D(b))
				if err != nil {
					t.Fatalf("[%s] Dot error: %v", tc.Name, err)
				}
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "det_2x2", "det_3x3":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got, err := numgo.Det(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] Det error: %v", tc.Name, err)
				}
				assertResultScalar(t, tc.Name, expected, got, tolExact)
			case "inv_2x2":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got, err := numgo.Inv(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] Inv error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "solve_2x2":
				var a [][]float64
				var b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got, err := numgo.Solve(toNDArray2D(a), toNDArray1D(b))
				if err != nil {
					t.Fatalf("[%s] Solve error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "svd_4x5", "svd_3x2":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				_, s, _, err := numgo.SVD(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] SVD error: %v", tc.Name, err)
				}
				// Compare singular values (sorted descending).
				var wantS []float64
				unmarshalField(t, expected, "s", &wantS)
				gotS := s.Data()
				// Sort both descending for comparison.
				sort.Sort(sort.Reverse(sort.Float64Slice(gotS)))
				sort.Sort(sort.Reverse(sort.Float64Slice(wantS)))
				assertClose(t, tc.Name+" singular_values", gotS, wantS, tolIterative)

			case "eig_symmetric_2x2", "eig_symmetric_3x3":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				vals, _, err := numgo.Eig(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] Eig error: %v", tc.Name, err)
				}
				var wantSorted []float64
				unmarshalField(t, expected, "eigenvalues_sorted", &wantSorted)
				gotVals := vals.Data()
				sort.Float64s(gotVals)
				assertClose(t, tc.Name+" eigenvalues", gotVals, wantSorted, tolIterative)

			case "cholesky_2x2", "cholesky_3x3":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got, err := numgo.Cholesky(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] Cholesky error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "qr_4x3":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				q, r, err := numgo.QR(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] QR error: %v", tc.Name, err)
				}
				// QR decomposition: signs of columns may differ.
				// Verify A = Q*R instead of comparing Q,R directly.
				qr, err := numgo.Matmul(q, r)
				if err != nil {
					t.Fatalf("[%s] Q*R multiply error: %v", tc.Name, err)
				}
				assertClose(t, tc.Name+" A=QR", flatResult(qr), flatten2D(a), tolIterative)
				// Also check Q is orthogonal: Q^T Q = I
				qt := q.T()
				qtq, _ := numgo.Matmul(qt, q)
				n := q.Shape()[1]
				eye := numgo.Eye(n)
				assertClose(t, tc.Name+" QtQ=I", flatResult(qtq), flatResult(eye), tolIterative)

			case "lstsq_3x2":
				var a [][]float64
				var b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got, err := numgo.Lstsq(toNDArray2D(a), toNDArray1D(b))
				if err != nil {
					t.Fatalf("[%s] Lstsq error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolIterative)
			case "norm_vec2", "norm_vec1":
				var a []float64
				var ord int
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "ord", &ord)
				got, err := numgo.Norm(toNDArray1D(a), ord, -1)
				if err != nil {
					t.Fatalf("[%s] Norm error: %v", tc.Name, err)
				}
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "trace_3x3":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got, err := numgo.Trace(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] Trace error: %v", tc.Name, err)
				}
				assertResultScalar(t, tc.Name, expected, got, tolExact)

			// ----- STATISTICS -----
			case "mean_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Mean(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "mean_axis0":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Mean(toNDArray2D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "mean_axis1":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Mean(toNDArray2D(a), 1)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "std_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Std(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "var_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Var(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "min_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Min(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "max_all":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Max(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "sum_all_stats":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sum(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "prod_all_stats":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Prod(toNDArray2D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "cumsum_1d":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Cumsum(toNDArray1D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "cumprod_1d":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Cumprod(toNDArray1D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "percentile_50", "percentile_25":
				var a []float64
				var q float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "q", &q)
				got := numgo.Percentile(toNDArray1D(a), q)
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "median":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Median(toNDArray1D(a))
				assertResultScalar(t, tc.Name, expected, got.Data()[0], tolExact)
			case "corrcoef":
				var x, y []float64
				unmarshalField(t, input, "x", &x)
				unmarshalField(t, input, "y", &y)
				got, err := numgo.Corrcoef(toNDArray1D(x), toNDArray1D(y))
				if err != nil {
					t.Fatalf("[%s] Corrcoef error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "cov_2d":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got, err := numgo.Cov(toNDArray2D(a))
				if err != nil {
					t.Fatalf("[%s] Cov error: %v", tc.Name, err)
				}
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- SORTING -----
			case "sort_1d":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sort(toNDArray1D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "argsort_1d":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.ArgSort(toNDArray1D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "unique":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Unique(toNDArray1D(a))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "searchsorted":
				var sorted, values []float64
				unmarshalField(t, input, "sorted", &sorted)
				unmarshalField(t, input, "values", &values)
				got := numgo.SearchSorted(toNDArray1D(sorted), toNDArray1D(values))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "sort_2d_axis0":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sort(toNDArray2D(a), 0)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "sort_2d_axis1":
				var a [][]float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Sort(toNDArray2D(a), 1)
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- LOGIC -----
			case "all_true":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.All(toNDArray1D(a))
				var wantBool bool
				unmarshalField(t, expected, "result", &wantBool)
				gotBool := got.Data()[0] != 0
				if gotBool != wantBool {
					t.Errorf("[%s] got %v, want %v", tc.Name, gotBool, wantBool)
				}
			case "all_false":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.All(toNDArray1D(a))
				var wantBool bool
				unmarshalField(t, expected, "result", &wantBool)
				gotBool := got.Data()[0] != 0
				if gotBool != wantBool {
					t.Errorf("[%s] got %v, want %v", tc.Name, gotBool, wantBool)
				}
			case "any_true":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Any(toNDArray1D(a))
				var wantBool bool
				unmarshalField(t, expected, "result", &wantBool)
				gotBool := got.Data()[0] != 0
				if gotBool != wantBool {
					t.Errorf("[%s] got %v, want %v", tc.Name, gotBool, wantBool)
				}
			case "any_false":
				var a []float64
				unmarshalField(t, input, "a", &a)
				got := numgo.Any(toNDArray1D(a))
				var wantBool bool
				unmarshalField(t, expected, "result", &wantBool)
				gotBool := got.Data()[0] != 0
				if gotBool != wantBool {
					t.Errorf("[%s] got %v, want %v", tc.Name, gotBool, wantBool)
				}
			case "allclose_true", "allclose_false":
				var a, b []float64
				var atol, rtol float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				unmarshalField(t, input, "atol", &atol)
				unmarshalField(t, input, "rtol", &rtol)
				got := numgo.AllClose(toNDArray1D(a), toNDArray1D(b), atol, rtol)
				var wantBool bool
				unmarshalField(t, expected, "result", &wantBool)
				if got != wantBool {
					t.Errorf("[%s] got %v, want %v", tc.Name, got, wantBool)
				}
			case "isnan":
				arr := parseSpecialFloats(t, input)
				got := numgo.Isnan(arr)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "isinf":
				arr := parseSpecialFloats(t, input)
				got := numgo.Isinf(arr)
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "greater":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.Greater(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "less":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.Less(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "equal":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.Equal(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- SET OPERATIONS -----
			case "intersect1d":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.Intersect1D(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "union1d":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.Union1D(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "setdiff1d":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.SetDiff1D(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)
			case "in1d":
				var a, b []float64
				unmarshalField(t, input, "a", &a)
				unmarshalField(t, input, "b", &b)
				got := numgo.In1d(toNDArray1D(a), toNDArray1D(b))
				assertResultArray(t, tc.Name, expected, got, tolExact)

			// ----- EINSUM -----
			case "einsum_matmul", "einsum_trace", "einsum_outer",
				"einsum_batch_matmul", "einsum_row_sums", "einsum_col_sums",
				"einsum_dot":
				var notation string
				unmarshalField(t, input, "notation", &notation)
				var operandsRaw []json.RawMessage
				unmarshalField(t, input, "operands", &operandsRaw)
				operands := make([]*numgo.NDArray, len(operandsRaw))
				for i, raw := range operandsRaw {
					operands[i] = ndArrayFromJSON(raw)
				}
				got, err := numgo.Einsum(notation, operands...)
				if err != nil {
					t.Fatalf("[%s] Einsum error: %v", tc.Name, err)
				}
				// Handle scalar result (einsum trace/dot returns shape [])
				var wantScalar float64
				if err := json.Unmarshal(expected["result"], &wantScalar); err == nil {
					// scalar expected
					gotData := got.Data()
					if len(gotData) != 1 {
						t.Fatalf("[%s] expected scalar, got %d elements", tc.Name, len(gotData))
					}
					assertScalarClose(t, tc.Name, gotData[0], wantScalar, tolExact)
				} else {
					assertResultArray(t, tc.Name, expected, got, tolExact)
				}

			default:
				t.Logf("unhandled test case: %s (skipping)", tc.Name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// shared test helpers
// ---------------------------------------------------------------------------

func unmarshalField(t *testing.T, m map[string]json.RawMessage, key string, v any) {
	t.Helper()
	raw, ok := m[key]
	if !ok {
		t.Fatalf("missing field %q in fixture", key)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("failed to unmarshal %q: %v", key, err)
	}
}

// testCreationShape calls a creation function with shape from input and compares.
func testCreationShape(t *testing.T, name string, expected map[string]json.RawMessage,
	fn func([]int) *numgo.NDArray, input map[string]json.RawMessage, tol float64) {
	t.Helper()
	var shape []int
	unmarshalField(t, input, "shape", &shape)
	got := fn(shape)
	assertResultArray(t, name, expected, got, tol)
}

// testBinaryOp2D loads two 2D arrays from "a","b", applies op, compares with expected "result".
func testBinaryOp2D(t *testing.T, name string, input, expected map[string]json.RawMessage,
	op func(*numgo.NDArray, *numgo.NDArray) *numgo.NDArray, tol float64) {
	t.Helper()
	var a, b [][]float64
	unmarshalField(t, input, "a", &a)
	unmarshalField(t, input, "b", &b)
	got := op(toNDArray2D(a), toNDArray2D(b))
	assertResultArray(t, name, expected, got, tol)
}

// testUnaryOp loads a 1D array from "a", applies op, compares with expected "result".
func testUnaryOp(t *testing.T, name string, input, expected map[string]json.RawMessage,
	op func(*numgo.NDArray) *numgo.NDArray, tol float64) {
	t.Helper()
	var a []float64
	unmarshalField(t, input, "a", &a)
	got := op(toNDArray1D(a))
	assertResultArray(t, name, expected, got, tol)
}

// assertResultArray compares an NDArray against the "result" field in expected.
// Handles both 1D and 2D expected values.
func assertResultArray(t *testing.T, name string, expected map[string]json.RawMessage, got *numgo.NDArray, tol float64) {
	t.Helper()
	raw := expected["result"]

	// Try as flat array first.
	var flat []float64
	if err := json.Unmarshal(raw, &flat); err == nil {
		assertClose(t, name, flatResult(got), flat, tol)
		return
	}

	// Try as 2D array.
	var arr2d [][]float64
	if err := json.Unmarshal(raw, &arr2d); err == nil {
		want := flatten2D(arr2d)
		assertClose(t, name, flatResult(got), want, tol)
		return
	}

	// Try as 3D array.
	var arr3d [][][]float64
	if err := json.Unmarshal(raw, &arr3d); err == nil {
		want := flatten3D(arr3d)
		assertClose(t, name, flatResult(got), want, tol)
		return
	}

	t.Fatalf("[%s] could not unmarshal expected result", name)
}

// assertResultScalar compares a scalar against the "result" field in expected.
func assertResultScalar(t *testing.T, name string, expected map[string]json.RawMessage, got float64, tol float64) {
	t.Helper()
	var want float64
	if err := json.Unmarshal(expected["result"], &want); err != nil {
		t.Fatalf("[%s] failed to unmarshal expected scalar: %v", name, err)
	}
	assertScalarClose(t, name, got, want, tol)
}

// parseSpecialFloats reads the "a_special" field which contains a mix of
// float64 and string markers ("NaN", "Inf", "-Inf") and returns an NDArray.
func parseSpecialFloats(t *testing.T, input map[string]json.RawMessage) *numgo.NDArray {
	t.Helper()
	var items []json.RawMessage
	unmarshalField(t, input, "a_special", &items)
	data := make([]float64, len(items))
	for i, raw := range items {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			switch s {
			case "NaN":
				data[i] = math.NaN()
			case "Inf":
				data[i] = math.Inf(1)
			case "-Inf":
				data[i] = math.Inf(-1)
			default:
				t.Fatalf("unknown special float: %s", s)
			}
			continue
		}
		var f float64
		if err := json.Unmarshal(raw, &f); err != nil {
			t.Fatalf("failed to parse special float item %d: %v", i, err)
		}
		data[i] = f
	}
	return numgo.FromSlice(data)
}

// ensure fmt is used
var _ = fmt.Sprintf
