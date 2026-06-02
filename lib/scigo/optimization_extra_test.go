//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// CurveFit
// ---------------------------------------------------------------------------

func TestCurveFit_Linear(t *testing.T) {
	// Fit y = a*x + b
	f := func(x float64, params []float64) float64 {
		return params[0]*x + params[1]
	}
	xdata := []float64{1, 2, 3, 4, 5}
	ydata := []float64{3, 5, 7, 9, 11} // y = 2x + 1

	params, err := CurveFit(f, xdata, ydata, []float64{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(params[0], 2, 1e-3) {
		t.Errorf("CurveFit linear: a=%v, want 2", params[0])
	}
	if !approxEqual(params[1], 1, 1e-3) {
		t.Errorf("CurveFit linear: b=%v, want 1", params[1])
	}
}

func TestCurveFit_Quadratic(t *testing.T) {
	// Fit y = a*x^2 + b
	f := func(x float64, params []float64) float64 {
		return params[0]*x*x + params[1]
	}
	xdata := []float64{0, 1, 2, 3, 4}
	ydata := []float64{1, 4, 13, 28, 49} // y = 3*x^2 + 1

	params, err := CurveFit(f, xdata, ydata, []float64{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(params[0], 3, 1e-2) {
		t.Errorf("CurveFit quadratic: a=%v, want 3", params[0])
	}
	if !approxEqual(params[1], 1, 1e-2) {
		t.Errorf("CurveFit quadratic: b=%v, want 1", params[1])
	}
}

func TestCurveFit_EmptyData(t *testing.T) {
	f := func(x float64, params []float64) float64 { return params[0] * x }
	_, err := CurveFit(f, []float64{}, []float64{}, []float64{1})
	if err == nil {
		t.Error("CurveFit should error on empty data")
	}
}

func TestCurveFit_MismatchedData(t *testing.T) {
	f := func(x float64, params []float64) float64 { return params[0] * x }
	_, err := CurveFit(f, []float64{1, 2}, []float64{1}, []float64{1})
	if err == nil {
		t.Error("CurveFit should error on mismatched data lengths")
	}
}

func TestCurveFit_EmptyParams(t *testing.T) {
	f := func(x float64, params []float64) float64 { return x }
	_, err := CurveFit(f, []float64{1}, []float64{1}, []float64{})
	if err == nil {
		t.Error("CurveFit should error on empty params")
	}
}

// ---------------------------------------------------------------------------
// LeastSquares
// ---------------------------------------------------------------------------

func TestLeastSquares_Quadratic(t *testing.T) {
	// Minimize sum((params[0]*x^2 + params[1] - y)^2)
	xdata := []float64{0, 1, 2, 3, 4}
	ydata := []float64{1, 2, 5, 10, 17} // y = x^2 + 1

	f := func(params []float64) []float64 {
		residuals := make([]float64, len(xdata))
		for i := range xdata {
			residuals[i] = params[0]*xdata[i]*xdata[i] + params[1] - ydata[i]
		}
		return residuals
	}

	res, err := LeastSquares(f, []float64{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.X[0], 1, 1e-2) {
		t.Errorf("LeastSquares: a=%v, want 1", res.X[0])
	}
	if !approxEqual(res.X[1], 1, 1e-2) {
		t.Errorf("LeastSquares: b=%v, want 1", res.X[1])
	}
}

func TestLeastSquares_EmptyX0(t *testing.T) {
	f := func(params []float64) []float64 { return []float64{0} }
	_, err := LeastSquares(f, []float64{})
	if err == nil {
		t.Error("LeastSquares should error on empty x0")
	}
}

// ---------------------------------------------------------------------------
// LinearSumAssignment
// ---------------------------------------------------------------------------

func TestLinearSumAssignment_Identity(t *testing.T) {
	// Diagonal cost matrix: optimal is identity assignment
	cost := [][]float64{
		{1, 100, 100},
		{100, 1, 100},
		{100, 100, 1},
	}
	rows, cols, err := LinearSumAssignment(cost)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 || len(cols) != 3 {
		t.Fatalf("LinearSumAssignment: got %d assignments, want 3", len(rows))
	}
	totalCost := 0.0
	for i := range rows {
		totalCost += cost[rows[i]][cols[i]]
	}
	if !approxEqual(totalCost, 3, 1e-10) {
		t.Errorf("LinearSumAssignment: total cost=%v, want 3", totalCost)
	}
}

func TestLinearSumAssignment_Known(t *testing.T) {
	cost := [][]float64{
		{4, 1, 3},
		{2, 0, 5},
		{3, 2, 2},
	}
	rows, cols, err := LinearSumAssignment(cost)
	if err != nil {
		t.Fatal(err)
	}
	totalCost := 0.0
	for i := range rows {
		totalCost += cost[rows[i]][cols[i]]
	}
	// Optimal: (0,1)=1, (1,0)=2, (2,2)=2 => total=5
	if !approxEqual(totalCost, 5, 1e-10) {
		t.Errorf("LinearSumAssignment known: total cost=%v, want 5", totalCost)
	}
}

func TestLinearSumAssignment_Rectangular(t *testing.T) {
	// More columns than rows
	cost := [][]float64{
		{5, 1, 3, 7},
		{2, 4, 6, 8},
	}
	rows, cols, err := LinearSumAssignment(cost)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("LinearSumAssignment rect: got %d assignments, want 2", len(rows))
	}
	totalCost := 0.0
	for i := range rows {
		totalCost += cost[rows[i]][cols[i]]
	}
	// Optimal: row 0 -> col 1 (1), row 1 -> col 0 (2) => total=3
	if !approxEqual(totalCost, 3, 1e-10) {
		t.Errorf("LinearSumAssignment rect: total cost=%v, want 3", totalCost)
	}
}

func TestLinearSumAssignment_Empty(t *testing.T) {
	_, _, err := LinearSumAssignment([][]float64{})
	if err == nil {
		t.Error("LinearSumAssignment should error on empty")
	}
}

// ---------------------------------------------------------------------------
// Linprog
// ---------------------------------------------------------------------------

func TestLinprog_Simple(t *testing.T) {
	// minimize -x1 - 2*x2
	// subject to x1 + x2 <= 4
	//            x1 <= 3
	//            x2 <= 3
	//            x1, x2 >= 0
	c := []float64{-1, -2}
	Aub := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	bub := []float64{4, 3, 3}

	res, err := Linprog(c, Aub, bub, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("Linprog simple: expected success")
	}
	// Optimal: x1=1, x2=3, fun = -1 - 6 = -7
	if !approxEqual(res.Fun, -7, 1e-6) {
		t.Errorf("Linprog simple: fun=%v, want -7", res.Fun)
	}
}

func TestLinprog_WithEquality(t *testing.T) {
	// minimize x1 + x2
	// subject to x1 + x2 = 1
	//            x1, x2 >= 0
	c := []float64{1, 1}
	Aeq := [][]float64{{1, 1}}
	beq := []float64{1}

	res, err := Linprog(c, nil, nil, Aeq, beq)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.Fun, 1, 1e-6) {
		t.Errorf("Linprog equality: fun=%v, want 1", res.Fun)
	}
}

func TestLinprog_EmptyC(t *testing.T) {
	_, err := Linprog([]float64{}, nil, nil, nil, nil)
	if err == nil {
		t.Error("Linprog should error on empty c")
	}
}

// ---------------------------------------------------------------------------
// DifferentialEvolution
// ---------------------------------------------------------------------------

func TestDifferentialEvolution_Sphere(t *testing.T) {
	// Minimize x^2 + y^2
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	bounds := [][2]float64{{-5, 5}, {-5, 5}}
	res, err := DifferentialEvolution(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.Fun, 0, 1e-6) {
		t.Errorf("DE sphere: fun=%v, want 0", res.Fun)
	}
	if !approxEqual(res.X[0], 0, 1e-3) {
		t.Errorf("DE sphere: x=%v, want 0", res.X[0])
	}
	if !approxEqual(res.X[1], 0, 1e-3) {
		t.Errorf("DE sphere: y=%v, want 0", res.X[1])
	}
}

func TestDifferentialEvolution_Rastrigin1D(t *testing.T) {
	// Rastrigin function has global min at 0
	f := func(x []float64) float64 {
		return 10 + x[0]*x[0] - 10*math.Cos(2*math.Pi*x[0])
	}
	bounds := [][2]float64{{-5.12, 5.12}}
	res, err := DifferentialEvolution(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if res.Fun > 1 {
		t.Errorf("DE Rastrigin: fun=%v, want close to 0", res.Fun)
	}
}

func TestDifferentialEvolution_EmptyBounds(t *testing.T) {
	_, err := DifferentialEvolution(func(x []float64) float64 { return 0 }, nil)
	if err == nil {
		t.Error("DE should error on empty bounds")
	}
}

// ---------------------------------------------------------------------------
// BasinHopping
// ---------------------------------------------------------------------------

func TestBasinHopping_Quadratic(t *testing.T) {
	f := func(x []float64) float64 { return (x[0]-3)*(x[0]-3) + (x[1]+1)*(x[1]+1) }
	res, err := BasinHopping(f, []float64{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.X[0], 3, 1e-3) {
		t.Errorf("BasinHopping: x=%v, want 3", res.X[0])
	}
	if !approxEqual(res.X[1], -1, 1e-3) {
		t.Errorf("BasinHopping: y=%v, want -1", res.X[1])
	}
	if !approxEqual(res.Fun, 0, 1e-5) {
		t.Errorf("BasinHopping: fun=%v, want 0", res.Fun)
	}
}

func TestBasinHopping_EmptyX0(t *testing.T) {
	_, err := BasinHopping(func(x []float64) float64 { return 0 }, []float64{})
	if err == nil {
		t.Error("BasinHopping should error on empty x0")
	}
}

// ---------------------------------------------------------------------------
// DualAnnealing
// ---------------------------------------------------------------------------

func TestDualAnnealing_Sphere(t *testing.T) {
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	bounds := [][2]float64{{-10, 10}, {-10, 10}}
	res, err := DualAnnealing(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if res.Fun > 1e-4 {
		t.Errorf("DualAnnealing sphere: fun=%v, want ~0", res.Fun)
	}
}

func TestDualAnnealing_EmptyBounds(t *testing.T) {
	_, err := DualAnnealing(func(x []float64) float64 { return 0 }, nil)
	if err == nil {
		t.Error("DualAnnealing should error on empty bounds")
	}
}

// ---------------------------------------------------------------------------
// SHGO
// ---------------------------------------------------------------------------

func TestSHGO_Sphere(t *testing.T) {
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	bounds := [][2]float64{{-5, 5}, {-5, 5}}
	res, err := SHGO(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if res.Fun > 1e-4 {
		t.Errorf("SHGO sphere: fun=%v, want ~0", res.Fun)
	}
}

func TestSHGO_Rosenbrock(t *testing.T) {
	f := func(x []float64) float64 {
		return 100*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]) + (1-x[0])*(1-x[0])
	}
	bounds := [][2]float64{{-2, 2}, {-1, 3}}
	res, err := SHGO(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if res.Fun > 1 {
		t.Errorf("SHGO Rosenbrock: fun=%v, want close to 0", res.Fun)
	}
}

func TestSHGO_EmptyBounds(t *testing.T) {
	_, err := SHGO(func(x []float64) float64 { return 0 }, nil)
	if err == nil {
		t.Error("SHGO should error on empty bounds")
	}
}

// ---------------------------------------------------------------------------
// Direct
// ---------------------------------------------------------------------------

func TestDirect_Sphere(t *testing.T) {
	f := func(x []float64) float64 { return x[0]*x[0] + x[1]*x[1] }
	bounds := [][2]float64{{-5, 5}, {-5, 5}}
	res, err := Direct(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if res.Fun > 1e-2 {
		t.Errorf("Direct sphere: fun=%v, want ~0", res.Fun)
	}
}

func TestDirect_1D(t *testing.T) {
	f := func(x []float64) float64 { return (x[0] - 3) * (x[0] - 3) }
	bounds := [][2]float64{{-10, 10}}
	res, err := Direct(f, bounds)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.X[0], 3, 0.5) {
		t.Errorf("Direct 1D: x=%v, want ~3", res.X[0])
	}
}

func TestDirect_EmptyBounds(t *testing.T) {
	_, err := Direct(func(x []float64) float64 { return 0 }, nil)
	if err == nil {
		t.Error("Direct should error on empty bounds")
	}
}

// ---------------------------------------------------------------------------
// MILP
// ---------------------------------------------------------------------------

func TestMILP_Simple(t *testing.T) {
	// minimize -x1 - 2*x2
	// subject to x1 + x2 <= 4
	//            x1 <= 3
	//            x2 <= 3
	//            x1, x2 >= 0, integer
	c := []float64{-1, -2}
	Aub := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	bub := []float64{4, 3, 3}
	integrality := []bool{true, true}

	res, err := MILP(c, Aub, bub, integrality)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Success {
		t.Error("MILP: expected success")
	}
	// Optimal integer solution: x1=1, x2=3, fun = -1 - 6 = -7
	if !approxEqual(res.Fun, -7, 1e-6) {
		t.Errorf("MILP: fun=%v, want -7", res.Fun)
	}
	// Check integrality.
	for i, v := range res.X {
		if integrality[i] && math.Abs(v-math.Round(v)) > 1e-6 {
			t.Errorf("MILP: x[%d]=%v is not integer", i, v)
		}
	}
}

func TestMILP_NoIntegrality(t *testing.T) {
	// Without integrality constraints, should behave like LP.
	c := []float64{-1, -2}
	Aub := [][]float64{
		{1, 1},
		{1, 0},
		{0, 1},
	}
	bub := []float64{4, 3, 3}

	res, err := MILP(c, Aub, bub, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !approxEqual(res.Fun, -7, 1e-6) {
		t.Errorf("MILP no int: fun=%v, want -7", res.Fun)
	}
}

func TestMILP_EmptyC(t *testing.T) {
	_, err := MILP([]float64{}, nil, nil, nil)
	if err == nil {
		t.Error("MILP should error on empty c")
	}
}

func TestMILP_FractionalLP(t *testing.T) {
	// Problem where LP relaxation gives non-integer solution but integer optimum exists.
	// minimize -x1 - x2
	// subject to 2*x1 + x2 <= 5
	//            x1 + 2*x2 <= 5
	//            x1, x2 >= 0, integer
	c := []float64{-1, -1}
	Aub := [][]float64{
		{2, 1},
		{1, 2},
	}
	bub := []float64{5, 5}
	integrality := []bool{true, true}

	res, err := MILP(c, Aub, bub, integrality)
	if err != nil {
		t.Fatal(err)
	}
	// Check that solution is integer.
	for i, v := range res.X {
		if math.Abs(v-math.Round(v)) > 1e-6 {
			t.Errorf("MILP fractional: x[%d]=%v is not integer", i, v)
		}
	}
	// Check feasibility: 2*x1+x2 <= 5 and x1+2*x2 <= 5.
	x1, x2 := res.X[0], res.X[1]
	if 2*x1+x2 > 5+1e-6 || x1+2*x2 > 5+1e-6 {
		t.Errorf("MILP fractional: solution not feasible: x=%v", res.X)
	}
}
