package scigo

import (
	"errors"
	"math"
)

// OptResult holds the result of a multivariate optimization.
type OptResult struct {
	X       []float64 // Solution vector
	Fun     float64   // Objective function value at the solution
	Success bool      // Whether the optimizer converged
	Nit     int       // Number of iterations performed
}

// ScalarResult holds the result of a scalar optimization.
type ScalarResult struct {
	X       float64 // Solution
	Fun     float64 // Objective function value at the solution
	Success bool    // Whether the optimizer converged
	Nit     int     // Number of iterations performed
}

// Minimize minimizes a multivariate function f starting from x0.
// Supported methods: "nelder-mead", "gradient-descent".
func Minimize(f func([]float64) float64, x0 []float64, method string) (*OptResult, error) {
	switch method {
	case "nelder-mead":
		return nelderMead(f, x0)
	case "gradient-descent":
		return gradientDescent(f, x0)
	default:
		return nil, errors.New("scigo: unknown optimization method: " + method)
	}
}

// nelderMead implements the Nelder-Mead simplex algorithm.
func nelderMead(f func([]float64) float64, x0 []float64) (*OptResult, error) {
	n := len(x0)
	if n == 0 {
		return nil, errors.New("scigo: x0 must not be empty")
	}

	const (
		maxIter = 10000
		tol     = 1e-10
		alpha   = 1.0 // reflection
		gamma   = 2.0 // expansion
		rho     = 0.5 // contraction
		sigma   = 0.5 // shrink
	)

	// Build initial simplex: n+1 vertices
	simplex := make([][]float64, n+1)
	vals := make([]float64, n+1)

	simplex[0] = make([]float64, n)
	copy(simplex[0], x0)
	vals[0] = f(simplex[0])

	for i := 1; i <= n; i++ {
		v := make([]float64, n)
		copy(v, x0)
		if math.Abs(v[i-1]) > 1e-10 {
			v[i-1] += 0.5 * v[i-1] // 50% perturbation for non-zero
		} else {
			v[i-1] = 1.0 // unit step for zero starting values
		}
		simplex[i] = v
		vals[i] = f(v)
	}

	// Helper to sort simplex by function values
	sortSimplex := func() {
		for i := 0; i < n+1; i++ {
			for j := i + 1; j < n+1; j++ {
				if vals[j] < vals[i] {
					vals[i], vals[j] = vals[j], vals[i]
					simplex[i], simplex[j] = simplex[j], simplex[i]
				}
			}
		}
	}

	var nit int
	for nit = 0; nit < maxIter; nit++ {
		sortSimplex()

		// Check convergence: range of function values
		fRange := math.Abs(vals[n] - vals[0])
		if fRange < tol {
			break
		}

		// Centroid of all points except the worst
		centroid := make([]float64, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				centroid[j] += simplex[i][j]
			}
		}
		for j := 0; j < n; j++ {
			centroid[j] /= float64(n)
		}

		// Reflection
		xr := make([]float64, n)
		for j := 0; j < n; j++ {
			xr[j] = centroid[j] + alpha*(centroid[j]-simplex[n][j])
		}
		fr := f(xr)

		if fr < vals[0] {
			// Expansion
			xe := make([]float64, n)
			for j := 0; j < n; j++ {
				xe[j] = centroid[j] + gamma*(xr[j]-centroid[j])
			}
			fe := f(xe)
			if fe < fr {
				simplex[n] = xe
				vals[n] = fe
			} else {
				simplex[n] = xr
				vals[n] = fr
			}
		} else if fr < vals[n-1] {
			simplex[n] = xr
			vals[n] = fr
		} else {
			// Contraction
			var xc []float64
			var fc float64
			if fr < vals[n] {
				// Outside contraction
				xc = make([]float64, n)
				for j := 0; j < n; j++ {
					xc[j] = centroid[j] + rho*(xr[j]-centroid[j])
				}
				fc = f(xc)
			} else {
				// Inside contraction
				xc = make([]float64, n)
				for j := 0; j < n; j++ {
					xc[j] = centroid[j] + rho*(simplex[n][j]-centroid[j])
				}
				fc = f(xc)
			}

			if fc < vals[n] {
				simplex[n] = xc
				vals[n] = fc
			} else {
				// Shrink
				for i := 1; i <= n; i++ {
					for j := 0; j < n; j++ {
						simplex[i][j] = simplex[0][j] + sigma*(simplex[i][j]-simplex[0][j])
					}
					vals[i] = f(simplex[i])
				}
			}
		}
	}

	sortSimplex()
	return &OptResult{
		X:       simplex[0],
		Fun:     vals[0],
		Success: nit < maxIter,
		Nit:     nit,
	}, nil
}

// gradientDescent implements gradient descent with numerical gradients.
func gradientDescent(f func([]float64) float64, x0 []float64) (*OptResult, error) {
	n := len(x0)
	if n == 0 {
		return nil, errors.New("scigo: x0 must not be empty")
	}

	const (
		maxIter = 10000
		tol     = 1e-10
		eps     = 1e-8
	)

	x := make([]float64, n)
	copy(x, x0)
	fval := f(x)

	grad := make([]float64, n)
	xTmp := make([]float64, n)

	var nit int
	for nit = 0; nit < maxIter; nit++ {
		// Compute numerical gradient (central differences)
		for i := 0; i < n; i++ {
			copy(xTmp, x)
			h := eps * math.Max(1, math.Abs(x[i]))
			xTmp[i] = x[i] + h
			fp := f(xTmp)
			xTmp[i] = x[i] - h
			fm := f(xTmp)
			grad[i] = (fp - fm) / (2 * h)
		}

		// Gradient norm for convergence check
		gnorm := 0.0
		for _, g := range grad {
			gnorm += g * g
		}
		gnorm = math.Sqrt(gnorm)
		if gnorm < tol {
			break
		}

		// Line search with backtracking (Armijo condition)
		lr := 1.0
		for ls := 0; ls < 30; ls++ {
			copy(xTmp, x)
			for i := 0; i < n; i++ {
				xTmp[i] -= lr * grad[i]
			}
			fnew := f(xTmp)
			if fnew < fval-1e-4*lr*gnorm*gnorm {
				copy(x, xTmp)
				fval = fnew
				break
			}
			lr *= 0.5
			if ls == 29 {
				// Take the step anyway with small learning rate
				for i := 0; i < n; i++ {
					x[i] -= lr * grad[i]
				}
				fval = f(x)
			}
		}
	}

	return &OptResult{
		X:       x,
		Fun:     fval,
		Success: nit < maxIter,
		Nit:     nit,
	}, nil
}

// MinimizeScalar finds a local minimum of a scalar function within bounds using Brent's method.
func MinimizeScalar(f func(float64) float64, bounds [2]float64) (*ScalarResult, error) {
	a, b := bounds[0], bounds[1]
	if a >= b {
		return nil, errors.New("scigo: bounds[0] must be less than bounds[1]")
	}

	const (
		maxIter = 500
		tol     = 1e-12
		goldenR = 0.3819660112501051 // (3 - sqrt(5)) / 2
		sqrtEps = 1.4901161193847656e-08
	)

	x := a + goldenR*(b-a)
	w, v := x, x
	fx := f(x)
	fw, fv := fx, fx
	d, e := 0.0, 0.0

	var nit int
	for nit = 0; nit < maxIter; nit++ {
		mid := 0.5 * (a + b)
		tol1 := sqrtEps*math.Abs(x) + tol/3
		tol2 := 2 * tol1

		if math.Abs(x-mid) <= tol2-0.5*(b-a) {
			return &ScalarResult{X: x, Fun: fx, Success: true, Nit: nit}, nil
		}

		// Try parabolic interpolation
		var u float64
		useGolden := true
		if math.Abs(e) > tol1 {
			r := (x - w) * (fx - fv)
			q := (x - v) * (fx - fw)
			p := (x-v)*q - (x-w)*r
			q = 2 * (q - r)
			if q > 0 {
				p = -p
			} else {
				q = -q
			}
			r = e
			e = d
			if math.Abs(p) < math.Abs(0.5*q*r) && p > q*(a-x) && p < q*(b-x) {
				d = p / q
				u = x + d
				if u-a < tol2 || b-u < tol2 {
					if x < mid {
						d = tol1
					} else {
						d = -tol1
					}
				}
				useGolden = false
			}
		}

		if useGolden {
			if x < mid {
				e = b - x
			} else {
				e = a - x
			}
			d = goldenR * e
		}

		if math.Abs(d) >= tol1 {
			u = x + d
		} else {
			if d > 0 {
				u = x + tol1
			} else {
				u = x - tol1
			}
		}

		fu := f(u)

		if fu <= fx {
			if u < x {
				b = x
			} else {
				a = x
			}
			v, w, x = w, x, u
			fv, fw, fx = fw, fx, fu
		} else {
			if u < x {
				a = u
			} else {
				b = u
			}
			if fu <= fw || w == x {
				v, w = w, u
				fv, fw = fw, fu
			} else if fu <= fv || v == x || v == w {
				v = u
				fv = fu
			}
		}
	}

	return &ScalarResult{X: x, Fun: fx, Success: false, Nit: nit}, nil
}

// RootScalar finds a root of f in the given bracket using Brent's method.
// Requires f(bracket[0]) and f(bracket[1]) to have opposite signs.
func RootScalar(f func(float64) float64, bracket [2]float64) (float64, error) {
	a, b := bracket[0], bracket[1]
	fa, fb := f(a), f(b)

	if fa*fb > 0 {
		return 0, errors.New("scigo: root is not bracketed, f(a) and f(b) must have opposite signs")
	}
	if fa == 0 {
		return a, nil
	}
	if fb == 0 {
		return b, nil
	}

	const (
		maxIter = 500
		tol     = 1e-14
	)

	c, fc := a, fa
	d := b - a
	e := d

	for i := 0; i < maxIter; i++ {
		if fb*fc > 0 {
			c, fc = a, fa
			d = b - a
			e = d
		}
		if math.Abs(fc) < math.Abs(fb) {
			a, b, c = b, c, b
			fa, fb, fc = fb, fc, fb
		}

		tol1 := 2*1e-16*math.Abs(b) + 0.5*tol
		m := 0.5 * (c - b)

		if math.Abs(m) <= tol1 || fb == 0 {
			return b, nil
		}

		if math.Abs(e) >= tol1 && math.Abs(fa) > math.Abs(fb) {
			// Try inverse quadratic interpolation
			var p, q float64
			s := fb / fa
			if a == c {
				p = 2 * m * s
				q = 1 - s
			} else {
				q = fa / fc
				r := fb / fc
				p = s * (2*m*q*(q-r) - (b-a)*(r-1))
				q = (q - 1) * (r - 1) * (s - 1)
			}
			if p > 0 {
				q = -q
			} else {
				p = -p
			}
			if 2*p < math.Min(3*m*q-math.Abs(tol1*q), math.Abs(e*q)) {
				e = d
				d = p / q
			} else {
				d = m
				e = m
			}
		} else {
			d = m
			e = m
		}

		a = b
		fa = fb
		if math.Abs(d) > tol1 {
			b += d
		} else {
			if m > 0 {
				b += tol1
			} else {
				b -= tol1
			}
		}
		fb = f(b)
	}

	return b, nil
}
