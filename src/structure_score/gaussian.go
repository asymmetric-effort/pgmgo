package structure_score

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// gaussianLL computes the Gaussian log-likelihood for a variable given its parents.
// It performs OLS regression of the variable on its parents and computes:
//
//	LL = -N/2 * ln(2*pi*var) - sum((x-mu)^2) / (2*var)
//	   = -N/2 * (ln(2*pi) + ln(var) + 1)
//
// where var is the residual variance (MLE estimate: RSS/N).
func gaussianLL(variable string, parents []string, data *tabgo.DataFrame) (ll float64, numParams int) {
	N := data.Len()
	if N == 0 {
		return 0, 0
	}
	nf := float64(N)

	yVals := data.Column(variable).Float64()

	// Number of parameters: intercept + len(parents) + 1 (variance)
	numParams = len(parents) + 2

	if len(parents) == 0 {
		// No parents: estimate mean and variance.
		mean := 0.0
		for _, v := range yVals {
			mean += v
		}
		mean /= nf

		rss := 0.0
		for _, v := range yVals {
			d := v - mean
			rss += d * d
		}
		variance := rss / nf
		if variance < 1e-300 {
			variance = 1e-300
		}

		// LL = -N/2 * (ln(2*pi) + ln(var) + 1)
		ll = -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)
		return ll, numParams
	}

	// OLS regression: y = X*beta + epsilon.
	// Build predictor columns.
	p := len(parents)
	np1 := p + 1 // including intercept

	// Compute X^T X and X^T y.
	xtx := make([]float64, np1*np1)
	xty := make([]float64, np1)

	parentData := make([][]float64, p)
	for i, pName := range parents {
		parentData[i] = data.Column(pName).Float64()
	}

	for row := 0; row < N; row++ {
		// Row of X: [1, parent1[row], parent2[row], ...]
		xi := make([]float64, np1)
		xi[0] = 1
		for j := 0; j < p; j++ {
			xi[j+1] = parentData[j][row]
		}
		for a := 0; a < np1; a++ {
			for b := 0; b < np1; b++ {
				xtx[a*np1+b] += xi[a] * xi[b]
			}
			xty[a] += xi[a] * yVals[row]
		}
	}

	// Solve via Gaussian elimination.
	beta := gaussSolve(xtx, xty, np1)

	// Compute residual sum of squares.
	rss := 0.0
	for row := 0; row < N; row++ {
		pred := beta[0]
		for j := 0; j < p; j++ {
			pred += beta[j+1] * parentData[j][row]
		}
		d := yVals[row] - pred
		rss += d * d
	}
	variance := rss / nf
	if variance < 1e-300 {
		variance = 1e-300
	}

	ll = -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)
	return ll, numParams
}

// gaussSolve solves A*x = b via Gaussian elimination with partial pivoting.
func gaussSolve(A []float64, b []float64, n int) []float64 {
	a := make([]float64, n*n)
	copy(a, A)
	x := make([]float64, n)
	copy(x, b)

	for col := 0; col < n; col++ {
		maxVal := math.Abs(a[col*n+col])
		maxRow := col
		for row := col + 1; row < n; row++ {
			if v := math.Abs(a[row*n+col]); v > maxVal {
				maxVal = v
				maxRow = row
			}
		}
		if maxRow != col {
			for j := 0; j < n; j++ {
				a[col*n+j], a[maxRow*n+j] = a[maxRow*n+j], a[col*n+j]
			}
			x[col], x[maxRow] = x[maxRow], x[col]
		}
		pivot := a[col*n+col]
		if math.Abs(pivot) < 1e-15 {
			continue
		}
		for row := col + 1; row < n; row++ {
			factor := a[row*n+col] / pivot
			for j := col; j < n; j++ {
				a[row*n+j] -= factor * a[col*n+j]
			}
			x[row] -= factor * x[col]
		}
	}

	for col := n - 1; col >= 0; col-- {
		if math.Abs(a[col*n+col]) < 1e-15 {
			x[col] = 0
			continue
		}
		for j := col + 1; j < n; j++ {
			x[col] -= a[col*n+j] * x[j]
		}
		x[col] /= a[col*n+col]
	}

	return x
}

// BICGauss implements the Gaussian BIC score.
// BIC_Gauss = LL - 0.5 * k * ln(N)
type BICGauss struct{}

// NewBICGauss creates a new BICGauss scorer.
func NewBICGauss() *BICGauss {
	return &BICGauss{}
}

// LocalScore computes the Gaussian BIC local score.
func (b *BICGauss) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}
	ll, numParams := gaussianLL(variable, parents, data)
	return ll - 0.5*float64(numParams)*math.Log(N)
}

// Score computes the total Gaussian BIC score.
func (b *BICGauss) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		total += b.LocalScore(v, parentMap[v], data)
	}
	return total
}

// AICGauss implements the Gaussian AIC score.
// AIC_Gauss = LL - k
type AICGauss struct{}

// NewAICGauss creates a new AICGauss scorer.
func NewAICGauss() *AICGauss {
	return &AICGauss{}
}

// LocalScore computes the Gaussian AIC local score.
func (a *AICGauss) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}
	ll, numParams := gaussianLL(variable, parents, data)
	return ll - float64(numParams)
}

// Score computes the total Gaussian AIC score.
func (a *AICGauss) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		total += a.LocalScore(v, parentMap[v], data)
	}
	return total
}

// LogLikelihoodGauss implements the pure Gaussian log-likelihood score with no penalty.
type LogLikelihoodGauss struct{}

// NewLogLikelihoodGauss creates a new LogLikelihoodGauss scorer.
func NewLogLikelihoodGauss() *LogLikelihoodGauss {
	return &LogLikelihoodGauss{}
}

// LocalScore computes the Gaussian log-likelihood local score.
func (l *LogLikelihoodGauss) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}
	ll, _ := gaussianLL(variable, parents, data)
	return ll
}

// Score computes the total Gaussian log-likelihood score.
func (l *LogLikelihoodGauss) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		total += l.LocalScore(v, parentMap[v], data)
	}
	return total
}
