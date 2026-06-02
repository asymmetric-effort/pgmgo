package structure_score

import (
	"fmt"
	"math"
	"sort"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// conditionalGaussianLL computes the conditional Gaussian log-likelihood for
// a continuous variable given a mix of discrete and continuous parents.
//
// It splits the data by discrete parent configurations (strata), computes a
// Gaussian regression of the variable on the continuous parents within each
// stratum, and sums the log-likelihoods.
//
// Returns the total log-likelihood and the number of free parameters.
func conditionalGaussianLL(variable string, discreteParents, continuousParents []string, data *tabgo.DataFrame) (ll float64, numParams int) {
	N := data.Len()
	if N == 0 {
		return 0, 0
	}

	yVals := data.Column(variable).Float64()

	// Parameters per stratum: intercept + len(continuousParents) + variance = len(cp) + 2
	paramsPerStratum := len(continuousParents) + 2

	if len(discreteParents) == 0 {
		// No discrete parents: just Gaussian regression on continuous parents.
		ll, numParams = gaussianLL(variable, continuousParents, data)
		return ll, numParams
	}

	// Build discrete parent configurations.
	sortedDiscrete := make([]string, len(discreteParents))
	copy(sortedDiscrete, discreteParents)
	sort.Strings(sortedDiscrete)

	discreteVals := make([][]string, len(sortedDiscrete))
	for i, p := range sortedDiscrete {
		raw := data.Column(p).Values()
		vals := make([]string, len(raw))
		for j, v := range raw {
			vals[j] = fmt.Sprintf("%v", v)
		}
		discreteVals[i] = vals
	}

	// Group rows by discrete parent config.
	strata := make(map[string][]int)
	for row := 0; row < N; row++ {
		key := ""
		for i, p := range sortedDiscrete {
			if i > 0 {
				key += ","
			}
			key += p + "=" + discreteVals[i][row]
		}
		strata[key] = append(strata[key], row)
	}

	// Get continuous parent data.
	cpData := make([][]float64, len(continuousParents))
	for i, p := range continuousParents {
		cpData[i] = data.Column(p).Float64()
	}

	numStrata := 0
	for _, rows := range strata {
		nStratum := len(rows)
		if nStratum == 0 {
			continue
		}
		numStrata++
		nf := float64(nStratum)

		// Extract y and continuous parent values for this stratum.
		yStratum := make([]float64, nStratum)
		for i, r := range rows {
			yStratum[i] = yVals[r]
		}

		if len(continuousParents) == 0 {
			// No continuous parents: just estimate mean and variance.
			mean := 0.0
			for _, v := range yStratum {
				mean += v
			}
			mean /= nf

			rss := 0.0
			for _, v := range yStratum {
				d := v - mean
				rss += d * d
			}
			variance := rss / nf
			if variance < 1e-300 {
				variance = 1e-300
			}
			ll += -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)
			continue
		}

		p := len(continuousParents)
		np1 := p + 1

		// Build X^T X and X^T y for this stratum.
		xtx := make([]float64, np1*np1)
		xty := make([]float64, np1)

		for idx, r := range rows {
			xi := make([]float64, np1)
			xi[0] = 1
			for j := 0; j < p; j++ {
				xi[j+1] = cpData[j][r]
			}
			for a := 0; a < np1; a++ {
				for b := 0; b < np1; b++ {
					xtx[a*np1+b] += xi[a] * xi[b]
				}
				xty[a] += xi[a] * yStratum[idx]
			}
		}

		beta := gaussSolve(xtx, xty, np1)

		rss := 0.0
		for idx, r := range rows {
			pred := beta[0]
			for j := 0; j < p; j++ {
				pred += beta[j+1] * cpData[j][r]
			}
			d := yStratum[idx] - pred
			rss += d * d
		}
		variance := rss / nf
		if variance < 1e-300 {
			variance = 1e-300
		}
		ll += -nf / 2.0 * (math.Log(2*math.Pi) + math.Log(variance) + 1)
	}

	numParams = numStrata * paramsPerStratum
	return ll, numParams
}

// isDiscreteColumn returns true if the column appears to contain discrete
// (non-float) values. It checks by trying to detect integer values.
func isDiscreteColumn(data *tabgo.DataFrame, name string) bool {
	vals := data.Column(name).Values()
	for _, v := range vals {
		switch v.(type) {
		case string:
			return true
		case int:
			return true
		case float64:
			// Check if it looks like an integer (common for discrete coded as float).
			f := v.(float64)
			if f != math.Floor(f) {
				return false
			}
		default:
			return true
		}
	}
	// All float64 values are integers -- treat as discrete.
	return true
}

// splitParents separates parents into discrete and continuous based on the data.
func splitParents(parents []string, data *tabgo.DataFrame) (discrete, continuous []string) {
	for _, p := range parents {
		if isDiscreteColumn(data, p) {
			discrete = append(discrete, p)
		} else {
			continuous = append(continuous, p)
		}
	}
	return
}

// BICCondGauss implements the conditional Gaussian BIC score.
// It handles mixed discrete and continuous parents by splitting the data into
// strata based on discrete parent configurations and fitting a Gaussian
// regression within each stratum.
type BICCondGauss struct{}

// NewBICCondGauss creates a new BICCondGauss scorer.
func NewBICCondGauss() *BICCondGauss {
	return &BICCondGauss{}
}

// LocalScore computes the conditional Gaussian BIC local score.
func (b *BICCondGauss) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}
	discrete, continuous := splitParents(parents, data)
	ll, numParams := conditionalGaussianLL(variable, discrete, continuous, data)
	return ll - 0.5*float64(numParams)*math.Log(N)
}

// Score computes the total conditional Gaussian BIC score.
func (b *BICCondGauss) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		total += b.LocalScore(v, parentMap[v], data)
	}
	return total
}

// AICCondGauss implements the conditional Gaussian AIC score.
type AICCondGauss struct{}

// NewAICCondGauss creates a new AICCondGauss scorer.
func NewAICCondGauss() *AICCondGauss {
	return &AICCondGauss{}
}

// LocalScore computes the conditional Gaussian AIC local score.
func (a *AICCondGauss) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	N := float64(data.Len())
	if N == 0 {
		return 0
	}
	discrete, continuous := splitParents(parents, data)
	ll, numParams := conditionalGaussianLL(variable, discrete, continuous, data)
	return ll - float64(numParams)
}

// Score computes the total conditional Gaussian AIC score.
func (a *AICCondGauss) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		total += a.LocalScore(v, parentMap[v], data)
	}
	return total
}

// LogLikelihoodCondGauss implements the pure conditional Gaussian log-likelihood score.
type LogLikelihoodCondGauss struct{}

// NewLogLikelihoodCondGauss creates a new LogLikelihoodCondGauss scorer.
func NewLogLikelihoodCondGauss() *LogLikelihoodCondGauss {
	return &LogLikelihoodCondGauss{}
}

// LocalScore computes the conditional Gaussian log-likelihood local score.
func (l *LogLikelihoodCondGauss) LocalScore(variable string, parents []string, data *tabgo.DataFrame) float64 {
	if data.Len() == 0 {
		return 0
	}
	discrete, continuous := splitParents(parents, data)
	ll, _ := conditionalGaussianLL(variable, discrete, continuous, data)
	return ll
}

// Score computes the total conditional Gaussian log-likelihood score.
func (l *LogLikelihoodCondGauss) Score(variables []string, parentMap map[string][]string, data *tabgo.DataFrame) float64 {
	total := 0.0
	for _, v := range variables {
		total += l.LocalScore(v, parentMap[v], data)
	}
	return total
}
