package learning

import (
	"fmt"
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// MirrorDescentEstimator estimates CPD parameters for a BayesianNetwork using
// mirror descent on the KL divergence (negative log-likelihood). Parameters
// are maintained in log-space and projected back to the probability simplex
// via exponentiation and normalization at each step.
type MirrorDescentEstimator struct {
	bn      *models.BayesianNetwork
	data    *tabgo.DataFrame
	lr      float64
	maxIter int
	tol     float64
}

// NewMirrorDescentEstimator creates a new MirrorDescentEstimator.
//
// bn is the Bayesian network whose CPD parameters will be estimated.
// data is the observed dataset with columns matching node names and integer
// state index values. lr is the learning rate. maxIter is the maximum number
// of mirror descent iterations.
func NewMirrorDescentEstimator(bn *models.BayesianNetwork, data *tabgo.DataFrame, lr float64, maxIter int) *MirrorDescentEstimator {
	return &MirrorDescentEstimator{
		bn:      bn,
		data:    data,
		lr:      lr,
		maxIter: maxIter,
		tol:     1e-8,
	}
}

// Estimate runs mirror descent to estimate CPD parameters for all nodes.
//
// Algorithm:
//  1. Initialize CPD parameters uniformly in log-space.
//  2. For each iteration, compute gradient of negative log-likelihood.
//  3. Mirror descent update: theta_new = log(params) - lr * gradient, then
//     params = exp(theta_new) / sum(exp(theta_new)) per parent configuration.
//  4. Converge when maximum parameter change < tolerance.
func (md *MirrorDescentEstimator) Estimate() error {
	if md.bn == nil {
		return fmt.Errorf("learning: BayesianNetwork is nil")
	}
	if md.data == nil {
		return fmt.Errorf("learning: data is nil")
	}

	nodes := md.bn.Nodes()
	if len(nodes) == 0 {
		return fmt.Errorf("learning: BayesianNetwork has no nodes")
	}

	// Validate columns.
	dataColumns := make(map[string]bool)
	for _, c := range md.data.Columns() {
		dataColumns[c] = true
	}
	for _, node := range nodes {
		if !dataColumns[node] {
			return fmt.Errorf("learning: data is missing column for node %q", node)
		}
	}

	nRows := md.data.Len()
	if nRows == 0 {
		// With no data, set uniform CPDs.
		return md.setUniformCPDs(nodes)
	}

	// Pre-extract column data as int slices.
	colData := make(map[string][]int, len(nodes))
	for _, node := range nodes {
		colData[node] = md.data.Column(node).Int()
	}

	// Determine cardinalities from data.
	cardMap := make(map[string]int, len(nodes))
	for _, node := range nodes {
		cardMap[node] = maxVal(colData[node]) + 1
		if cardMap[node] < 1 {
			cardMap[node] = 1
		}
	}

	// For each node, run mirror descent independently.
	for _, node := range nodes {
		parents := md.bn.Parents(node)
		nodeCard := cardMap[node]

		parentCards := make([]int, len(parents))
		numParentConfigs := 1
		for i, p := range parents {
			parentCards[i] = cardMap[p]
			numParentConfigs *= cardMap[p]
		}

		// Initialize parameters uniformly.
		// params[childState][parentConfig]
		params := make([][]float64, nodeCard)
		for cs := 0; cs < nodeCard; cs++ {
			params[cs] = make([]float64, numParentConfigs)
			for pc := 0; pc < numParentConfigs; pc++ {
				params[cs][pc] = 1.0 / float64(nodeCard)
			}
		}

		// Pre-compute parent configs and count occurrences per row.
		parentConfigs := make([]int, nRows)
		validRows := make([]bool, nRows)
		for row := 0; row < nRows; row++ {
			childState := colData[node][row]
			if childState < 0 || childState >= nodeCard {
				continue
			}

			pc := 0
			valid := true
			for i, p := range parents {
				pv := colData[p][row]
				if pv < 0 || pv >= parentCards[i] {
					valid = false
					break
				}
				pc = pc*parentCards[i] + pv
			}
			if !valid {
				continue
			}
			parentConfigs[row] = pc
			validRows[row] = true
		}

		// Pre-compute sufficient statistics: counts[cs][pc].
		counts := make([][]float64, nodeCard)
		for cs := 0; cs < nodeCard; cs++ {
			counts[cs] = make([]float64, numParentConfigs)
		}
		for row := 0; row < nRows; row++ {
			if !validRows[row] {
				continue
			}
			cs := colData[node][row]
			pc := parentConfigs[row]
			counts[cs][pc]++
		}

		// Mirror descent iterations using the exponentiated gradient
		// (mirror map = negative entropy on the simplex).
		//
		// The gradient of the NLL with respect to the natural (log)
		// parameters eta[cs][pc] = log(params[cs][pc]) is:
		//   dNLL/d_eta[cs][pc] = params[cs][pc] * N_pc - counts[cs][pc]
		// where N_pc = total count for parent config pc.
		//
		// Mirror descent update in dual space:
		//   eta_new = eta_old - lr * gradient
		//   params_new = softmax(eta_new) per parent config
		//
		// Initialize eta from log of uniform params.
		eta := make([][]float64, nodeCard)
		for cs := 0; cs < nodeCard; cs++ {
			eta[cs] = make([]float64, numParentConfigs)
			for pc := 0; pc < numParentConfigs; pc++ {
				eta[cs][pc] = 0.0 // log(1/K) is the same for all, shift to 0
			}
		}

		// Total counts per parent config.
		pcTotals := make([]float64, numParentConfigs)
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < nodeCard; cs++ {
				pcTotals[pc] += counts[cs][pc]
			}
		}

		for iter := 0; iter < md.maxIter; iter++ {
			newParams := make([][]float64, nodeCard)
			for cs := 0; cs < nodeCard; cs++ {
				newParams[cs] = make([]float64, numParentConfigs)
			}

			maxDelta := 0.0

			for pc := 0; pc < numParentConfigs; pc++ {
				if pcTotals[pc] == 0 {
					// No data for this parent config; keep uniform.
					for cs := 0; cs < nodeCard; cs++ {
						newParams[cs][pc] = 1.0 / float64(nodeCard)
					}
					continue
				}

				// Compute normalized gradient and update eta.
				for cs := 0; cs < nodeCard; cs++ {
					grad := params[cs][pc] - counts[cs][pc]/pcTotals[pc]
					eta[cs][pc] -= md.lr * grad
				}

				// Softmax to get new params.
				maxEta := eta[0][pc]
				for cs := 1; cs < nodeCard; cs++ {
					if eta[cs][pc] > maxEta {
						maxEta = eta[cs][pc]
					}
				}
				sumExp := 0.0
				for cs := 0; cs < nodeCard; cs++ {
					newParams[cs][pc] = math.Exp(eta[cs][pc] - maxEta)
					sumExp += newParams[cs][pc]
				}
				for cs := 0; cs < nodeCard; cs++ {
					newParams[cs][pc] /= sumExp
					delta := math.Abs(newParams[cs][pc] - params[cs][pc])
					if delta > maxDelta {
						maxDelta = delta
					}
				}
			}

			params = newParams

			if maxDelta < md.tol {
				break
			}
		}

		// Store the CPD.
		cpd, err := factors.NewTabularCPD(node, nodeCard, params, parents, parentCards)
		if err != nil {
			return fmt.Errorf("learning: failed to create CPD for %q: %w", node, err)
		}
		if err := md.bn.AddCPD(cpd); err != nil {
			return fmt.Errorf("learning: failed to add CPD for %q: %w", node, err)
		}
	}

	return nil
}

// GetParameters returns the CPD for the given node, or an error if none has
// been estimated.
func (md *MirrorDescentEstimator) GetParameters(node string) (*factors.TabularCPD, error) {
	if md.bn == nil {
		return nil, fmt.Errorf("learning: BayesianNetwork is nil")
	}
	cpd := md.bn.GetCPD(node)
	if cpd == nil {
		return nil, fmt.Errorf("learning: no CPD found for node %q", node)
	}
	return cpd, nil
}

// setUniformCPDs sets uniform CPDs for all nodes (used when data is empty).
func (md *MirrorDescentEstimator) setUniformCPDs(nodes []string) error {
	for _, node := range nodes {
		parents := md.bn.Parents(node)
		parentCards := make([]int, len(parents))
		numPC := 1
		for i := range parents {
			parentCards[i] = 1
			numPC *= 1
		}
		values := [][]float64{{1.0}}
		cpd, err := factors.NewTabularCPD(node, 1, values, parents, parentCards)
		if err != nil {
			return fmt.Errorf("learning: failed to create uniform CPD for %q: %w", node, err)
		}
		if err := md.bn.AddCPD(cpd); err != nil {
			return fmt.Errorf("learning: failed to add CPD for %q: %w", node, err)
		}
	}
	return nil
}
