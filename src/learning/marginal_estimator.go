package learning

import (
	"fmt"
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// MarginalEstimator estimates CPD parameters for a BayesianNetwork from
// marginal counts in the observed data and computes the log marginal
// likelihood of the data given the network structure.
type MarginalEstimator struct {
	bn   *models.BayesianNetwork
	data *tabgo.DataFrame
}

// NewMarginalEstimator creates a new MarginalEstimator.
func NewMarginalEstimator(bn *models.BayesianNetwork, data *tabgo.DataFrame) *MarginalEstimator {
	return &MarginalEstimator{
		bn:   bn,
		data: data,
	}
}

// Estimate computes the CPD for each node from marginal counts in the data.
// For each node, it counts occurrences of each (node_value, parent_values)
// combination, normalizes per parent configuration, and stores the resulting
// TabularCPD in the network.
func (me *MarginalEstimator) Estimate() error {
	if me.bn == nil {
		return fmt.Errorf("learning: BayesianNetwork is nil")
	}
	if me.data == nil {
		return fmt.Errorf("learning: data is nil")
	}

	nodes := me.bn.Nodes()
	if len(nodes) == 0 {
		return fmt.Errorf("learning: BayesianNetwork has no nodes")
	}

	// Validate columns.
	dataColumns := make(map[string]bool)
	for _, c := range me.data.Columns() {
		dataColumns[c] = true
	}
	for _, node := range nodes {
		if !dataColumns[node] {
			return fmt.Errorf("learning: data is missing column for node %q", node)
		}
	}

	nRows := me.data.Len()

	// Pre-extract column data.
	colData := make(map[string][]int, len(nodes))
	for _, node := range nodes {
		colData[node] = me.data.Column(node).Int()
	}

	// Determine cardinalities.
	cardMap := make(map[string]int, len(nodes))
	for _, node := range nodes {
		cardMap[node] = maxVal(colData[node]) + 1
		if cardMap[node] < 1 {
			cardMap[node] = 1
		}
	}

	for _, node := range nodes {
		parents := me.bn.Parents(node)
		nodeCard := cardMap[node]

		parentCards := make([]int, len(parents))
		numParentConfigs := 1
		for i, p := range parents {
			parentCards[i] = cardMap[p]
			numParentConfigs *= cardMap[p]
		}

		// Count occurrences: counts[childState][parentConfig].
		counts := make([][]float64, nodeCard)
		for cs := 0; cs < nodeCard; cs++ {
			counts[cs] = make([]float64, numParentConfigs)
		}

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

			counts[childState][pc]++
		}

		// Normalize each column.
		for pc := 0; pc < numParentConfigs; pc++ {
			colSum := 0.0
			for cs := 0; cs < nodeCard; cs++ {
				colSum += counts[cs][pc]
			}
			if colSum == 0 {
				uniform := 1.0 / float64(nodeCard)
				for cs := 0; cs < nodeCard; cs++ {
					counts[cs][pc] = uniform
				}
			} else {
				for cs := 0; cs < nodeCard; cs++ {
					counts[cs][pc] /= colSum
				}
			}
		}

		cpd, err := factors.NewTabularCPD(node, nodeCard, counts, parents, parentCards)
		if err != nil {
			return fmt.Errorf("learning: failed to create CPD for %q: %w", node, err)
		}
		if err := me.bn.AddCPD(cpd); err != nil {
			return fmt.Errorf("learning: failed to add CPD for %q: %w", node, err)
		}
	}

	return nil
}

// MarginalLikelihood computes the log marginal likelihood of the data given
// the network structure and estimated CPDs. This is computed as:
//
//	log P(D|G) = sum over data rows of sum over nodes of log P(node | parents)
//
// The CPDs must have been estimated (via Estimate) before calling this method.
func (me *MarginalEstimator) MarginalLikelihood() (float64, error) {
	if me.bn == nil {
		return 0, fmt.Errorf("learning: BayesianNetwork is nil")
	}
	if me.data == nil {
		return 0, fmt.Errorf("learning: data is nil")
	}

	nodes := me.bn.Nodes()
	nRows := me.data.Len()
	if nRows == 0 {
		return 0, nil
	}

	// Pre-extract column data.
	colData := make(map[string][]int, len(nodes))
	for _, node := range nodes {
		colData[node] = me.data.Column(node).Int()
	}

	logLikelihood := 0.0

	for _, node := range nodes {
		cpd := me.bn.GetCPD(node)
		if cpd == nil {
			return 0, fmt.Errorf("learning: no CPD found for node %q; call Estimate first", node)
		}

		parents := me.bn.Parents(node)
		nodeCard := cpd.VariableCard()
		evCard := cpd.EvidenceCard()
		numPC := 1
		for _, ec := range evCard {
			numPC *= ec
		}

		// Get the flat CPD values.
		flatVals := cpd.ToFactor().Values().Data()

		for row := 0; row < nRows; row++ {
			childState := colData[node][row]
			if childState < 0 || childState >= nodeCard {
				continue
			}

			pc := 0
			valid := true
			for i, p := range parents {
				pv := colData[p][row]
				if pv < 0 || pv >= evCard[i] {
					valid = false
					break
				}
				pc = pc*evCard[i] + pv
			}
			if !valid {
				continue
			}

			prob := flatVals[childState*numPC+pc]
			if prob <= 0 {
				return math.Inf(-1), nil
			}
			logLikelihood += math.Log(prob)
		}
	}

	return logLikelihood, nil
}
