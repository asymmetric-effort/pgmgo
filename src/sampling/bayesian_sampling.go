package sampling

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// BayesianModelSampling provides sampling-based approximate inference methods
// for Bayesian networks.
type BayesianModelSampling struct {
	bn  *models.BayesianNetwork
	rng *numgo.RNG
}

// NewBayesianModelSampling creates a new BayesianModelSampling instance.
// It validates the model by calling bn.CheckModel().
func NewBayesianModelSampling(bn *models.BayesianNetwork, seed int64) (*BayesianModelSampling, error) {
	if bn == nil {
		return nil, fmt.Errorf("sampling: bayesian network must not be nil")
	}
	if err := bn.CheckModel(); err != nil {
		return nil, fmt.Errorf("sampling: invalid model: %w", err)
	}
	return &BayesianModelSampling{
		bn:  bn,
		rng: numgo.NewRNG(seed),
	}, nil
}

// topologicalOrder computes a topological ordering of the BN nodes using
// Kahn's algorithm over the public API of the BayesianNetwork.
func topologicalOrder(bn *models.BayesianNetwork) ([]string, error) {
	nodes := bn.Nodes()

	// Build in-degree map.
	inDegree := make(map[string]int, len(nodes))
	children := make(map[string][]string, len(nodes))
	for _, n := range nodes {
		inDegree[n] = len(bn.Parents(n))
		children[n] = bn.Children(n)
	}

	// Seed queue with zero in-degree nodes.
	var queue []string
	for _, n := range nodes {
		if inDegree[n] == 0 {
			queue = append(queue, n)
		}
	}

	var order []string
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		order = append(order, node)
		for _, child := range children[node] {
			inDegree[child]--
			if inDegree[child] == 0 {
				queue = append(queue, child)
			}
		}
	}

	if len(order) != len(nodes) {
		return nil, fmt.Errorf("sampling: cycle detected in bayesian network")
	}
	return order, nil
}

// sampleFromCPD samples a single value from the CPD for a variable given the
// current assignment of parent values. It extracts the conditional distribution
// column for the parent configuration and samples from it using the RNG.
func (bms *BayesianModelSampling) sampleFromCPD(cpd *factors.TabularCPD, assignment map[string]int) int {
	variableCard := cpd.VariableCard()
	evidence := cpd.Evidence()
	evidenceCard := cpd.EvidenceCard()

	// Compute the parent configuration index (column index in the CPD table).
	// Parent configs are in row-major order of evidence variables.
	parentConfig := 0
	if len(evidence) > 0 {
		stride := 1
		for i := len(evidence) - 1; i >= 0; i-- {
			parentConfig += assignment[evidence[i]] * stride
			stride *= evidenceCard[i]
		}
	}

	// Extract the conditional distribution for this parent configuration.
	// The factor's flat layout: flat[childState * numParentConfigs + parentConfig]
	factor := cpd.ToFactor()
	data := factor.Values().Data()
	numParentConfigs := 1
	for _, ec := range evidenceCard {
		numParentConfigs *= ec
	}

	dist := make([]float64, variableCard)
	for childState := 0; childState < variableCard; childState++ {
		dist[childState] = data[childState*numParentConfigs+parentConfig]
	}

	// Sample from the distribution using cumulative sum and a uniform draw.
	u := bms.rng.Rand(1).Data()[0]
	cumSum := 0.0
	for i, p := range dist {
		cumSum += p
		if u < cumSum {
			return i
		}
	}
	// Fallback for floating-point edge case.
	return variableCard - 1
}

// cpdProbability returns P(variable=value | parents) from the CPD.
func cpdProbability(cpd *factors.TabularCPD, value int, assignment map[string]int) float64 {
	evidence := cpd.Evidence()
	evidenceCard := cpd.EvidenceCard()

	parentConfig := 0
	if len(evidence) > 0 {
		stride := 1
		for i := len(evidence) - 1; i >= 0; i-- {
			parentConfig += assignment[evidence[i]] * stride
			stride *= evidenceCard[i]
		}
	}

	numParentConfigs := 1
	for _, ec := range evidenceCard {
		numParentConfigs *= ec
	}

	factor := cpd.ToFactor()
	data := factor.Values().Data()
	return data[value*numParentConfigs+parentConfig]
}

// ForwardSample generates n samples from the Bayesian network by sampling each
// node in topological order from its CPD conditioned on parent values.
// Returns a DataFrame with columns = node names, rows = samples, values = int state indices.
func (bms *BayesianModelSampling) ForwardSample(n int) (*tabgo.DataFrame, error) {
	if n <= 0 {
		return nil, fmt.Errorf("sampling: n must be positive, got %d", n)
	}

	order, err := topologicalOrder(bms.bn)
	if err != nil {
		return nil, err
	}

	// Prepare column data: one slice per node.
	colData := make(map[string][]any, len(order))
	for _, node := range order {
		colData[node] = make([]any, n)
	}

	for i := 0; i < n; i++ {
		assignment := make(map[string]int, len(order))
		for _, node := range order {
			cpd := bms.bn.GetCPD(node)
			val := bms.sampleFromCPD(cpd, assignment)
			assignment[node] = val
			colData[node][i] = val
		}
	}

	// Build DataFrame.
	seriesMap := make(map[string]*tabgo.Series, len(order))
	for _, node := range order {
		seriesMap[node] = tabgo.NewSeries(node, colData[node])
	}
	return tabgo.NewDataFrame(seriesMap), nil
}

// RejectionSample generates n accepted samples from the Bayesian network
// that match the given evidence. It uses forward sampling and rejects any
// samples that do not match all evidence assignments.
func (bms *BayesianModelSampling) RejectionSample(n int, evidence map[string]int) (*tabgo.DataFrame, error) {
	if n <= 0 {
		return nil, fmt.Errorf("sampling: n must be positive, got %d", n)
	}

	// Validate evidence variables exist in the network.
	nodes := bms.bn.Nodes()
	nodeSet := make(map[string]bool, len(nodes))
	for _, nd := range nodes {
		nodeSet[nd] = true
	}
	for v := range evidence {
		if !nodeSet[v] {
			return nil, fmt.Errorf("sampling: evidence variable %q not in network", v)
		}
	}

	order, err := topologicalOrder(bms.bn)
	if err != nil {
		return nil, err
	}

	// Collect accepted samples.
	colData := make(map[string][]any, len(order))
	for _, node := range order {
		colData[node] = make([]any, 0, n)
	}

	accepted := 0
	for accepted < n {
		assignment := make(map[string]int, len(order))
		for _, node := range order {
			cpd := bms.bn.GetCPD(node)
			val := bms.sampleFromCPD(cpd, assignment)
			assignment[node] = val
		}

		// Check evidence.
		match := true
		for v, val := range evidence {
			if assignment[v] != val {
				match = false
				break
			}
		}
		if match {
			for _, node := range order {
				colData[node] = append(colData[node], assignment[node])
			}
			accepted++
		}
	}

	seriesMap := make(map[string]*tabgo.Series, len(order))
	for _, node := range order {
		seriesMap[node] = tabgo.NewSeries(node, colData[node])
	}
	return tabgo.NewDataFrame(seriesMap), nil
}

// LikelihoodWeightedSample generates n samples using likelihood weighting.
// Evidence nodes are fixed to their observed values; non-evidence nodes are
// sampled from their CPDs. The weight of each sample is the product of
// P(evidence_node = observed_value | parents) for all evidence nodes.
// Returns a DataFrame of samples and a slice of corresponding weights.
func (bms *BayesianModelSampling) LikelihoodWeightedSample(n int, evidence map[string]int) (*tabgo.DataFrame, []float64, error) {
	if n <= 0 {
		return nil, nil, fmt.Errorf("sampling: n must be positive, got %d", n)
	}

	// Validate evidence variables.
	nodes := bms.bn.Nodes()
	nodeSet := make(map[string]bool, len(nodes))
	for _, nd := range nodes {
		nodeSet[nd] = true
	}
	for v := range evidence {
		if !nodeSet[v] {
			return nil, nil, fmt.Errorf("sampling: evidence variable %q not in network", v)
		}
	}

	order, err := topologicalOrder(bms.bn)
	if err != nil {
		return nil, nil, err
	}

	colData := make(map[string][]any, len(order))
	for _, node := range order {
		colData[node] = make([]any, n)
	}
	weights := make([]float64, n)

	for i := 0; i < n; i++ {
		assignment := make(map[string]int, len(order))
		w := 1.0

		for _, node := range order {
			cpd := bms.bn.GetCPD(node)
			if evVal, isEvidence := evidence[node]; isEvidence {
				// Fix evidence node to its observed value.
				assignment[node] = evVal
				// Multiply weight by P(node=evVal | parents).
				w *= cpdProbability(cpd, evVal, assignment)
			} else {
				// Sample non-evidence node from its CPD.
				val := bms.sampleFromCPD(cpd, assignment)
				assignment[node] = val
			}
		}

		for _, node := range order {
			colData[node][i] = assignment[node]
		}
		weights[i] = w
	}

	seriesMap := make(map[string]*tabgo.Series, len(order))
	for _, node := range order {
		seriesMap[node] = tabgo.NewSeries(node, colData[node])
	}
	df := tabgo.NewDataFrame(seriesMap)
	return df, weights, nil
}
