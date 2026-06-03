package models

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// IndependenceAssertion represents a conditional independence statement
// X _|_ Y | Z for use with IsIMap. Event1 and Event2 are the two variable
// sets, and Given is the conditioning set.
type IndependenceAssertion struct {
	Event1 []string
	Event2 []string
	Given  []string
}

// LinearGaussianBayesianNetwork is a Bayesian network where every node has a
// LinearGaussianCPD instead of a TabularCPD. It embeds *BayesianNetwork for
// graph structure (nodes, edges) and stores LG CPDs in a separate map.
type LinearGaussianBayesianNetwork struct {
	*BayesianNetwork
	lgCPDs map[string]*factors.LinearGaussianCPD
}

// NewLinearGaussianBayesianNetwork creates a new empty LinearGaussianBayesianNetwork.
func NewLinearGaussianBayesianNetwork() *LinearGaussianBayesianNetwork {
	return &LinearGaussianBayesianNetwork{
		BayesianNetwork: NewBayesianNetwork(),
		lgCPDs:          make(map[string]*factors.LinearGaussianCPD),
	}
}

// AddLinearGaussianCPD stores a LinearGaussianCPD for its variable.
// It validates that the variable exists in the graph and that the CPD's
// evidence matches the node's parents in the DAG.
func (lgbn *LinearGaussianBayesianNetwork) AddLinearGaussianCPD(cpd *factors.LinearGaussianCPD) error {
	if cpd == nil {
		return fmt.Errorf("models: cpd must not be nil")
	}

	v := cpd.Variable()
	if !lgbn.dag.HasNode(v) {
		return fmt.Errorf("models: variable %q is not a node in the network", v)
	}

	// Verify evidence matches parents.
	parents := lgbn.dag.Parents(v) // sorted
	evidence := cpd.Evidence()
	sortedEvidence := make([]string, len(evidence))
	copy(sortedEvidence, evidence)
	sort.Strings(sortedEvidence)

	if len(parents) != len(sortedEvidence) {
		return fmt.Errorf("models: LG CPD for %q has evidence %v but node has parents %v",
			v, evidence, parents)
	}
	for i := range parents {
		if parents[i] != sortedEvidence[i] {
			return fmt.Errorf("models: LG CPD for %q has evidence %v but node has parents %v",
				v, evidence, parents)
		}
	}

	lgbn.lgCPDs[v] = cpd
	return nil
}

// GetLinearGaussianCPD returns the LinearGaussianCPD for the given variable,
// or nil if none is set.
func (lgbn *LinearGaussianBayesianNetwork) GetLinearGaussianCPD(variable string) *factors.LinearGaussianCPD {
	return lgbn.lgCPDs[variable]
}

// CheckModel validates the LinearGaussianBayesianNetwork:
//  1. Every node has a LinearGaussianCPD.
//  2. Each CPD's evidence matches the node's parents in the DAG.
//  3. Each CPD passes Validate().
func (lgbn *LinearGaussianBayesianNetwork) CheckModel() error {
	nodes := lgbn.dag.Nodes()

	for _, node := range nodes {
		cpd, ok := lgbn.lgCPDs[node]
		if !ok {
			return fmt.Errorf("models: node %q has no LinearGaussianCPD", node)
		}

		// Check evidence matches parents.
		parents := lgbn.dag.Parents(node) // sorted
		evidence := cpd.Evidence()
		sortedEvidence := make([]string, len(evidence))
		copy(sortedEvidence, evidence)
		sort.Strings(sortedEvidence)

		if len(parents) != len(sortedEvidence) {
			return fmt.Errorf("models: LG CPD for %q has evidence %v but node has parents %v",
				node, evidence, parents)
		}
		for i := range parents {
			if parents[i] != sortedEvidence[i] {
				return fmt.Errorf("models: LG CPD for %q has evidence %v but node has parents %v",
					node, evidence, parents)
			}
		}

		if err := cpd.Validate(); err != nil {
			return fmt.Errorf("models: LG CPD for %q failed validation: %w", node, err)
		}
	}
	return nil
}

// Copy returns a deep copy of the LinearGaussianBayesianNetwork.
func (lgbn *LinearGaussianBayesianNetwork) Copy() *LinearGaussianBayesianNetwork {
	newLGCPDs := make(map[string]*factors.LinearGaussianCPD, len(lgbn.lgCPDs))
	for k, v := range lgbn.lgCPDs {
		newLGCPDs[k] = v.Copy()
	}
	return &LinearGaussianBayesianNetwork{
		BayesianNetwork: lgbn.BayesianNetwork.Copy(),
		lgCPDs:          newLGCPDs,
	}
}

// Save writes the LinearGaussianBayesianNetwork to a file in a simple text
// format (BIF-like for LG networks).
func (lgbn *LinearGaussianBayesianNetwork) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("models: Save: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	if _, err := fmt.Fprintf(w, "network lg_bayesian_network {\n}\n"); err != nil {
		return err
	}

	nodes := lgbn.Nodes()
	for _, node := range nodes {
		cpd := lgbn.lgCPDs[node]
		if cpd == nil {
			return fmt.Errorf("models: variable %q has no LG CPD", node)
		}
		if _, err := fmt.Fprintf(w, "\nvariable %s {\n  type continuous;\n}\n", node); err != nil {
			return err
		}
	}

	for _, node := range nodes {
		cpd := lgbn.lgCPDs[node]
		evidence := cpd.Evidence()
		betas := cpd.Betas()

		if _, err := fmt.Fprintf(w, "\ndistribution %s", node); err != nil {
			return err
		}
		if len(evidence) > 0 {
			if _, err := fmt.Fprintf(w, " | %s", strings.Join(evidence, ", ")); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprintf(w, " {\n"); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "  mean %g;\n", cpd.Mean()); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "  variance %g;\n", cpd.Variance()); err != nil {
			return err
		}
		if len(betas) > 0 {
			betaStrs := make([]string, len(betas))
			for i, b := range betas {
				betaStrs[i] = fmt.Sprintf("%g", b)
			}
			if _, err := fmt.Fprintf(w, "  betas %s;\n", strings.Join(betaStrs, ", ")); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprintf(w, "}\n"); err != nil {
			return err
		}
	}

	return w.Flush()
}

// Load reads a LinearGaussianBayesianNetwork from a file written by Save.
func LoadLinearGaussianBayesianNetwork(filename string) (*LinearGaussianBayesianNetwork, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("models: Load: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var lines []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "//") {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("models: Load: %w", err)
	}

	bn := NewLinearGaussianBayesianNetwork()

	// Parse variables and distributions.
	type distInfo struct {
		node     string
		evidence []string
		mean     float64
		variance float64
		betas    []float64
	}
	var distributions []distInfo
	var variables []string

	i := 0
	for i < len(lines) {
		tokens := strings.Fields(lines[i])
		if len(tokens) == 0 {
			i++
			continue
		}

		switch tokens[0] {
		case "network":
			// Skip to closing brace.
			for i < len(lines) && !strings.Contains(lines[i], "}") {
				i++
			}
			i++
		case "variable":
			if len(tokens) < 2 {
				return nil, fmt.Errorf("models: Load: malformed variable line")
			}
			name := strings.TrimRight(tokens[1], " {")
			variables = append(variables, name)
			if err := bn.AddNode(name); err != nil {
				return nil, err
			}
			// Skip to closing brace.
			for i < len(lines) && !strings.HasSuffix(lines[i], "}") {
				i++
			}
			i++
		case "distribution":
			if len(tokens) < 2 {
				return nil, fmt.Errorf("models: Load: malformed distribution line")
			}
			di := distInfo{node: tokens[1]}
			// Parse evidence from the header line.
			fullLine := lines[i]
			if idx := strings.Index(fullLine, "|"); idx >= 0 {
				evPart := fullLine[idx+1:]
				evPart = strings.TrimRight(evPart, " {")
				evPart = strings.TrimSpace(evPart)
				for _, ev := range strings.Split(evPart, ",") {
					ev = strings.TrimSpace(ev)
					if ev != "" {
						di.evidence = append(di.evidence, ev)
					}
				}
			}
			i++
			// Parse body until closing brace.
			for i < len(lines) && !strings.HasPrefix(lines[i], "}") {
				line := lines[i]
				line = strings.TrimRight(line, ";")
				line = strings.TrimSpace(line)
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					switch parts[0] {
					case "mean":
						di.mean, _ = strconv.ParseFloat(parts[1], 64)
					case "variance":
						di.variance, _ = strconv.ParseFloat(parts[1], 64)
					case "betas":
						betaStr := strings.Join(parts[1:], " ")
						for _, bs := range strings.Split(betaStr, ",") {
							bs = strings.TrimSpace(bs)
							if bs != "" {
								v, _ := strconv.ParseFloat(bs, 64)
								di.betas = append(di.betas, v)
							}
						}
					}
				}
				i++
			}
			i++ // skip closing brace
			distributions = append(distributions, di)
		default:
			i++
		}
	}

	// Add edges and CPDs.
	for _, di := range distributions {
		for _, ev := range di.evidence {
			if !bn.dag.HasEdge(ev, di.node) {
				if err := bn.AddEdge(ev, di.node); err != nil {
					return nil, err
				}
			}
		}
		cpd, err := factors.NewLinearGaussianCPD(di.node, di.mean, di.betas, di.variance, di.evidence)
		if err != nil {
			return nil, fmt.Errorf("models: Load CPD for %q: %w", di.node, err)
		}
		if err := bn.AddLinearGaussianCPD(cpd); err != nil {
			return nil, err
		}
	}

	return bn, nil
}

// RemoveCPDs removes all LinearGaussianCPDs from the network.
func (lgbn *LinearGaussianBayesianNetwork) RemoveCPDs() {
	lgbn.lgCPDs = make(map[string]*factors.LinearGaussianCPD)
}

// GetRandomCPDs generates and assigns random LinearGaussianCPDs for all nodes.
// Each node gets betas drawn uniformly from [-1, 1], mean from [-5, 5], and
// variance from (0, 2].
func (lgbn *LinearGaussianBayesianNetwork) GetRandomCPDs() error {
	nodes := lgbn.Nodes()
	for _, node := range nodes {
		parents := lgbn.Parents(node)
		betas := make([]float64, len(parents))
		for i := range betas {
			betas[i] = rand.Float64()*2 - 1 // [-1, 1]
		}
		mean := rand.Float64()*10 - 5       // [-5, 5]
		variance := rand.Float64()*2 + 0.01 // (0.01, 2.01]

		cpd, err := factors.NewLinearGaussianCPD(node, mean, betas, variance, parents)
		if err != nil {
			return fmt.Errorf("models: GetRandomCPDs for %q: %w", node, err)
		}
		lgbn.lgCPDs[node] = cpd
	}
	return nil
}

// ToJointGaussian computes the joint Gaussian distribution implied by the
// linear Gaussian BN. Returns the mean vector and covariance matrix, with
// variables in sorted order.
func (lgbn *LinearGaussianBayesianNetwork) ToJointGaussian() ([]float64, [][]float64, error) {
	if err := lgbn.CheckModel(); err != nil {
		return nil, nil, fmt.Errorf("models: cannot compute joint Gaussian: %w", err)
	}

	vars := lgbn.Nodes() // sorted
	n := len(vars)
	if n == 0 {
		return nil, nil, nil
	}

	varIdx := make(map[string]int, n)
	for i, v := range vars {
		varIdx[v] = i
	}

	// Build B (coefficient matrix) and intercepts/variances.
	B := make([][]float64, n)
	intercepts := make([]float64, n)
	errVar := make([]float64, n)
	for i := 0; i < n; i++ {
		B[i] = make([]float64, n)
	}

	for _, v := range vars {
		cpd := lgbn.lgCPDs[v]
		vi := varIdx[v]
		intercepts[vi] = cpd.Mean()
		errVar[vi] = cpd.Variance()
		evidence := cpd.Evidence()
		betas := cpd.Betas()
		for j, p := range evidence {
			pi := varIdx[p]
			B[vi][pi] = betas[j]
		}
	}

	// Compute (I - B)
	imb := make([][]float64, n)
	for i := 0; i < n; i++ {
		imb[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				imb[i][j] = 1.0 - B[i][j]
			} else {
				imb[i][j] = -B[i][j]
			}
		}
	}

	inv, err := invertMatrix(imb)
	if err != nil {
		return nil, nil, fmt.Errorf("models: failed to invert (I-B): %w", err)
	}

	// Mean vector: mu = (I-B)^{-1} * intercepts
	mu := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			mu[i] += inv[i][j] * intercepts[j]
		}
	}

	// Covariance: Sigma = (I-B)^{-1} * Psi * ((I-B)^{-1})^T
	psi := make([][]float64, n)
	for i := 0; i < n; i++ {
		psi[i] = make([]float64, n)
		psi[i][i] = errVar[i]
	}

	invPsi := matMul(inv, psi, n)
	invT := transpose(inv, n)
	sigma := matMul(invPsi, invT, n)

	return mu, sigma, nil
}

// LogLikelihood computes the log-likelihood of the data given the model.
// Each row's log-likelihood is the sum of log P(x_i | parents(x_i)) for
// each variable.
func (lgbn *LinearGaussianBayesianNetwork) LogLikelihood(data *tabgo.DataFrame) (float64, error) {
	if data == nil {
		return 0, fmt.Errorf("models: data must not be nil")
	}
	if err := lgbn.CheckModel(); err != nil {
		return 0, fmt.Errorf("models: invalid model: %w", err)
	}

	nRows := data.Len()
	if nRows == 0 {
		return 0, nil
	}

	nodes := lgbn.Nodes()
	colData := make(map[string][]float64, len(nodes))
	for _, node := range nodes {
		colData[node] = data.Column(node).Float64()
	}

	totalLL := 0.0
	for row := 0; row < nRows; row++ {
		for _, node := range nodes {
			cpd := lgbn.lgCPDs[node]
			parentVals := make(map[string]float64)
			for _, ev := range cpd.Evidence() {
				parentVals[ev] = colData[ev][row]
			}
			totalLL += cpd.LogPDF(colData[node][row], parentVals)
		}
	}

	return totalLL, nil
}

// Simulate samples continuous data from the linear Gaussian BN.
func (lgbn *LinearGaussianBayesianNetwork) Simulate(nSamples int) (*tabgo.DataFrame, error) {
	if err := lgbn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: cannot simulate: %w", err)
	}
	if nSamples <= 0 {
		return nil, fmt.Errorf("models: nSamples must be positive, got %d", nSamples)
	}

	order, err := lgbn.dag.TopologicalOrder()
	if err != nil {
		return nil, fmt.Errorf("models: cannot get topological order: %w", err)
	}

	data := make(map[string][]float64, len(order))
	for _, v := range order {
		data[v] = make([]float64, nSamples)
	}

	for i := 0; i < nSamples; i++ {
		for _, v := range order {
			cpd := lgbn.lgCPDs[v]
			parentVals := make(map[string]float64)
			for _, ev := range cpd.Evidence() {
				parentVals[ev] = data[ev][i]
			}
			mu := cpd.ConditionalMean(parentVals)
			std := math.Sqrt(cpd.Variance())
			data[v][i] = mu + std*lgRandStdNormal()
		}
	}

	vars := lgbn.Nodes() // sorted
	rows := make([][]any, nSamples)
	for i := 0; i < nSamples; i++ {
		row := make([]any, len(vars))
		for j, v := range vars {
			row[j] = data[v][i]
		}
		rows[i] = row
	}

	return tabgo.NewDataFrameFromRows(vars, rows), nil
}

// lgRandStdNormal returns a sample from N(0,1) using Box-Muller.
func lgRandStdNormal() float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	for u1 == 0 {
		u1 = rand.Float64()
	}
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// GetCardinality returns an error for LinearGaussianBayesianNetwork since
// continuous variables do not have a finite cardinality.
func (lgbn *LinearGaussianBayesianNetwork) GetCardinality(node string) (int, error) {
	if !lgbn.dag.HasNode(node) {
		return 0, fmt.Errorf("models: node %q not in network", node)
	}
	return 0, fmt.Errorf("models: GetCardinality is not applicable for continuous variable %q", node)
}

// Fit estimates LinearGaussianCPD parameters from data using OLS per node.
func (lgbn *LinearGaussianBayesianNetwork) Fit(data *tabgo.DataFrame) error {
	if data == nil {
		return fmt.Errorf("models: data must not be nil")
	}
	nRows := data.Len()
	if nRows == 0 {
		return fmt.Errorf("models: data must not be empty")
	}

	nodes := lgbn.Nodes()
	colData := make(map[string][]float64, len(nodes))
	for _, node := range nodes {
		colData[node] = data.Column(node).Float64()
	}

	for _, node := range nodes {
		parents := lgbn.Parents(node)
		y := colData[node]

		if len(parents) == 0 {
			mean := 0.0
			for _, v := range y {
				mean += v
			}
			mean /= float64(nRows)

			variance := 0.0
			for _, v := range y {
				d := v - mean
				variance += d * d
			}
			if nRows > 1 {
				variance /= float64(nRows)
			}
			if variance <= 0 {
				variance = 1e-10
			}

			cpd, err := factors.NewLinearGaussianCPD(node, mean, nil, variance, nil)
			if err != nil {
				return fmt.Errorf("models: Fit CPD for %q: %w", node, err)
			}
			lgbn.lgCPDs[node] = cpd
			continue
		}

		nP := len(parents)
		k := nP + 1

		xtx := make([][]float64, k)
		for ii := range xtx {
			xtx[ii] = make([]float64, k)
		}
		xty := make([]float64, k)

		for row := 0; row < nRows; row++ {
			x := make([]float64, k)
			x[0] = 1.0
			for j, p := range parents {
				x[j+1] = colData[p][row]
			}
			for ii := 0; ii < k; ii++ {
				for jj := 0; jj < k; jj++ {
					xtx[ii][jj] += x[ii] * x[jj]
				}
				xty[ii] += x[ii] * y[row]
			}
		}

		inv, err := invertMatrix(xtx)
		if err != nil {
			return fmt.Errorf("models: OLS for %q: %w", node, err)
		}

		beta := make([]float64, k)
		for ii := 0; ii < k; ii++ {
			for jj := 0; jj < k; jj++ {
				beta[ii] += inv[ii][jj] * xty[jj]
			}
		}

		intercept := beta[0]
		betas := beta[1:]

		variance := 0.0
		for row := 0; row < nRows; row++ {
			predicted := intercept
			for j, p := range parents {
				predicted += betas[j] * colData[p][row]
			}
			d := y[row] - predicted
			variance += d * d
		}
		if nRows > 1 {
			variance /= float64(nRows)
		}
		if variance <= 0 {
			variance = 1e-10
		}

		cpd, err := factors.NewLinearGaussianCPD(node, intercept, betas, variance, parents)
		if err != nil {
			return fmt.Errorf("models: Fit CPD for %q: %w", node, err)
		}
		lgbn.lgCPDs[node] = cpd
	}

	return nil
}

// PredictProbability returns the log-PDF of each row in data given the model.
// The result is a slice of length data.Len() where each element is the
// log P(row | model).
func (lgbn *LinearGaussianBayesianNetwork) PredictProbability(data *tabgo.DataFrame) ([]float64, error) {
	if data == nil {
		return nil, fmt.Errorf("models: data must not be nil")
	}
	if err := lgbn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: invalid model: %w", err)
	}

	nRows := data.Len()
	nodes := lgbn.Nodes()
	colData := make(map[string][]float64, len(nodes))
	for _, node := range nodes {
		colData[node] = data.Column(node).Float64()
	}

	result := make([]float64, nRows)
	for row := 0; row < nRows; row++ {
		logP := 0.0
		for _, node := range nodes {
			cpd := lgbn.lgCPDs[node]
			parentVals := make(map[string]float64)
			for _, ev := range cpd.Evidence() {
				parentVals[ev] = colData[ev][row]
			}
			logP += cpd.LogPDF(colData[node][row], parentVals)
		}
		result[row] = logP
	}

	return result, nil
}

// Predict returns the predicted (conditional mean) value of each node for each
// row, evaluated in topological order. Returns a map from variable name to
// a slice of predicted values.
func (lgbn *LinearGaussianBayesianNetwork) Predict(data *tabgo.DataFrame) (map[string][]float64, error) {
	if data == nil {
		return nil, fmt.Errorf("models: data must not be nil")
	}
	if err := lgbn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: invalid model: %w", err)
	}

	nRows := data.Len()
	order, err := lgbn.dag.TopologicalOrder()
	if err != nil {
		return nil, fmt.Errorf("models: cannot get topological order: %w", err)
	}

	// Use actual data for root nodes, predicted for children.
	colData := make(map[string][]float64)
	for _, node := range order {
		colData[node] = data.Column(node).Float64()
	}

	predicted := make(map[string][]float64, len(order))
	for _, node := range order {
		cpd := lgbn.lgCPDs[node]
		parents := cpd.Evidence()
		preds := make([]float64, nRows)

		if len(parents) == 0 {
			// Root node: predicted value is the mean.
			for i := 0; i < nRows; i++ {
				preds[i] = cpd.Mean()
			}
		} else {
			for i := 0; i < nRows; i++ {
				parentVals := make(map[string]float64)
				for _, ev := range parents {
					parentVals[ev] = colData[ev][i]
				}
				preds[i] = cpd.ConditionalMean(parentVals)
			}
		}
		predicted[node] = preds
	}

	return predicted, nil
}

// ToMarkovModel converts the LG BN to a set of factors by computing the
// joint Gaussian and returning a single factor representing the joint
// distribution. For continuous models this returns an error since discrete
// factors are not directly applicable.
func (lgbn *LinearGaussianBayesianNetwork) ToMarkovModel() error {
	return fmt.Errorf("models: ToMarkovModel is not applicable for continuous linear Gaussian networks; use ToJointGaussian instead")
}

// IsIMap checks whether the network structure is an I-map of the given
// independence assertions. A BN is an I-map if every independence implied
// by its structure (via d-separation) is also present in the given set of
// independencies. The method checks all pairs of non-adjacent nodes: for
// each such pair (A, B), it uses the DAG's d-separation test with the
// parents of A as the conditioning set, and verifies that the resulting
// independence is contained in the provided assertions.
func (lgbn *LinearGaussianBayesianNetwork) IsIMap(independencies []IndependenceAssertion) (bool, error) {
	if err := lgbn.CheckModel(); err != nil {
		return false, fmt.Errorf("models: IsIMap requires a valid model: %w", err)
	}

	nodes := lgbn.Nodes() // sorted
	if len(nodes) == 0 {
		return true, nil
	}

	// Build a DiGraph for d-separation testing.
	dg := graphgo.NewDiGraph()
	for _, n := range nodes {
		dg.AddNode(n)
	}
	for _, e := range lgbn.Edges() {
		dg.AddEdge(e[0], e[1])
	}

	// Build adjacency set for quick non-adjacency checks.
	adjSet := make(map[string]map[string]bool)
	for _, n := range nodes {
		adjSet[n] = make(map[string]bool)
	}
	for _, e := range lgbn.Edges() {
		adjSet[e[0]][e[1]] = true
		adjSet[e[1]][e[0]] = true
	}

	// Build a lookup set from the provided independencies.
	type indepKey struct {
		a, b string
		z    string // sorted conditioning set joined
	}
	givenSet := make(map[indepKey]bool)
	for _, ia := range independencies {
		for _, x := range ia.Event1 {
			for _, y := range ia.Event2 {
				a, b := x, y
				if a > b {
					a, b = b, a
				}
				given := make([]string, len(ia.Given))
				copy(given, ia.Given)
				sort.Strings(given)
				givenSet[indepKey{a, b, strings.Join(given, ",")}] = true
			}
		}
	}

	// Check the local Markov property: for each node, it must be
	// independent of all non-descendants given its parents.
	for _, node := range nodes {
		parents := lgbn.Parents(node)
		zSet := make(map[string]bool, len(parents))
		for _, p := range parents {
			zSet[p] = true
		}
		nodeSet := map[string]bool{node: true}

		// Check against every other non-adjacent node.
		for _, other := range nodes {
			if other == node || adjSet[node][other] {
				continue
			}

			otherSet := map[string]bool{other: true}
			if graphgo.DSeparation(dg, nodeSet, otherSet, zSet) {
				// This independence is implied by the structure.
				// Check if it's in the provided independencies.
				a, b := node, other
				if a > b {
					a, b = b, a
				}
				condSorted := make([]string, len(parents))
				copy(condSorted, parents)
				sort.Strings(condSorted)
				key := indepKey{a, b, strings.Join(condSorted, ",")}
				if !givenSet[key] {
					return false, nil
				}
			}
		}
	}

	return true, nil
}

// GetRandom generates a random LinearGaussianBayesianNetwork with the given
// number of nodes and edges.
func GetRandomLinearGaussianBayesianNetwork(nNodes, nEdges int) (*LinearGaussianBayesianNetwork, error) {
	if nNodes <= 0 {
		return nil, fmt.Errorf("models: nNodes must be positive, got %d", nNodes)
	}
	maxEdges := nNodes * (nNodes - 1) / 2
	if nEdges < 0 || nEdges > maxEdges {
		return nil, fmt.Errorf("models: nEdges %d out of range [0, %d]", nEdges, maxEdges)
	}

	bn := NewLinearGaussianBayesianNetwork()
	nodeNames := make([]string, nNodes)
	for i := 0; i < nNodes; i++ {
		nodeNames[i] = fmt.Sprintf("X%d", i)
		if err := bn.AddNode(nodeNames[i]); err != nil {
			return nil, err
		}
	}

	type edgePair struct{ from, to int }
	var possibleEdges []edgePair
	for i := 0; i < nNodes; i++ {
		for j := i + 1; j < nNodes; j++ {
			possibleEdges = append(possibleEdges, edgePair{i, j})
		}
	}
	rand.Shuffle(len(possibleEdges), func(i, j int) {
		possibleEdges[i], possibleEdges[j] = possibleEdges[j], possibleEdges[i]
	})
	for e := 0; e < nEdges; e++ {
		ep := possibleEdges[e]
		if err := bn.AddEdge(nodeNames[ep.from], nodeNames[ep.to]); err != nil {
			return nil, err
		}
	}

	if err := bn.GetRandomCPDs(); err != nil {
		return nil, err
	}

	return bn, nil
}
