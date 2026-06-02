package models

import (
	"bufio"
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// RemoveNode removes a node from the network, along with its CPD and all
// connected edges (both incoming and outgoing).
func (bn *BayesianNetwork) RemoveNode(node string) error {
	if !bn.dag.HasNode(node) {
		return fmt.Errorf("models: node %q not found", node)
	}
	delete(bn.cpds, node)
	delete(bn.states, node)
	return bn.dag.RemoveNode(node)
}

// RemoveNodes removes multiple nodes from the network. If any node is not
// found, an error is returned and nodes already removed are not restored.
func (bn *BayesianNetwork) RemoveNodes(nodes ...string) error {
	for _, n := range nodes {
		if err := bn.RemoveNode(n); err != nil {
			return err
		}
	}
	return nil
}

// GetCardinality returns the cardinality of a node from its CPD.
// An error is returned if the node has no CPD.
func (bn *BayesianNetwork) GetCardinality(node string) (int, error) {
	cpd := bn.cpds[node]
	if cpd == nil {
		return 0, fmt.Errorf("models: node %q has no CPD", node)
	}
	return cpd.VariableCard(), nil
}

// ToJunctionTree converts this BayesianNetwork to a JunctionTree by
// delegating to NewJunctionTreeFromBN.
func (bn *BayesianNetwork) ToJunctionTree() (*JunctionTree, error) {
	return NewJunctionTreeFromBN(bn)
}

// ---------- local variable-elimination helpers (avoids circular import) ------

// veQuery performs variable elimination on the given factors, querying
// queryVars conditioned on evidence. Returns a normalized factor over queryVars.
func veQuery(factorList []*factors.DiscreteFactor, queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	working := make([]*factors.DiscreteFactor, len(factorList))
	for i, f := range factorList {
		working[i] = f.Copy()
	}

	var err error
	working, err = veReduceAll(working, evidence)
	if err != nil {
		return nil, err
	}

	allVars := make(map[string]bool)
	for _, f := range working {
		for _, v := range f.Variables() {
			allVars[v] = true
		}
	}

	keepSet := make(map[string]bool, len(queryVars)+len(evidence))
	for _, v := range queryVars {
		keepSet[v] = true
	}
	for v := range evidence {
		keepSet[v] = true
	}

	var elimVars []string
	for v := range allVars {
		if !keepSet[v] {
			elimVars = append(elimVars, v)
		}
	}
	sort.Strings(elimVars)

	for _, elimVar := range elimVars {
		working, err = veEliminateVariable(working, elimVar)
		if err != nil {
			return nil, err
		}
	}

	if len(working) == 0 {
		return nil, fmt.Errorf("models: no factors remain after elimination")
	}

	result, err := factors.FactorProduct(working...)
	if err != nil {
		return nil, err
	}
	result.Normalize()
	return result, nil
}

func veReduceAll(factorList []*factors.DiscreteFactor, evidence map[string]int) ([]*factors.DiscreteFactor, error) {
	result := make([]*factors.DiscreteFactor, 0, len(factorList))
	for _, f := range factorList {
		fVars := make(map[string]bool)
		for _, v := range f.Variables() {
			fVars[v] = true
		}
		applicable := make(map[string]int)
		for v, val := range evidence {
			if fVars[v] {
				applicable[v] = val
			}
		}
		reduced, err := f.Reduce(applicable)
		if err != nil {
			return nil, err
		}
		result = append(result, reduced)
	}
	return result, nil
}

func veEliminateVariable(factorList []*factors.DiscreteFactor, variable string) ([]*factors.DiscreteFactor, error) {
	var containing []*factors.DiscreteFactor
	var remaining []*factors.DiscreteFactor

	for _, f := range factorList {
		hasVar := false
		for _, v := range f.Variables() {
			if v == variable {
				hasVar = true
				break
			}
		}
		if hasVar {
			containing = append(containing, f)
		} else {
			remaining = append(remaining, f)
		}
	}

	if len(containing) == 0 {
		return factorList, nil
	}

	product, err := factors.FactorProduct(containing...)
	if err != nil {
		return nil, err
	}

	prodVars := product.Variables()
	if len(prodVars) == 1 && prodVars[0] == variable {
		return remaining, nil
	}

	marginalized, err := product.Marginalize([]string{variable})
	if err != nil {
		return nil, err
	}

	return append(remaining, marginalized), nil
}

func veMAP(factorList []*factors.DiscreteFactor, queryVars []string, evidence map[string]int) (map[string]int, error) {
	result, err := veQuery(factorList, queryVars, evidence)
	if err != nil {
		return nil, err
	}

	vars := result.Variables()
	card := result.Cardinality()
	totalSize := 1
	for _, c := range card {
		totalSize *= c
	}

	bestVal := -1.0
	bestAssignment := make(map[string]int, len(vars))

	for flat := 0; flat < totalSize; flat++ {
		assignment := make(map[string]int, len(vars))
		rem := flat
		for i := len(vars) - 1; i >= 0; i-- {
			assignment[vars[i]] = rem % card[i]
			rem /= card[i]
		}
		val := result.GetValue(assignment)
		if val > bestVal {
			bestVal = val
			for k, v := range assignment {
				bestAssignment[k] = v
			}
		}
	}

	return bestAssignment, nil
}

// -------------------------------------------------------------------------

// Predict fills in missing values (nil) in a DataFrame using variable
// elimination inference. For each row, variables with nil values are treated
// as query variables and non-nil variables are treated as evidence. The most
// likely (MAP) assignment is used to fill in missing values.
func (bn *BayesianNetwork) Predict(data *tabgo.DataFrame) (*tabgo.DataFrame, error) {
	if err := bn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: Predict requires a valid model: %w", err)
	}

	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		return nil, err
	}

	cols := data.Columns()
	nRows := data.Len()

	colVals := make(map[string][]any, len(cols))
	for _, c := range cols {
		colVals[c] = data.Column(c).Values()
	}

	resultCols := make(map[string][]any, len(cols))
	for _, c := range cols {
		resultCols[c] = make([]any, nRows)
	}

	for row := 0; row < nRows; row++ {
		evidence := make(map[string]int)
		var queryVars []string

		for _, c := range cols {
			val := colVals[c][row]
			if val == nil {
				queryVars = append(queryVars, c)
			} else {
				evidence[c] = toInt(val)
			}
		}

		if len(queryVars) == 0 {
			for _, c := range cols {
				resultCols[c][row] = colVals[c][row]
			}
			continue
		}

		mapAssignment, err := veMAP(markovFactors, queryVars, evidence)
		if err != nil {
			return nil, fmt.Errorf("models: Predict row %d: %w", row, err)
		}

		for _, c := range cols {
			if colVals[c][row] == nil {
				resultCols[c][row] = mapAssignment[c]
			} else {
				resultCols[c][row] = colVals[c][row]
			}
		}
	}

	seriesMap := make(map[string]*tabgo.Series, len(cols))
	for _, c := range cols {
		seriesMap[c] = tabgo.NewSeries(c, resultCols[c])
	}
	return tabgo.NewDataFrame(seriesMap), nil
}

// PredictProbability returns the probability distribution over missing
// variables for each row. The returned map contains variable name -> slice
// of probabilities (one per state, concatenated across rows).
func (bn *BayesianNetwork) PredictProbability(data *tabgo.DataFrame) (map[string][]float64, error) {
	if err := bn.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: PredictProbability requires a valid model: %w", err)
	}

	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		return nil, err
	}

	cols := data.Columns()
	nRows := data.Len()

	colVals := make(map[string][]any, len(cols))
	for _, c := range cols {
		colVals[c] = data.Column(c).Values()
	}

	result := make(map[string][]float64)

	for row := 0; row < nRows; row++ {
		evidence := make(map[string]int)
		var queryVars []string

		for _, c := range cols {
			val := colVals[c][row]
			if val == nil {
				queryVars = append(queryVars, c)
			} else {
				evidence[c] = toInt(val)
			}
		}

		if len(queryVars) == 0 {
			continue
		}

		resultFactor, err := veQuery(markovFactors, queryVars, evidence)
		if err != nil {
			return nil, fmt.Errorf("models: PredictProbability row %d: %w", row, err)
		}

		for _, qv := range queryVars {
			otherVars := make([]string, 0)
			for _, v := range resultFactor.Variables() {
				if v != qv {
					otherVars = append(otherVars, v)
				}
			}
			var marginal *factors.DiscreteFactor
			if len(otherVars) > 0 {
				marginal, err = resultFactor.Marginalize(otherVars)
				if err != nil {
					return nil, fmt.Errorf("models: PredictProbability marginalize %q: %w", qv, err)
				}
			} else {
				marginal = resultFactor.Copy()
			}
			probs := marginal.Values().Data()
			result[qv] = append(result[qv], probs...)
		}
	}

	return result, nil
}

// GetStateProbability computes P(states) -- the joint probability of a
// complete or partial state assignment -- using variable elimination.
func (bn *BayesianNetwork) GetStateProbability(states map[string]int) (float64, error) {
	if len(states) == 0 {
		return 0, fmt.Errorf("models: states must not be empty")
	}

	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		return 0, err
	}

	allNodes := bn.Nodes()

	specified := make(map[string]bool, len(states))
	for v := range states {
		specified[v] = true
	}

	allSpecified := true
	for _, n := range allNodes {
		if !specified[n] {
			allSpecified = false
			break
		}
	}

	if allSpecified {
		product, err := factors.FactorProduct(markovFactors...)
		if err != nil {
			return 0, fmt.Errorf("models: GetStateProbability: %w", err)
		}
		return product.GetValue(states), nil
	}

	specifiedVars := make([]string, 0, len(states))
	for _, n := range allNodes {
		if specified[n] {
			specifiedVars = append(specifiedVars, n)
		}
	}

	resultFactor, err := veQuery(markovFactors, specifiedVars, nil)
	if err != nil {
		return 0, fmt.Errorf("models: GetStateProbability: %w", err)
	}

	return resultFactor.GetValue(states), nil
}

// GetMarkovBlanket returns the Markov blanket of a node: its parents,
// children, and parents of its children (co-parents), excluding the node itself.
func (bn *BayesianNetwork) GetMarkovBlanket(node string) ([]string, error) {
	if !bn.dag.HasNode(node) {
		return nil, fmt.Errorf("models: node %q not found", node)
	}

	dg := graphgo.NewDiGraph()
	for _, n := range bn.Nodes() {
		dg.AddNode(n)
	}
	for _, e := range bn.Edges() {
		dg.AddEdge(e[0], e[1])
	}

	blanket := graphgo.MarkovBlanket(dg, node)

	result := make([]string, 0, len(blanket))
	for n := range blanket {
		result = append(result, n)
	}
	sort.Strings(result)
	return result, nil
}

// Do performs a causal intervention (do-calculus). For each node in the nodes
// map, the incoming edges are removed and the node's CPD is replaced with a
// delta distribution that assigns probability 1 to the specified state.
// Returns a new (mutilated) BayesianNetwork; the original is not modified.
func (bn *BayesianNetwork) Do(nodes map[string]int) (*BayesianNetwork, error) {
	if len(nodes) == 0 {
		return bn.Copy(), nil
	}

	mutilated := bn.Copy()

	for node, state := range nodes {
		if !mutilated.dag.HasNode(node) {
			return nil, fmt.Errorf("models: do-intervention on unknown node %q", node)
		}

		cpd := mutilated.cpds[node]
		if cpd == nil {
			return nil, fmt.Errorf("models: node %q has no CPD for do-intervention", node)
		}

		card := cpd.VariableCard()
		if state < 0 || state >= card {
			return nil, fmt.Errorf("models: do-intervention state %d out of range for %q (card %d)", state, node, card)
		}

		parents := mutilated.Parents(node)
		for _, p := range parents {
			_ = mutilated.dag.RemoveEdge(p, node)
		}

		vals := make([][]float64, card)
		for i := 0; i < card; i++ {
			if i == state {
				vals[i] = []float64{1.0}
			} else {
				vals[i] = []float64{0.0}
			}
		}
		deltaCPD, err := factors.NewTabularCPD(node, card, vals, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("models: Do failed to create delta CPD for %q: %w", node, err)
		}
		mutilated.cpds[node] = deltaCPD
	}

	return mutilated, nil
}

// IsIMap checks whether this BayesianNetwork is an I-map (independence map)
// of the given joint probability distribution. A BN is an I-map if every
// independence implied by the BN structure (via d-separation) also holds in
// the JPD.
func (bn *BayesianNetwork) IsIMap(jpd *factors.JointProbabilityDistribution) (bool, error) {
	if err := bn.CheckModel(); err != nil {
		return false, fmt.Errorf("models: IsIMap requires a valid model: %w", err)
	}

	dg := graphgo.NewDiGraph()
	for _, n := range bn.Nodes() {
		dg.AddNode(n)
	}
	for _, e := range bn.Edges() {
		dg.AddEdge(e[0], e[1])
	}

	nodes := bn.Nodes()
	adjSet := make(map[string]map[string]bool)
	for _, n := range nodes {
		adjSet[n] = make(map[string]bool)
	}
	for _, e := range bn.Edges() {
		adjSet[e[0]][e[1]] = true
		adjSet[e[1]][e[0]] = true
	}

	const atol = 1e-4

	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			a, b := nodes[i], nodes[j]
			if adjSet[a][b] {
				continue
			}

			parents := bn.Parents(a)
			xSet := map[string]bool{a: true}
			ySet := map[string]bool{b: true}
			zSet := make(map[string]bool, len(parents))
			for _, p := range parents {
				zSet[p] = true
			}

			if graphgo.DSeparation(dg, xSet, ySet, zSet) {
				if !jpd.CheckIndependence(a, b, parents, atol) {
					return false, nil
				}
			}
		}
	}

	return true, nil
}

// GetFactorizedProduct returns all CPDs converted to discrete factors.
// CheckModel is called first to ensure the network is valid.
func (bn *BayesianNetwork) GetFactorizedProduct() ([]*factors.DiscreteFactor, error) {
	return bn.ToMarkovFactors()
}

// Save writes the BayesianNetwork to a BIF file.
func (bn *BayesianNetwork) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("models: Save: %w", err)
	}
	defer f.Close()
	return bn.writeBIF(f)
}

// writeBIF serializes the BayesianNetwork in BIF format.
func (bn *BayesianNetwork) writeBIF(w io.Writer) error {
	if _, err := fmt.Fprintf(w, "network unknown {\n}\n"); err != nil {
		return err
	}
	nodes := bn.Nodes()
	for _, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("models: variable %q has no state names", node)
		}
		if _, err := fmt.Fprintf(w, "\nvariable %s {\n  type discrete [ %d ] { %s };\n}\n",
			node, len(states), strings.Join(states, ", ")); err != nil {
			return err
		}
	}
	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			return fmt.Errorf("models: variable %q has no CPD", node)
		}
		evidence := cpd.Evidence()
		if len(evidence) == 0 {
			if _, err := fmt.Fprintf(w, "\nprobability ( %s ) {\n", node); err != nil {
				return err
			}
		} else {
			if _, err := fmt.Fprintf(w, "\nprobability ( %s | %s ) {\n",
				node, strings.Join(evidence, ", ")); err != nil {
				return err
			}
		}
		data := cpd.ToFactor().Values().Data()
		childCard := cpd.VariableCard()
		evidenceCard := cpd.EvidenceCard()
		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}
		if len(evidence) == 0 {
			var parts []string
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, fmt.Sprintf("%.10g", data[cs]))
			}
			if _, err := fmt.Fprintf(w, "  table %s;\n", strings.Join(parts, ", ")); err != nil {
				return err
			}
		} else {
			for pc := 0; pc < numParentConfigs; pc++ {
				pNames := bifDecomposePC(pc, evidence, evidenceCard, bn)
				var valParts []string
				for cs := 0; cs < childCard; cs++ {
					valParts = append(valParts, fmt.Sprintf("%.10g", data[cs*numParentConfigs+pc]))
				}
				if _, err := fmt.Fprintf(w, "  (%s) %s;\n",
					strings.Join(pNames, ", "), strings.Join(valParts, ", ")); err != nil {
					return err
				}
			}
		}
		if _, err := fmt.Fprintf(w, "}\n"); err != nil {
			return err
		}
	}
	return nil
}

func bifDecomposePC(pc int, evidence []string, evidenceCard []int, bn *BayesianNetwork) []string {
	indices := make([]int, len(evidence))
	rem := pc
	for i := len(evidence) - 1; i >= 0; i-- {
		indices[i] = rem % evidenceCard[i]
		rem /= evidenceCard[i]
	}
	names := make([]string, len(evidence))
	for i, ev := range evidence {
		states := bn.GetStates(ev)
		if indices[i] < len(states) {
			names[i] = states[indices[i]]
		} else {
			names[i] = fmt.Sprintf("state%d", indices[i])
		}
	}
	return names
}

// LoadBayesianNetwork reads a BIF file and returns a new BayesianNetwork.
func LoadBayesianNetwork(filename string) (*BayesianNetwork, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("models: Load: %w", err)
	}
	defer f.Close()
	return loadBIF(f)
}

// bifVarMeta holds parsed variable metadata for BIF loading.
type bifVarMeta struct {
	name   string
	card   int
	states []string
}

func loadBIF(r io.Reader) (*BayesianNetwork, error) {
	scanner := bufio.NewScanner(r)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		if idx := strings.Index(line, "//"); idx >= 0 {
			line = line[:idx]
		}
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("models: Load: %w", err)
	}

	bn := NewBayesianNetwork()
	varMap := make(map[string]*bifVarMeta)

	i := 0
	for i < len(lines) {
		tokens := strings.Fields(lines[i])
		if len(tokens) == 0 {
			i++
			continue
		}
		switch tokens[0] {
		case "network":
			i = bifSkipBraces(lines, i)
		case "variable":
			if len(tokens) < 2 {
				return nil, fmt.Errorf("models: Load: malformed variable declaration")
			}
			name := strings.TrimRight(tokens[1], "{")
			blockContent, end := bifCollectBlock(lines, i)
			i = end
			states, err := bifParseVarBlock(name, blockContent)
			if err != nil {
				return nil, err
			}
			if err := bn.AddNode(name); err != nil {
				return nil, err
			}
			_ = bn.SetStates(name, states)
			varMap[name] = &bifVarMeta{name: name, card: len(states), states: states}
		case "probability":
			headerLine := lines[i]
			blockContent, end := bifCollectBlock(lines, i)
			i = end
			child, parents, err := bifParseProbHeader(headerLine)
			if err != nil {
				return nil, err
			}
			childInfo := varMap[child]
			if childInfo == nil {
				return nil, fmt.Errorf("models: Load: unknown variable %q", child)
			}
			for _, p := range parents {
				if err := bn.AddEdge(p, child); err != nil {
					if !strings.Contains(err.Error(), "already exists") {
						return nil, err
					}
				}
			}
			var parentInfos []*bifVarMeta
			for _, p := range parents {
				pi := varMap[p]
				if pi == nil {
					return nil, fmt.Errorf("models: Load: unknown parent %q", p)
				}
				parentInfos = append(parentInfos, pi)
			}
			cpd, err := bifParseProbBlock(childInfo, parents, parentInfos, blockContent)
			if err != nil {
				return nil, err
			}
			if err := bn.AddCPD(cpd); err != nil {
				return nil, err
			}
		default:
			i++
		}
	}
	return bn, nil
}

func bifSkipBraces(lines []string, i int) int {
	depth := 0
	for i < len(lines) {
		depth += strings.Count(lines[i], "{") - strings.Count(lines[i], "}")
		i++
		if depth <= 0 {
			break
		}
	}
	return i
}

func bifCollectBlock(lines []string, start int) ([]string, int) {
	depth := 0
	var raw []string
	i := start
	for i < len(lines) {
		depth += strings.Count(lines[i], "{") - strings.Count(lines[i], "}")
		raw = append(raw, lines[i])
		i++
		if depth <= 0 {
			break
		}
	}
	joined := strings.Join(raw, "\n")
	openIdx := strings.Index(joined, "{")
	closeIdx := strings.LastIndex(joined, "}")
	if openIdx < 0 || closeIdx <= openIdx {
		return nil, i
	}
	inner := joined[openIdx+1 : closeIdx]
	var content []string
	for _, line := range strings.Split(inner, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			content = append(content, line)
		}
	}
	return content, i
}

func bifParseVarBlock(name string, blockLines []string) ([]string, error) {
	for _, line := range blockLines {
		if strings.HasPrefix(strings.TrimSpace(line), "type") {
			line = strings.TrimRight(strings.TrimSpace(line), ";")
			ob := strings.Index(line, "{")
			cb := strings.LastIndex(line, "}")
			if ob < 0 || cb <= ob {
				return nil, fmt.Errorf("models: Load: malformed type for %q", name)
			}
			var states []string
			for _, p := range strings.Split(line[ob+1:cb], ",") {
				s := strings.TrimSpace(p)
				if s != "" {
					states = append(states, s)
				}
			}
			if len(states) == 0 {
				return nil, fmt.Errorf("models: Load: no states for %q", name)
			}
			return states, nil
		}
	}
	return nil, fmt.Errorf("models: Load: no type for variable %q", name)
}

func bifParseProbHeader(line string) (string, []string, error) {
	op := strings.Index(line, "(")
	cp := strings.LastIndex(line, ")")
	if op < 0 || cp <= op {
		return "", nil, fmt.Errorf("models: Load: malformed probability header")
	}
	inner := strings.TrimSpace(line[op+1 : cp])
	parts := strings.SplitN(inner, "|", 2)
	child := strings.TrimSpace(parts[0])
	var parents []string
	if len(parts) == 2 {
		for _, p := range strings.Split(parts[1], ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				parents = append(parents, p)
			}
		}
	}
	return child, parents, nil
}

func bifParseProbBlock(child *bifVarMeta, parents []string, parentInfos []*bifVarMeta, blockLines []string) (*factors.TabularCPD, error) {
	numParentConfigs := 1
	var evidenceCard []int
	for _, pi := range parentInfos {
		numParentConfigs *= pi.card
		evidenceCard = append(evidenceCard, pi.card)
	}

	values := make([][]float64, child.card)
	for i := range values {
		values[i] = make([]float64, numParentConfigs)
	}

	if len(parents) == 0 {
		for _, line := range blockLines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "table") {
				line = strings.TrimPrefix(line, "table")
				vals, err := bifParseFloats(line)
				if err != nil {
					return nil, err
				}
				if len(vals) != child.card {
					return nil, fmt.Errorf("models: Load: table for %q has %d values, expected %d", child.name, len(vals), child.card)
				}
				for i, v := range vals {
					values[i][0] = v
				}
				break
			}
		}
	} else {
		for _, line := range blockLines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "table") {
				line = strings.TrimPrefix(line, "table")
				vals, err := bifParseFloats(line)
				if err != nil {
					return nil, err
				}
				if len(vals) != child.card*numParentConfigs {
					return nil, fmt.Errorf("models: Load: table for %q has %d values, expected %d", child.name, len(vals), child.card*numParentConfigs)
				}
				idx := 0
				for pc := 0; pc < numParentConfigs; pc++ {
					for cs := 0; cs < child.card; cs++ {
						values[cs][pc] = vals[idx]
						idx++
					}
				}
				break
			}
			if !strings.HasPrefix(line, "(") {
				continue
			}
			closeParen := strings.Index(line, ")")
			if closeParen < 0 {
				return nil, fmt.Errorf("models: Load: malformed conditional line: %s", line)
			}
			stateStr := line[1:closeParen]
			valStr := line[closeParen+1:]

			var parentStates []string
			for _, s := range strings.Split(stateStr, ",") {
				s = strings.TrimSpace(s)
				if s != "" {
					parentStates = append(parentStates, s)
				}
			}
			if len(parentStates) != len(parents) {
				return nil, fmt.Errorf("models: Load: conditional has %d parent states, expected %d", len(parentStates), len(parents))
			}

			pc, err := bifParentConfigIdx(parentStates, parentInfos)
			if err != nil {
				return nil, err
			}
			vals, err := bifParseFloats(valStr)
			if err != nil {
				return nil, err
			}
			if len(vals) != child.card {
				return nil, fmt.Errorf("models: Load: conditional has %d values, expected %d", len(vals), child.card)
			}
			for cs := 0; cs < child.card; cs++ {
				values[cs][pc] = vals[cs]
			}
		}
	}

	return factors.NewTabularCPD(child.name, child.card, values, parents, evidenceCard)
}

func bifParentConfigIdx(parentStates []string, parentInfos []*bifVarMeta) (int, error) {
	idx := 0
	stride := 1
	for i := len(parentInfos) - 1; i >= 0; i-- {
		stateIdx := -1
		for j, s := range parentInfos[i].states {
			if s == parentStates[i] {
				stateIdx = j
				break
			}
		}
		if stateIdx < 0 {
			return 0, fmt.Errorf("models: Load: unknown state %q for parent %q", parentStates[i], parentInfos[i].name)
		}
		idx += stateIdx * stride
		stride *= parentInfos[i].card
	}
	return idx, nil
}

func bifParseFloats(s string) ([]float64, error) {
	s = strings.TrimRight(strings.TrimSpace(s), ";")
	var vals []float64
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.ParseFloat(p, 64)
		if err != nil {
			return nil, err
		}
		vals = append(vals, v)
	}
	return vals, nil
}

// FitUpdate performs an online parameter update using weighted counts.
func (bn *BayesianNetwork) FitUpdate(data *tabgo.DataFrame, nPrevSamples int) error {
	if nPrevSamples < 0 {
		return fmt.Errorf("models: nPrevSamples must be non-negative, got %d", nPrevSamples)
	}

	nodes := bn.Nodes()
	nRows := data.Len()
	if nRows == 0 {
		return nil
	}

	colVals := make(map[string][]any, len(nodes))
	for _, node := range nodes {
		colVals[node] = data.Column(node).Values()
	}

	for _, node := range nodes {
		cpd := bn.cpds[node]
		if cpd == nil {
			return fmt.Errorf("models: node %q has no CPD for FitUpdate", node)
		}

		parents := bn.Parents(node)
		childCard := cpd.VariableCard()
		evidenceCard := cpd.EvidenceCard()

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		counts := make([][]float64, childCard)
		for i := range counts {
			counts[i] = make([]float64, numParentConfigs)
		}
		parentConfigCounts := make([]float64, numParentConfigs)

		for row := 0; row < nRows; row++ {
			childVal := toInt(colVals[node][row])
			if childVal < 0 || childVal >= childCard {
				continue
			}

			pc := 0
			stride := 1
			valid := true
			for pi := len(parents) - 1; pi >= 0; pi-- {
				pVal := toInt(colVals[parents[pi]][row])
				if pVal < 0 || pVal >= evidenceCard[pi] {
					valid = false
					break
				}
				pc += pVal * stride
				stride *= evidenceCard[pi]
			}
			if !valid {
				continue
			}

			counts[childVal][pc]++
			parentConfigCounts[pc]++
		}

		oldFactor := cpd.ToFactor()
		oldData := oldFactor.Values().Data()

		newValues := make([][]float64, childCard)
		for cs := 0; cs < childCard; cs++ {
			newValues[cs] = make([]float64, numParentConfigs)
			for pc := 0; pc < numParentConfigs; pc++ {
				oldProb := oldData[cs*numParentConfigs+pc]
				newCount := counts[cs][pc]
				total := float64(nPrevSamples) + parentConfigCounts[pc]
				if total > 0 {
					newValues[cs][pc] = (float64(nPrevSamples)*oldProb + newCount) / total
				} else {
					newValues[cs][pc] = oldProb
				}
			}
		}

		newCPD, err := factors.NewTabularCPD(node, childCard, newValues, parents, evidenceCard)
		if err != nil {
			return fmt.Errorf("models: FitUpdate CPD for %q: %w", node, err)
		}
		bn.cpds[node] = newCPD
	}

	return nil
}

// GetRandomBayesianNetwork generates a random BayesianNetwork with the
// specified number of nodes, edges, and states per node.
func GetRandomBayesianNetwork(nNodes, nEdges, nStates int) (*BayesianNetwork, error) {
	if nNodes <= 0 {
		return nil, fmt.Errorf("models: nNodes must be positive, got %d", nNodes)
	}
	if nStates <= 0 {
		return nil, fmt.Errorf("models: nStates must be positive, got %d", nStates)
	}
	maxEdges := nNodes * (nNodes - 1) / 2
	if nEdges < 0 || nEdges > maxEdges {
		return nil, fmt.Errorf("models: nEdges %d out of range [0, %d]", nEdges, maxEdges)
	}

	bn := NewBayesianNetwork()
	nodeNames := make([]string, nNodes)
	for i := 0; i < nNodes; i++ {
		nodeNames[i] = fmt.Sprintf("X%d", i)
		if err := bn.AddNode(nodeNames[i]); err != nil {
			return nil, err
		}
		stateNames := make([]string, nStates)
		for s := 0; s < nStates; s++ {
			stateNames[s] = fmt.Sprintf("s%d", s)
		}
		_ = bn.SetStates(nodeNames[i], stateNames)
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

	for _, node := range nodeNames {
		parents := bn.Parents(node)
		var evidenceCard []int
		numParentConfigs := 1
		for range parents {
			numParentConfigs *= nStates
			evidenceCard = append(evidenceCard, nStates)
		}

		values := make([][]float64, nStates)
		for cs := 0; cs < nStates; cs++ {
			values[cs] = make([]float64, numParentConfigs)
		}
		for pc := 0; pc < numParentConfigs; pc++ {
			sum := 0.0
			for cs := 0; cs < nStates; cs++ {
				v := rand.Float64() + 0.001
				values[cs][pc] = v
				sum += v
			}
			for cs := 0; cs < nStates; cs++ {
				values[cs][pc] /= sum
			}
		}

		cpd, err := factors.NewTabularCPD(node, nStates, values, parents, evidenceCard)
		if err != nil {
			return nil, fmt.Errorf("models: GetRandomBayesianNetwork CPD for %q: %w", node, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			return nil, err
		}
	}

	return bn, nil
}

// toInt converts an any value to int.
func toInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int8:
		return int(n)
	case int16:
		return int(n)
	case int32:
		return int(n)
	case int64:
		return int(n)
	case uint:
		return int(n)
	case uint8:
		return int(n)
	case uint16:
		return int(n)
	case uint32:
		return int(n)
	case uint64:
		return int(n)
	case float64:
		return int(n)
	case float32:
		return int(n)
	default:
		return 0
	}
}
