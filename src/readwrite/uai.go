package readwrite

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// ReadUAI parses a UAI format file and returns a BayesianNetwork.
// UAI format:
//
//	BAYES
//	N (number of variables)
//	card1 card2 ... cardN (cardinalities)
//	M (number of factors)
//	numVars var1 var2 ... (factor scopes, one per line)
//	... (blank line)
//	numEntries (per factor table)
//	val1 val2 ...
func ReadUAI(r io.Reader) (*models.BayesianNetwork, error) {
	tokens, err := uaiTokenize(r)
	if err != nil {
		return nil, err
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("readwrite: empty UAI file")
	}

	pos := 0
	next := func() (string, error) {
		if pos >= len(tokens) {
			return "", fmt.Errorf("readwrite: unexpected end of UAI file")
		}
		t := tokens[pos]
		pos++
		return t, nil
	}
	nextInt := func() (int, error) {
		t, err := next()
		if err != nil {
			return 0, err
		}
		v, err := strconv.Atoi(t)
		if err != nil {
			return 0, fmt.Errorf("readwrite: expected integer, got %q: %w", t, err)
		}
		return v, nil
	}
	nextFloat := func() (float64, error) {
		t, err := next()
		if err != nil {
			return 0, err
		}
		v, err := strconv.ParseFloat(t, 64)
		if err != nil {
			return 0, fmt.Errorf("readwrite: expected float, got %q: %w", t, err)
		}
		return v, nil
	}

	// Type.
	typeStr, err := next()
	if err != nil {
		return nil, err
	}
	if typeStr != "BAYES" {
		return nil, fmt.Errorf("readwrite: UAI type %q not supported, expected BAYES", typeStr)
	}

	// Number of variables.
	numVars, err := nextInt()
	if err != nil {
		return nil, fmt.Errorf("readwrite: error reading variable count: %w", err)
	}

	// Cardinalities.
	cards := make([]int, numVars)
	for i := 0; i < numVars; i++ {
		cards[i], err = nextInt()
		if err != nil {
			return nil, fmt.Errorf("readwrite: error reading cardinality %d: %w", i, err)
		}
	}

	// Number of factors.
	numFactors, err := nextInt()
	if err != nil {
		return nil, fmt.Errorf("readwrite: error reading factor count: %w", err)
	}

	// Factor scopes.
	type scope struct {
		vars []int
	}
	scopes := make([]scope, numFactors)
	for f := 0; f < numFactors; f++ {
		numScopeVars, err := nextInt()
		if err != nil {
			return nil, fmt.Errorf("readwrite: error reading scope size for factor %d: %w", f, err)
		}
		scopes[f].vars = make([]int, numScopeVars)
		for j := 0; j < numScopeVars; j++ {
			scopes[f].vars[j], err = nextInt()
			if err != nil {
				return nil, fmt.Errorf("readwrite: error reading scope var for factor %d: %w", f, err)
			}
		}
	}

	// Build network.
	bn := models.NewBayesianNetwork()

	// Create variable names: V0, V1, ...
	varNames := make([]string, numVars)
	for i := 0; i < numVars; i++ {
		varNames[i] = fmt.Sprintf("V%d", i)
		if err := bn.AddNode(varNames[i]); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
		// Generate state names: s0, s1, ...
		states := make([]string, cards[i])
		for s := 0; s < cards[i]; s++ {
			states[s] = fmt.Sprintf("s%d", s)
		}
		if err := bn.SetStates(varNames[i], states); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
	}

	// Factor tables.
	for f := 0; f < numFactors; f++ {
		numEntries, err := nextInt()
		if err != nil {
			return nil, fmt.Errorf("readwrite: error reading entry count for factor %d: %w", f, err)
		}
		vals := make([]float64, numEntries)
		for j := 0; j < numEntries; j++ {
			vals[j], err = nextFloat()
			if err != nil {
				return nil, fmt.Errorf("readwrite: error reading value for factor %d: %w", f, err)
			}
		}

		sv := scopes[f].vars
		if len(sv) == 0 {
			continue
		}

		// Last variable in scope is the child (for BAYES type).
		childIdx := sv[len(sv)-1]
		childName := varNames[childIdx]
		childCard := cards[childIdx]

		var parents []string
		var evidenceCard []int
		for _, pi := range sv[:len(sv)-1] {
			parents = append(parents, varNames[pi])
			evidenceCard = append(evidenceCard, cards[pi])
			if err := bn.AddEdge(varNames[pi], childName); err != nil {
				if !strings.Contains(err.Error(), "already exists") {
					return nil, fmt.Errorf("readwrite: %w", err)
				}
			}
		}

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		if len(vals) != childCard*numParentConfigs {
			return nil, fmt.Errorf("readwrite: factor %d has %d values, expected %d",
				f, len(vals), childCard*numParentConfigs)
		}

		// UAI table ordering: last variable in scope changes fastest.
		// Since child is last in scope, ordering is: parent configs outer, child states inner.
		// values[childState][parentConfig]
		values := make([][]float64, childCard)
		for cs := 0; cs < childCard; cs++ {
			values[cs] = make([]float64, numParentConfigs)
		}

		idx := 0
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < childCard; cs++ {
				values[cs][pc] = vals[idx]
				idx++
			}
		}

		cpd, err := factors.NewTabularCPD(childName, childCard, values, parents, evidenceCard)
		if err != nil {
			return nil, fmt.Errorf("readwrite: failed to create CPD for %q: %w", childName, err)
		}
		if err := bn.AddCPD(cpd); err != nil {
			return nil, fmt.Errorf("readwrite: %w", err)
		}
	}

	return bn, nil
}

// WriteUAI serializes a BayesianNetwork to UAI format.
func WriteUAI(w io.Writer, bn *models.BayesianNetwork) error {
	nodes := bn.Nodes()
	numVars := len(nodes)

	// Build name-to-index map.
	nameToIdx := make(map[string]int, numVars)
	for i, n := range nodes {
		nameToIdx[n] = i
	}

	// Header.
	if _, err := fmt.Fprintf(w, "BAYES\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	// Number of variables.
	if _, err := fmt.Fprintf(w, "%d\n", numVars); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	// Cardinalities.
	cardParts := make([]string, numVars)
	for i, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("readwrite: variable %q has no state names", node)
		}
		cardParts[i] = strconv.Itoa(len(states))
	}
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(cardParts, " ")); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	// Number of factors (one per node).
	if _, err := fmt.Fprintf(w, "%d\n", numVars); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	// Factor scopes.
	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			return fmt.Errorf("readwrite: variable %q has no CPD", node)
		}
		evidence := cpd.Evidence()
		scopeSize := 1 + len(evidence)
		parts := []string{strconv.Itoa(scopeSize)}
		for _, ev := range evidence {
			parts = append(parts, strconv.Itoa(nameToIdx[ev]))
		}
		parts = append(parts, strconv.Itoa(nameToIdx[node]))
		if _, err := fmt.Fprintf(w, "%s\n", strings.Join(parts, " ")); err != nil {
			return fmt.Errorf("readwrite: write error: %w", err)
		}
	}

	// Factor tables.
	if _, err := fmt.Fprint(w, "\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		evidenceCard := cpd.EvidenceCard()
		childCard := cpd.VariableCard()

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		data := cpd.ToFactor().Values().Data()
		totalEntries := childCard * numParentConfigs

		if _, err := fmt.Fprintf(w, "%d\n", totalEntries); err != nil {
			return fmt.Errorf("readwrite: write error: %w", err)
		}

		// UAI: parent configs outer, child states inner.
		var parts []string
		for pc := 0; pc < numParentConfigs; pc++ {
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, formatFloat(data[cs*numParentConfigs+pc]))
			}
		}
		if _, err := fmt.Fprintf(w, "%s\n", strings.Join(parts, " ")); err != nil {
			return fmt.Errorf("readwrite: write error: %w", err)
		}
	}

	return nil
}

// uaiTokenize reads all whitespace-separated tokens from a UAI file.
func uaiTokenize(r io.Reader) ([]string, error) {
	scanner := bufio.NewScanner(r)
	var tokens []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		tokens = append(tokens, strings.Fields(line)...)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("readwrite: error reading UAI: %w", err)
	}
	return tokens, nil
}
