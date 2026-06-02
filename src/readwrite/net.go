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

// ReadNET parses a Hugin NET format file and returns a BayesianNetwork.
// NET format uses: node X { states = ("s0" "s1"); } and
// potential (X | Y) { data = ((0.3 0.7)(0.8 0.2)); }
func ReadNET(r io.Reader) (*models.BayesianNetwork, error) {
	lines, err := netReadLines(r)
	if err != nil {
		return nil, err
	}

	bn := models.NewBayesianNetwork()

	type varInfo struct {
		card   int
		states []string
	}
	varMap := make(map[string]*varInfo)

	i := 0
	for i < len(lines) {
		tokens := strings.Fields(lines[i])
		if len(tokens) == 0 {
			i++
			continue
		}

		switch tokens[0] {
		case "net":
			i = netSkipBlock(lines, i)

		case "node":
			if len(tokens) < 2 {
				return nil, fmt.Errorf("readwrite: malformed node declaration")
			}
			name := strings.TrimRight(tokens[1], "{")
			name = strings.TrimSpace(name)
			if name == "" {
				return nil, fmt.Errorf("readwrite: malformed node declaration: missing name")
			}

			blockContent, end := netCollectBlock(lines, i)
			i = end

			states, err := netParseNodeBlock(name, blockContent)
			if err != nil {
				return nil, err
			}

			if err := bn.AddNode(name); err != nil {
				return nil, fmt.Errorf("readwrite: %w", err)
			}
			if err := bn.SetStates(name, states); err != nil {
				return nil, fmt.Errorf("readwrite: %w", err)
			}
			varMap[name] = &varInfo{card: len(states), states: states}

		case "potential":
			headerLine := lines[i]
			blockContent, end := netCollectBlock(lines, i)
			i = end

			child, parents, err := netParsePotentialHeader(headerLine)
			if err != nil {
				return nil, err
			}

			childInfo := varMap[child]
			if childInfo == nil {
				return nil, fmt.Errorf("readwrite: potential references unknown variable %q", child)
			}

			for _, p := range parents {
				if err := bn.AddEdge(p, child); err != nil {
					if !strings.Contains(err.Error(), "already exists") {
						return nil, fmt.Errorf("readwrite: %w", err)
					}
				}
			}

			var parentInfos []struct {
				card   int
				states []string
			}
			var evidenceCard []int
			for _, p := range parents {
				pi := varMap[p]
				if pi == nil {
					return nil, fmt.Errorf("readwrite: potential references unknown parent %q", p)
				}
				parentInfos = append(parentInfos, struct {
					card   int
					states []string
				}{pi.card, pi.states})
				evidenceCard = append(evidenceCard, pi.card)
			}

			numParentConfigs := 1
			for _, ec := range evidenceCard {
				numParentConfigs *= ec
			}

			// Parse data from block.
			vals, err := netParseDataBlock(blockContent)
			if err != nil {
				return nil, fmt.Errorf("readwrite: error parsing potential data for %q: %w", child, err)
			}

			expectedLen := childInfo.card * numParentConfigs
			if len(vals) != expectedLen {
				return nil, fmt.Errorf("readwrite: potential for %q has %d values, expected %d",
					child, len(vals), expectedLen)
			}

			// NET data ordering: parent configs outer, child states inner.
			// values[childState][parentConfig]
			values := make([][]float64, childInfo.card)
			for cs := 0; cs < childInfo.card; cs++ {
				values[cs] = make([]float64, numParentConfigs)
			}

			idx := 0
			for pc := 0; pc < numParentConfigs; pc++ {
				for cs := 0; cs < childInfo.card; cs++ {
					values[cs][pc] = vals[idx]
					idx++
				}
			}

			cpd, err := factors.NewTabularCPD(child, childInfo.card, values, parents, evidenceCard)
			if err != nil {
				return nil, fmt.Errorf("readwrite: failed to create CPD for %q: %w", child, err)
			}
			if err := bn.AddCPD(cpd); err != nil {
				return nil, fmt.Errorf("readwrite: %w", err)
			}

		default:
			i++
		}
	}

	return bn, nil
}

// WriteNET serializes a BayesianNetwork to Hugin NET format.
func WriteNET(w io.Writer, bn *models.BayesianNetwork) error {
	// Net header.
	if _, err := fmt.Fprintf(w, "net\n{\n}\n"); err != nil {
		return fmt.Errorf("readwrite: write error: %w", err)
	}

	nodes := bn.Nodes()

	// Write node blocks.
	for _, node := range nodes {
		states := bn.GetStates(node)
		if len(states) == 0 {
			return fmt.Errorf("readwrite: variable %q has no state names", node)
		}

		quotedStates := make([]string, len(states))
		for i, s := range states {
			quotedStates[i] = fmt.Sprintf("%q", s)
		}

		if _, err := fmt.Fprintf(w, "\nnode %s\n{\n  states = (%s);\n}\n",
			node, strings.Join(quotedStates, " ")); err != nil {
			return fmt.Errorf("readwrite: write error: %w", err)
		}
	}

	// Write potential blocks.
	for _, node := range nodes {
		cpd := bn.GetCPD(node)
		if cpd == nil {
			return fmt.Errorf("readwrite: variable %q has no CPD", node)
		}

		evidence := cpd.Evidence()
		evidenceCard := cpd.EvidenceCard()
		childCard := cpd.VariableCard()

		numParentConfigs := 1
		for _, ec := range evidenceCard {
			numParentConfigs *= ec
		}

		data := cpd.ToFactor().Values().Data()

		// Header.
		if len(evidence) == 0 {
			if _, err := fmt.Fprintf(w, "\npotential (%s)\n{\n", node); err != nil {
				return fmt.Errorf("readwrite: write error: %w", err)
			}
		} else {
			if _, err := fmt.Fprintf(w, "\npotential (%s | %s)\n{\n",
				node, strings.Join(evidence, " ")); err != nil {
				return fmt.Errorf("readwrite: write error: %w", err)
			}
		}

		// Data: nested parentheses.
		// For each parent config, write (val1 val2 ... valN) for child states.
		if _, err := fmt.Fprint(w, "  data = "); err != nil {
			return fmt.Errorf("readwrite: write error: %w", err)
		}

		if numParentConfigs == 1 {
			// No parents: data = (val1 val2);
			var parts []string
			for cs := 0; cs < childCard; cs++ {
				parts = append(parts, formatFloat(data[cs]))
			}
			if _, err := fmt.Fprintf(w, "(%s);\n", strings.Join(parts, " ")); err != nil {
				return fmt.Errorf("readwrite: write error: %w", err)
			}
		} else {
			// With parents: data = ((val1 val2)(val3 val4));
			// Outer parens group all parent configs.
			if _, err := fmt.Fprint(w, "("); err != nil {
				return fmt.Errorf("readwrite: write error: %w", err)
			}
			for pc := 0; pc < numParentConfigs; pc++ {
				var parts []string
				for cs := 0; cs < childCard; cs++ {
					parts = append(parts, formatFloat(data[cs*numParentConfigs+pc]))
				}
				if _, err := fmt.Fprintf(w, "(%s)", strings.Join(parts, " ")); err != nil {
					return fmt.Errorf("readwrite: write error: %w", err)
				}
			}
			if _, err := fmt.Fprint(w, ");\n"); err != nil {
				return fmt.Errorf("readwrite: write error: %w", err)
			}
		}

		if _, err := fmt.Fprint(w, "}\n"); err != nil {
			return fmt.Errorf("readwrite: write error: %w", err)
		}
	}

	return nil
}

// netReadLines reads all lines, strips comments, returns non-empty trimmed lines.
func netReadLines(r io.Reader) ([]string, error) {
	scanner := bufio.NewScanner(r)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		// Strip % comments (Hugin style).
		if idx := strings.Index(line, "%"); idx >= 0 {
			line = line[:idx]
		}
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("readwrite: error reading NET: %w", err)
	}
	return lines, nil
}

// netSkipBlock advances past a { ... } block starting at line i.
func netSkipBlock(lines []string, i int) int {
	depth := 0
	opened := false
	for i < len(lines) {
		depth += strings.Count(lines[i], "{") - strings.Count(lines[i], "}")
		if depth > 0 {
			opened = true
		}
		i++
		if opened && depth <= 0 {
			break
		}
	}
	return i
}

// netCollectBlock returns content lines inside { } block starting at i.
func netCollectBlock(lines []string, start int) ([]string, int) {
	depth := 0
	opened := false
	var raw []string
	i := start
	for i < len(lines) {
		depth += strings.Count(lines[i], "{") - strings.Count(lines[i], "}")
		if depth > 0 {
			opened = true
		}
		raw = append(raw, lines[i])
		i++
		if opened && depth <= 0 {
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

// netParseNodeBlock parses lines inside a node { } block to extract states.
func netParseNodeBlock(name string, blockLines []string) ([]string, error) {
	for _, line := range blockLines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "states") {
			return netParseStates(name, line)
		}
	}
	return nil, fmt.Errorf("readwrite: no states declaration for node %q", name)
}

// netParseStates parses: states = ("s0" "s1" "s2");
func netParseStates(varName, line string) ([]string, error) {
	line = strings.TrimRight(strings.TrimSpace(line), ";")

	openParen := strings.Index(line, "(")
	closeParen := strings.LastIndex(line, ")")
	if openParen < 0 || closeParen <= openParen {
		return nil, fmt.Errorf("readwrite: malformed states declaration for %q", varName)
	}

	inner := line[openParen+1 : closeParen]
	var states []string

	// Parse quoted strings.
	inQuote := false
	var current strings.Builder
	for _, ch := range inner {
		if ch == '"' {
			if inQuote {
				states = append(states, current.String())
				current.Reset()
			}
			inQuote = !inQuote
			continue
		}
		if inQuote {
			current.WriteRune(ch)
		}
	}

	if len(states) == 0 {
		return nil, fmt.Errorf("readwrite: no states found for node %q", varName)
	}
	return states, nil
}

// netParsePotentialHeader parses: potential (X | Y Z) { or potential (X) {
func netParsePotentialHeader(line string) (string, []string, error) {
	openParen := strings.Index(line, "(")
	closeParen := strings.LastIndex(line, ")")
	if openParen < 0 || closeParen <= openParen {
		return "", nil, fmt.Errorf("readwrite: malformed potential header: %s", line)
	}

	inner := strings.TrimSpace(line[openParen+1 : closeParen])
	parts := strings.SplitN(inner, "|", 2)
	child := strings.TrimSpace(parts[0])
	if child == "" {
		return "", nil, fmt.Errorf("readwrite: empty variable in potential header")
	}

	var parents []string
	if len(parts) == 2 {
		for _, p := range strings.Fields(strings.TrimSpace(parts[1])) {
			p = strings.TrimSpace(p)
			if p != "" {
				parents = append(parents, p)
			}
		}
	}
	return child, parents, nil
}

// netParseDataBlock extracts float values from the data = (...) block.
func netParseDataBlock(blockLines []string) ([]float64, error) {
	// Join all block lines and find the data = ... part.
	joined := strings.Join(blockLines, " ")

	// Find "data" keyword.
	dataIdx := strings.Index(joined, "data")
	if dataIdx < 0 {
		return nil, fmt.Errorf("no data declaration found")
	}
	rest := joined[dataIdx+4:]

	// Find = sign.
	eqIdx := strings.Index(rest, "=")
	if eqIdx < 0 {
		return nil, fmt.Errorf("no = sign in data declaration")
	}
	rest = rest[eqIdx+1:]

	// Strip semicolons and parentheses, parse all numbers.
	rest = strings.TrimRight(strings.TrimSpace(rest), ";")
	rest = strings.ReplaceAll(rest, "(", " ")
	rest = strings.ReplaceAll(rest, ")", " ")

	var vals []float64
	for _, f := range strings.Fields(rest) {
		v, err := strconv.ParseFloat(f, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid probability value %q: %w", f, err)
		}
		vals = append(vals, v)
	}

	return vals, nil
}
