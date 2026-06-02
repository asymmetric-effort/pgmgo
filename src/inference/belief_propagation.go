package inference

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// edgeKey returns a canonical key for the edge between two cliques.
func edgeKey(i, j int) string {
	if i > j {
		i, j = j, i
	}
	return fmt.Sprintf("%d-%d", i, j)
}

// parseEdgeKey extracts the two clique indices from a separator key string.
// It supports both the "%d-%d" format (used internally by edgeKey) and the
// NUL-separated format produced by graphgo.BuildJunctionTree ("0\x001").
// Returns (-1, -1) if parsing fails.
func parseEdgeKey(k string) (int, int) {
	// Try hyphen-separated format first.
	var a, b int
	if n, _ := fmt.Sscanf(k, "%d-%d", &a, &b); n == 2 {
		return a, b
	}
	// Try NUL-byte separated format.
	parts := strings.Split(k, "\x00")
	if len(parts) == 2 {
		a, errA := strconv.Atoi(parts[0])
		b, errB := strconv.Atoi(parts[1])
		if errA == nil && errB == nil {
			return a, b
		}
	}
	return -1, -1
}

// normalizeEdgeKey converts a separator key from any supported format to the
// canonical edgeKey format ("%d-%d" with smaller index first).
func normalizeEdgeKey(k string) string {
	a, b := parseEdgeKey(k)
	if a < 0 {
		return k // fallback: return as-is
	}
	return edgeKey(a, b)
}

// BeliefPropagation performs exact inference on a junction tree using the
// Hugin-style message-passing (collect/distribute) algorithm. It operates
// on cliques and factors directly, avoiding circular imports with models.
type BeliefPropagation struct {
	// cliques[i] is the set of variable names in clique i.
	cliques [][]string

	// neighbors[i] lists the indices of cliques adjacent to clique i
	// in the junction tree.
	neighbors [][]int

	// separators maps an edge key (edgeKey(i,j)) to the separator
	// variables between cliques i and j.
	separators map[string][]string

	// initialFactors maps clique index to the original factors assigned
	// to that clique.
	initialFactors map[int][]*factors.DiscreteFactor

	// potentials[i] is the current belief (potential) for clique i.
	potentials []*factors.DiscreteFactor

	// messages stores the last message sent along each directed edge.
	// Key is "src->dst".
	messages map[string]*factors.DiscreteFactor

	// calibrated is true after Calibrate has completed successfully.
	calibrated bool

	// cardMap caches variable cardinalities extracted from factors.
	cardMap map[string]int
}

// NewBeliefPropagation creates a new BeliefPropagation engine.
//
// Parameters:
//   - cliques: each element is the set of variable names in one clique.
//   - separators: maps edgeKey(i,j) to the separator variables. The
//     separator structure implicitly defines the junction tree edges.
//   - cliqueFactors: maps clique index to factors assigned to that clique.
func NewBeliefPropagation(
	cliques [][]string,
	separators map[string][]string,
	cliqueFactors map[int][]*factors.DiscreteFactor,
) *BeliefPropagation {
	// Deep-copy cliques.
	cliqueCopy := make([][]string, len(cliques))
	for i, c := range cliques {
		cliqueCopy[i] = make([]string, len(c))
		copy(cliqueCopy[i], c)
	}

	// Deep-copy separators, normalizing keys to edgeKey format ("%d-%d").
	// Incoming keys may use different delimiters (e.g. NUL byte from graphgo).
	sepCopy := make(map[string][]string, len(separators))
	for k, v := range separators {
		s := make([]string, len(v))
		copy(s, v)
		normalizedKey := normalizeEdgeKey(k)
		sepCopy[normalizedKey] = s
	}

	// Deep-copy factors.
	facCopy := make(map[int][]*factors.DiscreteFactor, len(cliqueFactors))
	for idx, fl := range cliqueFactors {
		copied := make([]*factors.DiscreteFactor, len(fl))
		for i, f := range fl {
			copied[i] = f.Copy()
		}
		facCopy[idx] = copied
	}

	// Build neighbor lists from separator keys.
	neighbors := make([][]int, len(cliques))
	for k := range separators {
		a, b := parseEdgeKey(k)
		if a >= 0 && b >= 0 && a < len(cliques) && b < len(cliques) {
			neighbors[a] = append(neighbors[a], b)
			neighbors[b] = append(neighbors[b], a)
		}
	}

	// Build cardinality map from all factors.
	cardMap := make(map[string]int)
	for _, fl := range facCopy {
		for _, f := range fl {
			vars := f.Variables()
			card := f.Cardinality()
			for i, v := range vars {
				if _, ok := cardMap[v]; !ok {
					cardMap[v] = card[i]
				}
			}
		}
	}

	return &BeliefPropagation{
		cliques:        cliqueCopy,
		neighbors:      neighbors,
		separators:     sepCopy,
		initialFactors: facCopy,
		potentials:     make([]*factors.DiscreteFactor, len(cliques)),
		messages:       make(map[string]*factors.DiscreteFactor),
		calibrated:     false,
		cardMap:        cardMap,
	}
}

// msgKey returns the directed message key from clique src to clique dst.
func msgKey(src, dst int) string {
	return fmt.Sprintf("%d->%d", src, dst)
}

// initializePotentials creates the initial potential for each clique by
// multiplying all factors assigned to it. If a clique has no factors, a
// uniform potential over the clique's variables is created.
func (bp *BeliefPropagation) initializePotentials() error {
	for i, vars := range bp.cliques {
		fl := bp.initialFactors[i]
		if len(fl) == 0 {
			// Create a uniform (all-ones) factor over the clique variables.
			card := make([]int, len(vars))
			size := 1
			for j, v := range vars {
				c, ok := bp.cardMap[v]
				if !ok {
					return fmt.Errorf("belief_propagation: unknown cardinality for variable %q", v)
				}
				card[j] = c
				size *= c
			}
			vals := make([]float64, size)
			for k := range vals {
				vals[k] = 1.0
			}
			f, err := factors.NewDiscreteFactor(vars, card, vals)
			if err != nil {
				return fmt.Errorf("belief_propagation: failed to create uniform potential for clique %d: %w", i, err)
			}
			bp.potentials[i] = f
		} else {
			prod, err := factors.FactorProduct(fl...)
			if err != nil {
				return fmt.Errorf("belief_propagation: failed to compute initial potential for clique %d: %w", i, err)
			}
			bp.potentials[i] = prod
		}
	}
	return nil
}

// Calibrate runs the collect-distribute message passing algorithm on the
// junction tree until all clique potentials are calibrated.
//
// Steps:
//  1. Initialize clique potentials by multiplying assigned factors.
//  2. Choose clique 0 as root.
//  3. Collect phase: leaves send messages toward root (post-order).
//  4. Distribute phase: root sends messages toward leaves (pre-order).
//  5. Update each clique potential by multiplying in all incoming messages.
func (bp *BeliefPropagation) Calibrate() error {
	if err := bp.initializePotentials(); err != nil {
		return err
	}

	if len(bp.cliques) == 0 {
		bp.calibrated = true
		return nil
	}

	if len(bp.cliques) == 1 {
		// Single clique: already calibrated.
		bp.calibrated = true
		return nil
	}

	root := 0

	// Build a rooted tree via BFS to get parent pointers and ordering.
	parent := make([]int, len(bp.cliques))
	for i := range parent {
		parent[i] = -1
	}
	visited := make([]bool, len(bp.cliques))
	visited[root] = true
	queue := []int{root}
	// bfsOrder will give us a BFS traversal from root.
	var bfsOrder []int
	bfsOrder = append(bfsOrder, root)

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for _, nb := range bp.neighbors[curr] {
			if !visited[nb] {
				visited[nb] = true
				parent[nb] = curr
				queue = append(queue, nb)
				bfsOrder = append(bfsOrder, nb)
			}
		}
	}

	// Collect phase: process in reverse BFS order (leaves first).
	for idx := len(bfsOrder) - 1; idx >= 1; idx-- {
		child := bfsOrder[idx]
		par := parent[child]
		msg, err := bp.computeMessage(child, par)
		if err != nil {
			return fmt.Errorf("belief_propagation: collect message %d->%d failed: %w", child, par, err)
		}
		bp.messages[msgKey(child, par)] = msg
	}

	// Distribute phase: process in BFS order (root first).
	for idx := 0; idx < len(bfsOrder)-1; idx++ {
		par := bfsOrder[idx]
		// Send to each child.
		for _, nb := range bp.neighbors[par] {
			if parent[nb] == par {
				msg, err := bp.computeMessage(par, nb)
				if err != nil {
					return fmt.Errorf("belief_propagation: distribute message %d->%d failed: %w", par, nb, err)
				}
				bp.messages[msgKey(par, nb)] = msg
			}
		}
	}

	// Update potentials: each clique's belief = initial potential * all incoming messages.
	for i := range bp.cliques {
		belief := bp.potentials[i]
		for _, nb := range bp.neighbors[i] {
			key := msgKey(nb, i)
			if msg, ok := bp.messages[key]; ok {
				prod, err := factors.FactorProduct(belief, msg)
				if err != nil {
					return fmt.Errorf("belief_propagation: failed to absorb message into clique %d: %w", i, err)
				}
				belief = prod
			}
		}
		bp.potentials[i] = belief
	}

	bp.calibrated = true
	return nil
}

// computeMessage computes the message from clique src to clique dst.
// message(src->dst) = marginalize(potential_src * product(incoming messages
// to src except from dst), down to separator(src,dst)).
func (bp *BeliefPropagation) computeMessage(src, dst int) (*factors.DiscreteFactor, error) {
	// Start with the initial potential of src.
	current := bp.potentials[src]

	// Multiply in all incoming messages to src except from dst.
	for _, nb := range bp.neighbors[src] {
		if nb == dst {
			continue
		}
		key := msgKey(nb, src)
		if msg, ok := bp.messages[key]; ok {
			prod, err := factors.FactorProduct(current, msg)
			if err != nil {
				return nil, err
			}
			current = prod
		}
	}

	// Marginalize down to separator variables.
	sepKey := edgeKey(src, dst)
	sepVars := bp.separators[sepKey]

	// Variables to marginalize out = clique vars - separator vars.
	sepSet := make(map[string]bool, len(sepVars))
	for _, v := range sepVars {
		sepSet[v] = true
	}

	currentVars := current.Variables()
	var margVars []string
	for _, v := range currentVars {
		if !sepSet[v] {
			margVars = append(margVars, v)
		}
	}

	if len(margVars) == 0 {
		// Nothing to marginalize; message is the full potential.
		return current.Copy(), nil
	}

	msg, err := current.Marginalize(margVars)
	if err != nil {
		return nil, err
	}
	return msg, nil
}

// Query computes P(queryVars | evidence) after calibration.
//
// When evidence is given, the method enters evidence by multiplying
// indicator factors (delta functions) into the calibrated clique beliefs,
// then extracts the answer from the appropriate clique. For each evidence
// variable, an indicator factor is applied to every clique that contains
// that variable.
func (bp *BeliefPropagation) Query(queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("belief_propagation: queryVars must not be empty")
	}
	if !bp.calibrated {
		return nil, fmt.Errorf("belief_propagation: must call Calibrate() before Query()")
	}

	// Find a clique containing all query vars.
	cliqueIdx := bp.findClique(queryVars)
	if cliqueIdx < 0 {
		return nil, fmt.Errorf("belief_propagation: no clique contains all query variables %v", queryVars)
	}

	if len(evidence) == 0 {
		// No evidence: just extract from calibrated belief.
		return bp.extractFromBelief(bp.potentials[cliqueIdx], queryVars)
	}

	// With evidence: re-calibrate with evidence entered as indicator
	// factors. We create indicator factors for each evidence variable
	// and assign them to one clique that contains that variable.
	evidenceFactors := make(map[int][]*factors.DiscreteFactor)
	for v, val := range evidence {
		// Find a clique containing this evidence variable.
		found := false
		for ci, clique := range bp.cliques {
			for _, cv := range clique {
				if cv == v {
					card, ok := bp.cardMap[v]
					if !ok {
						return nil, fmt.Errorf("belief_propagation: unknown cardinality for evidence variable %q", v)
					}
					// Create indicator factor: 1 at val, 0 elsewhere.
					vals := make([]float64, card)
					vals[val] = 1.0
					indicator, err := factors.NewDiscreteFactor([]string{v}, []int{card}, vals)
					if err != nil {
						return nil, fmt.Errorf("belief_propagation: failed to create indicator for %q: %w", v, err)
					}
					evidenceFactors[ci] = append(evidenceFactors[ci], indicator)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("belief_propagation: evidence variable %q not found in any clique", v)
		}
	}

	// Create a new BP with the original factors plus indicator factors.
	augFactors := make(map[int][]*factors.DiscreteFactor, len(bp.initialFactors))
	for idx, fl := range bp.initialFactors {
		augFactors[idx] = make([]*factors.DiscreteFactor, len(fl))
		for i, f := range fl {
			augFactors[idx][i] = f.Copy()
		}
	}
	for idx, indicators := range evidenceFactors {
		for _, ind := range indicators {
			augFactors[idx] = append(augFactors[idx], ind)
		}
	}

	tmp := NewBeliefPropagation(bp.cliques, bp.separators, augFactors)
	if err := tmp.Calibrate(); err != nil {
		return nil, fmt.Errorf("belief_propagation: re-calibration with evidence failed: %w", err)
	}

	return bp.extractFromBelief(tmp.potentials[cliqueIdx], queryVars)
}

// extractFromBelief marginalizes a clique belief to the query variables
// and normalizes.
func (bp *BeliefPropagation) extractFromBelief(belief *factors.DiscreteFactor, queryVars []string) (*factors.DiscreteFactor, error) {
	result := belief.Copy()

	querySet := make(map[string]bool, len(queryVars))
	for _, v := range queryVars {
		querySet[v] = true
	}

	beliefVars := result.Variables()
	var margVars []string
	for _, v := range beliefVars {
		if !querySet[v] {
			margVars = append(margVars, v)
		}
	}

	if len(margVars) > 0 {
		marg, err := result.Marginalize(margVars)
		if err != nil {
			return nil, fmt.Errorf("belief_propagation: marginalization failed: %w", err)
		}
		result = marg
	}

	result.Normalize()
	return result, nil
}

// findClique returns the index of a clique that contains all the given
// variables, or -1 if none is found.
func (bp *BeliefPropagation) findClique(vars []string) int {
	for i, clique := range bp.cliques {
		cSet := make(map[string]bool, len(clique))
		for _, v := range clique {
			cSet[v] = true
		}
		allFound := true
		for _, v := range vars {
			if !cSet[v] {
				allFound = false
				break
			}
		}
		if allFound {
			return i
		}
	}
	return -1
}

// GetCliqueBelief returns the calibrated potential for the given clique.
// Returns nil if the index is out of range or Calibrate has not been called.
func (bp *BeliefPropagation) GetCliqueBelief(cliqueIndex int) *factors.DiscreteFactor {
	if cliqueIndex < 0 || cliqueIndex >= len(bp.potentials) {
		return nil
	}
	if bp.potentials[cliqueIndex] == nil {
		return nil
	}
	return bp.potentials[cliqueIndex].Copy()
}

// IsCalibrated returns true if Calibrate has been called successfully.
func (bp *BeliefPropagation) IsCalibrated() bool {
	return bp.calibrated
}

// MaxCalibrate runs the collect-distribute message passing algorithm using
// max-product messages instead of sum-product. After max-calibration, each
// clique potential represents the max-marginal over its variables.
func (bp *BeliefPropagation) MaxCalibrate() error {
	if err := bp.initializePotentials(); err != nil {
		return err
	}

	if len(bp.cliques) == 0 {
		bp.calibrated = true
		return nil
	}

	if len(bp.cliques) == 1 {
		bp.calibrated = true
		return nil
	}

	root := 0

	parent := make([]int, len(bp.cliques))
	for i := range parent {
		parent[i] = -1
	}
	visited := make([]bool, len(bp.cliques))
	visited[root] = true
	queue := []int{root}
	var bfsOrder []int
	bfsOrder = append(bfsOrder, root)

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]
		for _, nb := range bp.neighbors[curr] {
			if !visited[nb] {
				visited[nb] = true
				parent[nb] = curr
				queue = append(queue, nb)
				bfsOrder = append(bfsOrder, nb)
			}
		}
	}

	// Collect phase with max messages.
	for idx := len(bfsOrder) - 1; idx >= 1; idx-- {
		child := bfsOrder[idx]
		par := parent[child]
		msg, err := bp.computeMaxMessage(child, par)
		if err != nil {
			return fmt.Errorf("belief_propagation: max collect message %d->%d failed: %w", child, par, err)
		}
		bp.messages[msgKey(child, par)] = msg
	}

	// Distribute phase with max messages.
	for idx := 0; idx < len(bfsOrder)-1; idx++ {
		par := bfsOrder[idx]
		for _, nb := range bp.neighbors[par] {
			if parent[nb] == par {
				msg, err := bp.computeMaxMessage(par, nb)
				if err != nil {
					return fmt.Errorf("belief_propagation: max distribute message %d->%d failed: %w", par, nb, err)
				}
				bp.messages[msgKey(par, nb)] = msg
			}
		}
	}

	// Update potentials.
	for i := range bp.cliques {
		belief := bp.potentials[i]
		for _, nb := range bp.neighbors[i] {
			key := msgKey(nb, i)
			if msg, ok := bp.messages[key]; ok {
				prod, err := factors.FactorProduct(belief, msg)
				if err != nil {
					return fmt.Errorf("belief_propagation: failed to absorb max message into clique %d: %w", i, err)
				}
				belief = prod
			}
		}
		bp.potentials[i] = belief
	}

	bp.calibrated = true
	return nil
}

// computeMaxMessage computes the max-product message from clique src to dst.
// Like computeMessage but uses max instead of sum for marginalization.
func (bp *BeliefPropagation) computeMaxMessage(src, dst int) (*factors.DiscreteFactor, error) {
	current := bp.potentials[src]

	for _, nb := range bp.neighbors[src] {
		if nb == dst {
			continue
		}
		key := msgKey(nb, src)
		if msg, ok := bp.messages[key]; ok {
			prod, err := factors.FactorProduct(current, msg)
			if err != nil {
				return nil, err
			}
			current = prod
		}
	}

	sepKey := edgeKey(src, dst)
	sepVars := bp.separators[sepKey]

	sepSet := make(map[string]bool, len(sepVars))
	for _, v := range sepVars {
		sepSet[v] = true
	}

	currentVars := current.Variables()
	var margVars []string
	for _, v := range currentVars {
		if !sepSet[v] {
			margVars = append(margVars, v)
		}
	}

	if len(margVars) == 0 {
		return current.Copy(), nil
	}

	// Max-marginalize: for each variable, take the max instead of sum.
	result := current
	for _, mv := range margVars {
		maxed, err := maxMarginalizeOne(result, mv)
		if err != nil {
			return nil, err
		}
		result = maxed
	}
	return result, nil
}

// maxMarginalizeOne maximizes out a single variable from a factor.
func maxMarginalizeOne(f *factors.DiscreteFactor, variable string) (*factors.DiscreteFactor, error) {
	fVars := f.Variables()
	fCard := f.Cardinality()
	idx := -1
	for i, v := range fVars {
		if v == variable {
			idx = i
			break
		}
	}
	if idx == -1 {
		return nil, fmt.Errorf("belief_propagation: variable %q not in factor", variable)
	}

	var newVars []string
	var newCard []int
	for i, v := range fVars {
		if i != idx {
			newVars = append(newVars, v)
			newCard = append(newCard, fCard[i])
		}
	}

	newSize := 1
	for _, c := range newCard {
		newSize *= c
	}

	newValues := make([]float64, newSize)
	initialized := make([]bool, newSize)

	totalSize := 1
	for _, c := range fCard {
		totalSize *= c
	}
	data := f.Values().Data()

	for flat := 0; flat < totalSize; flat++ {
		// Decompose flat index.
		assignment := make(map[string]int, len(fVars))
		rem := flat
		for i := len(fVars) - 1; i >= 0; i-- {
			assignment[fVars[i]] = rem % fCard[i]
			rem /= fCard[i]
		}
		// Compute flat index in new factor.
		newFlat := 0
		stride := 1
		for i := len(newVars) - 1; i >= 0; i-- {
			newFlat += assignment[newVars[i]] * stride
			stride *= newCard[i]
		}
		val := data[flat]
		if !initialized[newFlat] || val > newValues[newFlat] {
			newValues[newFlat] = val
			initialized[newFlat] = true
		}
	}

	return factors.NewDiscreteFactor(newVars, newCard, newValues)
}

// MAPQuery performs MAP inference via max-calibration. It finds the
// most probable joint assignment to queryVars given evidence.
func (bp *BeliefPropagation) MAPQuery(queryVars []string, evidence map[string]int) (map[string]int, error) {
	if len(queryVars) == 0 {
		return nil, fmt.Errorf("belief_propagation: queryVars must not be empty")
	}

	// Build augmented factors with evidence indicators.
	augFactors := make(map[int][]*factors.DiscreteFactor, len(bp.initialFactors))
	for idx, fl := range bp.initialFactors {
		augFactors[idx] = make([]*factors.DiscreteFactor, len(fl))
		for i, f := range fl {
			augFactors[idx][i] = f.Copy()
		}
	}

	for v, val := range evidence {
		found := false
		for ci, clique := range bp.cliques {
			for _, cv := range clique {
				if cv == v {
					card, ok := bp.cardMap[v]
					if !ok {
						return nil, fmt.Errorf("belief_propagation: unknown cardinality for evidence variable %q", v)
					}
					vals := make([]float64, card)
					vals[val] = 1.0
					indicator, err := factors.NewDiscreteFactor([]string{v}, []int{card}, vals)
					if err != nil {
						return nil, fmt.Errorf("belief_propagation: failed to create indicator for %q: %w", v, err)
					}
					augFactors[ci] = append(augFactors[ci], indicator)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("belief_propagation: evidence variable %q not found in any clique", v)
		}
	}

	tmp := NewBeliefPropagation(bp.cliques, bp.separators, augFactors)
	if err := tmp.MaxCalibrate(); err != nil {
		return nil, fmt.Errorf("belief_propagation: max-calibration failed: %w", err)
	}

	// Find a clique containing all query vars.
	cliqueIdx := tmp.findClique(queryVars)
	if cliqueIdx < 0 {
		return nil, fmt.Errorf("belief_propagation: no clique contains all query variables %v", queryVars)
	}

	belief := tmp.potentials[cliqueIdx]

	// Marginalize to query vars using max.
	querySet := make(map[string]bool, len(queryVars))
	for _, v := range queryVars {
		querySet[v] = true
	}
	beliefVars := belief.Variables()
	var margVars []string
	for _, v := range beliefVars {
		if !querySet[v] {
			margVars = append(margVars, v)
		}
	}
	result := belief
	for _, mv := range margVars {
		maxed, err := maxMarginalizeOne(result, mv)
		if err != nil {
			return nil, err
		}
		result = maxed
	}

	// Find the assignment that maximizes the result factor.
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

// GetCliques returns all cliques in the junction tree.
func (bp *BeliefPropagation) GetCliques() [][]string {
	result := make([][]string, len(bp.cliques))
	for i, c := range bp.cliques {
		cp := make([]string, len(c))
		copy(cp, c)
		result[i] = cp
	}
	return result
}

// GetSepsetBeliefs returns the calibrated separator beliefs. Each separator
// belief is computed by marginalizing the adjacent clique belief down to
// the separator variables. Returns nil values if not calibrated.
func (bp *BeliefPropagation) GetSepsetBeliefs() map[string]*factors.DiscreteFactor {
	result := make(map[string]*factors.DiscreteFactor, len(bp.separators))

	if !bp.calibrated {
		for k := range bp.separators {
			result[k] = nil
		}
		return result
	}

	for k, sepVars := range bp.separators {
		// Parse the edge key to find an adjacent clique.
		a, b := parseEdgeKey(k)
		if a < 0 || a >= len(bp.potentials) {
			result[k] = nil
			continue
		}

		// Marginalize clique a's belief to separator vars.
		belief := bp.potentials[a]
		if belief == nil {
			result[k] = nil
			continue
		}

		sepSet := make(map[string]bool, len(sepVars))
		for _, v := range sepVars {
			sepSet[v] = true
		}

		beliefVars := belief.Variables()
		var margVars []string
		for _, v := range beliefVars {
			if !sepSet[v] {
				margVars = append(margVars, v)
			}
		}

		if len(margVars) == 0 {
			result[k] = belief.Copy()
			continue
		}

		marg, err := belief.Marginalize(margVars)
		if err != nil {
			// If marginalization fails, just use clique b.
			if b >= 0 && b < len(bp.potentials) && bp.potentials[b] != nil {
				beliefB := bp.potentials[b]
				beliefBVars := beliefB.Variables()
				var margVarsB []string
				for _, v := range beliefBVars {
					if !sepSet[v] {
						margVarsB = append(margVarsB, v)
					}
				}
				if len(margVarsB) > 0 {
					margB, errB := beliefB.Marginalize(margVarsB)
					if errB == nil {
						result[k] = margB
						continue
					}
				}
			}
			result[k] = nil
			continue
		}
		result[k] = marg
	}

	return result
}

// String returns a human-readable summary of the junction tree structure.
func (bp *BeliefPropagation) String() string {
	var b strings.Builder
	b.WriteString("BeliefPropagation(\n")
	for i, c := range bp.cliques {
		sorted := make([]string, len(c))
		copy(sorted, c)
		sort.Strings(sorted)
		b.WriteString(fmt.Sprintf("  clique %d: {%s}\n", i, strings.Join(sorted, ", ")))
	}
	for k, v := range bp.separators {
		sorted := make([]string, len(v))
		copy(sorted, v)
		sort.Strings(sorted)
		b.WriteString(fmt.Sprintf("  separator %s: {%s}\n", k, strings.Join(sorted, ", ")))
	}
	b.WriteString(fmt.Sprintf("  calibrated: %v\n", bp.calibrated))
	b.WriteString(")")
	return b.String()
}
