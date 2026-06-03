package base

import (
	"fmt"
	"math/rand"
	"sort"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

// DAG is a directed acyclic graph. It wraps a graphgo.DiGraph and enforces
// acyclicity on every edge addition.
type DAG struct {
	g *graphgo.DiGraph
}

// NewDAG creates a new empty DAG.
func NewDAG() *DAG {
	return &DAG{g: graphgo.NewDiGraph()}
}

// AddNode adds a node to the DAG. Returns an error if the node already exists.
func (d *DAG) AddNode(node string) error {
	if d.g.HasNode(node) {
		return fmt.Errorf("base: node %q already exists", node)
	}
	d.g.AddNode(node)
	return nil
}

// AddNodes adds multiple nodes. Returns an error if any node already exists;
// nodes added before the error are retained.
func (d *DAG) AddNodes(nodes ...string) error {
	for _, n := range nodes {
		if err := d.AddNode(n); err != nil {
			return err
		}
	}
	return nil
}

// RemoveNode removes a node and all its incident edges.
// Returns an error if the node does not exist.
func (d *DAG) RemoveNode(node string) error {
	if !d.g.HasNode(node) {
		return fmt.Errorf("base: node %q not found", node)
	}
	d.g.RemoveNode(node)
	return nil
}

// AddEdge adds a directed edge from -> to. Both nodes must already exist.
// Returns an error if the edge would create a cycle, if either node does not
// exist, or if the edge already exists.
func (d *DAG) AddEdge(from, to string) error {
	if !d.g.HasNode(from) {
		return fmt.Errorf("base: node %q not found", from)
	}
	if !d.g.HasNode(to) {
		return fmt.Errorf("base: node %q not found", to)
	}
	if d.g.HasEdge(from, to) {
		return fmt.Errorf("base: edge (%q, %q) already exists", from, to)
	}

	// Temporarily add the edge and check acyclicity.
	d.g.AddEdge(from, to)
	if !graphgo.IsDAG(d.g) {
		// Revert: remove the edge we just added.
		_ = d.g.RemoveEdge(from, to)
		return fmt.Errorf("base: edge (%q, %q) would create a cycle", from, to)
	}
	return nil
}

// RemoveEdge removes a directed edge. Returns an error if it does not exist.
func (d *DAG) RemoveEdge(from, to string) error {
	return d.g.RemoveEdge(from, to)
}

// HasNode returns true if the node exists in the DAG.
func (d *DAG) HasNode(node string) bool {
	return d.g.HasNode(node)
}

// HasEdge returns true if the directed edge exists.
func (d *DAG) HasEdge(from, to string) bool {
	return d.g.HasEdge(from, to)
}

// Nodes returns a sorted list of all nodes.
func (d *DAG) Nodes() []string {
	nodes := d.g.Nodes()
	sort.Strings(nodes)
	return nodes
}

// Edges returns all directed edges, sorted lexicographically.
func (d *DAG) Edges() []graphgo.Edge {
	edges := d.g.Edges()
	sort.Slice(edges, func(i, j int) bool {
		if edges[i].Src != edges[j].Src {
			return edges[i].Src < edges[j].Src
		}
		return edges[i].Dst < edges[j].Dst
	})
	return edges
}

// Parents returns the sorted parents (predecessors) of a node.
func (d *DAG) Parents(node string) []string {
	p := d.g.Parents(node)
	sort.Strings(p)
	return p
}

// Children returns the sorted children (successors) of a node.
func (d *DAG) Children(node string) []string {
	c := d.g.Children(node)
	sort.Strings(c)
	return c
}

// GetRoots returns nodes with no parents, sorted.
func (d *DAG) GetRoots() []string {
	var roots []string
	for _, n := range d.Nodes() {
		if d.g.InDegree(n) == 0 {
			roots = append(roots, n)
		}
	}
	return roots
}

// GetLeaves returns nodes with no children, sorted.
func (d *DAG) GetLeaves() []string {
	var leaves []string
	for _, n := range d.Nodes() {
		if d.g.OutDegree(n) == 0 {
			leaves = append(leaves, n)
		}
	}
	return leaves
}

// TopologicalOrder returns a topological ordering of the DAG.
func (d *DAG) TopologicalOrder() ([]string, error) {
	return graphgo.TopologicalSort(d.g)
}

// Copy returns a deep copy of the DAG.
func (d *DAG) Copy() *DAG {
	return &DAG{g: d.g.Copy()}
}

// DiGraph returns the underlying graphgo.DiGraph. This is useful for passing
// to library functions that operate on DiGraphs.
func (d *DAG) DiGraph() *graphgo.DiGraph {
	return d.g
}

// AddEdgesFrom adds all edges from the given list. Each element is [from, to].
// Stops on the first error.
func (d *DAG) AddEdgesFrom(edges [][2]string) error {
	for _, e := range edges {
		if err := d.AddEdge(e[0], e[1]); err != nil {
			return err
		}
	}
	return nil
}

// Moralize returns the moral graph of this DAG as an undirected graphgo.Graph.
// The moral graph connects all co-parents and drops edge directions.
func (d *DAG) Moralize() *graphgo.Graph {
	return graphgo.Moralize(d.g)
}

// GetIndependencies enumerates d-separation-based conditional independence
// assertions for all pairs of non-adjacent nodes. For each pair (X, Y) that
// are not directly connected, it checks whether they are d-separated given
// the parents of Y (and symmetrically the parents of X).
func (d *DAG) GetIndependencies() [][3][]string {
	nodes := d.Nodes()
	var result [][3][]string

	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			x, y := nodes[i], nodes[j]
			if d.HasEdge(x, y) || d.HasEdge(y, x) {
				continue
			}
			// Try conditioning on parents of Y.
			parentsY := d.Parents(y)
			xSet := map[string]bool{x: true}
			ySet := map[string]bool{y: true}
			zSet := make(map[string]bool, len(parentsY))
			for _, p := range parentsY {
				if p != x {
					zSet[p] = true
				}
			}
			if graphgo.DSeparation(d.g, xSet, ySet, zSet) {
				condSet := make([]string, 0, len(zSet))
				for v := range zSet {
					condSet = append(condSet, v)
				}
				sort.Strings(condSet)
				result = append(result, [3][]string{{x}, {y}, condSet})
			}
		}
	}
	return result
}

// LocalIndependencies returns the local independence assertions for a node.
// A node X is independent of its non-descendants given its parents.
func (d *DAG) LocalIndependencies(node string) [][3][]string {
	parents := d.Parents(node)
	descendants := graphgo.Descendants(d.g, node)
	descendants[node] = true

	parentSet := make(map[string]bool, len(parents))
	for _, p := range parents {
		parentSet[p] = true
	}

	var nonDescNonParents []string
	for _, n := range d.Nodes() {
		if !descendants[n] && !parentSet[n] && n != node {
			nonDescNonParents = append(nonDescNonParents, n)
		}
	}

	if len(nonDescNonParents) == 0 {
		return nil
	}

	return [][3][]string{{[]string{node}, nonDescNonParents, parents}}
}

// IsIEquivalent checks whether this DAG is I-equivalent to another DAG,
// meaning they have the same skeleton and same set of v-structures (immoralities).
func (d *DAG) IsIEquivalent(other *DAG) bool {
	nodes1 := d.Nodes()
	nodes2 := other.Nodes()
	if len(nodes1) != len(nodes2) {
		return false
	}
	for i := range nodes1 {
		if nodes1[i] != nodes2[i] {
			return false
		}
	}

	// Check same skeleton.
	for _, n := range nodes1 {
		adj1 := make(map[string]bool)
		for _, c := range d.Children(n) {
			adj1[c] = true
		}
		for _, p := range d.Parents(n) {
			adj1[p] = true
		}

		adj2 := make(map[string]bool)
		for _, c := range other.Children(n) {
			adj2[c] = true
		}
		for _, p := range other.Parents(n) {
			adj2[p] = true
		}

		if len(adj1) != len(adj2) {
			return false
		}
		for v := range adj1 {
			if !adj2[v] {
				return false
			}
		}
	}

	// Check same v-structures.
	imm1 := d.GetImmoralities()
	imm2 := other.GetImmoralities()
	if len(imm1) != len(imm2) {
		return false
	}
	immSet := make(map[[3]string]bool, len(imm1))
	for _, im := range imm1 {
		immSet[im] = true
	}
	for _, im := range imm2 {
		if !immSet[im] {
			return false
		}
	}
	return true
}

// GetImmoralities returns all v-structures (immoralities) in the DAG.
// A v-structure is a triple (parent1, child, parent2) where parent1 and
// parent2 are both parents of child but are not adjacent to each other.
// Each triple is returned with parent1 < parent2 lexicographically.
func (d *DAG) GetImmoralities() [][3]string {
	var immoralities [][3]string
	for _, node := range d.Nodes() {
		parents := d.Parents(node)
		for i := 0; i < len(parents); i++ {
			for j := i + 1; j < len(parents); j++ {
				p1, p2 := parents[i], parents[j]
				if !d.HasEdge(p1, p2) && !d.HasEdge(p2, p1) {
					if p1 > p2 {
						p1, p2 = p2, p1
					}
					immoralities = append(immoralities, [3]string{p1, node, p2})
				}
			}
		}
	}
	sort.Slice(immoralities, func(i, j int) bool {
		if immoralities[i][0] != immoralities[j][0] {
			return immoralities[i][0] < immoralities[j][0]
		}
		if immoralities[i][1] != immoralities[j][1] {
			return immoralities[i][1] < immoralities[j][1]
		}
		return immoralities[i][2] < immoralities[j][2]
	})
	return immoralities
}

// IsDConnected checks whether x and y are d-connected (NOT d-separated)
// given the observed set z.
func (d *DAG) IsDConnected(x, y string, z []string) bool {
	xSet := map[string]bool{x: true}
	ySet := map[string]bool{y: true}
	zSet := make(map[string]bool, len(z))
	for _, v := range z {
		zSet[v] = true
	}
	return !graphgo.DSeparation(d.g, xSet, ySet, zSet)
}

// MinimalDSeparator finds a minimal d-separating set between x and y.
// Returns the set and true if one exists, or nil and false if x and y
// cannot be d-separated.
func (d *DAG) MinimalDSeparator(x, y string) ([]string, bool) {
	allNodes := d.Nodes()
	var candidates []string
	for _, n := range allNodes {
		if n != x && n != y {
			candidates = append(candidates, n)
		}
	}

	xSet := map[string]bool{x: true}
	ySet := map[string]bool{y: true}
	fullSet := make(map[string]bool, len(candidates))
	for _, c := range candidates {
		fullSet[c] = true
	}
	if !graphgo.DSeparation(d.g, xSet, ySet, fullSet) {
		return nil, false
	}

	minimal := make([]string, len(candidates))
	copy(minimal, candidates)

	for i := 0; i < len(minimal); {
		candidate := make([]string, 0, len(minimal)-1)
		candidate = append(candidate, minimal[:i]...)
		candidate = append(candidate, minimal[i+1:]...)

		zSet := make(map[string]bool, len(candidate))
		for _, v := range candidate {
			zSet[v] = true
		}
		if graphgo.DSeparation(d.g, xSet, ySet, zSet) {
			minimal = candidate
		} else {
			i++
		}
	}

	return minimal, true
}

// GetMarkovBlanket returns the Markov blanket of a node: its parents, children,
// and co-parents (other parents of its children).
func (d *DAG) GetMarkovBlanket(node string) []string {
	blanket := graphgo.MarkovBlanket(d.g, node)
	result := make([]string, 0, len(blanket))
	for v := range blanket {
		result = append(result, v)
	}
	sort.Strings(result)
	return result
}

// ActiveTrailNodes returns the set of nodes reachable from the given node
// via active trails given the observed set.
func (d *DAG) ActiveTrailNodes(node string, observed []string) []string {
	obsSet := make(map[string]bool, len(observed))
	for _, v := range observed {
		obsSet[v] = true
	}

	nodeSet := map[string]bool{node: true}
	var active []string

	for _, n := range d.Nodes() {
		if n == node {
			continue
		}
		nSet := map[string]bool{n: true}
		if !graphgo.DSeparation(d.g, nodeSet, nSet, obsSet) {
			active = append(active, n)
		}
	}

	sort.Strings(active)
	return active
}

// GetAncestors returns all ancestors of a node (not including the node itself).
func (d *DAG) GetAncestors(node string) []string {
	anc := graphgo.Ancestors(d.g, node)
	result := make([]string, 0, len(anc))
	for v := range anc {
		result = append(result, v)
	}
	sort.Strings(result)
	return result
}

// GetDescendants returns all descendants of a node (not including the node itself).
func (d *DAG) GetDescendants(node string) []string {
	desc := graphgo.Descendants(d.g, node)
	result := make([]string, 0, len(desc))
	for v := range desc {
		result = append(result, v)
	}
	sort.Strings(result)
	return result
}

// ToPDAG converts this DAG to a PDAG (CPDAG) representing the Markov
// equivalence class. Compelled edges remain directed; reversible edges
// become undirected.
func (d *DAG) ToPDAG() *graphgo.PDAG {
	pdag := graphgo.NewPDAG()
	nodes := d.Nodes()
	for _, n := range nodes {
		pdag.AddNode(n)
	}

	compelled := make(map[[2]string]bool)
	for _, n := range nodes {
		parents := d.Parents(n)
		for i := 0; i < len(parents); i++ {
			for j := i + 1; j < len(parents); j++ {
				if !d.HasEdge(parents[i], parents[j]) && !d.HasEdge(parents[j], parents[i]) {
					compelled[[2]string{parents[i], n}] = true
					compelled[[2]string{parents[j], n}] = true
				}
			}
		}
	}

	edges := d.Edges()
	for _, e := range edges {
		if compelled[[2]string{e.Src, e.Dst}] {
			pdag.AddDirectedEdge(e.Src, e.Dst)
		} else {
			pdag.AddUndirectedEdge(e.Src, e.Dst)
		}
	}

	graphgo.ApplyMeekRules(pdag)
	return pdag
}

// Do returns a new DAG representing the interventional graph after performing
// do(node), which removes all incoming edges to the given node.
func (d *DAG) Do(node string) *DAG {
	newDAG := d.Copy()
	parents := newDAG.Parents(node)
	for _, p := range parents {
		_ = newDAG.RemoveEdge(p, node)
	}
	return newDAG
}

// GetAncestralGraph returns the ancestral graph: the subgraph induced by the
// given nodes and all their ancestors.
func (d *DAG) GetAncestralGraph(nodes []string) *DAG {
	keep := make(map[string]bool)
	for _, n := range nodes {
		keep[n] = true
		for v := range graphgo.Ancestors(d.g, n) {
			keep[v] = true
		}
	}

	var keepList []string
	for v := range keep {
		keepList = append(keepList, v)
	}
	sort.Strings(keepList)

	result := NewDAG()
	for _, n := range keepList {
		_ = result.AddNode(n)
	}
	for _, e := range d.Edges() {
		if keep[e.Src] && keep[e.Dst] {
			_ = result.AddEdge(e.Src, e.Dst)
		}
	}
	return result
}

// FromLavaan parses a DAG from lavaan-style syntax. Each line has the form
// "Y ~ X1 + X2" meaning edges X1->Y and X2->Y. Blank lines and lines without
// "~" are ignored.
func FromLavaan(syntax string) (*DAG, error) {
	if strings.TrimSpace(syntax) == "" {
		return nil, fmt.Errorf("base: empty lavaan syntax")
	}

	d := NewDAG()
	lines := strings.Split(syntax, "\n")
	parsed := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || !strings.Contains(line, "~") {
			continue
		}
		parts := strings.SplitN(line, "~", 2)
		if len(parts) != 2 {
			continue
		}
		child := strings.TrimSpace(parts[0])
		if child == "" {
			return nil, fmt.Errorf("base: empty child variable in lavaan line %q", line)
		}
		parentStr := strings.TrimSpace(parts[1])
		if parentStr == "" {
			return nil, fmt.Errorf("base: empty parent list in lavaan line %q", line)
		}
		parentTokens := strings.Split(parentStr, "+")
		for _, tok := range parentTokens {
			parent := strings.TrimSpace(tok)
			if parent == "" {
				continue
			}
			if !d.HasNode(parent) {
				if err := d.AddNode(parent); err != nil {
					return nil, err
				}
			}
			if !d.HasNode(child) {
				if err := d.AddNode(child); err != nil {
					return nil, err
				}
			}
			if err := d.AddEdge(parent, child); err != nil {
				return nil, fmt.Errorf("base: failed to add edge %q -> %q: %w", parent, child, err)
			}
			parsed = true
		}
	}
	if !parsed {
		return nil, fmt.Errorf("base: no valid lavaan lines found")
	}
	return d, nil
}

// FromDagitty parses a DAG from dagitty-style syntax. The expected format is
// "dag { X -> Y; Y -> Z }" or multiline with one edge per line/semicolon.
func FromDagitty(syntax string) (*DAG, error) {
	if strings.TrimSpace(syntax) == "" {
		return nil, fmt.Errorf("base: empty dagitty syntax")
	}

	// Strip outer "dag {" and "}".
	s := strings.TrimSpace(syntax)
	if strings.HasPrefix(s, "dag") {
		s = strings.TrimPrefix(s, "dag")
		s = strings.TrimSpace(s)
	}
	s = strings.TrimPrefix(s, "{")
	s = strings.TrimSuffix(s, "}")
	s = strings.TrimSpace(s)

	if s == "" {
		return nil, fmt.Errorf("base: no edges found in dagitty syntax")
	}

	d := NewDAG()
	// Split by semicolons and newlines.
	s = strings.ReplaceAll(s, "\n", ";")
	segments := strings.Split(s, ";")
	parsed := false
	for _, seg := range segments {
		seg = strings.TrimSpace(seg)
		if seg == "" {
			continue
		}
		// Look for "->" tokens.
		parts := strings.Split(seg, "->")
		if len(parts) < 2 {
			continue
		}
		for i := 0; i < len(parts)-1; i++ {
			src := strings.TrimSpace(parts[i])
			dst := strings.TrimSpace(parts[i+1])
			if src == "" || dst == "" {
				continue
			}
			if !d.HasNode(src) {
				if err := d.AddNode(src); err != nil {
					return nil, err
				}
			}
			if !d.HasNode(dst) {
				if err := d.AddNode(dst); err != nil {
					return nil, err
				}
			}
			if err := d.AddEdge(src, dst); err != nil {
				return nil, fmt.Errorf("base: failed to add edge %q -> %q: %w", src, dst, err)
			}
			parsed = true
		}
	}
	if !parsed {
		return nil, fmt.Errorf("base: no valid edges found in dagitty syntax")
	}
	return d, nil
}

// OutDegreeIter returns a map from each node to its out-degree.
func (d *DAG) OutDegreeIter() map[string]int {
	result := make(map[string]int)
	for _, n := range d.Nodes() {
		result[n] = len(d.Children(n))
	}
	return result
}

// InDegreeIter returns a map from each node to its in-degree.
func (d *DAG) InDegreeIter() map[string]int {
	result := make(map[string]int)
	for _, n := range d.Nodes() {
		result[n] = len(d.Parents(n))
	}
	return result
}

// GetRandomDAG generates a random DAG with nNodes nodes (named "X0", "X1", ...)
// and exactly nEdges edges. Edges are only added from lower-indexed to
// higher-indexed nodes to guarantee acyclicity.
func GetRandomDAG(nNodes, nEdges int, seed int64) (*DAG, error) {
	if nNodes <= 0 {
		return nil, fmt.Errorf("base: nNodes must be positive, got %d", nNodes)
	}
	maxEdges := nNodes * (nNodes - 1) / 2
	if nEdges < 0 || nEdges > maxEdges {
		return nil, fmt.Errorf("base: nEdges %d out of range [0, %d] for %d nodes", nEdges, maxEdges, nNodes)
	}

	rng := rand.New(rand.NewSource(seed))
	d := NewDAG()

	nodeNames := make([]string, nNodes)
	for i := 0; i < nNodes; i++ {
		nodeNames[i] = fmt.Sprintf("X%d", i)
		_ = d.AddNode(nodeNames[i])
	}

	// Collect all possible edges (i < j guarantees DAG).
	type edgePair struct{ i, j int }
	allEdges := make([]edgePair, 0, maxEdges)
	for i := 0; i < nNodes; i++ {
		for j := i + 1; j < nNodes; j++ {
			allEdges = append(allEdges, edgePair{i, j})
		}
	}

	// Shuffle and pick the first nEdges.
	rng.Shuffle(len(allEdges), func(i, j int) {
		allEdges[i], allEdges[j] = allEdges[j], allEdges[i]
	})
	for k := 0; k < nEdges; k++ {
		e := allEdges[k]
		_ = d.AddEdge(nodeNames[e.i], nodeNames[e.j])
	}

	return d, nil
}

// ToGraphviz returns a Graphviz DOT representation of the DAG.
func (d *DAG) ToGraphviz() string {
	var b strings.Builder
	b.WriteString("digraph {\n")
	for _, n := range d.Nodes() {
		b.WriteString(fmt.Sprintf("  %q;\n", n))
	}
	for _, e := range d.Edges() {
		b.WriteString(fmt.Sprintf("  %q -> %q;\n", e.Src, e.Dst))
	}
	b.WriteString("}\n")
	return b.String()
}

// ToLavaan returns a Lavaan-style syntax string representation.
func (d *DAG) ToLavaan() string {
	childParents := make(map[string][]string)
	for _, e := range d.Edges() {
		childParents[e.Dst] = append(childParents[e.Dst], e.Src)
	}

	var b strings.Builder
	for _, node := range d.Nodes() {
		parents := childParents[node]
		if len(parents) > 0 {
			sort.Strings(parents)
			b.WriteString(fmt.Sprintf("%s ~ %s\n", node, strings.Join(parents, " + ")))
		}
	}
	return b.String()
}

// ToDagitty returns a Dagitty-style syntax string representation.
func (d *DAG) ToDagitty() string {
	var b strings.Builder
	b.WriteString("dag {\n")
	for _, e := range d.Edges() {
		b.WriteString(fmt.Sprintf("  %s -> %s\n", e.Src, e.Dst))
	}
	b.WriteString("}\n")
	return b.String()
}

// ToDaft returns a Daft-style (Python library) representation string.
func (d *DAG) ToDaft() string {
	var b strings.Builder
	b.WriteString("# Daft representation\n")
	b.WriteString("import daft\n\n")
	b.WriteString("pgm = daft.PGM()\n")
	for i, n := range d.Nodes() {
		b.WriteString(fmt.Sprintf("pgm.add_node(%q, %q, %d, 0)\n", n, n, i))
	}
	for _, e := range d.Edges() {
		b.WriteString(fmt.Sprintf("pgm.add_edge(%q, %q)\n", e.Src, e.Dst))
	}
	b.WriteString("pgm.render()\n")
	return b.String()
}

// EdgeStrength computes a simple edge strength metric for each edge based on
// how many d-separation relationships the edge participates in.
func (d *DAG) EdgeStrength() map[[2]string]float64 {
	strengths := make(map[[2]string]float64)
	edges := d.Edges()

	for _, e := range edges {
		key := [2]string{e.Src, e.Dst}
		modified := d.Copy()
		_ = modified.RemoveEdge(e.Src, e.Dst)

		count := 0.0
		nodes := d.Nodes()
		for i := 0; i < len(nodes); i++ {
			for j := i + 1; j < len(nodes); j++ {
				x, y := nodes[i], nodes[j]
				xSet := map[string]bool{x: true}
				ySet := map[string]bool{y: true}

				origSep := graphgo.DSeparation(d.g, xSet, ySet, map[string]bool{})
				modSep := graphgo.DSeparation(modified.g, xSet, ySet, map[string]bool{})

				if !origSep && modSep {
					count++
				}
			}
		}
		strengths[key] = count
	}

	return strengths
}

// DAGStats holds summary statistics for a DAG.
type DAGStats struct {
	NumNodes  int
	NumEdges  int
	NumRoots  int
	NumLeaves int
}

// GetStats returns summary statistics about the DAG.
func (d *DAG) GetStats() DAGStats {
	return DAGStats{
		NumNodes:  len(d.Nodes()),
		NumEdges:  len(d.Edges()),
		NumRoots:  len(d.GetRoots()),
		NumLeaves: len(d.GetLeaves()),
	}
}

// GetRandom generates a random DAG with the given nodes and approximately
// edgeProbability chance of each valid edge being present.
func GetRandom(nodes []string, edgeProbability float64, seed int64) *DAG {
	rng := rand.New(rand.NewSource(seed))
	d := NewDAG()

	sorted := make([]string, len(nodes))
	copy(sorted, nodes)
	sort.Strings(sorted)

	for _, n := range sorted {
		_ = d.AddNode(n)
	}

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if rng.Float64() < edgeProbability {
				_ = d.AddEdge(sorted[i], sorted[j])
			}
		}
	}

	return d
}
