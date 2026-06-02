package metrics

import "github.com/asymmetric-effort/pgmgo/lib/graphgo"

// undirectedPair returns a canonical (sorted) pair key for an unordered edge.
func undirectedPair(a, b string) [2]string {
	if a < b {
		return [2]string{a, b}
	}
	return [2]string{b, a}
}

// allNodes returns the union of nodes from two digraphs.
func allNodes(g1, g2 *graphgo.DiGraph) map[string]bool {
	nodes := make(map[string]bool)
	for _, n := range g1.Nodes() {
		nodes[n] = true
	}
	for _, n := range g2.Nodes() {
		nodes[n] = true
	}
	return nodes
}

// SHD computes the Structural Hamming Distance between the true and estimated
// directed graphs. It counts the number of edge additions, deletions, and
// reversals needed to transform estimated into true. Each reversal counts as
// one operation (not two).
func SHD(trueG, estimated *graphgo.DiGraph) int {
	// Build sets of directed edges for both graphs.
	trueEdges := make(map[[2]string]bool)
	for _, e := range trueG.Edges() {
		trueEdges[[2]string{e.Src, e.Dst}] = true
	}
	estEdges := make(map[[2]string]bool)
	for _, e := range estimated.Edges() {
		estEdges[[2]string{e.Src, e.Dst}] = true
	}

	// Collect all undirected pairs that appear in either graph.
	allPairs := make(map[[2]string]bool)
	for e := range trueEdges {
		allPairs[undirectedPair(e[0], e[1])] = true
	}
	for e := range estEdges {
		allPairs[undirectedPair(e[0], e[1])] = true
	}

	dist := 0
	for pair := range allPairs {
		a, b := pair[0], pair[1]
		// Check which directed edges exist in each graph for this pair.
		tAB := trueEdges[[2]string{a, b}]
		tBA := trueEdges[[2]string{b, a}]
		eAB := estEdges[[2]string{a, b}]
		eBA := estEdges[[2]string{b, a}]

		// Compare the edge configuration for this pair.
		// Possible states per graph: none, A->B, B->A, both.
		// We count minimal edits.
		if tAB == eAB && tBA == eBA {
			// Identical configuration for this pair.
			continue
		}

		// Count how many directed edges differ.
		diffs := 0
		if tAB != eAB {
			diffs++
		}
		if tBA != eBA {
			diffs++
		}

		if diffs == 2 {
			// Check if it's a pure reversal: one graph has A->B, other has B->A.
			if (tAB && eBA && !tBA && !eAB) || (tBA && eAB && !tAB && !eBA) {
				dist++ // single reversal
			} else {
				dist += 2 // two independent changes
			}
		} else {
			dist++ // one addition or deletion
		}
	}
	return dist
}

// AdjacencyConfusionMatrix computes the confusion matrix treating edges as
// undirected. For every unordered pair of nodes:
//   - TP: edge exists in both graphs (regardless of direction)
//   - FP: edge exists in estimated only
//   - FN: edge exists in true only
//   - TN: edge exists in neither
func AdjacencyConfusionMatrix(trueG, estimated *graphgo.DiGraph) (tp, fp, tn, fn int) {
	nodes := allNodes(trueG, estimated)

	// Build undirected adjacency sets.
	trueAdj := make(map[[2]string]bool)
	for _, e := range trueG.Edges() {
		trueAdj[undirectedPair(e.Src, e.Dst)] = true
	}
	estAdj := make(map[[2]string]bool)
	for _, e := range estimated.Edges() {
		estAdj[undirectedPair(e.Src, e.Dst)] = true
	}

	// Enumerate all unordered pairs.
	nodeList := make([]string, 0, len(nodes))
	for n := range nodes {
		nodeList = append(nodeList, n)
	}
	for i := 0; i < len(nodeList); i++ {
		for j := i + 1; j < len(nodeList); j++ {
			pair := undirectedPair(nodeList[i], nodeList[j])
			inTrue := trueAdj[pair]
			inEst := estAdj[pair]
			switch {
			case inTrue && inEst:
				tp++
			case !inTrue && inEst:
				fp++
			case inTrue && !inEst:
				fn++
			default:
				tn++
			}
		}
	}
	return
}

// OrientationConfusionMatrix computes the orientation confusion matrix among
// edges that are adjacency true positives (present in both graphs as undirected
// edges). For each such pair:
//   - TP: same directed edge exists in both
//   - FP: estimated has an orientation not in true
//   - FN: true has an orientation not in estimated
//   - TN: neither graph has a particular directed edge for this pair
//
// Since each undirected pair corresponds to two possible directed edges (A->B
// and B->A), we evaluate both directions for each adjacency-TP pair.
func OrientationConfusionMatrix(trueG, estimated *graphgo.DiGraph) (tp, fp, tn, fn int) {
	// Find adjacency true positives.
	trueAdj := make(map[[2]string]bool)
	for _, e := range trueG.Edges() {
		trueAdj[undirectedPair(e.Src, e.Dst)] = true
	}
	estAdj := make(map[[2]string]bool)
	for _, e := range estimated.Edges() {
		estAdj[undirectedPair(e.Src, e.Dst)] = true
	}

	// Collect adjacency TP pairs.
	var adjTP [][2]string
	for pair := range trueAdj {
		if estAdj[pair] {
			adjTP = append(adjTP, pair)
		}
	}

	// For each adjacency TP, check both possible directed edges.
	for _, pair := range adjTP {
		a, b := pair[0], pair[1]
		for _, dir := range [][2]string{{a, b}, {b, a}} {
			inTrue := trueG.HasEdge(dir[0], dir[1])
			inEst := estimated.HasEdge(dir[0], dir[1])
			switch {
			case inTrue && inEst:
				tp++
			case !inTrue && inEst:
				fp++
			case inTrue && !inEst:
				fn++
			default:
				tn++
			}
		}
	}
	return
}
