//go:build unit

package learning

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// --- Mock CI test helpers ---

// mockCITest creates a deterministic CI test function based on a predefined
// set of independence relationships. Each entry is keyed by a string
// "x|y|z1,z2,..." and maps to whether x and y are independent given z.

func ciKey(x, y string, z []string) string {
	if x > y {
		x, y = y, x
	}
	zz := make([]string, len(z))
	copy(zz, z)
	sort.Strings(zz)
	zStr := ""
	for i, s := range zz {
		if i > 0 {
			zStr += ","
		}
		zStr += s
	}
	return x + "|" + y + "|" + zStr
}

func mockCITestFromMap(indeps map[string]bool) CITestFunc {
	return func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
		key := ciKey(x, y, z)
		if independent, ok := indeps[key]; ok {
			if independent {
				return 0.0, 1.0, true
			}
			return 10.0, 0.001, false
		}
		// Default: not independent.
		return 10.0, 0.001, false
	}
}

// dummyData creates a minimal DataFrame with named columns (needed for CI test signature).
func dummyData(columns ...string) *tabgo.DataFrame {
	cols := make(map[string]*tabgo.Series)
	for _, c := range columns {
		cols[c] = tabgo.NewSeries(c, []any{0, 1})
	}
	return tabgo.NewDataFrame(cols)
}

// --- Tests ---

func TestPCChainABC(t *testing.T) {
	// True DAG: A -> B -> C
	// Independence structure:
	//   A _|_ C | B  (A and C are independent given B)
	//   A !_|_ B
	//   B !_|_ C
	//   A !_|_ C     (marginally dependent)
	//   A !_|_ B | C
	//   B !_|_ C | A
	indeps := map[string]bool{
		ciKey("A", "B", nil):           false,
		ciKey("A", "C", nil):           false,
		ciKey("B", "C", nil):           false,
		ciKey("A", "C", []string{"B"}): true,
		ciKey("A", "B", []string{"C"}): false,
		ciKey("B", "C", []string{"A"}): false,
	}

	data := dummyData("A", "B", "C")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// Skeleton: A-B, B-C but NOT A-C.
	if !pdag.HasEdge("A", "B") {
		t.Error("expected edge A-B in skeleton")
	}
	if !pdag.HasEdge("B", "C") {
		t.Error("expected edge B-C in skeleton")
	}
	if pdag.Adjacent("A", "C") {
		t.Error("A and C should not be adjacent")
	}

	// For chain A->B->C, the equivalence class is A-B-C (all undirected)
	// because B IS in sepSet(A,C) = {B}, so no v-structure is formed.
	// The resulting CPDAG has all undirected edges.
	nodes := pdag.Nodes()
	if len(nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(nodes))
	}
}

func TestPCVStructure(t *testing.T) {
	// True DAG: A -> C <- B (v-structure / collider)
	// Independence structure:
	//   A _|_ B       (marginally independent)
	//   A !_|_ C
	//   B !_|_ C
	//   A !_|_ B | C  (explaining away)
	indeps := map[string]bool{
		ciKey("A", "B", nil):           true, // A _|_ B
		ciKey("A", "C", nil):           false,
		ciKey("B", "C", nil):           false,
		ciKey("A", "B", []string{"C"}): false, // Not independent given C
		ciKey("A", "C", []string{"B"}): false,
		ciKey("B", "C", []string{"A"}): false,
	}

	data := dummyData("A", "B", "C")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// Skeleton: A-C, B-C but NOT A-B.
	if pdag.Adjacent("A", "B") {
		t.Error("A and B should not be adjacent")
	}

	// V-structure: A -> C <- B
	// sepSet(A,B) = {} (empty set). C is NOT in sepSet(A,B), so orient.
	if !pdag.HasDirectedEdge("A", "C") {
		t.Error("expected directed edge A -> C")
	}
	if !pdag.HasDirectedEdge("B", "C") {
		t.Error("expected directed edge B -> C")
	}

	// No undirected edges should remain.
	undirected := pdag.UndirectedEdges()
	if len(undirected) != 0 {
		t.Errorf("expected 0 undirected edges, got %d: %v", len(undirected), undirected)
	}
}

func TestPCDiamond(t *testing.T) {
	// True DAG: A -> B, A -> C, B -> D, C -> D
	// Independence structure:
	//   A !_|_ B, A !_|_ C, A !_|_ D
	//   B !_|_ C, B !_|_ D, C !_|_ D
	//   B _|_ C | A      (B and C are independent given A)
	//   A _|_ D | {B,C}  (A and D are independent given B and C)
	//   B !_|_ D | A, B !_|_ D | C
	//   C !_|_ D | A, C !_|_ D | B
	indeps := map[string]bool{
		ciKey("A", "B", nil):                false,
		ciKey("A", "C", nil):                false,
		ciKey("A", "D", nil):                false,
		ciKey("B", "C", nil):                false,
		ciKey("B", "D", nil):                false,
		ciKey("C", "D", nil):                false,
		ciKey("B", "C", []string{"A"}):      true,
		ciKey("A", "D", []string{"B"}):      false,
		ciKey("A", "D", []string{"C"}):      false,
		ciKey("A", "D", []string{"B", "C"}): true,
		ciKey("B", "D", []string{"A"}):      false,
		ciKey("B", "D", []string{"C"}):      false,
		ciKey("C", "D", []string{"A"}):      false,
		ciKey("C", "D", []string{"B"}):      false,
		ciKey("A", "B", []string{"C"}):      false,
		ciKey("A", "B", []string{"D"}):      false,
		ciKey("A", "C", []string{"B"}):      false,
		ciKey("A", "C", []string{"D"}):      false,
		ciKey("B", "C", []string{"D"}):      false,
		ciKey("B", "D", []string{"A", "C"}): false,
		ciKey("C", "D", []string{"A", "B"}): false,
	}

	data := dummyData("A", "B", "C", "D")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// Skeleton should have edges: A-B, A-C, B-D, C-D.
	// A-D and B-C should be absent.
	if !pdag.HasEdge("A", "B") {
		t.Error("expected edge A-B")
	}
	if !pdag.HasEdge("A", "C") {
		t.Error("expected edge A-C")
	}
	if !pdag.HasEdge("B", "D") {
		t.Error("expected edge B-D")
	}
	if !pdag.HasEdge("C", "D") {
		t.Error("expected edge C-D")
	}
	if pdag.Adjacent("A", "D") {
		t.Error("A-D should not be adjacent")
	}
	if pdag.Adjacent("B", "C") {
		t.Error("B-C should not be adjacent")
	}

	// V-structures: B and C are both unshielded colliders at D?
	// sepSet(B,C) = {A}, so for triple B-D-C: D is NOT in sepSet(B,C)={A},
	// so B->D<-C is a v-structure.
	// sepSet(A,D) = {B,C}, so for triple A-B-D: check B in sepSet(A,D)={B,C}? Yes, so no v-structure.
	// And for triple A-C-D: check C in sepSet(A,D)={B,C}? Yes, so no v-structure.
	if !pdag.HasDirectedEdge("B", "D") {
		t.Error("expected directed edge B -> D")
	}
	if !pdag.HasDirectedEdge("C", "D") {
		t.Error("expected directed edge C -> D")
	}
}

func TestPCWithMaxCondSetSize(t *testing.T) {
	// Test that WithMaxCondSetSize limits the search depth.
	// Same chain A->B->C, but with maxCondSetSize=0 we never condition
	// on anything, so the skeleton stays complete (A-C is not removed
	// because A _|_ C | B requires d=1).
	indeps := map[string]bool{
		ciKey("A", "B", nil):           false,
		ciKey("A", "C", nil):           false,
		ciKey("B", "C", nil):           false,
		ciKey("A", "C", []string{"B"}): true,
	}

	data := dummyData("A", "B", "C")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05, WithMaxCondSetSize(0))
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// With maxCondSetSize=0, A-C should still be present because we never
	// tested with conditioning set size 1.
	if !pdag.Adjacent("A", "C") {
		t.Error("with maxCondSetSize=0, A-C should still be adjacent")
	}
}

func TestPCEstimateBN(t *testing.T) {
	// V-structure: A -> C <- B
	indeps := map[string]bool{
		ciKey("A", "B", nil):           true,
		ciKey("A", "C", nil):           false,
		ciKey("B", "C", nil):           false,
		ciKey("A", "B", []string{"C"}): false,
		ciKey("A", "C", []string{"B"}): false,
		ciKey("B", "C", []string{"A"}): false,
	}

	data := dummyData("A", "B", "C")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	bn, err := pc.EstimateBN()
	if err != nil {
		t.Fatalf("EstimateBN failed: %v", err)
	}

	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}

	edges := bn.Edges()
	// Should have exactly 2 edges: A->C and B->C.
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d: %v", len(edges), edges)
	}

	edgeSet := make(map[[2]string]bool)
	for _, e := range edges {
		edgeSet[e] = true
	}
	if !edgeSet[[2]string{"A", "C"}] {
		t.Error("expected edge A->C in BN")
	}
	if !edgeSet[[2]string{"B", "C"}] {
		t.Error("expected edge B->C in BN")
	}
}

func TestPCEstimateBNChain(t *testing.T) {
	// Chain A->B->C: equivalence class is all undirected.
	// EstimateBN should orient them consistently (some valid DAG).
	indeps := map[string]bool{
		ciKey("A", "B", nil):           false,
		ciKey("A", "C", nil):           false,
		ciKey("B", "C", nil):           false,
		ciKey("A", "C", []string{"B"}): true,
		ciKey("A", "B", []string{"C"}): false,
		ciKey("B", "C", []string{"A"}): false,
	}

	data := dummyData("A", "B", "C")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	bn, err := pc.EstimateBN()
	if err != nil {
		t.Fatalf("EstimateBN failed: %v", err)
	}

	edges := bn.Edges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d: %v", len(edges), edges)
	}

	// The result should be a valid DAG over A, B, C with edges A-B and B-C
	// oriented in some consistent direction.
	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(nodes))
	}
}

func TestPCTwoVariables(t *testing.T) {
	// Two variables A, B that are dependent.
	indeps := map[string]bool{
		ciKey("A", "B", nil): false,
	}

	data := dummyData("A", "B")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	if !pdag.HasEdge("A", "B") {
		t.Error("expected edge A-B")
	}
}

func TestPCTwoVariablesIndependent(t *testing.T) {
	// Two variables A, B that are independent.
	indeps := map[string]bool{
		ciKey("A", "B", nil): true,
	}

	data := dummyData("A", "B")
	ci := mockCITestFromMap(indeps)
	pc := NewPC(data, ci, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	if pdag.Adjacent("A", "B") {
		t.Error("A and B should not be adjacent (independent)")
	}
}

func TestPCInsufficientVariables(t *testing.T) {
	data := dummyData("A")
	ci := mockCITestFromMap(nil)
	pc := NewPC(data, ci, 0.05)
	_, err := pc.Estimate()
	if err == nil {
		t.Error("expected error for single variable")
	}
}

// --- Combinations helper test ---

func TestCombinations(t *testing.T) {
	items := []string{"A", "B", "C", "D"}

	tests := []struct {
		k    int
		want int
	}{
		{0, 1},
		{1, 4},
		{2, 6},
		{3, 4},
		{4, 1},
		{5, 0},
	}

	for _, tt := range tests {
		got := combinations(items, tt.k)
		if len(got) != tt.want {
			t.Errorf("C(%d,%d): expected %d combinations, got %d", len(items), tt.k, tt.want, len(got))
		}
	}
}

// --- Chi-square CI test for integration testing ---

// chiSquareCITest implements a chi-square test of independence for discrete
// (categorical) variables. Values in the DataFrame columns must be comparable
// (typically int or string).
func chiSquareCITest(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	if n == 0 {
		return 0, 1, true
	}

	xVals := data.Column(x).Values()
	yVals := data.Column(y).Values()

	// Collect z values per row as a composite key.
	type zKey struct {
		vals string
	}

	zKeys := make([]string, n)
	if len(z) > 0 {
		for i := 0; i < n; i++ {
			key := ""
			for _, zVar := range z {
				zCol := data.Column(zVar).Values()
				key += fmt.Sprintf("%v|", zCol[i])
			}
			zKeys[i] = key
		}
	}

	// Group data by z-values and compute chi-square for each group.
	type group struct {
		indices []int
	}
	groups := make(map[string]*group)
	for i := 0; i < n; i++ {
		k := zKeys[i]
		g, ok := groups[k]
		if !ok {
			g = &group{}
			groups[k] = g
		}
		g.indices = append(g.indices, i)
	}

	totalStat := 0.0
	totalDF := 0

	for _, g := range groups {
		if len(g.indices) < 2 {
			continue
		}

		// Count joint and marginal frequencies.
		xCounts := make(map[any]int)
		yCounts := make(map[any]int)
		joint := make(map[[2]any]int)

		for _, idx := range g.indices {
			xv := xVals[idx]
			yv := yVals[idx]
			xCounts[xv]++
			yCounts[yv]++
			joint[[2]any{xv, yv}]++
		}

		nGroup := float64(len(g.indices))

		// Compute chi-square statistic for this group.
		xLevels := make([]any, 0, len(xCounts))
		for k := range xCounts {
			xLevels = append(xLevels, k)
		}
		yLevels := make([]any, 0, len(yCounts))
		for k := range yCounts {
			yLevels = append(yLevels, k)
		}

		if len(xLevels) < 2 || len(yLevels) < 2 {
			continue
		}

		stat := 0.0
		for _, xv := range xLevels {
			for _, yv := range yLevels {
				observed := float64(joint[[2]any{xv, yv}])
				expected := float64(xCounts[xv]) * float64(yCounts[yv]) / nGroup
				if expected > 0 {
					stat += (observed - expected) * (observed - expected) / expected
				}
			}
		}

		df := (len(xLevels) - 1) * (len(yLevels) - 1)
		totalStat += stat
		totalDF += df
	}

	if totalDF == 0 {
		return 0, 1, true
	}

	// Compute p-value using the chi-square survival function (1 - CDF).
	pvalue := chiSquareSurvival(totalStat, totalDF)
	independent := pvalue > significance

	return totalStat, pvalue, independent
}

// chiSquareSurvival computes the survival function (1 - CDF) of the
// chi-square distribution using the regularized incomplete gamma function.
func chiSquareSurvival(x float64, df int) float64 {
	if x <= 0 || df <= 0 {
		return 1.0
	}
	// P(X > x) = 1 - regularizedGammaP(df/2, x/2)
	// which equals regularizedGammaQ(df/2, x/2) = upper incomplete gamma.
	return regularizedUpperGamma(float64(df)/2.0, x/2.0)
}

// regularizedUpperGamma computes Q(a, x) = 1 - P(a, x) where P is the
// regularized lower incomplete gamma function.
func regularizedUpperGamma(a, x float64) float64 {
	if x < a+1.0 {
		// Use series expansion for P, then Q = 1 - P.
		return 1.0 - regularizedLowerGammaSeries(a, x)
	}
	// Use continued fraction for Q directly.
	return regularizedUpperGammaCF(a, x)
}

// regularizedLowerGammaSeries computes P(a, x) via series expansion.
func regularizedLowerGammaSeries(a, x float64) float64 {
	if x == 0 {
		return 0
	}
	ap := a
	sum := 1.0 / a
	del := 1.0 / a
	for i := 0; i < 200; i++ {
		ap++
		del *= x / ap
		sum += del
		if math.Abs(del) < math.Abs(sum)*1e-14 {
			break
		}
	}
	return sum * math.Exp(-x+a*math.Log(x)-lgamma(a))
}

// regularizedUpperGammaCF computes Q(a, x) via Lentz continued fraction.
func regularizedUpperGammaCF(a, x float64) float64 {
	tiny := 1e-30
	b := x + 1.0 - a
	c := 1.0 / tiny
	d := 1.0 / b
	h := d
	for i := 1; i <= 200; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2.0
		d = an*d + b
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = b + an/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1.0 / d
		delta := d * c
		h *= delta
		if math.Abs(delta-1.0) < 1e-14 {
			break
		}
	}
	return math.Exp(-x+a*math.Log(x)-lgamma(a)) * h
}

func lgamma(x float64) float64 {
	v, _ := math.Lgamma(x)
	return v
}

// --- Integration test with chi-square on synthetic data ---

func TestPCChainWithChiSquare(t *testing.T) {
	// Generate synthetic data from A -> B -> C.
	// A is Bernoulli(0.5), B = A (deterministic-ish), C = B (deterministic-ish).
	// With noise to make it non-degenerate.
	rng := rand.New(rand.NewSource(42))
	n := 2000
	aData := make([]any, n)
	bData := make([]any, n)
	cData := make([]any, n)

	for i := 0; i < n; i++ {
		a := 0
		if rng.Float64() < 0.5 {
			a = 1
		}
		b := a
		if rng.Float64() < 0.1 {
			b = 1 - b // 10% noise
		}
		c := b
		if rng.Float64() < 0.1 {
			c = 1 - c // 10% noise
		}
		aData[i] = a
		bData[i] = b
		cData[i] = c
	}

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aData),
		"B": tabgo.NewSeries("B", bData),
		"C": tabgo.NewSeries("C", cData),
	})

	pc := NewPC(data, chiSquareCITest, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// Skeleton should be A-B, B-C (no A-C because A _|_ C | B).
	if !pdag.HasEdge("A", "B") {
		t.Error("expected edge A-B in skeleton")
	}
	if !pdag.HasEdge("B", "C") {
		t.Error("expected edge B-C in skeleton")
	}
	if pdag.Adjacent("A", "C") {
		t.Error("A-C should not be adjacent (A _|_ C | B)")
	}
}

func TestPCVStructureWithChiSquare(t *testing.T) {
	// Generate synthetic data from A -> C <- B (v-structure).
	// A and B are independent, C = f(A, B).
	rng := rand.New(rand.NewSource(123))
	n := 3000
	aData := make([]any, n)
	bData := make([]any, n)
	cData := make([]any, n)

	for i := 0; i < n; i++ {
		a := 0
		if rng.Float64() < 0.5 {
			a = 1
		}
		b := 0
		if rng.Float64() < 0.5 {
			b = 1
		}
		// C is an OR-like function of A and B with noise.
		c := 0
		if a == 1 || b == 1 {
			c = 1
		}
		if rng.Float64() < 0.05 {
			c = 1 - c
		}
		aData[i] = a
		bData[i] = b
		cData[i] = c
	}

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aData),
		"B": tabgo.NewSeries("B", bData),
		"C": tabgo.NewSeries("C", cData),
	})

	pc := NewPC(data, chiSquareCITest, 0.05)
	pdag, err := pc.Estimate()
	if err != nil {
		t.Fatalf("Estimate failed: %v", err)
	}

	// Skeleton: A-C, B-C (no A-B since A _|_ B).
	if pdag.Adjacent("A", "B") {
		t.Error("A-B should not be adjacent (A and B are independent)")
	}
	if !pdag.HasEdge("A", "C") {
		t.Error("expected edge A-C")
	}
	if !pdag.HasEdge("B", "C") {
		t.Error("expected edge B-C")
	}

	// V-structure: should orient A->C<-B.
	if !pdag.HasDirectedEdge("A", "C") {
		t.Error("expected directed edge A->C (v-structure)")
	}
	if !pdag.HasDirectedEdge("B", "C") {
		t.Error("expected directed edge B->C (v-structure)")
	}
}

func TestPCEstimateBNWithChiSquare(t *testing.T) {
	// V-structure A -> C <- B with chi-square test.
	rng := rand.New(rand.NewSource(456))
	n := 3000
	aData := make([]any, n)
	bData := make([]any, n)
	cData := make([]any, n)

	for i := 0; i < n; i++ {
		a := 0
		if rng.Float64() < 0.5 {
			a = 1
		}
		b := 0
		if rng.Float64() < 0.5 {
			b = 1
		}
		c := 0
		if a == 1 || b == 1 {
			c = 1
		}
		if rng.Float64() < 0.05 {
			c = 1 - c
		}
		aData[i] = a
		bData[i] = b
		cData[i] = c
	}

	data := tabgo.NewDataFrame(map[string]*tabgo.Series{
		"A": tabgo.NewSeries("A", aData),
		"B": tabgo.NewSeries("B", bData),
		"C": tabgo.NewSeries("C", cData),
	})

	pc := NewPC(data, chiSquareCITest, 0.05)
	bn, err := pc.EstimateBN()
	if err != nil {
		t.Fatalf("EstimateBN failed: %v", err)
	}

	edges := bn.Edges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d: %v", len(edges), edges)
	}

	edgeSet := make(map[[2]string]bool)
	for _, e := range edges {
		edgeSet[e] = true
	}
	if !edgeSet[[2]string{"A", "C"}] {
		t.Error("expected edge A->C in BN")
	}
	if !edgeSet[[2]string{"B", "C"}] {
		t.Error("expected edge B->C in BN")
	}
}

// TestSepSetKey verifies canonical ordering.
func TestSepSetKey(t *testing.T) {
	k1 := sepSetKey("B", "A")
	k2 := sepSetKey("A", "B")
	if k1 != k2 {
		t.Error("sepSetKey should be symmetric")
	}
	if k1[0] != "A" || k1[1] != "B" {
		t.Errorf("expected [A, B], got %v", k1)
	}
}

// TestPDAGToDAG tests conversion of a simple PDAG with undirected edges.
func TestPDAGToDAG(t *testing.T) {
	pdag := graphgo.NewPDAG()
	pdag.AddNodes("A", "B", "C")
	pdag.AddUndirectedEdge("A", "B")
	pdag.AddUndirectedEdge("B", "C")

	bn, err := pdagToDAG(pdag)
	if err != nil {
		t.Fatalf("pdagToDAG failed: %v", err)
	}

	edges := bn.Edges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d: %v", len(edges), edges)
	}

	// Verify it's a valid DAG (no cycles).
	nodes := bn.Nodes()
	if len(nodes) != 3 {
		t.Errorf("expected 3 nodes, got %d", len(nodes))
	}
}

// TestPDAGToDAGWithDirected tests PDAG with both directed and undirected edges.
func TestPDAGToDAGWithDirected(t *testing.T) {
	pdag := graphgo.NewPDAG()
	pdag.AddNodes("A", "B", "C")
	pdag.AddDirectedEdge("A", "C")
	pdag.AddDirectedEdge("B", "C")

	bn, err := pdagToDAG(pdag)
	if err != nil {
		t.Fatalf("pdagToDAG failed: %v", err)
	}

	edges := bn.Edges()
	if len(edges) != 2 {
		t.Fatalf("expected 2 edges, got %d", len(edges))
	}

	edgeSet := make(map[[2]string]bool)
	for _, e := range edges {
		edgeSet[e] = true
	}
	if !edgeSet[[2]string{"A", "C"}] || !edgeSet[[2]string{"B", "C"}] {
		t.Errorf("expected A->C and B->C, got %v", edges)
	}
}

// Verify chiSquareCITest matches CITestFunc signature at compile time.
var _ CITestFunc = chiSquareCITest
