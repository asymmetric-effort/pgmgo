//go:build unit

package structure_score

import (
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

func TestBDs_NewBDs(t *testing.T) {
	bds := NewBDs(1.0, 0.5)
	if bds == nil {
		t.Fatal("NewBDs returned nil")
	}
	if bds.equivalentSampleSize != 1.0 {
		t.Errorf("ESS: got %f, want 1.0", bds.equivalentSampleSize)
	}
	if bds.structurePrior != 0.5 {
		t.Errorf("structurePrior: got %f, want 0.5", bds.structurePrior)
	}
}

func TestBDs_LocalScore_NoParents(t *testing.T) {
	data := makeSmallData()
	bds := NewBDs(1.0, 0.5)

	score := bds.LocalScore("A", nil, data)

	// With no parents, structurePrior penalty is 0.
	// Should be a valid negative score.
	if score >= 0 {
		t.Errorf("BDs LocalScore(A, nil) should be negative, got %f", score)
	}
}

func TestBDs_LocalScore_WithParents(t *testing.T) {
	data := makeSmallData()
	bds := NewBDs(1.0, 0.5)

	score := bds.LocalScore("B", []string{"A"}, data)

	// Should be a valid negative score.
	if score >= 0 {
		t.Errorf("BDs LocalScore(B, [A]) should be negative, got %f", score)
	}
}

func TestBDs_StructurePriorPenalizesParents(t *testing.T) {
	data := makeTestData()

	// High structure prior should heavily penalize parents.
	bdsHigh := NewBDs(1.0, 10.0)
	scoreNoParent := bdsHigh.LocalScore("Z", nil, data)
	scoreWithParent := bdsHigh.LocalScore("Z", []string{"X"}, data)

	if scoreWithParent > scoreNoParent {
		t.Errorf("BDs with high structure prior should penalize parents: no_parent=%f, with_parent=%f",
			scoreNoParent, scoreWithParent)
	}
}

func TestBDs_ZeroStructurePrior(t *testing.T) {
	data := makeSmallData()

	// With zero structure prior and same ESS, BDs should differ from BDeu
	// because of the quadratic scaling of pseudo-counts.
	bds := NewBDs(1.0, 0.0)
	bdeu := NewBDeu(1.0)

	scoreBDs := bds.LocalScore("B", []string{"A"}, data)
	scoreBDeu := bdeu.LocalScore("B", []string{"A"}, data)

	// They should not be equal (different pseudo-count formulas).
	if approxEqual(scoreBDs, scoreBDeu, 1e-10) {
		t.Errorf("BDs and BDeu should differ (different pseudo-count scaling): BDs=%f, BDeu=%f",
			scoreBDs, scoreBDeu)
	}
}

func TestBDs_NoParentsMatchesBDeu(t *testing.T) {
	data := makeSmallData()

	// With no parents, numParentCfgs=1, so alpha_jk = ESS/(1^2 * card) = ESS/card
	// which is the same as BDeu: ESS/(1 * card). So they should match
	// when structurePrior=0.
	bds := NewBDs(1.0, 0.0)
	bdeu := NewBDeu(1.0)

	scoreBDs := bds.LocalScore("A", nil, data)
	scoreBDeu := bdeu.LocalScore("A", nil, data)

	if !approxEqual(scoreBDs, scoreBDeu, 1e-10) {
		t.Errorf("BDs and BDeu should match with no parents: BDs=%f, BDeu=%f",
			scoreBDs, scoreBDeu)
	}
}

func TestBDs_FavorsSparserStructures(t *testing.T) {
	// BDs with structure prior should prefer simpler structures
	// more aggressively than BDeu.
	data := makeTestData()

	bds := NewBDs(1.0, 1.0)

	// Z is independent of X. BDs should penalize adding X as parent more
	// than BDeu does (due to structure prior + quadratic pseudo-count scaling).
	noParent := bds.LocalScore("Z", nil, data)
	withParent := bds.LocalScore("Z", []string{"X"}, data)

	if withParent >= noParent {
		t.Errorf("BDs should penalize unnecessary parent: no_parent=%f, with_parent=%f",
			noParent, withParent)
	}
}

func TestBDs_Score_SumOfLocalScores(t *testing.T) {
	data := makeTestData()
	bds := NewBDs(1.0, 0.5)

	variables := []string{"X", "Y", "Z"}
	parentMap := map[string][]string{
		"X": {},
		"Y": {"X"},
		"Z": {},
	}

	totalScore := bds.Score(variables, parentMap, data)
	sumLocal := 0.0
	for _, v := range variables {
		sumLocal += bds.LocalScore(v, parentMap[v], data)
	}

	if !approxEqual(totalScore, sumLocal, 1e-10) {
		t.Errorf("BDs Score != sum of LocalScore: %f != %f", totalScore, sumLocal)
	}
}

func TestBDs_EmptyData(t *testing.T) {
	data := tabgo.NewDataFrameFromRows([]string{"A", "B"}, nil)
	bds := NewBDs(1.0, 0.5)
	score := bds.LocalScore("A", nil, data)
	// With empty data, all parent configs have 0 count, score should be 0
	// or at least finite.
	if score > 0 {
		t.Errorf("BDs LocalScore on empty data should not be positive, got %f", score)
	}
}

func TestBDs_InterfaceCompliance(t *testing.T) {
	var _ StructureScore = NewBDs(1.0, 0.5)
}

func TestBDs_DifferentStructurePriors(t *testing.T) {
	data := makeSmallData()

	bds0 := NewBDs(1.0, 0.0)
	bds1 := NewBDs(1.0, 1.0)
	bds5 := NewBDs(1.0, 5.0)

	score0 := bds0.LocalScore("B", []string{"A"}, data)
	score1 := bds1.LocalScore("B", []string{"A"}, data)
	score5 := bds5.LocalScore("B", []string{"A"}, data)

	// Higher structure prior => lower score (more penalty for having parents).
	if score1 >= score0 {
		t.Errorf("BDs score should decrease with structure prior: sp=0 got %f, sp=1 got %f", score0, score1)
	}
	if score5 >= score1 {
		t.Errorf("BDs score should decrease with structure prior: sp=1 got %f, sp=5 got %f", score1, score5)
	}
}
