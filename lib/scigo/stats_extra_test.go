//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Chi2Contingency
// ---------------------------------------------------------------------------

func TestChi2Contingency_Independent(t *testing.T) {
	// Perfectly proportional table: no association.
	observed := [][]float64{
		{10, 20},
		{10, 20},
	}
	stat, pval, dof, expected := Chi2Contingency(observed)
	if dof != 1 {
		t.Errorf("Chi2Contingency: dof=%v, want 1", dof)
	}
	if !approxEqual(stat, 0, 1e-10) {
		t.Errorf("Chi2Contingency: statistic=%v, want 0", stat)
	}
	if !approxEqual(pval, 1, 1e-6) {
		t.Errorf("Chi2Contingency: pvalue=%v, want 1", pval)
	}
	// Expected should equal observed for proportional table.
	for i := range expected {
		for j := range expected[i] {
			if !approxEqual(expected[i][j], observed[i][j], 1e-10) {
				t.Errorf("Chi2Contingency: expected[%d][%d]=%v, want %v", i, j, expected[i][j], observed[i][j])
			}
		}
	}
}

func TestChi2Contingency_Known(t *testing.T) {
	// Classic example: gender vs. handedness.
	observed := [][]float64{
		{43, 9},
		{44, 4},
	}
	stat, pval, dof, _ := Chi2Contingency(observed)
	if dof != 1 {
		t.Errorf("dof=%v, want 1", dof)
	}
	if stat <= 0 {
		t.Errorf("statistic=%v, should be > 0", stat)
	}
	// With this small table the association is weak, pval should be moderate.
	if pval < 0 || pval > 1 {
		t.Errorf("pvalue=%v, out of range", pval)
	}
}

func TestChi2Contingency_3x3(t *testing.T) {
	observed := [][]float64{
		{10, 20, 30},
		{6, 9, 17},
		{14, 21, 28},
	}
	_, _, dof, _ := Chi2Contingency(observed)
	if dof != 4 {
		t.Errorf("3x3 dof=%v, want 4", dof)
	}
}

func TestChi2Contingency_PanicRows(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on fewer than 2 rows")
		}
	}()
	Chi2Contingency([][]float64{{1, 2}})
}

func TestChi2Contingency_PanicCols(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on fewer than 2 columns")
		}
	}()
	Chi2Contingency([][]float64{{1}, {2}})
}

// ---------------------------------------------------------------------------
// FisherExact
// ---------------------------------------------------------------------------

func TestFisherExact_NoAssociation(t *testing.T) {
	// Perfectly proportional 2x2 table.
	table := [2][2]int{{5, 5}, {5, 5}}
	or, pval := FisherExact(table)
	if !approxEqual(or, 1, 1e-10) {
		t.Errorf("FisherExact no assoc: oddsRatio=%v, want 1", or)
	}
	if !approxEqual(pval, 1, 1e-2) {
		t.Errorf("FisherExact no assoc: pvalue=%v, want ~1", pval)
	}
}

func TestFisherExact_StrongAssociation(t *testing.T) {
	// Strong association.
	table := [2][2]int{{10, 0}, {0, 10}}
	or, pval := FisherExact(table)
	if !math.IsInf(or, 1) {
		t.Errorf("FisherExact strong: oddsRatio=%v, want +Inf", or)
	}
	if pval > 0.001 {
		t.Errorf("FisherExact strong: pvalue=%v, want < 0.001", pval)
	}
}

func TestFisherExact_Known(t *testing.T) {
	// Lady tasting tea: classic example.
	// Table: [[3,1],[1,3]]
	table := [2][2]int{{3, 1}, {1, 3}}
	or, pval := FisherExact(table)
	if or <= 0 {
		t.Errorf("FisherExact known: oddsRatio=%v, should be > 0", or)
	}
	// p-value for this table is about 0.4857 (two-sided).
	if pval < 0.2 || pval > 0.8 {
		t.Errorf("FisherExact known: pvalue=%v, expected moderate", pval)
	}
}

// ---------------------------------------------------------------------------
// PointBiserialR
// ---------------------------------------------------------------------------

func TestPointBiserialR_Perfect(t *testing.T) {
	// Perfect positive association.
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	y := []bool{false, false, false, false, true, true, true, true}
	r, pval := PointBiserialR(x, y)
	if r <= 0 {
		t.Errorf("PointBiserialR perfect: r=%v, want > 0", r)
	}
	if pval > 0.05 {
		t.Errorf("PointBiserialR perfect: pval=%v, want < 0.05", pval)
	}
}

func TestPointBiserialR_NoAssociation(t *testing.T) {
	// Random-ish, weak association.
	x := []float64{1, 2, 3, 4, 5, 6}
	y := []bool{true, false, true, false, true, false}
	r, _ := PointBiserialR(x, y)
	// Should be close to 0 for alternating pattern with linear x.
	if math.Abs(r) > 0.5 {
		t.Errorf("PointBiserialR no assoc: r=%v, expected close to 0", r)
	}
}

func TestPointBiserialR_PanicLength(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on mismatched lengths")
		}
	}()
	PointBiserialR([]float64{1, 2}, []bool{true})
}

func TestPointBiserialR_PanicTooFew(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on fewer than 3 elements")
		}
	}()
	PointBiserialR([]float64{1, 2}, []bool{true, false})
}
