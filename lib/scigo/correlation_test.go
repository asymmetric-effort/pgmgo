//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Pearson Correlation Tests
// ---------------------------------------------------------------------------

func TestPearsonCorrelationPerfectPositive(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{2, 4, 6, 8, 10}
	r, pval := PearsonCorrelation(x, y)
	if !approxEqual(r, 1.0, 1e-12) {
		t.Errorf("Perfect positive correlation: r = %v, want 1.0", r)
	}
	if pval > 1e-10 {
		t.Errorf("Perfect positive correlation: p-value = %v, want ~0", pval)
	}
}

func TestPearsonCorrelationPerfectNegative(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{10, 8, 6, 4, 2}
	r, pval := PearsonCorrelation(x, y)
	if !approxEqual(r, -1.0, 1e-12) {
		t.Errorf("Perfect negative correlation: r = %v, want -1.0", r)
	}
	if pval > 1e-10 {
		t.Errorf("Perfect negative correlation: p-value = %v, want ~0", pval)
	}
}

func TestPearsonCorrelationUncorrelated(t *testing.T) {
	// Construct data that is exactly uncorrelated
	x := []float64{1, -1, 1, -1}
	y := []float64{1, 1, -1, -1}
	r, pval := PearsonCorrelation(x, y)
	if !approxEqual(r, 0, 1e-12) {
		t.Errorf("Uncorrelated data: r = %v, want 0", r)
	}
	if !approxEqual(pval, 1, 1e-6) {
		t.Errorf("Uncorrelated data: p-value = %v, want 1.0", pval)
	}
}

func TestPearsonCorrelationKnownValue(t *testing.T) {
	// Known example: x = [1,2,3,4,5], y = [1,3,2,5,4]
	// Manual computation:
	// mean(x)=3, mean(y)=3
	// sxy = (-2)(-2)+(-1)(0)+(0)(-1)+(1)(2)+(2)(1) = 4+0+0+2+2 = 8
	// sxx = 4+1+0+1+4 = 10, syy = 4+0+1+4+1 = 10
	// r = 8/sqrt(100) = 0.8
	x := []float64{1, 2, 3, 4, 5}
	y := []float64{1, 3, 2, 5, 4}
	r, pval := PearsonCorrelation(x, y)
	if !approxEqual(r, 0.8, 1e-10) {
		t.Errorf("Known correlation: r = %v, want 0.8", r)
	}
	// p-value for r=0.8, n=5, df=3: t = 0.8*sqrt(3/(1-0.64)) = 0.8*sqrt(3/0.36) = 0.8*2.887 = 2.309
	// Should be a moderate p-value
	if pval < 0.05 || pval > 0.5 {
		t.Errorf("Known correlation: p-value = %v, expected between 0.05 and 0.5", pval)
	}
}

func TestPearsonCorrelationConstantInput(t *testing.T) {
	x := []float64{5, 5, 5, 5}
	y := []float64{1, 2, 3, 4}
	r, pval := PearsonCorrelation(x, y)
	if r != 0 {
		t.Errorf("Constant x: r = %v, want 0", r)
	}
	if pval != 1 {
		t.Errorf("Constant x: p-value = %v, want 1", pval)
	}
}

func TestPearsonCorrelationPanicDifferentLengths(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on different lengths")
		}
	}()
	PearsonCorrelation([]float64{1, 2, 3}, []float64{1, 2})
}

func TestPearsonCorrelationPanicTooFew(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on fewer than 3 data points")
		}
	}()
	PearsonCorrelation([]float64{1, 2}, []float64{3, 4})
}

// ---------------------------------------------------------------------------
// Partial Correlation Tests
// ---------------------------------------------------------------------------

func TestPartialCorrelationNoConditioning(t *testing.T) {
	// With no conditioning set, partial correlation = Pearson correlation
	data := [][]float64{
		{1, 2, 10},
		{2, 4, 20},
		{3, 6, 30},
		{4, 8, 40},
		{5, 10, 50},
	}
	r, pval := PartialCorrelation(data, 0, 1, nil)
	rDirect, pDirect := PearsonCorrelation(
		[]float64{1, 2, 3, 4, 5},
		[]float64{2, 4, 6, 8, 10},
	)
	if !approxEqual(r, rDirect, 1e-10) {
		t.Errorf("No conditioning: r = %v, want %v", r, rDirect)
	}
	if !approxEqual(pval, pDirect, 1e-8) {
		t.Errorf("No conditioning: p-value = %v, want %v", pval, pDirect)
	}
}

func TestPartialCorrelationThreeVariables(t *testing.T) {
	// X = 2*Z + small noise, Y = 3*Z + small noise, Z = [1..10]
	// Marginal r(X,Y) should be very high since both are driven by Z.
	// After controlling for Z, partial r should drop substantially.
	x := []float64{2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1}
	y := []float64{3.1, 5.9, 9.1, 11.9, 15.1, 17.9, 21.1, 23.9, 27.1, 29.9}
	z := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

	data := make([][]float64, 10)
	for i := range data {
		data[i] = []float64{x[i], y[i], z[i]}
	}

	// X and Y should be very highly correlated marginally
	rMarginal, _ := PearsonCorrelation(x, y)
	if rMarginal < 0.99 {
		t.Errorf("X and Y should be highly correlated marginally, got r = %v", rMarginal)
	}

	// After controlling for Z, partial correlation should drop significantly
	rPartial, _ := PartialCorrelation(data, 0, 1, []int{2})
	if math.Abs(rPartial) > 0.9 {
		t.Errorf("Partial correlation controlling for Z should drop: r = %v", rPartial)
	}
}

func TestPartialCorrelationKnownResult(t *testing.T) {
	// Manual 3-variable test with known result.
	// Use the recursive formula directly:
	// r(0,1|2) = (r01 - r02*r12) / sqrt((1-r02^2)*(1-r12^2))
	//
	// Data chosen so computations are tractable:
	data := [][]float64{
		{1, 2, 1},
		{2, 1, 2},
		{3, 4, 2},
		{4, 3, 3},
		{5, 5, 4},
	}

	col := func(j int) []float64 {
		c := make([]float64, len(data))
		for i := range data {
			c[i] = data[i][j]
		}
		return c
	}

	r01, _ := PearsonCorrelation(col(0), col(1))
	r02, _ := PearsonCorrelation(col(0), col(2))
	r12, _ := PearsonCorrelation(col(1), col(2))

	expected := (r01 - r02*r12) / math.Sqrt((1-r02*r02)*(1-r12*r12))

	r, _ := PartialCorrelation(data, 0, 1, []int{2})
	if !approxEqual(r, expected, 1e-10) {
		t.Errorf("Partial correlation: got %v, want %v", r, expected)
	}
}

func TestPartialCorrelationPanicFewObs(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic with fewer than 3 observations")
		}
	}()
	data := [][]float64{{1, 2}, {3, 4}}
	PartialCorrelation(data, 0, 1, nil)
}

// ---------------------------------------------------------------------------
// Fisher Z Transform Tests
// ---------------------------------------------------------------------------

func TestFisherZTransformZero(t *testing.T) {
	// Z(0) = 0
	z := FisherZTransform(0, 100)
	if !approxEqual(z, 0, 1e-15) {
		t.Errorf("FisherZ(0) = %v, want 0", z)
	}
}

func TestFisherZTransformSymmetry(t *testing.T) {
	// Z(-r) = -Z(r)
	z1 := FisherZTransform(0.5, 100)
	z2 := FisherZTransform(-0.5, 100)
	if !approxEqual(z1, -z2, 1e-15) {
		t.Errorf("FisherZ should be antisymmetric: Z(0.5) = %v, Z(-0.5) = %v", z1, z2)
	}
}

func TestFisherZTransformKnownValues(t *testing.T) {
	tests := []struct {
		r, wantZ float64
	}{
		{0.5, 0.5 * math.Log(3)},        // 0.5*ln(1.5/0.5) = 0.5*ln(3)
		{0.9, 0.5 * math.Log(19)},       // 0.5*ln(1.9/0.1) = 0.5*ln(19)
		{-0.3, 0.5 * math.Log(0.7/1.3)}, // 0.5*ln((1-0.3)/(1+0.3))
	}
	for _, tc := range tests {
		got := FisherZTransform(tc.r, 50)
		if !approxEqual(got, tc.wantZ, 1e-12) {
			t.Errorf("FisherZ(%v) = %v, want %v", tc.r, got, tc.wantZ)
		}
	}
}

func TestFisherZTransformMonotonic(t *testing.T) {
	// Z should be monotonically increasing
	prev := FisherZTransform(-0.99, 100)
	for r := -0.9; r <= 0.99; r += 0.1 {
		cur := FisherZTransform(r, 100)
		if cur <= prev {
			t.Errorf("FisherZ not monotonically increasing at r=%v: prev=%v, cur=%v", r, prev, cur)
		}
		prev = cur
	}
}

func TestFisherZTransformBoundary(t *testing.T) {
	// Z(1) -> +Inf, Z(-1) -> -Inf
	z1 := FisherZTransform(1, 100)
	if !math.IsInf(z1, 1) {
		t.Errorf("FisherZ(1) = %v, want +Inf", z1)
	}
	z2 := FisherZTransform(-1, 100)
	if !math.IsInf(z2, -1) {
		t.Errorf("FisherZ(-1) = %v, want -Inf", z2)
	}
}

func TestFisherZTransformSmallR(t *testing.T) {
	// For small r, Z(r) ~ r (first-order approximation)
	r := 0.01
	z := FisherZTransform(r, 100)
	if !approxEqual(z, r, 1e-4) {
		t.Errorf("FisherZ(%v) = %v, want approximately %v for small r", r, z, r)
	}
}

// ---------------------------------------------------------------------------
// Power Divergence Test
// ---------------------------------------------------------------------------

func TestPowerDivergenceChiSquare(t *testing.T) {
	// lambda=1 should give Pearson chi-squared when sum(observed) == sum(expected)
	// The power divergence formula with lambda=1 gives sum(O^2/E) - sum(O),
	// and Pearson gives sum(O^2/E) - 2*sum(O) + sum(E). These are equal when sum(O)=sum(E).
	observed := []float64{16, 18, 16, 14, 12, 12}
	total := 0.0
	for _, o := range observed {
		total += o
	}
	k := float64(len(observed))
	expected := make([]float64, len(observed))
	for i := range expected {
		expected[i] = total / k
	}

	statPD, pvalPD := PowerDivergenceTest(observed, expected, 1.0)
	statCS, pvalCS := ChiSquareTest(observed, expected)

	if !approxEqual(statPD, statCS, 1e-8) {
		t.Errorf("PowerDivergence(lambda=1) statistic = %v, ChiSquare = %v", statPD, statCS)
	}
	if !approxEqual(pvalPD, pvalCS, 1e-8) {
		t.Errorf("PowerDivergence(lambda=1) p-value = %v, ChiSquare = %v", pvalPD, pvalCS)
	}
}

func TestPowerDivergenceGTest(t *testing.T) {
	// lambda=0 should give G-test
	observed := []float64{20, 30}
	expected := []float64{25, 25}

	statPD, pvalPD := PowerDivergenceTest(observed, expected, 0.0)
	statG, pvalG := GTest(observed, expected)

	if !approxEqual(statPD, statG, 1e-10) {
		t.Errorf("PowerDivergence(lambda=0) statistic = %v, GTest = %v", statPD, statG)
	}
	if !approxEqual(pvalPD, pvalG, 1e-10) {
		t.Errorf("PowerDivergence(lambda=0) p-value = %v, GTest = %v", pvalPD, pvalG)
	}
}

func TestPowerDivergenceModifiedLLR(t *testing.T) {
	// lambda=-1: modified log-likelihood ratio
	// statistic = 2 * sum(expected * ln(expected/observed))
	observed := []float64{20, 30}
	expected := []float64{25, 25}

	stat, _ := PowerDivergenceTest(observed, expected, -1.0)
	wantStat := 2 * (25*math.Log(25.0/20.0) + 25*math.Log(25.0/30.0))

	if !approxEqual(stat, wantStat, 1e-10) {
		t.Errorf("PowerDivergence(lambda=-1) statistic = %v, want %v", stat, wantStat)
	}
}

func TestPowerDivergencePerfectFit(t *testing.T) {
	// Perfect fit should give statistic = 0
	observed := []float64{10, 10, 10}
	expected := []float64{10, 10, 10}

	for _, lambda := range []float64{-1, 0, 0.5, 1, 2.0 / 3.0} {
		stat, pval := PowerDivergenceTest(observed, expected, lambda)
		if !approxEqual(stat, 0, 1e-10) {
			t.Errorf("Perfect fit (lambda=%v): statistic = %v, want 0", lambda, stat)
		}
		if !approxEqual(pval, 1, 1e-6) {
			t.Errorf("Perfect fit (lambda=%v): p-value = %v, want 1", lambda, pval)
		}
	}
}

func TestPowerDivergenceCressieRead(t *testing.T) {
	// lambda=2/3: Cressie-Read statistic
	// General formula: (2/(lambda*(lambda+1))) * sum(O * ((O/E)^lambda - 1))
	observed := []float64{15, 25, 10}
	expected := []float64{20, 15, 15}
	lambda := 2.0 / 3.0

	stat, pval := PowerDivergenceTest(observed, expected, lambda)

	// Verify manually
	var wantStat float64
	for i := range observed {
		ratio := observed[i] / expected[i]
		wantStat += observed[i] * (math.Pow(ratio, lambda) - 1)
	}
	wantStat *= 2.0 / (lambda * (lambda + 1))

	if !approxEqual(stat, wantStat, 1e-10) {
		t.Errorf("Cressie-Read statistic = %v, want %v", stat, wantStat)
	}
	if pval <= 0 || pval >= 1 {
		t.Errorf("Cressie-Read p-value = %v, expected in (0, 1)", pval)
	}
}

func TestPowerDivergencePanicDifferentLengths(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on different lengths")
		}
	}()
	PowerDivergenceTest([]float64{1, 2}, []float64{1}, 1.0)
}

func TestPowerDivergencePanicTooFew(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("Should panic on fewer than 2 categories")
		}
	}()
	PowerDivergenceTest([]float64{1}, []float64{1}, 1.0)
}

// ---------------------------------------------------------------------------
// TDistribution Tests (from distributions.go addition)
// ---------------------------------------------------------------------------

func TestTDistributionPanicZeroDF(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("NewTDistribution(0) should panic")
		}
	}()
	NewTDistribution(0)
}

func TestTDistributionMeanVar(t *testing.T) {
	td := NewTDistribution(5)
	if td.Mean() != 0 {
		t.Errorf("t(5).Mean() = %v, want 0", td.Mean())
	}
	if !approxEqual(td.Var(), 5.0/3.0, 1e-12) {
		t.Errorf("t(5).Var() = %v, want %v", td.Var(), 5.0/3.0)
	}

	td1 := NewTDistribution(1) // Cauchy: mean undefined
	if !math.IsNaN(td1.Mean()) {
		t.Errorf("t(1).Mean() = %v, want NaN", td1.Mean())
	}
	if !math.IsNaN(td1.Var()) {
		t.Errorf("t(1).Var() = %v, want NaN", td1.Var())
	}

	td15 := NewTDistribution(1.5)
	if td15.Mean() != 0 {
		t.Errorf("t(1.5).Mean() = %v, want 0", td15.Mean())
	}
	if !math.IsInf(td15.Var(), 1) {
		t.Errorf("t(1.5).Var() = %v, want +Inf", td15.Var())
	}
}

func TestTDistributionCDFSymmetry(t *testing.T) {
	td := NewTDistribution(10)
	// CDF(0) = 0.5
	if !approxEqual(td.CDF(0), 0.5, 1e-10) {
		t.Errorf("t(10).CDF(0) = %v, want 0.5", td.CDF(0))
	}
	// CDF(-x) + CDF(x) = 1
	for _, x := range []float64{0.5, 1, 2, 3} {
		sum := td.CDF(x) + td.CDF(-x)
		if !approxEqual(sum, 1.0, 1e-8) {
			t.Errorf("t(10).CDF(%v) + CDF(%v) = %v, want 1.0", x, -x, sum)
		}
	}
}

func TestTDistributionCDFKnownValues(t *testing.T) {
	// For large df, t-distribution approaches standard normal
	td := NewTDistribution(1000)
	n := NewNormal(0, 1)
	for _, x := range []float64{-2, -1, 0, 1, 2} {
		got := td.CDF(x)
		want := n.CDF(x)
		if !approxEqual(got, want, 1e-3) {
			t.Errorf("t(1000).CDF(%v) = %v, want ~%v (normal approx)", x, got, want)
		}
	}
}

func TestTDistributionPDFSymmetry(t *testing.T) {
	td := NewTDistribution(5)
	for _, x := range []float64{0.5, 1, 2, 3} {
		if !approxEqual(td.PDF(x), td.PDF(-x), 1e-12) {
			t.Errorf("t(5).PDF(%v) != PDF(%v): %v vs %v", x, -x, td.PDF(x), td.PDF(-x))
		}
	}
}

func TestTDistributionPDFLogPDFConsistency(t *testing.T) {
	td := NewTDistribution(5)
	for _, x := range []float64{-2, -1, 0, 1, 2} {
		got := td.LogPDF(x)
		want := math.Log(td.PDF(x))
		if !approxEqual(got, want, 1e-10) {
			t.Errorf("t(5).LogPDF(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestTDistributionSurvivalFunction(t *testing.T) {
	td := NewTDistribution(10)
	for _, x := range []float64{0.5, 1, 2, 3} {
		got := td.SurvivalFunction(x)
		want := 1 - td.CDF(x)
		if !approxEqual(got, want, 1e-12) {
			t.Errorf("t(10).SF(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestTDistributionPPFRoundTrip(t *testing.T) {
	td := NewTDistribution(10)
	for _, p := range []float64{0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99} {
		x := td.PPF(p)
		got := td.CDF(x)
		if !approxEqual(got, p, 1e-6) {
			t.Errorf("t(10).CDF(PPF(%v)) = %v, want %v", p, got, p)
		}
	}
}
