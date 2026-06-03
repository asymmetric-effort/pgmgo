//go:build unit

package factors

import (
	"math"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/numgo"
)

// ---------------------------------------------------------------------------
// NewLinearGaussianCPD — construction and validation
// ---------------------------------------------------------------------------

func TestNewLinearGaussianCPD_NoParents(t *testing.T) {
	cpd, err := NewLinearGaussianCPD("X", 5.0, nil, 1.0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if cpd.Variable() != "X" {
		t.Errorf("Variable() = %q, want X", cpd.Variable())
	}
	if cpd.Mean() != 5.0 {
		t.Errorf("Mean() = %f, want 5.0", cpd.Mean())
	}
	if cpd.Variance() != 1.0 {
		t.Errorf("Variance() = %f, want 1.0", cpd.Variance())
	}
	if len(cpd.Betas()) != 0 {
		t.Errorf("Betas() should be empty, got %v", cpd.Betas())
	}
	if len(cpd.Evidence()) != 0 {
		t.Errorf("Evidence() should be empty, got %v", cpd.Evidence())
	}
}

func TestNewLinearGaussianCPD_WithParents(t *testing.T) {
	cpd, err := NewLinearGaussianCPD("Y", 2.0,
		[]float64{0.5, -1.3},
		4.0,
		[]string{"A", "B"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if cpd.Variable() != "Y" {
		t.Errorf("Variable() = %q, want Y", cpd.Variable())
	}
	betas := cpd.Betas()
	if len(betas) != 2 || betas[0] != 0.5 || betas[1] != -1.3 {
		t.Errorf("Betas() = %v, want [0.5, -1.3]", betas)
	}
	ev := cpd.Evidence()
	if len(ev) != 2 || ev[0] != "A" || ev[1] != "B" {
		t.Errorf("Evidence() = %v, want [A, B]", ev)
	}
}

func TestNewLinearGaussianCPD_BetasEvidenceMismatch(t *testing.T) {
	_, err := NewLinearGaussianCPD("X", 0, []float64{1.0, 2.0}, 1.0, []string{"A"})
	if err == nil {
		t.Fatal("expected error for betas/evidence length mismatch")
	}
}

func TestNewLinearGaussianCPD_ZeroVariance(t *testing.T) {
	_, err := NewLinearGaussianCPD("X", 0, nil, 0, nil)
	if err == nil {
		t.Fatal("expected error for zero variance")
	}
}

func TestNewLinearGaussianCPD_NegativeVariance(t *testing.T) {
	_, err := NewLinearGaussianCPD("X", 0, nil, -1.0, nil)
	if err == nil {
		t.Fatal("expected error for negative variance")
	}
}

// ---------------------------------------------------------------------------
// ConditionalMean
// ---------------------------------------------------------------------------

func TestConditionalMean_NoParents(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 3.0, nil, 1.0, nil)
	mu := cpd.ConditionalMean(nil)
	if mu != 3.0 {
		t.Errorf("ConditionalMean() = %f, want 3.0", mu)
	}
}

func TestConditionalMean_WithParents(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 2.0,
		[]float64{0.5, -1.0, 3.0},
		1.0,
		[]string{"A", "B", "C"},
	)
	parents := map[string]float64{"A": 4.0, "B": 2.0, "C": 1.0}
	// expected: 2.0 + 0.5*4.0 + (-1.0)*2.0 + 3.0*1.0 = 2 + 2 - 2 + 3 = 5
	mu := cpd.ConditionalMean(parents)
	if math.Abs(mu-5.0) > 1e-12 {
		t.Errorf("ConditionalMean() = %f, want 5.0", mu)
	}
}

func TestConditionalMean_NegativeBetas(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 0,
		[]float64{-2.0},
		1.0,
		[]string{"P"},
	)
	parents := map[string]float64{"P": 3.0}
	// expected: 0 + (-2)*3 = -6
	mu := cpd.ConditionalMean(parents)
	if math.Abs(mu-(-6.0)) > 1e-12 {
		t.Errorf("ConditionalMean() = %f, want -6.0", mu)
	}
}

// ---------------------------------------------------------------------------
// ConditionalVariance
// ---------------------------------------------------------------------------

func TestConditionalVariance(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 0, []float64{1.0}, 9.0, []string{"A"})
	if cpd.ConditionalVariance() != 9.0 {
		t.Errorf("ConditionalVariance() = %f, want 9.0", cpd.ConditionalVariance())
	}
}

// ---------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------

func TestPDF_StandardNormal(t *testing.T) {
	// N(0, 1) at x=0 should be 1/sqrt(2*pi) ~ 0.3989
	cpd, _ := NewLinearGaussianCPD("X", 0, nil, 1.0, nil)
	p := cpd.PDF(0, nil)
	expected := 1.0 / math.Sqrt(2*math.Pi)
	if math.Abs(p-expected) > 1e-10 {
		t.Errorf("PDF(0) = %f, want %f", p, expected)
	}
}

func TestPDF_AtMean(t *testing.T) {
	// N(5, 4) at x=5 should be 1/sqrt(2*pi*4)
	cpd, _ := NewLinearGaussianCPD("X", 5.0, nil, 4.0, nil)
	p := cpd.PDF(5.0, nil)
	expected := 1.0 / math.Sqrt(2*math.Pi*4.0)
	if math.Abs(p-expected) > 1e-10 {
		t.Errorf("PDF(5) = %f, want %f", p, expected)
	}
}

func TestPDF_OffMean(t *testing.T) {
	// N(0, 1) at x=1: exp(-0.5)/sqrt(2*pi)
	cpd, _ := NewLinearGaussianCPD("X", 0, nil, 1.0, nil)
	p := cpd.PDF(1.0, nil)
	expected := math.Exp(-0.5) / math.Sqrt(2*math.Pi)
	if math.Abs(p-expected) > 1e-10 {
		t.Errorf("PDF(1) = %f, want %f", p, expected)
	}
}

func TestPDF_WithParents(t *testing.T) {
	// X | A=2 ~ N(1 + 0.5*2, 1) = N(2, 1)
	// PDF at x=2 should be 1/sqrt(2*pi)
	cpd, _ := NewLinearGaussianCPD("X", 1.0, []float64{0.5}, 1.0, []string{"A"})
	parents := map[string]float64{"A": 2.0}
	p := cpd.PDF(2.0, parents)
	expected := 1.0 / math.Sqrt(2*math.Pi)
	if math.Abs(p-expected) > 1e-10 {
		t.Errorf("PDF(2) = %f, want %f", p, expected)
	}
}

func TestPDF_Symmetry(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 3.0, nil, 2.0, nil)
	// f(3+d) should equal f(3-d) for any d
	p1 := cpd.PDF(3.0+1.5, nil)
	p2 := cpd.PDF(3.0-1.5, nil)
	if math.Abs(p1-p2) > 1e-12 {
		t.Errorf("PDF not symmetric: PDF(4.5) = %f, PDF(1.5) = %f", p1, p2)
	}
}

// ---------------------------------------------------------------------------
// LogPDF
// ---------------------------------------------------------------------------

func TestLogPDF_ConsistencyWithPDF(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 2.0,
		[]float64{1.0, -0.5},
		3.0,
		[]string{"A", "B"},
	)
	parents := map[string]float64{"A": 1.0, "B": 4.0}
	for _, x := range []float64{-2.0, 0, 1.5, 3.0, 10.0} {
		logp := cpd.LogPDF(x, parents)
		p := cpd.PDF(x, parents)
		if math.Abs(logp-math.Log(p)) > 1e-10 {
			t.Errorf("LogPDF(%f) = %f, but log(PDF(%f)) = %f", x, logp, x, math.Log(p))
		}
	}
}

func TestLogPDF_StandardNormalAtZero(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 0, nil, 1.0, nil)
	logp := cpd.LogPDF(0, nil)
	expected := -0.5 * math.Log(2*math.Pi)
	if math.Abs(logp-expected) > 1e-10 {
		t.Errorf("LogPDF(0) = %f, want %f", logp, expected)
	}
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

func TestSample_MeanConvergence(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 5.0,
		[]float64{2.0},
		1.0,
		[]string{"A"},
	)
	parents := map[string]float64{"A": 3.0}
	expectedMean := 5.0 + 2.0*3.0 // = 11.0

	rng := numgo.NewRNG(42)
	n := 10000
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += cpd.Sample(parents, rng)
	}
	sampleMean := sum / float64(n)
	// With n=10000 and variance=1, SE = 1/100 = 0.01, so 0.1 is ~10 SE.
	if math.Abs(sampleMean-expectedMean) > 0.1 {
		t.Errorf("sample mean = %f, want ~%f", sampleMean, expectedMean)
	}
}

func TestSample_VarianceConvergence(t *testing.T) {
	variance := 4.0
	cpd, _ := NewLinearGaussianCPD("X", 0, nil, variance, nil)

	rng := numgo.NewRNG(123)
	n := 10000
	sum := 0.0
	sumSq := 0.0
	for i := 0; i < n; i++ {
		x := cpd.Sample(nil, rng)
		sum += x
		sumSq += x * x
	}
	sampleMean := sum / float64(n)
	sampleVar := sumSq/float64(n) - sampleMean*sampleMean
	// Tolerance: with n=10000 and var=4, SE of variance estimator ~ var*sqrt(2/n) ~ 0.089
	if math.Abs(sampleVar-variance) > 0.5 {
		t.Errorf("sample variance = %f, want ~%f", sampleVar, variance)
	}
}

func TestSample_NoParents(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 100.0, nil, 0.01, nil)
	rng := numgo.NewRNG(7)
	// With tiny variance, samples should be very close to the mean.
	for i := 0; i < 100; i++ {
		x := cpd.Sample(nil, rng)
		if math.Abs(x-100.0) > 1.0 {
			t.Errorf("sample %f is too far from mean 100.0", x)
		}
	}
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

func TestValidate_Valid(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 1.0, []float64{0.5}, 2.0, []string{"A"})
	if err := cpd.Validate(); err != nil {
		t.Errorf("Validate() = %v, want nil", err)
	}
}

func TestValidate_InvalidVariance(t *testing.T) {
	cpd := &LinearGaussianCPD{
		variable: "X",
		mean:     0,
		betas:    nil,
		variance: -1.0,
		evidence: nil,
	}
	if err := cpd.Validate(); err == nil {
		t.Error("Validate() should return error for negative variance")
	}
}

func TestValidate_BetasEvidenceMismatch(t *testing.T) {
	cpd := &LinearGaussianCPD{
		variable: "X",
		mean:     0,
		betas:    []float64{1.0, 2.0},
		variance: 1.0,
		evidence: []string{"A"},
	}
	if err := cpd.Validate(); err == nil {
		t.Error("Validate() should return error for betas/evidence mismatch")
	}
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

func TestCopy_Independence(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 2.0, []float64{0.5, -1.0}, 3.0, []string{"A", "B"})
	cpy := cpd.Copy()

	// Verify values match.
	if cpy.Variable() != cpd.Variable() {
		t.Errorf("copy Variable() = %q, want %q", cpy.Variable(), cpd.Variable())
	}
	if cpy.Mean() != cpd.Mean() {
		t.Errorf("copy Mean() = %f, want %f", cpy.Mean(), cpd.Mean())
	}
	if cpy.Variance() != cpd.Variance() {
		t.Errorf("copy Variance() = %f, want %f", cpy.Variance(), cpd.Variance())
	}

	// Mutate copy and ensure original is unaffected.
	cpy.betas[0] = 999.0
	cpy.evidence[0] = "Z"
	if cpd.betas[0] != 0.5 {
		t.Error("modifying copy's betas affected original")
	}
	if cpd.evidence[0] != "A" {
		t.Error("modifying copy's evidence affected original")
	}
}

// ---------------------------------------------------------------------------
// String
// ---------------------------------------------------------------------------

func TestString_NoParents(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 5.0, nil, 1.0, nil)
	s := cpd.String()
	if s == "" {
		t.Error("String() should not be empty")
	}
	// Should contain the variable name and variance.
	if !containsSubstring(s, "X") || !containsSubstring(s, "5.0000") || !containsSubstring(s, "1.0000") {
		t.Errorf("String() = %q, expected to contain variable, mean, variance", s)
	}
}

func TestString_WithParents(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("Y", 1.0, []float64{0.5}, 2.0, []string{"A"})
	s := cpd.String()
	if !containsSubstring(s, "betas") || !containsSubstring(s, "evidence") {
		t.Errorf("String() = %q, expected betas and evidence info", s)
	}
}

// containsSubstring checks if s contains substr.
func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || indexSubstring(s, substr) >= 0)
}

func indexSubstring(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// PDF integration — area under curve should be ~1
// ---------------------------------------------------------------------------

func TestPDF_IntegrationApprox(t *testing.T) {
	cpd, _ := NewLinearGaussianCPD("X", 3.0, []float64{1.0}, 2.0, []string{"A"})
	parents := map[string]float64{"A": 2.0}
	// Conditional mean = 3 + 1*2 = 5, variance = 2, std ~ 1.414
	// Integrate from mean-10*std to mean+10*std with small step.
	mu := cpd.ConditionalMean(parents)
	std := math.Sqrt(cpd.Variance())
	lo := mu - 10*std
	hi := mu + 10*std
	steps := 10000
	dx := (hi - lo) / float64(steps)
	area := 0.0
	for i := 0; i < steps; i++ {
		x := lo + (float64(i)+0.5)*dx
		area += cpd.PDF(x, parents) * dx
	}
	if math.Abs(area-1.0) > 1e-6 {
		t.Errorf("numerical integral of PDF = %f, want ~1.0", area)
	}
}

// ---------------------------------------------------------------------------
// GetRandomLinearGaussianCPD
// ---------------------------------------------------------------------------

func TestGetRandomLinearGaussianCPD_NoEvidence(t *testing.T) {
	cpd, err := GetRandomLinearGaussianCPD("X", nil, 42)
	if err != nil {
		t.Fatalf("GetRandomLinearGaussianCPD: %v", err)
	}
	if cpd.Variable() != "X" {
		t.Errorf("expected variable X, got %q", cpd.Variable())
	}
	if len(cpd.Betas()) != 0 {
		t.Errorf("expected 0 betas, got %d", len(cpd.Betas()))
	}
	if cpd.Variance() <= 0 {
		t.Errorf("expected positive variance, got %f", cpd.Variance())
	}
	if err := cpd.Validate(); err != nil {
		t.Errorf("validation failed: %v", err)
	}
}

func TestGetRandomLinearGaussianCPD_WithEvidence(t *testing.T) {
	cpd, err := GetRandomLinearGaussianCPD("Y", []string{"X1", "X2"}, 99)
	if err != nil {
		t.Fatalf("GetRandomLinearGaussianCPD: %v", err)
	}
	if len(cpd.Betas()) != 2 {
		t.Errorf("expected 2 betas, got %d", len(cpd.Betas()))
	}
	ev := cpd.Evidence()
	if len(ev) != 2 || ev[0] != "X1" || ev[1] != "X2" {
		t.Errorf("expected evidence [X1, X2], got %v", ev)
	}
}

func TestGetRandomLinearGaussianCPD_Deterministic(t *testing.T) {
	cpd1, _ := GetRandomLinearGaussianCPD("Z", []string{"A"}, 123)
	cpd2, _ := GetRandomLinearGaussianCPD("Z", []string{"A"}, 123)
	if cpd1.Mean() != cpd2.Mean() {
		t.Errorf("means differ: %f vs %f", cpd1.Mean(), cpd2.Mean())
	}
	if cpd1.Variance() != cpd2.Variance() {
		t.Errorf("variances differ: %f vs %f", cpd1.Variance(), cpd2.Variance())
	}
	b1 := cpd1.Betas()
	b2 := cpd2.Betas()
	for i := range b1 {
		if b1[i] != b2[i] {
			t.Errorf("beta[%d] differs: %f vs %f", i, b1[i], b2[i])
		}
	}
}

func TestGetRandomLinearGaussianCPD_EmptyVariable(t *testing.T) {
	_, err := GetRandomLinearGaussianCPD("", nil, 1)
	if err == nil {
		t.Error("expected error for empty variable name")
	}
}
