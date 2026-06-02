//go:build unit

package numgo

import (
	"math"
	"testing"
)

func TestNewRNG(t *testing.T) {
	rng := NewRNG(42)
	if rng == nil || rng.src == nil {
		t.Fatal("NewRNG returned nil")
	}
}

// ---------------------------------------------------------------------------
// Rand
// ---------------------------------------------------------------------------

func TestRandShape(t *testing.T) {
	rng := NewRNG(1)
	a := rng.Rand(3, 4)
	if a.Shape()[0] != 3 || a.Shape()[1] != 4 {
		t.Fatalf("Rand: unexpected shape %v", a.Shape())
	}
	if a.Size() != 12 {
		t.Fatalf("Rand: expected size 12, got %d", a.Size())
	}
}

func TestRandRange(t *testing.T) {
	rng := NewRNG(2)
	a := rng.Rand(10000)
	for _, v := range a.Data() {
		if v < 0 || v >= 1 {
			t.Fatalf("Rand: value %f outside [0,1)", v)
		}
	}
}

func TestRandMean(t *testing.T) {
	rng := NewRNG(3)
	a := rng.Rand(100000)
	mean := computeMean(a.Data())
	if math.Abs(mean-0.5) > 0.01 {
		t.Fatalf("Rand: expected mean ~0.5, got %f", mean)
	}
}

// ---------------------------------------------------------------------------
// Randn
// ---------------------------------------------------------------------------

func TestRandnShape(t *testing.T) {
	rng := NewRNG(10)
	a := rng.Randn(5, 3)
	if a.Shape()[0] != 5 || a.Shape()[1] != 3 {
		t.Fatalf("Randn: unexpected shape %v", a.Shape())
	}
}

func TestRandnStatistics(t *testing.T) {
	rng := NewRNG(11)
	n := 200000
	a := rng.Randn(n)
	data := a.Data()
	mean := computeMean(data)
	stddev := computeStd(data, mean)
	if math.Abs(mean) > 0.02 {
		t.Fatalf("Randn: expected mean ~0, got %f", mean)
	}
	if math.Abs(stddev-1.0) > 0.02 {
		t.Fatalf("Randn: expected std ~1, got %f", stddev)
	}
}

func TestRandnOddSize(t *testing.T) {
	rng := NewRNG(12)
	a := rng.Randn(7)
	if a.Size() != 7 {
		t.Fatalf("Randn: expected size 7, got %d", a.Size())
	}
}

// ---------------------------------------------------------------------------
// RandInt
// ---------------------------------------------------------------------------

func TestRandIntRange(t *testing.T) {
	rng := NewRNG(20)
	a := rng.RandInt(5, 15, 10000)
	for _, v := range a.Data() {
		iv := int(v)
		if iv < 5 || iv >= 15 {
			t.Fatalf("RandInt: value %d outside [5,15)", iv)
		}
	}
}

func TestRandIntShape(t *testing.T) {
	rng := NewRNG(21)
	a := rng.RandInt(0, 10, 2, 3)
	if a.Shape()[0] != 2 || a.Shape()[1] != 3 {
		t.Fatalf("RandInt: unexpected shape %v", a.Shape())
	}
}

func TestRandIntPanicsInvalidRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for low >= high")
		}
	}()
	rng := NewRNG(22)
	rng.RandInt(10, 5, 10)
}

// ---------------------------------------------------------------------------
// Choice
// ---------------------------------------------------------------------------

func TestChoiceWithReplacement(t *testing.T) {
	rng := NewRNG(30)
	result := rng.Choice(5, 20, true)
	if len(result) != 20 {
		t.Fatalf("Choice: expected 20 elements, got %d", len(result))
	}
	for _, v := range result {
		if v < 0 || v >= 5 {
			t.Fatalf("Choice: value %d out of range [0,5)", v)
		}
	}
}

func TestChoiceWithoutReplacement(t *testing.T) {
	rng := NewRNG(31)
	result := rng.Choice(10, 10, false)
	if len(result) != 10 {
		t.Fatalf("Choice: expected 10 elements, got %d", len(result))
	}
	seen := make(map[int]bool)
	for _, v := range result {
		if v < 0 || v >= 10 {
			t.Fatalf("Choice: value %d out of range", v)
		}
		if seen[v] {
			t.Fatalf("Choice without replacement: duplicate value %d", v)
		}
		seen[v] = true
	}
}

func TestChoiceWithoutReplacementPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for size > n without replacement")
		}
	}()
	rng := NewRNG(32)
	rng.Choice(3, 5, false)
}

func TestChoiceEmptySize(t *testing.T) {
	rng := NewRNG(33)
	result := rng.Choice(10, 0, false)
	if len(result) != 0 {
		t.Fatalf("Choice: expected empty, got %d", len(result))
	}
}

// ---------------------------------------------------------------------------
// Shuffle
// ---------------------------------------------------------------------------

func TestShuffle1D(t *testing.T) {
	rng := NewRNG(40)
	a := FromSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
	rng.Shuffle(a)
	// Check all elements are still present.
	sum := 0.0
	for _, v := range a.Data() {
		sum += v
	}
	if sum != 55 {
		t.Fatalf("Shuffle: elements changed, sum=%f expected 55", sum)
	}
	// Very unlikely to remain sorted after shuffle with 10 elements.
	sorted := true
	data := a.Data()
	for i := 1; i < len(data); i++ {
		if data[i] < data[i-1] {
			sorted = false
			break
		}
	}
	if sorted {
		t.Fatal("Shuffle: array appears unchanged (very unlikely)")
	}
}

func TestShuffle2D(t *testing.T) {
	rng := NewRNG(41)
	a := NewNDArray([]int{4, 3}, []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	rng.Shuffle(a)
	// Sum should be preserved.
	sum := 0.0
	for _, v := range a.Data() {
		sum += v
	}
	if sum != 78 {
		t.Fatalf("Shuffle2D: elements changed, sum=%f expected 78", sum)
	}
	// Each row should remain intact (row sums: 6, 15, 24, 33).
	rowSums := make(map[float64]bool)
	for i := 0; i < 4; i++ {
		rs := a.Get(i, 0) + a.Get(i, 1) + a.Get(i, 2)
		rowSums[rs] = true
	}
	for _, expected := range []float64{6, 15, 24, 33} {
		if !rowSums[expected] {
			t.Fatalf("Shuffle2D: missing row with sum %f", expected)
		}
	}
}

// ---------------------------------------------------------------------------
// Normal
// ---------------------------------------------------------------------------

func TestNormalStatistics(t *testing.T) {
	rng := NewRNG(50)
	n := 200000
	mean := 5.0
	std := 2.0
	a := rng.Normal(mean, std, n)
	data := a.Data()
	gotMean := computeMean(data)
	gotStd := computeStd(data, gotMean)
	if math.Abs(gotMean-mean) > 0.05 {
		t.Fatalf("Normal: expected mean ~%f, got %f", mean, gotMean)
	}
	if math.Abs(gotStd-std) > 0.05 {
		t.Fatalf("Normal: expected std ~%f, got %f", std, gotStd)
	}
}

func TestNormalShape(t *testing.T) {
	rng := NewRNG(51)
	a := rng.Normal(0, 1, 3, 4)
	if a.Shape()[0] != 3 || a.Shape()[1] != 4 {
		t.Fatalf("Normal: unexpected shape %v", a.Shape())
	}
}

// ---------------------------------------------------------------------------
// Uniform
// ---------------------------------------------------------------------------

func TestUniformRange(t *testing.T) {
	rng := NewRNG(60)
	a := rng.Uniform(-3.0, 7.0, 50000)
	for _, v := range a.Data() {
		if v < -3.0 || v >= 7.0 {
			t.Fatalf("Uniform: value %f outside [-3,7)", v)
		}
	}
}

func TestUniformStatistics(t *testing.T) {
	rng := NewRNG(61)
	low, high := 2.0, 8.0
	n := 200000
	a := rng.Uniform(low, high, n)
	data := a.Data()
	mean := computeMean(data)
	expectedMean := (low + high) / 2.0
	if math.Abs(mean-expectedMean) > 0.05 {
		t.Fatalf("Uniform: expected mean ~%f, got %f", expectedMean, mean)
	}
}

func TestUniformPanicsInvalidRange(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for low >= high")
		}
	}()
	rng := NewRNG(62)
	rng.Uniform(5.0, 3.0, 10)
}

// ---------------------------------------------------------------------------
// Dirichlet
// ---------------------------------------------------------------------------

func TestDirichletSumsToOne(t *testing.T) {
	rng := NewRNG(70)
	alpha := []float64{1.0, 2.0, 3.0}
	sample := rng.Dirichlet(alpha)
	if len(sample) != 3 {
		t.Fatalf("Dirichlet: expected length 3, got %d", len(sample))
	}
	sum := 0.0
	for _, v := range sample {
		sum += v
		if v < 0 {
			t.Fatalf("Dirichlet: negative value %f", v)
		}
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Fatalf("Dirichlet: sum=%f, expected 1.0", sum)
	}
}

func TestDirichletMean(t *testing.T) {
	rng := NewRNG(71)
	alpha := []float64{2.0, 3.0, 5.0}
	alphaSum := 10.0
	n := 100000
	means := make([]float64, 3)
	for i := 0; i < n; i++ {
		s := rng.Dirichlet(alpha)
		for j := range s {
			means[j] += s[j]
		}
	}
	for j := range means {
		means[j] /= float64(n)
		expected := alpha[j] / alphaSum
		if math.Abs(means[j]-expected) > 0.01 {
			t.Fatalf("Dirichlet: mean[%d]=%f, expected ~%f", j, means[j], expected)
		}
	}
}

func TestDirichletSmallAlpha(t *testing.T) {
	// Test that alpha < 1 works (exercises the boost path).
	rng := NewRNG(72)
	alpha := []float64{0.1, 0.5, 0.3}
	sample := rng.Dirichlet(alpha)
	sum := 0.0
	for _, v := range sample {
		sum += v
		if v < 0 {
			t.Fatalf("Dirichlet small alpha: negative value %f", v)
		}
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Fatalf("Dirichlet small alpha: sum=%f, expected 1.0", sum)
	}
}

func TestDirichletPanicsEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for empty alpha")
		}
	}()
	rng := NewRNG(73)
	rng.Dirichlet([]float64{})
}

// ---------------------------------------------------------------------------
// Multinomial
// ---------------------------------------------------------------------------

func TestMultinomialSumsToN(t *testing.T) {
	rng := NewRNG(80)
	pvals := []float64{0.2, 0.3, 0.5}
	n := 1000
	counts := rng.Multinomial(n, pvals)
	if len(counts) != 3 {
		t.Fatalf("Multinomial: expected 3 bins, got %d", len(counts))
	}
	total := 0
	for _, c := range counts {
		if c < 0 {
			t.Fatalf("Multinomial: negative count %d", c)
		}
		total += c
	}
	if total != n {
		t.Fatalf("Multinomial: counts sum to %d, expected %d", total, n)
	}
}

func TestMultinomialProportions(t *testing.T) {
	rng := NewRNG(81)
	pvals := []float64{0.1, 0.3, 0.6}
	trials := 500000
	counts := rng.Multinomial(trials, pvals)
	for i, p := range pvals {
		got := float64(counts[i]) / float64(trials)
		if math.Abs(got-p) > 0.01 {
			t.Fatalf("Multinomial: bin %d proportion=%f, expected ~%f", i, got, p)
		}
	}
}

func TestMultinomialZeroTrials(t *testing.T) {
	rng := NewRNG(82)
	counts := rng.Multinomial(0, []float64{0.5, 0.5})
	for _, c := range counts {
		if c != 0 {
			t.Fatalf("Multinomial(0): expected all zeros, got %d", c)
		}
	}
}

func TestMultinomialPanicsEmpty(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic for empty pvals")
		}
	}()
	rng := NewRNG(83)
	rng.Multinomial(10, []float64{})
}

// ---------------------------------------------------------------------------
// Seeded reproducibility
// ---------------------------------------------------------------------------

func TestSeededReproducibility(t *testing.T) {
	rng1 := NewRNG(999)
	rng2 := NewRNG(999)
	a1 := rng1.Rand(100)
	a2 := rng2.Rand(100)
	for i := range a1.Data() {
		if a1.Data()[i] != a2.Data()[i] {
			t.Fatalf("Reproducibility: mismatch at index %d", i)
		}
	}
}

func TestSeededReproducibilityRandn(t *testing.T) {
	rng1 := NewRNG(1000)
	rng2 := NewRNG(1000)
	a1 := rng1.Randn(100)
	a2 := rng2.Randn(100)
	for i := range a1.Data() {
		if a1.Data()[i] != a2.Data()[i] {
			t.Fatalf("Reproducibility Randn: mismatch at index %d", i)
		}
	}
}

func TestSeededReproducibilityNormal(t *testing.T) {
	rng1 := NewRNG(1001)
	rng2 := NewRNG(1001)
	a1 := rng1.Normal(5, 2, 50)
	a2 := rng2.Normal(5, 2, 50)
	for i := range a1.Data() {
		if a1.Data()[i] != a2.Data()[i] {
			t.Fatalf("Reproducibility Normal: mismatch at index %d", i)
		}
	}
}

func TestDifferentSeedsProduceDifferentResults(t *testing.T) {
	rng1 := NewRNG(1)
	rng2 := NewRNG(2)
	a1 := rng1.Rand(100)
	a2 := rng2.Rand(100)
	allSame := true
	for i := range a1.Data() {
		if a1.Data()[i] != a2.Data()[i] {
			allSame = false
			break
		}
	}
	if allSame {
		t.Fatal("Different seeds produced identical output")
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func computeMean(data []float64) float64 {
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func computeStd(data []float64, mean float64) float64 {
	sumSq := 0.0
	for _, v := range data {
		d := v - mean
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(len(data)))
}
