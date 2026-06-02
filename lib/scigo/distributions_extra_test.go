//go:build unit

package scigo

import (
	"math"
	"testing"
)

// ---------------------------------------------------------------------------
// Beta Distribution Tests
// ---------------------------------------------------------------------------

func TestBetaPanic(t *testing.T) {
	tests := []struct {
		name        string
		alpha, beta float64
	}{
		{"zero alpha", 0, 1},
		{"negative alpha", -1, 1},
		{"zero beta", 1, 0},
		{"negative beta", 1, -1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			NewBeta(tc.alpha, tc.beta)
		})
	}
}

func TestBetaMeanVar(t *testing.T) {
	tests := []struct {
		alpha, beta, wantMean, wantVar float64
	}{
		{1, 1, 0.5, 1.0 / 12.0},
		{2, 2, 0.5, 1.0 / 20.0},
		{2, 5, 2.0 / 7.0, 10.0 / (49.0 * 8.0)},
		{0.5, 0.5, 0.5, 0.125},
	}
	for _, tc := range tests {
		b := NewBeta(tc.alpha, tc.beta)
		if !approxEqual(b.Mean(), tc.wantMean, 1e-12) {
			t.Errorf("Beta(%v,%v).Mean() = %v, want %v", tc.alpha, tc.beta, b.Mean(), tc.wantMean)
		}
		if !approxEqual(b.Var(), tc.wantVar, 1e-12) {
			t.Errorf("Beta(%v,%v).Var() = %v, want %v", tc.alpha, tc.beta, b.Var(), tc.wantVar)
		}
	}
}

func TestBetaPDF(t *testing.T) {
	// Beta(1,1) is Uniform(0,1), PDF = 1 everywhere on [0,1]
	b := NewBeta(1, 1)
	for _, x := range []float64{0.0, 0.25, 0.5, 0.75, 1.0} {
		got := b.PDF(x)
		if !approxEqual(got, 1.0, 1e-10) {
			t.Errorf("Beta(1,1).PDF(%v) = %v, want 1.0", x, got)
		}
	}

	// Beta(2,2) at x=0.5: PDF = 6*0.5*0.5 = 1.5
	b2 := NewBeta(2, 2)
	got := b2.PDF(0.5)
	if !approxEqual(got, 1.5, 1e-10) {
		t.Errorf("Beta(2,2).PDF(0.5) = %v, want 1.5", got)
	}

	// Outside [0,1] should be 0
	if b2.PDF(-0.1) != 0 {
		t.Error("Beta PDF outside [0,1] should be 0")
	}
	if b2.PDF(1.1) != 0 {
		t.Error("Beta PDF outside [0,1] should be 0")
	}

	// Beta(2,5) at x=0.3: known value
	// PDF = x^(a-1)*(1-x)^(b-1) / B(a,b)
	// B(2,5) = Gamma(2)*Gamma(5)/Gamma(7) = 1*24/720 = 1/30
	// PDF(0.3) = 0.3^1 * 0.7^4 / (1/30) = 0.3 * 0.2401 * 30 = 2.1609
	b3 := NewBeta(2, 5)
	got = b3.PDF(0.3)
	want := 0.3 * math.Pow(0.7, 4) * 30
	if !approxEqual(got, want, 1e-10) {
		t.Errorf("Beta(2,5).PDF(0.3) = %v, want %v", got, want)
	}
}

func TestBetaCDF(t *testing.T) {
	// Beta(1,1) CDF should equal x on [0,1]
	b := NewBeta(1, 1)
	for _, x := range []float64{0.0, 0.25, 0.5, 0.75, 1.0} {
		got := b.CDF(x)
		if !approxEqual(got, x, 1e-10) {
			t.Errorf("Beta(1,1).CDF(%v) = %v, want %v", x, got, x)
		}
	}

	// Boundaries
	b2 := NewBeta(2, 3)
	if b2.CDF(-0.1) != 0 {
		t.Error("Beta CDF below 0 should be 0")
	}
	if b2.CDF(1.1) != 1 {
		t.Error("Beta CDF above 1 should be 1")
	}
}

func TestBetaPPF(t *testing.T) {
	b := NewBeta(2, 5)

	// Round-trip: PPF(CDF(x)) = x
	for _, x := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		p := b.CDF(x)
		got := b.PPF(p)
		if !approxEqual(got, x, 1e-6) {
			t.Errorf("Beta(2,5).PPF(CDF(%v)) = %v, want %v", x, got, x)
		}
	}

	// Boundaries
	if b.PPF(0) != 0 {
		t.Errorf("Beta PPF(0) = %v, want 0", b.PPF(0))
	}
	if b.PPF(1) != 1 {
		t.Errorf("Beta PPF(1) = %v, want 1", b.PPF(1))
	}
}

// ---------------------------------------------------------------------------
// Gamma Distribution Tests
// ---------------------------------------------------------------------------

func TestGammaPanic(t *testing.T) {
	tests := []struct {
		name         string
		shape, scale float64
	}{
		{"zero shape", 0, 1},
		{"negative shape", -1, 1},
		{"zero scale", 1, 0},
		{"negative scale", 1, -1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			NewGamma(tc.shape, tc.scale)
		})
	}
}

func TestGammaMeanVar(t *testing.T) {
	tests := []struct {
		shape, scale, wantMean, wantVar float64
	}{
		{1, 1, 1, 1},
		{2, 3, 6, 18},
		{5, 2, 10, 20},
		{0.5, 4, 2, 8},
	}
	for _, tc := range tests {
		g := NewGamma(tc.shape, tc.scale)
		if !approxEqual(g.Mean(), tc.wantMean, 1e-12) {
			t.Errorf("Gamma(%v,%v).Mean() = %v, want %v", tc.shape, tc.scale, g.Mean(), tc.wantMean)
		}
		if !approxEqual(g.Var(), tc.wantVar, 1e-12) {
			t.Errorf("Gamma(%v,%v).Var() = %v, want %v", tc.shape, tc.scale, g.Var(), tc.wantVar)
		}
	}
}

func TestGammaPDF(t *testing.T) {
	// Gamma(1,1) = Exponential(1): PDF(x) = exp(-x)
	g := NewGamma(1, 1)
	for _, x := range []float64{0.5, 1, 2, 5} {
		got := g.PDF(x)
		want := math.Exp(-x)
		if !approxEqual(got, want, 1e-10) {
			t.Errorf("Gamma(1,1).PDF(%v) = %v, want %v", x, got, want)
		}
	}

	// PDF for negative x should be 0
	if g.PDF(-1) != 0 {
		t.Error("Gamma PDF for negative x should be 0")
	}

	// Gamma(2,1) PDF at x=1: x*exp(-x)/Gamma(2) = 1*exp(-1)/1 = exp(-1)
	g2 := NewGamma(2, 1)
	got := g2.PDF(1)
	want := math.Exp(-1)
	if !approxEqual(got, want, 1e-10) {
		t.Errorf("Gamma(2,1).PDF(1) = %v, want %v", got, want)
	}
}

func TestGammaCDF(t *testing.T) {
	// Gamma(1,1) CDF(x) = 1 - exp(-x)
	g := NewGamma(1, 1)
	for _, x := range []float64{0.5, 1, 2, 5} {
		got := g.CDF(x)
		want := 1 - math.Exp(-x)
		if !approxEqual(got, want, 1e-8) {
			t.Errorf("Gamma(1,1).CDF(%v) = %v, want %v", x, got, want)
		}
	}

	if g.CDF(0) != 0 {
		t.Error("Gamma CDF(0) should be 0")
	}
	if g.CDF(-1) != 0 {
		t.Error("Gamma CDF for negative x should be 0")
	}
}

func TestGammaLogPDF(t *testing.T) {
	g := NewGamma(3, 2)
	for _, x := range []float64{0.5, 1, 2, 5, 10} {
		got := g.LogPDF(x)
		want := math.Log(g.PDF(x))
		if !approxEqual(got, want, 1e-10) {
			t.Errorf("Gamma(3,2).LogPDF(%v) = %v, want %v", x, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// Exponential Distribution Tests
// ---------------------------------------------------------------------------

func TestExponentialPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for rate=0")
		}
	}()
	NewExponential(0)
}

func TestExponentialMeanVar(t *testing.T) {
	tests := []struct {
		rate, wantMean, wantVar float64
	}{
		{1, 1, 1},
		{2, 0.5, 0.25},
		{0.5, 2, 4},
	}
	for _, tc := range tests {
		e := NewExponential(tc.rate)
		if !approxEqual(e.Mean(), tc.wantMean, 1e-12) {
			t.Errorf("Exponential(%v).Mean() = %v, want %v", tc.rate, e.Mean(), tc.wantMean)
		}
		if !approxEqual(e.Var(), tc.wantVar, 1e-12) {
			t.Errorf("Exponential(%v).Var() = %v, want %v", tc.rate, e.Var(), tc.wantVar)
		}
	}
}

func TestExponentialPDF(t *testing.T) {
	e := NewExponential(2)
	// PDF(0) = rate = 2
	if !approxEqual(e.PDF(0), 2, 1e-12) {
		t.Errorf("Exponential(2).PDF(0) = %v, want 2", e.PDF(0))
	}
	// PDF(1) = 2*exp(-2)
	got := e.PDF(1)
	want := 2 * math.Exp(-2)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Exponential(2).PDF(1) = %v, want %v", got, want)
	}
	// PDF for negative x should be 0
	if e.PDF(-1) != 0 {
		t.Error("Exponential PDF for negative x should be 0")
	}
}

func TestExponentialCDF(t *testing.T) {
	e := NewExponential(2)
	// CDF(0) = 0
	if e.CDF(0) != 0 {
		t.Errorf("Exponential(2).CDF(0) = %v, want 0", e.CDF(0))
	}
	// CDF(1) = 1 - exp(-2)
	got := e.CDF(1)
	want := 1 - math.Exp(-2)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Exponential(2).CDF(1) = %v, want %v", got, want)
	}
}

func TestExponentialPPF(t *testing.T) {
	e := NewExponential(2)

	// Round-trip
	for _, x := range []float64{0.1, 0.5, 1, 2, 5} {
		p := e.CDF(x)
		got := e.PPF(p)
		if !approxEqual(got, x, 1e-10) {
			t.Errorf("Exponential(2).PPF(CDF(%v)) = %v, want %v", x, got, x)
		}
	}

	// Boundaries
	if e.PPF(0) != 0 {
		t.Errorf("Exponential PPF(0) = %v, want 0", e.PPF(0))
	}
	if !math.IsInf(e.PPF(1), 1) {
		t.Error("Exponential PPF(1) should be +Inf")
	}

	// Known value: PPF(0.5) = ln(2)/rate
	got := e.PPF(0.5)
	want := math.Ln2 / 2
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Exponential(2).PPF(0.5) = %v, want %v", got, want)
	}
}

func TestExponentialLogPDF(t *testing.T) {
	e := NewExponential(3)
	for _, x := range []float64{0.1, 0.5, 1, 2} {
		got := e.LogPDF(x)
		want := math.Log(e.PDF(x))
		if !approxEqual(got, want, 1e-12) {
			t.Errorf("Exponential(3).LogPDF(%v) = %v, want %v", x, got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// Uniform Distribution Tests
// ---------------------------------------------------------------------------

func TestUniformPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for low >= high")
		}
	}()
	NewUniform(5, 5)
}

func TestUniformMeanVar(t *testing.T) {
	u := NewUniform(2, 8)
	if !approxEqual(u.Mean(), 5, 1e-12) {
		t.Errorf("Uniform(2,8).Mean() = %v, want 5", u.Mean())
	}
	// Var = (8-2)^2 / 12 = 36/12 = 3
	if !approxEqual(u.Var(), 3, 1e-12) {
		t.Errorf("Uniform(2,8).Var() = %v, want 3", u.Var())
	}
}

func TestUniformPDF(t *testing.T) {
	u := NewUniform(0, 10)
	// PDF = 0.1 on [0,10]
	for _, x := range []float64{0, 2, 5, 8, 10} {
		got := u.PDF(x)
		if !approxEqual(got, 0.1, 1e-12) {
			t.Errorf("Uniform(0,10).PDF(%v) = %v, want 0.1", x, got)
		}
	}
	// Outside
	if u.PDF(-1) != 0 {
		t.Error("Uniform PDF outside range should be 0")
	}
	if u.PDF(11) != 0 {
		t.Error("Uniform PDF outside range should be 0")
	}
}

func TestUniformCDF(t *testing.T) {
	u := NewUniform(0, 10)
	tests := []struct {
		x, want float64
	}{
		{-1, 0},
		{0, 0},
		{5, 0.5},
		{10, 1},
		{11, 1},
	}
	for _, tc := range tests {
		got := u.CDF(tc.x)
		if !approxEqual(got, tc.want, 1e-12) {
			t.Errorf("Uniform(0,10).CDF(%v) = %v, want %v", tc.x, got, tc.want)
		}
	}
}

func TestUniformPPF(t *testing.T) {
	u := NewUniform(2, 8)

	// PPF(0) = low, PPF(1) = high
	if u.PPF(0) != 2 {
		t.Errorf("Uniform(2,8).PPF(0) = %v, want 2", u.PPF(0))
	}
	if u.PPF(1) != 8 {
		t.Errorf("Uniform(2,8).PPF(1) = %v, want 8", u.PPF(1))
	}

	// PPF(0.5) = 5
	if !approxEqual(u.PPF(0.5), 5, 1e-12) {
		t.Errorf("Uniform(2,8).PPF(0.5) = %v, want 5", u.PPF(0.5))
	}

	// Round-trip
	for _, x := range []float64{2.5, 4, 5, 6, 7.5} {
		p := u.CDF(x)
		got := u.PPF(p)
		if !approxEqual(got, x, 1e-10) {
			t.Errorf("Uniform(2,8).PPF(CDF(%v)) = %v, want %v", x, got, x)
		}
	}
}

func TestUniformLogPDF(t *testing.T) {
	u := NewUniform(0, 10)
	got := u.LogPDF(5)
	want := math.Log(0.1)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Uniform(0,10).LogPDF(5) = %v, want %v", got, want)
	}
	if !math.IsInf(u.LogPDF(-1), -1) {
		t.Error("Uniform LogPDF outside range should be -Inf")
	}
}

// ---------------------------------------------------------------------------
// Poisson Distribution Tests
// ---------------------------------------------------------------------------

func TestPoissonPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for lambda=0")
		}
	}()
	NewPoisson(0)
}

func TestPoissonMeanVar(t *testing.T) {
	p := NewPoisson(5)
	if p.Mean() != 5 {
		t.Errorf("Poisson(5).Mean() = %v, want 5", p.Mean())
	}
	if p.Var() != 5 {
		t.Errorf("Poisson(5).Var() = %v, want 5", p.Var())
	}
}

func TestPoissonPMF(t *testing.T) {
	p := NewPoisson(3)

	// PMF(0) = exp(-3)
	got := p.PMF(0)
	want := math.Exp(-3)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Poisson(3).PMF(0) = %v, want %v", got, want)
	}

	// PMF(1) = 3*exp(-3)
	got = p.PMF(1)
	want = 3 * math.Exp(-3)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Poisson(3).PMF(1) = %v, want %v", got, want)
	}

	// PMF(3) = 3^3 * exp(-3) / 6 = 27*exp(-3)/6 = 4.5*exp(-3)
	got = p.PMF(3)
	want = 4.5 * math.Exp(-3)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Poisson(3).PMF(3) = %v, want %v", got, want)
	}

	// PMF for negative k should be 0
	if p.PMF(-1) != 0 {
		t.Error("Poisson PMF for negative k should be 0")
	}

	// Sum of PMFs for large range should be close to 1
	sum := 0.0
	for k := 0; k < 30; k++ {
		sum += p.PMF(k)
	}
	if !approxEqual(sum, 1.0, 1e-10) {
		t.Errorf("Sum of Poisson(3) PMFs 0..29 = %v, want ~1.0", sum)
	}
}

func TestPoissonCDF(t *testing.T) {
	p := NewPoisson(3)

	// CDF(0) = PMF(0) = exp(-3)
	got := p.CDF(0)
	want := math.Exp(-3)
	if !approxEqual(got, want, 1e-8) {
		t.Errorf("Poisson(3).CDF(0) = %v, want %v", got, want)
	}

	// CDF should be monotonically non-decreasing
	prev := 0.0
	for k := 0; k <= 20; k++ {
		c := p.CDF(k)
		if c < prev-1e-10 {
			t.Errorf("Poisson(3).CDF(%d) = %v < CDF(%d) = %v", k, c, k-1, prev)
		}
		prev = c
	}

	// CDF for large k should be close to 1
	got = p.CDF(20)
	if !approxEqual(got, 1.0, 1e-8) {
		t.Errorf("Poisson(3).CDF(20) = %v, want ~1.0", got)
	}

	// CDF for negative k should be 0
	if p.CDF(-1) != 0 {
		t.Error("Poisson CDF for negative k should be 0")
	}

	// Verify CDF matches sum of PMFs
	p2 := NewPoisson(5)
	for _, k := range []int{0, 1, 3, 5, 10} {
		sum := 0.0
		for i := 0; i <= k; i++ {
			sum += p2.PMF(i)
		}
		got := p2.CDF(k)
		if !approxEqual(got, sum, 1e-8) {
			t.Errorf("Poisson(5).CDF(%d) = %v, sum of PMFs = %v", k, got, sum)
		}
	}
}

// ---------------------------------------------------------------------------
// Binomial Distribution Tests
// ---------------------------------------------------------------------------

func TestBinomialPanic(t *testing.T) {
	tests := []struct {
		name string
		n    int
		p    float64
	}{
		{"negative n", -1, 0.5},
		{"p below 0", 10, -0.1},
		{"p above 1", 10, 1.1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Error("expected panic")
				}
			}()
			NewBinomial(tc.n, tc.p)
		})
	}
}

func TestBinomialMeanVar(t *testing.T) {
	b := NewBinomial(10, 0.3)
	if !approxEqual(b.Mean(), 3, 1e-12) {
		t.Errorf("Binomial(10,0.3).Mean() = %v, want 3", b.Mean())
	}
	// Var = 10 * 0.3 * 0.7 = 2.1
	if !approxEqual(b.Var(), 2.1, 1e-12) {
		t.Errorf("Binomial(10,0.3).Var() = %v, want 2.1", b.Var())
	}
}

func TestBinomialPMF(t *testing.T) {
	b := NewBinomial(10, 0.5)

	// PMF(5) = C(10,5) * 0.5^10 = 252/1024
	got := b.PMF(5)
	want := 252.0 / 1024.0
	if !approxEqual(got, want, 1e-10) {
		t.Errorf("Binomial(10,0.5).PMF(5) = %v, want %v", got, want)
	}

	// PMF(0) = 0.5^10
	got = b.PMF(0)
	want = math.Pow(0.5, 10)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Binomial(10,0.5).PMF(0) = %v, want %v", got, want)
	}

	// PMF(10) = 0.5^10
	got = b.PMF(10)
	want = math.Pow(0.5, 10)
	if !approxEqual(got, want, 1e-12) {
		t.Errorf("Binomial(10,0.5).PMF(10) = %v, want %v", got, want)
	}

	// Out of range
	if b.PMF(-1) != 0 {
		t.Error("Binomial PMF for k<0 should be 0")
	}
	if b.PMF(11) != 0 {
		t.Error("Binomial PMF for k>n should be 0")
	}

	// Sum of PMFs should be 1
	sum := 0.0
	for k := 0; k <= 10; k++ {
		sum += b.PMF(k)
	}
	if !approxEqual(sum, 1.0, 1e-10) {
		t.Errorf("Sum of Binomial(10,0.5) PMFs = %v, want 1.0", sum)
	}

	// Edge cases: p=0 and p=1
	b0 := NewBinomial(5, 0)
	if b0.PMF(0) != 1 {
		t.Errorf("Binomial(5,0).PMF(0) = %v, want 1", b0.PMF(0))
	}
	if b0.PMF(1) != 0 {
		t.Errorf("Binomial(5,0).PMF(1) = %v, want 0", b0.PMF(1))
	}

	b1 := NewBinomial(5, 1)
	if b1.PMF(5) != 1 {
		t.Errorf("Binomial(5,1).PMF(5) = %v, want 1", b1.PMF(5))
	}
	if b1.PMF(4) != 0 {
		t.Errorf("Binomial(5,1).PMF(4) = %v, want 0", b1.PMF(4))
	}
}

func TestBinomialCDF(t *testing.T) {
	b := NewBinomial(10, 0.3)

	// CDF should match sum of PMFs
	for _, k := range []int{0, 1, 3, 5, 7, 10} {
		sum := 0.0
		for i := 0; i <= k; i++ {
			sum += b.PMF(i)
		}
		got := b.CDF(k)
		if !approxEqual(got, sum, 1e-6) {
			t.Errorf("Binomial(10,0.3).CDF(%d) = %v, sum of PMFs = %v", k, got, sum)
		}
	}

	// CDF for negative k
	if b.CDF(-1) != 0 {
		t.Error("Binomial CDF for negative k should be 0")
	}

	// CDF(n) = 1
	if b.CDF(10) != 1 {
		t.Errorf("Binomial(10,0.3).CDF(10) = %v, want 1", b.CDF(10))
	}
}

func TestBinomialKnownValues(t *testing.T) {
	// Binomial(20, 0.4): known PMF values
	b := NewBinomial(20, 0.4)
	// PMF(8) = C(20,8) * 0.4^8 * 0.6^12
	// C(20,8) = 125970
	want := 125970.0 * math.Pow(0.4, 8) * math.Pow(0.6, 12)
	got := b.PMF(8)
	if !approxEqual(got, want, 1e-10) {
		t.Errorf("Binomial(20,0.4).PMF(8) = %v, want %v", got, want)
	}
}
