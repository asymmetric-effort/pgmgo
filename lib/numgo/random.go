package numgo

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// RNG wraps a seeded random source for reproducible random number generation.
type RNG struct {
	src *rand.Rand
}

// NewRNG creates a new RNG with the given seed.
func NewRNG(seed int64) *RNG {
	return &RNG{
		src: rand.New(rand.NewPCG(uint64(seed), uint64(seed>>1)^0xa02bdbf7bb3c0785)),
	}
}

// Rand returns an NDArray of the given shape with values drawn uniformly from [0, 1).
func (r *RNG) Rand(shape ...int) *NDArray {
	size := product(shape)
	data := make([]float64, size)
	for i := range data {
		data[i] = r.src.Float64()
	}
	return NewNDArray(shape, data)
}

// Randn returns an NDArray of the given shape with values drawn from the
// standard normal distribution (mean=0, std=1) using the Box-Muller transform.
func (r *RNG) Randn(shape ...int) *NDArray {
	size := product(shape)
	data := make([]float64, size)
	for i := 0; i < size; i += 2 {
		u1 := r.src.Float64()
		u2 := r.src.Float64()
		// Avoid log(0).
		for u1 == 0 {
			u1 = r.src.Float64()
		}
		z0 := math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
		z1 := math.Sqrt(-2*math.Log(u1)) * math.Sin(2*math.Pi*u2)
		data[i] = z0
		if i+1 < size {
			data[i+1] = z1
		}
	}
	return NewNDArray(shape, data)
}

// RandInt returns an NDArray of the given shape with integer values drawn
// uniformly from [low, high).
func (r *RNG) RandInt(low, high int, shape ...int) *NDArray {
	if low >= high {
		panic(fmt.Sprintf("numgo: RandInt requires low < high, got low=%d high=%d", low, high))
	}
	size := product(shape)
	data := make([]float64, size)
	span := high - low
	for i := range data {
		data[i] = float64(r.src.IntN(span) + low)
	}
	return NewNDArray(shape, data)
}

// Choice returns a slice of random indices in [0, n).
// If replace is true, indices may repeat; otherwise they are unique
// and size must be <= n.
func (r *RNG) Choice(n int, size int, replace bool) []int {
	if n <= 0 {
		panic("numgo: Choice requires n > 0")
	}
	if size < 0 {
		panic("numgo: Choice requires size >= 0")
	}
	if !replace && size > n {
		panic(fmt.Sprintf("numgo: Choice without replacement requires size <= n, got size=%d n=%d", size, n))
	}

	if replace {
		result := make([]int, size)
		for i := range result {
			result[i] = r.src.IntN(n)
		}
		return result
	}

	// Without replacement: Fisher-Yates partial shuffle.
	pool := make([]int, n)
	for i := range pool {
		pool[i] = i
	}
	for i := 0; i < size; i++ {
		j := i + r.src.IntN(n-i)
		pool[i], pool[j] = pool[j], pool[i]
	}
	result := make([]int, size)
	copy(result, pool[:size])
	return result
}

// Shuffle performs an in-place Fisher-Yates shuffle on the first axis of the
// array. For 1-D arrays this shuffles elements; for N-D arrays it shuffles
// the sub-arrays along axis 0.
func (r *RNG) Shuffle(a *NDArray) {
	n := a.shape[0]
	if a.Ndim() == 1 {
		for i := n - 1; i > 0; i-- {
			j := r.src.IntN(i + 1)
			a.data[i], a.data[j] = a.data[j], a.data[i]
		}
		return
	}

	// For N-D: swap entire sub-arrays along axis 0.
	stride := a.strides[0]
	tmp := make([]float64, stride)
	for i := n - 1; i > 0; i-- {
		j := r.src.IntN(i + 1)
		if i != j {
			iOff := i * stride
			jOff := j * stride
			copy(tmp, a.data[iOff:iOff+stride])
			copy(a.data[iOff:iOff+stride], a.data[jOff:jOff+stride])
			copy(a.data[jOff:jOff+stride], tmp)
		}
	}
}

// Normal returns an NDArray of the given shape with values drawn from a
// normal distribution with the specified mean and standard deviation.
func (r *RNG) Normal(mean, std float64, shape ...int) *NDArray {
	a := r.Randn(shape...)
	data := a.Data()
	for i := range data {
		data[i] = data[i]*std + mean
	}
	return NewNDArray(shape, data)
}

// Uniform returns an NDArray of the given shape with values drawn uniformly
// from [low, high).
func (r *RNG) Uniform(low, high float64, shape ...int) *NDArray {
	if low >= high {
		panic(fmt.Sprintf("numgo: Uniform requires low < high, got low=%f high=%f", low, high))
	}
	a := r.Rand(shape...)
	span := high - low
	data := a.Data()
	for i := range data {
		data[i] = data[i]*span + low
	}
	return NewNDArray(shape, data)
}

// gammaVariate generates a single sample from the Gamma(alpha, 1) distribution
// using the Marsaglia and Tsang method for alpha >= 1, with a boost for alpha < 1.
func (r *RNG) gammaVariate(alpha float64) float64 {
	if alpha <= 0 {
		panic("numgo: gammaVariate requires alpha > 0")
	}

	if alpha < 1.0 {
		// Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
		return r.gammaVariate(alpha+1.0) * math.Pow(r.src.Float64(), 1.0/alpha)
	}

	// Marsaglia and Tsang's method for alpha >= 1.
	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		var x, v float64
		for {
			x = r.boxMullerSingle()
			v = 1.0 + c*x
			if v > 0 {
				break
			}
		}
		v = v * v * v
		u := r.src.Float64()
		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

// boxMullerSingle returns a single standard normal variate.
func (r *RNG) boxMullerSingle() float64 {
	u1 := r.src.Float64()
	for u1 == 0 {
		u1 = r.src.Float64()
	}
	u2 := r.src.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// Dirichlet draws a single sample from the Dirichlet distribution with
// parameter vector alpha. It returns a probability vector of length len(alpha).
// The implementation draws independent Gamma(alpha_i, 1) samples and normalizes.
func (r *RNG) Dirichlet(alpha []float64) []float64 {
	if len(alpha) == 0 {
		panic("numgo: Dirichlet requires non-empty alpha")
	}
	samples := make([]float64, len(alpha))
	total := 0.0
	for i, a := range alpha {
		samples[i] = r.gammaVariate(a)
		total += samples[i]
	}
	for i := range samples {
		samples[i] /= total
	}
	return samples
}

// Multinomial draws a single sample from the multinomial distribution:
// distribute n trials among len(pvals) categories with the given probabilities.
// Returns a slice of counts (length len(pvals)) summing to n.
func (r *RNG) Multinomial(n int, pvals []float64) []int {
	if n < 0 {
		panic("numgo: Multinomial requires n >= 0")
	}
	if len(pvals) == 0 {
		panic("numgo: Multinomial requires non-empty pvals")
	}

	// Build cumulative probabilities.
	cumulative := make([]float64, len(pvals))
	cumulative[0] = pvals[0]
	for i := 1; i < len(pvals); i++ {
		cumulative[i] = cumulative[i-1] + pvals[i]
	}
	// Normalize to handle floating-point drift.
	total := cumulative[len(cumulative)-1]

	counts := make([]int, len(pvals))
	for trial := 0; trial < n; trial++ {
		u := r.src.Float64() * total
		// Binary search for the bucket.
		lo, hi := 0, len(cumulative)-1
		for lo < hi {
			mid := (lo + hi) / 2
			if cumulative[mid] <= u {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		counts[lo]++
	}
	return counts
}
