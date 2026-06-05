package scigo

import (
	"errors"
	"math"
	"math/rand"
)

// MultivariateNormal represents a multivariate normal distribution.
type MultivariateNormal struct {
	mean  []float64
	cov   [][]float64
	cholL [][]float64 // Lower-triangular Cholesky factor: cov = L * L^T
	dim   int
}

// NewMultivariateNormal creates a MultivariateNormal distribution from a mean
// vector and covariance matrix. The covariance matrix must be symmetric and
// positive definite. Internally it computes the Cholesky decomposition.
func NewMultivariateNormal(mean []float64, cov [][]float64) (*MultivariateNormal, error) {
	n := len(mean)
	if n == 0 {
		return nil, errors.New("scigo.NewMultivariateNormal: mean must not be empty")
	}
	if len(cov) != n {
		return nil, errors.New("scigo.NewMultivariateNormal: cov rows must match mean length")
	}
	for i, row := range cov {
		if len(row) != n {
			return nil, errors.New("scigo.NewMultivariateNormal: cov must be square")
		}
		// Check symmetry.
		for j := 0; j < i; j++ {
			if math.Abs(cov[i][j]-cov[j][i]) > 1e-10 {
				return nil, errors.New("scigo.NewMultivariateNormal: cov must be symmetric")
			}
		}
	}

	cholL, err := ChoFactor(cov)
	if err != nil {
		return nil, errors.New("scigo.NewMultivariateNormal: " + err.Error())
	}

	// Deep copy mean and cov.
	meanCopy := make([]float64, n)
	copy(meanCopy, mean)
	covCopy := make([][]float64, n)
	for i := range cov {
		covCopy[i] = make([]float64, n)
		copy(covCopy[i], cov[i])
	}

	return &MultivariateNormal{
		mean:  meanCopy,
		cov:   covCopy,
		cholL: cholL,
		dim:   n,
	}, nil
}

// Dim returns the dimensionality.
func (mvn *MultivariateNormal) Dim() int {
	return mvn.dim
}

// MeanVec returns a copy of the mean vector.
func (mvn *MultivariateNormal) MeanVec() []float64 {
	m := make([]float64, mvn.dim)
	copy(m, mvn.mean)
	return m
}

// CovMatrix returns a copy of the covariance matrix.
func (mvn *MultivariateNormal) CovMatrix() [][]float64 {
	c := make([][]float64, mvn.dim)
	for i := range c {
		c[i] = make([]float64, mvn.dim)
		copy(c[i], mvn.cov[i])
	}
	return c
}

// PDF returns the probability density function evaluated at x.
func (mvn *MultivariateNormal) PDF(x []float64) float64 {
	return math.Exp(mvn.LogPDF(x))
}

// LogPDF returns the natural logarithm of the PDF evaluated at x.
func (mvn *MultivariateNormal) LogPDF(x []float64) float64 {
	n := mvn.dim
	if len(x) != n {
		return math.Inf(-1)
	}

	// diff = x - mean
	diff := make([]float64, n)
	for i := 0; i < n; i++ {
		diff[i] = x[i] - mvn.mean[i]
	}

	// Solve L * y = diff via forward substitution.
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := diff[i]
		for j := 0; j < i; j++ {
			s -= mvn.cholL[i][j] * y[j]
		}
		y[i] = s / mvn.cholL[i][i]
	}

	// Mahalanobis^2 = y^T * y
	maha2 := 0.0
	for _, v := range y {
		maha2 += v * v
	}

	// log det(cov) = 2 * sum(log(L[i][i]))
	logDet := 0.0
	for i := 0; i < n; i++ {
		logDet += math.Log(mvn.cholL[i][i])
	}
	logDet *= 2

	return -0.5 * (float64(n)*math.Log(2*math.Pi) + logDet + maha2)
}

// Sample generates n samples from the distribution, returning an n x dim matrix.
// Each row is one sample. seed controls the random number generator.
func (mvn *MultivariateNormal) Sample(nSamples int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	d := mvn.dim
	samples := make([][]float64, nSamples)

	for s := 0; s < nSamples; s++ {
		z := make([]float64, d)
		for i := 0; i < d; i++ {
			z[i] = rng.NormFloat64()
		}
		// x = mean + L * z
		x := make([]float64, d)
		for i := 0; i < d; i++ {
			x[i] = mvn.mean[i]
			for j := 0; j <= i; j++ {
				x[i] += mvn.cholL[i][j] * z[j]
			}
		}
		samples[s] = x
	}
	return samples
}

// Mahalanobis returns the Mahalanobis distance of x from the distribution mean.
func (mvn *MultivariateNormal) Mahalanobis(x []float64) float64 {
	n := mvn.dim
	diff := make([]float64, n)
	for i := 0; i < n; i++ {
		diff[i] = x[i] - mvn.mean[i]
	}

	// Solve L * y = diff.
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		s := diff[i]
		for j := 0; j < i; j++ {
			s -= mvn.cholL[i][j] * y[j]
		}
		y[i] = s / mvn.cholL[i][i]
	}

	maha2 := 0.0
	for _, v := range y {
		maha2 += v * v
	}
	return math.Sqrt(maha2)
}

// MarginalDistribution returns the marginal distribution over the specified indices.
func (mvn *MultivariateNormal) MarginalDistribution(indices []int) *MultivariateNormal {
	k := len(indices)
	newMean := make([]float64, k)
	newCov := make([][]float64, k)
	for i := 0; i < k; i++ {
		newMean[i] = mvn.mean[indices[i]]
		newCov[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			newCov[i][j] = mvn.cov[indices[i]][indices[j]]
		}
	}
	result, _ := NewMultivariateNormal(newMean, newCov)
	return result
}

// ConditionalDistribution returns the conditional distribution given observed values.
// observed maps dimension index to observed value.
// Returns the conditional distribution over the unobserved dimensions.
func (mvn *MultivariateNormal) ConditionalDistribution(observed map[int]float64) *MultivariateNormal {
	n := mvn.dim

	// Partition into observed (2) and unobserved (1) indices.
	var idx1, idx2 []int
	for i := 0; i < n; i++ {
		if _, ok := observed[i]; ok {
			idx2 = append(idx2, i)
		} else {
			idx1 = append(idx1, i)
		}
	}

	n1 := len(idx1)
	n2 := len(idx2)
	if n1 == 0 || n2 == 0 {
		return mvn
	}

	// Extract submatrices.
	mu1 := make([]float64, n1)
	mu2 := make([]float64, n2)
	for i, ii := range idx1 {
		mu1[i] = mvn.mean[ii]
	}
	for i, ii := range idx2 {
		mu2[i] = mvn.mean[ii]
	}

	sig11 := make([][]float64, n1)
	sig12 := make([][]float64, n1)
	sig22 := make([][]float64, n2)
	for i := 0; i < n1; i++ {
		sig11[i] = make([]float64, n1)
		sig12[i] = make([]float64, n2)
		for j := 0; j < n1; j++ {
			sig11[i][j] = mvn.cov[idx1[i]][idx1[j]]
		}
		for j := 0; j < n2; j++ {
			sig12[i][j] = mvn.cov[idx1[i]][idx2[j]]
		}
	}
	for i := 0; i < n2; i++ {
		sig22[i] = make([]float64, n2)
		for j := 0; j < n2; j++ {
			sig22[i][j] = mvn.cov[idx2[i]][idx2[j]]
		}
	}

	// sig22Inv = sig22^{-1}
	sig22Inv, err := matInverse(sig22)
	if err != nil {
		return mvn
	}

	// x2 - mu2
	diff2 := make([]float64, n2)
	for i, ii := range idx2 {
		diff2[i] = observed[ii] - mu2[i]
	}

	// condMean = mu1 + sig12 * sig22Inv * diff2
	condMean := make([]float64, n1)
	for i := 0; i < n1; i++ {
		condMean[i] = mu1[i]
		for j := 0; j < n2; j++ {
			t := 0.0
			for k := 0; k < n2; k++ {
				t += sig22Inv[j][k] * diff2[k]
			}
			condMean[i] += sig12[i][j] * t
		}
	}

	// condCov = sig11 - sig12 * sig22Inv * sig12^T
	// sig12 * sig22Inv
	tmp := make([][]float64, n1)
	for i := 0; i < n1; i++ {
		tmp[i] = make([]float64, n2)
		for j := 0; j < n2; j++ {
			for k := 0; k < n2; k++ {
				tmp[i][j] += sig12[i][k] * sig22Inv[k][j]
			}
		}
	}
	// tmp * sig12^T
	condCov := make([][]float64, n1)
	for i := 0; i < n1; i++ {
		condCov[i] = make([]float64, n1)
		for j := 0; j < n1; j++ {
			condCov[i][j] = sig11[i][j]
			for k := 0; k < n2; k++ {
				condCov[i][j] -= tmp[i][k] * sig12[j][k]
			}
		}
	}

	// Symmetrize.
	for i := 0; i < n1; i++ {
		for j := i + 1; j < n1; j++ {
			avg := (condCov[i][j] + condCov[j][i]) / 2
			condCov[i][j] = avg
			condCov[j][i] = avg
		}
	}

	result, _ := NewMultivariateNormal(condMean, condCov)
	return result
}

// ---------------------------------------------------------------------------
// Gaussian Copula
// ---------------------------------------------------------------------------

// GaussianCopula represents a Gaussian copula parameterized by a correlation matrix.
type GaussianCopula struct {
	corr  [][]float64
	cholL [][]float64
	dim   int
}

// NewGaussianCopula creates a Gaussian copula from a correlation matrix.
// The matrix must be symmetric positive definite with ones on the diagonal.
func NewGaussianCopula(corr [][]float64) (*GaussianCopula, error) {
	n := len(corr)
	if n == 0 {
		return nil, errors.New("scigo.NewGaussianCopula: empty correlation matrix")
	}
	for i, row := range corr {
		if len(row) != n {
			return nil, errors.New("scigo.NewGaussianCopula: correlation matrix must be square")
		}
		if math.Abs(row[i]-1.0) > 1e-10 {
			return nil, errors.New("scigo.NewGaussianCopula: diagonal must be 1")
		}
		for j := 0; j < i; j++ {
			if math.Abs(corr[i][j]-corr[j][i]) > 1e-10 {
				return nil, errors.New("scigo.NewGaussianCopula: matrix must be symmetric")
			}
		}
	}

	cholL, err := ChoFactor(corr)
	if err != nil {
		return nil, errors.New("scigo.NewGaussianCopula: " + err.Error())
	}

	corrCopy := make([][]float64, n)
	for i := range corr {
		corrCopy[i] = make([]float64, n)
		copy(corrCopy[i], corr[i])
	}

	return &GaussianCopula{
		corr:  corrCopy,
		cholL: cholL,
		dim:   n,
	}, nil
}

// Dim returns the dimensionality of the copula.
func (gc *GaussianCopula) Dim() int {
	return gc.dim
}

// CorrMatrix returns a copy of the correlation matrix.
func (gc *GaussianCopula) CorrMatrix() [][]float64 {
	c := make([][]float64, gc.dim)
	for i := range c {
		c[i] = make([]float64, gc.dim)
		copy(c[i], gc.corr[i])
	}
	return c
}

// Sample generates n samples from the copula, returning uniform [0,1] values
// with the specified correlation structure. Each row is one sample.
func (gc *GaussianCopula) Sample(nSamples int, seed int64) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	d := gc.dim
	stdNorm := NewNormal(0, 1)

	samples := make([][]float64, nSamples)
	for s := 0; s < nSamples; s++ {
		// Generate standard normal vector.
		z := make([]float64, d)
		for i := 0; i < d; i++ {
			z[i] = rng.NormFloat64()
		}
		// Correlate: y = L * z
		u := make([]float64, d)
		for i := 0; i < d; i++ {
			y := 0.0
			for j := 0; j <= i; j++ {
				y += gc.cholL[i][j] * z[j]
			}
			// Transform to uniform via standard normal CDF.
			u[i] = stdNorm.CDF(y)
		}
		samples[s] = u
	}
	return samples
}
