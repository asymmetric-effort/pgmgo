package tabgo

import (
	"math"
	"sort"
)

// RollingSkew computes the rolling sample skewness over a window.
func RollingSkew(s *Series, window int) *Series {
	vals := s.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) < 3 {
			data[i] = nil
			continue
		}
		data[i] = skewness(w)
	}
	return NewSeries(s.Name(), data)
}

// RollingKurtosis computes the rolling excess kurtosis over a window.
func RollingKurtosis(s *Series, window int) *Series {
	vals := s.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) < 4 {
			data[i] = nil
			continue
		}
		data[i] = kurtosis(w)
	}
	return NewSeries(s.Name(), data)
}

// RollingQuantile computes the rolling quantile over a window.
// q should be in [0,1].
func RollingQuantile(s *Series, window int, q float64) *Series {
	vals := s.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) == 0 {
			data[i] = nil
			continue
		}
		data[i] = quantile(w, q)
	}
	return NewSeries(s.Name(), data)
}

// RollingZscore computes the rolling z-score: (x - mean) / std.
func RollingZscore(s *Series, window int) *Series {
	vals := s.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) < 2 {
			data[i] = nil
			continue
		}
		m := mean(w)
		sd := stddev(w)
		if sd < 1e-12 {
			data[i] = 0.0
		} else {
			data[i] = (toFloat64(vals[i]) - m) / sd
		}
	}
	return NewSeries(s.Name(), data)
}

// RollingCov computes the rolling sample covariance between two series.
func RollingCov(s1, s2 *Series, window int) *Series {
	v1 := s1.Values()
	v2 := s2.Values()
	n := len(v1)
	if len(v2) < n {
		n = len(v2)
	}
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w1 := extractWindow(v1, i, window)
		w2 := extractWindow(v2, i, window)
		if len(w1) < 2 || len(w2) < 2 || len(w1) != len(w2) {
			data[i] = nil
			continue
		}
		data[i] = covariance(w1, w2)
	}
	return NewSeries(s1.Name()+"_"+s2.Name()+"_cov", data)
}

// RollingSharpe computes the rolling Sharpe ratio: (mean(r) - rf) / std(r).
func RollingSharpe(returns *Series, window int, riskFreeRate float64) *Series {
	vals := returns.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) < 2 {
			data[i] = nil
			continue
		}
		m := mean(w)
		sd := stddev(w)
		if sd < 1e-12 {
			data[i] = 0.0
		} else {
			data[i] = (m - riskFreeRate) / sd
		}
	}
	return NewSeries(returns.Name(), data)
}

// RollingSortino computes the rolling Sortino ratio using downside deviation.
func RollingSortino(returns *Series, window int, riskFreeRate float64) *Series {
	vals := returns.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) < 2 {
			data[i] = nil
			continue
		}
		m := mean(w)
		dd := downsideDeviation(w, riskFreeRate)
		if dd < 1e-12 {
			data[i] = 0.0
		} else {
			data[i] = (m - riskFreeRate) / dd
		}
	}
	return NewSeries(returns.Name(), data)
}

// RollingMaxDrawdown computes the rolling maximum drawdown within each window.
func RollingMaxDrawdown(s *Series, window int) *Series {
	vals := s.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) == 0 {
			data[i] = nil
			continue
		}
		data[i] = maxDrawdown(w)
	}
	return NewSeries(s.Name(), data)
}

// RollingVaR computes the rolling Value at Risk (historical simulation).
// confidence is e.g. 0.95 for 95% VaR.
func RollingVaR(returns *Series, window int, confidence float64) *Series {
	vals := returns.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) == 0 {
			data[i] = nil
			continue
		}
		// VaR is the negative of the (1-confidence) quantile
		data[i] = -quantile(w, 1-confidence)
	}
	return NewSeries(returns.Name(), data)
}

// RollingCVaR computes the rolling Conditional Value at Risk (Expected Shortfall).
func RollingCVaR(returns *Series, window int, confidence float64) *Series {
	vals := returns.Values()
	n := len(vals)
	data := make([]any, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			data[i] = nil
			continue
		}
		w := extractWindow(vals, i, window)
		if len(w) == 0 {
			data[i] = nil
			continue
		}
		threshold := quantile(w, 1-confidence)
		// Average of all returns <= threshold
		var sum float64
		var count int
		for _, v := range w {
			if v <= threshold {
				sum += v
				count++
			}
		}
		if count == 0 {
			data[i] = -threshold
		} else {
			data[i] = -(sum / float64(count))
		}
	}
	return NewSeries(returns.Name(), data)
}

// extractWindow extracts float64 values from a window ending at index i.
func extractWindow(vals []any, i, window int) []float64 {
	w := make([]float64, 0, window)
	for j := i - window + 1; j <= i; j++ {
		if vals[j] != nil {
			w = append(w, toFloat64(vals[j]))
		}
	}
	return w
}

// skewness computes sample skewness (Fisher's definition).
func skewness(vals []float64) float64 {
	n := float64(len(vals))
	m := mean(vals)
	var m2, m3 float64
	for _, v := range vals {
		d := v - m
		m2 += d * d
		m3 += d * d * d
	}
	m2 /= n
	m3 /= n
	if m2 == 0 {
		return 0
	}
	// Adjusted Fisher-Pearson skewness
	return (math.Sqrt(n*(n-1)) / (n - 2)) * (m3 / math.Pow(m2, 1.5))
}

// kurtosis computes sample excess kurtosis.
func kurtosis(vals []float64) float64 {
	n := float64(len(vals))
	m := mean(vals)
	var m2, m4 float64
	for _, v := range vals {
		d := v - m
		d2 := d * d
		m2 += d2
		m4 += d2 * d2
	}
	m2 /= n
	m4 /= n
	if m2 == 0 {
		return 0
	}
	// Excess kurtosis with bias correction
	rawKurt := m4 / (m2 * m2)
	return ((n-1)/((n-2)*(n-3)))*((n+1)*rawKurt-3*(n-1)) + 0
}

// quantile computes the q-th quantile using linear interpolation.
func quantile(vals []float64, q float64) float64 {
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	n := float64(len(sorted))
	idx := q * (n - 1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi || hi >= len(sorted) {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

// downsideDeviation computes the downside deviation relative to threshold.
func downsideDeviation(vals []float64, threshold float64) float64 {
	var ss float64
	var n int
	for _, v := range vals {
		if v < threshold {
			d := v - threshold
			ss += d * d
			n++
		}
	}
	if n < 2 {
		return 0
	}
	return math.Sqrt(ss / float64(n-1))
}

// maxDrawdown computes the maximum drawdown in a price series.
func maxDrawdown(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	peak := vals[0]
	maxDD := 0.0
	for _, v := range vals {
		if v > peak {
			peak = v
		}
		if peak > 0 {
			dd := (peak - v) / peak
			if dd > maxDD {
				maxDD = dd
			}
		}
	}
	return maxDD
}
