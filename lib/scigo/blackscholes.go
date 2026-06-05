package scigo

import (
	"errors"
	"math"
	"math/rand"
)

// Greeks holds the partial derivatives of the Black-Scholes option price
// with respect to various parameters.
type Greeks struct {
	Delta float64 // dPrice/dS
	Gamma float64 // d²Price/dS²
	Theta float64 // dPrice/dT (per year)
	Vega  float64 // dPrice/dSigma (per 1 unit change in sigma)
	Rho   float64 // dPrice/dR (per 1 unit change in r)
}

// cdfNormal returns the standard normal CDF.
func cdfNormal(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

// pdfNormal returns the standard normal PDF.
func pdfNormal(x float64) float64 {
	return math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
}

// bsD1D2 computes the d1 and d2 parameters for Black-Scholes.
func bsD1D2(S, K, T, r, sigma float64) (float64, float64) {
	d1 := (math.Log(S/K) + (r+0.5*sigma*sigma)*T) / (sigma * math.Sqrt(T))
	d2 := d1 - sigma*math.Sqrt(T)
	return d1, d2
}

// BlackScholesCall computes the Black-Scholes price of a European call option.
// S is the underlying price, K is the strike, T is time to expiration (years),
// r is the risk-free rate, and sigma is the volatility.
func BlackScholesCall(S, K, T, r, sigma float64) float64 {
	if T <= 0 {
		// At expiration
		return math.Max(S-K, 0)
	}
	d1, d2 := bsD1D2(S, K, T, r, sigma)
	return S*cdfNormal(d1) - K*math.Exp(-r*T)*cdfNormal(d2)
}

// BlackScholesPut computes the Black-Scholes price of a European put option.
// S is the underlying price, K is the strike, T is time to expiration (years),
// r is the risk-free rate, and sigma is the volatility.
func BlackScholesPut(S, K, T, r, sigma float64) float64 {
	if T <= 0 {
		return math.Max(K-S, 0)
	}
	d1, d2 := bsD1D2(S, K, T, r, sigma)
	return K*math.Exp(-r*T)*cdfNormal(-d2) - S*cdfNormal(-d1)
}

// BlackScholesGreeks computes the Greeks for a European call option.
// S is the underlying price, K is the strike, T is time to expiration (years),
// r is the risk-free rate, and sigma is the volatility.
func BlackScholesGreeks(S, K, T, r, sigma float64) *Greeks {
	if T <= 0 {
		delta := 0.0
		if S > K {
			delta = 1.0
		}
		return &Greeks{Delta: delta}
	}

	d1, d2 := bsD1D2(S, K, T, r, sigma)
	sqrtT := math.Sqrt(T)
	nd1 := pdfNormal(d1)
	Nd1 := cdfNormal(d1)
	Nd2 := cdfNormal(d2)
	discount := math.Exp(-r * T)

	delta := Nd1
	gamma := nd1 / (S * sigma * sqrtT)
	theta := -(S*nd1*sigma)/(2*sqrtT) - r*K*discount*Nd2
	vega := S * nd1 * sqrtT
	rho := K * T * discount * Nd2

	return &Greeks{
		Delta: delta,
		Gamma: gamma,
		Theta: theta,
		Vega:  vega,
		Rho:   rho,
	}
}

// ImpliedVolatility computes the implied volatility from an option price using bisection.
// price is the observed option price, S is the underlying price, K is the strike,
// T is time to expiration, r is the risk-free rate.
// optionType must be "call" or "put".
func ImpliedVolatility(price, S, K, T, r float64, optionType string) (float64, error) {
	if price <= 0 {
		return 0, errors.New("scigo: ImpliedVolatility price must be positive")
	}
	if T <= 0 {
		return 0, errors.New("scigo: ImpliedVolatility T must be positive")
	}
	if optionType != "call" && optionType != "put" {
		return 0, errors.New("scigo: ImpliedVolatility optionType must be 'call' or 'put'")
	}

	pricer := BlackScholesCall
	if optionType == "put" {
		pricer = BlackScholesPut
	}

	lo := 1e-6
	hi := 10.0
	tol := 1e-10
	maxIter := 200

	fLo := pricer(S, K, T, r, lo) - price
	fHi := pricer(S, K, T, r, hi) - price

	if fLo*fHi > 0 {
		return 0, errors.New("scigo: ImpliedVolatility no root found in [1e-6, 10.0]")
	}

	for i := 0; i < maxIter; i++ {
		mid := (lo + hi) / 2
		fMid := pricer(S, K, T, r, mid) - price
		if math.Abs(fMid) < tol || (hi-lo)/2 < tol {
			return mid, nil
		}
		if fMid*fLo < 0 {
			hi = mid
			fHi = fMid
		} else {
			lo = mid
			fLo = fMid
		}
	}

	return (lo + hi) / 2, nil
}

// BinomialTree prices a European or American option using a Cox-Ross-Rubinstein
// binomial tree with n time steps. optionType must be "call" or "put".
// If american is true, early exercise is allowed.
func BinomialTree(S, K, T, r, sigma float64, n int, optionType string, american bool) float64 {
	if n <= 0 {
		panic("scigo: BinomialTree n must be positive")
	}

	dt := T / float64(n)
	u := math.Exp(sigma * math.Sqrt(dt))
	d := 1.0 / u
	p := (math.Exp(r*dt) - d) / (u - d)
	discount := math.Exp(-r * dt)

	// Build terminal payoff
	values := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		sT := S * math.Pow(u, float64(n-i)) * math.Pow(d, float64(i))
		if optionType == "call" {
			values[i] = math.Max(sT-K, 0)
		} else {
			values[i] = math.Max(K-sT, 0)
		}
	}

	// Step backwards through the tree
	for step := n - 1; step >= 0; step-- {
		for i := 0; i <= step; i++ {
			values[i] = discount * (p*values[i] + (1-p)*values[i+1])
			if american {
				sNode := S * math.Pow(u, float64(step-i)) * math.Pow(d, float64(i))
				var exercise float64
				if optionType == "call" {
					exercise = math.Max(sNode-K, 0)
				} else {
					exercise = math.Max(K-sNode, 0)
				}
				if exercise > values[i] {
					values[i] = exercise
				}
			}
		}
	}

	return values[0]
}

// MonteCarloPricing prices a derivative using Monte Carlo simulation.
// S is the initial price, K is the strike (used only if relevant to payoff),
// T is time to expiration, r is the risk-free rate, sigma is volatility,
// nPaths is the number of simulation paths, seed controls the RNG,
// and payoff is a function that maps the terminal stock price to the option payoff.
func MonteCarloPricing(S, K, T, r, sigma float64, nPaths int, seed int64, payoff func(float64) float64) float64 {
	if nPaths <= 0 {
		panic("scigo: MonteCarloPricing nPaths must be positive")
	}

	rng := rand.New(rand.NewSource(seed))
	drift := (r - 0.5*sigma*sigma) * T
	vol := sigma * math.Sqrt(T)
	discount := math.Exp(-r * T)

	sum := 0.0
	for i := 0; i < nPaths; i++ {
		z := rng.NormFloat64()
		sT := S * math.Exp(drift+vol*z)
		sum += payoff(sT)
	}

	return discount * sum / float64(nPaths)
}
