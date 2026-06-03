//go:build unit

package metrics

import (
	"math"
	"testing"
)

// These tests target specific numerical edge cases in the continued fraction
// implementations where intermediate variables (d, c) fall below the tiny
// threshold (1e-30), triggering safety fallbacks.

// TestFinalRIB_InitialDTiny covers line 82-84: 1 - (a+b)*x/(a+1) ≈ 0.
// We need x = (a+1)/(a+b) while also x <= (a+1)/(a+b+2) to avoid the symmetry swap.
// (a+1)/(a+b) <= (a+1)/(a+b+2) is impossible (a+b < a+b+2), so the
// threshold check at line 61 will route to 1-I(1-x,b,a). We need the
// recursive call's d_init to be tiny.
// For the recursive call: 1 - (b+a)*(1-x)/(b+1) ≈ 0 → (1-x) = (b+1)/(a+b).
// So x = 1 - (b+1)/(a+b) = (a-1)/(a+b).
// And the recursive call checks: (1-x) > (b+1)/(b+a+2) → (b+1)/(a+b) > (b+1)/(a+b+2).
// This is always true, so the recursive call also swaps... infinite recursion won't happen
// because the symmetry changes x so that eventually x <= threshold.
// Instead, let's make (a+b)*x/(a+1) = 1 directly: x = (a+1)/(a+b), ensuring no swap.
// The swap happens when x > (a+1)/(a+b+2). We need x <= (a+1)/(a+b+2).
// But (a+1)/(a+b) > (a+1)/(a+b+2) always. So we can never avoid the swap with d_init=tiny.
// After the swap, the new x' = 1-x = 1-(a+1)/(a+b) = (b-1)/(a+b), new a'=b, new b'=a.
// The new d_init = 1 - (b+a)*x'/(b+1) = 1 - (a+b)*(b-1)/((a+b)*(b+1)) = 1 - (b-1)/(b+1) = 2/(b+1).
// For this to be tiny: b+1 ≈ 2e30. That's too large.
// The loop d/c tiny branches require num*d ≈ -1 or num/c ≈ -1 during iteration.
// For the even step: num = m*(b-m)*x / ((a+2m-1)*(a+2m)), d_prev is known.
// When b=m exactly, num=0 so d=1+0=1, c=1+0=1 -- not tiny.
// When m > b, num becomes negative. The key is finding parameters where
// 1+num*d_prev is within 1e-30 of zero. This is essentially impossible
// with IEEE 754 arithmetic without contrived values.
//
// Since these 7 statements are unreachable under normal floating-point arithmetic
// (they protect against numerical underflow in the Lentz CF method), we focus
// on maximizing other coverage. The test below exercises nearby paths.
func TestFinalRIB_NearTinyPaths(t *testing.T) {
	// Exercise parameter ranges that make CF terms very small (close to tiny threshold).
	testCases := [][3]float64{
		// Parameters that produce very small CF terms.
		{1e-8, 1e-8, 0.5},          // very small a,b
		{0.001, 0.001, 0.5},        // small a,b
		{1e-6, 1000.0, 1e-10},      // tiny x, large b
		{1000.0, 1e-6, 1.0 - 1e-6}, // large a, tiny b, x near 1
		{0.5, 0.5, 0.5},            // symmetric
		{1e-10, 1e-10, 0.5},        // extremely small a,b
	}
	for _, tc := range testCases {
		val := regularizedIncompleteBeta(tc[2], tc[0], tc[1])
		if math.IsNaN(val) {
			t.Errorf("NaN for x=%e, a=%e, b=%e", tc[2], tc[0], tc[1])
		}
	}
}

// TestFinalUpperGamma_NearTinyPaths exercises paths close to the tiny thresholds
// in upperGammaCF.
func TestFinalUpperGamma_NearTinyPaths(t *testing.T) {
	testCases := [][2]float64{
		{1e-8, 1e-8},   // extremely small a, x
		{0.001, 0.001}, // small values
		{1e-6, 1000.0}, // tiny a, large x
		{1000.0, 1e-6}, // large a, tiny x
		{0.5, 1e-15},   // fractional a with near-zero x
	}
	for _, tc := range testCases {
		val := upperGammaCF(tc[0], tc[1])
		if math.IsNaN(val) {
			t.Errorf("NaN for a=%e, x=%e", tc[0], tc[1])
		}
	}
}
