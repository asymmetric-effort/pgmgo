package models

import (
	"fmt"
	"math"
	"math/rand"
)

// MarkovChain represents a discrete-time Markov chain with a finite state
// space. The transition matrix T[i][j] gives the probability of transitioning
// from state i to state j.
type MarkovChain struct {
	transitionMatrix [][]float64
	stateNames       []string
}

// NewMarkovChain creates a new MarkovChain. transitionMatrix must be square
// and each row must sum to 1 (within tolerance). stateNames must match the
// number of states, or be nil (in which case states are unnamed).
func NewMarkovChain(transitionMatrix [][]float64, stateNames []string) (*MarkovChain, error) {
	n := len(transitionMatrix)
	if n == 0 {
		return nil, fmt.Errorf("models: transition matrix must not be empty")
	}

	const tol = 1e-6
	for i, row := range transitionMatrix {
		if len(row) != n {
			return nil, fmt.Errorf("models: transition matrix row %d has length %d, expected %d", i, len(row), n)
		}
		sum := 0.0
		for _, v := range row {
			if v < -tol {
				return nil, fmt.Errorf("models: transition matrix has negative value %f at row %d", v, i)
			}
			sum += v
		}
		if math.Abs(sum-1.0) > tol {
			return nil, fmt.Errorf("models: transition matrix row %d sums to %f, expected 1.0", i, sum)
		}
	}

	if stateNames != nil && len(stateNames) != n {
		return nil, fmt.Errorf("models: stateNames length %d does not match matrix size %d", len(stateNames), n)
	}

	// Deep copy.
	mat := make([][]float64, n)
	for i, row := range transitionMatrix {
		mat[i] = make([]float64, n)
		copy(mat[i], row)
	}

	var names []string
	if stateNames != nil {
		names = make([]string, n)
		copy(names, stateNames)
	}

	return &MarkovChain{
		transitionMatrix: mat,
		stateNames:       names,
	}, nil
}

// NumStates returns the number of states.
func (mc *MarkovChain) NumStates() int {
	return len(mc.transitionMatrix)
}

// StateNames returns a copy of the state names, or nil if unnamed.
func (mc *MarkovChain) StateNames() []string {
	if mc.stateNames == nil {
		return nil
	}
	names := make([]string, len(mc.stateNames))
	copy(names, mc.stateNames)
	return names
}

// TransitionMatrix returns a deep copy of the transition matrix.
func (mc *MarkovChain) TransitionMatrix() [][]float64 {
	n := len(mc.transitionMatrix)
	mat := make([][]float64, n)
	for i, row := range mc.transitionMatrix {
		mat[i] = make([]float64, n)
		copy(mat[i], row)
	}
	return mat
}

// StationaryDistribution computes the stationary distribution pi such that
// pi * T = pi, using the power iteration method.
func (mc *MarkovChain) StationaryDistribution() ([]float64, error) {
	n := mc.NumStates()
	if n == 0 {
		return nil, fmt.Errorf("models: empty Markov chain")
	}

	// Initialize uniform distribution.
	pi := make([]float64, n)
	for i := range pi {
		pi[i] = 1.0 / float64(n)
	}

	const maxIter = 10000
	const tol = 1e-10

	piNext := make([]float64, n)

	for iter := 0; iter < maxIter; iter++ {
		// pi_next[j] = sum_i pi[i] * T[i][j]
		for j := 0; j < n; j++ {
			piNext[j] = 0
			for i := 0; i < n; i++ {
				piNext[j] += pi[i] * mc.transitionMatrix[i][j]
			}
		}

		// Check convergence.
		maxDiff := 0.0
		for j := 0; j < n; j++ {
			d := math.Abs(piNext[j] - pi[j])
			if d > maxDiff {
				maxDiff = d
			}
		}

		copy(pi, piNext)

		if maxDiff < tol {
			return pi, nil
		}
	}

	return pi, nil
}

// Sample generates a sequence of n state indices starting from startState.
// seed controls the random number generator (use 0 for non-deterministic).
func (mc *MarkovChain) Sample(n int, startState int, seed int64) ([]int, error) {
	if n <= 0 {
		return nil, fmt.Errorf("models: n must be positive, got %d", n)
	}
	numStates := mc.NumStates()
	if startState < 0 || startState >= numStates {
		return nil, fmt.Errorf("models: startState %d out of range [0, %d)", startState, numStates)
	}

	rng := rand.New(rand.NewSource(seed))
	samples := make([]int, n)
	samples[0] = startState

	for i := 1; i < n; i++ {
		current := samples[i-1]
		u := rng.Float64()
		cumSum := 0.0
		next := numStates - 1
		for s := 0; s < numStates; s++ {
			cumSum += mc.transitionMatrix[current][s]
			if u < cumSum {
				next = s
				break
			}
		}
		samples[i] = next
	}

	return samples, nil
}

// IsAbsorbing returns true if the Markov chain has at least one absorbing
// state (a state i where T[i][i] == 1).
func (mc *MarkovChain) IsAbsorbing() bool {
	for i, row := range mc.transitionMatrix {
		if row[i] == 1.0 {
			return true
		}
	}
	return false
}

// IsErgodic returns true if the Markov chain is ergodic (irreducible and
// aperiodic). Irreducibility is checked by verifying that all states can
// reach all other states. Aperiodicity is checked by verifying that the GCD
// of return times to each state is 1 (approximated by checking that T^k has
// all positive entries for some k).
func (mc *MarkovChain) IsErgodic() bool {
	n := mc.NumStates()
	if n == 0 {
		return false
	}

	// Check irreducibility: every state must be reachable from every other state.
	for start := 0; start < n; start++ {
		visited := make([]bool, n)
		visited[start] = true
		queue := []int{start}
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			for next := 0; next < n; next++ {
				if mc.transitionMatrix[cur][next] > 0 && !visited[next] {
					visited[next] = true
					queue = append(queue, next)
				}
			}
		}
		for _, v := range visited {
			if !v {
				return false
			}
		}
	}

	// Check aperiodicity: compute T^k and check if all entries are positive.
	// Use matrix power with max n^2 iterations.
	mat := mc.TransitionMatrix()
	maxIter := n * n
	if maxIter < 2 {
		maxIter = 2
	}
	if maxIter > 100 {
		maxIter = 100
	}

	for iter := 0; iter < maxIter; iter++ {
		allPositive := true
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if mat[i][j] <= 0 {
					allPositive = false
					break
				}
			}
			if !allPositive {
				break
			}
		}
		if allPositive {
			return true
		}

		// mat = mat * T
		next := make([][]float64, n)
		for i := 0; i < n; i++ {
			next[i] = make([]float64, n)
			for j := 0; j < n; j++ {
				for k := 0; k < n; k++ {
					next[i][j] += mat[i][k] * mc.transitionMatrix[k][j]
				}
			}
		}
		mat = next
	}

	return false
}
