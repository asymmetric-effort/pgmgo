// Command sampling demonstrates forward sampling and likelihood-weighted
// sampling from a Bayesian network, comparing empirical marginals with
// exact inference results.
package main

import (
	"fmt"
	"log"
	"math"

	"github.com/asymmetric-effort/pgmgo/example_models"
	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/src/sampling"
)

func main() {
	// Build the Student network.
	bn := example_models.Student()

	fmt.Println("=== Student Bayesian Network — Sampling ===")
	fmt.Println()

	// --- Forward Sampling ---
	sampler, err := sampling.NewBayesianModelSampling(bn, 12345)
	if err != nil {
		log.Fatalf("Failed to create sampler: %v", err)
	}

	nSamples := 50000
	samples, err := sampler.ForwardSample(nSamples)
	if err != nil {
		log.Fatalf("Forward sampling failed: %v", err)
	}

	fmt.Printf("Forward sampled %d instances\n\n", nSamples)

	// Compute empirical marginal P(G) from forward samples.
	gradeVals := samples.Column("G").Values()
	gradeCounts := make([]int, 3)
	for _, v := range gradeVals {
		gradeCounts[v.(int)]++
	}

	fmt.Println("=== Empirical P(G) from Forward Sampling ===")
	gradeStates := bn.GetStates("G")
	for i, state := range gradeStates {
		fmt.Printf("  P(G=%s) = %.4f\n", state, float64(gradeCounts[i])/float64(nSamples))
	}

	// Exact marginal P(G) via Variable Elimination.
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		log.Fatal(err)
	}
	ve := inference.NewVariableElimination(markovFactors)
	exactG, err := ve.Query([]string{"G"}, nil)
	if err != nil {
		log.Fatalf("Exact query failed: %v", err)
	}

	fmt.Println("\n=== Exact P(G) via Variable Elimination ===")
	for i, state := range gradeStates {
		prob := exactG.GetValue(map[string]int{"G": i})
		fmt.Printf("  P(G=%s) = %.4f\n", state, prob)
	}

	// --- Likelihood-Weighted Sampling with Evidence ---
	fmt.Println("\n=== Likelihood-Weighted Sampling: P(G | D=Easy, I=High) ===")

	sampler2, err := sampling.NewBayesianModelSampling(bn, 67890)
	if err != nil {
		log.Fatal(err)
	}

	evidence := map[string]int{"D": 0, "I": 1} // D=Easy, I=High
	lwSamples, weights, err := sampler2.LikelihoodWeightedSample(nSamples, evidence)
	if err != nil {
		log.Fatalf("Likelihood-weighted sampling failed: %v", err)
	}

	// Compute weighted empirical marginal P(G | evidence).
	lwGradeVals := lwSamples.Column("G").Values()
	weightedCounts := make([]float64, 3)
	totalWeight := 0.0
	for i, v := range lwGradeVals {
		weightedCounts[v.(int)] += weights[i]
		totalWeight += weights[i]
	}

	fmt.Println("\nEmpirical (likelihood-weighted):")
	for i, state := range gradeStates {
		fmt.Printf("  P(G=%s | D=Easy, I=High) = %.4f\n", state, weightedCounts[i]/totalWeight)
	}

	// Exact P(G | D=Easy, I=High).
	markovFactors2, err := bn.ToMarkovFactors()
	if err != nil {
		log.Fatal(err)
	}
	ve2 := inference.NewVariableElimination(markovFactors2)
	exactCondG, err := ve2.Query([]string{"G"}, evidence)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("\nExact:")
	for i, state := range gradeStates {
		prob := exactCondG.GetValue(map[string]int{"G": i})
		fmt.Printf("  P(G=%s | D=Easy, I=High) = %.4f\n", state, prob)
	}

	// Compute and display error.
	fmt.Println("\n=== Approximation Error ===")
	maxErr := 0.0
	for i, state := range gradeStates {
		empirical := weightedCounts[i] / totalWeight
		exact := exactCondG.GetValue(map[string]int{"G": i})
		err := math.Abs(empirical - exact)
		if err > maxErr {
			maxErr = err
		}
		fmt.Printf("  |empirical - exact| for G=%s: %.4f\n", state, err)
	}
	fmt.Printf("\nMax absolute error: %.4f\n", maxErr)
}
