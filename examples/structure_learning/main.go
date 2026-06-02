// Command structure_learning demonstrates learning a Bayesian network
// structure from synthetic data using HillClimbSearch with BIC scoring.
package main

import (
	"fmt"
	"log"

	"github.com/asymmetric-effort/pgmgo/example_models"
	"github.com/asymmetric-effort/pgmgo/src/learning"
	"github.com/asymmetric-effort/pgmgo/src/sampling"
	"github.com/asymmetric-effort/pgmgo/src/structure_score"
)

func main() {
	// Build the true network: Water Sprinkler (simple 4-node network).
	trueBN := example_models.WaterSprinkler()

	fmt.Println("=== True Network Structure ===")
	trueEdges := trueBN.Edges()
	for _, e := range trueEdges {
		fmt.Printf("  %s -> %s\n", e[0], e[1])
	}

	// Generate synthetic data by forward sampling.
	sampler, err := sampling.NewBayesianModelSampling(trueBN, 42)
	if err != nil {
		log.Fatalf("Failed to create sampler: %v", err)
	}

	data, err := sampler.ForwardSample(5000)
	if err != nil {
		log.Fatalf("Failed to generate samples: %v", err)
	}
	fmt.Printf("\nGenerated %d samples with columns: %v\n", data.Len(), data.Columns())

	// Run HillClimbSearch with BIC scoring.
	bic := structure_score.NewBIC()
	hc := learning.NewHillClimbSearch(data, bic.LocalScore)

	learnedBN, err := hc.Estimate()
	if err != nil {
		log.Fatalf("Structure learning failed: %v", err)
	}

	fmt.Println("\n=== Learned Network Structure ===")
	learnedEdges := learnedBN.Edges()
	for _, e := range learnedEdges {
		fmt.Printf("  %s -> %s\n", e[0], e[1])
	}

	// Compare with true structure.
	trueSet := make(map[[2]string]bool)
	for _, e := range trueEdges {
		trueSet[e] = true
	}

	learnedSet := make(map[[2]string]bool)
	for _, e := range learnedEdges {
		learnedSet[e] = true
	}

	fmt.Println("\n=== Comparison ===")
	// Edges in true network that were recovered.
	recovered := 0
	for e := range trueSet {
		if learnedSet[e] {
			recovered++
			fmt.Printf("  RECOVERED: %s -> %s\n", e[0], e[1])
		} else {
			// Check if the reverse was learned (correct skeleton, wrong direction).
			rev := [2]string{e[1], e[0]}
			if learnedSet[rev] {
				fmt.Printf("  REVERSED:  %s -> %s (learned as %s -> %s)\n", e[0], e[1], e[1], e[0])
			} else {
				fmt.Printf("  MISSING:   %s -> %s\n", e[0], e[1])
			}
		}
	}

	// Extra edges in learned network.
	for e := range learnedSet {
		if !trueSet[e] {
			rev := [2]string{e[1], e[0]}
			if !trueSet[rev] {
				fmt.Printf("  EXTRA:     %s -> %s\n", e[0], e[1])
			}
		}
	}

	fmt.Printf("\nRecovered %d/%d true edges exactly\n", recovered, len(trueEdges))
}
