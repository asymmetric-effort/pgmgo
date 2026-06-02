// Command basic_bn demonstrates building a Bayesian network, adding CPDs,
// validating the model, and running a variable elimination query.
package main

import (
	"fmt"
	"log"

	"github.com/asymmetric-effort/pgmgo/example_models"
	"github.com/asymmetric-effort/pgmgo/src/inference"
)

func main() {
	// Build the classic Student network (D, I, G, L, S).
	bn := example_models.Student()

	fmt.Println("=== Student Bayesian Network ===")
	fmt.Println("Nodes:", bn.Nodes())
	fmt.Println("Edges:")
	for _, e := range bn.Edges() {
		fmt.Printf("  %s -> %s\n", e[0], e[1])
	}

	// Validate the model.
	if err := bn.CheckModel(); err != nil {
		log.Fatalf("Model validation failed: %v", err)
	}
	fmt.Println("\nModel validation: PASSED")

	// Convert to Markov factors for inference.
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		log.Fatalf("Failed to convert to Markov factors: %v", err)
	}

	// Run a Variable Elimination query: P(G | D=0, I=1)
	// D=0 means "Easy", I=1 means "High"
	ve := inference.NewVariableElimination(markovFactors)
	evidence := map[string]int{"D": 0, "I": 1}
	result, err := ve.Query([]string{"G"}, evidence)
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}

	fmt.Println("\n=== Query: P(G | D=Easy, I=High) ===")
	gradeStates := bn.GetStates("G") // ["A", "B", "C"]
	for i, state := range gradeStates {
		assignment := map[string]int{"G": i}
		prob := result.GetValue(assignment)
		fmt.Printf("  P(G=%s | D=Easy, I=High) = %.4f\n", state, prob)
	}

	// Also compute MAP assignment.
	ve2 := inference.NewVariableElimination(markovFactors)
	mapAssignment, err := ve2.MAP([]string{"G"}, evidence)
	if err != nil {
		log.Fatalf("MAP query failed: %v", err)
	}
	fmt.Printf("\nMAP assignment: G=%s\n", gradeStates[mapAssignment["G"]])
}
