// Command causal_inference demonstrates causal reasoning with a Bayesian
// network, showing the difference between observational P(Y|X=1) and
// interventional P(Y|do(X=1)) queries, and computing the ATE.
package main

import (
	"fmt"
	"log"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// buildConfoundedNetwork creates a simple network with confounding:
//
//	Z -> X, Z -> Y, X -> Y
//
// Z is a confounder: it causes both the treatment (X) and the outcome (Y).
//
//	Z: {0, 1} — confounder (e.g., socioeconomic status)
//	X: {0, 1} — treatment (e.g., exercise)
//	Y: {0, 1} — outcome (e.g., health)
func buildConfoundedNetwork() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"X", "Y", "Z"} {
		if err := bn.AddNode(node); err != nil {
			log.Fatal(err)
		}
	}
	if err := bn.AddEdge("Z", "X"); err != nil {
		log.Fatal(err)
	}
	if err := bn.AddEdge("Z", "Y"); err != nil {
		log.Fatal(err)
	}
	if err := bn.AddEdge("X", "Y"); err != nil {
		log.Fatal(err)
	}

	if err := bn.SetStates("Z", []string{"Low", "High"}); err != nil {
		log.Fatal(err)
	}
	if err := bn.SetStates("X", []string{"No", "Yes"}); err != nil {
		log.Fatal(err)
	}
	if err := bn.SetStates("Y", []string{"Bad", "Good"}); err != nil {
		log.Fatal(err)
	}

	// P(Z)
	cpdZ, err := factors.NewTabularCPD("Z", 2, [][]float64{
		{0.5}, // Low
		{0.5}, // High
	}, nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	if err := bn.AddCPD(cpdZ); err != nil {
		log.Fatal(err)
	}

	// P(X | Z): High SES people are more likely to exercise
	cpdX, err := factors.NewTabularCPD("X", 2, [][]float64{
		{0.8, 0.2}, // No exercise
		{0.2, 0.8}, // Yes exercise
	}, []string{"Z"}, []int{2})
	if err != nil {
		log.Fatal(err)
	}
	if err := bn.AddCPD(cpdX); err != nil {
		log.Fatal(err)
	}

	// P(Y | X, Z): Both exercise and high SES improve health
	// Columns: X=No,Z=Low | X=No,Z=High | X=Yes,Z=Low | X=Yes,Z=High
	cpdY, err := factors.NewTabularCPD("Y", 2, [][]float64{
		{0.9, 0.5, 0.6, 0.2}, // Bad health
		{0.1, 0.5, 0.4, 0.8}, // Good health
	}, []string{"X", "Z"}, []int{2, 2})
	if err != nil {
		log.Fatal(err)
	}
	if err := bn.AddCPD(cpdY); err != nil {
		log.Fatal(err)
	}

	if err := bn.CheckModel(); err != nil {
		log.Fatalf("Model validation failed: %v", err)
	}
	return bn
}

func main() {
	bn := buildConfoundedNetwork()

	fmt.Println("=== Confounded Network: Z -> X, Z -> Y, X -> Y ===")
	fmt.Println("Z = confounder (socioeconomic status)")
	fmt.Println("X = treatment (exercise)")
	fmt.Println("Y = outcome (health)")
	fmt.Println()

	// 1. Observational query: P(Y | X=1) — conditioning on exercise=Yes
	markovFactors, err := bn.ToMarkovFactors()
	if err != nil {
		log.Fatal(err)
	}
	ve := inference.NewVariableElimination(markovFactors)
	obsResult, err := ve.Query([]string{"Y"}, map[string]int{"X": 1})
	if err != nil {
		log.Fatalf("Observational query failed: %v", err)
	}

	fmt.Println("=== Observational: P(Y | X=Yes) ===")
	fmt.Printf("  P(Y=Bad  | X=Yes) = %.4f\n", obsResult.GetValue(map[string]int{"Y": 0}))
	fmt.Printf("  P(Y=Good | X=Yes) = %.4f\n", obsResult.GetValue(map[string]int{"Y": 1}))
	fmt.Println()

	// 2. Interventional query: P(Y | do(X=1)) — using do-calculus
	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		log.Fatal(err)
	}
	doResult, err := ci.Query([]string{"Y"}, map[string]int{"X": 1}, nil)
	if err != nil {
		log.Fatalf("Interventional query failed: %v", err)
	}

	fmt.Println("=== Interventional: P(Y | do(X=Yes)) ===")
	fmt.Printf("  P(Y=Bad  | do(X=Yes)) = %.4f\n", doResult.GetValue(map[string]int{"Y": 0}))
	fmt.Printf("  P(Y=Good | do(X=Yes)) = %.4f\n", doResult.GetValue(map[string]int{"Y": 1}))
	fmt.Println()

	// 3. Also show P(Y | do(X=0)) for comparison
	doResult0, err := ci.Query([]string{"Y"}, map[string]int{"X": 0}, nil)
	if err != nil {
		log.Fatalf("Interventional query failed: %v", err)
	}
	fmt.Println("=== Interventional: P(Y | do(X=No)) ===")
	fmt.Printf("  P(Y=Bad  | do(X=No)) = %.4f\n", doResult0.GetValue(map[string]int{"Y": 0}))
	fmt.Printf("  P(Y=Good | do(X=No)) = %.4f\n", doResult0.GetValue(map[string]int{"Y": 1}))
	fmt.Println()

	// 4. Compute Average Treatment Effect (ATE)
	ate, err := ci.ATE("X", "Y", [2]int{0, 1})
	if err != nil {
		log.Fatalf("ATE computation failed: %v", err)
	}
	fmt.Println("=== Average Treatment Effect ===")
	fmt.Printf("  ATE = E[Y | do(X=Yes)] - E[Y | do(X=No)] = %.4f\n", ate)
	fmt.Println()

	// Explain the difference
	fmt.Println("=== Explanation ===")
	fmt.Println("The observational P(Y=Good | X=Yes) is higher than the interventional")
	fmt.Println("P(Y=Good | do(X=Yes)) because of confounding: people who exercise (X=Yes)")
	fmt.Println("tend to have higher socioeconomic status (Z=High), which independently")
	fmt.Println("improves health. The do-operator removes this confounding bias by")
	fmt.Println("intervening on X directly, cutting the Z->X edge.")
}
