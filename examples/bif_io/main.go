// Command bif_io demonstrates writing a Bayesian network to BIF format
// and reading it back, verifying the round-trip preserves structure and CPDs.
package main

import (
	"bytes"
	"fmt"
	"log"
	"math"

	"github.com/asymmetric-effort/pgmgo/example_models"
	"github.com/asymmetric-effort/pgmgo/src/readwrite"
)

func main() {
	// Build the Water Sprinkler network.
	original := example_models.WaterSprinkler()

	fmt.Println("=== Original Network ===")
	fmt.Println("Nodes:", original.Nodes())
	fmt.Println("Edges:")
	for _, e := range original.Edges() {
		fmt.Printf("  %s -> %s\n", e[0], e[1])
	}

	// Write to BIF format (in-memory buffer).
	var buf bytes.Buffer
	if err := readwrite.WriteBIF(&buf, original); err != nil {
		log.Fatalf("WriteBIF failed: %v", err)
	}

	bifContent := buf.String()
	fmt.Println("\n=== BIF Output ===")
	fmt.Println(bifContent)

	// Read back from BIF.
	restored, err := readwrite.ReadBIF(bytes.NewReader(buf.Bytes()))
	if err != nil {
		log.Fatalf("ReadBIF failed: %v", err)
	}

	// Verify round-trip.
	fmt.Println("=== Round-Trip Verification ===")

	// Check nodes.
	origNodes := original.Nodes()
	restNodes := restored.Nodes()
	nodesMatch := len(origNodes) == len(restNodes)
	if nodesMatch {
		for i := range origNodes {
			if origNodes[i] != restNodes[i] {
				nodesMatch = false
				break
			}
		}
	}
	fmt.Printf("Nodes match: %v (%v)\n", nodesMatch, restNodes)

	// Check edges.
	origEdges := original.Edges()
	restEdges := restored.Edges()
	edgesMatch := len(origEdges) == len(restEdges)
	if edgesMatch {
		for i := range origEdges {
			if origEdges[i] != restEdges[i] {
				edgesMatch = false
				break
			}
		}
	}
	fmt.Printf("Edges match: %v\n", edgesMatch)

	// Check CPD values.
	cpdsMatch := true
	const tol = 1e-6
	for _, node := range origNodes {
		origCPD := original.GetCPD(node)
		restCPD := restored.GetCPD(node)
		if restCPD == nil {
			fmt.Printf("  CPD for %s: MISSING in restored\n", node)
			cpdsMatch = false
			continue
		}

		origData := origCPD.ToFactor().Values().Data()
		restData := restCPD.ToFactor().Values().Data()

		if len(origData) != len(restData) {
			fmt.Printf("  CPD for %s: size mismatch (%d vs %d)\n", node, len(origData), len(restData))
			cpdsMatch = false
			continue
		}

		maxDiff := 0.0
		for i := range origData {
			diff := math.Abs(origData[i] - restData[i])
			if diff > maxDiff {
				maxDiff = diff
			}
		}

		ok := maxDiff < tol
		if !ok {
			cpdsMatch = false
		}
		fmt.Printf("  CPD for %s: max diff = %.2e [%v]\n", node, maxDiff, ok)
	}
	fmt.Printf("All CPDs match: %v\n", cpdsMatch)

	// Validate restored model.
	if err := restored.CheckModel(); err != nil {
		fmt.Printf("Restored model validation: FAILED (%v)\n", err)
	} else {
		fmt.Println("Restored model validation: PASSED")
	}
}
