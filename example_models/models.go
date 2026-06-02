// Package example_models provides factory functions that return fully
// parameterized, well-known Bayesian networks from the literature.
// Each function creates the network structure, adds CPDs with real
// probability values, sets state names, and validates the model.
package example_models

import (
	"log"

	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/models"
)

// mustCPD is a helper that panics on error (used only in factory functions
// where the parameters are known to be correct at compile time).
func mustCPD(variable string, variableCard int, values [][]float64,
	evidence []string, evidenceCard []int) *factors.TabularCPD {
	cpd, err := factors.NewTabularCPD(variable, variableCard, values, evidence, evidenceCard)
	if err != nil {
		log.Panicf("example_models: failed to create CPD for %s: %v", variable, err)
	}
	return cpd
}

// mustAdd is a helper that panics on AddNode/AddEdge/AddCPD/SetStates errors.
func must(err error) {
	if err != nil {
		log.Panicf("example_models: %v", err)
	}
}

// Student returns the classic Student Bayesian network with 5 nodes:
//
//	D (Difficulty): 2 states {Easy, Hard}
//	I (Intelligence): 2 states {Low, High}
//	G (Grade): 3 states {A, B, C}
//	L (Letter): 2 states {Weak, Strong}
//	S (SAT): 2 states {Low, High}
//
// Edges: D->G, I->G, I->S, G->L
// CPD values from Koller & Friedman (2009).
func Student() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"D", "G", "I", "L", "S"} {
		must(bn.AddNode(node))
	}
	must(bn.AddEdge("D", "G"))
	must(bn.AddEdge("I", "G"))
	must(bn.AddEdge("I", "S"))
	must(bn.AddEdge("G", "L"))

	must(bn.SetStates("D", []string{"Easy", "Hard"}))
	must(bn.SetStates("I", []string{"Low", "High"}))
	must(bn.SetStates("G", []string{"A", "B", "C"}))
	must(bn.SetStates("L", []string{"Weak", "Strong"}))
	must(bn.SetStates("S", []string{"Low", "High"}))

	// P(D)
	must(bn.AddCPD(mustCPD("D", 2, [][]float64{
		{0.6}, // Easy
		{0.4}, // Hard
	}, nil, nil)))

	// P(I)
	must(bn.AddCPD(mustCPD("I", 2, [][]float64{
		{0.7}, // Low
		{0.3}, // High
	}, nil, nil)))

	// P(G | D, I) — columns: D=Easy,I=Low | D=Easy,I=High | D=Hard,I=Low | D=Hard,I=High
	must(bn.AddCPD(mustCPD("G", 3, [][]float64{
		{0.3, 0.9, 0.05, 0.5},  // A
		{0.4, 0.08, 0.25, 0.3}, // B
		{0.3, 0.02, 0.7, 0.2},  // C
	}, []string{"D", "I"}, []int{2, 2})))

	// P(L | G) — columns: G=A | G=B | G=C
	must(bn.AddCPD(mustCPD("L", 2, [][]float64{
		{0.1, 0.4, 0.99}, // Weak
		{0.9, 0.6, 0.01}, // Strong
	}, []string{"G"}, []int{3})))

	// P(S | I) — columns: I=Low | I=High
	must(bn.AddCPD(mustCPD("S", 2, [][]float64{
		{0.95, 0.2}, // Low
		{0.05, 0.8}, // High
	}, []string{"I"}, []int{2})))

	if err := bn.CheckModel(); err != nil {
		log.Panicf("example_models: Student network validation failed: %v", err)
	}
	return bn
}

// Asia returns the Asia (chest clinic) Bayesian network with 8 nodes.
// This network was introduced by Lauritzen & Spiegelhalter (1988).
//
// Nodes:
//
//	Asia (A): visit to Asia — {No, Yes}
//	Tub (T): tuberculosis — {No, Yes}
//	Smoke (S): smoker — {No, Yes}
//	Lung (L): lung cancer — {No, Yes}
//	Bronc (B): bronchitis — {No, Yes}
//	Either (E): tuberculosis or lung cancer — {No, Yes}
//	Xray (X): positive X-ray — {No, Yes}
//	Dysp (D): dyspnoea — {No, Yes}
//
// Edges: A->T, S->L, S->B, T->E, L->E, E->X, E->D, B->D
func Asia() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"Asia", "Bronc", "Dysp", "Either", "Lung", "Smoke", "Tub", "Xray"} {
		must(bn.AddNode(node))
	}
	must(bn.AddEdge("Asia", "Tub"))
	must(bn.AddEdge("Smoke", "Lung"))
	must(bn.AddEdge("Smoke", "Bronc"))
	must(bn.AddEdge("Tub", "Either"))
	must(bn.AddEdge("Lung", "Either"))
	must(bn.AddEdge("Either", "Xray"))
	must(bn.AddEdge("Either", "Dysp"))
	must(bn.AddEdge("Bronc", "Dysp"))

	must(bn.SetStates("Asia", []string{"No", "Yes"}))
	must(bn.SetStates("Tub", []string{"No", "Yes"}))
	must(bn.SetStates("Smoke", []string{"No", "Yes"}))
	must(bn.SetStates("Lung", []string{"No", "Yes"}))
	must(bn.SetStates("Bronc", []string{"No", "Yes"}))
	must(bn.SetStates("Either", []string{"No", "Yes"}))
	must(bn.SetStates("Xray", []string{"No", "Yes"}))
	must(bn.SetStates("Dysp", []string{"No", "Yes"}))

	// P(Asia)
	must(bn.AddCPD(mustCPD("Asia", 2, [][]float64{
		{0.99}, // No
		{0.01}, // Yes
	}, nil, nil)))

	// P(Smoke)
	must(bn.AddCPD(mustCPD("Smoke", 2, [][]float64{
		{0.5}, // No
		{0.5}, // Yes
	}, nil, nil)))

	// P(Tub | Asia)
	must(bn.AddCPD(mustCPD("Tub", 2, [][]float64{
		{0.99, 0.95}, // No
		{0.01, 0.05}, // Yes
	}, []string{"Asia"}, []int{2})))

	// P(Lung | Smoke)
	must(bn.AddCPD(mustCPD("Lung", 2, [][]float64{
		{0.99, 0.90}, // No
		{0.01, 0.10}, // Yes
	}, []string{"Smoke"}, []int{2})))

	// P(Bronc | Smoke)
	must(bn.AddCPD(mustCPD("Bronc", 2, [][]float64{
		{0.70, 0.40}, // No
		{0.30, 0.60}, // Yes
	}, []string{"Smoke"}, []int{2})))

	// P(Either | Lung, Tub) — deterministic OR gate
	// Columns: Lung=No,Tub=No | Lung=No,Tub=Yes | Lung=Yes,Tub=No | Lung=Yes,Tub=Yes
	must(bn.AddCPD(mustCPD("Either", 2, [][]float64{
		{1.0, 0.0, 0.0, 0.0}, // No
		{0.0, 1.0, 1.0, 1.0}, // Yes
	}, []string{"Lung", "Tub"}, []int{2, 2})))

	// P(Xray | Either)
	must(bn.AddCPD(mustCPD("Xray", 2, [][]float64{
		{0.95, 0.02}, // No
		{0.05, 0.98}, // Yes
	}, []string{"Either"}, []int{2})))

	// P(Dysp | Bronc, Either)
	// Columns: Bronc=No,Either=No | Bronc=No,Either=Yes | Bronc=Yes,Either=No | Bronc=Yes,Either=Yes
	must(bn.AddCPD(mustCPD("Dysp", 2, [][]float64{
		{0.90, 0.30, 0.20, 0.10}, // No
		{0.10, 0.70, 0.80, 0.90}, // Yes
	}, []string{"Bronc", "Either"}, []int{2, 2})))

	if err := bn.CheckModel(); err != nil {
		log.Panicf("example_models: Asia network validation failed: %v", err)
	}
	return bn
}

// Alarm returns a simplified Alarm Bayesian network with 5 nodes.
// This is a compact version of the classic Burglary-Alarm network
// from Pearl (1988).
//
// Nodes:
//
//	Burglary (B): {No, Yes}
//	Earthquake (E): {No, Yes}
//	Alarm (A): {No, Yes}
//	JohnCalls (J): {No, Yes}
//	MaryCalls (M): {No, Yes}
//
// Edges: B->A, E->A, A->J, A->M
func Alarm() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"Alarm", "Burglary", "Earthquake", "JohnCalls", "MaryCalls"} {
		must(bn.AddNode(node))
	}
	must(bn.AddEdge("Burglary", "Alarm"))
	must(bn.AddEdge("Earthquake", "Alarm"))
	must(bn.AddEdge("Alarm", "JohnCalls"))
	must(bn.AddEdge("Alarm", "MaryCalls"))

	must(bn.SetStates("Burglary", []string{"No", "Yes"}))
	must(bn.SetStates("Earthquake", []string{"No", "Yes"}))
	must(bn.SetStates("Alarm", []string{"No", "Yes"}))
	must(bn.SetStates("JohnCalls", []string{"No", "Yes"}))
	must(bn.SetStates("MaryCalls", []string{"No", "Yes"}))

	// P(Burglary)
	must(bn.AddCPD(mustCPD("Burglary", 2, [][]float64{
		{0.999}, // No
		{0.001}, // Yes
	}, nil, nil)))

	// P(Earthquake)
	must(bn.AddCPD(mustCPD("Earthquake", 2, [][]float64{
		{0.998}, // No
		{0.002}, // Yes
	}, nil, nil)))

	// P(Alarm | Burglary, Earthquake)
	// Columns: B=No,E=No | B=No,E=Yes | B=Yes,E=No | B=Yes,E=Yes
	must(bn.AddCPD(mustCPD("Alarm", 2, [][]float64{
		{0.999, 0.71, 0.06, 0.05}, // No
		{0.001, 0.29, 0.94, 0.95}, // Yes
	}, []string{"Burglary", "Earthquake"}, []int{2, 2})))

	// P(JohnCalls | Alarm)
	must(bn.AddCPD(mustCPD("JohnCalls", 2, [][]float64{
		{0.95, 0.10}, // No
		{0.05, 0.90}, // Yes
	}, []string{"Alarm"}, []int{2})))

	// P(MaryCalls | Alarm)
	must(bn.AddCPD(mustCPD("MaryCalls", 2, [][]float64{
		{0.99, 0.30}, // No
		{0.01, 0.70}, // Yes
	}, []string{"Alarm"}, []int{2})))

	if err := bn.CheckModel(); err != nil {
		log.Panicf("example_models: Alarm network validation failed: %v", err)
	}
	return bn
}

// Cancer returns the simple Cancer diagnosis Bayesian network with 5 nodes.
//
// Nodes:
//
//	Pollution (P): {Low, High}
//	Smoker (S): {No, Yes}
//	Cancer (C): {No, Yes}
//	Xray (X): {Negative, Positive}
//	Dyspnoea (D): {No, Yes}
//
// Edges: P->C, S->C, C->X, C->D
func Cancer() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"Cancer", "Dyspnoea", "Pollution", "Smoker", "Xray"} {
		must(bn.AddNode(node))
	}
	must(bn.AddEdge("Pollution", "Cancer"))
	must(bn.AddEdge("Smoker", "Cancer"))
	must(bn.AddEdge("Cancer", "Xray"))
	must(bn.AddEdge("Cancer", "Dyspnoea"))

	must(bn.SetStates("Pollution", []string{"Low", "High"}))
	must(bn.SetStates("Smoker", []string{"No", "Yes"}))
	must(bn.SetStates("Cancer", []string{"No", "Yes"}))
	must(bn.SetStates("Xray", []string{"Negative", "Positive"}))
	must(bn.SetStates("Dyspnoea", []string{"No", "Yes"}))

	// P(Pollution)
	must(bn.AddCPD(mustCPD("Pollution", 2, [][]float64{
		{0.9}, // Low
		{0.1}, // High
	}, nil, nil)))

	// P(Smoker)
	must(bn.AddCPD(mustCPD("Smoker", 2, [][]float64{
		{0.7}, // No
		{0.3}, // Yes
	}, nil, nil)))

	// P(Cancer | Pollution, Smoker)
	// Columns: P=Low,S=No | P=Low,S=Yes | P=High,S=No | P=High,S=Yes
	must(bn.AddCPD(mustCPD("Cancer", 2, [][]float64{
		{0.999, 0.97, 0.95, 0.92}, // No
		{0.001, 0.03, 0.05, 0.08}, // Yes
	}, []string{"Pollution", "Smoker"}, []int{2, 2})))

	// P(Xray | Cancer)
	must(bn.AddCPD(mustCPD("Xray", 2, [][]float64{
		{0.80, 0.10}, // Negative
		{0.20, 0.90}, // Positive
	}, []string{"Cancer"}, []int{2})))

	// P(Dyspnoea | Cancer)
	must(bn.AddCPD(mustCPD("Dyspnoea", 2, [][]float64{
		{0.70, 0.35}, // No
		{0.30, 0.65}, // Yes
	}, []string{"Cancer"}, []int{2})))

	if err := bn.CheckModel(); err != nil {
		log.Panicf("example_models: Cancer network validation failed: %v", err)
	}
	return bn
}

// Sachs returns the Sachs protein signaling Bayesian network with 11 nodes.
// This network was learned from flow cytometry data by Sachs et al. (2005).
// Only the structure is provided (no CPDs) since the full parameterization
// is too large to hardcode.
//
// Nodes: Raf, Mek, Plcg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk
//
// Note: CheckModel() will fail on this network because CPDs are not set.
// Use it for structure-only tasks such as structure learning evaluation.
func Sachs() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	nodes := []string{"Akt", "Erk", "Jnk", "Mek", "P38", "PIP2", "PIP3", "PKA", "PKC", "Plcg", "Raf"}
	for _, node := range nodes {
		must(bn.AddNode(node))
	}

	// Set state names (3 discretization levels as in Sachs et al.)
	for _, node := range nodes {
		must(bn.SetStates(node, []string{"Low", "Medium", "High"}))
	}

	// Edges from Sachs et al. (2005) consensus network
	edges := [][2]string{
		{"PKC", "Raf"},
		{"PKC", "Mek"},
		{"PKC", "PKA"},
		{"PKC", "Jnk"},
		{"PKC", "P38"},
		{"PKA", "Raf"},
		{"PKA", "Mek"},
		{"PKA", "Erk"},
		{"PKA", "Akt"},
		{"PKA", "Jnk"},
		{"PKA", "P38"},
		{"Raf", "Mek"},
		{"Mek", "Erk"},
		{"Erk", "Akt"},
		{"Plcg", "PIP2"},
		{"Plcg", "PIP3"},
		{"PIP3", "PIP2"},
	}
	for _, e := range edges {
		must(bn.AddEdge(e[0], e[1]))
	}

	return bn
}

// WaterSprinkler returns the classic Rain/Sprinkler/WetGrass Bayesian network.
//
// Nodes:
//
//	Cloudy (C): {No, Yes}
//	Sprinkler (S): {Off, On}
//	Rain (R): {No, Yes}
//	WetGrass (W): {No, Yes}
//
// Edges: Cloudy->Sprinkler, Cloudy->Rain, Sprinkler->WetGrass, Rain->WetGrass
func WaterSprinkler() *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()

	for _, node := range []string{"Cloudy", "Rain", "Sprinkler", "WetGrass"} {
		must(bn.AddNode(node))
	}
	must(bn.AddEdge("Cloudy", "Sprinkler"))
	must(bn.AddEdge("Cloudy", "Rain"))
	must(bn.AddEdge("Sprinkler", "WetGrass"))
	must(bn.AddEdge("Rain", "WetGrass"))

	must(bn.SetStates("Cloudy", []string{"No", "Yes"}))
	must(bn.SetStates("Sprinkler", []string{"Off", "On"}))
	must(bn.SetStates("Rain", []string{"No", "Yes"}))
	must(bn.SetStates("WetGrass", []string{"No", "Yes"}))

	// P(Cloudy)
	must(bn.AddCPD(mustCPD("Cloudy", 2, [][]float64{
		{0.5}, // No
		{0.5}, // Yes
	}, nil, nil)))

	// P(Sprinkler | Cloudy)
	must(bn.AddCPD(mustCPD("Sprinkler", 2, [][]float64{
		{0.5, 0.9}, // Off
		{0.5, 0.1}, // On
	}, []string{"Cloudy"}, []int{2})))

	// P(Rain | Cloudy)
	must(bn.AddCPD(mustCPD("Rain", 2, [][]float64{
		{0.8, 0.2}, // No
		{0.2, 0.8}, // Yes
	}, []string{"Cloudy"}, []int{2})))

	// P(WetGrass | Rain, Sprinkler)
	// Columns: Rain=No,S=Off | Rain=No,S=On | Rain=Yes,S=Off | Rain=Yes,S=On
	must(bn.AddCPD(mustCPD("WetGrass", 2, [][]float64{
		{1.0, 0.1, 0.1, 0.01}, // No
		{0.0, 0.9, 0.9, 0.99}, // Yes
	}, []string{"Rain", "Sprinkler"}, []int{2, 2})))

	if err := bn.CheckModel(); err != nil {
		log.Panicf("example_models: WaterSprinkler network validation failed: %v", err)
	}
	return bn
}
