package main

import (
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/asymmetric-effort/pgmgo/lib/graphgo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/ci_tests"
	"github.com/asymmetric-effort/pgmgo/src/factors"
	"github.com/asymmetric-effort/pgmgo/src/inference"
	"github.com/asymmetric-effort/pgmgo/src/learning"
	"github.com/asymmetric-effort/pgmgo/src/metrics"
	"github.com/asymmetric-effort/pgmgo/src/models"
	"github.com/asymmetric-effort/pgmgo/src/readwrite"
	"github.com/asymmetric-effort/pgmgo/src/sampling"
	"github.com/asymmetric-effort/pgmgo/src/structure_score"
)

// bifWriter abstracts BIF model serialization to enable testing of write error paths.
type bifWriter interface {
	Write(w io.Writer, bn *models.BayesianNetwork) error
}

// bifWriterFunc adapts a function to the bifWriter interface.
type bifWriterFunc func(w io.Writer, bn *models.BayesianNetwork) error

func (f bifWriterFunc) Write(w io.Writer, bn *models.BayesianNetwork) error {
	return f(w, bn)
}

// defaultBIFWriter is the production bifWriter using readwrite.WriteBIF.
var defaultBIFWriter bifWriter = bifWriterFunc(readwrite.WriteBIF)

// formatWriterMap maps format names to bifWriter implementations for convert output.
var formatWriterMap = map[string]bifWriter{
	"bif":    bifWriterFunc(readwrite.WriteBIF),
	"xmlbif": bifWriterFunc(readwrite.WriteXMLBIF),
	"net":    bifWriterFunc(readwrite.WriteNET),
	"uai":    bifWriterFunc(readwrite.WriteUAI),
	"xdsl":   bifWriterFunc(readwrite.WriteXDSL),
}

// columnLookup abstracts DataFrame column access to enable testing of nil-column paths.
type columnLookup interface {
	Column(name string) *tabgo.Series
}

// dataFrameColumnLookup wraps a *tabgo.DataFrame and returns nil instead of panicking
// when a column is not found.
type dataFrameColumnLookup struct {
	df *tabgo.DataFrame
}

func (d *dataFrameColumnLookup) Column(name string) *tabgo.Series {
	defer func() { recover() }()
	return d.df.Column(name)
}

const version = "0.0.45"

func main() {
	os.Exit(run(os.Args))
}

// run is the testable core of main(). It processes command-line
// arguments and returns an exit code instead of calling os.Exit.
func run(args []string) int {
	if len(args) < 2 {
		printUsage()
		return 0
	}

	switch args[1] {
	case "version", "--version", "-v":
		fmt.Printf("pgmgo %s\n", version)
		return 0
	case "help", "--help", "-h":
		printUsage()
		return 0
	case "validate":
		return runValidate(args[2:])
	case "query":
		return runQuery(args[2:])
	case "map":
		return runMAP(args[2:])
	case "learn":
		return runLearn(args[2:])
	case "fit":
		return runFit(args[2:])
	case "sample":
		return runSample(args[2:])
	case "info":
		return runInfo(args[2:])
	case "convert":
		return runConvert(args[2:])
	case "compare":
		return runCompare(args[2:])
	case "do":
		return runDo(args[2:])
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", args[1])
		printUsage()
		return 1
	}
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

// runValidate validates a BIF model file. It returns an exit code:
//
//	0 on success, 1 on error, 2 on invalid input.
func runValidate(args []string) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "error: validate requires a file argument")
		fmt.Fprintln(os.Stderr, "usage: pgmgo validate <file>")
		return 2
	}

	filePath := args[0]
	return validateBIFFile(filePath)
}

// validateBIFFile opens and validates a BIF file, printing results to stdout/stderr.
// Returns 0 on success, 1 on validation error, 2 on invalid input (e.g., file not found).
func validateBIFFile(filePath string) int {
	f, err := os.Open(filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot open file: %v\n", err)
		return 2
	}
	defer f.Close()

	bn, err := readwrite.ReadBIF(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to parse BIF file: %v\n", err)
		return 1
	}

	if err := bn.CheckModel(); err != nil {
		fmt.Fprintf(os.Stderr, "error: model validation failed: %v\n", err)
		return 1
	}

	fmt.Printf("model %s is valid (%d nodes, %d edges)\n", filePath, len(bn.Nodes()), len(bn.Edges()))
	return 0
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

func runQuery(args []string) int {
	fs := flag.NewFlagSet("query", flag.ContinueOnError)
	variables := fs.String("variables", "", "comma-separated query variables")
	evidence := fs.String("evidence", "", "comma-separated evidence E1=v1,E2=v2")
	method := fs.String("method", "ve", "inference method: ve, bp, approx")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "error: query requires a file argument")
		fmt.Fprintln(os.Stderr, "usage: pgmgo query <file> --variables V1,V2 [--evidence E1=v1] [--method ve|bp|approx]")
		return 2
	}
	filePath := fs.Arg(0)
	queryVars := parseCSVList(*variables)
	if len(queryVars) == 0 {
		fmt.Fprintln(os.Stderr, "error: --variables is required")
		return 2
	}
	evidenceMap, err := parseEvidenceMap(*evidence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: bad --evidence: %v\n", err)
		return 2
	}
	return queryBIF(filePath, queryVars, evidenceMap, *method)
}

func queryBIF(filePath string, queryVars []string, evidence map[string]int, method string) int {
	bn, code := loadBIF(filePath)
	if code != 0 {
		return code
	}

	var result *factors.DiscreteFactor
	var err error

	switch method {
	case "ve":
		result, err = queryVE(bn, queryVars, evidence)
	case "bp":
		result, err = queryBP(bn, queryVars, evidence)
	case "approx":
		result, err = queryApprox(bn, queryVars, evidence)
	default:
		fmt.Fprintf(os.Stderr, "error: unknown method %q (use ve, bp, approx)\n", method)
		return 2
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: inference failed: %v\n", err)
		return 1
	}

	printFactor(result)
	return 0
}

func queryVE(bn *models.BayesianNetwork, queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	facs, err := bn.ToMarkovFactors()
	if err != nil {
		return nil, err
	}
	ve := inference.NewVariableElimination(facs)
	return ve.Query(queryVars, evidence)
}

func queryBP(bn *models.BayesianNetwork, queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	jt, err := bn.ToJunctionTree()
	if err != nil {
		return nil, err
	}
	cliques := jt.Cliques()
	separators := jt.SeparatorSets()
	cliqueFactors := make(map[int][]*factors.DiscreteFactor)
	for i, c := range cliques {
		fs := jt.GetCliqueFactors(c)
		if len(fs) > 0 {
			cliqueFactors[i] = fs
		}
	}
	bp := inference.NewBeliefPropagation(cliques, separators, cliqueFactors)
	if err := bp.Calibrate(); err != nil {
		return nil, err
	}
	return bp.Query(queryVars, evidence)
}

func queryApprox(bn *models.BayesianNetwork, queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	facs, err := bn.ToMarkovFactors()
	if err != nil {
		return nil, err
	}
	ai := inference.NewApproxInference(facs, 0)
	return ai.Query(queryVars, evidence, 10000)
}

// ---------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------

func runMAP(args []string) int {
	fs := flag.NewFlagSet("map", flag.ContinueOnError)
	variables := fs.String("variables", "", "comma-separated query variables")
	evidence := fs.String("evidence", "", "comma-separated evidence E1=v1,E2=v2")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "error: map requires a file argument")
		fmt.Fprintln(os.Stderr, "usage: pgmgo map <file> --variables V1,V2 [--evidence E1=v1]")
		return 2
	}
	filePath := fs.Arg(0)
	queryVars := parseCSVList(*variables)
	if len(queryVars) == 0 {
		fmt.Fprintln(os.Stderr, "error: --variables is required")
		return 2
	}
	evidenceMap, err := parseEvidenceMap(*evidence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: bad --evidence: %v\n", err)
		return 2
	}
	return mapBIF(filePath, queryVars, evidenceMap)
}

func mapBIF(filePath string, queryVars []string, evidence map[string]int) int {
	bn, code := loadBIF(filePath)
	if code != 0 {
		return code
	}

	facs, err := bn.ToMarkovFactors()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return 1
	}
	ve := inference.NewVariableElimination(facs)
	result, err := ve.MAP(queryVars, evidence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: MAP inference failed: %v\n", err)
		return 1
	}

	fmt.Println("MAP assignment:")
	keys := make([]string, 0, len(result))
	for k := range result {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Printf("  %s = %d\n", k, result[k])
	}
	return 0
}

// ---------------------------------------------------------------------------
// learn
// ---------------------------------------------------------------------------

func runLearn(args []string) int {
	fs := flag.NewFlagSet("learn", flag.ContinueOnError)
	dataPath := fs.String("data", "", "path to CSV data file")
	method := fs.String("method", "hillclimb", "structure learning method: hillclimb, pc, ges, exhaustive, tree")
	scoreName := fs.String("score", "bic", "scoring function: bic, bdeu, k2 (for score-based methods)")
	significance := fs.Float64("significance", 0.05, "significance level for constraint-based methods")
	output := fs.String("output", "", "output BIF file path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *dataPath == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "error: --data and --output are required")
		fmt.Fprintln(os.Stderr, "usage: pgmgo learn --data <csv> --method <method> --output <bif>")
		return 2
	}
	return learnStructure(*dataPath, *method, *scoreName, *significance, *output)
}

func learnStructure(dataPath, method, scoreName string, significance float64, output string) int {
	data, err := tabgo.ReadCSV(dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot read CSV: %v\n", err)
		return 2
	}

	scoreFn := getScoreFunc(scoreName)

	var bn *models.BayesianNetwork

	switch method {
	case "hillclimb":
		hc := learning.NewHillClimbSearch(data, scoreFn)
		bn, err = hc.Estimate()
	case "pc":
		ciTest := adaptCITest(ci_tests.ChiSquare)
		pc := learning.NewPC(data, ciTest, significance)
		bn, err = pc.EstimateBN()
	case "ges":
		ges := learning.NewGES(data, scoreFn)
		pdag, gerr := ges.Estimate()
		if gerr != nil {
			err = gerr
			break
		}
		bn = pdagToBN(pdag)
	case "exhaustive":
		es := learning.NewExhaustiveSearch(data, scoreFn)
		bn, err = es.Estimate()
	case "tree":
		ts := learning.NewTreeSearch(data)
		bn, err = ts.Estimate()
	default:
		fmt.Fprintf(os.Stderr, "error: unknown method %q\n", method)
		return 2
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: structure learning failed: %v\n", err)
		return 1
	}

	// Set state names from data for each node so WriteBIF succeeds.
	setStatesFromData(bn, data)

	return learnStructureFinalize(bn, data, output)
}

// learnStructureFinalize is the testable implementation of the MLE fit and write
// step. Accepts a BN and data for mock injection of MLE failure paths.
func learnStructureFinalize(bn *models.BayesianNetwork, data *tabgo.DataFrame, output string) int {
	return learnStructureFinalizeImpl(bn, data, output, defaultBIFWriter)
}

// learnStructureFinalizeImpl is the testable implementation of learnStructureFinalize.
// Accepts a bifWriter for mock injection of write failures.
func learnStructureFinalizeImpl(bn *models.BayesianNetwork, data *tabgo.DataFrame, output string, writer bifWriter) int {
	// Fit MLE parameters so the output file has CPDs.
	mle := learning.NewMLE(bn, data)
	if mleErr := mle.Estimate(); mleErr != nil {
		// If MLE fails (e.g. insufficient data), generate uniform CPDs.
		nStates := 2
		for _, node := range bn.Nodes() {
			states := bn.GetStates(node)
			if len(states) > nStates {
				nStates = len(states)
			}
		}
		if rErr := bn.GetRandomCPDs(nStates, 0); rErr != nil {
			fmt.Fprintf(os.Stderr, "error: cannot generate CPDs: %v\n", rErr)
			return 1
		}
	}

	if err := writeBIFFileImpl(output, bn, writer); err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot write output: %v\n", err)
		return 1
	}

	fmt.Printf("learned structure with %d nodes, %d edges -> %s\n", len(bn.Nodes()), len(bn.Edges()), output)
	return 0
}

// ---------------------------------------------------------------------------
// fit
// ---------------------------------------------------------------------------

func runFit(args []string) int {
	fs := flag.NewFlagSet("fit", flag.ContinueOnError)
	modelPath := fs.String("model", "", "input BIF model file (structure)")
	dataPath := fs.String("data", "", "path to CSV data file")
	method := fs.String("method", "mle", "parameter learning method: mle, bayesian, em")
	output := fs.String("output", "", "output BIF file path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *modelPath == "" || *dataPath == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "error: --model, --data, and --output are required")
		fmt.Fprintln(os.Stderr, "usage: pgmgo fit --model <bif> --data <csv> --method <method> --output <bif>")
		return 2
	}
	return fitParameters(*modelPath, *dataPath, *method, *output)
}

func fitParameters(modelPath, dataPath, method, output string) int {
	bn, code := loadBIF(modelPath)
	if code != 0 {
		return code
	}

	data, err := tabgo.ReadCSV(dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot read CSV: %v\n", err)
		return 2
	}

	switch method {
	case "mle":
		mle := learning.NewMLE(bn, data)
		err = mle.Estimate()
	case "bayesian":
		be := learning.NewBayesianEstimator(bn, data, learning.BDeu, 5.0)
		err = be.Estimate()
	case "em":
		em := learning.NewEM(bn, data, nil, 100, 1e-6)
		err = em.Estimate()
	default:
		fmt.Fprintf(os.Stderr, "error: unknown method %q (use mle, bayesian, em)\n", method)
		return 2
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: parameter learning failed: %v\n", err)
		return 1
	}

	if err := writeBIFFile(output, bn); err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot write output: %v\n", err)
		return 1
	}

	fmt.Printf("fitted parameters for %d nodes -> %s\n", len(bn.Nodes()), output)
	return 0
}

// ---------------------------------------------------------------------------
// sample
// ---------------------------------------------------------------------------

func runSample(args []string) int {
	fs := flag.NewFlagSet("sample", flag.ContinueOnError)
	modelPath := fs.String("model", "", "input BIF model file")
	n := fs.Int("n", 100, "number of samples to generate")
	output := fs.String("output", "", "output CSV file path")
	method := fs.String("method", "forward", "sampling method: forward, rejection, gibbs")
	evidence := fs.String("evidence", "", "comma-separated evidence E1=v1,E2=v2 (for rejection/gibbs)")
	seed := fs.Int64("seed", 0, "random seed (0 = non-deterministic)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *modelPath == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "error: --model and --output are required")
		fmt.Fprintln(os.Stderr, "usage: pgmgo sample --model <bif> --n <count> --output <csv>")
		return 2
	}
	evidenceMap, err := parseEvidenceMap(*evidence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: bad --evidence: %v\n", err)
		return 2
	}
	if *seed == 0 {
		*seed = rand.Int63()
	}
	return sampleModel(*modelPath, *n, *output, *method, evidenceMap, *seed)
}

func sampleModel(modelPath string, n int, output, method string, evidence map[string]int, seed int64) int {
	bn, code := loadBIF(modelPath)
	if code != 0 {
		return code
	}

	var df *tabgo.DataFrame
	var err error

	switch method {
	case "forward":
		bms, serr := sampling.NewBayesianModelSampling(bn, seed)
		if serr != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", serr)
			return 1
		}
		df, err = bms.ForwardSample(n)
	case "rejection":
		bms, serr := sampling.NewBayesianModelSampling(bn, seed)
		if serr != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", serr)
			return 1
		}
		df, err = bms.RejectionSample(n, evidence)
	case "gibbs":
		gs, serr := sampling.NewGibbsSampling(bn, seed)
		if serr != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", serr)
			return 1
		}
		df, err = gs.Sample(n, 100, 1, evidence)
	default:
		fmt.Fprintf(os.Stderr, "error: unknown method %q (use forward, rejection, gibbs)\n", method)
		return 2
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: sampling failed: %v\n", err)
		return 1
	}

	if err := tabgo.WriteCSV(df, output); err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot write CSV: %v\n", err)
		return 1
	}

	fmt.Printf("generated %d samples -> %s\n", n, output)
	return 0
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

func runInfo(args []string) int {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "error: info requires a file argument")
		fmt.Fprintln(os.Stderr, "usage: pgmgo info <file>")
		return 2
	}
	return infoBIF(args[0])
}

func infoBIF(filePath string) int {
	bn, code := loadBIF(filePath)
	if code != 0 {
		return code
	}

	nodes := bn.Nodes()
	edges := bn.Edges()

	fmt.Printf("Model: %s\n", filePath)
	fmt.Printf("Nodes: %d\n", len(nodes))
	fmt.Printf("Edges: %d\n", len(edges))
	fmt.Println()

	fmt.Println("Node list:")
	for _, n := range nodes {
		states := bn.GetStates(n)
		fmt.Printf("  %s (states: %s)\n", n, strings.Join(states, ", "))
	}
	fmt.Println()

	fmt.Println("Edge list:")
	for _, e := range edges {
		fmt.Printf("  %s -> %s\n", e[0], e[1])
	}
	fmt.Println()

	fmt.Println("CPD summary:")
	for _, n := range nodes {
		cpd := bn.GetCPD(n)
		if cpd == nil {
			fmt.Printf("  %s: no CPD\n", n)
			continue
		}
		parents := cpd.Evidence()
		if len(parents) == 0 {
			fmt.Printf("  %s: %d states, no parents\n", n, cpd.VariableCard())
		} else {
			fmt.Printf("  %s: %d states, parents: %s\n", n, cpd.VariableCard(), strings.Join(parents, ", "))
		}
	}

	return 0
}

// ---------------------------------------------------------------------------
// convert
// ---------------------------------------------------------------------------

func runConvert(args []string) int {
	fs := flag.NewFlagSet("convert", flag.ContinueOnError)
	input := fs.String("input", "", "input file path")
	fromFmt := fs.String("from", "", "input format: bif, xmlbif, net, uai, xdsl")
	toFmt := fs.String("to", "", "output format: bif, xmlbif, net, uai, xdsl")
	output := fs.String("output", "", "output file path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *input == "" || *fromFmt == "" || *toFmt == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "error: --input, --from, --to, and --output are required")
		fmt.Fprintln(os.Stderr, "usage: pgmgo convert --input <file> --from <format> --to <format> --output <file>")
		return 2
	}
	return convertModel(*input, *fromFmt, *toFmt, *output)
}

func convertModel(input, fromFmt, toFmt, output string) int {
	return convertModelImpl(input, fromFmt, toFmt, output, formatWriterMap)
}

// convertModelImpl is the testable implementation of convertModel.
// Accepts a writer map for mock injection of write failures.
func convertModelImpl(input, fromFmt, toFmt, output string, writers map[string]bifWriter) int {
	f, err := os.Open(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot open input: %v\n", err)
		return 2
	}
	defer f.Close()

	var bn *models.BayesianNetwork
	switch fromFmt {
	case "bif":
		bn, err = readwrite.ReadBIF(f)
	case "xmlbif":
		bn, err = readwrite.ReadXMLBIF(f)
	case "net":
		bn, err = readwrite.ReadNET(f)
	case "uai":
		bn, err = readwrite.ReadUAI(f)
	case "xdsl":
		bn, err = readwrite.ReadXDSL(f)
	default:
		fmt.Fprintf(os.Stderr, "error: unknown input format %q\n", fromFmt)
		return 2
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to read %s: %v\n", fromFmt, err)
		return 1
	}

	outFile, err := os.Create(output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot create output: %v\n", err)
		return 1
	}
	defer outFile.Close()

	writer, ok := writers[toFmt]
	if !ok {
		fmt.Fprintf(os.Stderr, "error: unknown output format %q\n", toFmt)
		return 2
	}
	if err := writer.Write(outFile, bn); err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to write %s: %v\n", toFmt, err)
		return 1
	}

	fmt.Printf("converted %s (%s) -> %s (%s)\n", input, fromFmt, output, toFmt)
	return 0
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

func runCompare(args []string) int {
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	truePath := fs.String("true", "", "true model BIF file")
	estPath := fs.String("estimated", "", "estimated model BIF file")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *truePath == "" || *estPath == "" {
		fmt.Fprintln(os.Stderr, "error: --true and --estimated are required")
		fmt.Fprintln(os.Stderr, "usage: pgmgo compare --true <bif> --estimated <bif>")
		return 2
	}
	return compareModels(*truePath, *estPath)
}

func compareModels(truePath, estPath string) int {
	trueBN, code := loadBIF(truePath)
	if code != 0 {
		return code
	}
	estBN, code := loadBIF(estPath)
	if code != 0 {
		return code
	}

	trueG := bnToDigraph(trueBN)
	estG := bnToDigraph(estBN)

	shd := metrics.SHD(trueG, estG)
	aTP, aFP, aTN, aFN := metrics.AdjacencyConfusionMatrix(trueG, estG)
	oTP, oFP, oTN, oFN := metrics.OrientationConfusionMatrix(trueG, estG)

	fmt.Printf("Structural Hamming Distance (SHD): %d\n", shd)
	fmt.Println()
	fmt.Println("Adjacency confusion matrix:")
	fmt.Printf("  TP=%d  FP=%d  TN=%d  FN=%d\n", aTP, aFP, aTN, aFN)
	fmt.Println()
	fmt.Println("Orientation confusion matrix:")
	fmt.Printf("  TP=%d  FP=%d  TN=%d  FN=%d\n", oTP, oFP, oTN, oFN)

	return 0
}

// ---------------------------------------------------------------------------
// do
// ---------------------------------------------------------------------------

func runDo(args []string) int {
	fs := flag.NewFlagSet("do", flag.ContinueOnError)
	intervention := fs.String("intervention", "", "intervention X=v (comma-separated)")
	queryVar := fs.String("query", "", "query variable")
	evidence := fs.String("evidence", "", "comma-separated evidence E1=v1,E2=v2")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "error: do requires a file argument")
		fmt.Fprintln(os.Stderr, "usage: pgmgo do <file> --intervention X=v --query Y [--evidence E1=v1]")
		return 2
	}
	filePath := fs.Arg(0)
	if *intervention == "" || *queryVar == "" {
		fmt.Fprintln(os.Stderr, "error: --intervention and --query are required")
		return 2
	}
	doVars, err := parseEvidenceMap(*intervention)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: bad --intervention: %v\n", err)
		return 2
	}
	evidenceMap, err := parseEvidenceMap(*evidence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: bad --evidence: %v\n", err)
		return 2
	}
	return doCausalQuery(filePath, doVars, *queryVar, evidenceMap)
}

func doCausalQuery(filePath string, doVars map[string]int, queryVar string, evidence map[string]int) int {
	bn, code := loadBIF(filePath)
	if code != 0 {
		return code
	}

	ci, err := inference.NewCausalInference(bn)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return 1
	}

	result, err := ci.Query([]string{queryVar}, doVars, evidence)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: causal query failed: %v\n", err)
		return 1
	}

	printFactor(result)
	return 0
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

func loadBIF(filePath string) (*models.BayesianNetwork, int) {
	f, err := os.Open(filePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot open file: %v\n", err)
		return nil, 2
	}
	defer f.Close()

	bn, err := readwrite.ReadBIF(f)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to parse BIF file: %v\n", err)
		return nil, 1
	}
	return bn, 0
}

func writeBIFFile(path string, bn *models.BayesianNetwork) error {
	return writeBIFFileImpl(path, bn, defaultBIFWriter)
}

// writeBIFFileImpl is the testable implementation of writeBIFFile.
// Accepts a bifWriter for mock injection of write failures.
func writeBIFFileImpl(path string, bn *models.BayesianNetwork, writer bifWriter) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return writer.Write(f, bn)
}

func parseCSVList(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	return result
}

func parseEvidenceMap(s string) (map[string]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	result := make(map[string]int)
	pairs := strings.Split(s, ",")
	for _, pair := range pairs {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		eqIdx := strings.Index(pair, "=")
		if eqIdx < 0 {
			return nil, fmt.Errorf("invalid evidence %q (expected KEY=VALUE)", pair)
		}
		key := strings.TrimSpace(pair[:eqIdx])
		valStr := strings.TrimSpace(pair[eqIdx+1:])
		val, err := strconv.Atoi(valStr)
		if err != nil {
			return nil, fmt.Errorf("invalid evidence value %q for %q: %v", valStr, key, err)
		}
		result[key] = val
	}
	return result, nil
}

func printFactor(f *factors.DiscreteFactor) {
	vars := f.Variables()
	cards := f.Cardinality()
	total := 1
	for _, c := range cards {
		total *= c
	}

	// Print header.
	fmt.Println(strings.Join(vars, "\t") + "\tP")

	data := f.Values().Data()
	// Print rows — compute assignment from flat index (row-major).
	for i := 0; i < total; i++ {
		rem := i
		indices := make([]int, len(vars))
		for j := len(vars) - 1; j >= 0; j-- {
			indices[j] = rem % cards[j]
			rem /= cards[j]
		}
		for _, idx := range indices {
			fmt.Printf("%d\t", idx)
		}
		fmt.Printf("%.6f\n", data[i])
	}
}

// setStatesFromData infers state names from unique values in each column
// and sets them on the BN nodes that don't already have states.
func setStatesFromData(bn *models.BayesianNetwork, data *tabgo.DataFrame) {
	setStatesFromDataImpl(bn, data, &dataFrameColumnLookup{df: data})
}

// setStatesFromDataImpl is the testable implementation of setStatesFromData.
// Accepts a columnLookup for mock injection of missing-column paths.
func setStatesFromDataImpl(bn *models.BayesianNetwork, data *tabgo.DataFrame, lookup columnLookup) {
	for _, node := range bn.Nodes() {
		existing := bn.GetStates(node)
		if len(existing) > 0 {
			continue
		}
		col := lookup.Column(node)
		if col == nil {
			continue
		}
		uniq := col.Unique()
		stateNames := make([]string, len(uniq))
		for i, v := range uniq {
			stateNames[i] = fmt.Sprintf("%v", v)
		}
		sort.Strings(stateNames)
		bn.SetStates(node, stateNames)
	}
}

func bnToDigraph(bn *models.BayesianNetwork) *graphgo.DiGraph {
	g := graphgo.NewDiGraph()
	for _, n := range bn.Nodes() {
		g.AddNode(n)
	}
	for _, e := range bn.Edges() {
		g.AddEdge(e[0], e[1])
	}
	return g
}

func getScoreFunc(name string) learning.ScoreFunc {
	switch name {
	case "bdeu":
		scorer := structure_score.NewBDeu(5.0)
		return scorer.LocalScore
	case "k2":
		scorer := structure_score.NewK2()
		return scorer.LocalScore
	default: // "bic"
		scorer := structure_score.NewBIC()
		return scorer.LocalScore
	}
}

// adaptCITest wraps the ci_tests.CITest type to the learning.CITestFunc type.
func adaptCITest(test ci_tests.CITest) learning.CITestFunc {
	return func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
		return test(x, y, z, data, significance)
	}
}

// pdagToBN converts a graphgo.PDAG to a BayesianNetwork by orienting
// remaining undirected edges in a consistent direction.
func pdagToBN(pdag *graphgo.PDAG) *models.BayesianNetwork {
	bn := models.NewBayesianNetwork()
	for _, n := range pdag.Nodes() {
		bn.AddNode(n)
	}
	for _, e := range pdag.DirectedEdges() {
		bn.AddEdge(e[0], e[1])
	}
	// Orient undirected edges lexicographically (simple heuristic).
	for _, e := range pdag.UndirectedEdges() {
		if e[0] < e[1] {
			bn.AddEdge(e[0], e[1])
		} else {
			bn.AddEdge(e[1], e[0])
		}
	}
	return bn
}

func printUsage() {
	fmt.Println("Usage: pgmgo [command] [options]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  version                    Print version information")
	fmt.Println("  help                       Show this help message")
	fmt.Println("  validate <file>            Validate a BIF model file")
	fmt.Println("  query <file> [flags]       Run inference query")
	fmt.Println("  map <file> [flags]         Run MAP query")
	fmt.Println("  learn [flags]              Learn structure from data")
	fmt.Println("  fit [flags]                Fit parameters to data")
	fmt.Println("  sample [flags]             Generate samples from model")
	fmt.Println("  info <file>                Print model summary")
	fmt.Println("  convert [flags]            Convert between formats")
	fmt.Println("  compare [flags]            Compare two structures")
	fmt.Println("  do <file> [flags]          Causal do-calculus query")
	fmt.Println()
	fmt.Println("Options:")
	fmt.Println("  -h, --help       Show help")
	fmt.Println("  -v, --version    Show version")
}
