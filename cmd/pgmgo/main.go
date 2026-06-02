package main

import (
	"fmt"
	"os"

	"github.com/asymmetric-effort/pgmgo/src/readwrite"
)

const version = "0.0.15"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	switch os.Args[1] {
	case "version", "--version", "-v":
		fmt.Printf("pgmgo %s\n", version)
	case "help", "--help", "-h":
		printUsage()
	case "validate":
		os.Exit(runValidate(os.Args[2:]))
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

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

func printUsage() {
	fmt.Println("Usage: pgmgo [command] [options]")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  version          Print version information")
	fmt.Println("  help             Show this help message")
	fmt.Println("  validate <file>  Validate a BIF model file")
	fmt.Println()
	fmt.Println("Options:")
	fmt.Println("  -h, --help       Show help")
	fmt.Println("  -v, --version    Show version")
}
