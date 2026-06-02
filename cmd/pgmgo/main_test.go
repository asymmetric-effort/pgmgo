//go:build unit

package main

import (
	"os"
	"path/filepath"
	"testing"
)

const validBIF = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}

variable Sprinkler {
  type discrete [ 2 ] { True, False };
}

probability ( Rain ) {
  table 0.2, 0.8;
}

probability ( Sprinkler | Rain ) {
  (True) 0.01, 0.99;
  (False) 0.4, 0.6;
}
`

const invalidBIF = `network unknown {
}

variable Rain {
  type discrete [ 2 ] { True, False };
}
`

const malformedBIF = `this is not a valid BIF file {{{
  broken syntax !!!
`

func writeTempBIF(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bif")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp BIF file: %v", err)
	}
	return path
}

func TestValidateBIFFile_ValidModel(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := validateBIFFile(path)
	if code != 0 {
		t.Errorf("expected exit code 0 for valid model, got %d", code)
	}
}

func TestValidateBIFFile_MissingCPD(t *testing.T) {
	// Model has a node with no CPD, so CheckModel should fail.
	path := writeTempBIF(t, invalidBIF)
	code := validateBIFFile(path)
	if code != 1 {
		t.Errorf("expected exit code 1 for model with missing CPD, got %d", code)
	}
}

func TestValidateBIFFile_FileNotFound(t *testing.T) {
	code := validateBIFFile("/nonexistent/path/to/file.bif")
	if code != 2 {
		t.Errorf("expected exit code 2 for missing file, got %d", code)
	}
}

func TestRunValidate_NoArgs(t *testing.T) {
	code := runValidate([]string{})
	if code != 2 {
		t.Errorf("expected exit code 2 for missing argument, got %d", code)
	}
}

func TestRunValidate_ValidFile(t *testing.T) {
	path := writeTempBIF(t, validBIF)
	code := runValidate([]string{path})
	if code != 0 {
		t.Errorf("expected exit code 0 for valid file, got %d", code)
	}
}

func TestVersion(t *testing.T) {
	if version != "0.0.19" {
		t.Errorf("expected version 0.0.19, got %s", version)
	}
}
