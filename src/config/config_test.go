//go:build unit

package config

import (
	"testing"
)

func TestDefaultValues(t *testing.T) {
	cfg := GetDefault()
	if cfg.DefaultInferenceMethod != "variable_elimination" {
		t.Errorf("DefaultInferenceMethod = %q, want %q", cfg.DefaultInferenceMethod, "variable_elimination")
	}
	if cfg.DefaultSignificance != 0.05 {
		t.Errorf("DefaultSignificance = %f, want 0.05", cfg.DefaultSignificance)
	}
	if cfg.Verbose {
		t.Error("Verbose should be false by default")
	}
	if cfg.Seed != 0 {
		t.Errorf("Seed = %d, want 0", cfg.Seed)
	}
}

func TestSetDefault(t *testing.T) {
	// Save original and restore after test.
	original := GetDefault()
	defer SetDefault(original)

	custom := Config{
		DefaultInferenceMethod: "belief_propagation",
		DefaultSignificance:    0.01,
		Verbose:                true,
		Seed:                   42,
	}
	SetDefault(custom)

	got := GetDefault()
	if got.DefaultInferenceMethod != "belief_propagation" {
		t.Errorf("DefaultInferenceMethod = %q, want %q", got.DefaultInferenceMethod, "belief_propagation")
	}
	if got.DefaultSignificance != 0.01 {
		t.Errorf("DefaultSignificance = %f, want 0.01", got.DefaultSignificance)
	}
	if !got.Verbose {
		t.Error("Verbose should be true")
	}
	if got.Seed != 42 {
		t.Errorf("Seed = %d, want 42", got.Seed)
	}
}

func TestDefaultPackageVar(t *testing.T) {
	// Save original and restore after test.
	original := GetDefault()
	defer SetDefault(original)

	custom := Config{
		DefaultInferenceMethod: "test_method",
		DefaultSignificance:    0.10,
		Verbose:                true,
		Seed:                   99,
	}
	SetDefault(custom)

	// The Default package variable should also be updated.
	if Default.DefaultInferenceMethod != "test_method" {
		t.Errorf("Default.DefaultInferenceMethod = %q, want %q", Default.DefaultInferenceMethod, "test_method")
	}
	if Default.Seed != 99 {
		t.Errorf("Default.Seed = %d, want 99", Default.Seed)
	}
}

func TestGetDefaultReturnsCopy(t *testing.T) {
	cfg := GetDefault()
	cfg.Verbose = !cfg.Verbose

	// Modifying the returned copy should not affect the global.
	got := GetDefault()
	if got.Verbose == cfg.Verbose {
		t.Error("GetDefault should return a copy, not a reference")
	}
}
