package config

import "sync"

// Config holds global configuration defaults for pgmgo.
type Config struct {
	// DefaultInferenceMethod is the name of the default inference algorithm
	// (e.g. "variable_elimination", "belief_propagation").
	DefaultInferenceMethod string

	// DefaultSignificance is the default significance level for statistical
	// tests (e.g. conditional independence tests).
	DefaultSignificance float64

	// Verbose controls whether verbose logging is enabled.
	Verbose bool

	// Seed is the default random seed. A value of 0 means use a random seed.
	Seed int64
}

var (
	mu      sync.RWMutex
	current = Config{
		DefaultInferenceMethod: "variable_elimination",
		DefaultSignificance:    0.05,
		Verbose:                false,
		Seed:                   0,
	}
)

// Default is the package-level default configuration.
// Use GetDefault/SetDefault for thread-safe access.
var Default = current

// SetDefault replaces the global default configuration.
func SetDefault(cfg Config) {
	mu.Lock()
	defer mu.Unlock()
	current = cfg
	Default = cfg
}

// GetDefault returns a copy of the current global default configuration.
func GetDefault() Config {
	mu.RLock()
	defer mu.RUnlock()
	return current
}
