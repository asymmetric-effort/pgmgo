# config

Package `config` holds global configuration defaults for pgmgo, providing thread-safe access to shared settings.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/config`

## Types

### Config

```go
type Config struct {
    DefaultInferenceMethod string  // e.g., "variable_elimination", "belief_propagation"
    DefaultSignificance    float64 // default significance level for CI tests (default: 0.05)
    Verbose                bool    // enable verbose logging (default: false)
    Seed                   int64   // default random seed; 0 means use random seed
}
```

## Functions

| Function | Description |
|----------|-------------|
| `GetDefault()` | Return a copy of the current global configuration (thread-safe) |
| `SetDefault(cfg)` | Replace the global configuration (thread-safe) |

## Default Values

| Field | Default |
|-------|---------|
| `DefaultInferenceMethod` | `"variable_elimination"` |
| `DefaultSignificance` | `0.05` |
| `Verbose` | `false` |
| `Seed` | `0` |

## Usage

```go
import "github.com/asymmetric-effort/pgmgo/src/config"

// Read current defaults
cfg := config.GetDefault()
fmt.Println(cfg.DefaultSignificance) // 0.05

// Update defaults
cfg.Verbose = true
cfg.DefaultInferenceMethod = "belief_propagation"
cfg.Seed = 42
config.SetDefault(cfg)

// Access the package-level variable directly (not thread-safe)
fmt.Println(config.Default.Verbose)
```
