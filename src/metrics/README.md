# metrics

Package `metrics` provides model evaluation functions including structural Hamming distance, confusion matrices, correlation scores, Fisher's C test, and structure score metrics.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/metrics`

## Structure Comparison

| Function | Description |
|----------|-------------|
| `SHD(trueG, estimated)` | Structural Hamming Distance: count of edge additions, deletions, and reversals (reversals count as 1) |
| `AdjacencyConfusionMatrix(trueG, estimated)` | TP, FP, TN, FN for edge presence (ignoring direction) |
| `OrientationConfusionMatrix(trueG, estimated)` | TP, FP, TN, FN for edge orientation among shared adjacencies |

## Model Fit

| Function | Description |
|----------|-------------|
| `CorrelationScore(edges, data)` | Average absolute Pearson correlation along edges |
| `FisherC(edges, data)` | Fisher's C statistic for testing d-separation implications against data |
| `ImpliedCIs(edges, allVars)` | Enumerate conditional independence statements implied by a DAG structure |

## Structure Scoring

| Function | Description |
|----------|-------------|
| `StructureScoreMetric(variables, parentMap, data, scorer)` | Total score by summing local scores over all variables |

## Usage

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/metrics"
    "github.com/asymmetric-effort/pgmgo/lib/graphgo"
)

trueG := graphgo.NewDiGraph()
trueG.AddNode("X"); trueG.AddNode("Y"); trueG.AddNode("Z")
trueG.AddEdge("X", "Y")
trueG.AddEdge("Y", "Z")

estG := graphgo.NewDiGraph()
estG.AddNode("X"); estG.AddNode("Y"); estG.AddNode("Z")
estG.AddEdge("Y", "X") // reversed
estG.AddEdge("Y", "Z") // correct

// Structural Hamming Distance
dist := metrics.SHD(trueG, estG) // 1 (one reversal)

// Adjacency confusion matrix
tp, fp, tn, fn := metrics.AdjacencyConfusionMatrix(trueG, estG)

// Orientation confusion matrix
tp, fp, tn, fn = metrics.OrientationConfusionMatrix(trueG, estG)

// Correlation score along edges
edges := [][2]string{{"X", "Y"}, {"Y", "Z"}}
corr := metrics.CorrelationScore(edges, data)

// Fisher's C test for model fit
cStat, pval := metrics.FisherC(edges, data)

// Structure score using any LocalScorer
import "github.com/asymmetric-effort/pgmgo/src/structure_score"
bic := structure_score.NewBIC()
parentMap := map[string][]string{"X": {}, "Y": {"X"}, "Z": {"Y"}}
total := metrics.StructureScoreMetric([]string{"X", "Y", "Z"}, parentMap, data, bic)
```
