# prediction

Package `prediction` provides causal prediction methods including Double Machine Learning, naive adjustment regression, and instrumental variable regression.

**Import path:** `github.com/asymmetric-effort/pgmgo/src/prediction`

## Types

| Type | Description |
|------|-------------|
| `DoubleMLRegressor` | Double Machine Learning for ATE estimation with cross-fitting |
| `NaiveAdjustmentRegressor` | Back-door adjustment via OLS: outcome ~ treatment + adjustment set |
| `NaiveIVRegressor` | Two-stage least squares (2SLS) instrumental variable regression |

## DoubleML

Estimates the Average Treatment Effect (ATE) using cross-fitting to avoid overfitting bias.

```go
import (
    "github.com/asymmetric-effort/pgmgo/src/prediction"
    "github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

dml := prediction.NewDoubleMLRegressor(
    "Treatment",
    "Outcome",
    []string{"Confounder1", "Confounder2"},
)
dml.SetNSplits(5) // 5-fold cross-fitting

err := dml.Fit(data)
ate := dml.ATE()           // average treatment effect
se := dml.StandardError()  // standard error of ATE
ci := dml.ConfidenceInterval(0.95)
```

## NaiveAdjustmentRegressor

Estimates causal effects by regressing the outcome on treatment plus adjustment variables (back-door adjustment).

```go
adj := prediction.NewNaiveAdjustmentRegressor(
    "Treatment",
    "Outcome",
    []string{"Confounder1", "Confounder2"},
)

err := adj.Fit(data)
ate := adj.ATE()            // treatment coefficient
se := adj.StandardError()   // SE of treatment coefficient
predicted := adj.Predict(newData)
residuals := adj.Residuals()
```

## NaiveIVRegressor

Estimates causal effects using two-stage least squares when confounders are unobserved but valid instruments exist.

```go
iv := prediction.NewNaiveIVRegressor(
    "Treatment",
    "Outcome",
    []string{"Instrument1", "Instrument2"},
)

err := iv.Fit(data)
ate := iv.ATE()             // estimated causal effect
se := iv.StandardError()    // standard error
fstat := iv.FirstStageFStat() // instrument strength diagnostic
```
