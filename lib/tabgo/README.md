# tabgo

Package `tabgo` provides tabular data structures and operations for loading, filtering, grouping, and transforming columnar data. It serves as the pandas equivalent for pgmgo.

```
import "github.com/asymmetric-effort/pgmgo/lib/tabgo"
```

## Series

A `Series` is a named column of data with elements of type `any`.

```go
s := tabgo.NewSeries("age", []any{25, 30, 35, 40})

s.Name()         // "age"
s.Len()          // 4
s.Values()       // []any copy of data
s.Float64()      // []float64{25, 30, 35, 40}
s.Int()          // []int{25, 30, 35, 40}

s.Unique()       // distinct values in order of first appearance
s.NUnique()      // number of distinct values
s.ValueCounts()  // map[any]int of value frequencies
s.IsNA()         // []bool where true means nil
```

## DataFrame

A `DataFrame` is an ordered collection of named Series (columns) of equal length.

### Creation

```go
// From map of Series (columns sorted alphabetically)
df := tabgo.NewDataFrame(map[string]*tabgo.Series{
    "name": tabgo.NewSeries("name", []any{"Alice", "Bob", "Carol"}),
    "age":  tabgo.NewSeries("age", []any{30, 25, 35}),
})

// From rows
df := tabgo.NewDataFrameFromRows(
    []string{"name", "age"},
    [][]any{
        {"Alice", 30},
        {"Bob", 25},
        {"Carol", 35},
    },
)
```

### Properties and Access

```go
df.Columns()     // []string column names in order
df.Len()         // number of rows
df.Column("age") // returns *Series (panics if not found)
```

### Selection and Filtering

```go
// Select specific columns
subset := df.Select("name", "age")

// Filter rows by predicate
adults := df.Filter(func(row map[string]any) bool {
    return row["age"].(int) >= 30
})

// Head and tail
first5 := df.Head(5)
last5 := df.Tail(5)

// Deep copy
cp := df.Copy()
```

## CSV I/O

```go
// Read from file
df, err := tabgo.ReadCSV("data.csv")

// Read from io.Reader
df, err := tabgo.ReadCSVFromReader(reader)

// Read from string
df, err := tabgo.ReadCSVFromString("name,age\nAlice,30\nBob,25\n")

// Write to file
err := tabgo.WriteCSV(df, "output.csv")
```

All CSV values are stored as strings. Use `Series.Float64()` or `Series.Int()` for numeric conversion.

## GroupBy Operations

```go
df := tabgo.NewDataFrameFromRows(
    []string{"dept", "name", "salary"},
    [][]any{
        {"eng", "Alice", 100.0},
        {"eng", "Bob", 120.0},
        {"sales", "Carol", 90.0},
        {"sales", "Dave", 95.0},
    },
)

gb := df.GroupBy("dept")

// Count rows per group
counts := gb.Count()  // columns: dept, count

// Sum numeric columns per group
totals := gb.Sum("salary")  // columns: dept, salary

// Mean of numeric columns per group
avgs := gb.Mean("salary")   // columns: dept, salary

// Get sub-DataFrames for each group
groups := gb.Groups()  // map[string]*DataFrame

// Apply a custom function to each group, then concatenate results
result := gb.Apply(func(sub *tabgo.DataFrame) *tabgo.DataFrame {
    // transform each group sub-DataFrame
    return sub
})
```

## Merge and Concat

### Inner Join

```go
left := tabgo.NewDataFrameFromRows(
    []string{"id", "name"},
    [][]any{{1, "Alice"}, {2, "Bob"}, {3, "Carol"}},
)
right := tabgo.NewDataFrameFromRows(
    []string{"id", "score"},
    [][]any{{1, 95}, {2, 87}, {4, 72}},
)

// Inner join on "id"
merged, err := tabgo.Merge(left, right, []string{"id"}, "inner")
// Result has columns: id, name, score
// Rows: {1, "Alice", 95}, {2, "Bob", 87}
```

### Vertical Concatenation

```go
df1 := tabgo.NewDataFrameFromRows([]string{"a", "b"}, [][]any{{1, 2}})
df2 := tabgo.NewDataFrameFromRows([]string{"a", "b"}, [][]any{{3, 4}})

combined, err := tabgo.Concat([]*tabgo.DataFrame{df1, df2})
// All DataFrames must have the same columns
```

## Missing Data Handling

Missing values are represented as `nil`.

```go
df := tabgo.NewDataFrameFromRows(
    []string{"a", "b"},
    [][]any{{1, nil}, {nil, 3}, {4, 5}},
)

// Check for missing values in a Series
df.Column("a").IsNA()  // []bool{false, true, false}

// Drop rows with any nil value
clean := df.DropNA()   // only row {4, 5} remains

// Fill all nil values with a default
filled := df.FillNA(0)

// Fill nil values in a specific column
filled := df.FillNAColumn("b", -1)
```

## API Summary

| Category | Types / Functions |
|---|---|
| **Series** | `Series`, `NewSeries` |
| **Series Methods** | `Name`, `Len`, `Values`, `Float64`, `Int`, `Unique`, `NUnique`, `ValueCounts`, `IsNA` |
| **DataFrame** | `DataFrame`, `NewDataFrame`, `NewDataFrameFromRows` |
| **DataFrame Methods** | `Columns`, `Len`, `Column`, `Select`, `Filter`, `Head`, `Tail`, `Copy` |
| **Missing Data** | `DropNA`, `FillNA`, `FillNAColumn` |
| **CSV I/O** | `ReadCSV`, `ReadCSVFromReader`, `ReadCSVFromString`, `WriteCSV` |
| **GroupBy** | `GroupBy`, `Groups`, `Count`, `Sum`, `Mean`, `Apply` |
| **Merge/Concat** | `Merge` (inner join), `Concat` |
