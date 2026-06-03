package tabgo

import (
	"fmt"
	"math"
	"strings"
	"unsafe"
)

// Dtypes returns a map of column name to its detected dtype string.
// Possible values: "float64", "int", "string", "bool", "mixed", "empty".
func (df *DataFrame) Dtypes() map[string]string {
	names := df.Columns()
	result := make(map[string]string, len(names))
	for _, n := range names {
		result[n] = df.Column(n).Dtype()
	}
	return result
}

// Info returns a summary string with column names, non-null counts, and dtypes.
func (df *DataFrame) Info() string {
	names := df.Columns()
	nRows := df.Len()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("DataFrame: %d rows x %d columns\n", nRows, len(names)))
	sb.WriteString(fmt.Sprintf("%-20s %-15s %-10s\n", "Column", "Non-Null", "Dtype"))
	sb.WriteString(strings.Repeat("-", 45))
	sb.WriteString("\n")

	for _, n := range names {
		s := df.Column(n)
		nonNull := s.Count()
		dtype := s.Dtype()
		sb.WriteString(fmt.Sprintf("%-20s %-15s %-10s\n", n, fmt.Sprintf("%d non-null", nonNull), dtype))
	}
	return sb.String()
}

// Memory returns the approximate memory usage in bytes of the DataFrame.
func (df *DataFrame) Memory() int {
	total := 0
	for _, col := range df.columns {
		vals := col.Values()
		// Base slice overhead.
		total += int(unsafe.Sizeof(vals))
		for _, v := range vals {
			total += estimateSize(v)
		}
		// Series name.
		total += len(col.Name())
	}
	return total
}

// estimateSize returns an approximate size in bytes for a value.
func estimateSize(v any) int {
	if v == nil {
		return 0
	}
	switch val := v.(type) {
	case float64:
		return 8
	case float32:
		return 4
	case int:
		return 8
	case int8:
		return 1
	case int16:
		return 2
	case int32:
		return 4
	case int64:
		return 8
	case string:
		return len(val) + 16
	case bool:
		return 1
	default:
		_ = val
		return 16
	}
}

// Nunique returns the number of unique values per column.
func (df *DataFrame) Nunique() map[string]int {
	names := df.Columns()
	result := make(map[string]int, len(names))
	for _, n := range names {
		result[n] = df.Column(n).NUnique()
	}
	return result
}

// Duplicated returns a boolean slice where true indicates the row is a duplicate
// of an earlier row. The first occurrence is marked false.
func (df *DataFrame) Duplicated() []bool {
	nRows := df.Len()
	names := df.Columns()
	allVals := make([][]any, len(names))
	for i, n := range names {
		allVals[i] = df.Column(n).Values()
	}

	seen := make(map[string]bool, nRows)
	result := make([]bool, nRows)
	for r := 0; r < nRows; r++ {
		parts := make([]string, len(names))
		for c := range names {
			parts[c] = fmt.Sprintf("%v", allVals[c][r])
		}
		key := strings.Join(parts, "\x00")
		if seen[key] {
			result[r] = true
		} else {
			seen[key] = true
		}
	}
	return result
}

// DropDuplicates returns a new DataFrame with duplicate rows removed.
// The first occurrence of each row is kept.
func (df *DataFrame) DropDuplicates() *DataFrame {
	dups := df.Duplicated()
	return df.Filter(func(row map[string]any) bool {
		// We need to use index-based logic, not row map.
		return true
	}).dropDuplicatesImpl(dups)
}

// dropDuplicatesImpl filters using the pre-computed duplicate flags.
func (df *DataFrame) dropDuplicatesImpl(dups []bool) *DataFrame {
	names := df.Columns()
	allVals := make([][]any, len(names))
	for i, n := range names {
		allVals[i] = df.Column(n).Values()
	}

	var kept []int
	for i, dup := range dups {
		if !dup {
			kept = append(kept, i)
		}
	}

	newCols := make([]*Series, len(names))
	newIdx := make(map[string]int, len(names))
	for i, n := range names {
		data := make([]any, len(kept))
		for j, r := range kept {
			data[j] = allVals[i][r]
		}
		newCols[i] = NewSeries(n, data)
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}
}

// Replace returns a new DataFrame where all occurrences of old are replaced with new_.
func (df *DataFrame) Replace(old, new_ any) *DataFrame {
	names := df.Columns()
	nRows := df.Len()
	newCols := make([]*Series, len(names))
	newIdx := make(map[string]int, len(names))
	for i, n := range names {
		vals := df.Column(n).Values()
		newVals := make([]any, nRows)
		for j, v := range vals {
			if fmt.Sprintf("%v", v) == fmt.Sprintf("%v", old) {
				newVals[j] = new_
			} else {
				newVals[j] = v
			}
		}
		newCols[i] = NewSeries(n, newVals)
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}
}

// Clip returns a new DataFrame with numeric values clipped to [lower, upper].
// Non-numeric values are left unchanged.
func (df *DataFrame) Clip(lower, upper float64) *DataFrame {
	names := df.Columns()
	nRows := df.Len()
	newCols := make([]*Series, len(names))
	newIdx := make(map[string]int, len(names))
	for i, n := range names {
		vals := df.Column(n).Values()
		newVals := make([]any, nRows)
		for j, v := range vals {
			if v == nil || !isNumeric(v) {
				newVals[j] = v
				continue
			}
			f := toFloat64(v)
			if f < lower {
				f = lower
			}
			if f > upper {
				f = upper
			}
			newVals[j] = f
		}
		newCols[i] = NewSeries(n, newVals)
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}
}

// Abs returns a new DataFrame with absolute values of all numeric columns.
// Non-numeric values are left unchanged.
func (df *DataFrame) Abs() *DataFrame {
	names := df.Columns()
	nRows := df.Len()
	newCols := make([]*Series, len(names))
	newIdx := make(map[string]int, len(names))
	for i, n := range names {
		vals := df.Column(n).Values()
		newVals := make([]any, nRows)
		for j, v := range vals {
			if v == nil || !isNumeric(v) {
				newVals[j] = v
				continue
			}
			newVals[j] = math.Abs(toFloat64(v))
		}
		newCols[i] = NewSeries(n, newVals)
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}
}

// Round returns a new DataFrame with numeric values rounded to the given number
// of decimal places. Non-numeric values are left unchanged.
func (df *DataFrame) Round(decimals int) *DataFrame {
	pow := math.Pow(10, float64(decimals))
	names := df.Columns()
	nRows := df.Len()
	newCols := make([]*Series, len(names))
	newIdx := make(map[string]int, len(names))
	for i, n := range names {
		vals := df.Column(n).Values()
		newVals := make([]any, nRows)
		for j, v := range vals {
			if v == nil || !isNumeric(v) {
				newVals[j] = v
				continue
			}
			newVals[j] = math.Round(toFloat64(v)*pow) / pow
		}
		newCols[i] = NewSeries(n, newVals)
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}
}

// Pipe passes the DataFrame through a transformation function.
// This enables method chaining with custom functions.
func (df *DataFrame) Pipe(fn func(*DataFrame) *DataFrame) *DataFrame {
	return fn(df)
}

// Shift returns a new DataFrame where each column's values are shifted by the
// given number of periods. Positive periods shift values down (introducing nil
// at the top); negative periods shift values up (introducing nil at the bottom).
func (df *DataFrame) Shift(periods int) *DataFrame {
	names := df.Columns()
	nRows := df.Len()
	newCols := make([]*Series, len(names))
	newIdx := make(map[string]int, len(names))
	for i, n := range names {
		vals := df.Column(n).Values()
		newVals := shiftSlice(vals, nRows, periods)
		newCols[i] = NewSeries(n, newVals)
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}
}

// shiftSlice shifts a slice of values by periods.
func shiftSlice(vals []any, n int, periods int) []any {
	out := make([]any, n)
	if periods >= 0 {
		for i := 0; i < n; i++ {
			src := i - periods
			if src >= 0 && src < n {
				out[i] = vals[src]
			} else {
				out[i] = nil
			}
		}
	} else {
		for i := 0; i < n; i++ {
			src := i - periods
			if src >= 0 && src < n {
				out[i] = vals[src]
			} else {
				out[i] = nil
			}
		}
	}
	return out
}
