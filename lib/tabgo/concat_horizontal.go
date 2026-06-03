package tabgo

import "fmt"

// ConcatHorizontal concatenates multiple DataFrames side by side (axis=1).
// All DataFrames must have the same number of rows.
// If column names conflict, a suffix "_N" (where N is the frame index) is appended.
func ConcatHorizontal(frames []*DataFrame) (*DataFrame, error) {
	if len(frames) == 0 {
		return NewDataFrameFromRows(nil, nil), nil
	}

	nRows := frames[0].Len()
	for i, f := range frames[1:] {
		if f.Len() != nRows {
			return nil, fmt.Errorf("tabgo: ConcatHorizontal: frame %d has %d rows, expected %d", i+1, f.Len(), nRows)
		}
	}

	// Collect all columns, resolving name conflicts.
	usedNames := make(map[string]int)
	var allCols []*Series
	var allNames []string

	for fi, f := range frames {
		for _, n := range f.Columns() {
			name := n
			if usedNames[name] > 0 {
				name = fmt.Sprintf("%s_%d", n, fi)
			}
			usedNames[n]++
			allNames = append(allNames, name)
			allCols = append(allCols, NewSeries(name, f.Column(n).Values()))
		}
	}

	newIdx := make(map[string]int, len(allNames))
	newCols := make([]*Series, len(allNames))
	for i, n := range allNames {
		newCols[i] = allCols[i]
		newIdx[n] = i
	}
	return &DataFrame{columns: newCols, index: newIdx}, nil
}
