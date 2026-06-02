package tabgo

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"
)

// ReadCSV reads a CSV file and returns a DataFrame.
// The first row is treated as column headers.
// All values are stored as strings.
func ReadCSV(path string) (*DataFrame, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return ReadCSVFromReader(f)
}

// ReadCSVFromReader reads CSV data from an io.Reader and returns a DataFrame.
// The first row is treated as column headers.
// All values are stored as strings.
func ReadCSVFromReader(rd io.Reader) (*DataFrame, error) {
	r := csv.NewReader(rd)
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) == 0 {
		return NewDataFrameFromRows(nil, nil), nil
	}

	headers := records[0]
	rows := make([][]any, 0, len(records)-1)
	for _, rec := range records[1:] {
		row := make([]any, len(rec))
		for i, v := range rec {
			row[i] = v
		}
		rows = append(rows, row)
	}
	return NewDataFrameFromRows(headers, rows), nil
}

// ReadCSVFromString reads CSV data from a string and returns a DataFrame.
func ReadCSVFromString(s string) (*DataFrame, error) {
	return ReadCSVFromReader(strings.NewReader(s))
}

// WriteCSV writes a DataFrame to a CSV file.
// Values are converted to strings via their default formatting.
func WriteCSV(df *DataFrame, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	names := df.Columns()
	// write header
	if err := w.Write(names); err != nil {
		return err
	}

	nRows := df.Len()
	// pre-fetch column values
	allVals := make([][]any, len(names))
	for i, n := range names {
		allVals[i] = df.Column(n).Values()
	}

	record := make([]string, len(names))
	for r := 0; r < nRows; r++ {
		for c := range names {
			record[c] = anyToString(allVals[c][r])
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}
	return w.Error()
}

// anyToString converts a value to its string representation.
func anyToString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%v", v)
}
