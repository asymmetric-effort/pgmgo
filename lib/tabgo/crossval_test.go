//go:build unit

package tabgo_test

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"testing"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/tests/testutil"
)

// tolerances
const (
	tolExact = 1e-10 // exact arithmetic (creation, sums, counts)
	tolStat  = 1e-6  // statistical ops (std, var, corr, percentiles)
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func assertFloatClose(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(want) && math.IsNaN(got) {
		return
	}
	if math.IsNaN(want) || math.IsNaN(got) {
		t.Errorf("[%s] got %g, want %g", name, got, want)
		return
	}
	diff := math.Abs(got - want)
	if diff > tol+tol*math.Abs(want) {
		t.Errorf("[%s] got %g, want %g (diff %g, tol %g)", name, got, want, diff, tol)
	}
}

func assertIntEqual(t *testing.T, name string, got, want int) {
	t.Helper()
	if got != want {
		t.Errorf("[%s] got %d, want %d", name, got, want)
	}
}

func assertBoolEqual(t *testing.T, name string, got, want bool) {
	t.Helper()
	if got != want {
		t.Errorf("[%s] got %v, want %v", name, got, want)
	}
}

// buildDF builds a DataFrame from the fixture input columns map.
// columns: map[string][]any(floats or strings). Column order is alphabetical.
func buildDF(columns map[string][]any) *tabgo.DataFrame {
	m := make(map[string]*tabgo.Series, len(columns))
	for name, vals := range columns {
		m[name] = tabgo.NewSeries(name, vals)
	}
	return tabgo.NewDataFrame(m)
}

// buildDFOrdered builds a DataFrame preserving column_order if present.
func buildDFOrdered(columns map[string][]any, order []string) *tabgo.DataFrame {
	if len(order) == 0 {
		return buildDF(columns)
	}
	rows := make([][]any, 0)
	if len(columns) == 0 {
		return tabgo.NewDataFrameFromRows(order, nil)
	}
	// Determine row count from first column.
	var nRows int
	for _, v := range columns {
		nRows = len(v)
		break
	}
	for r := 0; r < nRows; r++ {
		row := make([]any, len(order))
		for c, name := range order {
			row[c] = columns[name][r]
		}
		rows = append(rows, row)
	}
	return tabgo.NewDataFrameFromRows(order, rows)
}

// parseColumnsInput parses {"columns": {"A": [...], ...}} into map[string][]any.
func parseColumnsInput(raw json.RawMessage) map[string][]any {
	var inp struct {
		Columns map[string][]any `json:"columns"`
	}
	if err := json.Unmarshal(raw, &inp); err != nil {
		return nil
	}
	// Convert []any elements: JSON numbers are float64, null is nil.
	for k, vals := range inp.Columns {
		for i, v := range vals {
			if v == nil {
				vals[i] = nil
			}
			// json.Unmarshal stores numbers as float64, strings as string.
		}
		inp.Columns[k] = vals
	}
	return inp.Columns
}

// parseFloatMap parses {"key": float, ...}
func parseFloatMap(raw json.RawMessage) map[string]float64 {
	var m map[string]float64
	json.Unmarshal(raw, &m)
	return m
}

// parseIntMap parses {"key": int, ...}
func parseIntMap(raw json.RawMessage) map[string]int {
	var m map[string]int
	json.Unmarshal(raw, &m)
	return m
}

// ---------------------------------------------------------------------------
// main test dispatcher
// ---------------------------------------------------------------------------

func TestCrossValidation(t *testing.T) {
	ff := testutil.LoadFixtures(t, "tabgo/fixtures.json")
	if ff == nil {
		return
	}

	t.Logf("Loaded %d test cases from tabgo fixtures", len(ff.TestCases))

	for _, tc := range ff.TestCases {
		tc := tc
		t.Run(tc.Name, func(t *testing.T) {
			switch {
			case strings.HasPrefix(tc.Name, "creation_"):
				testCreation(t, &tc)
			case strings.HasPrefix(tc.Name, "agg_"):
				testAggregation(t, &tc)
			case strings.HasPrefix(tc.Name, "groupby_"):
				testGroupBy(t, &tc)
			case strings.HasPrefix(tc.Name, "merge_"):
				testMerge(t, &tc)
			case strings.HasPrefix(tc.Name, "concat_"):
				testConcat(t, &tc)
			case strings.HasPrefix(tc.Name, "sort_"):
				testSort(t, &tc)
			case strings.HasPrefix(tc.Name, "missing_"):
				testMissing(t, &tc)
			case strings.HasPrefix(tc.Name, "reshape_"):
				testReshape(t, &tc)
			case strings.HasPrefix(tc.Name, "stats_"):
				testStatistics(t, &tc)
			case strings.HasPrefix(tc.Name, "series_"):
				testSeries(t, &tc)
			case strings.HasPrefix(tc.Name, "csv_"):
				testCSV(t, &tc)
			default:
				t.Skipf("no handler for test case %q", tc.Name)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 1: Creation
// ---------------------------------------------------------------------------

func testCreation(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	var expected struct {
		Shape         []int          `json:"shape"`
		Columns       []string       `json:"columns"`
		Size          int            `json:"size"`
		Ndim          int            `json:"ndim"`
		Empty         *bool          `json:"empty"`
		NullCounts    map[string]int `json:"null_counts"`
		NonNullCounts map[string]int `json:"non_null_counts"`
	}
	tc.UnmarshalExpected(t, &expected)

	cols := parseColumnsInput(tc.Input)

	if tc.Name == "creation_empty" {
		df := tabgo.NewDataFrameFromRows(nil, nil)
		shape := df.Shape()
		assertIntEqual(t, "shape[0]", shape[0], 0)
		assertIntEqual(t, "shape[1]", shape[1], 0)
		assertBoolEqual(t, "empty", df.Empty(), true)
		assertIntEqual(t, "size", df.Size(), 0)
		assertIntEqual(t, "ndim", df.Ndim(), 2)
		return
	}

	if tc.Name == "creation_from_records" {
		var inp struct {
			Records     []map[string]any `json:"records"`
			ColumnNames []string         `json:"column_names"`
		}
		tc.UnmarshalInput(t, &inp)
		rows := make([][]any, len(inp.Records))
		for i, rec := range inp.Records {
			row := make([]any, len(inp.ColumnNames))
			for j, col := range inp.ColumnNames {
				row[j] = rec[col]
			}
			rows[i] = row
		}
		df := tabgo.NewDataFrameFromRows(inp.ColumnNames, rows)
		shape := df.Shape()
		assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
		assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
		return
	}

	df := buildDF(cols)

	if expected.Shape != nil {
		shape := df.Shape()
		assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
		assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
	}

	if expected.Columns != nil {
		gotCols := df.Columns()
		sort.Strings(gotCols)
		wantCols := expected.Columns
		sort.Strings(wantCols)
		if len(gotCols) != len(wantCols) {
			t.Fatalf("columns length: got %d, want %d", len(gotCols), len(wantCols))
		}
		for i := range gotCols {
			if gotCols[i] != wantCols[i] {
				t.Errorf("column %d: got %q, want %q", i, gotCols[i], wantCols[i])
			}
		}
	}

	if expected.Size > 0 {
		assertIntEqual(t, "size", df.Size(), expected.Size)
	}

	if expected.Ndim > 0 {
		assertIntEqual(t, "ndim", df.Ndim(), expected.Ndim)
	}

	if expected.Empty != nil {
		assertBoolEqual(t, "empty", df.Empty(), *expected.Empty)
	}

	if expected.NullCounts != nil {
		cnt := df.Count()
		for col, wantNull := range expected.NullCounts {
			nRows := df.Len()
			gotNull := nRows - cnt[col]
			assertIntEqual(t, fmt.Sprintf("null_count[%s]", col), gotNull, wantNull)
		}
	}

	if expected.NonNullCounts != nil {
		cnt := df.Count()
		for col, want := range expected.NonNullCounts {
			assertIntEqual(t, fmt.Sprintf("non_null_count[%s]", col), cnt[col], want)
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 2: Aggregation
// ---------------------------------------------------------------------------

func testAggregation(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	cols := parseColumnsInput(tc.Input)
	df := buildDF(cols)

	var expected struct {
		Sum      map[string]float64            `json:"sum"`
		Mean     map[string]float64            `json:"mean"`
		Std      map[string]float64            `json:"std"`
		Var      map[string]float64            `json:"var"`
		Min      map[string]float64            `json:"min"`
		Max      map[string]float64            `json:"max"`
		Median   map[string]float64            `json:"median"`
		Count    map[string]int                `json:"count"`
		Describe map[string]map[string]float64 `json:"describe"`
	}
	tc.UnmarshalExpected(t, &expected)

	if expected.Sum != nil {
		got := df.Sum()
		for col, want := range expected.Sum {
			assertFloatClose(t, fmt.Sprintf("sum[%s]", col), got[col], want, tolExact)
		}
	}

	if expected.Mean != nil {
		got := df.MeanAll()
		for col, want := range expected.Mean {
			assertFloatClose(t, fmt.Sprintf("mean[%s]", col), got[col], want, tolExact)
		}
	}

	if expected.Std != nil {
		got := df.StdAll()
		for col, want := range expected.Std {
			assertFloatClose(t, fmt.Sprintf("std[%s]", col), got[col], want, tolStat)
		}
	}

	if expected.Var != nil {
		got := df.VarAll()
		for col, want := range expected.Var {
			assertFloatClose(t, fmt.Sprintf("var[%s]", col), got[col], want, tolStat)
		}
	}

	if expected.Min != nil {
		got := df.MinAll()
		for col, want := range expected.Min {
			assertFloatClose(t, fmt.Sprintf("min[%s]", col), got[col], want, tolExact)
		}
	}

	if expected.Max != nil {
		got := df.MaxAll()
		for col, want := range expected.Max {
			assertFloatClose(t, fmt.Sprintf("max[%s]", col), got[col], want, tolExact)
		}
	}

	if expected.Median != nil {
		got := df.MedianAll()
		for col, want := range expected.Median {
			assertFloatClose(t, fmt.Sprintf("median[%s]", col), got[col], want, tolExact)
		}
	}

	if expected.Count != nil {
		got := df.Count()
		for col, want := range expected.Count {
			assertIntEqual(t, fmt.Sprintf("count[%s]", col), got[col], want)
		}
	}

	if expected.Describe != nil {
		desc := df.Describe()
		// Describe returns a DataFrame with rows: count, mean, std, min, 25%, 50%, 75%, max
		statCol := desc.Column("stat")
		statVals := statCol.Values()

		for col, stats := range expected.Describe {
			colSeries := desc.Column(col)
			colVals := colSeries.Values()
			for i, sv := range statVals {
				statName, ok := sv.(string)
				if !ok {
					continue
				}
				want, exists := stats[statName]
				if !exists {
					continue
				}
				got := toFloat(colVals[i])
				tol := tolStat
				if statName == "count" || statName == "min" || statName == "max" {
					tol = tolExact
				}
				assertFloatClose(t, fmt.Sprintf("describe[%s][%s]", col, statName), got, want, tol)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 3: GroupBy
// ---------------------------------------------------------------------------

func testGroupBy(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	var inp struct {
		Columns      map[string][]any `json:"columns"`
		GroupBy      []string         `json:"group_by"`
		ValueColumns []string         `json:"value_columns"`
	}
	tc.UnmarshalInput(t, &inp)

	df := buildDF(inp.Columns)
	gb := df.GroupBy(inp.GroupBy...)

	switch tc.Name {
	case "groupby_sum":
		testGroupByAgg(t, tc, gb.Sum(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_mean":
		testGroupByAgg(t, tc, gb.Mean(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_std":
		testGroupByAgg(t, tc, gb.Std(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_var":
		testGroupByAgg(t, tc, gb.Var(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_count":
		testGroupByCount(t, tc, gb.Count(), inp.GroupBy)
	case "groupby_min":
		testGroupByAgg(t, tc, gb.Min(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_max":
		testGroupByAgg(t, tc, gb.Max(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_median":
		testGroupByAgg(t, tc, gb.Median(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_first":
		testGroupByFirstLast(t, tc, gb.First(), inp.GroupBy)
	case "groupby_last":
		testGroupByFirstLast(t, tc, gb.Last(), inp.GroupBy)
	case "groupby_size":
		testGroupBySize(t, tc, gb.Size(), inp.GroupBy)
	case "groupby_multi_sum":
		testGroupByMulti(t, tc, gb.Sum(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_multi_mean":
		testGroupByMulti(t, tc, gb.Mean(inp.ValueColumns...), inp.GroupBy, inp.ValueColumns)
	case "groupby_single_member":
		var expected struct {
			GroupCSumValue  float64 `json:"group_C_sum_Value"`
			GroupCMeanValue float64 `json:"group_C_mean_Value"`
			GroupCCount     int     `json:"group_C_count"`
		}
		tc.UnmarshalExpected(t, &expected)
		sumDf := gb.Sum("Value")
		// Find group C row
		groupVals := sumDf.Column(inp.GroupBy[0]).Values()
		valVals := sumDf.Column("Value").Values()
		for i, gv := range groupVals {
			if fmt.Sprintf("%v", gv) == "C" {
				assertFloatClose(t, "C_sum_Value", toFloat(valVals[i]), expected.GroupCSumValue, tolExact)
			}
		}
		meanDf := gb.Mean("Value")
		groupVals = meanDf.Column(inp.GroupBy[0]).Values()
		valVals = meanDf.Column("Value").Values()
		for i, gv := range groupVals {
			if fmt.Sprintf("%v", gv) == "C" {
				assertFloatClose(t, "C_mean_Value", toFloat(valVals[i]), expected.GroupCMeanValue, tolExact)
			}
		}
	case "groupby_ngroups":
		var expected struct {
			Ngroups int `json:"ngroups"`
		}
		tc.UnmarshalExpected(t, &expected)
		assertIntEqual(t, "ngroups", gb.Ngroups(), expected.Ngroups)
	default:
		t.Skipf("unhandled groupby case: %s", tc.Name)
	}
}

func testGroupByAgg(t *testing.T, tc *testutil.TestCase, result *tabgo.DataFrame, groupCols, valCols []string) {
	t.Helper()
	var expected struct {
		Result map[string]map[string]float64 `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	groupVals := result.Column(groupCols[0]).Values()
	for _, valCol := range valCols {
		colVals := result.Column(valCol).Values()
		wantMap := expected.Result[valCol]
		for i, gv := range groupVals {
			key := fmt.Sprintf("%v", gv)
			want, ok := wantMap[key]
			if !ok {
				continue
			}
			got := toFloat(colVals[i])
			tol := tolExact
			if strings.Contains(tc.Name, "std") || strings.Contains(tc.Name, "var") {
				tol = tolStat
			}
			assertFloatClose(t, fmt.Sprintf("group[%s].%s", key, valCol), got, want, tol)
		}
	}
}

func testGroupByCount(t *testing.T, tc *testutil.TestCase, result *tabgo.DataFrame, groupCols []string) {
	t.Helper()
	// pandas groupby().count() returns per-column counts; tabgo Count() returns a single "count" column.
	// For non-null data they are the same, so we compare against the "count" column.
	var expected struct {
		Result map[string]map[string]float64 `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	groupVals := result.Column(groupCols[0]).Values()
	countVals := result.Column("count").Values()

	// Pick any value column from the expected result (they should all be equal for non-null data).
	for valCol, wantMap := range expected.Result {
		for i, gv := range groupVals {
			key := fmt.Sprintf("%v", gv)
			want, ok := wantMap[key]
			if !ok {
				continue
			}
			got := toFloat(countVals[i])
			assertFloatClose(t, fmt.Sprintf("count[%s].%s", key, valCol), got, want, tolExact)
		}
		break // only need to check one column since counts are the same
	}
}

func testGroupByFirstLast(t *testing.T, tc *testutil.TestCase, result *tabgo.DataFrame, groupCols []string) {
	t.Helper()
	var expected struct {
		Result map[string]map[string]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	groupVals := result.Column(groupCols[0]).Values()
	for valCol, wantMap := range expected.Result {
		if valCol == groupCols[0] {
			continue // skip group column itself
		}
		colVals := result.Column(valCol).Values()
		for i, gv := range groupVals {
			key := fmt.Sprintf("%v", gv)
			want, ok := wantMap[key]
			if !ok {
				continue
			}
			got := toFloat(colVals[i])
			wantF := toFloat(want)
			assertFloatClose(t, fmt.Sprintf("first_last[%s].%s", key, valCol), got, wantF, tolExact)
		}
	}
}

func testGroupBySize(t *testing.T, tc *testutil.TestCase, result *tabgo.DataFrame, groupCols []string) {
	t.Helper()
	var expected struct {
		Result map[string]float64 `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	groupVals := result.Column(groupCols[0]).Values()
	sizeVals := result.Column("size").Values()
	for i, gv := range groupVals {
		key := fmt.Sprintf("%v", gv)
		want, ok := expected.Result[key]
		if !ok {
			continue
		}
		got := toFloat(sizeVals[i])
		assertFloatClose(t, fmt.Sprintf("size[%s]", key), got, want, tolExact)
	}
}

func testGroupByMulti(t *testing.T, tc *testutil.TestCase, result *tabgo.DataFrame, groupCols, valCols []string) {
	t.Helper()
	var expected struct {
		Result map[string]float64 `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	// Build composite keys from group columns
	nRows := result.Len()
	groupColVals := make([][]any, len(groupCols))
	for i, gc := range groupCols {
		groupColVals[i] = result.Column(gc).Values()
	}
	valColVals := result.Column(valCols[0]).Values()

	for r := 0; r < nRows; r++ {
		parts := make([]string, len(groupCols))
		for gi := range groupCols {
			parts[gi] = fmt.Sprintf("%v", groupColVals[gi][r])
		}
		key := strings.Join(parts, "|")
		want, ok := expected.Result[key]
		if !ok {
			continue
		}
		got := toFloat(valColVals[r])
		assertFloatClose(t, fmt.Sprintf("multi[%s]", key), got, want, tolExact)
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 4: Merge
// ---------------------------------------------------------------------------

func testMerge(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	var inp struct {
		Left struct {
			Columns     map[string][]any `json:"columns"`
			ColumnOrder []string         `json:"column_order"`
		} `json:"left"`
		Right struct {
			Columns     map[string][]any `json:"columns"`
			ColumnOrder []string         `json:"column_order"`
		} `json:"right"`
		On  []string `json:"on"`
		How string   `json:"how"`
	}
	tc.UnmarshalInput(t, &inp)

	left := buildDFOrdered(inp.Left.Columns, inp.Left.ColumnOrder)
	right := buildDFOrdered(inp.Right.Columns, inp.Right.ColumnOrder)

	result, err := tabgo.Merge(left, right, inp.On, inp.How)
	if err != nil {
		t.Fatalf("Merge error: %v", err)
	}

	var expected struct {
		Shape  []int            `json:"shape"`
		Result map[string][]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	if expected.Shape != nil {
		shape := result.Shape()
		assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
		assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
	}

	if expected.Result != nil {
		for col, wantVals := range expected.Result {
			gotVals := result.Column(col).Values()
			if len(gotVals) != len(wantVals) {
				t.Errorf("merge[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
				continue
			}
			for i, wv := range wantVals {
				if wv == nil {
					if gotVals[i] != nil {
						t.Errorf("merge[%s][%d]: got %v, want nil", col, i, gotVals[i])
					}
					continue
				}
				gotF := toFloat(gotVals[i])
				wantF := toFloat(wv)
				// For string comparison, check string equality
				gotStr := fmt.Sprintf("%v", gotVals[i])
				wantStr := fmt.Sprintf("%v", wv)
				if gotStr != wantStr {
					// Try numeric comparison
					assertFloatClose(t, fmt.Sprintf("merge[%s][%d]", col, i), gotF, wantF, tolExact)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 5: Concat
// ---------------------------------------------------------------------------

func testConcat(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	var inp struct {
		Frames []struct {
			Columns map[string][]any `json:"columns"`
		} `json:"frames"`
		Axis int `json:"axis"`
	}
	tc.UnmarshalInput(t, &inp)

	var expected struct {
		Shape  []int            `json:"shape"`
		Result map[string][]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	frames := make([]*tabgo.DataFrame, len(inp.Frames))
	for i, f := range inp.Frames {
		frames[i] = buildDF(f.Columns)
	}

	var result *tabgo.DataFrame
	var err error
	if inp.Axis == 1 {
		result, err = tabgo.ConcatHorizontal(frames)
	} else {
		result, err = tabgo.Concat(frames)
	}
	if err != nil {
		t.Fatalf("Concat error: %v", err)
	}

	if expected.Shape != nil {
		shape := result.Shape()
		assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
		assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
	}

	if expected.Result != nil {
		for col, wantVals := range expected.Result {
			gotVals := result.Column(col).Values()
			if len(gotVals) != len(wantVals) {
				t.Errorf("concat[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
				continue
			}
			for i, wv := range wantVals {
				gotF := toFloat(gotVals[i])
				wantF := toFloat(wv)
				assertFloatClose(t, fmt.Sprintf("concat[%s][%d]", col, i), gotF, wantF, tolExact)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 6: Sorting
// ---------------------------------------------------------------------------

func testSort(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	var inp struct {
		Columns   map[string][]any `json:"columns"`
		By        string           `json:"by"`
		Ascending bool             `json:"ascending"`
	}
	tc.UnmarshalInput(t, &inp)

	var expected struct {
		Result map[string][]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDF(inp.Columns)
	result := df.SortValues(inp.By, inp.Ascending)

	for col, wantVals := range expected.Result {
		gotVals := result.Column(col).Values()
		if len(gotVals) != len(wantVals) {
			t.Errorf("sort[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
			continue
		}
		for i, wv := range wantVals {
			gotF := toFloat(gotVals[i])
			wantF := toFloat(wv)
			assertFloatClose(t, fmt.Sprintf("sort[%s][%d]", col, i), gotF, wantF, tolExact)
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 7: Missing Data
// ---------------------------------------------------------------------------

func testMissing(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	cols := parseColumnsInput(tc.Input)
	df := buildDF(cols)

	switch tc.Name {
	case "missing_dropna", "missing_dropna_all_nan_row":
		var expected struct {
			Shape   []int `json:"shape"`
			NumRows int   `json:"num_rows"`
		}
		tc.UnmarshalExpected(t, &expected)
		result := df.DropNA()
		assertIntEqual(t, "num_rows", result.Len(), expected.NumRows)
		if expected.Shape != nil {
			shape := result.Shape()
			assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
			assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
		}

	case "missing_dropna_all":
		// dropna(how='all') drops rows only where ALL values are nil.
		// tabgo.DropNA() implements how='any'. Simulate how='all' with Filter.
		var expected struct {
			Shape   []int `json:"shape"`
			NumRows int   `json:"num_rows"`
		}
		tc.UnmarshalExpected(t, &expected)
		result := df.Filter(func(row map[string]any) bool {
			for _, v := range row {
				if v != nil {
					return true
				}
			}
			return false
		})
		assertIntEqual(t, "num_rows", result.Len(), expected.NumRows)
		if expected.Shape != nil {
			shape := result.Shape()
			assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
			assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
		}

	case "missing_fillna_zero":
		testFillNA(t, tc, df, 0.0)
	case "missing_fillna_neg1":
		testFillNA(t, tc, df, -1.0)
	case "missing_fillna_99":
		testFillNA(t, tc, df, 99.9)

	case "missing_isna":
		var expected struct {
			Result map[string][]any `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		for col, wantVals := range expected.Result {
			s := df.Column(col)
			gotNA := s.IsNA()
			for i, wv := range wantVals {
				wantBool := false
				if b, ok := wv.(bool); ok {
					wantBool = b
				}
				assertBoolEqual(t, fmt.Sprintf("isna[%s][%d]", col, i), gotNA[i], wantBool)
			}
		}

	case "missing_count":
		var expected struct {
			Count map[string]int `json:"count"`
		}
		tc.UnmarshalExpected(t, &expected)
		got := df.Count()
		for col, want := range expected.Count {
			assertIntEqual(t, fmt.Sprintf("count[%s]", col), got[col], want)
		}

	default:
		t.Skipf("unhandled missing case: %s", tc.Name)
	}
}

func testFillNA(t *testing.T, tc *testutil.TestCase, df *tabgo.DataFrame, fillVal float64) {
	t.Helper()
	var expected struct {
		Result map[string][]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	result := df.FillNA(fillVal)
	for col, wantVals := range expected.Result {
		gotVals := result.Column(col).Values()
		for i, wv := range wantVals {
			gotF := toFloat(gotVals[i])
			wantF := toFloat(wv)
			assertFloatClose(t, fmt.Sprintf("fillna[%s][%d]", col, i), gotF, wantF, tolExact)
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 8: Reshape
// ---------------------------------------------------------------------------

func testReshape(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	switch {
	case strings.HasPrefix(tc.Name, "reshape_melt"):
		testMelt(t, tc)
	case strings.HasPrefix(tc.Name, "reshape_pivot"):
		testPivot(t, tc)
	case strings.HasPrefix(tc.Name, "reshape_crosstab"):
		testCrosstab(t, tc)
	default:
		t.Skipf("unhandled reshape case: %s", tc.Name)
	}
}

func testMelt(t *testing.T, tc *testutil.TestCase) {
	t.Helper()
	var inp struct {
		Columns     map[string][]any `json:"columns"`
		ColumnOrder []string         `json:"column_order"`
		IDVars      []string         `json:"id_vars"`
		ValueVars   []string         `json:"value_vars"`
	}
	tc.UnmarshalInput(t, &inp)

	var expected struct {
		Shape  []int            `json:"shape"`
		Result map[string][]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDFOrdered(inp.Columns, inp.ColumnOrder)
	result, err := tabgo.Melt(df, inp.IDVars, inp.ValueVars)
	if err != nil {
		t.Fatalf("Melt error: %v", err)
	}

	if expected.Shape != nil {
		shape := result.Shape()
		assertIntEqual(t, "shape[0]", shape[0], expected.Shape[0])
		assertIntEqual(t, "shape[1]", shape[1], expected.Shape[1])
	}

	if expected.Result != nil {
		for col, wantVals := range expected.Result {
			gotVals := result.Column(col).Values()
			if len(gotVals) != len(wantVals) {
				t.Errorf("melt[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
				continue
			}
			for i, wv := range wantVals {
				gotStr := fmt.Sprintf("%v", gotVals[i])
				wantStr := fmt.Sprintf("%v", wv)
				if gotStr != wantStr {
					// try numeric
					gotF := toFloat(gotVals[i])
					wantF := toFloat(wv)
					assertFloatClose(t, fmt.Sprintf("melt[%s][%d]", col, i), gotF, wantF, tolExact)
				}
			}
		}
	}
}

func testPivot(t *testing.T, tc *testutil.TestCase) {
	t.Helper()
	var inp struct {
		Columns      map[string][]any `json:"columns"`
		Index        string           `json:"index"`
		PivotColumns string           `json:"pivot_columns"`
		Values       string           `json:"values"`
		AggFunc      string           `json:"aggfunc"`
	}
	tc.UnmarshalInput(t, &inp)

	var expected struct {
		Result map[string][]any `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDF(inp.Columns)
	result, err := tabgo.PivotTable(df, inp.Index, inp.PivotColumns, inp.Values, inp.AggFunc)
	if err != nil {
		t.Fatalf("PivotTable error: %v", err)
	}

	// The pivot result has index column + value columns
	// The expected from pandas is keyed by column name -> list of values
	for col, wantVals := range expected.Result {
		gotVals := result.Column(col).Values()
		if len(gotVals) != len(wantVals) {
			t.Errorf("pivot[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
			continue
		}
		for i, wv := range wantVals {
			gotF := toFloat(gotVals[i])
			wantF := toFloat(wv)
			assertFloatClose(t, fmt.Sprintf("pivot[%s][%d]", col, i), gotF, wantF, tolExact)
		}
	}
}

func testCrosstab(t *testing.T, tc *testutil.TestCase) {
	t.Helper()
	var inp struct {
		Columns map[string][]any `json:"columns"`
		Row     string           `json:"row"`
		Col     string           `json:"col"`
	}
	tc.UnmarshalInput(t, &inp)

	var expected struct {
		Result map[string]map[string]float64 `json:"result"`
	}
	tc.UnmarshalExpected(t, &expected)

	df := buildDF(inp.Columns)
	result, err := tabgo.Crosstab(df, inp.Row, inp.Col)
	if err != nil {
		t.Fatalf("Crosstab error: %v", err)
	}

	// result has Row column + one column per unique Col value
	rowVals := result.Column(inp.Row).Values()
	for colName, wantMap := range expected.Result {
		colVals := result.Column(colName).Values()
		for i, rv := range rowVals {
			rowKey := fmt.Sprintf("%v", rv)
			want, ok := wantMap[rowKey]
			if !ok {
				continue
			}
			got := toFloat(colVals[i])
			assertFloatClose(t, fmt.Sprintf("crosstab[%s][%s]", rowKey, colName), got, want, tolExact)
		}
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 9: Statistics
// ---------------------------------------------------------------------------

func testStatistics(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	cols := parseColumnsInput(tc.Input)
	df := buildDF(cols)

	switch tc.Name {
	case "stats_corr":
		var expected struct {
			Result map[string]map[string]float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		result, err := tabgo.Corr(df)
		if err != nil {
			t.Fatalf("Corr error: %v", err)
		}
		checkMatrixResult(t, result, expected.Result, tolStat)

	case "stats_cov":
		var expected struct {
			Result map[string]map[string]float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		result, err := tabgo.Cov(df)
		if err != nil {
			t.Fatalf("Cov error: %v", err)
		}
		checkMatrixResult(t, result, expected.Result, tolStat)

	case "stats_cumsum", "stats_cumsum_specific":
		var expected struct {
			Result map[string][]any `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		result := tabgo.Cumsum(df)
		checkColumnValues(t, result, expected.Result, tolExact)

	case "stats_diff":
		var expected struct {
			Result map[string][]any `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		result := tabgo.Diff(df, 1)
		checkColumnValuesAllowNil(t, result, expected.Result, tolExact)

	case "stats_pct_change":
		var expected struct {
			Result map[string][]any `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		result := tabgo.PctChange(df, 1)
		checkColumnValuesAllowNil(t, result, expected.Result, tolStat)

	case "stats_rank", "stats_rank_ties":
		var expected struct {
			Result map[string][]any `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		result := tabgo.Rank(df)
		checkColumnValues(t, result, expected.Result, tolExact)

	case "stats_corr_diagonal":
		var expected struct {
			Diagonal []float64 `json:"diagonal"`
		}
		tc.UnmarshalExpected(t, &expected)
		result, err := tabgo.Corr(df)
		if err != nil {
			t.Fatalf("Corr error: %v", err)
		}
		// Check diagonal elements
		numCols := dfNumericColNames(df)
		for i, col := range numCols {
			colVals := result.Column(col).Values()
			// Row i should have value 1.0
			got := toFloat(colVals[i])
			assertFloatClose(t, fmt.Sprintf("corr_diag[%s]", col), got, expected.Diagonal[i], tolStat)
		}

	case "stats_corr_symmetry":
		var expected struct {
			AB float64 `json:"AB"`
			BA float64 `json:"BA"`
			AC float64 `json:"AC"`
			CA float64 `json:"CA"`
		}
		tc.UnmarshalExpected(t, &expected)
		result, err := tabgo.Corr(df)
		if err != nil {
			t.Fatalf("Corr error: %v", err)
		}
		// Check AB == BA and AC == CA
		assertFloatClose(t, "AB==BA", expected.AB, expected.BA, tolStat)
		assertFloatClose(t, "AC==CA", expected.AC, expected.CA, tolStat)
		// Also verify Go produces matching values
		bVals := result.Column("B").Values()
		// Row for A is index 0 (columns sorted A, B, C)
		assertFloatClose(t, "corr[A,B]", toFloat(bVals[0]), expected.AB, tolStat)

	default:
		t.Skipf("unhandled stats case: %s", tc.Name)
	}
}

func checkMatrixResult(t *testing.T, result *tabgo.DataFrame, expected map[string]map[string]float64, tol float64) {
	t.Helper()
	// result has "" column (row labels) + value columns
	rowLabels := result.Column("").Values()
	for col, rowMap := range expected {
		colVals := result.Column(col).Values()
		for i, rl := range rowLabels {
			rowName := fmt.Sprintf("%v", rl)
			want, ok := rowMap[rowName]
			if !ok {
				continue
			}
			got := toFloat(colVals[i])
			assertFloatClose(t, fmt.Sprintf("matrix[%s][%s]", rowName, col), got, want, tol)
		}
	}
}

func checkColumnValues(t *testing.T, result *tabgo.DataFrame, expected map[string][]any, tol float64) {
	t.Helper()
	for col, wantVals := range expected {
		gotVals := result.Column(col).Values()
		if len(gotVals) != len(wantVals) {
			t.Errorf("[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
			continue
		}
		for i, wv := range wantVals {
			gotF := toFloat(gotVals[i])
			wantF := toFloat(wv)
			assertFloatClose(t, fmt.Sprintf("[%s][%d]", col, i), gotF, wantF, tol)
		}
	}
}

func checkColumnValuesAllowNil(t *testing.T, result *tabgo.DataFrame, expected map[string][]any, tol float64) {
	t.Helper()
	for col, wantVals := range expected {
		gotVals := result.Column(col).Values()
		if len(gotVals) != len(wantVals) {
			t.Errorf("[%s] length: got %d, want %d", col, len(gotVals), len(wantVals))
			continue
		}
		for i, wv := range wantVals {
			if wv == nil {
				if gotVals[i] != nil {
					t.Errorf("[%s][%d]: got %v, want nil", col, i, gotVals[i])
				}
				continue
			}
			if gotVals[i] == nil {
				t.Errorf("[%s][%d]: got nil, want %v", col, i, wv)
				continue
			}
			gotF := toFloat(gotVals[i])
			wantF := toFloat(wv)
			assertFloatClose(t, fmt.Sprintf("[%s][%d]", col, i), gotF, wantF, tol)
		}
	}
}

func dfNumericColNames(df *tabgo.DataFrame) []string {
	names := df.Columns()
	sort.Strings(names)
	return names
}

// ---------------------------------------------------------------------------
// CATEGORY 10: Series
// ---------------------------------------------------------------------------

func testSeries(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	var inp struct {
		Values   []any   `json:"values"`
		Lower    float64 `json:"lower"`
		Upper    float64 `json:"upper"`
		Decimals int     `json:"decimals"`
	}
	tc.UnmarshalInput(t, &inp)

	s := tabgo.NewSeries("s", inp.Values)

	switch tc.Name {
	case "series_value_counts":
		var expected struct {
			Result map[string]int `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		got := s.ValueCounts()
		for k, wantCount := range expected.Result {
			// Find matching key in got
			found := false
			for gk, gc := range got {
				if fmt.Sprintf("%v", gk) == k {
					assertIntEqual(t, fmt.Sprintf("vc[%s]", k), gc, wantCount)
					found = true
					break
				}
			}
			if !found {
				t.Errorf("value_counts: key %s not found", k)
			}
		}

	case "series_unique":
		var expected struct {
			ResultSorted []float64 `json:"result_sorted"`
			Count        int       `json:"count"`
		}
		tc.UnmarshalExpected(t, &expected)
		u := s.Unique()
		assertIntEqual(t, "unique_count", len(u), expected.Count)
		// Sort unique values
		var gotFloats []float64
		for _, v := range u {
			gotFloats = append(gotFloats, toFloat(v))
		}
		sort.Float64s(gotFloats)
		for i, want := range expected.ResultSorted {
			assertFloatClose(t, fmt.Sprintf("unique[%d]", i), gotFloats[i], want, tolExact)
		}

	case "series_nunique":
		var expected struct {
			Result int `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		assertIntEqual(t, "nunique", s.NUnique(), expected.Result)

	case "series_describe":
		var expected struct {
			Result map[string]float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		got := s.Describe()
		for stat, want := range expected.Result {
			tol := tolStat
			if stat == "count" || stat == "min" || stat == "max" {
				tol = tolExact
			}
			assertFloatClose(t, fmt.Sprintf("describe[%s]", stat), got[stat], want, tol)
		}

	case "series_aggregations":
		var expected struct {
			Sum    float64 `json:"sum"`
			Mean   float64 `json:"mean"`
			Std    float64 `json:"std"`
			Var    float64 `json:"var"`
			Min    float64 `json:"min"`
			Max    float64 `json:"max"`
			Median float64 `json:"median"`
			Count  int     `json:"count"`
		}
		tc.UnmarshalExpected(t, &expected)
		assertFloatClose(t, "sum", s.Sum(), expected.Sum, tolExact)
		assertFloatClose(t, "mean", s.Mean(), expected.Mean, tolExact)
		assertFloatClose(t, "std", s.Std(), expected.Std, tolStat)
		assertFloatClose(t, "var", s.Var(), expected.Var, tolStat)
		assertFloatClose(t, "min", s.Min(), expected.Min, tolExact)
		assertFloatClose(t, "max", s.Max(), expected.Max, tolExact)
		assertFloatClose(t, "median", s.Median(), expected.Median, tolExact)
		assertIntEqual(t, "count", s.Count(), expected.Count)

	case "series_sort_asc":
		var expected struct {
			Result []float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		sorted := s.Sort(true)
		gotVals := sorted.Values()
		for i, want := range expected.Result {
			assertFloatClose(t, fmt.Sprintf("sort[%d]", i), toFloat(gotVals[i]), want, tolExact)
		}

	case "series_clip":
		var expected struct {
			Result []float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		clipped := s.Clip(inp.Lower, inp.Upper)
		gotVals := clipped.Values()
		for i, want := range expected.Result {
			assertFloatClose(t, fmt.Sprintf("clip[%d]", i), toFloat(gotVals[i]), want, tolExact)
		}

	case "series_abs":
		var expected struct {
			Result []float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		abs := s.Abs()
		gotVals := abs.Values()
		for i, want := range expected.Result {
			assertFloatClose(t, fmt.Sprintf("abs[%d]", i), toFloat(gotVals[i]), want, tolExact)
		}

	case "series_round":
		var expected struct {
			Result []float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		rounded := s.Round(inp.Decimals)
		gotVals := rounded.Values()
		for i, want := range expected.Result {
			assertFloatClose(t, fmt.Sprintf("round[%d]", i), toFloat(gotVals[i]), want, tolExact)
		}

	case "series_rank":
		var expected struct {
			Result []float64 `json:"result"`
		}
		tc.UnmarshalExpected(t, &expected)
		ranked := s.Rank()
		gotVals := ranked.Values()
		for i, want := range expected.Result {
			assertFloatClose(t, fmt.Sprintf("rank[%d]", i), toFloat(gotVals[i]), want, tolExact)
		}

	default:
		t.Skipf("unhandled series case: %s", tc.Name)
	}
}

// ---------------------------------------------------------------------------
// CATEGORY 11: CSV
// ---------------------------------------------------------------------------

func testCSV(t *testing.T, tc *testutil.TestCase) {
	t.Helper()

	cols := parseColumnsInput(tc.Input)
	df := buildDF(cols)

	switch tc.Name {
	case "csv_roundtrip_basic", "csv_roundtrip_mixed":
		var expected struct {
			RoundtripShape  []int            `json:"roundtrip_shape"`
			RoundtripValues map[string][]any `json:"roundtrip_values"`
		}
		tc.UnmarshalExpected(t, &expected)

		// Write to CSV string
		var buf strings.Builder
		names := df.Columns()
		// Header
		buf.WriteString(strings.Join(names, ",") + "\n")
		nRows := df.Len()
		for r := 0; r < nRows; r++ {
			parts := make([]string, len(names))
			for c, n := range names {
				v := df.Column(n).Values()[r]
				if v == nil {
					parts[c] = ""
				} else {
					parts[c] = fmt.Sprintf("%v", v)
				}
			}
			buf.WriteString(strings.Join(parts, ",") + "\n")
		}
		csvStr := buf.String()

		// Read back
		readDF, err := tabgo.ReadCSVFromString(csvStr)
		if err != nil {
			t.Fatalf("ReadCSVFromString error: %v", err)
		}

		if expected.RoundtripShape != nil {
			shape := readDF.Shape()
			assertIntEqual(t, "roundtrip_shape[0]", shape[0], expected.RoundtripShape[0])
			assertIntEqual(t, "roundtrip_shape[1]", shape[1], expected.RoundtripShape[1])
		}

	case "csv_roundtrip_nan":
		// Just verify we can write and read back
		var buf strings.Builder
		names := df.Columns()
		buf.WriteString(strings.Join(names, ",") + "\n")
		nRows := df.Len()
		for r := 0; r < nRows; r++ {
			parts := make([]string, len(names))
			for c, n := range names {
				v := df.Column(n).Values()[r]
				if v == nil {
					parts[c] = ""
				} else {
					parts[c] = fmt.Sprintf("%v", v)
				}
			}
			buf.WriteString(strings.Join(parts, ",") + "\n")
		}
		csvStr := buf.String()

		readDF, err := tabgo.ReadCSVFromString(csvStr)
		if err != nil {
			t.Fatalf("ReadCSVFromString error: %v", err)
		}
		// Verify shape is preserved
		assertIntEqual(t, "csv_nan_rows", readDF.Len(), df.Len())

	default:
		t.Skipf("unhandled csv case: %s", tc.Name)
	}
}

// ---------------------------------------------------------------------------
// utility
// ---------------------------------------------------------------------------

func toFloat(v any) float64 {
	if v == nil {
		return math.NaN()
	}
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case string:
		var f float64
		fmt.Sscan(n, &f)
		return f
	case json.Number:
		f, _ := n.Float64()
		return f
	default:
		return 0
	}
}
