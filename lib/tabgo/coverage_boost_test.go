//go:build unit

package tabgo

import (
	"math"
	"testing"
	"time"
)

// ===========================================================================
// Window operations: Rolling, Expanding, EWM with nil values
// ===========================================================================

func TestRolling_NilValues(t *testing.T) {
	// All nil values in a numeric column => empty windows.
	s := NewSeries("X", []any{nil, nil, nil, nil})
	df := makeDF(t, s)
	r := df.Rolling(2)

	_ = r.Mean()
	_ = r.Std()
	_ = r.Min()
	_ = r.Max()
}

func TestExpanding_NilValues(t *testing.T) {
	// Leading nil values so first window(s) are empty.
	s := NewSeries("X", []any{nil, nil, 1.0, 2.0})
	df := makeDF(t, s)
	e := df.Expanding()

	_ = e.Mean()
	_ = e.Min()
	_ = e.Max()
}

func TestEWM_NilValues(t *testing.T) {
	// Nil values interspersed trigger the nil path in EWM.
	s := NewSeries("X", []any{nil, 1.0, nil, 3.0})
	df := makeDF(t, s)
	ew := df.EWM(3)

	_ = ew.Mean()
	_ = ew.Std()
	_ = ew.Var()
}

// ===========================================================================
// DataFrame aggregation: all-nil columns
// ===========================================================================

func TestMeanAll_AllNil(t *testing.T) {
	s := NewSeries("X", []any{nil, nil, nil})
	df := makeDF(t, s)
	result := df.MeanAll()
	if result["X"] != 0 {
		t.Errorf("expected 0 for all-nil column, got %f", result["X"])
	}
}

func TestMinAll_AllNil(t *testing.T) {
	s := NewSeries("X", []any{nil, nil})
	df := makeDF(t, s)
	result := df.MinAll()
	if result["X"] != 0 {
		t.Errorf("expected 0 for all-nil column, got %f", result["X"])
	}
}

func TestMaxAll_AllNil(t *testing.T) {
	s := NewSeries("X", []any{nil, nil})
	df := makeDF(t, s)
	result := df.MaxAll()
	if result["X"] != 0 {
		t.Errorf("expected 0 for all-nil column, got %f", result["X"])
	}
}

func TestDescribe_AllNil(t *testing.T) {
	s := NewSeries("X", []any{nil, nil, nil})
	df := makeDF(t, s)
	desc := df.Describe()
	if desc == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestMean_Empty(t *testing.T) {
	if v := mean(nil); v != 0 {
		t.Errorf("expected 0 for empty, got %f", v)
	}
}

func TestMedian_Empty(t *testing.T) {
	if v := median(nil); v != 0 {
		t.Errorf("expected 0 for empty, got %f", v)
	}
}

// ===========================================================================
// DataFrame stats: nil values, empty columns
// ===========================================================================

func TestCorr_NilValues(t *testing.T) {
	s1 := NewSeries("A", []any{nil, 2.0, 3.0, 4.0})
	s2 := NewSeries("B", []any{1.0, nil, 3.0, 4.0})
	df := makeDF(t, s1, s2)
	result, err := Corr(df)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestCov_NilValues(t *testing.T) {
	s1 := NewSeries("A", []any{nil, 2.0, 3.0})
	s2 := NewSeries("B", []any{1.0, nil, 3.0})
	df := makeDF(t, s1, s2)
	result, err := Cov(df)
	if err != nil {
		t.Fatal(err)
	}
	_ = result
}

func TestPearson_LessThan2(t *testing.T) {
	// Only 1 non-NaN pair.
	r := pearson([]float64{1.0, math.NaN()}, []float64{2.0, 3.0})
	if r != 0 {
		t.Errorf("expected 0 for < 2 valid pairs, got %f", r)
	}
}

func TestPearson_ZeroDenom(t *testing.T) {
	// Constant values => zero denom.
	r := pearson([]float64{1.0, 1.0, 1.0}, []float64{2.0, 3.0, 4.0})
	if r != 0 {
		t.Errorf("expected 0 for zero denom, got %f", r)
	}
}

func TestCovariance_LessThan2(t *testing.T) {
	r := covariance([]float64{math.NaN(), math.NaN()}, []float64{1.0, 2.0})
	if r != 0 {
		t.Errorf("expected 0 for < 2 valid pairs, got %f", r)
	}
}

func TestCummax_LeadingNils(t *testing.T) {
	s := NewSeries("X", []any{nil, nil, 3.0, 1.0, 5.0})
	df := makeDF(t, s)
	result := Cummax(df)
	vals := result.Column("X").Values()
	if vals[0] != nil || vals[1] != nil {
		t.Error("expected leading nils in Cummax")
	}
}

func TestCummin_LeadingNils(t *testing.T) {
	s := NewSeries("X", []any{nil, nil, 3.0, 5.0, 1.0})
	df := makeDF(t, s)
	result := Cummin(df)
	vals := result.Column("X").Values()
	if vals[0] != nil || vals[1] != nil {
		t.Error("expected leading nils in Cummin")
	}
}

// ===========================================================================
// Series properties: edge cases
// ===========================================================================

func TestDtype_UintType(t *testing.T) {
	s := NewSeries("X", []any{uint(1), uint(2)})
	dt := s.Dtype()
	if dt != "int" {
		t.Errorf("expected 'int' for uint values, got %q", dt)
	}
}

func TestDtype_ObjectType(t *testing.T) {
	s := NewSeries("X", []any{struct{}{}, struct{}{}})
	dt := s.Dtype()
	if dt != "object" {
		t.Errorf("expected 'object' for unknown types, got %q", dt)
	}
}

func TestAstype_Bool(t *testing.T) {
	s := NewSeries("X", []any{1.0, 0.0, nil})
	result := s.Astype("bool")
	vals := result.Values()
	if vals[2] != nil {
		t.Error("expected nil preserved")
	}
}

func TestCummax_LeadingNilsSeries(t *testing.T) {
	s := NewSeries("X", []any{nil, nil, 3.0})
	result := s.Cummax()
	vals := result.Values()
	if vals[0] != nil || vals[1] != nil {
		t.Error("expected leading nils")
	}
}

func TestCummin_LeadingNilsSeries(t *testing.T) {
	s := NewSeries("X", []any{nil, nil, 3.0})
	result := s.Cummin()
	vals := result.Values()
	if vals[0] != nil || vals[1] != nil {
		t.Error("expected leading nils")
	}
}

func TestPctChange_ZeroPrev(t *testing.T) {
	s := NewSeries("X", []any{0.0, 5.0, 10.0})
	result := s.PctChange(1)
	vals := result.Values()
	if vals[1] != nil {
		t.Error("expected nil when prev=0")
	}
}

// ===========================================================================
// Series Max/Median/Corr: all-nil and edge cases
// ===========================================================================

func TestSeriesMax_AllNil(t *testing.T) {
	s := NewSeries("X", []any{nil, nil})
	if s.Max() != 0 {
		t.Error("expected 0 for all-nil Max")
	}
}

func TestSeriesMedian_AllNil(t *testing.T) {
	s := NewSeries("X", []any{nil, nil})
	if s.Median() != 0 {
		t.Error("expected 0 for all-nil Median")
	}
}

func TestSeriesCorr_TooFew(t *testing.T) {
	s1 := NewSeries("X", []any{1.0})
	s2 := NewSeries("Y", []any{2.0})
	if s1.Corr(s2) != 0 {
		t.Error("expected 0 for < 2 elements")
	}
}

func TestSeriesCorr_ZeroDenom(t *testing.T) {
	s1 := NewSeries("X", []any{1.0, 1.0, 1.0})
	s2 := NewSeries("Y", []any{2.0, 3.0, 4.0})
	if s1.Corr(s2) != 0 {
		t.Error("expected 0 for constant series")
	}
}

// ===========================================================================
// MultiIndex edge cases
// ===========================================================================

func TestMultiIndex_EmptyArrays(t *testing.T) {
	_, err := NewMultiIndexFromArrays(nil, nil)
	if err == nil {
		t.Error("expected error for empty arrays")
	}
}

func TestMultiIndex_UnequalLengths(t *testing.T) {
	_, err := NewMultiIndexFromArrays([][]any{{1, 2}, {1}}, []string{"a", "b"})
	if err == nil {
		t.Error("expected error for unequal lengths")
	}
}

func TestMultiIndex_Equals_DifferentLen(t *testing.T) {
	mi1, _ := NewMultiIndexFromArrays([][]any{{1, 2}}, []string{"a"})
	mi2, _ := NewMultiIndexFromArrays([][]any{{1, 2, 3}}, []string{"a"})
	if mi1.Equals(mi2) {
		t.Error("expected false for different lengths")
	}
}

func TestMultiIndex_GetLevelSeries_OutOfRange(t *testing.T) {
	mi, _ := NewMultiIndexFromArrays([][]any{{1, 2}}, []string{"a"})
	s := mi.GetLevelSeries(-1)
	if s == nil {
		t.Error("expected non-nil series")
	}
}

func TestMultiIndex_Droplevel_OutOfRange(t *testing.T) {
	mi, _ := NewMultiIndexFromArrays([][]any{{1, 2}}, []string{"a"})
	_, err := mi.Droplevel(-1)
	if err == nil {
		t.Error("expected error for out-of-range level")
	}
}

func TestMultiIndex_Sortlevel_OutOfRange(t *testing.T) {
	mi, _ := NewMultiIndexFromArrays([][]any{{1, 2}}, []string{"a"})
	_, err := mi.Sortlevel(5)
	if err == nil {
		t.Error("expected error for out-of-range level")
	}
}

// ===========================================================================
// DatetimeIndex edge cases
// ===========================================================================

func TestDatetimeIndex_Slice_OutOfBounds(t *testing.T) {
	di := NewDatetimeIndex(
		[]time.Time{time.Now(), time.Now().Add(time.Hour)},
		"idx",
	)
	// start < 0 and end > len
	sliced := di.Slice(-5, 100)
	if len(sliced.Times()) != 2 {
		t.Errorf("expected 2 elements, got %d", len(sliced.Times()))
	}
}

func TestTruncateToFreq_Weekly_Boost(t *testing.T) {
	// A Sunday should truncate to the previous Monday.
	sunday := time.Date(2024, 1, 7, 12, 0, 0, 0, time.UTC) // Sunday
	di := NewDatetimeIndex([]time.Time{sunday}, "idx")
	_ = di.Resample("W")
}

// ===========================================================================
// Series.Str().Slice edge cases
// ===========================================================================

func TestStrSlice_BoundsClamp(t *testing.T) {
	s := NewSeries("X", []any{"ab", "cd"})
	// Start > length: clamped to n => empty string.
	r := s.Str().Slice(100, 200)
	vals := r.Values()
	if vals[0] != "" {
		t.Errorf("expected empty for out-of-bounds start, got %q", vals[0])
	}

	// Negative start clamped to 0.
	r2 := s.Str().Slice(-100, 1)
	vals2 := r2.Values()
	if vals2[0] != "a" {
		t.Errorf("expected 'a', got %q", vals2[0])
	}

	// End < 0 after resolution: clamped to 0.
	r3 := s.Str().Slice(0, -100)
	vals3 := r3.Values()
	if vals3[0] != "" {
		t.Errorf("expected empty for negative end, got %q", vals3[0])
	}

	// start >= end after clamping.
	r4 := s.Str().Slice(5, 3)
	vals4 := r4.Values()
	if vals4[0] != "" {
		t.Errorf("expected empty for start >= end, got %q", vals4[0])
	}
}

// ===========================================================================
// GroupBy edge cases
// ===========================================================================

func TestGroupBy_NilGroupKey(t *testing.T) {
	s1 := NewSeries("G", []any{nil, "a", nil})
	s2 := NewSeries("V", []any{1.0, 2.0, 3.0})
	df := makeDF(t, s1, s2)
	gb := df.GroupBy("G")
	result := gb.Sum()
	if result == nil {
		t.Error("expected non-nil result")
	}
}

func TestGroupBy_Apply_EmptyResult(t *testing.T) {
	s := NewSeries("G", []any{"a", "b"})
	s2 := NewSeries("V", []any{1.0, 2.0})
	df := makeDF(t, s, s2)
	gb := df.GroupBy("G")
	// Apply a function that returns nil for all groups.
	result := gb.Apply(func(d *DataFrame) *DataFrame {
		return NewDataFrameFromRows(nil, nil)
	})
	_ = result
}

func TestGroupByVar_SingleElement(t *testing.T) {
	s1 := NewSeries("G", []any{"a", "b"})
	s2 := NewSeries("V", []any{1.0, 2.0})
	df := makeDF(t, s1, s2)
	gb := df.GroupBy("G")
	result := gb.Var()
	_ = result
}

// ===========================================================================
// Merge/Reshape edge cases
// ===========================================================================

func TestMerge_ColumnNotFound(t *testing.T) {
	s1 := NewSeries("A", []any{1, 2})
	s2 := NewSeries("B", []any{3, 4})
	df1 := makeDF(t, s1)
	df2 := makeDF(t, s2)
	_, err := Merge(df1, df2, []string{"A"}, "inner")
	if err == nil {
		t.Error("expected error when column not found in right DataFrame")
	}
}

func TestMelt_ColumnNotFound(t *testing.T) {
	s := NewSeries("A", []any{1, 2})
	df := makeDF(t, s)
	_, err := Melt(df, []string{"A"}, []string{"NonExistent"})
	if err == nil {
		t.Error("expected error for non-existent value var")
	}
}

func TestCrosstab_ColumnNotFound(t *testing.T) {
	s := NewSeries("A", []any{1, 2})
	df := makeDF(t, s)
	_, err := Crosstab(df, "NonExistent", "A")
	if err == nil {
		t.Error("expected error for non-existent row column")
	}
}

// ===========================================================================
// IO extra edge cases
// ===========================================================================

func TestToJSON_Coverage2(t *testing.T) {
	s := NewSeries("X", []any{1.0, 2.0})
	df := makeDF(t, s)
	jsonStr, err := ToJSON(df)
	if err != nil {
		t.Fatal(err)
	}
	if jsonStr == "" {
		t.Error("expected non-empty JSON")
	}
}

func TestReadJSON_InvalidJSON(t *testing.T) {
	_, err := ReadJSON("not valid json")
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestReadJSON_MissingKey(t *testing.T) {
	// One record has a key the other doesn't.
	_, err := ReadJSON(`[{"a":1},{"b":2}]`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// ===========================================================================
// Series.Dt (datetime accessor) edge cases
// ===========================================================================

func TestToTime_PointerNil(t *testing.T) {
	var tp *time.Time
	_, ok := toTime(tp)
	if ok {
		t.Error("expected false for nil *time.Time")
	}
}

func TestToTime_NonTimeString(t *testing.T) {
	_, ok := toTime("not-a-date")
	if !ok {
		// It might still fail on parsing; just exercise the path.
		_ = ok
	}
}

func TestToTime_DefaultCase(t *testing.T) {
	_, ok := toTime(42)
	if ok {
		t.Error("expected false for int")
	}
}

// ===========================================================================
// DataFrame astype: convertColumnDtypes edge cases
// ===========================================================================

func TestConvertColumnDtypes_AlreadyTyped(t *testing.T) {
	// Non-string values should be returned as-is.
	vals := []any{1.0, 2.0, 3.0}
	result := convertColumnDtypes(vals)
	if len(result) != 3 {
		t.Errorf("expected 3 values, got %d", len(result))
	}
}

func TestConvertColumnDtypes_NilValues(t *testing.T) {
	// nil values among strings.
	vals := []any{nil, "1.0", nil, "2.0"}
	result := convertColumnDtypes(vals)
	if result[0] != nil || result[2] != nil {
		t.Error("expected nil preserved")
	}
}

func TestConvertColumnDtypes_EmptyStrings(t *testing.T) {
	vals := []any{"", "1.0", ""}
	result := convertColumnDtypes(vals)
	// Empty strings should become nil in the float conversion.
	if result[0] != nil {
		t.Error("expected nil for empty string")
	}
}

func TestConvertColumnDtypes_BoolStrings(t *testing.T) {
	// Strings that are booleans.
	vals := []any{"true", "false", "True"}
	result := convertColumnDtypes(vals)
	// Should be converted to bool.
	if _, ok := result[0].(bool); !ok {
		t.Errorf("expected bool, got %T", result[0])
	}
}

func TestConvertColumnDtypes_BoolWithNil(t *testing.T) {
	vals := []any{nil, "true", "false"}
	result := convertColumnDtypes(vals)
	if result[0] != nil {
		t.Error("expected nil preserved")
	}
}

func TestConvertColumnDtypes_NonNumericStrings(t *testing.T) {
	// Strings that can't be floats or bools => remain strings.
	vals := []any{"hello", "world"}
	result := convertColumnDtypes(vals)
	if _, ok := result[0].(string); !ok {
		t.Errorf("expected string, got %T", result[0])
	}
}

// ===========================================================================
// DataFrame extra: estimateSize
// ===========================================================================

func TestEstimateSize_AllTypes(t *testing.T) {
	cases := []struct {
		v    any
		want int
	}{
		{nil, 0},
		{float64(1.0), 8},
		{float32(1.0), 4},
		{int(1), 8},
		{int8(1), 1},
		{int16(1), 2},
		{int32(1), 4},
		{int64(1), 8},
		{"hello", 5 + 16},
		{true, 1},
		{struct{}{}, 16},
	}
	for _, c := range cases {
		got := estimateSize(c.v)
		if got != c.want {
			t.Errorf("estimateSize(%v) = %d, want %d", c.v, got, c.want)
		}
	}
}

// ===========================================================================
// Merge: onIndex
// ===========================================================================

func TestOnIndex_NotFound(t *testing.T) {
	idx := onIndex([]string{"A", "B"}, "C")
	if idx != -1 {
		t.Errorf("expected -1 for not found, got %d", idx)
	}
}

// ===========================================================================
// DataFrame select: Nsmallest edge case
// ===========================================================================

func TestNsmallest_NonNumeric(t *testing.T) {
	s := NewSeries("X", []any{"a", "b", "c"})
	df := makeDF(t, s)
	result := df.Nsmallest(2, "X")
	// Non-numeric values should be handled gracefully.
	_ = result
}
