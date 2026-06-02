//go:build unit

package datasets

import (
	"testing"
)

func TestAsia(t *testing.T) {
	df, err := Asia()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 1000 {
		t.Errorf("Asia: expected 1000 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 8 {
		t.Errorf("Asia: expected 8 columns, got %d", len(cols))
	}
}

func TestAlarm(t *testing.T) {
	df, err := Alarm()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 1000 {
		t.Errorf("Alarm: expected 1000 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 5 {
		t.Errorf("Alarm: expected 5 columns, got %d", len(cols))
	}
}

func TestSachs(t *testing.T) {
	df, err := Sachs()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 500 {
		t.Errorf("Sachs: expected 500 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 11 {
		t.Errorf("Sachs: expected 11 columns, got %d", len(cols))
	}
}

func TestCancer(t *testing.T) {
	df, err := Cancer()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 1000 {
		t.Errorf("Cancer: expected 1000 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 5 {
		t.Errorf("Cancer: expected 5 columns, got %d", len(cols))
	}
}

func TestStudent(t *testing.T) {
	df, err := Student()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 1000 {
		t.Errorf("Student: expected 1000 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 5 {
		t.Errorf("Student: expected 5 columns, got %d", len(cols))
	}
}

func TestSprinkler(t *testing.T) {
	df, err := Sprinkler()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 1000 {
		t.Errorf("Sprinkler: expected 1000 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 4 {
		t.Errorf("Sprinkler: expected 4 columns, got %d", len(cols))
	}
}

func TestSurvey(t *testing.T) {
	df, err := Survey()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 500 {
		t.Errorf("Survey: expected 500 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 5 {
		t.Errorf("Survey: expected 5 columns, got %d", len(cols))
	}
}

func TestTitanic(t *testing.T) {
	df, err := Titanic()
	if err != nil {
		t.Fatal(err)
	}
	if df.Len() != 800 {
		t.Errorf("Titanic: expected 800 rows, got %d", df.Len())
	}
	cols := df.Columns()
	if len(cols) != 4 {
		t.Errorf("Titanic: expected 4 columns, got %d", len(cols))
	}
}

func TestList(t *testing.T) {
	names := List()
	expected := []string{"alarm", "asia", "cancer", "sachs", "sprinkler", "student", "survey", "titanic"}
	if len(names) != len(expected) {
		t.Fatalf("List: expected %d datasets, got %d", len(expected), len(names))
	}
	for i, name := range expected {
		if names[i] != name {
			t.Errorf("List[%d]: expected %q, got %q", i, name, names[i])
		}
	}
}

func TestLoad(t *testing.T) {
	for _, name := range List() {
		df, err := Load(name)
		if err != nil {
			t.Errorf("Load(%q): %v", name, err)
			continue
		}
		if df.Len() == 0 {
			t.Errorf("Load(%q): got 0 rows", name)
		}
	}
}

func TestLoadUnknown(t *testing.T) {
	_, err := Load("nonexistent")
	if err == nil {
		t.Error("Load(nonexistent): expected error, got nil")
	}
}
