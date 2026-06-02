//go:build unit

package example_models

import "testing"

func TestStudentCheckModel(t *testing.T) {
	bn := Student()
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("Student CheckModel failed: %v", err)
	}
	if len(bn.Nodes()) != 5 {
		t.Fatalf("Student should have 5 nodes, got %d", len(bn.Nodes()))
	}
	if len(bn.Edges()) != 4 {
		t.Fatalf("Student should have 4 edges, got %d", len(bn.Edges()))
	}
}

func TestAsiaCheckModel(t *testing.T) {
	bn := Asia()
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("Asia CheckModel failed: %v", err)
	}
	if len(bn.Nodes()) != 8 {
		t.Fatalf("Asia should have 8 nodes, got %d", len(bn.Nodes()))
	}
	if len(bn.Edges()) != 8 {
		t.Fatalf("Asia should have 8 edges, got %d", len(bn.Edges()))
	}
}

func TestAlarmCheckModel(t *testing.T) {
	bn := Alarm()
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("Alarm CheckModel failed: %v", err)
	}
	if len(bn.Nodes()) != 5 {
		t.Fatalf("Alarm should have 5 nodes, got %d", len(bn.Nodes()))
	}
	if len(bn.Edges()) != 4 {
		t.Fatalf("Alarm should have 4 edges, got %d", len(bn.Edges()))
	}
}

func TestCancerCheckModel(t *testing.T) {
	bn := Cancer()
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("Cancer CheckModel failed: %v", err)
	}
	if len(bn.Nodes()) != 5 {
		t.Fatalf("Cancer should have 5 nodes, got %d", len(bn.Nodes()))
	}
	if len(bn.Edges()) != 4 {
		t.Fatalf("Cancer should have 4 edges, got %d", len(bn.Edges()))
	}
}

func TestSachsStructure(t *testing.T) {
	bn := Sachs()
	if len(bn.Nodes()) != 11 {
		t.Fatalf("Sachs should have 11 nodes, got %d", len(bn.Nodes()))
	}
	if len(bn.Edges()) != 17 {
		t.Fatalf("Sachs should have 17 edges, got %d", len(bn.Edges()))
	}
	// Sachs has no CPDs, so CheckModel should fail
	if err := bn.CheckModel(); err == nil {
		t.Fatal("Sachs CheckModel should fail (no CPDs)")
	}
}

func TestWaterSprinklerCheckModel(t *testing.T) {
	bn := WaterSprinkler()
	if err := bn.CheckModel(); err != nil {
		t.Fatalf("WaterSprinkler CheckModel failed: %v", err)
	}
	if len(bn.Nodes()) != 4 {
		t.Fatalf("WaterSprinkler should have 4 nodes, got %d", len(bn.Nodes()))
	}
	if len(bn.Edges()) != 4 {
		t.Fatalf("WaterSprinkler should have 4 edges, got %d", len(bn.Edges()))
	}
}
