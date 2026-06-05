//go:build unit

package blas

import (
	"runtime"
	"testing"
)

func TestUseBLAS_Level1_BelowThreshold(t *testing.T) {
	if UseBLAS(1, 63) {
		t.Error("expected false for size below Level1Threshold")
	}
}

func TestUseBLAS_Level1_AtThreshold(t *testing.T) {
	if !UseBLAS(1, 64) {
		t.Error("expected true for size at Level1Threshold")
	}
}

func TestUseBLAS_Level1_AboveThreshold(t *testing.T) {
	if !UseBLAS(1, 1000) {
		t.Error("expected true for size above Level1Threshold")
	}
}

func TestUseBLAS_Level2_BelowThreshold(t *testing.T) {
	if UseBLAS(2, 31) {
		t.Error("expected false for size below Level2Threshold")
	}
}

func TestUseBLAS_Level2_AtThreshold(t *testing.T) {
	if !UseBLAS(2, 32) {
		t.Error("expected true for size at Level2Threshold")
	}
}

func TestUseBLAS_Level3_BelowThreshold(t *testing.T) {
	if UseBLAS(3, 63) {
		t.Error("expected false for size below Level3Threshold")
	}
}

func TestUseBLAS_Level3_AtThreshold(t *testing.T) {
	if !UseBLAS(3, 64) {
		t.Error("expected true for size at Level3Threshold")
	}
}

func TestUseBLAS_InvalidLevel(t *testing.T) {
	if UseBLAS(0, 1000) {
		t.Error("expected false for invalid level 0")
	}
	if UseBLAS(4, 1000) {
		t.Error("expected false for invalid level 4")
	}
	if UseBLAS(-1, 1000) {
		t.Error("expected false for negative level")
	}
}

func TestUseBLAS_ZeroSize(t *testing.T) {
	if UseBLAS(1, 0) {
		t.Error("expected false for zero size")
	}
}

func TestUseBLAS_NegativeSize(t *testing.T) {
	if UseBLAS(1, -1) {
		t.Error("expected false for negative size")
	}
}

func TestThresholdConstants(t *testing.T) {
	if Level1Threshold != 64 {
		t.Errorf("Level1Threshold = %d, want 64", Level1Threshold)
	}
	if Level2Threshold != 32 {
		t.Errorf("Level2Threshold = %d, want 32", Level2Threshold)
	}
	if Level3Threshold != 64 {
		t.Errorf("Level3Threshold = %d, want 64", Level3Threshold)
	}
}

func TestHasAVX2(t *testing.T) {
	got := HasAVX2()
	if runtime.GOARCH == "amd64" && !got {
		t.Error("expected HasAVX2() == true on amd64")
	}
	if runtime.GOARCH != "amd64" && got {
		t.Error("expected HasAVX2() == false on non-amd64")
	}
}
