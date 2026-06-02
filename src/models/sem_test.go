//go:build unit

package models

import (
	"math"
	"testing"
)

func TestNewSEM(t *testing.T) {
	s := NewSEM()
	if s == nil {
		t.Fatal("NewSEM returned nil")
	}
	if len(s.Variables()) != 0 {
		t.Errorf("expected 0 variables, got %d", len(s.Variables()))
	}
}

func TestSEMAddEquation(t *testing.T) {
	s := NewSEM()
	err := s.AddEquation("Y", []string{"X"}, []float64{0.5}, 1.0, 0.25)
	if err != nil {
		t.Fatalf("AddEquation: %v", err)
	}

	// X should have been auto-added.
	vars := s.Variables()
	if len(vars) != 2 {
		t.Fatalf("expected 2 variables, got %d", len(vars))
	}

	eq := s.GetEquation("Y")
	if eq == nil {
		t.Fatal("GetEquation(Y) returned nil")
	}
	if eq.Intercept != 1.0 {
		t.Errorf("expected intercept 1.0, got %f", eq.Intercept)
	}
	if eq.Variance != 0.25 {
		t.Errorf("expected variance 0.25, got %f", eq.Variance)
	}
	if len(eq.Parents) != 1 || eq.Parents[0] != "X" {
		t.Errorf("expected parents [X], got %v", eq.Parents)
	}
	if len(eq.Coefficients) != 1 || eq.Coefficients[0] != 0.5 {
		t.Errorf("expected coefficients [0.5], got %v", eq.Coefficients)
	}
}

func TestSEMAddEquationNoParents(t *testing.T) {
	s := NewSEM()
	err := s.AddEquation("X", nil, nil, 0.0, 1.0)
	if err != nil {
		t.Fatalf("AddEquation: %v", err)
	}
	eq := s.GetEquation("X")
	if eq == nil {
		t.Fatal("GetEquation(X) returned nil")
	}
	if len(eq.Parents) != 0 {
		t.Errorf("expected no parents, got %v", eq.Parents)
	}
}

func TestSEMAddEquationMismatchedLengths(t *testing.T) {
	s := NewSEM()
	err := s.AddEquation("Y", []string{"X"}, []float64{0.5, 0.3}, 0.0, 1.0)
	if err == nil {
		t.Error("expected error for mismatched parents/coefficients lengths")
	}
}

func TestSEMAddEquationCycle(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)
	err := s.AddEquation("X", []string{"Y"}, []float64{0.5}, 0.0, 1.0)
	if err == nil {
		t.Error("expected error for cycle-creating equation")
	}
}

func TestSEMCheckModel(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 0.5)

	if err := s.CheckModel(); err != nil {
		t.Fatalf("CheckModel: %v", err)
	}
}

func TestSEMCheckModelMissingEquation(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)
	// X has no equation.

	if err := s.CheckModel(); err == nil {
		t.Error("expected error for missing equation")
	}
}

func TestSEMCheckModelNegativeVariance(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, -1.0)

	if err := s.CheckModel(); err == nil {
		t.Error("expected error for negative variance")
	}
}

func TestSEMGetEquationNonexistent(t *testing.T) {
	s := NewSEM()
	if s.GetEquation("Z") != nil {
		t.Error("expected nil for nonexistent equation")
	}
}

func TestSEMImpliedCovarianceMatrixSingleVar(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 4.0)

	cov, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		t.Fatalf("ImpliedCovarianceMatrix: %v", err)
	}

	if len(cov) != 1 || len(cov[0]) != 1 {
		t.Fatalf("expected 1x1 matrix, got %dx%d", len(cov), len(cov[0]))
	}
	if math.Abs(cov[0][0]-4.0) > 1e-6 {
		t.Errorf("expected Var(X)=4.0, got %f", cov[0][0])
	}
}

func TestSEMImpliedCovarianceMatrixTwoVars(t *testing.T) {
	// X = e_x, Var(e_x) = 1.0
	// Y = 0.5*X + e_y, Var(e_y) = 0.5
	// => Var(Y) = 0.5^2 * Var(X) + Var(e_y) = 0.25 + 0.5 = 0.75
	// => Cov(X,Y) = 0.5 * Var(X) = 0.5
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 0.5)

	cov, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		t.Fatalf("ImpliedCovarianceMatrix: %v", err)
	}

	// Variables sorted: X=0, Y=1
	if len(cov) != 2 {
		t.Fatalf("expected 2x2 matrix, got %dx%d", len(cov), len(cov[0]))
	}

	if math.Abs(cov[0][0]-1.0) > 1e-6 {
		t.Errorf("Var(X): expected 1.0, got %f", cov[0][0])
	}
	if math.Abs(cov[1][1]-0.75) > 1e-6 {
		t.Errorf("Var(Y): expected 0.75, got %f", cov[1][1])
	}
	if math.Abs(cov[0][1]-0.5) > 1e-6 {
		t.Errorf("Cov(X,Y): expected 0.5, got %f", cov[0][1])
	}
	if math.Abs(cov[1][0]-0.5) > 1e-6 {
		t.Errorf("Cov(Y,X): expected 0.5, got %f", cov[1][0])
	}
}

func TestSEMImpliedCovarianceMatrixChain(t *testing.T) {
	// X = e_x, Var(e_x) = 1
	// Y = 2*X + e_y, Var(e_y) = 1
	// Z = 3*Y + e_z, Var(e_z) = 1
	// => Var(X) = 1
	// => Var(Y) = 4*1 + 1 = 5
	// => Var(Z) = 9*5 + 1 = 46
	// => Cov(X,Y) = 2*1 = 2
	// => Cov(X,Z) = 3*2 = 6
	// => Cov(Y,Z) = 3*5 = 15
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{2.0}, 0.0, 1.0)
	_ = s.AddEquation("Z", []string{"Y"}, []float64{3.0}, 0.0, 1.0)

	cov, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		t.Fatalf("ImpliedCovarianceMatrix: %v", err)
	}

	// Variables sorted: X=0, Y=1, Z=2
	if math.Abs(cov[0][0]-1.0) > 1e-6 {
		t.Errorf("Var(X): expected 1.0, got %f", cov[0][0])
	}
	if math.Abs(cov[1][1]-5.0) > 1e-6 {
		t.Errorf("Var(Y): expected 5.0, got %f", cov[1][1])
	}
	if math.Abs(cov[2][2]-46.0) > 1e-6 {
		t.Errorf("Var(Z): expected 46.0, got %f", cov[2][2])
	}
	if math.Abs(cov[0][1]-2.0) > 1e-6 {
		t.Errorf("Cov(X,Y): expected 2.0, got %f", cov[0][1])
	}
	if math.Abs(cov[0][2]-6.0) > 1e-6 {
		t.Errorf("Cov(X,Z): expected 6.0, got %f", cov[0][2])
	}
	if math.Abs(cov[1][2]-15.0) > 1e-6 {
		t.Errorf("Cov(Y,Z): expected 15.0, got %f", cov[1][2])
	}
}

func TestSEMImpliedCovarianceMatrixInvalidModel(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 1.0)
	// X has no equation.

	_, err := s.ImpliedCovarianceMatrix()
	if err == nil {
		t.Error("expected error for invalid model")
	}
}

func TestSEMImpliedCovarianceMatrixEmpty(t *testing.T) {
	s := NewSEM()
	cov, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		t.Fatalf("ImpliedCovarianceMatrix: %v", err)
	}
	if cov != nil {
		t.Errorf("expected nil for empty SEM, got %v", cov)
	}
}

func TestSEMImpliedCovarianceSymmetric(t *testing.T) {
	s := NewSEM()
	_ = s.AddEquation("X", nil, nil, 0.0, 1.0)
	_ = s.AddEquation("Y", []string{"X"}, []float64{0.5}, 0.0, 0.5)

	cov, err := s.ImpliedCovarianceMatrix()
	if err != nil {
		t.Fatalf("ImpliedCovarianceMatrix: %v", err)
	}

	n := len(cov)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(cov[i][j]-cov[j][i]) > 1e-10 {
				t.Errorf("covariance matrix not symmetric: cov[%d][%d]=%f != cov[%d][%d]=%f",
					i, j, cov[i][j], j, i, cov[j][i])
			}
		}
	}
}
