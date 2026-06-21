package dataset_test

import (
	"testing"

	"github.com/nakurai/goblas-ai/dataset"
)

func TestSequenceFromValues(t *testing.T) {
	vals := []float64{1, 2, 3, 4}
	s := dataset.SequenceFromValues("x", vals)

	if s.Len() != 4 {
		t.Errorf("Len = %d, want 4", s.Len())
	}
	if s.Dim() != 1 {
		t.Errorf("Dim = %d, want 1", s.Dim())
	}
	if len(s.Vars) != 1 || s.Vars[0] != "x" {
		t.Errorf("Vars = %v, want [x]", s.Vars)
	}
	if got := s.Step(2)[0]; got != 3 {
		t.Errorf("Step(2) = %g, want 3", got)
	}

	// The input slice must be copied, not aliased.
	vals[0] = 99
	if got := s.Step(0)[0]; got != 1 {
		t.Errorf("mutating the input changed the sequence: Step(0) = %g, want 1", got)
	}
}

func TestNewSequence(t *testing.T) {
	rows := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	s, err := dataset.NewSequence([]string{"x", "y"}, rows)
	if err != nil {
		t.Fatalf("NewSequence: %v", err)
	}
	if s.Len() != 3 || s.Dim() != 2 {
		t.Errorf("Len/Dim = %d/%d, want 3/2", s.Len(), s.Dim())
	}
	if got := s.Step(1); got[0] != 3 || got[1] != 4 {
		t.Errorf("Step(1) = %v, want [3 4]", got)
	}
}

func TestNewSequenceRaggedRow(t *testing.T) {
	rows := [][]float64{{1, 2}, {3}}
	if _, err := dataset.NewSequence([]string{"x", "y"}, rows); err == nil {
		t.Error("expected error for a row whose length does not match len(vars)")
	}
}
