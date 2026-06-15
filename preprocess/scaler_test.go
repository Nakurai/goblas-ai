package preprocess

import (
	"math"
	"testing"

	"github.com/nakurai/goblas-ai/dataset"
)

func TestStandardScaler(t *testing.T) {
	// Column 0: 1,2,3 -> mean 2, std sqrt(2/3). Column 1: constant 5 -> std 1.
	x := dataset.Matrix{Rows: 3, Cols: 2, Data: []float64{1, 5, 2, 5, 3, 5}}
	s := NewStandardScaler(2)
	s.FitMatrix(x)

	if math.Abs(s.Mean[0]-2) > 1e-12 {
		t.Errorf("mean[0] = %v, want 2", s.Mean[0])
	}
	wantStd := math.Sqrt(2.0 / 3.0)
	if math.Abs(s.Std[0]-wantStd) > 1e-12 {
		t.Errorf("std[0] = %v, want %v", s.Std[0], wantStd)
	}
	if s.Std[1] != 1 {
		t.Errorf("std[1] = %v, want 1 (constant column guarded)", s.Std[1])
	}

	row := []float64{2, 5}
	s.Apply(row)
	if math.Abs(row[0]) > 1e-12 || math.Abs(row[1]) > 1e-12 {
		t.Errorf("Apply(mean row) = %v, want ~[0 0]", row)
	}
}

func TestStandardScalerStreaming(t *testing.T) {
	// Observing in two chunks must equal observing all at once.
	s := NewStandardScaler(1)
	s.Observe(dataset.Matrix{Rows: 2, Cols: 1, Data: []float64{1, 2}})
	s.Observe(dataset.Matrix{Rows: 2, Cols: 1, Data: []float64{3, 4}})
	s.Fit()
	if math.Abs(s.Mean[0]-2.5) > 1e-12 {
		t.Errorf("streaming mean = %v, want 2.5", s.Mean[0])
	}
}
