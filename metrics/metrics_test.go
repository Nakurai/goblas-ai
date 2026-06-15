package metrics

import (
	"math"
	"testing"
)

func TestMetricsKnownValues(t *testing.T) {
	yTrue := []float64{1, 2, 3, 4}
	yPred := []float64{1, 2, 3, 5} // one error of size 1

	if got := MSE(yTrue, yPred); math.Abs(got-0.25) > 1e-12 {
		t.Errorf("MSE = %v, want 0.25", got)
	}
	if got := RMSE(yTrue, yPred); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("RMSE = %v, want 0.5", got)
	}
	if got := MAE(yTrue, yPred); math.Abs(got-0.25) > 1e-12 {
		t.Errorf("MAE = %v, want 0.25", got)
	}
}

func TestR2PerfectAndBaseline(t *testing.T) {
	y := []float64{10, 20, 30, 40}
	if got := R2(y, y); math.Abs(got-1) > 1e-12 {
		t.Errorf("R2 perfect = %v, want 1", got)
	}
	// Predicting the mean everywhere gives R2 = 0.
	mean := 25.0
	flat := []float64{mean, mean, mean, mean}
	if got := R2(y, flat); math.Abs(got) > 1e-12 {
		t.Errorf("R2 baseline = %v, want 0", got)
	}
}

func TestR2ConstantTarget(t *testing.T) {
	y := []float64{5, 5, 5}
	if got := R2(y, []float64{5, 5, 5}); got != 1 {
		t.Errorf("R2 constant exact = %v, want 1", got)
	}
	if got := R2(y, []float64{5, 6, 5}); got != 0 {
		t.Errorf("R2 constant inexact = %v, want 0", got)
	}
}
