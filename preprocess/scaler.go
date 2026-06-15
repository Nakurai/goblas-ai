// Package preprocess contains feature transformations applied before training
// and, identically, before prediction. Keeping the transformation together with
// the model is what prevents "train/serve skew": the bug where a model is fed
// differently-scaled numbers at prediction time than it saw during training.
package preprocess

import (
	"math"

	"github.com/nakurai/goblas-ai/dataset"
)

// StandardScaler centers and scales each feature so that, across the training
// data, it has an average (mean) of 0 and a standard deviation of 1.
//
// Why this matters: a "standard deviation" is a measure of how spread out a
// feature's values are. Features measured on very different scales — say house
// size in the thousands of square feet versus number of bedrooms in single
// digits — can make some training methods slow or unstable. Rescaling every
// feature to a common footing fixes that.
//
// The scaler can be fitted in a streaming fashion: call Observe once per batch
// to accumulate statistics, then Fit to finalize them. This keeps memory
// constant regardless of dataset size.
type StandardScaler struct {
	Mean []float64 // per-feature mean, valid after Fit
	Std  []float64 // per-feature standard deviation, valid after Fit

	// streaming accumulators
	n     int64
	sum   []float64
	sumSq []float64
}

// NewStandardScaler creates a scaler for p features.
func NewStandardScaler(p int) *StandardScaler {
	return &StandardScaler{
		sum:   make([]float64, p),
		sumSq: make([]float64, p),
	}
}

// Observe folds one batch of rows into the running statistics. Call it once per
// batch during a pass over the data, then call Fit.
func (s *StandardScaler) Observe(x dataset.Matrix) {
	for i := 0; i < x.Rows; i++ {
		row := x.Row(i)
		for j, v := range row {
			s.sum[j] += v
			s.sumSq[j] += v * v
		}
	}
	s.n += int64(x.Rows)
}

// Fit finalizes Mean and Std from everything Observe has seen. A feature that
// never varies (standard deviation 0) is given a standard deviation of 1 so that
// transforming it yields 0 rather than dividing by zero.
func (s *StandardScaler) Fit() {
	p := len(s.sum)
	s.Mean = make([]float64, p)
	s.Std = make([]float64, p)
	n := float64(s.n)
	for j := 0; j < p; j++ {
		mean := s.sum[j] / n
		s.Mean[j] = mean
		variance := s.sumSq[j]/n - mean*mean
		if variance < 0 {
			variance = 0 // guard against tiny negative values from rounding
		}
		std := math.Sqrt(variance)
		if std == 0 {
			std = 1
		}
		s.Std[j] = std
	}
}

// FitMatrix is a convenience that fits the scaler from a single in-memory matrix.
func (s *StandardScaler) FitMatrix(x dataset.Matrix) {
	s.Observe(x)
	s.Fit()
}

// Apply transforms a feature row in place: each value becomes (value - mean) /
// std for its column. The row length must equal the number of features.
func (s *StandardScaler) Apply(row []float64) {
	for j := range row {
		row[j] = (row[j] - s.Mean[j]) / s.Std[j]
	}
}
