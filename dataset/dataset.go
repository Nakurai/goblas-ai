// Package dataset provides the data abstractions goblas-ai trains on.
//
// The central idea is the Provider interface: a source of (features, target)
// rows delivered in mini-batches. The same Provider works whether the data
// lives entirely in memory (Frame) or is streamed row-by-row from a file on
// disk (CSVStream), so training code never needs to know which it is — and
// large files never have to fit in RAM.
package dataset

import "iter"

// Matrix is a dense, row-major matrix of float64 values. Row i, column j lives
// at Data[i*Cols+j]. It deliberately avoids any third-party matrix type so it
// can appear in goblas-ai's public API.
type Matrix struct {
	Rows int
	Cols int
	Data []float64 // row-major; len(Data) == Rows*Cols
}

// NewMatrix allocates a zeroed rows×cols matrix.
func NewMatrix(rows, cols int) Matrix {
	return Matrix{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// At returns the element at row i, column j.
func (m Matrix) At(i, j int) float64 { return m.Data[i*m.Cols+j] }

// Set assigns v to row i, column j.
func (m Matrix) Set(i, j int, v float64) { m.Data[i*m.Cols+j] = v }

// Row returns a slice aliasing row i (length Cols). Mutating it mutates the
// matrix.
func (m Matrix) Row(i int) []float64 { return m.Data[i*m.Cols : i*m.Cols+m.Cols] }

// Batch is a contiguous group of training rows: a feature matrix X (one row per
// example) and the matching target values Y (one per row).
type Batch struct {
	X Matrix
	Y []float64
}

// Provider is a source of training data delivered in mini-batches.
//
// Implementations must allow Batches to be called more than once: training may
// make several passes over the data (epochs), and each call must replay the
// same rows in the same order.
type Provider interface {
	// Batches streams the data in groups of at most size rows. A size <= 0
	// means "one batch containing every row". The iterator yields a non-nil
	// error if reading fails (e.g. a malformed CSV line); callers should stop
	// on the first error.
	Batches(size int) iter.Seq2[Batch, error]

	// FeatureNames returns the column names of the features, in column order.
	FeatureNames() []string

	// NFeatures returns the number of feature columns.
	NFeatures() int
}
