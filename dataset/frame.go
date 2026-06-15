package dataset

import (
	"iter"
	"math/rand"
)

// Frame is an in-memory dataset: a feature matrix X (n rows × p columns) and a
// target vector y (length n). It is the convenient choice when the data fits in
// RAM. Frame implements Provider.
type Frame struct {
	features []string
	x        Matrix
	y        []float64
}

// NewFrame builds a Frame from a feature matrix and target vector. It panics if
// the shapes are inconsistent, since that indicates a programming error rather
// than bad input data.
func NewFrame(features []string, x Matrix, y []float64) *Frame {
	if x.Rows != len(y) {
		panic("dataset: number of rows in X must equal len(y)")
	}
	if x.Cols != len(features) {
		panic("dataset: number of feature names must equal number of columns in X")
	}
	return &Frame{features: features, x: x, y: y}
}

// Len returns the number of rows.
func (f *Frame) Len() int { return f.x.Rows }

// Features returns the feature matrix. Useful for prediction and evaluation,
// e.g. model.PredictBatch(test.Features()).
func (f *Frame) Features() Matrix { return f.x }

// Targets returns the target vector.
func (f *Frame) Targets() []float64 { return f.y }

// FeatureNames implements Provider.
func (f *Frame) FeatureNames() []string { return f.features }

// NFeatures implements Provider.
func (f *Frame) NFeatures() int { return f.x.Cols }

// Batches implements Provider, slicing the in-memory rows into groups. The
// emitted batches alias the Frame's storage (no copy); callers must not retain
// or mutate them across iterations.
func (f *Frame) Batches(size int) iter.Seq2[Batch, error] {
	if size <= 0 {
		size = f.x.Rows
	}
	return func(yield func(Batch, error) bool) {
		for start := 0; start < f.x.Rows; start += size {
			end := start + size
			if end > f.x.Rows {
				end = f.x.Rows
			}
			b := Batch{
				X: Matrix{
					Rows: end - start,
					Cols: f.x.Cols,
					Data: f.x.Data[start*f.x.Cols : end*f.x.Cols],
				},
				Y: f.y[start:end],
			}
			if !yield(b, nil) {
				return
			}
		}
	}
}

// Split partitions the Frame into a training set and a test set. testFrac is the
// fraction of rows (0..1) assigned to the test set; seed makes the split
// reproducible. Rows are shuffled before splitting so ordering in the source
// data does not bias the result.
func (f *Frame) Split(testFrac float64, seed int64) (train, test *Frame) {
	n := f.x.Rows
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	nTest := int(float64(n) * testFrac)

	build := func(idx []int) *Frame {
		x := NewMatrix(len(idx), f.x.Cols)
		y := make([]float64, len(idx))
		for i, src := range idx {
			copy(x.Row(i), f.x.Row(src))
			y[i] = f.y[src]
		}
		return NewFrame(f.features, x, y)
	}
	return build(perm[nTest:]), build(perm[:nTest])
}
