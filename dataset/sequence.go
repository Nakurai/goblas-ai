package dataset

import (
	"encoding/csv"
	"fmt"
	"os"
)

// Sequence is an ordered multivariate time series: T rows (time steps) by d
// columns (variables). Row t is the state of the system at time t, and the order
// of the rows carries meaning — it is the signal, not an incidental arrangement.
//
// Sequence is deliberately separate from the row-shuffling tools used for
// independent tabular data (Frame, SplitCSV), so a time series cannot be
// accidentally reordered.
type Sequence struct {
	Vars []string
	Data Matrix // T×d, row t is the state at time t
}

// Len returns the number of time steps (T).
func (s *Sequence) Len() int { return s.Data.Rows }

// Dim returns the number of variables (d).
func (s *Sequence) Dim() int { return s.Data.Cols }

// Step returns the state at time t (a slice of length d, aliasing the storage).
func (s *Sequence) Step(t int) []float64 { return s.Data.Row(t) }

// SequenceFromCSV loads a multivariate time series from a CSV file. If cols is
// empty, every column is used, in file order; otherwise only the named columns
// are loaded, in the order given. Rows are kept in file order — that order is
// taken to be time order.
func SequenceFromCSV(path string, cols ...string) (*Sequence, error) {
	if len(cols) == 0 {
		header, err := readHeader(path)
		if err != nil {
			return nil, err
		}
		cols = header
	}
	m, err := ReadMatrix(path, cols)
	if err != nil {
		return nil, err
	}
	return &Sequence{Vars: cols, Data: m}, nil
}

// SplitChrono splits the sequence in time: the first (1-testFrac) of the steps
// become the training sequence and the final testFrac become the test sequence.
// There is no shuffling — for a time series you must always test on the future,
// using the past to train.
func (s *Sequence) SplitChrono(testFrac float64) (train, test *Sequence) {
	t := s.Len()
	d := s.Dim()
	nTest := int(float64(t) * testFrac)
	nTrain := t - nTest

	train = &Sequence{
		Vars: s.Vars,
		Data: Matrix{Rows: nTrain, Cols: d, Data: s.Data.Data[:nTrain*d]},
	}
	test = &Sequence{
		Vars: s.Vars,
		Data: Matrix{Rows: nTest, Cols: d, Data: s.Data.Data[nTrain*d:]},
	}
	return train, test
}

// readHeader returns the column names from the first row of a CSV file.
func readHeader(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("dataset: open csv: %w", err)
	}
	defer f.Close()
	header, err := csv.NewReader(f).Read()
	if err != nil {
		return nil, fmt.Errorf("dataset: read csv header: %w", err)
	}
	return header, nil
}
