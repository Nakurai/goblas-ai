package dataset

import (
	"encoding/csv"
	"fmt"
	"io"
	"iter"
	"math/rand"
	"os"
	"strconv"
)

// CSVStream is a Provider backed by a CSV file on disk. Rows are read and parsed
// on demand, so a file far larger than available memory can be used for
// training: at no point is the whole file held in RAM.
//
// The CSV must have a header row. Every column except the named target column is
// treated as a numeric feature, in the order the columns appear.
type CSVStream struct {
	path      string
	features  []string
	featIdx   []int // source column index for each feature
	targetIdx int

	// Optional row filter, used to carve a reproducible train/test split out of
	// a single file without loading it (see SplitCSV).
	filtered bool
	seed     int64
	testFrac float64
	wantTest bool
}

// OpenCSV prepares a streaming Provider over the CSV file at path, using the
// column named target as the value to predict. It reads only the header to
// discover the columns; the data rows are read later, during training.
func OpenCSV(path, target string) (*CSVStream, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("dataset: open csv: %w", err)
	}
	defer f.Close()

	header, err := csv.NewReader(f).Read()
	if err != nil {
		return nil, fmt.Errorf("dataset: read csv header: %w", err)
	}

	s := &CSVStream{path: path, targetIdx: -1}
	for i, name := range header {
		if name == target {
			s.targetIdx = i
			continue
		}
		s.features = append(s.features, name)
		s.featIdx = append(s.featIdx, i)
	}
	if s.targetIdx == -1 {
		return nil, fmt.Errorf("dataset: target column %q not found in header", target)
	}
	if len(s.features) == 0 {
		return nil, fmt.Errorf("dataset: csv has no feature columns besides target %q", target)
	}
	return s, nil
}

// FeatureNames implements Provider.
func (s *CSVStream) FeatureNames() []string { return s.features }

// NFeatures implements Provider.
func (s *CSVStream) NFeatures() int { return len(s.features) }

// eachRow streams parsed rows from the file, applying the optional split filter.
// It re-opens the file on every call so the Provider can be iterated repeatedly
// (once per training epoch).
func (s *CSVStream) eachRow(fn func(x []float64, y float64) error) error {
	f, err := os.Open(s.path)
	if err != nil {
		return fmt.Errorf("dataset: open csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.ReuseRecord = true // reuse the backing slice between records to reduce allocations

	// Skip header.
	if _, err := r.Read(); err != nil {
		return fmt.Errorf("dataset: read csv header: %w", err)
	}

	var prng *rand.Rand
	if s.filtered {
		prng = rand.New(rand.NewSource(s.seed))
	}

	row := make([]float64, len(s.featIdx))
	line := 1 // header was line 1
	for {
		rec, err := r.Read()
		line++
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("dataset: read csv line %d: %w", line, err)
		}

		// Deterministic train/test assignment must happen for every data row,
		// before any skip, so train and test scans stay perfectly complementary.
		if s.filtered {
			keep := (prng.Float64() < s.testFrac) == s.wantTest
			if !keep {
				continue
			}
		}

		y, err := strconv.ParseFloat(rec[s.targetIdx], 64)
		if err != nil {
			return fmt.Errorf("dataset: line %d: parse target: %w", line, err)
		}
		for j, idx := range s.featIdx {
			v, err := strconv.ParseFloat(rec[idx], 64)
			if err != nil {
				return fmt.Errorf("dataset: line %d: parse feature %q: %w", line, s.features[j], err)
			}
			row[j] = v
		}
		if err := fn(row, y); err != nil {
			return err
		}
	}
}

// Batches implements Provider, accumulating streamed rows into mini-batches.
func (s *CSVStream) Batches(size int) iter.Seq2[Batch, error] {
	return func(yield func(Batch, error) bool) {
		p := len(s.featIdx)
		batchSize := size
		if batchSize <= 0 {
			batchSize = 4096 // a sane streaming default when the caller wants "all"
		}

		x := make([]float64, 0, batchSize*p)
		y := make([]float64, 0, batchSize)
		stopped := false

		flush := func() bool {
			if len(y) == 0 {
				return true
			}
			b := Batch{X: Matrix{Rows: len(y), Cols: p, Data: x}, Y: y}
			if !yield(b, nil) {
				return false
			}
			x = make([]float64, 0, batchSize*p)
			y = make([]float64, 0, batchSize)
			return true
		}

		err := s.eachRow(func(rx []float64, ry float64) error {
			x = append(x, rx...)
			y = append(y, ry)
			if len(y) >= batchSize {
				if !flush() {
					stopped = true
					return errStop
				}
			}
			return nil
		})
		if stopped {
			return
		}
		if err != nil {
			yield(Batch{}, err)
			return
		}
		flush()
	}
}

// errStop is a sentinel used to abort eachRow when the consumer stops early.
var errStop = fmt.Errorf("dataset: iteration stopped")

// FrameFromCSV loads an entire CSV file into an in-memory Frame. Convenient for
// small datasets and examples; prefer OpenCSV for data that may not fit in RAM.
func FrameFromCSV(path, target string) (*Frame, error) {
	s, err := OpenCSV(path, target)
	if err != nil {
		return nil, err
	}
	p := s.NFeatures()
	var data []float64
	var y []float64
	if err := s.eachRow(func(rx []float64, ry float64) error {
		data = append(data, rx...)
		y = append(y, ry)
		return nil
	}); err != nil {
		return nil, err
	}
	x := Matrix{Rows: len(y), Cols: p, Data: data}
	return NewFrame(s.features, x, y), nil
}

// ReadMatrix loads the named columns from a CSV file into an in-memory matrix,
// one row per data row, columns in the order given. It is used for prediction,
// where the input may not contain a target column. It errors if any requested
// column is missing or any value is non-numeric.
func ReadMatrix(path string, columns []string) (Matrix, error) {
	f, err := os.Open(path)
	if err != nil {
		return Matrix{}, fmt.Errorf("dataset: open csv: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	header, err := r.Read()
	if err != nil {
		return Matrix{}, fmt.Errorf("dataset: read csv header: %w", err)
	}
	pos := make(map[string]int, len(header))
	for i, name := range header {
		pos[name] = i
	}
	idx := make([]int, len(columns))
	for j, name := range columns {
		p, ok := pos[name]
		if !ok {
			return Matrix{}, fmt.Errorf("dataset: column %q not found in %s", name, path)
		}
		idx[j] = p
	}

	var data []float64
	rows := 0
	line := 1
	for {
		rec, err := r.Read()
		line++
		if err == io.EOF {
			break
		}
		if err != nil {
			return Matrix{}, fmt.Errorf("dataset: read csv line %d: %w", line, err)
		}
		for j, c := range idx {
			v, err := strconv.ParseFloat(rec[c], 64)
			if err != nil {
				return Matrix{}, fmt.Errorf("dataset: line %d: parse column %q: %w", line, columns[j], err)
			}
			data = append(data, v)
		}
		rows++
	}
	return Matrix{Rows: rows, Cols: len(columns), Data: data}, nil
}

// SplitCSV carves a single CSV file into a streaming training Provider and a
// streaming test Provider, without loading the file into memory. testFrac is the
// fraction of rows assigned to the test set; seed makes the split reproducible.
//
// The split is deterministic per row: the two Providers replay the same
// pseudo-random sequence over the rows, so every row lands in exactly one of the
// two sets and the assignment is stable across passes.
func SplitCSV(path, target string, testFrac float64, seed int64) (train, test Provider, err error) {
	base, err := OpenCSV(path, target)
	if err != nil {
		return nil, nil, err
	}
	mk := func(wantTest bool) *CSVStream {
		c := *base
		c.filtered = true
		c.seed = seed
		c.testFrac = testFrac
		c.wantTest = wantTest
		return &c
	}
	return mk(false), mk(true), nil
}
