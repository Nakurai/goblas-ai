package dataset_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/nakurai/goblas-ai/dataset"
)

func writeCSV(t *testing.T, rows int) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "data.csv")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Fprintln(f, "x1,x2,y")
	for i := 0; i < rows; i++ {
		fmt.Fprintf(f, "%d,%d,%d\n", i, i*2, i*3)
	}
	f.Close()
	return path
}

func TestFrameFromCSV(t *testing.T) {
	path := writeCSV(t, 5)
	frame, err := dataset.FrameFromCSV(path, "y")
	if err != nil {
		t.Fatal(err)
	}
	if frame.Len() != 5 {
		t.Errorf("rows = %d, want 5", frame.Len())
	}
	if got := frame.FeatureNames(); len(got) != 2 || got[0] != "x1" || got[1] != "x2" {
		t.Errorf("features = %v, want [x1 x2]", got)
	}
	if frame.Features().At(3, 1) != 6 { // row 3, x2 = 3*2
		t.Errorf("X[3,1] = %v, want 6", frame.Features().At(3, 1))
	}
	if frame.Targets()[4] != 12 { // y = 4*3
		t.Errorf("y[4] = %v, want 12", frame.Targets()[4])
	}
}

func TestCSVStreamBatching(t *testing.T) {
	path := writeCSV(t, 10)
	s, err := dataset.OpenCSV(path, "y")
	if err != nil {
		t.Fatal(err)
	}
	var total, batches int
	for b, err := range s.Batches(3) {
		if err != nil {
			t.Fatal(err)
		}
		total += b.X.Rows
		batches++
	}
	if total != 10 {
		t.Errorf("streamed %d rows, want 10", total)
	}
	if batches != 4 { // 3+3+3+1
		t.Errorf("got %d batches, want 4", batches)
	}
	// Iterating again must replay the same data (needed for multi-epoch training).
	total = 0
	for b := range s.Batches(0) {
		total += b.X.Rows
	}
	if total != 10 {
		t.Errorf("second pass streamed %d rows, want 10", total)
	}
}

func TestSplitCSVComplementary(t *testing.T) {
	path := writeCSV(t, 1000)
	train, test, err := dataset.SplitCSV(path, "y", 0.2, 42)
	if err != nil {
		t.Fatal(err)
	}
	count := func(p dataset.Provider) (n int, seen map[float64]bool) {
		seen = map[float64]bool{}
		for b := range p.Batches(64) {
			for _, y := range b.Y {
				seen[y] = true
				n++
			}
		}
		return n, seen
	}
	nTrain, trainSeen := count(train)
	nTest, testSeen := count(test)

	if nTrain+nTest != 1000 {
		t.Errorf("train+test = %d, want 1000 (every row assigned exactly once)", nTrain+nTest)
	}
	// Disjoint: no target value appears in both sets (targets are unique here).
	for y := range testSeen {
		if trainSeen[y] {
			t.Errorf("row with y=%v appears in both train and test", y)
		}
	}
	// Roughly 20% in test.
	if nTest < 150 || nTest > 250 {
		t.Errorf("test size = %d, want ~200", nTest)
	}
}

func TestReadMatrix(t *testing.T) {
	path := writeCSV(t, 4)
	m, err := dataset.ReadMatrix(path, []string{"x2", "x1"}) // note: reversed order
	if err != nil {
		t.Fatal(err)
	}
	if m.Rows != 4 || m.Cols != 2 {
		t.Fatalf("shape = %dx%d, want 4x2", m.Rows, m.Cols)
	}
	if m.At(2, 0) != 4 || m.At(2, 1) != 2 { // row 2: x2=4, x1=2
		t.Errorf("row 2 = [%v %v], want [4 2]", m.At(2, 0), m.At(2, 1))
	}
	if _, err := dataset.ReadMatrix(path, []string{"missing"}); err == nil {
		t.Error("expected error for missing column")
	}
}
