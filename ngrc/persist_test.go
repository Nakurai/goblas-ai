package ngrc_test

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/nakurai/goblas-ai/ngrc"
)

func TestSaveLoadForecastIdentical(t *testing.T) {
	seq := ar2Series(300, 0.4, 0.5)
	m := ngrc.New(ngrc.WithTaps(2), ngrc.WithOrder(2))
	if err := m.Fit(context.Background(), seq); err != nil {
		t.Fatal(err)
	}
	want, err := m.Forecast(40)
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "ngrc.gobl")
	if err := m.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := ngrc.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Forecast(40)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(want) {
		t.Fatalf("forecast length %d, want %d", len(got), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if math.Abs(want[i][j]-got[i][j]) > 1e-12 {
				t.Fatalf("forecast mismatch after reload at step %d var %d: %v vs %v", i, j, want[i][j], got[i][j])
			}
		}
	}
}
