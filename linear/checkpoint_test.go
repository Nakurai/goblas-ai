package linear_test

import (
	"context"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/linear"
)

// makeData builds a small deterministic linear dataset.
func makeData(n, p int, seed int64) *dataset.Frame {
	rng := rand.New(rand.NewSource(seed))
	x := dataset.NewMatrix(n, p)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		var v float64
		for j := 0; j < p; j++ {
			f := rng.NormFloat64()
			x.Set(i, j, f)
			v += float64(j+1) * f
		}
		y[i] = v
	}
	feat := make([]string, p)
	for j := range feat {
		feat[j] = "f" + string(rune('0'+j))
	}
	return dataset.NewFrame(feat, x, y)
}

// TestResumeEqualsContinuous verifies that training 5 epochs, saving, reloading,
// and training 5 more epochs gives exactly the same model as training 10 epochs
// straight through. SGD here is deterministic, so the results must match.
func TestResumeEqualsContinuous(t *testing.T) {
	data := makeData(800, 3, 1)
	opts := []linear.Option{
		linear.WithSolver(linear.SGD),
		linear.WithStandardize(true),
		linear.WithLearningRate(0.05),
		linear.WithBatchSize(100),
	}

	// Continuous: 10 epochs.
	cont := linear.NewRegression(append(opts, linear.WithEpochs(10))...)
	if err := cont.Fit(context.Background(), data); err != nil {
		t.Fatal(err)
	}

	// Interrupted: 5 epochs, save, reload, 5 more.
	first := linear.NewRegression(append(opts, linear.WithEpochs(5))...)
	if err := first.Fit(context.Background(), data); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "ckpt.gobl")
	if err := first.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	resumed, err := linear.LoadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.Fit(context.Background(), data); err != nil { // 5 more epochs
		t.Fatal(err)
	}

	contPred, _ := cont.PredictBatch(data.Features())
	resPred, _ := resumed.PredictBatch(data.Features())
	for i := range contPred {
		if math.Abs(contPred[i]-resPred[i]) > 1e-9 {
			t.Fatalf("resume diverged at row %d: %v vs %v", i, contPred[i], resPred[i])
		}
	}
}

// TestCancellationCheckpoints verifies that cancelling training writes a usable
// checkpoint that can be loaded and continued.
func TestCancellationCheckpoints(t *testing.T) {
	data := makeData(800, 3, 2)
	path := filepath.Join(t.TempDir(), "ckpt.gobl")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately; training should checkpoint and stop

	m := linear.NewRegression(
		linear.WithSolver(linear.SGD),
		linear.WithStandardize(false), // skip the initial scaler pass so we reach the epoch loop
		linear.WithEpochs(100),
		linear.WithCheckpoint(path, 1),
	)
	err := m.Fit(ctx, data)
	if err != context.Canceled {
		t.Fatalf("Fit error = %v, want context.Canceled", err)
	}
	// The checkpoint written on cancel must be loadable.
	if _, err := linear.LoadFile(path); err != nil {
		t.Fatalf("checkpoint not loadable: %v", err)
	}
}

func TestAutoSelectsClosedFormForFewFeatures(t *testing.T) {
	data := makeData(200, 3, 3)
	m := linear.NewRegression() // Auto
	if err := m.Fit(context.Background(), data); err != nil {
		t.Fatal(err)
	}
	preds, _ := m.PredictBatch(data.Features())
	// Closed form on near-noiseless data should fit almost perfectly.
	var ssRes float64
	ys := data.Targets()
	for i := range ys {
		d := ys[i] - preds[i]
		ssRes += d * d
	}
	if ssRes > 1e-6 {
		t.Errorf("Auto fit residual too high (%v); expected closed-form near-exact fit", ssRes)
	}
}
