package linear_test

import (
	"context"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/linear"
	"github.com/nakurai/goblas-ai/metrics"
	"github.com/nakurai/goblas-ai/model"
)

// makeLinearData builds a synthetic dataset y = intercept + Xβ + small noise.
func makeLinearData(n, p int, trueCoef []float64, intercept float64, seed int64) *dataset.Frame {
	rng := rand.New(rand.NewSource(seed))
	x := dataset.NewMatrix(n, p)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		v := intercept
		for j := 0; j < p; j++ {
			f := rng.NormFloat64()*float64(j+1) + float64(j) // varied scales
			x.Set(i, j, f)
			v += trueCoef[j] * f
		}
		y[i] = v + rng.NormFloat64()*0.01 // tiny noise
	}
	features := make([]string, p)
	for j := range features {
		features[j] = "f" + string(rune('0'+j))
	}
	return dataset.NewFrame(features, x, y)
}

func TestClosedFormRecoversCoefficients(t *testing.T) {
	trueCoef := []float64{2.0, -3.0, 0.5}
	data := makeLinearData(2000, 3, trueCoef, 5.0, 1)

	lr := linear.NewRegression(linear.WithSolver(linear.ClosedForm))
	if err := lr.Fit(context.Background(), data); err != nil {
		t.Fatalf("fit: %v", err)
	}

	coef, intercept := lr.Coefficients()
	for j, want := range trueCoef {
		if math.Abs(coef[j]-want) > 0.05 {
			t.Errorf("coef[%d] = %.4f, want ~%.4f", j, coef[j], want)
		}
	}
	if math.Abs(intercept-5.0) > 0.05 {
		t.Errorf("intercept = %.4f, want ~5.0", intercept)
	}

	preds, _ := lr.PredictBatch(data.Features())
	if r2 := metrics.R2(data.Targets(), preds); r2 < 0.999 {
		t.Errorf("R2 = %.5f, want > 0.999", r2)
	}
}

func TestSGDAgreesWithClosedForm(t *testing.T) {
	trueCoef := []float64{2.0, -3.0, 0.5}
	data := makeLinearData(2000, 3, trueCoef, 5.0, 2)

	cf := linear.NewRegression(linear.WithSolver(linear.ClosedForm))
	if err := cf.Fit(context.Background(), data); err != nil {
		t.Fatalf("closed-form fit: %v", err)
	}
	sgd := linear.NewRegression(
		linear.WithSolver(linear.SGD),
		linear.WithEpochs(300),
		linear.WithLearningRate(0.05),
	)
	if err := sgd.Fit(context.Background(), data); err != nil {
		t.Fatalf("sgd fit: %v", err)
	}

	cfPred, _ := cf.PredictBatch(data.Features())
	sgdPred, _ := sgd.PredictBatch(data.Features())
	if rmse := metrics.RMSE(cfPred, sgdPred); rmse > 0.1 {
		t.Errorf("closed-form vs SGD predictions disagree: RMSE=%.4f", rmse)
	}
}

func TestSaveLoadRoundTrip(t *testing.T) {
	data := makeLinearData(500, 3, []float64{1, 2, 3}, 0.5, 3)
	lr := linear.NewRegression(linear.WithSolver(linear.ClosedForm))
	if err := lr.Fit(context.Background(), data); err != nil {
		t.Fatalf("fit: %v", err)
	}
	want, _ := lr.PredictBatch(data.Features())

	path := filepath.Join(t.TempDir(), "m.gobl")
	if err := lr.SaveFile(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := model.LoadFile(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got, err := loaded.PredictBatch(data.Features())
	if err != nil {
		t.Fatalf("predict after load: %v", err)
	}
	for i := range want {
		if math.Abs(want[i]-got[i]) > 1e-9 {
			t.Fatalf("prediction mismatch at %d: %v vs %v", i, want[i], got[i])
		}
	}
}

func TestOnlinePartialFit(t *testing.T) {
	trueCoef := []float64{1.5, -2.0}
	data := makeLinearData(3000, 2, trueCoef, 0, 4)

	// Online learning on raw features (standardization disabled).
	lr := linear.NewRegression(
		linear.WithStandardize(false),
		linear.WithIntercept(false),
		linear.WithLearningRate(0.001),
	)
	ctx := context.Background()
	for pass := 0; pass < 50; pass++ {
		for batch, err := range data.Batches(32) {
			if err != nil {
				t.Fatal(err)
			}
			if err := lr.PartialFit(ctx, batch); err != nil {
				t.Fatal(err)
			}
		}
	}
	coef, _ := lr.Coefficients()
	for j, want := range trueCoef {
		if math.Abs(coef[j]-want) > 0.1 {
			t.Errorf("online coef[%d] = %.4f, want ~%.4f", j, coef[j], want)
		}
	}
}
