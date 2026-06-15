package ngrc_test

import (
	"context"
	"math"
	"testing"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/metrics"
	"github.com/nakurai/goblas-ai/ngrc"
)

// ar2Series builds x(t+1) = a*x(t) + b*x(t-1), a stable linear recurrence NG-RC
// should recover exactly with 2 taps and order 1.
func ar2Series(n int, a, b float64) *dataset.Sequence {
	data := make([]float64, n)
	data[0], data[1] = 1.0, 0.8
	for t := 2; t < n; t++ {
		data[t] = a*data[t-1] + b*data[t-2]
	}
	return &dataset.Sequence{
		Vars: []string{"x"},
		Data: dataset.Matrix{Rows: n, Cols: 1, Data: data},
	}
}

func TestRecoversLinearRecurrence(t *testing.T) {
	seq := ar2Series(300, 0.4, 0.5)
	train, test := seq.SplitChrono(0.2)

	m := ngrc.New(
		ngrc.WithTaps(2),
		ngrc.WithOrder(1),
		ngrc.WithStandardize(false),
		ngrc.WithPredictDelta(false),
		ngrc.WithRidge(1e-10),
	)
	if err := m.Fit(context.Background(), train); err != nil {
		t.Fatal(err)
	}

	// One-step-ahead over the held-out tail: the model is primed with the end of
	// training (which directly precedes the test segment in time).
	var preds, truth []float64
	for i := 0; i < test.Len()-1; i++ {
		p, err := m.Step(test.Step(i))
		if err != nil {
			t.Fatal(err)
		}
		preds = append(preds, p[0])
		truth = append(truth, test.Step(i + 1)[0])
	}
	if rmse := metrics.RMSE(truth, preds); rmse > 1e-6 {
		t.Errorf("one-step RMSE = %g, want ~0 (exact recurrence)", rmse)
	}
}

func TestAutonomousForecastTracks(t *testing.T) {
	seq := ar2Series(400, 0.4, 0.5)
	// Use the whole series to train, then forecast and compare to a fresh,
	// independently generated continuation of the same recurrence.
	m := ngrc.New(
		ngrc.WithTaps(2), ngrc.WithOrder(1),
		ngrc.WithStandardize(false), ngrc.WithPredictDelta(false),
		ngrc.WithRidge(1e-10),
	)
	if err := m.Fit(context.Background(), seq); err != nil {
		t.Fatal(err)
	}
	// Continue the true recurrence beyond the training data.
	last1 := seq.Step(seq.Len() - 1)[0]
	last2 := seq.Step(seq.Len() - 2)[0]
	const horizon = 30
	wantNext := make([]float64, horizon)
	p2, p1 := last2, last1
	for i := 0; i < horizon; i++ {
		next := 0.4*p1 + 0.5*p2
		wantNext[i] = next
		p2, p1 = p1, next
	}

	got, err := m.Forecast(horizon)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < horizon; i++ {
		if math.Abs(got[i][0]-wantNext[i]) > 1e-4 {
			t.Fatalf("forecast step %d = %g, want %g", i, got[i][0], wantNext[i])
		}
	}
}

func TestMultivariateRotation(t *testing.T) {
	// A 2-D rotation: next state depends only on the current state (1 tap).
	const n = 400
	theta := 0.1
	c, s := math.Cos(theta), math.Sin(theta)
	data := make([]float64, n*2)
	data[0], data[1] = 1.0, 0.0
	for t := 1; t < n; t++ {
		x, y := data[(t-1)*2], data[(t-1)*2+1]
		data[t*2] = c*x - s*y
		data[t*2+1] = s*x + c*y
	}
	seq := &dataset.Sequence{Vars: []string{"x", "y"}, Data: dataset.Matrix{Rows: n, Cols: 2, Data: data}}

	m := ngrc.New(
		ngrc.WithTaps(1), ngrc.WithOrder(1),
		ngrc.WithStandardize(false), ngrc.WithPredictDelta(false),
		ngrc.WithRidge(1e-10),
	)
	if err := m.Fit(context.Background(), seq); err != nil {
		t.Fatal(err)
	}
	got, err := m.Forecast(50)
	if err != nil {
		t.Fatal(err)
	}
	// Rotation preserves radius; the forecast should stay on the unit circle.
	for i, st := range got {
		r := math.Hypot(st[0], st[1])
		if math.Abs(r-1) > 1e-3 {
			t.Errorf("forecast step %d radius = %g, want ~1", i, r)
		}
	}
}

func TestStepRequiresTraining(t *testing.T) {
	m := ngrc.New()
	if _, err := m.Step([]float64{1}); err == nil {
		t.Error("expected error from Step on an untrained model")
	}
}

func TestQuadraticDynamics(t *testing.T) {
	// Logistic map x(t+1) = r*x(t)*(1-x(t)) — needs order >= 2 to capture.
	const n = 500
	r := 3.6
	data := make([]float64, n)
	data[0] = 0.4
	for t := 1; t < n; t++ {
		data[t] = r * data[t-1] * (1 - data[t-1])
	}
	seq := &dataset.Sequence{Vars: []string{"x"}, Data: dataset.Matrix{Rows: n, Cols: 1, Data: data}}
	train, test := seq.SplitChrono(0.2)

	m := ngrc.New(
		ngrc.WithTaps(1), ngrc.WithOrder(2),
		ngrc.WithStandardize(false), ngrc.WithPredictDelta(false),
		ngrc.WithRidge(1e-12),
	)
	if err := m.Fit(context.Background(), train); err != nil {
		t.Fatal(err)
	}
	var preds, truth []float64
	for i := 0; i < test.Len()-1; i++ {
		p, _ := m.Step(test.Step(i))
		preds = append(preds, p[0])
		truth = append(truth, test.Step(i + 1)[0])
	}
	if rmse := metrics.RMSE(truth, preds); rmse > 1e-6 {
		t.Errorf("logistic-map one-step RMSE = %g, want ~0 (order-2 captures it exactly)", rmse)
	}
}
