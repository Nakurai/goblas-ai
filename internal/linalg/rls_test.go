package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// TestRLSConvergesToLeastSquares feeds linear samples one at a time and checks
// the recursive solution converges to the known generating weights, matching
// what a batch least-squares solve would recover.
func TestRLSConvergesToLeastSquares(t *testing.T) {
	const m, nOut, n = 4, 2, 4000
	rng := rand.New(rand.NewSource(7))

	// True readout W (m×nOut, row-major).
	wTrue := make([]float64, m*nOut)
	for i := range wTrue {
		wTrue[i] = rng.NormFloat64()
	}

	// lambda 1 (no forgetting) for a stationary problem; large p0 = fast start.
	r := NewRLS(m, nOut, 1.0, 1e3, nil)
	o := make([]float64, m)
	y := make([]float64, nOut)
	for s := 0; s < n; s++ {
		for i := 0; i < m; i++ {
			o[i] = rng.NormFloat64()
		}
		for j := 0; j < nOut; j++ {
			var v float64
			for i := 0; i < m; i++ {
				v += wTrue[i*nOut+j] * o[i]
			}
			y[j] = v + rng.NormFloat64()*1e-3 // tiny noise
		}
		r.Update(o, y)
	}

	got := r.Weights()
	for i := range wTrue {
		if math.Abs(got[i]-wTrue[i]) > 1e-2 {
			t.Errorf("weight[%d] = %.4f, want ~%.4f", i, got[i], wTrue[i])
		}
	}
}

// TestRLSWithCovMatchesRidge checks that starting RLS from a batch ridge solution
// (weights and inverse covariance) leaves the solution essentially unchanged when
// the same rows are then streamed once more — i.e. the warm start is consistent.
func TestRLSWithCovMatchesRidge(t *testing.T) {
	const m, nOut, n = 3, 1, 500
	const ridge = 1e-6
	rng := rand.New(rand.NewSource(11))

	rows := make([][]float64, n)
	targets := make([][]float64, n)
	wTrue := []float64{1.5, -2.0, 0.7}
	for s := 0; s < n; s++ {
		o := make([]float64, m)
		var v float64
		for i := 0; i < m; i++ {
			o[i] = rng.NormFloat64()
			v += wTrue[i] * o[i]
		}
		rows[s] = o
		targets[s] = []float64{v}
	}

	rn := NewRidgeNormal(m, nOut)
	for s := 0; s < n; s++ {
		rn.Accumulate(rows[s], 1, m, targets[s], nOut)
	}
	wBatch, err := rn.Solve(ridge)
	if err != nil {
		t.Fatal(err)
	}
	p0, err := rn.InvCovariance(ridge)
	if err != nil {
		t.Fatal(err)
	}

	r := NewRLSWithCov(m, nOut, 1.0, p0, wBatch)
	// Stream a few fresh consistent samples; the solution should barely move.
	for s := 0; s < 50; s++ {
		o := make([]float64, m)
		var v float64
		for i := 0; i < m; i++ {
			o[i] = rng.NormFloat64()
			v += wTrue[i] * o[i]
		}
		r.Update(o, []float64{v})
	}
	got := r.Weights()
	for i := range wTrue {
		if math.Abs(got[i]-wTrue[i]) > 1e-3 {
			t.Errorf("weight[%d] = %.5f, want ~%.5f", i, got[i], wTrue[i])
		}
	}
}
