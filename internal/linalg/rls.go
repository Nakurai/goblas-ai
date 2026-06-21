package linalg

import "gonum.org/v1/gonum/mat"

// RLS solves a multi-output linear readout online by recursive least squares
// (RLS): the running analogue of the batch ridge solve in RidgeNormal. It maps
// feature vectors o (length M) to targets y (length nOut) through a readout W
// (M×nOut), refining W one sample at a time instead of accumulating the whole
// problem and solving once.
//
// It keeps an estimate P of the inverse feature-covariance (M×M, shared across
// all outputs) and applies a rank-1 update per sample. Because the gain is
// computed through P, RLS effectively whitens the features as it goes, so it is
// far less sensitive to feature scaling than plain gradient descent. The
// forgetting factor lambda in (0,1] discounts older samples: 1 weights all
// history equally, while a value slightly below 1 lets the readout track
// dynamics that drift over time.
type RLS struct {
	m      int
	nOut   int
	lambda float64
	p      *mat.Dense // M×M, inverse feature-covariance estimate
	w      *mat.Dense // M×nOut, readout
}

// NewRLS prepares an RLS estimator with the initial covariance P = p0·I. A large
// p0 means low confidence in the starting weights (fast adaptation); a small p0
// means high confidence (slow adaptation). w0 is the row-major M×nOut starting
// readout; pass nil for zeros.
func NewRLS(m, nOut int, lambda, p0 float64, w0 []float64) *RLS {
	p := mat.NewDense(m, m, nil)
	for i := 0; i < m; i++ {
		p.Set(i, i, p0)
	}
	return &RLS{m: m, nOut: nOut, lambda: lambda, p: p, w: newReadout(m, nOut, w0)}
}

// NewRLSWithCov prepares an RLS estimator from a full initial covariance p0 (M×M,
// row-major), so it can continue a batch ridge solution online seamlessly —
// p0 is typically RidgeNormal.InvCovariance. w0 is the row-major M×nOut starting
// readout; pass nil for zeros.
func NewRLSWithCov(m, nOut int, lambda float64, p0, w0 []float64) *RLS {
	p := mat.NewDense(m, m, append([]float64(nil), p0...))
	return &RLS{m: m, nOut: nOut, lambda: lambda, p: p, w: newReadout(m, nOut, w0)}
}

func newReadout(m, nOut int, w0 []float64) *mat.Dense {
	if w0 == nil {
		return mat.NewDense(m, nOut, nil)
	}
	return mat.NewDense(m, nOut, append([]float64(nil), w0...))
}

// Update folds one (o, y) sample into the readout, where o is the length-M
// feature vector and y the length-nOut target.
func (r *RLS) Update(o, y []float64) {
	oVec := mat.NewVecDense(r.m, o)

	// pi = P·o, gain k = pi / (lambda + oᵀ·pi).
	var pi mat.VecDense
	pi.MulVec(r.p, oVec)
	denom := r.lambda + mat.Dot(oVec, &pi)
	var k mat.VecDense
	k.ScaleVec(1/denom, &pi)

	// error e = y - Wᵀ·o.
	var pred mat.VecDense
	pred.MulVec(r.w.T(), oVec)
	e := mat.NewVecDense(r.nOut, append([]float64(nil), y...))
	e.SubVec(e, &pred)

	// W += k·eᵀ (rank-1, M×nOut).
	var dw mat.Dense
	dw.Mul(&k, e.T())
	r.w.Add(r.w, &dw)

	// P = (P - k·piᵀ) / lambda.
	var kpi mat.Dense
	kpi.Mul(&k, pi.T())
	r.p.Sub(r.p, &kpi)
	r.p.Scale(1/r.lambda, r.p)
}

// Weights returns the current readout W (M×nOut, row-major).
func (r *RLS) Weights() []float64 {
	out := make([]float64, r.m*r.nOut)
	for i := 0; i < r.m; i++ {
		for j := 0; j < r.nOut; j++ {
			out[i*r.nOut+j] = r.w.At(i, j)
		}
	}
	return out
}
