package linalg

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

// RidgeNormal accumulates the pieces of a multi-output ridge-regression problem:
// the M×M matrix G = OᵀO and the M×nOut matrix C = OᵀY, one batch of rows at a
// time. Solving (G + ridge·I)·W = C yields the readout W (M×nOut).
//
// "Ridge" regression is ordinary least squares plus a penalty (ridge) on large
// weights, which keeps the solution stable when features are correlated — common
// in NG-RC, where polynomial features overlap heavily. Accumulating G and C
// incrementally keeps memory proportional to M² regardless of how long the input
// sequence is. The matrix products run through Gonum, which is backed by goblas.
type RidgeNormal struct {
	m    int
	nOut int
	g    *mat.Dense // M×M, accumulates OᵀO
	c    *mat.Dense // M×nOut, accumulates OᵀY
}

// NewRidgeNormal prepares an accumulator for feature dimension m and nOut output
// targets.
func NewRidgeNormal(m, nOut int) *RidgeNormal {
	return &RidgeNormal{
		m:    m,
		nOut: nOut,
		g:    mat.NewDense(m, m, nil),
		c:    mat.NewDense(m, nOut, nil),
	}
}

// Accumulate folds one batch into the running totals. o is the row-major feature
// matrix (rows×m), and y is the row-major target matrix (rows×nOut).
func (r *RidgeNormal) Accumulate(o []float64, rows, m int, y []float64, nOut int) {
	if m != r.m || nOut != r.nOut {
		panic("linalg: RidgeNormal dimensions do not match")
	}
	oMat := mat.NewDense(rows, m, o)
	yMat := mat.NewDense(rows, nOut, y)

	var oto mat.Dense
	oto.Mul(oMat.T(), oMat) // OᵀO
	r.g.Add(r.g, &oto)

	var oty mat.Dense
	oty.Mul(oMat.T(), yMat) // OᵀY
	r.c.Add(r.c, &oty)
}

// Solve returns the readout W (M×nOut, row-major) satisfying
// (G + ridge·I)·W = C. ridge must be > 0 for a guaranteed solution; a larger
// value yields a smoother, more stable readout.
func (r *RidgeNormal) Solve(ridge float64) ([]float64, error) {
	// A = G + ridge·I (symmetric positive-definite for ridge > 0).
	aData := make([]float64, r.m*r.m)
	copy(aData, r.g.RawMatrix().Data)
	for i := 0; i < r.m; i++ {
		aData[i*r.m+i] += ridge
	}
	sym := mat.NewSymDense(r.m, aData)

	var chol mat.Cholesky
	if !chol.Factorize(sym) {
		return nil, errors.New("linalg: ridge system is not positive-definite; try a larger ridge value")
	}
	var w mat.Dense
	if err := chol.SolveTo(&w, r.c); err != nil {
		return nil, err
	}

	out := make([]float64, r.m*r.nOut)
	for i := 0; i < r.m; i++ {
		for j := 0; j < r.nOut; j++ {
			out[i*r.nOut+j] = w.At(i, j)
		}
	}
	return out, nil
}

// InvCovariance returns P = (G + ridge·I)⁻¹ (M×M, row-major): the inverse
// feature-covariance behind this batch solution. A recursive least-squares
// estimator started from this P (see NewRLSWithCov) continues the batch solution
// online as if the batch rows had been fed one at a time. ridge must be > 0.
func (r *RidgeNormal) InvCovariance(ridge float64) ([]float64, error) {
	aData := make([]float64, r.m*r.m)
	copy(aData, r.g.RawMatrix().Data)
	for i := 0; i < r.m; i++ {
		aData[i*r.m+i] += ridge
	}
	sym := mat.NewSymDense(r.m, aData)

	var chol mat.Cholesky
	if !chol.Factorize(sym) {
		return nil, errors.New("linalg: ridge system is not positive-definite; try a larger ridge value")
	}
	var inv mat.SymDense
	if err := chol.InverseTo(&inv); err != nil {
		return nil, err
	}

	out := make([]float64, r.m*r.m)
	for i := 0; i < r.m; i++ {
		for j := 0; j < r.m; j++ {
			out[i*r.m+j] = inv.At(i, j)
		}
	}
	return out, nil
}
