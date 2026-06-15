// Package linalg is the internal linear-algebra backend for goblas-ai.
//
// It wires the pure-Go goblas BLAS implementation into Gonum (via blasadapt)
// so that Gonum's high-level routines (Cholesky, Solve, SVD) run on goblas
// kernels with no CGo. This package is internal on purpose: Gonum types must
// never appear in goblas-ai's public API, so the math backend can be swapped
// later without breaking users.
package linalg

import (
	"errors"

	"github.com/nakurai/goblas/blasadapt"
	"gonum.org/v1/gonum/mat"
)

func init() {
	// Register goblas as the BLAS backend used by gonum/mat and gonum's LAPACK.
	blasadapt.Use()
}

// SolveSPD solves the linear system A·x = b where A is a symmetric
// positive-definite p×p matrix stored row-major in a (len(a) == p*p) and b has
// length p. It is the workhorse behind the linear-regression normal equation
// (XᵀX)·β = Xᵀy.
//
// It first attempts a Cholesky factorization (fast, accurate for well-behaved
// data). If A is not positive-definite — which happens when features are
// collinear or the data is ill-conditioned — it adds a tiny ridge term to the
// diagonal and retries, which is numerically equivalent to a minuscule amount
// of L2 regularization and almost never changes the fitted model in practice.
func SolveSPD(a []float64, p int, b []float64) ([]float64, error) {
	if p <= 0 {
		return nil, errors.New("linalg: matrix dimension must be positive")
	}
	if len(a) != p*p {
		return nil, errors.New("linalg: matrix data length does not match dimension")
	}
	if len(b) != p {
		return nil, errors.New("linalg: right-hand side length does not match dimension")
	}

	if x, ok := choleskySolve(a, p, b); ok {
		return x, nil
	}

	// Fallback: add a small ridge proportional to the average diagonal magnitude.
	var trace float64
	for i := 0; i < p; i++ {
		trace += a[i*p+i]
	}
	ridge := 1e-8
	if trace > 0 {
		ridge = 1e-8 * (trace / float64(p))
	}
	ridged := make([]float64, len(a))
	copy(ridged, a)
	for i := 0; i < p; i++ {
		ridged[i*p+i] += ridge
	}
	if x, ok := choleskySolve(ridged, p, b); ok {
		return x, nil
	}
	return nil, errors.New("linalg: system is singular and could not be solved")
}

// choleskySolve attempts a Cholesky-based solve, returning ok=false if the
// matrix is not positive-definite.
func choleskySolve(a []float64, p int, b []float64) (x []float64, ok bool) {
	// NewSymDense copies the data, so the caller's slice is never mutated.
	sym := mat.NewSymDense(p, a)
	var chol mat.Cholesky
	if !chol.Factorize(sym) {
		return nil, false
	}
	dst := mat.NewVecDense(p, nil)
	if err := chol.SolveVecTo(dst, mat.NewVecDense(p, b)); err != nil {
		return nil, false
	}
	out := make([]float64, p)
	copy(out, dst.RawVector().Data)
	return out, true
}
