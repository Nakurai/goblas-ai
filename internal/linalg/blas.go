package linalg

import "github.com/nakurai/goblas"

// The goblas BLAS routines are column-major, while goblas-ai stores matrices
// row-major. Conveniently, a row-major rows×cols matrix occupies exactly the
// same memory as a column-major cols×rows matrix — i.e. its transpose. So these
// helpers pass the row-major data straight to goblas and simply flip the
// transpose flag, with no copying or explicit transpose.

// MatVec computes y = Z·x, where Z is a row-major rows×cols matrix (len
// rows*cols), x has length cols, and y has length rows. Runs on goblas.
func MatVec(z []float64, rows, cols int, x, y []float64) {
	if rows == 0 || cols == 0 {
		return
	}
	// Treat z as the column-major transpose Zᵀ (cols×rows, lda=cols); op = Trans
	// recovers Z.
	goblas.Dgemv(goblas.Trans, cols, rows, 1, z, cols, x, 1, 0, y, 1)
}

// MatTVec computes y = Zᵀ·x, where Z is a row-major rows×cols matrix, x has
// length rows, and y has length cols. Runs on goblas.
func MatTVec(z []float64, rows, cols int, x, y []float64) {
	if rows == 0 || cols == 0 {
		return
	}
	goblas.Dgemv(goblas.NoTrans, cols, rows, 1, z, cols, x, 1, 0, y, 1)
}

// Axpy computes y += alpha*x (in place). Runs on goblas.
func Axpy(alpha float64, x, y []float64) {
	goblas.Daxpy(len(x), alpha, x, 1, y, 1)
}
