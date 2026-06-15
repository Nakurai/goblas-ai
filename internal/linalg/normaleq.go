package linalg

import "gonum.org/v1/gonum/mat"

// NormalEquations accumulates the pieces of the linear-regression "normal
// equation" — the p×p matrix A = ZᵀZ and the length-p vector b = Zᵀy — one batch
// of rows at a time. Solving A·β = b yields the least-squares coefficients β.
//
// Accumulating incrementally means the full design matrix never has to be held
// in memory: an arbitrarily large dataset can be streamed through Accumulate in
// batches, keeping memory proportional to p² rather than to the number of rows.
// The matrix products run through Gonum, which is backed by goblas.
type NormalEquations struct {
	p int
	a *mat.Dense // p×p, accumulates ZᵀZ
	b *mat.Dense // p×1, accumulates Zᵀy
}

// NewNormalEquations prepares an accumulator for a design matrix with p columns
// (features, plus one extra column if an intercept is being fitted).
func NewNormalEquations(p int) *NormalEquations {
	return &NormalEquations{
		p: p,
		a: mat.NewDense(p, p, nil),
		b: mat.NewDense(p, 1, nil),
	}
}

// Accumulate folds one batch into the running totals. z is the row-major design
// matrix for the batch (rows×cols, where cols must equal p), and y holds the
// matching target values (length rows).
func (ne *NormalEquations) Accumulate(z []float64, rows, cols int, y []float64) {
	if cols != ne.p {
		panic("linalg: design matrix column count does not match NormalEquations dimension")
	}
	zMat := mat.NewDense(rows, cols, z)
	yMat := mat.NewDense(rows, 1, y)

	var ztz mat.Dense
	ztz.Mul(zMat.T(), zMat) // ZᵀZ
	ne.a.Add(ne.a, &ztz)

	var zty mat.Dense
	zty.Mul(zMat.T(), yMat) // Zᵀy
	ne.b.Add(ne.b, &zty)
}

// Solve returns the coefficients β that satisfy A·β = b. It is valid to call
// Solve after any number of Accumulate calls.
func (ne *NormalEquations) Solve() ([]float64, error) {
	return SolveSPD(ne.a.RawMatrix().Data, ne.p, ne.b.RawMatrix().Data)
}
