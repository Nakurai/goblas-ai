// Package metrics provides the standard ways to measure how good a regression
// model's predictions are. Each function compares the values the model predicted
// against the true values.
package metrics

import "math"

// requireSameLen guards against comparing mismatched slices, which is always a
// programming error rather than bad data.
func requireSameLen(a, b []float64) {
	if len(a) != len(b) {
		panic("metrics: yTrue and yPred must have the same length")
	}
	if len(a) == 0 {
		panic("metrics: cannot compute a metric over zero values")
	}
}

// MSE is the Mean Squared Error: the average of the squared differences between
// predicted and true values. Squaring makes every error positive and punishes
// big misses much more than small ones. Lower is better; 0 means perfect.
func MSE(yTrue, yPred []float64) float64 {
	requireSameLen(yTrue, yPred)
	var sum float64
	for i := range yTrue {
		d := yPred[i] - yTrue[i]
		sum += d * d
	}
	return sum / float64(len(yTrue))
}

// RMSE is the Root Mean Squared Error: the square root of the MSE. Taking the
// square root brings the number back into the same units as the target (e.g.
// dollars instead of dollars-squared), which makes it easier to interpret.
// Lower is better; 0 means perfect.
func RMSE(yTrue, yPred []float64) float64 {
	return math.Sqrt(MSE(yTrue, yPred))
}

// MAE is the Mean Absolute Error: the average size of the errors, ignoring their
// direction. Unlike MSE it does not over-emphasize large errors, so it is a good
// "typical error" summary. Lower is better; 0 means perfect.
func MAE(yTrue, yPred []float64) float64 {
	requireSameLen(yTrue, yPred)
	var sum float64
	for i := range yTrue {
		sum += math.Abs(yPred[i] - yTrue[i])
	}
	return sum / float64(len(yTrue))
}

// R2 is the coefficient of determination, usually read as "R squared". It is the
// fraction of the variation in the target that the model explains:
//   - 1.0 means the model predicts perfectly.
//   - 0.0 means the model is no better than always guessing the average target.
//   - Negative means the model is worse than guessing the average.
//
// It is unitless, which makes it handy for comparing models across different
// datasets.
func R2(yTrue, yPred []float64) float64 {
	requireSameLen(yTrue, yPred)
	var mean float64
	for _, v := range yTrue {
		mean += v
	}
	mean /= float64(len(yTrue))

	var ssRes, ssTot float64 // residual sum of squares, total sum of squares
	for i := range yTrue {
		dRes := yTrue[i] - yPred[i]
		ssRes += dRes * dRes
		dTot := yTrue[i] - mean
		ssTot += dTot * dTot
	}
	if ssTot == 0 {
		// The target never varies; "explained variance" is undefined. Report a
		// perfect score only if the model also nailed the constant value.
		if ssRes == 0 {
			return 1
		}
		return 0
	}
	return 1 - ssRes/ssTot
}
