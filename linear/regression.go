// Package linear implements linear regression: a model that predicts a numeric
// target as a weighted sum of the input features plus a constant.
//
// In plain terms, it learns one number (a "coefficient") for each feature, plus
// one overall offset (the "intercept"). A prediction is then:
//
//	prediction = intercept + coef[0]*feature[0] + coef[1]*feature[1] + ...
//
// It is the right first model for most "predict a quantity" problems: fast,
// needs little data, and its coefficients are easy to interpret.
//
// Two training methods are provided and chosen automatically (see Solver):
//   - the closed-form "normal equation", which computes the exact best-fit
//     coefficients in one shot; and
//   - mini-batch stochastic gradient descent (SGD), which learns iteratively and
//     supports streaming and online learning.
package linear

import (
	"context"
	"fmt"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/internal/linalg"
	"github.com/nakurai/goblas-ai/preprocess"
)

// Solver selects how the model is trained.
type Solver int

const (
	// Auto lets the library pick: the exact closed-form solver for a modest
	// number of features, and SGD when there are many features. Either way the
	// data is streamed, so dataset size is not the deciding factor.
	Auto Solver = iota
	// ClosedForm computes the exact least-squares solution via the normal
	// equation. Fast and hyperparameter-free for a modest number of features.
	ClosedForm
	// SGD trains iteratively with mini-batch stochastic gradient descent. Scales
	// to very wide data and supports online learning via PartialFit.
	SGD
)

func (s Solver) String() string {
	switch s {
	case ClosedForm:
		return "closed_form"
	case SGD:
		return "sgd"
	default:
		return "auto"
	}
}

// closedFormFeatureLimit is the number of features at or below which Auto uses
// the exact closed-form solver. Above it, the p×p normal-equation solve becomes
// the bottleneck and SGD is preferred.
const closedFormFeatureLimit = 1000

// Regression is a linear-regression model. Create one with NewRegression, train
// it with Fit (or PartialFit for online learning), and make predictions with
// Predict / PredictBatch. It implements model.Estimator, model.Predictor,
// model.OnlineEstimator and model.Persistable.
type Regression struct {
	// hyperparameters
	solver       Solver
	fitIntercept bool
	standardize  bool
	batchSize    int

	// SGD hyperparameters (see sgd.go)
	epochs       int
	learningRate float64
	l2           float64
	checkpoint   *checkpointConfig

	// learned state
	coef      []float64                  // coefficients in the (standardized) feature space
	intercept float64                    // overall offset
	scaler    *preprocess.StandardScaler // nil when standardize is disabled
	features  []string
	nFeatures int
	steps     int64 // number of SGD updates applied (for online/resume bookkeeping)
	fitted    bool
}

// Option configures a Regression at construction time.
type Option func(*Regression)

// WithSolver overrides the training method (default Auto).
func WithSolver(s Solver) Option { return func(r *Regression) { r.solver = s } }

// WithIntercept controls whether an overall offset term is fitted (default true).
// Leave it on unless you have a specific reason to force predictions through zero.
func WithIntercept(b bool) Option { return func(r *Regression) { r.fitIntercept = b } }

// WithStandardize controls whether features are centered and scaled before
// training (default true). The scaling is stored in the model and reapplied
// automatically at prediction time.
func WithStandardize(b bool) Option { return func(r *Regression) { r.standardize = b } }

// WithBatchSize sets how many rows are processed at a time while streaming
// (default 8192). Larger uses more memory; smaller uses less.
func WithBatchSize(n int) Option { return func(r *Regression) { r.batchSize = n } }

// NewRegression creates a linear-regression model with sensible defaults: Auto
// solver, an intercept term, feature standardization on, and a streaming batch
// size of 8192. Pass Options to override.
func NewRegression(opts ...Option) *Regression {
	r := &Regression{
		solver:       Auto,
		fitIntercept: true,
		standardize:  true,
		batchSize:    8192,
		epochs:       10,
		learningRate: 0.01,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Fit trains the model on data, honouring ctx cancellation. The solver is chosen
// per the configured Solver (Auto picks one based on the number of features).
func (r *Regression) Fit(ctx context.Context, data dataset.Provider) error {
	r.features = append([]string(nil), data.FeatureNames()...)
	r.nFeatures = data.NFeatures()

	solver := r.solver
	if solver == Auto {
		if r.nFeatures <= closedFormFeatureLimit {
			solver = ClosedForm
		} else {
			solver = SGD
		}
	}

	switch solver {
	case ClosedForm:
		return r.fitClosedForm(ctx, data)
	case SGD:
		return r.fitSGD(ctx, data)
	default:
		return fmt.Errorf("linear: unknown solver %v", solver)
	}
}

// fitClosedForm computes the exact least-squares solution by streaming the data
// once to fit the scaler (if enabled) and once more to accumulate the normal
// equation, then solving it.
func (r *Regression) fitClosedForm(ctx context.Context, data dataset.Provider) error {
	p := r.nFeatures

	if r.standardize {
		sc := preprocess.NewStandardScaler(p)
		for batch, err := range data.Batches(r.batchSize) {
			if err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			sc.Observe(batch.X)
		}
		sc.Fit()
		r.scaler = sc
	}

	cols := p
	if r.fitIntercept {
		cols = p + 1
	}
	ne := linalg.NewNormalEquations(cols)
	for batch, err := range data.Batches(r.batchSize) {
		if err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		z := r.buildDesign(batch.X)
		ne.Accumulate(z, batch.X.Rows, cols, batch.Y)
	}

	sol, err := ne.Solve()
	if err != nil {
		return err
	}
	if r.fitIntercept {
		r.coef = sol[:p]
		r.intercept = sol[p]
	} else {
		r.coef = sol
		r.intercept = 0
	}
	r.fitted = true
	return nil
}

// buildDesign turns a raw feature matrix into the design matrix the solver
// consumes: features are standardized (if enabled) and, when fitting an
// intercept, a trailing column of ones is appended.
func (r *Regression) buildDesign(x dataset.Matrix) []float64 {
	p := x.Cols
	cols := p
	if r.fitIntercept {
		cols = p + 1
	}
	z := make([]float64, x.Rows*cols)
	for i := 0; i < x.Rows; i++ {
		dst := z[i*cols : i*cols+cols]
		copy(dst[:p], x.Row(i))
		if r.scaler != nil {
			r.scaler.Apply(dst[:p])
		}
		if r.fitIntercept {
			dst[p] = 1
		}
	}
	return z
}

// Predict returns the model's prediction for a single row of raw features.
func (r *Regression) Predict(row []float64) (float64, error) {
	if !r.fitted {
		return 0, fmt.Errorf("linear: model is not trained")
	}
	if len(row) != r.nFeatures {
		return 0, fmt.Errorf("linear: expected %d features, got %d", r.nFeatures, len(row))
	}
	return r.predictTransformed(row), nil
}

// PredictBatch returns one prediction per row of X (raw features).
func (r *Regression) PredictBatch(x dataset.Matrix) ([]float64, error) {
	if !r.fitted {
		return nil, fmt.Errorf("linear: model is not trained")
	}
	if x.Cols != r.nFeatures {
		return nil, fmt.Errorf("linear: expected %d features, got %d", r.nFeatures, x.Cols)
	}
	out := make([]float64, x.Rows)
	for i := 0; i < x.Rows; i++ {
		out[i] = r.predictTransformed(x.Row(i))
	}
	return out, nil
}

// predictTransformed computes the prediction for a raw feature row, applying the
// stored scaler on the fly without mutating the caller's slice.
func (r *Regression) predictTransformed(row []float64) float64 {
	sum := r.intercept
	if r.scaler != nil {
		for j, v := range row {
			std := (v - r.scaler.Mean[j]) / r.scaler.Std[j]
			sum += r.coef[j] * std
		}
	} else {
		for j, v := range row {
			sum += r.coef[j] * v
		}
	}
	return sum
}

// Coefficients returns the model's coefficients expressed in the ORIGINAL,
// unscaled feature space (one per feature), regardless of whether
// standardization was used. These are the numbers to read for interpretation:
// "holding everything else fixed, increasing this feature by 1 changes the
// prediction by this much." The paired intercept is also returned.
func (r *Regression) Coefficients() (coef []float64, intercept float64) {
	coef = make([]float64, len(r.coef))
	intercept = r.intercept
	if r.scaler == nil {
		copy(coef, r.coef)
		return coef, intercept
	}
	for j := range r.coef {
		coef[j] = r.coef[j] / r.scaler.Std[j]
		intercept -= r.coef[j] * r.scaler.Mean[j] / r.scaler.Std[j]
	}
	return coef, intercept
}

// FeatureNames returns the feature column names captured during training.
func (r *Regression) FeatureNames() []string { return r.features }
