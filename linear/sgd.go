package linear

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/internal/linalg"
	"github.com/nakurai/goblas-ai/model"
	"github.com/nakurai/goblas-ai/preprocess"
)

// checkpointConfig controls periodic saving during SGD training.
type checkpointConfig struct {
	path       string
	everySteps int
}

// WithEpochs sets how many full passes SGD makes over the data (default 10).
// More epochs can improve the fit but take longer.
func WithEpochs(n int) Option { return func(r *Regression) { r.epochs = n } }

// WithLearningRate sets the SGD step size (default 0.01). It controls how big an
// adjustment each update makes: too large can overshoot and diverge, too small
// learns slowly. The default is a safe starting point for standardized features.
func WithLearningRate(lr float64) Option { return func(r *Regression) { r.learningRate = lr } }

// WithL2 adds L2 regularization (also called "ridge"), a penalty on large
// coefficients that helps prevent overfitting (default 0, off). Larger values
// pull coefficients more strongly toward zero.
func WithL2(lambda float64) Option { return func(r *Regression) { r.l2 = lambda } }

// WithCheckpoint makes SGD periodically save the model to path, every
// everySteps mini-batch updates, and also on cancellation. Combined with
// LoadFile, this lets long or interruptible training resume where it left off.
func WithCheckpoint(path string, everySteps int) Option {
	return func(r *Regression) { r.checkpoint = &checkpointConfig{path: path, everySteps: everySteps} }
}

// fitSGD trains with mini-batch stochastic gradient descent. It streams the data
// once to fit the scaler (if enabled), then makes r.epochs passes, updating the
// coefficients after every batch. It checkpoints periodically and on
// cancellation, and resumes cleanly if coefficients are already present.
func (r *Regression) fitSGD(ctx context.Context, data dataset.Provider) error {
	p := r.nFeatures

	if r.standardize && r.scaler == nil {
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

	if r.coef == nil {
		r.coef = make([]float64, p)
	}

	for epoch := 0; epoch < r.epochs; epoch++ {
		for batch, err := range data.Batches(r.batchSize) {
			if err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				// Save progress before bailing out so training can resume.
				if cerr := r.writeCheckpoint(); cerr != nil {
					return fmt.Errorf("linear: checkpoint on cancel: %w (after %v)", cerr, err)
				}
				return err
			}
			r.sgdStep(batch)
			if r.shouldCheckpoint() {
				if err := r.writeCheckpoint(); err != nil {
					return fmt.Errorf("linear: write checkpoint: %w", err)
				}
			}
		}
	}
	r.fitted = true
	return nil
}

// sgdStep applies one gradient-descent update from a single batch, on
// standardized features. The matrix-vector products run on goblas.
func (r *Regression) sgdStep(batch dataset.Batch) {
	rows := batch.X.Rows
	p := batch.X.Cols
	if rows == 0 {
		return
	}

	// Standardized feature matrix (no intercept column; intercept handled below).
	z := make([]float64, rows*p)
	for i := 0; i < rows; i++ {
		dst := z[i*p : i*p+p]
		copy(dst, batch.X.Row(i))
		if r.scaler != nil {
			r.scaler.Apply(dst)
		}
	}

	pred := make([]float64, rows)
	linalg.MatVec(z, rows, p, r.coef, pred)
	if r.fitIntercept {
		for i := range pred {
			pred[i] += r.intercept
		}
	}

	// residual = prediction - truth
	resid := make([]float64, rows)
	for i := range resid {
		resid[i] = pred[i] - batch.Y[i]
	}

	// gradient of mean-squared error w.r.t. coefficients, plus L2 term
	grad := make([]float64, p)
	linalg.MatTVec(z, rows, p, resid, grad)
	scale := 2.0 / float64(rows)
	for j := range grad {
		grad[j] = grad[j]*scale + 2*r.l2*r.coef[j]
	}
	linalg.Axpy(-r.learningRate, grad, r.coef) // coef -= lr * grad

	if r.fitIntercept {
		var s float64
		for _, v := range resid {
			s += v
		}
		r.intercept -= r.learningRate * scale * s
	}
	r.steps++
}

func (r *Regression) shouldCheckpoint() bool {
	if r.checkpoint == nil || r.checkpoint.everySteps <= 0 {
		return false
	}
	return r.steps%int64(r.checkpoint.everySteps) == 0
}

// writeCheckpoint saves the current model atomically (write to a temp file, then
// rename) so an interrupted write never corrupts the checkpoint.
func (r *Regression) writeCheckpoint() error {
	r.fitted = true
	dir := filepath.Dir(r.checkpoint.path)
	tmp, err := os.CreateTemp(dir, ".ckpt-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if err := model.Save(tmp, r); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	return os.Rename(tmpName, r.checkpoint.path)
}

// PartialFit performs a single online-learning update from one batch of new
// data, without restarting training. The model keeps accumulating knowledge
// across calls, which is what enables continuous / streaming learning.
//
// On the very first call the model sizes itself to the batch. Note that feature
// standardization needs statistics over the whole dataset, so for purely online
// use either disable standardization (WithStandardize(false)) or bootstrap the
// scaler with an initial Fit; otherwise PartialFit trains on raw features.
func (r *Regression) PartialFit(ctx context.Context, batch dataset.Batch) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if r.nFeatures == 0 {
		r.nFeatures = batch.X.Cols
	}
	if batch.X.Cols != r.nFeatures {
		return fmt.Errorf("linear: expected %d features, got %d", r.nFeatures, batch.X.Cols)
	}
	if r.coef == nil {
		r.coef = make([]float64, r.nFeatures)
	}
	r.sgdStep(batch)
	r.fitted = true
	return nil
}

// LoadFile loads a trained linear-regression model from path as a concrete
// *Regression, so training can be continued (e.g. resuming from a checkpoint or
// further online learning). Use model.LoadFile instead when you only need to
// make predictions.
func LoadFile(path string) (*Regression, error) {
	p, err := model.LoadFile(path)
	if err != nil {
		return nil, err
	}
	r, ok := p.(*Regression)
	if !ok {
		return nil, fmt.Errorf("linear: %s is not a linear-regression model", path)
	}
	return r, nil
}
