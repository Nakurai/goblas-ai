// Package model defines the small, shared contracts that every goblas-ai
// algorithm implements, plus the on-disk format used to save and reload trained
// models.
//
// The contracts are intentionally split so that a program that only needs to
// *use* a trained model (a Predictor) does not have to depend on any of the
// training or data-loading machinery.
package model

import (
	"context"

	"github.com/nakurai/goblas-ai/dataset"
)

// Estimator is the training contract. Fit learns model parameters from a
// streaming data Provider. Implementations should honour ctx cancellation by
// stopping promptly (and, where they support it, writing a checkpoint first).
type Estimator interface {
	Fit(ctx context.Context, data dataset.Provider) error
}

// Predictor is the reuse / inference contract — the only thing a deployment
// binary needs. A model loaded with Load or LoadFile is returned as a Predictor.
//
// Predictions are made from raw, untransformed features: any preprocessing the
// model learned during training (such as feature scaling) is applied
// automatically inside these methods, so callers never have to reproduce it.
type Predictor interface {
	// Predict returns the prediction for a single row of features.
	Predict(row []float64) (float64, error)
	// PredictBatch returns one prediction per row of X.
	PredictBatch(X dataset.Matrix) ([]float64, error)
}

// OnlineEstimator is the optional online-learning contract. PartialFit updates
// the model from a single batch of new data without restarting training, which
// is what makes continuous / streaming learning and checkpoint-resume possible.
// Algorithms that cannot learn incrementally simply do not implement it.
type OnlineEstimator interface {
	PartialFit(ctx context.Context, batch dataset.Batch) error
}

// Persistable is what the save/load machinery needs from an algorithm. The
// container format (magic bytes, versioning, metadata) is handled centrally;
// the algorithm only has to serialize and restore its own learned weights and
// report a few descriptive parameters.
type Persistable interface {
	// Algorithm returns the registry key identifying this algorithm, e.g.
	// "linear_regression".
	Algorithm() string
	// MarshalWeights serializes the complete learned state needed to reproduce
	// predictions and to resume training.
	MarshalWeights() ([]byte, error)
	// UnmarshalWeights restores state previously produced by MarshalWeights.
	UnmarshalWeights([]byte) error
	// Params returns human-readable, self-describing metadata (hyperparameters,
	// feature names, etc.) stored alongside the model for tooling like
	// `goblasai inspect`.
	Params() map[string]any
}
