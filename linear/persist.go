package linear

import (
	"encoding/json"
	"io"

	"github.com/nakurai/goblas-ai/model"
	"github.com/nakurai/goblas-ai/preprocess"
)

// algorithmName is this model's key in the model registry.
const algorithmName = "linear_regression"

func init() {
	// Registering here means a blank import of this package is enough for
	// model.Load / model.LoadFile to reconstruct linear-regression models:
	//   import _ "github.com/nakurai/goblas-ai/linear"
	model.Register(algorithmName, func() model.Persistable { return &Regression{} })
}

// weightState is the JSON-serializable snapshot of a trained model. It captures
// everything needed both to make predictions and to resume training.
type weightState struct {
	Coef         []float64 `json:"coef"`
	Intercept    float64   `json:"intercept"`
	Features     []string  `json:"features"`
	NFeatures    int       `json:"n_features"`
	FitIntercept bool      `json:"fit_intercept"`
	Standardize  bool      `json:"standardize"`
	Solver       string    `json:"solver"`
	ScalerMean   []float64 `json:"scaler_mean,omitempty"`
	ScalerStd    []float64 `json:"scaler_std,omitempty"`

	// SGD bookkeeping, so online training can resume exactly.
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
	L2           float64 `json:"l2"`
	BatchSize    int     `json:"batch_size"`
	Steps        int64   `json:"steps"`
}

// Algorithm implements model.Persistable.
func (r *Regression) Algorithm() string { return algorithmName }

// MarshalWeights implements model.Persistable.
func (r *Regression) MarshalWeights() ([]byte, error) {
	ws := weightState{
		Coef:         r.coef,
		Intercept:    r.intercept,
		Features:     r.features,
		NFeatures:    r.nFeatures,
		FitIntercept: r.fitIntercept,
		Standardize:  r.standardize,
		Solver:       r.solver.String(),
		Epochs:       r.epochs,
		LearningRate: r.learningRate,
		L2:           r.l2,
		BatchSize:    r.batchSize,
		Steps:        r.steps,
	}
	if r.scaler != nil {
		ws.ScalerMean = r.scaler.Mean
		ws.ScalerStd = r.scaler.Std
	}
	return json.Marshal(ws)
}

// UnmarshalWeights implements model.Persistable.
func (r *Regression) UnmarshalWeights(data []byte) error {
	var ws weightState
	if err := json.Unmarshal(data, &ws); err != nil {
		return err
	}
	r.coef = ws.Coef
	r.intercept = ws.Intercept
	r.features = ws.Features
	r.nFeatures = ws.NFeatures
	r.fitIntercept = ws.FitIntercept
	r.standardize = ws.Standardize
	r.batchSize = ws.BatchSize
	if r.batchSize <= 0 {
		r.batchSize = 8192 // default for models saved before batch size was recorded
	}
	r.epochs = ws.Epochs
	r.learningRate = ws.LearningRate
	r.l2 = ws.L2
	r.steps = ws.Steps
	switch ws.Solver {
	case "closed_form":
		r.solver = ClosedForm
	case "sgd":
		r.solver = SGD
	default:
		r.solver = Auto
	}
	if ws.ScalerMean != nil {
		r.scaler = &preprocess.StandardScaler{Mean: ws.ScalerMean, Std: ws.ScalerStd}
	}
	r.fitted = true
	return nil
}

// Params implements model.Persistable, returning self-describing metadata for
// tooling. Coefficients are reported in the original (unscaled) feature space
// for interpretability.
func (r *Regression) Params() map[string]any {
	coef, intercept := r.Coefficients()
	perFeature := make(map[string]float64, len(coef))
	for i, name := range r.features {
		if i < len(coef) {
			perFeature[name] = coef[i]
		}
	}
	return map[string]any{
		"n_features":    r.nFeatures,
		"feature_names": r.features,
		"fit_intercept": r.fitIntercept,
		"standardize":   r.standardize,
		"solver":        r.solver.String(),
		"intercept":     intercept,
		"coefficients":  perFeature,
		"sgd_steps":     r.steps,
	}
}

// Save writes the trained model to w in the .gobl container format.
func (r *Regression) Save(w io.Writer) error { return model.Save(w, r) }

// SaveFile writes the trained model to the file at path.
func (r *Regression) SaveFile(path string) error { return model.SaveFile(path, r) }
