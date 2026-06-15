package ngrc

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/nakurai/goblas-ai/model"
	"github.com/nakurai/goblas-ai/preprocess"
)

// algorithmName is this model's key in the model registry.
const algorithmName = "ng_reservoir"

func init() {
	// A blank import of this package is enough for model.LoadModel to reconstruct
	// NG-RC models:  import _ "github.com/nakurai/goblas-ai/ngrc"
	model.Register(algorithmName, func() model.Persistable { return &Model{} })
}

// weightState is the JSON-serializable snapshot of a trained NG-RC model. The
// monomial layout is rebuilt from the hyperparameters on load, so it is not
// stored.
type weightState struct {
	K            int         `json:"taps"`
	S            int         `json:"stride"`
	Order        int         `json:"order"`
	Constant     bool        `json:"constant"`
	Ridge        float64     `json:"ridge"`
	Standardize  bool        `json:"standardize"`
	PredictDelta bool        `json:"predict_delta"`
	D            int         `json:"n_vars"`
	Vars         []string    `json:"vars"`
	Wout         []float64   `json:"wout"`
	ScalerMean   []float64   `json:"scaler_mean,omitempty"`
	ScalerStd    []float64   `json:"scaler_std,omitempty"`
	Seed         [][]float64 `json:"seed"`
}

// Algorithm implements model.Persistable.
func (m *Model) Algorithm() string { return algorithmName }

// MarshalWeights implements model.Persistable.
func (m *Model) MarshalWeights() ([]byte, error) {
	ws := weightState{
		K:            m.k,
		S:            m.s,
		Order:        m.order,
		Constant:     m.constant,
		Ridge:        m.ridge,
		Standardize:  m.standardize,
		PredictDelta: m.predictDelta,
		D:            m.d,
		Vars:         m.vars,
		Wout:         m.wout,
		Seed:         m.seed,
	}
	if m.scaler != nil {
		ws.ScalerMean = m.scaler.Mean
		ws.ScalerStd = m.scaler.Std
	}
	return json.Marshal(ws)
}

// UnmarshalWeights implements model.Persistable, rebuilding the feature layout
// and reseeding the forecast buffer so the model is ready to use.
func (m *Model) UnmarshalWeights(data []byte) error {
	var ws weightState
	if err := json.Unmarshal(data, &ws); err != nil {
		return err
	}
	m.k = ws.K
	m.s = ws.S
	m.order = ws.Order
	m.constant = ws.Constant
	m.ridge = ws.Ridge
	m.standardize = ws.Standardize
	m.predictDelta = ws.PredictDelta
	m.d = ws.D
	m.vars = ws.Vars
	m.wout = ws.Wout
	m.seed = ws.Seed
	if ws.ScalerMean != nil {
		m.scaler = &preprocess.StandardScaler{Mean: ws.ScalerMean, Std: ws.ScalerStd}
	}
	m.spec = newFeatureSpec(m.d, m.k, m.s, m.order, m.constant)
	m.fitted = true
	m.Reset()
	return nil
}

// Params implements model.Persistable, returning self-describing metadata for
// tooling such as `goblasai inspect`.
func (m *Model) Params() map[string]any {
	return map[string]any{
		"taps":          m.k,
		"stride":        m.s,
		"order":         m.order,
		"constant":      m.constant,
		"ridge":         m.ridge,
		"standardize":   m.standardize,
		"predict_delta": m.predictDelta,
		"n_vars":        m.d,
		"vars":          m.vars,
		"feature_dim":   featureDim(m),
		"warmup":        warmupOf(m),
	}
}

// featureDim / warmupOf read derived sizes, rebuilding the spec if needed (e.g.
// before training).
func featureDim(m *Model) int {
	if m.spec == nil {
		return 0
	}
	return m.spec.mTotal
}

func warmupOf(m *Model) int {
	if m.spec == nil {
		return 0
	}
	return m.spec.warmup()
}

// Save writes the trained model to w in the .gobl container format.
func (m *Model) Save(w io.Writer) error { return model.Save(w, m) }

// SaveFile writes the trained model to the file at path.
func (m *Model) SaveFile(path string) error { return model.SaveFile(path, m) }

// LoadFile loads a trained NG-RC model from path. Use this (rather than
// model.LoadFile) because NG-RC is a stateful sequence model used through Step
// and Forecast, not row-by-row prediction.
func LoadFile(path string) (*Model, error) {
	p, err := model.LoadModelFile(path)
	if err != nil {
		return nil, err
	}
	m, ok := p.(*Model)
	if !ok {
		return nil, fmt.Errorf("ngrc: %s is not an NG-RC model", path)
	}
	return m, nil
}
