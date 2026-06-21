// Package ngrc implements Next-Generation Reservoir Computing (NG-RC), a
// lightweight, accurate method for forecasting time series — data where each
// reading depends on what came just before, like sensor traces, prices over
// time, or the state of a physical system.
//
// What it does, in plain terms: to predict the next reading, NG-RC looks at the
// last few readings and at simple combinations of them (products of pairs), then
// applies one learned linear formula. There is no neural network and no
// randomness, so training is fast, needs little data, and gives the same result
// every time.
//
// A trained model can be used two ways (see Step and Forecast):
//   - one-step-ahead: given the latest real reading, predict the next one; or
//   - autonomous forecasting: predict the next reading, feed that prediction
//     back in, and repeat to simulate the future with no new data.
package ngrc

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/internal/linalg"
	"github.com/nakurai/goblas-ai/preprocess"
)

// Model is a trained (or trainable) NG-RC time-series model. Create one with
// New, train it with Fit, then predict with Step or Forecast. It implements
// model.Persistable so it can be saved and reloaded.
type Model struct {
	// hyperparameters
	k            int     // number of delay taps (how far back to look)
	s            int     // stride between taps (spacing)
	order        int     // polynomial order of the feature combinations
	constant     bool    // include a constant bias feature
	ridge        float64 // ridge (L2) regularization strength
	standardize  bool    // scale variables before training
	predictDelta bool    // predict the change between steps rather than the absolute value
	online       bool    // keep learning from new readings via RLS (see Update)
	forget       float64 // RLS forgetting factor in (0,1]

	// learned state
	spec   *featureSpec
	wout   []float64                  // readout, M×d row-major
	d      int                        // number of variables
	vars   []string                   // variable names
	scaler *preprocess.StandardScaler // nil if standardize is off

	// seed window: the last (k-1)*s+1 raw states from training, so a loaded model
	// can Forecast immediately.
	seed [][]float64

	// inference buffer: recent scaled states, oldest first (runtime only).
	buf    [][]float64
	fitted bool

	// online (RLS) training state, set up by Fit or PrimeRandom when online is on.
	rls *linalg.RLS
}

// Option configures a Model at construction time.
type Option func(*Model)

// WithTaps sets how many past readings to look at, including the current one
// (default 2). More taps capture longer memory at the cost of a bigger model.
func WithTaps(k int) Option { return func(m *Model) { m.k = k } }

// WithStride sets the spacing between the taps (default 1). A stride of 2 looks
// at every other reading, reaching further back for the same number of taps.
func WithStride(s int) Option { return func(m *Model) { m.s = s } }

// WithOrder sets how complex the feature combinations are (default 2). Order 1
// uses only the readings themselves; order 2 also uses products of pairs, which
// is what lets NG-RC capture nonlinear dynamics.
func WithOrder(order int) Option { return func(m *Model) { m.order = order } }

// WithRidge sets the regularization strength (default 1e-6). Larger values make
// the model smoother and more stable when features are highly correlated.
func WithRidge(alpha float64) Option { return func(m *Model) { m.ridge = alpha } }

// WithConstant controls whether a constant bias feature is included (default true).
func WithConstant(b bool) Option { return func(m *Model) { m.constant = b } }

// WithStandardize controls whether variables are scaled to a common range before
// training (default true); the scaling is stored in the model and applied
// automatically during prediction.
func WithStandardize(b bool) Option { return func(m *Model) { m.standardize = b } }

// WithPredictDelta controls whether the model predicts the change from one step
// to the next (default true, usually more accurate for smooth data) or the
// absolute next value (false).
func WithPredictDelta(b bool) Option { return func(m *Model) { m.predictDelta = b } }

// WithOnline enables online training via recursive least squares (RLS), so the
// model keeps learning from new readings through Update. forget is the forgetting
// factor in (0,1]: 1 weights all history equally, while a value slightly below 1
// (e.g. 0.999) discounts older data so the model can track slowly drifting
// dynamics. The online state is initialized by Fit or PrimeRandom.
func WithOnline(forget float64) Option {
	return func(m *Model) { m.online = true; m.forget = forget }
}

// New creates an NG-RC model with sensible defaults: 2 taps, stride 1, order 2,
// a small ridge, standardization on, and change-prediction on. Pass Options to
// override.
func New(opts ...Option) *Model {
	m := &Model{
		k:            2,
		s:            1,
		order:        2,
		constant:     true,
		ridge:        1e-6,
		standardize:  true,
		predictDelta: true,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// blockRows bounds how many feature rows are accumulated at once, keeping memory
// in check for long sequences.
const blockRows = 1024

// Fit trains the model on a time series, honouring ctx cancellation.
func (m *Model) Fit(ctx context.Context, seq *dataset.Sequence) error {
	m.d = seq.Dim()
	m.vars = append([]string(nil), seq.Vars...)
	m.spec = newFeatureSpec(m.d, m.k, m.s, m.order, m.constant)

	warm := m.spec.warmup()
	T := seq.Len()
	// Need at least one training pair: a step t in [warm, T-2].
	if T < warm+2 {
		return fmt.Errorf("ngrc: series too short: have %d steps, need at least %d for these settings", T, warm+2)
	}

	if m.standardize {
		sc := preprocess.NewStandardScaler(m.d)
		sc.Observe(seq.Data)
		sc.Fit()
		m.scaler = sc
	}

	M := m.spec.mTotal
	rn := linalg.NewRidgeNormal(M, m.d)

	oBlock := make([]float64, 0, blockRows*M)
	yBlock := make([]float64, 0, blockRows*m.d)
	count := 0
	flush := func() {
		if count == 0 {
			return
		}
		rn.Accumulate(oBlock, count, M, yBlock, m.d)
		oBlock = oBlock[:0]
		yBlock = yBlock[:0]
		count = 0
	}

	lin := make([]float64, m.spec.m)
	o := make([]float64, M)
	for t := warm; t <= T-2; t++ {
		if t%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		// Assemble the linear (delay-embedded) part from scaled states.
		for tap := 0; tap < m.k; tap++ {
			st := m.scale(seq.Step(t - tap*m.s))
			copy(lin[tap*m.d:(tap+1)*m.d], st)
		}
		m.spec.build(o, lin)
		oBlock = append(oBlock, o...)

		// Target: the next step, as a change (default) or absolute value.
		next := m.scale(seq.Step(t + 1))
		if m.predictDelta {
			cur := m.scale(seq.Step(t))
			for j := 0; j < m.d; j++ {
				yBlock = append(yBlock, next[j]-cur[j])
			}
		} else {
			yBlock = append(yBlock, next...)
		}

		count++
		if count == blockRows {
			flush()
		}
	}
	flush()

	wout, err := rn.Solve(m.ridge)
	if err != nil {
		return err
	}
	m.wout = wout

	// Store the trailing raw window so Forecast works straight after training (or
	// after a reload).
	L := warm + 1
	m.seed = make([][]float64, L)
	for i := 0; i < L; i++ {
		m.seed[i] = append([]float64(nil), seq.Step(T-L+i)...)
	}
	m.fitted = true
	m.Reset()

	// If online learning is on, seed the RLS estimator from the batch covariance
	// so Update continues this fit seamlessly.
	if m.online {
		p0, err := rn.InvCovariance(m.ridge)
		if err != nil {
			return err
		}
		m.rls = linalg.NewRLSWithCov(M, m.d, m.forget, p0, m.wout)
	}
	return nil
}

// Step predicts the next reading given the latest real reading. It records the
// reading internally, so calling Step repeatedly as new data arrives produces a
// running one-step-ahead forecast. The model must be primed first — either by
// training (Fit) or by Prime/Reset — with enough history to fill the delay
// window; until then Step returns an error.
func (m *Model) Step(nextInput []float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("ngrc: model is not trained")
	}
	if len(nextInput) != m.d {
		return nil, fmt.Errorf("ngrc: expected %d variables, got %d", m.d, len(nextInput))
	}
	m.push(m.scale(nextInput))
	L := m.spec.warmup() + 1
	if len(m.buf) < L {
		return nil, fmt.Errorf("ngrc: need %d readings to start predicting, have %d", L, len(m.buf))
	}
	return m.unscale(m.predictNext(m.buf)), nil
}

// Forecast autonomously simulates the next n readings, feeding each prediction
// back in as if it were real. It rolls forward from the current buffer (set by
// Fit, Reset, or Prime) without changing that buffer, so repeated calls are
// reproducible.
func (m *Model) Forecast(n int) ([][]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("ngrc: model is not trained")
	}
	L := m.spec.warmup() + 1
	if len(m.buf) < L {
		return nil, fmt.Errorf("ngrc: not enough history to forecast: have %d, need %d (call Prime or Reset)", len(m.buf), L)
	}
	// Work on a copy so the model's buffer is left untouched.
	work := make([][]float64, len(m.buf))
	for i, st := range m.buf {
		work[i] = append([]float64(nil), st...)
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		next := m.predictNext(work)
		out[i] = m.unscale(next)
		work = pushTrim(work, next, L)
	}
	return out, nil
}

// Prime sets the model's buffer from a provided window of recent readings, so
// Step/Forecast continue from that context instead of the end of training. The
// window must contain at least warmup+1 readings.
func (m *Model) Prime(window *dataset.Sequence) error {
	if !m.fitted {
		return fmt.Errorf("ngrc: model is not trained")
	}
	L := m.spec.warmup() + 1
	if window.Len() < L {
		return fmt.Errorf("ngrc: priming window has %d readings, need at least %d", window.Len(), L)
	}
	m.buf = m.buf[:0]
	for t := window.Len() - L; t < window.Len(); t++ {
		m.push(m.scale(window.Step(t)))
	}
	return nil
}

// Reset reseeds the buffer from the window stored at the end of training, so
// Forecast restarts from where training left off.
func (m *Model) Reset() {
	L := m.spec.warmup() + 1
	m.buf = make([][]float64, 0, L)
	for _, raw := range m.seed {
		m.push(m.scale(raw))
	}
}

// Update feeds the next real reading into an online model (one constructed with
// WithOnline). It is a learning variant of Step: when the delay window is full it
// first applies one recursive-least-squares update to the readout — using the
// current window as features and this reading as the target — then records the
// reading and returns the model's one-step-ahead prediction for the following
// step. Until enough readings have arrived to fill the window it records the
// reading and returns a "need more readings" error, like Step.
func (m *Model) Update(reading []float64) ([]float64, error) {
	if !m.fitted {
		return nil, fmt.Errorf("ngrc: model is not trained")
	}
	if m.rls == nil {
		return nil, fmt.Errorf("ngrc: online learning is not enabled; construct with WithOnline")
	}
	if len(reading) != m.d {
		return nil, fmt.Errorf("ngrc: expected %d variables, got %d", m.d, len(reading))
	}

	scaled := m.scale(reading)
	L := m.spec.warmup() + 1

	// With a full window, form a training pair (features now, target = this
	// reading) and refine the readout before recording the reading.
	if len(m.buf) >= L {
		lin := m.assembleLin(m.buf)
		o := make([]float64, m.spec.mTotal)
		m.spec.build(o, lin)

		y := make([]float64, m.d)
		if m.predictDelta {
			cur := m.buf[len(m.buf)-1]
			for j := 0; j < m.d; j++ {
				y[j] = scaled[j] - cur[j]
			}
		} else {
			copy(y, scaled)
		}
		m.rls.Update(o, y)
		copy(m.wout, m.rls.Weights()) // make predictNext see the updated readout
	}

	m.push(scaled)
	if len(m.buf) < L {
		return nil, fmt.Errorf("ngrc: need %d readings to start predicting, have %d", L, len(m.buf))
	}
	return m.unscale(m.predictNext(m.buf)), nil
}

// PrimeRandom sets the model up with random readout weights instead of training,
// so an online model (WithOnline) can start learning from scratch through Update.
// vars names the variables and fixes their count. If window is non-nil and
// standardization is on, the scaler is fit from it and the delay buffer is seeded
// from its tail; if window is nil, the model runs in raw space (standardization
// off) and the buffer fills as Update is called. seed makes the random weights
// reproducible.
func (m *Model) PrimeRandom(vars []string, window *dataset.Sequence, seed int64) error {
	if len(vars) == 0 {
		return fmt.Errorf("ngrc: PrimeRandom needs at least one variable")
	}
	m.d = len(vars)
	m.vars = append([]string(nil), vars...)
	m.spec = newFeatureSpec(m.d, m.k, m.s, m.order, m.constant)

	// Standardization needs data to estimate scale; without a window, fall back to
	// raw space.
	if m.standardize && window != nil {
		if window.Dim() != m.d {
			return fmt.Errorf("ngrc: window has %d variables, want %d", window.Dim(), m.d)
		}
		sc := preprocess.NewStandardScaler(m.d)
		sc.Observe(window.Data)
		sc.Fit()
		m.scaler = sc
	} else {
		m.scaler = nil
	}

	// Small random readout (M×d, row-major). Small weights keep a delta-model's
	// first predictions close to "no change" until RLS adapts.
	M := m.spec.mTotal
	rng := rand.New(rand.NewSource(seed))
	m.wout = make([]float64, M*m.d)
	const weightScale = 0.01
	for i := range m.wout {
		m.wout[i] = rng.NormFloat64() * weightScale
	}

	m.fitted = true

	// Seed the buffer (and seed window for Reset) from the tail of window, if given.
	m.buf = m.buf[:0]
	m.seed = nil
	if window != nil {
		L := m.spec.warmup() + 1
		start := window.Len() - L
		if start < 0 {
			start = 0
		}
		m.seed = make([][]float64, 0, window.Len()-start)
		for t := start; t < window.Len(); t++ {
			m.seed = append(m.seed, append([]float64(nil), window.Step(t)...))
			m.push(m.scale(window.Step(t)))
		}
	}

	// Large initial covariance: low confidence in the random weights, so RLS adapts
	// quickly.
	if m.online {
		m.rls = linalg.NewRLS(M, m.d, m.forget, 1/m.ridge, m.wout)
	}
	return nil
}

// Vars returns the variable names, in column order.
func (m *Model) Vars() []string { return m.vars }

// predictNext computes the next scaled state from a buffer of scaled states.
func (m *Model) predictNext(buf [][]float64) []float64 {
	lin := m.assembleLin(buf)
	o := make([]float64, m.spec.mTotal)
	m.spec.build(o, lin)
	pred := make([]float64, m.d)
	linalg.MatTVec(m.wout, m.spec.mTotal, m.d, o, pred) // pred = woutᵀ · o
	if m.predictDelta {
		cur := buf[len(buf)-1]
		for j := range pred {
			pred[j] += cur[j]
		}
	}
	return pred
}

// assembleLin gathers the delay-embedded linear part from a buffer of scaled
// states (most recent last).
func (m *Model) assembleLin(buf [][]float64) []float64 {
	last := len(buf) - 1
	lin := make([]float64, m.spec.m)
	for tap := 0; tap < m.k; tap++ {
		st := buf[last-tap*m.s]
		copy(lin[tap*m.d:(tap+1)*m.d], st)
	}
	return lin
}

func (m *Model) push(scaledState []float64) {
	m.buf = pushTrim(m.buf, scaledState, m.spec.warmup()+1)
}

func pushTrim(buf [][]float64, st []float64, L int) [][]float64 {
	buf = append(buf, st)
	if len(buf) > L {
		buf = buf[len(buf)-L:]
	}
	return buf
}

func (m *Model) scale(raw []float64) []float64 {
	out := make([]float64, m.d)
	if m.scaler == nil {
		copy(out, raw)
		return out
	}
	for j := 0; j < m.d; j++ {
		out[j] = (raw[j] - m.scaler.Mean[j]) / m.scaler.Std[j]
	}
	return out
}

func (m *Model) unscale(scaled []float64) []float64 {
	out := make([]float64, m.d)
	if m.scaler == nil {
		copy(out, scaled)
		return out
	}
	for j := 0; j < m.d; j++ {
		out[j] = scaled[j]*m.scaler.Std[j] + m.scaler.Mean[j]
	}
	return out
}
