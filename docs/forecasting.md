# Forecasting time series with NG-RC

This guide explains how to predict the future of a **time series** using
Next-Generation Reservoir Computing (NG-RC). No machine-learning background is
assumed; every term is defined as it appears.

## What is a time series?

A **time series** is data recorded in order over time: a temperature read every
minute, a stock price every day, the position of a moving object every fraction
of a second. The defining feature is that **order matters** — each reading is
related to the ones just before it. That is different from the tabular data used
for [linear regression](training.md), where each row stands on its own.

A time series can have several values per time step. For example, tracking an
object in 3-D gives three numbers — x, y, z — at each step. We call each of these
a **variable**, and a single time step a **state** (the full set of variables at
that moment).

As a CSV, a time series is just rows in time order, one column per variable:

```csv
x,y,z
2.80,4.51,14.28
3.18,5.25,13.83
3.59,6.05,13.50
...
```

## What does "forecasting" mean?

**Forecasting** is predicting future readings from past ones. Two flavours,
which NG-RC supports from the same trained model:

- **One-step-ahead**: given the latest *real* reading, predict the very next one.
  You keep feeding in real data as it arrives. Best for live monitoring and
  short-horizon "what comes next?".
- **Autonomous forecasting**: predict the next reading, then feed that prediction
  back in as if it were real, and repeat — generating a whole simulated future
  with no new data. Best for "show me the next 200 steps". Because it builds on
  its own guesses, it drifts further out.

## What NG-RC does, in plain terms

To predict the next state, NG-RC looks at:

1. **The recent past** — the current state and a few previous states. How many it
   looks at is the number of **taps**; how far apart they are is the **stride**.
2. **Simple combinations of those numbers** — specifically their products (e.g.
   value A times value B). This is what lets it capture curved, nonlinear
   behaviour. How complex the combinations get is the **order** (order 2 = include
   products of pairs).

It then applies **one learned linear formula** to all of those quantities to get
the next state. That formula is the only thing "trained", and training it is a
single fast calculation — no neural network, and no randomness, so you get the
same model every time.

This makes NG-RC unusually well-suited to the goals of this library: it learns
accurately from **short** histories and runs comfortably on a **CPU**.

## The settings (hyperparameters), explained

All have sensible defaults — you can ignore them at first.

- **taps** (default 2): how many readings back to look, including the current one.
  More taps = longer memory, bigger model.
- **stride** (default 1): the spacing between taps. Stride 2 looks at every other
  reading, reaching further back for the same number of taps.
- **order** (default 2): how complex the combinations are. 1 = just the readings;
  2 = also products of pairs (needed for curved/nonlinear systems).
- **ridge** (default 1e-6): how much to "smooth" the fit for stability. Increase
  it if forecasts look erratic or training complains the problem is unstable.
- **predict-the-change** (default on): the model learns how much each value
  *changes* from one step to the next, rather than the next value directly. This
  is usually more accurate for smooth data. You rarely need to turn it off.

## A complete example

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/metrics"
	"github.com/nakurai/goblas-ai/ngrc"
)

func main() {
	// 1. Load the series (all columns, in time order).
	seq, err := dataset.SequenceFromCSV("lorenz.csv")
	if err != nil {
		log.Fatal(err)
	}

	// 2. Split by TIME: train on the earlier part, test on the later part.
	//    (Never shuffle a time series — you must test on the future.)
	train, test := seq.SplitChrono(0.2)

	// 3. Train with defaults.
	m := ngrc.New()
	if err := m.Fit(context.Background(), train); err != nil {
		log.Fatal(err)
	}

	// 4. Measure one-step-ahead accuracy on the held-out future. After training,
	//    the model is primed with the end of the training data, which comes right
	//    before the test segment.
	var preds, truth []float64
	for i := 0; i < test.Len()-1; i++ {
		p, _ := m.Step(test.Step(i)) // predict the step after test[i]
		preds = append(preds, p...)
		truth = append(truth, test.Step(i+1)...)
	}
	fmt.Println("one-step R2:", metrics.R2(truth, preds)) // 1.0 is perfect

	// 5. Save, then later reload and simulate the future autonomously.
	m.SaveFile("lorenz.gobl")
	loaded, _ := ngrc.LoadFile("lorenz.gobl")
	future, _ := loaded.Forecast(200) // 200 simulated steps, each a []float64
	fmt.Println("first forecast step:", future[0])
}
```

A runnable version is in [examples/forecast/main.go](../examples/forecast/main.go):

```sh
go run ./examples/forecast
```

## Using it from the command line

```sh
# Train on a CSV time series (all columns by default; or pass --columns x,y,z).
goblasai train --algo ng_reservoir --data lorenz.csv --out lorenz.gobl

# See what was trained.
goblasai inspect lorenz.gobl

# Simulate the next 200 steps into a CSV.
goblasai forecast --model lorenz.gobl --steps 200 --out future.csv
```

Useful training flags: `--taps`, `--stride`, `--order`, `--ridge`, `--columns`,
and `--test-frac` (to report one-step accuracy on a held-out tail).

## Priming for live use

In a running program you typically want one-step predictions on a live feed.
After loading a model, give it a recent window of real readings, then call `Step`
as each new reading arrives:

```go
m, _ := ngrc.LoadFile("lorenz.gobl")
m.Prime(recentWindow)            // a *dataset.Sequence of the latest readings
for reading := range liveFeed {  // each new real reading
	next, _ := m.Step(reading)   // prediction for the following step
	use(next)
}
```

`Prime` needs at least `taps` readings (more precisely `(taps-1)*stride + 1`).
If you just want to continue from where training ended, call `Reset()` instead.

## Building a sequence in memory

You don't need a CSV to make a `dataset.Sequence`. For data you already have in
Go, use `SequenceFromValues` (one variable) or `NewSequence` (several):

```go
// One variable, in time order.
s := dataset.SequenceFromValues("x", []float64{1.0, 1.2, 0.9, ...})

// Several variables: one row per time step, each of length len(vars).
s, err := dataset.NewSequence([]string{"x", "y"}, [][]float64{
	{1.0, 4.5},
	{1.2, 4.4},
	// ...
})
```

## Online learning (and starting from scratch)

Sometimes you want the model to keep learning as new readings arrive, instead of
freezing after a single `Fit`. Enable that with `WithOnline`, then drive the
model with `Update` instead of `Step`. `Update` is a learning version of `Step`:
it refines the model from the latest reading (via **recursive least squares**,
the online counterpart of the ridge fit), records it, and returns the
one-step-ahead prediction for the next step.

```go
m := ngrc.New(ngrc.WithOnline(0.999)) // forgetting factor in (0,1]
m.Fit(context.Background(), train)     // optional warm start
for reading := range liveFeed {
	next, _ := m.Update(reading)        // learn, record, then predict
	use(next)
}
```

The **forgetting factor** controls how fast the model adapts: `1.0` weights all
history equally (best for fixed dynamics), while a value slightly below 1 (e.g.
`0.999`) discounts old data so the model can track dynamics that drift over time.

To skip training entirely and learn purely online, prime the model with **random
weights** using `PrimeRandom`, then feed it readings:

```go
m := ngrc.New(ngrc.WithOnline(0.999), ngrc.WithStandardize(false))
m.PrimeRandom([]string{"x"}, nil, 42) // random readout, raw space, seed 42
for reading := range liveFeed {
	next, _ := m.Update(reading)
	use(next)
}
```

`PrimeRandom`'s second argument is an optional warmup window (`*dataset.Sequence`).
Pass one to fit the standardizing scaler and seed the delay buffer from real data;
pass `nil` to run in raw space (standardization off) and let the buffer fill as
readings arrive. Standardization centers and scales each variable, which helps
when variables have large offsets or very different magnitudes; recursive least
squares is fairly scale-robust on its own, so raw space is fine when your
variables are already comparable in size.

## Tips and limits

- **Always split by time** (`SplitChrono`), never randomly. Testing on shuffled
  data would let the model "peek" at the future and look better than it is.
- **Autonomous forecasts drift.** One-step accuracy is the honest measure of
  quality; long autonomous rollouts of chaotic systems will eventually diverge
  from reality even when the model is excellent — that is the nature of chaos,
  not a bug.
- **Start with defaults**, then, if needed, raise `order` for more nonlinearity,
  add `taps` for longer memory, or raise `ridge` if results look unstable.
- v1 forecasts **all** variables of the series together. Predicting some
  variables from others (cross-prediction) is a planned addition.

## See also

- [training.md](training.md) — what R²/RMSE mean and how training works generally.
- [deployment.md](deployment.md) — reusing saved models in a Go program.
- [data-cleaning.md](data-cleaning.md) — getting your CSV into numeric shape.
