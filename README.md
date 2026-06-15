# goblas-ai

**Machine learning you can build directly into a Go program — no Python, no
external services, no C libraries.**

goblas-ai is a pure-Go machine-learning library designed to run on ordinary
computers (a laptop, a small server, a container) rather than a data centre. It
is built on top of [goblas](https://github.com/nakurai/goblas), a pure-Go
implementation of the standard linear-algebra routines, so it compiles into a
single self-contained binary with no CGo and no system dependencies.

It is written for **Go developers, not data scientists**. Every concept is
explained in plain language in the [docs](docs/), the APIs have sensible
defaults, and the goal throughout is to make the right thing the easy thing.

> **Algorithms available today:**
> - **Linear regression** — predict a number from other numbers. End to end: data
>   loading, training, evaluation, saving, prediction, and ONNX export.
> - **Next-Generation Reservoir Computing (NG-RC)** — forecast time series
>   (sensor traces, prices, physical systems) accurately from short histories.
>
> The architecture is built so that more algorithms slot in behind the same
> interfaces.

**AI Disclaimer**: Since it looks like this is a polarizing topic, let's make it clear that this entire repo has been generated. Use with caution, at your own risks.

## What it's good for

- **Predicting a number** from other numbers: house prices from size and
  location, delivery time from distance and load, demand from price and season.
  This is called *regression*, and linear regression is the right first tool for
  it. → [docs/training.md](docs/training.md)
- **Forecasting a time series**: predicting future values of something that
  evolves over time, from its recent past — and even simulating its trajectory
  forward. This is what NG-RC does. → [docs/forecasting.md](docs/forecasting.md)
- **Embedding a model inside a Go service** so predictions happen in-process,
  with no network call to a separate ML system.

## Strengths

- **Pure Go.** Compiles to one static binary. No Python runtime, no CGo, no
  native libraries to install.
- **Runs on modest hardware.** Designed for CPUs and limited memory.
- **Handles data bigger than memory.** Input files are *streamed* — read a piece
  at a time — so you can train on files larger than your RAM.
- **Interruptible and resumable.** Long training runs can save checkpoints and
  pick up where they left off.
- **Learns continuously.** Supports *online learning*: updating a model as new
  data arrives, without retraining from scratch.
- **Deploy-ready.** Save models in a native format for reuse in Go, or export to
  **ONNX**, an industry-standard format other tools and languages can read.
- **Simple defaults.** `linear.NewRegression()` already does the sensible thing.

## Limitations (honestly)

- **CPU-focused.** It is not meant for training very large deep neural networks
  or models that need internet-scale data and GPU clusters.
- **Two algorithms so far** — linear regression and NG-RC. More are planned.
- **Numbers in, numbers out.** Inputs must be numeric columns. Text and
  categorical data must be converted to numbers first (see
  [docs/data-cleaning.md](docs/data-cleaning.md)).
- **NG-RC forecasts all series variables together** (cross-prediction — inferring
  some variables from others — is a planned addition), and long autonomous
  forecasts of chaotic systems will eventually drift (the nature of chaos).
- **Not a research replacement.** For exploratory data science, Python's
  ecosystem is still richer. goblas-ai is about *shipping* models inside Go apps.

## Install

```sh
go get github.com/nakurai/goblas-ai
```

## Quick start (library)

```go
package main

import (
	"context"
	"fmt"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/linear"
	"github.com/nakurai/goblas-ai/metrics"
	"github.com/nakurai/goblas-ai/model"
)

func main() {
	// Split a CSV into a training set and a held-out test set (20%).
	train, test, _ := dataset.SplitCSV("housing.csv", "price", 0.2, 1)

	// Train with sensible defaults.
	lr := linear.NewRegression()
	lr.Fit(context.Background(), train)

	// Measure quality on data the model never saw.
	var preds, truth []float64
	for b := range test.Batches(4096) {
		p, _ := lr.PredictBatch(b.X)
		preds = append(preds, p...)
		truth = append(truth, b.Y...)
	}
	fmt.Println("R2:", metrics.R2(truth, preds))

	// Save it, then reuse it later (this is what a deployed app does).
	lr.SaveFile("price.gobl")
	reloaded, _ := model.LoadFile("price.gobl")
	price, _ := reloaded.Predict([]float64{2000, 3, 10}) // raw features
	fmt.Println("predicted price:", price)
}
```

A complete, runnable version is in [examples/main.go](examples/main.go):

```sh
go run ./examples
```

## Quick start (forecasting a time series)

```go
seq, _ := dataset.SequenceFromCSV("lorenz.csv") // columns in time order
train, test := seq.SplitChrono(0.2)             // split by TIME, never shuffled

m := ngrc.New()                                 // sensible defaults
m.Fit(context.Background(), train)

// One-step-ahead on the held-out future:
next, _ := m.Step(test.Step(0))                 // predict the step after test[0]

// Or simulate the future autonomously:
future, _ := m.Forecast(200)                    // 200 predicted states

m.SaveFile("lorenz.gobl")                        // reuse later via ngrc.LoadFile
```

Runnable version: [examples/forecast/main.go](examples/forecast/main.go) — `go run ./examples/forecast`.

## Quick start (command line)

The `goblasai` tool does the whole workflow without writing Go:

```sh
go install github.com/nakurai/goblas-ai/cmd/goblasai@latest

goblasai train   --data housing.csv --target price --out price.gobl
goblasai inspect price.gobl
goblasai predict --model price.gobl --data new_houses.csv --out predictions.csv
goblasai export-onnx price.gobl price.onnx

# Time-series forecasting with NG-RC:
goblasai train    --algo ng_reservoir --data lorenz.csv --out lorenz.gobl
goblasai forecast --model lorenz.gobl --steps 200 --out future.csv
```

## Features and where to learn more

| Feature | What it does | Guide |
|---|---|---|
| Streaming CSV + in-memory data | Load data from files of any size, or from memory | [docs/data-cleaning.md](docs/data-cleaning.md) |
| Linear regression (two solvers) | Exact closed-form fit, or iterative SGD for big/streaming data | [docs/training.md](docs/training.md) |
| Time-series forecasting (NG-RC) | One-step-ahead and autonomous multi-step forecasting | [docs/forecasting.md](docs/forecasting.md) |
| Feature scaling | Automatic, stored with the model, applied at prediction time | [docs/data-cleaning.md](docs/data-cleaning.md) |
| Evaluation metrics | R², RMSE, MAE — explained in plain terms | [docs/training.md](docs/training.md) |
| Online learning & checkpoints | Update models continuously; resume interrupted training | [docs/online-learning.md](docs/online-learning.md) |
| Native save/load | Reuse trained models in any Go program | [docs/deployment.md](docs/deployment.md) |
| ONNX export | Run your (linear) model in other languages and tools | [docs/deployment.md](docs/deployment.md) |

## Project layout

```
dataset/      load data: streaming CSV, in-memory Frame, ordered Sequence, splits
preprocess/   feature scaling (StandardScaler)
linear/       linear regression (closed-form + SGD solvers)
ngrc/         Next-Generation Reservoir Computing (time-series forecasting)
metrics/      R², RMSE, MAE
model/        shared interfaces, save/load, the model registry
onnx/         ONNX export
cmd/goblasai/ the command-line tool
examples/     complete, runnable workflows (tabular and forecasting)
docs/         plain-language guides
```

## License

See repository.
