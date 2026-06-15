# Deploying and reusing a trained model

Training produces a file. This guide explains how to *use* that file — both
inside a Go program and from other tools — and what makes reuse safe by default.

## Two files, two purposes

goblas-ai can save a trained model in two formats, and they serve different
goals:

- **The native `.gobl` file** — for reuse *inside Go programs*. It is
  self-contained and pure-Go to load. Use it for checkpoints, online learning,
  and running the model in your own Go service.
- **The ONNX `.onnx` file** — for reuse *outside Go*. ONNX is an industry-standard
  model format that many tools and languages (Python, C++, the ONNX Runtime, and
  others) can read. Use it when something that isn't a Go program needs to run
  your model.

## Reusing a model in Go

This is the common case and it is deliberately minimal. A program that only needs
to make predictions imports two things: the `model` package, and the algorithm's
package as a **blank import** (the leading `_`). The blank import does nothing
visible except register the algorithm so the loader knows how to rebuild it.

```go
package main

import (
	"fmt"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/model"
	_ "github.com/nakurai/goblas-ai/linear" // registers "linear_regression"
)

func main() {
	m, err := model.LoadFile("price.gobl")
	if err != nil {
		panic(err)
	}

	// Predict for one house: raw features, in the order they were trained.
	price, _ := m.Predict([]float64{2000, 3, 10})
	fmt.Println("predicted price:", price)

	// Or predict for many rows at once.
	rows := dataset.Matrix{Rows: 2, Cols: 3, Data: []float64{
		2000, 3, 10,
		1500, 2, 25,
	}}
	prices, _ := m.PredictBatch(rows)
	fmt.Println(prices)
}
```

`model.LoadFile` returns a `model.Predictor` — an object that can only make
predictions. That is intentional: deployment code shouldn't be able to
accidentally pull in training or data-loading machinery, so the inference path
stays small.

### You pass raw features — always

A frequent and painful bug in machine learning is *train/serve skew*: the model
was trained on adjusted numbers (for example standardized features), but at
prediction time it is fed the raw numbers, so it silently produces wrong answers.

goblas-ai closes this trap by storing any feature scaling **inside** the model
file and reapplying it automatically inside `Predict`. You pass the same raw
feature values you would put in a spreadsheet, both when training and when
predicting. There is nothing to reproduce by hand.

### If you also need to keep training

`model.LoadFile` gives you a predict-only object. If instead you want to continue
training (resume a checkpoint, or do online learning), load it as the concrete
type:

```go
lr, _ := linear.LoadFile("price.gobl") // *linear.Regression, still trainable
lr.PartialFit(ctx, batch)
```

You can also reach the learned coefficients for inspection or auditing:

```go
coef, intercept := lr.Coefficients() // in the original feature units
```

## Predicting from the command line

If you'd rather not write Go, the `goblasai` tool predicts straight from a CSV.
It reads the feature column names from the model file and matches them against the
input CSV's header, so column order in the input doesn't matter.

```sh
goblasai predict --model price.gobl --data new_houses.csv --out predictions.csv
```

`predictions.csv` will contain a single `prediction` column, one row per input
row.

## Inspecting a saved model

To see what a file contains — algorithm, when it was made, its settings, and its
coefficients — without writing code:

```sh
goblasai inspect price.gobl
```

This is handy for verifying you're shipping the model you think you are.

## Exporting to ONNX (using the model outside Go)

```go
lr.ExportONNX("price.onnx")
```

or from the command line:

```sh
goblasai export-onnx price.gobl price.onnx
```

The exported ONNX model takes **raw features** as input, exactly like the Go
`Predict` call: the feature scaling learned during training is folded into the
exported numbers, so consumers don't need to know it ever happened. The model's
input is named `input` (a row of features) and its output is named `output` (the
prediction).

Any ONNX-compatible runtime can then load `price.onnx` and run predictions. This
is the bridge for deploying a model trained in Go into, say, a Python service or
an edge runtime that speaks ONNX.

## Choosing between native and ONNX

- Running predictions **in a Go program**? Use the native `.gobl` file with
  `model.LoadFile`. It's the simplest and keeps everything pure Go.
- Need the model in **another language or an ONNX runtime**? Export to `.onnx`.
- Doing **online learning or resuming training**? Use the native file with
  `linear.LoadFile`. (ONNX stores a frozen model for prediction only — it is not
  meant for continued training.)

## Where to go next

- New to the workflow? Start at [data-cleaning.md](data-cleaning.md).
- Want to understand the training options? See [training.md](training.md).
- Updating models over time? See [online-learning.md](online-learning.md).
