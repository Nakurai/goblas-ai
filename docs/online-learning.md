# Online learning and resumable training

This guide covers two related abilities: updating a model continuously as new
data arrives (**online learning**), and stopping and resuming a long training run
without losing progress (**checkpointing**).

## What "online learning" means

The usual way to train, called *batch training*, is: gather all your data, train
once, get a model. **Online learning** is different — the model learns
**incrementally**, absorbing new data a little at a time, and improving as it
goes. You never have to retrain from scratch.

This is useful when:

- data arrives over time (a stream of orders, sensor readings, user events), and
- you want the model to stay current without periodically rebuilding it from the
  full history (which may be too large or too slow to reprocess).

Online learning is possible because of the SGD solver (see
[training.md](training.md)): SGD already learns by making small updates from
batches of rows, so feeding it new batches over time is a natural extension.

## The `PartialFit` method

Online learning is done with `PartialFit`, which applies a single update from one
batch of new data:

```go
lr := linear.NewRegression(
	linear.WithStandardize(false), // see the note below
)

ctx := context.Background()
for batch := range newData.Batches(64) {
	if err := lr.PartialFit(ctx, batch); err != nil {
		log.Fatal(err)
	}
}
// lr now reflects everything it has seen so far and can predict immediately.
price, _ := lr.Predict([]float64{2000, 3, 10})
```

Each call to `PartialFit` nudges the model using just that batch, then returns.
You can keep calling it forever as new data shows up.

### Important: standardization and online learning

Feature standardization (the automatic scaling described in
[data-cleaning.md](data-cleaning.md)) needs statistics computed over the whole
dataset — the average and spread of each feature. In a never-ending stream there
is no "whole dataset" up front, so for purely online use you have two clean
options:

1. **Turn standardization off** with `WithStandardize(false)`, and make sure your
   features are already on similar, modest scales. This is the simplest path.
2. **Bootstrap the scaling once** from an initial chunk of data with a normal
   `Fit`, then switch to `PartialFit` for ongoing updates. The bootstrapped
   scaling is then reused for every later update.

If you call `PartialFit` with standardization left on but no scaling has been
established, the model will simply train on the raw feature values.

## Bootstrapping an online application

A practical pattern for a long-running service:

```go
// 1. On first launch, if no saved model exists, train an initial one
//    from whatever history you have. This also establishes feature scaling.
lr := linear.NewRegression()
lr.Fit(ctx, initialHistory)
lr.SaveFile("model.gobl")

// 2. From then on, load the existing model at startup...
lr, _ := linear.LoadFile("model.gobl") // returns the trainable model

// 3. ...and update it as new data arrives.
for batch := range liveStream.Batches(64) {
	lr.PartialFit(ctx, batch)
}

// 4. Periodically persist, so a restart doesn't lose recent learning.
lr.SaveFile("model.gobl")
```

`linear.LoadFile` returns the full, trainable model (a `*linear.Regression`), so
you can keep teaching it. (If you only need to *predict*, use `model.LoadFile`
instead — see [deployment.md](deployment.md).)

## Maintaining an online model

- **Save regularly.** Treat the saved file as your source of truth and rewrite it
  on a schedule (say every few minutes, or every N batches), so an unexpected
  restart loses at most a little recent learning.
- **Watch for drift.** Over time the world can change in ways that make old
  patterns wrong. Keep evaluating the model on recent data (the same R²/RMSE from
  [training.md](training.md)); if scores slip, consider retraining a fresh model
  from recent history and swapping it in.
- **Keep raw data if you can.** Online learning does not require keeping all
  history, but having some recent raw data makes it possible to re-bootstrap or
  re-evaluate when needed.

## Resumable training (checkpointing)

Separately from online learning, a single long training run can be made
**interruptible**. Two mechanisms work together.

### Automatic checkpoints during training

`WithCheckpoint` tells the SGD solver to save the model to a file periodically —
every N updates — and also if training is cancelled:

```go
lr := linear.NewRegression(
	linear.WithSolver(linear.SGD),
	linear.WithEpochs(1000),
	linear.WithCheckpoint("checkpoint.gobl", 500), // save every 500 updates
)
lr.Fit(ctx, data)
```

The checkpoint is written safely (to a temporary file, then renamed), so an
interrupted save never leaves a corrupted file behind.

### Cancelling cleanly

Training honours Go's `context` cancellation. When the context is cancelled — for
example on a shutdown signal — `Fit` writes one final checkpoint and then returns
the cancellation error:

```go
ctx, cancel := context.WithCancel(context.Background())
// ... arrange for cancel() to be called on shutdown ...

err := lr.Fit(ctx, data)
if errors.Is(err, context.Canceled) {
	// training stopped early; checkpoint.gobl holds the latest progress
}
```

### Resuming

To continue, load the checkpoint and call `Fit` again. Because all of the
training state (coefficients, scaling, batch size, learning rate) is saved,
resuming continues exactly where it stopped:

```go
lr, _ := linear.LoadFile("checkpoint.gobl")
lr.Fit(ctx, data) // continues from the saved state
```

## Next step

When your model is ready to use in production, see
[deployment.md](deployment.md).
