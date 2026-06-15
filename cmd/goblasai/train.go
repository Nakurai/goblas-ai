package main

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/linear"
	"github.com/nakurai/goblas-ai/metrics"
	"github.com/nakurai/goblas-ai/model"
	"github.com/nakurai/goblas-ai/ngrc"
)

func runTrain(args []string) error {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	data := fs.String("data", "", "path to the training CSV (required)")
	target := fs.String("target", "", "linear_regression: name of the target column (required)")
	out := fs.String("out", "model.gobl", "path to write the trained model")
	algo := fs.String("algo", "linear_regression", "algorithm: linear_regression or ng_reservoir")
	solverName := fs.String("solver", "auto", "linear_regression: auto, closed_form, or sgd")
	standardize := fs.Bool("standardize", true, "scale features/variables before training")
	intercept := fs.Bool("intercept", true, "linear_regression: fit an overall offset term")
	epochs := fs.Int("epochs", 10, "linear_regression SGD: number of passes over the data")
	lr := fs.Float64("lr", 0.01, "linear_regression SGD: learning rate")
	l2 := fs.Float64("l2", 0, "linear_regression SGD: L2 regularization strength")
	testFrac := fs.Float64("test-frac", 0.2, "fraction held out to report test metrics (0 to disable)")
	seed := fs.Int64("seed", 1, "linear_regression: random seed for the train/test split")
	// ng_reservoir flags
	columns := fs.String("columns", "", "ng_reservoir: comma-separated series columns (default: all)")
	taps := fs.Int("taps", 2, "ng_reservoir: number of delay taps")
	stride := fs.Int("stride", 1, "ng_reservoir: stride between taps")
	order := fs.Int("order", 2, "ng_reservoir: polynomial order")
	ridge := fs.Float64("ridge", 1e-6, "ng_reservoir: ridge regularization strength")
	delta := fs.Bool("delta", true, "ng_reservoir: predict the change between steps")
	fs.Parse(args)

	if *data == "" {
		return fmt.Errorf("--data is required")
	}

	switch *algo {
	case "ng_reservoir":
		return trainNGRC(*data, *out, *columns, *taps, *stride, *order, *ridge, *delta, *standardize, *testFrac)
	case "linear_regression":
		// handled below
	default:
		return fmt.Errorf("unknown algorithm %q (use linear_regression or ng_reservoir)", *algo)
	}

	if *target == "" {
		return fmt.Errorf("--target is required for linear_regression")
	}

	solver, err := parseSolver(*solverName)
	if err != nil {
		return err
	}

	var trainData, testData dataset.Provider
	if *testFrac > 0 {
		trainData, testData, err = dataset.SplitCSV(*data, *target, *testFrac, *seed)
	} else {
		trainData, err = dataset.OpenCSV(*data, *target)
	}
	if err != nil {
		return err
	}

	lrModel := linear.NewRegression(
		linear.WithSolver(solver),
		linear.WithStandardize(*standardize),
		linear.WithIntercept(*intercept),
		linear.WithEpochs(*epochs),
		linear.WithLearningRate(*lr),
		linear.WithL2(*l2),
	)
	fmt.Printf("training %s (solver=%s) on %s ...\n", *algo, *solverName, *data)
	if err := lrModel.Fit(context.Background(), trainData); err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	if err := lrModel.SaveFile(*out); err != nil {
		return err
	}
	fmt.Printf("saved model to %s\n", *out)

	if testData != nil {
		preds, ys, err := evaluate(lrModel, testData)
		if err != nil {
			return err
		}
		if len(ys) > 0 {
			fmt.Printf("test metrics on %d held-out rows:\n", len(ys))
			fmt.Printf("  R2   = %.4f  (1.0 is perfect, 0 means no better than the average)\n", metrics.R2(ys, preds))
			fmt.Printf("  RMSE = %.4f  (typical error, in the target's units)\n", metrics.RMSE(ys, preds))
			fmt.Printf("  MAE  = %.4f  (average absolute error)\n", metrics.MAE(ys, preds))
		}
	}
	return nil
}

func parseSolver(name string) (linear.Solver, error) {
	switch name {
	case "auto", "":
		return linear.Auto, nil
	case "closed_form":
		return linear.ClosedForm, nil
	case "sgd":
		return linear.SGD, nil
	default:
		return linear.Auto, fmt.Errorf("unknown solver %q", name)
	}
}

// trainNGRC trains a Next-Generation Reservoir Computing model on a time series.
func trainNGRC(data, out, columns string, taps, stride, order int, ridge float64, delta, standardize bool, testFrac float64) error {
	var cols []string
	if columns != "" {
		cols = strings.Split(columns, ",")
	}
	seq, err := dataset.SequenceFromCSV(data, cols...)
	if err != nil {
		return err
	}

	trainSeq := seq
	var testSeq *dataset.Sequence
	if testFrac > 0 {
		trainSeq, testSeq = seq.SplitChrono(testFrac)
	}

	m := ngrc.New(
		ngrc.WithTaps(taps),
		ngrc.WithStride(stride),
		ngrc.WithOrder(order),
		ngrc.WithRidge(ridge),
		ngrc.WithPredictDelta(delta),
		ngrc.WithStandardize(standardize),
	)
	fmt.Printf("training ng_reservoir (taps=%d stride=%d order=%d) on %s ...\n", taps, stride, order, data)
	if err := m.Fit(context.Background(), trainSeq); err != nil {
		return fmt.Errorf("training failed: %w", err)
	}
	if err := m.SaveFile(out); err != nil {
		return err
	}
	fmt.Printf("saved model to %s\n", out)

	if testSeq != nil && testSeq.Len() > 1 {
		// One-step-ahead over the held-out tail. The model is primed with the end
		// of training, which directly precedes the test segment.
		var preds, truth []float64
		for i := 0; i < testSeq.Len()-1; i++ {
			p, err := m.Step(testSeq.Step(i))
			if err != nil {
				return err
			}
			preds = append(preds, p...)
			truth = append(truth, testSeq.Step(i+1)...)
		}
		fmt.Printf("one-step-ahead metrics on %d held-out steps:\n", testSeq.Len()-1)
		fmt.Printf("  R2   = %.4f  (1.0 is perfect, 0 means no better than the average)\n", metrics.R2(truth, preds))
		fmt.Printf("  RMSE = %.4f  (typical error, in the variables' units)\n", metrics.RMSE(truth, preds))
		fmt.Printf("  MAE  = %.4f  (average absolute error)\n", metrics.MAE(truth, preds))
	}
	return nil
}

// evaluate streams a Provider through a trained model, returning predictions and
// true targets for metric computation.
func evaluate(p model.Predictor, data dataset.Provider) (preds, ys []float64, err error) {
	for batch, berr := range data.Batches(8192) {
		if berr != nil {
			return nil, nil, berr
		}
		bp, perr := p.PredictBatch(batch.X)
		if perr != nil {
			return nil, nil, perr
		}
		preds = append(preds, bp...)
		ys = append(ys, batch.Y...)
	}
	return preds, ys, nil
}
