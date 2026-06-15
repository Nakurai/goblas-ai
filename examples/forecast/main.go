// Command forecast demonstrates Next-Generation Reservoir Computing end to end:
// load a time series, train, measure one-step accuracy on held-out data, save,
// reload, and autonomously forecast the future.
//
// Run it from the repository root:
//
//	go run ./examples/forecast
//
// The data is a Lorenz system — a classic chaotic dynamical system — which NG-RC
// can learn from a short trace and then forecast.
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
	const csvPath = "examples/data/lorenz.csv"

	// 1. Load the multivariate time series (columns x, y, z), in time order.
	seq, err := dataset.SequenceFromCSV(csvPath)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("loaded %d steps of %d variables: %v\n", seq.Len(), seq.Dim(), seq.Vars)

	// 2. Split chronologically: train on the past, test on the future.
	train, test := seq.SplitChrono(0.2)

	// 3. Train with defaults (2 taps, order 2 — captures the quadratic dynamics).
	m := ngrc.New()
	if err := m.Fit(context.Background(), train); err != nil {
		log.Fatal(err)
	}

	// 4. One-step-ahead accuracy on the held-out tail. After training, the model
	//    is primed with the end of the training data, which precedes the test set.
	var preds, truth []float64
	for i := 0; i < test.Len()-1; i++ {
		p, err := m.Step(test.Step(i))
		if err != nil {
			log.Fatal(err)
		}
		preds = append(preds, p...)
		truth = append(truth, test.Step(i+1)...)
	}
	fmt.Printf("one-step-ahead R2   = %.5f (1.0 is perfect)\n", metrics.R2(truth, preds))
	fmt.Printf("one-step-ahead RMSE = %.4f\n", metrics.RMSE(truth, preds))

	// 5. Save and reload (this is how a deployed program reuses the model).
	if err := m.SaveFile("examples/lorenz.gobl"); err != nil {
		log.Fatal(err)
	}
	loaded, err := ngrc.LoadFile("examples/lorenz.gobl")
	if err != nil {
		log.Fatal(err)
	}

	// 6. Autonomously forecast 10 steps into the future, feeding predictions back.
	future, err := loaded.Forecast(10)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("\nAutonomous 10-step forecast (x, y, z):")
	for i, state := range future {
		fmt.Printf("  +%2d: % .3f % .3f % .3f\n", i+1, state[0], state[1], state[2])
	}
}
