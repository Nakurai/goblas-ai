// Command example demonstrates a complete goblas-ai workflow end to end:
// load data, split it, train a linear-regression model, evaluate it on held-out
// data, save it, reload it, predict, and export to ONNX.
//
// Run it from the repository root:
//
//	go run ./examples
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/linear"
	"github.com/nakurai/goblas-ai/metrics"
	"github.com/nakurai/goblas-ai/model"
)

func main() {
	const csvPath = "examples/data/housing.csv"
	const target = "price"

	// 1. Split the file into training and test sets without loading it all into
	//    memory. 20% of rows are held out to measure how well the model
	//    generalizes to data it did not learn from.
	train, test, err := dataset.SplitCSV(csvPath, target, 0.2, 1)
	if err != nil {
		log.Fatal(err)
	}

	// 2. Train. Defaults are sensible: features are standardized and an
	//    intercept is fitted; the solver is chosen automatically.
	lr := linear.NewRegression()
	if err := lr.Fit(context.Background(), train); err != nil {
		log.Fatal(err)
	}

	// 3. Evaluate on the held-out test set.
	var preds, truth []float64
	for batch, err := range test.Batches(4096) {
		if err != nil {
			log.Fatal(err)
		}
		bp, _ := lr.PredictBatch(batch.X)
		preds = append(preds, bp...)
		truth = append(truth, batch.Y...)
	}
	fmt.Printf("Test R2   = %.4f (1.0 is perfect)\n", metrics.R2(truth, preds))
	fmt.Printf("Test RMSE = %.0f (typical error, in price units)\n", metrics.RMSE(truth, preds))

	// 4. Inspect what the model learned, in the original feature units.
	coef, intercept := lr.Coefficients()
	fmt.Println("\nLearned relationship (price ≈ ...):")
	for i, name := range lr.FeatureNames() {
		fmt.Printf("  %+0.2f per unit of %s\n", coef[i], name)
	}
	fmt.Printf("  %+0.2f base price (intercept)\n", intercept)

	// 5. Save and reload — this is exactly how a deployed program reuses a model.
	if err := lr.SaveFile("examples/housing.gobl"); err != nil {
		log.Fatal(err)
	}
	loaded, err := model.LoadFile("examples/housing.gobl")
	if err != nil {
		log.Fatal(err)
	}

	// 6. Predict for a new house: 2000 sqft, 3 bedrooms, 10 years old.
	price, err := loaded.Predict([]float64{2000, 3, 10})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nPredicted price for [2000 sqft, 3 beds, 10 yrs]: %.0f\n", price)

	// 7. Export to ONNX for use outside Go.
	if err := lr.ExportONNX("examples/housing.onnx"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Exported examples/housing.onnx")
}
