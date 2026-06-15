// Command goblasai is a small command-line front end to the goblas-ai library.
// It covers the whole workflow — train a model, make predictions, inspect a
// saved model, and export to ONNX — so the library can be used without writing
// any Go code.
//
// Usage:
//
//	goblasai train      --data train.csv --target price --out model.gobl
//	goblasai predict    --model model.gobl --data new.csv --out preds.csv
//	goblasai inspect    model.gobl
//	goblasai export-onnx model.gobl model.onnx
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "train":
		err = runTrain(os.Args[2:])
	case "predict":
		err = runPredict(os.Args[2:])
	case "forecast":
		err = runForecast(os.Args[2:])
	case "inspect":
		err = runInspect(os.Args[2:])
	case "export-onnx":
		err = runExportONNX(os.Args[2:])
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `goblasai — pure-Go machine learning from the command line

Commands:
  train        Train a model from a CSV file (linear_regression or ng_reservoir)
  predict      Predict targets for a CSV using a trained model
  forecast     Simulate future steps of a time series with an ng_reservoir model
  inspect      Print a saved model's details
  export-onnx  Export a saved model to the ONNX format

Run "goblasai <command> -h" for command-specific flags.
`)
}
