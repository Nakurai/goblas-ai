package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/nakurai/goblas-ai/dataset"
	"github.com/nakurai/goblas-ai/model"
)

func runPredict(args []string) error {
	fs := flag.NewFlagSet("predict", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to the trained model (required)")
	data := fs.String("data", "", "path to the input CSV (required)")
	out := fs.String("out", "", "path to write predictions CSV (default: stdout)")
	fs.Parse(args)

	if *modelPath == "" || *data == "" {
		return fmt.Errorf("both --model and --data are required")
	}

	// The model file records the feature columns and their order; read them so
	// the input CSV columns are matched up correctly.
	meta, err := model.ReadMetadataFile(*modelPath)
	if err != nil {
		return err
	}
	features, err := featureNames(meta)
	if err != nil {
		return err
	}

	predictor, err := model.LoadFile(*modelPath)
	if err != nil {
		return err
	}
	x, err := dataset.ReadMatrix(*data, features)
	if err != nil {
		return err
	}
	preds, err := predictor.PredictBatch(x)
	if err != nil {
		return err
	}

	w := os.Stdout
	if *out != "" {
		f, err := os.Create(*out)
		if err != nil {
			return err
		}
		defer f.Close()
		w = f
	}
	bw := bufio.NewWriter(w)
	defer bw.Flush()
	cw := csv.NewWriter(bw)
	cw.Write([]string{"prediction"})
	for _, p := range preds {
		cw.Write([]string{strconv.FormatFloat(p, 'g', -1, 64)})
	}
	cw.Flush()
	if *out != "" {
		fmt.Fprintf(os.Stderr, "wrote %d predictions to %s\n", len(preds), *out)
	}
	return nil
}

// featureNames extracts the ordered feature column names from model metadata.
func featureNames(meta model.Metadata) ([]string, error) {
	raw, ok := meta.Params["feature_names"]
	if !ok {
		return nil, fmt.Errorf("model metadata has no feature_names")
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("model metadata feature_names has unexpected type %T", raw)
	}
	names := make([]string, len(list))
	for i, v := range list {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("feature name %d is not a string", i)
		}
		names[i] = s
	}
	return names, nil
}
