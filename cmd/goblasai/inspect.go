package main

import (
	"encoding/json"
	"fmt"

	"github.com/nakurai/goblas-ai/model"
)

func runInspect(args []string) error {
	if len(args) != 1 {
		return fmt.Errorf("usage: goblasai inspect <model.gobl>")
	}
	path := args[0]
	meta, err := model.ReadMetadataFile(path)
	if err != nil {
		return err
	}

	fmt.Printf("model:        %s\n", path)
	fmt.Printf("algorithm:    %s\n", meta.Algorithm)
	fmt.Printf("created:      %s\n", meta.CreatedAt)
	fmt.Printf("file format:  v%d\n", meta.FormatVersion)
	fmt.Println("parameters:")
	pretty, _ := json.MarshalIndent(meta.Params, "  ", "  ")
	fmt.Printf("  %s\n", pretty)

	// Confirm the model actually loads (validates the weights, not just the
	// header). LoadModelFile works for any algorithm, including the stateful
	// NG-RC model that is not a row-by-row Predictor.
	if _, err := model.LoadModelFile(path); err != nil {
		return fmt.Errorf("model header is readable but weights failed to load: %w", err)
	}
	return nil
}
