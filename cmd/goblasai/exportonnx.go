package main

import (
	"fmt"

	"github.com/nakurai/goblas-ai/linear"
)

func runExportONNX(args []string) error {
	if len(args) != 2 {
		return fmt.Errorf("usage: goblasai export-onnx <model.gobl> <out.onnx>")
	}
	in, out := args[0], args[1]
	m, err := linear.LoadFile(in)
	if err != nil {
		return err
	}
	if err := m.ExportONNX(out); err != nil {
		return err
	}
	fmt.Printf("exported %s to %s\n", in, out)
	return nil
}
