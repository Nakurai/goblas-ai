package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/nakurai/goblas-ai/ngrc"
)

func runForecast(args []string) error {
	fs := flag.NewFlagSet("forecast", flag.ExitOnError)
	modelPath := fs.String("model", "", "path to a trained ng_reservoir model (required)")
	steps := fs.Int("steps", 100, "number of future steps to simulate")
	out := fs.String("out", "", "path to write the forecast CSV (default: stdout)")
	fs.Parse(args)

	if *modelPath == "" {
		return fmt.Errorf("--model is required")
	}
	if *steps <= 0 {
		return fmt.Errorf("--steps must be positive")
	}

	m, err := ngrc.LoadFile(*modelPath)
	if err != nil {
		return err
	}
	preds, err := m.Forecast(*steps)
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
	cw.Write(m.Vars())
	for _, state := range preds {
		rec := make([]string, len(state))
		for j, v := range state {
			rec[j] = strconv.FormatFloat(v, 'g', -1, 64)
		}
		cw.Write(rec)
	}
	cw.Flush()
	if *out != "" {
		fmt.Fprintf(os.Stderr, "wrote %d forecast steps to %s\n", *steps, *out)
	}
	return nil
}
