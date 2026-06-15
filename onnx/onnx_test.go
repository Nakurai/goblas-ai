package onnx

import (
	"bytes"
	"testing"
)

func TestExportLinearProducesModel(t *testing.T) {
	var buf bytes.Buffer
	coef := []float64{2.0, -3.0, 0.5}
	if err := ExportLinear(&buf, coef, 5.0, []string{"a", "b", "c"}); err != nil {
		t.Fatalf("export: %v", err)
	}
	out := buf.Bytes()
	if len(out) == 0 {
		t.Fatal("export produced no bytes")
	}
	// The first field is ir_version (field 1, varint): tag byte 0x08.
	if out[0] != 0x08 {
		t.Errorf("unexpected first byte %#x, want 0x08 (ir_version tag)", out[0])
	}
	// Producer name must appear in the stream.
	if !bytes.Contains(out, []byte("goblas-ai")) {
		t.Error("producer name not found in model bytes")
	}
	if !bytes.Contains(out, []byte("Gemm")) {
		t.Error("Gemm op_type not found in model bytes")
	}
}

func TestExportLinearRejectsEmpty(t *testing.T) {
	var buf bytes.Buffer
	if err := ExportLinear(&buf, nil, 0, nil); err == nil {
		t.Fatal("expected error for empty coefficients")
	}
}
