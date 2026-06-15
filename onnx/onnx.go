// Package onnx exports trained linear models to the ONNX format — an
// industry-standard, cross-platform file for machine-learning models. Exporting
// to ONNX lets a model trained with goblas-ai be run by other tools and
// languages (Python, C++, the ONNX Runtime, etc.), which is the usual path for
// deploying outside a Go program.
package onnx

import (
	"fmt"
	"io"
	"os"
)

// ONNX constants. Opset 13 (and the matching IR version 7) is broadly supported
// by ONNX runtimes.
const (
	irVersion    = 7
	opsetVersion = 13
	elemFloat    = 1 // TensorProto.DataType FLOAT
	attrFloat    = 1 // AttributeProto.AttributeType FLOAT
)

// ExportLinear writes a linear model y = X·coef + intercept to w as an ONNX
// model. coef must be in the original (unscaled) feature space, so the exported
// graph consumes raw features — any training-time standardization must already
// be folded into coef and intercept (linear.Regression.Coefficients does this).
//
// The graph is a single Gemm (general matrix multiply) node:
//
//	output = input · W + B
//
// with input shape [N, p], W shape [p, 1], and bias B shape [1].
func ExportLinear(w io.Writer, coef []float64, intercept float64, featureNames []string) error {
	p := len(coef)
	if p == 0 {
		return fmt.Errorf("onnx: model has no coefficients")
	}

	coef32 := make([]float32, p)
	for i, c := range coef {
		coef32[i] = float32(c)
	}

	model := buildModel(coef32, float32(intercept), p)
	if _, err := w.Write(model); err != nil {
		return fmt.Errorf("onnx: write model: %w", err)
	}
	return nil
}

// ExportLinearFile writes the ONNX model to the file at path.
func ExportLinearFile(path string, coef []float64, intercept float64, featureNames []string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("onnx: create file: %w", err)
	}
	if err := ExportLinear(f, coef, intercept, featureNames); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// buildModel assembles the full ModelProto bytes.
func buildModel(coef []float32, intercept float32, p int) []byte {
	var m pbuf
	m.int64Field(1, irVersion)                        // ir_version
	m.stringField(2, "goblas-ai")                     // producer_name
	m.messageField(7, buildGraph(coef, intercept, p)) // graph
	m.messageField(8, buildOpset())                   // opset_import
	return m.b
}

func buildOpset() []byte {
	var o pbuf
	o.stringField(1, "")          // domain (default ONNX domain)
	o.int64Field(2, opsetVersion) // version
	return o.b
}

func buildGraph(coef []float32, intercept float32, p int) []byte {
	var g pbuf
	g.messageField(1, buildGemmNode())                    // node
	g.stringField(2, "linear_regression")                 // name
	g.messageField(5, buildWeightTensor(coef, p))         // initializer: W
	g.messageField(5, buildBiasTensor(intercept))         // initializer: B
	g.messageField(11, buildValueInfo("input", int64(p))) // graph input
	g.messageField(12, buildValueInfo("output", 1))       // graph output
	return g.b
}

func buildGemmNode() []byte {
	var n pbuf
	n.stringField(1, "input") // input A
	n.stringField(1, "W")     // input B
	n.stringField(1, "B")     // input C (bias)
	n.stringField(2, "output")
	n.stringField(3, "gemm")
	n.stringField(4, "Gemm")
	n.messageField(5, buildFloatAttr("alpha", 1.0))
	n.messageField(5, buildFloatAttr("beta", 1.0))
	return n.b
}

func buildFloatAttr(name string, f float32) []byte {
	var a pbuf
	a.stringField(1, name)      // name
	a.floatField(2, f)          // f
	a.int64Field(20, attrFloat) // type = FLOAT
	return a.b
}

// buildWeightTensor builds the W initializer with shape [p, 1].
func buildWeightTensor(coef []float32, p int) []byte {
	var t pbuf
	t.packedInt64(1, []int64{int64(p), 1}) // dims
	t.int64Field(2, elemFloat)             // data_type
	t.packedFloats(4, coef)                // float_data
	t.stringField(8, "W")                  // name
	return t.b
}

// buildBiasTensor builds the B initializer with shape [1].
func buildBiasTensor(intercept float32) []byte {
	var t pbuf
	t.packedInt64(1, []int64{1})            // dims
	t.int64Field(2, elemFloat)              // data_type
	t.packedFloats(4, []float32{intercept}) // float_data
	t.stringField(8, "B")                   // name
	return t.b
}

// buildValueInfo describes a graph input/output: a float tensor of shape
// [N, lastDim], where N is the (dynamic) batch dimension.
func buildValueInfo(name string, lastDim int64) []byte {
	var vi pbuf
	vi.stringField(1, name)                      // name
	vi.messageField(2, buildTensorType(lastDim)) // type
	return vi.b
}

func buildTensorType(lastDim int64) []byte {
	var tt pbuf
	tt.messageField(1, buildTensorTypeInner(lastDim)) // tensor_type
	return tt.b
}

func buildTensorTypeInner(lastDim int64) []byte {
	var inner pbuf
	inner.int64Field(1, elemFloat)             // elem_type
	inner.messageField(2, buildShape(lastDim)) // shape
	return inner.b
}

func buildShape(lastDim int64) []byte {
	var s pbuf
	s.messageField(1, buildDimParam("N")) // dynamic batch dimension
	s.messageField(1, buildDimValue(lastDim))
	return s.b
}

func buildDimParam(name string) []byte {
	var d pbuf
	d.stringField(2, name) // dim_param
	return d.b
}

func buildDimValue(v int64) []byte {
	var d pbuf
	d.int64Field(1, v) // dim_value
	return d.b
}
