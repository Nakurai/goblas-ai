package linear

import (
	"fmt"

	"github.com/nakurai/goblas-ai/onnx"
)

// ExportONNX writes the trained model to path in the ONNX format, for deployment
// in non-Go runtimes. The exported graph consumes raw (unscaled) features: any
// standardization learned during training is folded into the exported
// coefficients, so consumers feed the model the same raw inputs you would pass
// to Predict.
func (r *Regression) ExportONNX(path string) error {
	if !r.fitted {
		return fmt.Errorf("linear: model is not trained")
	}
	coef, intercept := r.Coefficients()
	return onnx.ExportLinearFile(path, coef, intercept, r.features)
}
