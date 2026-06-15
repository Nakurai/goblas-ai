package ngrc

import (
	"reflect"
	"testing"
)

func TestFeatureSpecDims(t *testing.T) {
	// d=1, k=2, order=2, constant: m = 2; quadratic monomials = [00],[01],[11] = 3;
	// M = 1 (const) + 2 (linear) + 3 (quadratic) = 6.
	fs := newFeatureSpec(1, 2, 1, 2, true)
	if fs.m != 2 {
		t.Errorf("m = %d, want 2", fs.m)
	}
	if fs.mTotal != 6 {
		t.Errorf("M = %d, want 6", fs.mTotal)
	}
	if fs.warmup() != 1 {
		t.Errorf("warmup = %d, want 1", fs.warmup())
	}
	want := [][]int{{0, 0}, {0, 1}, {1, 1}}
	if !reflect.DeepEqual(fs.monomials, want) {
		t.Errorf("monomials = %v, want %v", fs.monomials, want)
	}
}

func TestFeatureSpecLinearOnly(t *testing.T) {
	// order 1 means no monomials.
	fs := newFeatureSpec(2, 3, 1, 1, false)
	if len(fs.monomials) != 0 {
		t.Errorf("expected no monomials, got %v", fs.monomials)
	}
	if fs.mTotal != 6 { // m = d*k = 6, no constant
		t.Errorf("M = %d, want 6", fs.mTotal)
	}
}

func TestBuild(t *testing.T) {
	fs := newFeatureSpec(1, 2, 1, 2, true)
	lin := []float64{2, 3} // x(t)=2, x(t-1)=3
	dst := make([]float64, fs.mTotal)
	fs.build(dst, lin)
	// [const, lin0, lin1, lin0*lin0, lin0*lin1, lin1*lin1]
	want := []float64{1, 2, 3, 4, 6, 9}
	if !reflect.DeepEqual(dst, want) {
		t.Errorf("build = %v, want %v", dst, want)
	}
}

func TestStrideWarmup(t *testing.T) {
	fs := newFeatureSpec(1, 3, 2, 1, false) // 3 taps, stride 2
	if fs.warmup() != 4 {                   // (3-1)*2
		t.Errorf("warmup = %d, want 4", fs.warmup())
	}
}
