package ngrc

// featureSpec describes how NG-RC turns a window of recent states into one
// feature vector. The recipe has two parts:
//
//  1. Linear part — the current state plus a few past states at a fixed spacing
//     ("time-delay embedding"). With k taps and stride s, it gathers states at
//     offsets 0, s, 2s, …, (k-1)s into the past, giving m = d*k numbers.
//  2. Nonlinear part — the unique polynomial products of those m numbers, up to
//     the chosen order (order 2 means every pair product, including squares),
//     plus an optional constant 1.
//
// The full feature vector has length M = [constant?] + m + (number of monomials).
type featureSpec struct {
	d         int // variables per state
	k         int // number of delay taps
	s         int // stride between taps
	order     int // highest polynomial order (>=1)
	constant  bool
	m         int     // d*k, length of the linear part
	monomials [][]int // each entry lists indices into the linear part to multiply
	mTotal    int     // M, total feature length
}

func newFeatureSpec(d, k, s, order int, constant bool) *featureSpec {
	if order < 1 {
		order = 1
	}
	m := d * k
	var monomials [][]int
	for deg := 2; deg <= order; deg++ {
		monomials = append(monomials, combosWithReplacement(m, deg)...)
	}
	total := m + len(monomials)
	if constant {
		total++
	}
	return &featureSpec{
		d:         d,
		k:         k,
		s:         s,
		order:     order,
		constant:  constant,
		m:         m,
		monomials: monomials,
		mTotal:    total,
	}
}

// warmup is how many initial steps are needed before a full delay embedding can
// be formed: the oldest tap reaches (k-1)*s steps into the past.
func (fs *featureSpec) warmup() int { return (fs.k - 1) * fs.s }

// build writes the full feature vector for one time step into dst (length
// mTotal), given the already-assembled linear part (length m).
func (fs *featureSpec) build(dst, lin []float64) {
	idx := 0
	if fs.constant {
		dst[0] = 1
		idx = 1
	}
	copy(dst[idx:idx+fs.m], lin)
	idx += fs.m
	for _, mono := range fs.monomials {
		prod := 1.0
		for _, ix := range mono {
			prod *= lin[ix]
		}
		dst[idx] = prod
		idx++
	}
}

// combosWithReplacement returns every non-decreasing tuple of length k drawn from
// indices [0, n). These enumerate the unique monomials of a given degree: e.g.
// for n=2, k=2 it yields [0 0], [0 1], [1 1] — the products x0·x0, x0·x1, x1·x1.
func combosWithReplacement(n, k int) [][]int {
	var res [][]int
	cur := make([]int, k)
	var rec func(start, depth int)
	rec = func(start, depth int) {
		if depth == k {
			c := make([]int, k)
			copy(c, cur)
			res = append(res, c)
			return
		}
		for i := start; i < n; i++ {
			cur[depth] = i
			rec(i, depth+1)
		}
	}
	rec(0, 0)
	return res
}
