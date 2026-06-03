package tabgo

// Shift returns a new Series where values are shifted by the given number of periods.
// Positive periods shift values down (introducing nil at the top);
// negative periods shift values up (introducing nil at the bottom).
func (s *Series) Shift(periods int) *Series {
	n := len(s.values)
	out := make([]any, n)
	for i := 0; i < n; i++ {
		src := i - periods
		if src >= 0 && src < n {
			out[i] = s.values[src]
		} else {
			out[i] = nil
		}
	}
	return &Series{name: s.name, values: out}
}
