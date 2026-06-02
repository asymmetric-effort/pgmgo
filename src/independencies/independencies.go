package independencies

import "strings"

// Independencies represents a collection of IndependenceAssertion values.
type Independencies struct {
	assertions []*IndependenceAssertion
}

// NewIndependencies creates a new empty Independencies collection.
func NewIndependencies() *Independencies {
	return &Independencies{
		assertions: make([]*IndependenceAssertion, 0),
	}
}

// Add appends one or more assertions to the collection, skipping duplicates.
func (ind *Independencies) Add(assertions ...*IndependenceAssertion) {
	for _, a := range assertions {
		if a == nil {
			continue
		}
		if !ind.Contains(a) {
			ind.assertions = append(ind.assertions, a)
		}
	}
}

// Remove removes an assertion from the collection (by equality).
func (ind *Independencies) Remove(assertion *IndependenceAssertion) {
	if assertion == nil {
		return
	}
	for i, a := range ind.assertions {
		if a.Equals(assertion) {
			ind.assertions = append(ind.assertions[:i], ind.assertions[i+1:]...)
			return
		}
	}
}

// Contains returns true if the collection contains an assertion equal to the given one.
func (ind *Independencies) Contains(assertion *IndependenceAssertion) bool {
	if assertion == nil {
		return false
	}
	for _, a := range ind.assertions {
		if a.Equals(assertion) {
			return true
		}
	}
	return false
}

// GetAssertions returns a copy of the assertion slice.
func (ind *Independencies) GetAssertions() []*IndependenceAssertion {
	result := make([]*IndependenceAssertion, len(ind.assertions))
	copy(result, ind.assertions)
	return result
}

// Len returns the number of assertions in the collection.
func (ind *Independencies) Len() int {
	return len(ind.assertions)
}

// IsEquivalent returns true if this collection contains the same set of assertions
// as other (order independent).
func (ind *Independencies) IsEquivalent(other *Independencies) bool {
	if other == nil {
		return false
	}
	if ind.Len() != other.Len() {
		return false
	}
	for _, a := range ind.assertions {
		if !other.Contains(a) {
			return false
		}
	}
	return true
}

// String returns a human-readable representation of all assertions in the collection.
func (ind *Independencies) String() string {
	if len(ind.assertions) == 0 {
		return "{}"
	}
	parts := make([]string, len(ind.assertions))
	for i, a := range ind.assertions {
		parts[i] = a.String()
	}
	return "{\n  " + strings.Join(parts, ",\n  ") + "\n}"
}
