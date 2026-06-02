package independencies

import (
	"fmt"
	"sort"
	"strings"
)

// IndependenceAssertion represents a conditional independence assertion X ⊥ Y | Z,
// where X and Y are sets of variables and Z is the conditioning set (which may be empty).
type IndependenceAssertion struct {
	event1 []string
	event2 []string
	given  []string
}

// NewIndependenceAssertion creates a new IndependenceAssertion with the given event sets
// and conditioning set. Each input slice is copied and sorted internally.
func NewIndependenceAssertion(event1, event2, given []string) *IndependenceAssertion {
	return &IndependenceAssertion{
		event1: copyAndSort(event1),
		event2: copyAndSort(event2),
		given:  copyAndSort(given),
	}
}

// Event1 returns a copy of the first event set (X).
func (a *IndependenceAssertion) Event1() []string {
	return copySlice(a.event1)
}

// Event2 returns a copy of the second event set (Y).
func (a *IndependenceAssertion) Event2() []string {
	return copySlice(a.event2)
}

// Given returns a copy of the conditioning set (Z).
func (a *IndependenceAssertion) Given() []string {
	return copySlice(a.given)
}

// Equals returns true if this assertion is set-equal to other.
// Two assertions are equal if they have the same Event1, Event2, and Given sets
// (order independent), considering that X ⊥ Y | Z is the same as Y ⊥ X | Z.
func (a *IndependenceAssertion) Equals(other *IndependenceAssertion) bool {
	if other == nil {
		return false
	}
	if !setsEqual(a.given, other.given) {
		return false
	}
	// X ⊥ Y | Z == Y ⊥ X | Z (symmetry of independence)
	if setsEqual(a.event1, other.event1) && setsEqual(a.event2, other.event2) {
		return true
	}
	if setsEqual(a.event1, other.event2) && setsEqual(a.event2, other.event1) {
		return true
	}
	return false
}

// Contains returns true if this assertion implies other. An assertion X ⊥ Y | Z
// implies X' ⊥ Y' | Z if X' ⊆ X and Y' ⊆ Y (or the symmetric case), and the
// conditioning sets are equal.
func (a *IndependenceAssertion) Contains(other *IndependenceAssertion) bool {
	if other == nil {
		return true
	}
	if !setsEqual(a.given, other.given) {
		return false
	}
	// Check both orientations due to symmetry
	if isSubset(other.event1, a.event1) && isSubset(other.event2, a.event2) {
		return true
	}
	if isSubset(other.event1, a.event2) && isSubset(other.event2, a.event1) {
		return true
	}
	return false
}

// String returns a human-readable representation using "X ⊥ Y | Z" notation.
func (a *IndependenceAssertion) String() string {
	x := formatSet(a.event1)
	y := formatSet(a.event2)
	if len(a.given) == 0 {
		return fmt.Sprintf("%s ⊥ %s", x, y)
	}
	z := formatSet(a.given)
	return fmt.Sprintf("%s ⊥ %s | %s", x, y, z)
}

// LatexString returns a LaTeX representation of the independence assertion.
func (a *IndependenceAssertion) LatexString() string {
	x := formatLatexSet(a.event1)
	y := formatLatexSet(a.event2)
	if len(a.given) == 0 {
		return fmt.Sprintf("%s \\perp %s", x, y)
	}
	z := formatLatexSet(a.given)
	return fmt.Sprintf("%s \\perp %s \\mid %s", x, y, z)
}

// copyAndSort returns a sorted copy of the input slice.
func copyAndSort(s []string) []string {
	c := copySlice(s)
	sort.Strings(c)
	return c
}

// copySlice returns a copy of the input slice.
func copySlice(s []string) []string {
	if s == nil {
		return nil
	}
	c := make([]string, len(s))
	copy(c, s)
	return c
}

// setsEqual returns true if two sorted string slices represent the same set.
func setsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// isSubset returns true if every element of sub is contained in super.
// Both slices must be sorted.
func isSubset(sub, super []string) bool {
	j := 0
	for i := 0; i < len(sub); i++ {
		for j < len(super) && super[j] < sub[i] {
			j++
		}
		if j >= len(super) || super[j] != sub[i] {
			return false
		}
		j++
	}
	return true
}

// formatSet formats a set of variables for display.
func formatSet(vars []string) string {
	if len(vars) == 1 {
		return vars[0]
	}
	return "{" + strings.Join(vars, ", ") + "}"
}

// formatLatexSet formats a set of variables for LaTeX display.
func formatLatexSet(vars []string) string {
	if len(vars) == 1 {
		return vars[0]
	}
	return "\\{" + strings.Join(vars, ", ") + "\\}"
}
