package factors

import (
	"fmt"
	"sort"
)

// FactorSet represents an unordered set of discrete factors.
type FactorSet struct {
	factors []*DiscreteFactor
}

// NewFactorSet creates a new FactorSet containing the given factors.
func NewFactorSet(factors ...*DiscreteFactor) *FactorSet {
	fs := &FactorSet{
		factors: make([]*DiscreteFactor, 0, len(factors)),
	}
	for _, f := range factors {
		if f != nil {
			fs.factors = append(fs.factors, f)
		}
	}
	return fs
}

// Add adds a factor to the set.
func (fs *FactorSet) Add(f *DiscreteFactor) {
	if f == nil {
		return
	}
	fs.factors = append(fs.factors, f)
}

// Remove removes the first occurrence of the given factor (by pointer equality)
// from the set. Returns true if the factor was found and removed.
func (fs *FactorSet) Remove(f *DiscreteFactor) bool {
	for i, existing := range fs.factors {
		if existing == f {
			fs.factors = append(fs.factors[:i], fs.factors[i+1:]...)
			return true
		}
	}
	return false
}

// Contains returns true if the set contains the given factor (by pointer equality).
func (fs *FactorSet) Contains(f *DiscreteFactor) bool {
	for _, existing := range fs.factors {
		if existing == f {
			return true
		}
	}
	return false
}

// Product computes the product of all factors in the set using FactorProduct.
// Returns an error if the set is empty.
func (fs *FactorSet) Product() (*DiscreteFactor, error) {
	if len(fs.factors) == 0 {
		return nil, fmt.Errorf("factors: cannot compute product of empty FactorSet")
	}
	return FactorProduct(fs.factors...)
}

// GetFactorsOf returns all factors in the set that involve the given variable.
func (fs *FactorSet) GetFactorsOf(variable string) []*DiscreteFactor {
	var result []*DiscreteFactor
	for _, f := range fs.factors {
		for _, v := range f.variables {
			if v == variable {
				result = append(result, f)
				break
			}
		}
	}
	return result
}

// Len returns the number of factors in the set.
func (fs *FactorSet) Len() int {
	return len(fs.factors)
}

// FactorDict maps variable names to discrete factors.
type FactorDict struct {
	entries map[string]*DiscreteFactor
}

// NewFactorDict creates a new empty FactorDict.
func NewFactorDict() *FactorDict {
	return &FactorDict{
		entries: make(map[string]*DiscreteFactor),
	}
}

// Set associates a factor with the given key.
func (fd *FactorDict) Set(key string, f *DiscreteFactor) {
	fd.entries[key] = f
}

// Get returns the factor associated with the given key, or nil if not found.
func (fd *FactorDict) Get(key string) *DiscreteFactor {
	return fd.entries[key]
}

// Keys returns the keys in sorted order.
func (fd *FactorDict) Keys() []string {
	keys := make([]string, 0, len(fd.entries))
	for k := range fd.entries {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Len returns the number of entries in the dict.
func (fd *FactorDict) Len() int {
	return len(fd.entries)
}
