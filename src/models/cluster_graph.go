package models

import (
	"fmt"
	"sort"
	"strings"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// Cluster represents a cluster of variables with associated factors.
type Cluster struct {
	Variables []string
	Factors   []*factors.DiscreteFactor
}

// ClusterEdge represents an edge between two clusters with a separation set.
type ClusterEdge struct {
	Cluster1 int
	Cluster2 int
	SepSet   []string
}

// ClusterGraph represents a cluster graph — a graph of clusters of variables
// with factors, connected by edges with separation sets. Unlike a junction
// tree, a cluster graph need not be a tree and need not satisfy the running
// intersection property.
type ClusterGraph struct {
	clusters []Cluster
	edges    []ClusterEdge
}

// NewClusterGraph creates a new empty ClusterGraph.
func NewClusterGraph() *ClusterGraph {
	return &ClusterGraph{}
}

// AddCluster adds a cluster with the given variables and factors to the graph.
// Returns the index of the newly added cluster.
func (cg *ClusterGraph) AddCluster(variables []string, clusterFactors []*factors.DiscreteFactor) int {
	vars := make([]string, len(variables))
	copy(vars, variables)
	sort.Strings(vars)

	fs := make([]*factors.DiscreteFactor, len(clusterFactors))
	copy(fs, clusterFactors)

	cg.clusters = append(cg.clusters, Cluster{
		Variables: vars,
		Factors:   fs,
	})
	return len(cg.clusters) - 1
}

// AddEdge adds an edge between two clusters with a separation set.
func (cg *ClusterGraph) AddEdge(cluster1, cluster2 int, sepSet []string) error {
	if cluster1 < 0 || cluster1 >= len(cg.clusters) {
		return fmt.Errorf("models: cluster1 index %d out of range [0, %d)", cluster1, len(cg.clusters))
	}
	if cluster2 < 0 || cluster2 >= len(cg.clusters) {
		return fmt.Errorf("models: cluster2 index %d out of range [0, %d)", cluster2, len(cg.clusters))
	}
	if cluster1 == cluster2 {
		return fmt.Errorf("models: self-loops not allowed (cluster1 == cluster2 == %d)", cluster1)
	}

	ss := make([]string, len(sepSet))
	copy(ss, sepSet)
	sort.Strings(ss)

	cg.edges = append(cg.edges, ClusterEdge{
		Cluster1: cluster1,
		Cluster2: cluster2,
		SepSet:   ss,
	})
	return nil
}

// Clusters returns a copy of all clusters.
func (cg *ClusterGraph) Clusters() []Cluster {
	result := make([]Cluster, len(cg.clusters))
	for i, c := range cg.clusters {
		vars := make([]string, len(c.Variables))
		copy(vars, c.Variables)
		fs := make([]*factors.DiscreteFactor, len(c.Factors))
		copy(fs, c.Factors)
		result[i] = Cluster{Variables: vars, Factors: fs}
	}
	return result
}

// Edges returns a copy of all edges.
func (cg *ClusterGraph) Edges() []ClusterEdge {
	result := make([]ClusterEdge, len(cg.edges))
	for i, e := range cg.edges {
		ss := make([]string, len(e.SepSet))
		copy(ss, e.SepSet)
		result[i] = ClusterEdge{
			Cluster1: e.Cluster1,
			Cluster2: e.Cluster2,
			SepSet:   ss,
		}
	}
	return result
}

// CheckModel validates the cluster graph:
//  1. At least one cluster exists.
//  2. Each edge's separation set must be a subset of the intersection of the
//     two connected clusters' variable sets.
//  3. No duplicate edges between the same pair of clusters.
func (cg *ClusterGraph) CheckModel() error {
	if len(cg.clusters) == 0 {
		return fmt.Errorf("models: cluster graph has no clusters")
	}

	// Check edges.
	type edgePair struct{ a, b int }
	seen := make(map[edgePair]bool)

	for i, e := range cg.edges {
		// Canonical ordering for duplicate check.
		a, b := e.Cluster1, e.Cluster2
		if a > b {
			a, b = b, a
		}
		ep := edgePair{a, b}
		if seen[ep] {
			return fmt.Errorf("models: duplicate edge between clusters %d and %d", a, b)
		}
		seen[ep] = true

		// Check that the separation set is a subset of the intersection of the
		// two clusters' variables.
		vars1 := toStringSet(cg.clusters[e.Cluster1].Variables)
		vars2 := toStringSet(cg.clusters[e.Cluster2].Variables)

		// Compute intersection.
		inter := make(map[string]bool)
		for v := range vars1 {
			if vars2[v] {
				inter[v] = true
			}
		}

		for _, sv := range e.SepSet {
			if !inter[sv] {
				return fmt.Errorf("models: edge %d sep-set variable %q is not in the intersection of clusters %d and %d (intersection: %v)",
					i, sv, e.Cluster1, e.Cluster2, setToSortedSlice(inter))
			}
		}
	}

	return nil
}

// toStringSet converts a string slice to a set.
func toStringSet(s []string) map[string]bool {
	m := make(map[string]bool, len(s))
	for _, v := range s {
		m[v] = true
	}
	return m
}

// setToSortedSlice converts a string set to a sorted slice for display.
func setToSortedSlice(m map[string]bool) string {
	s := make([]string, 0, len(m))
	for v := range m {
		s = append(s, v)
	}
	sort.Strings(s)
	return strings.Join(s, ", ")
}
