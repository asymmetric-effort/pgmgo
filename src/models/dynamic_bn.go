package models

import (
	"fmt"
	"sort"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// DynamicBayesianNetwork represents a two-time-slice Bayesian network (2TBN).
// It consists of an initial network for t=0 and a transition network for t→t+1.
// Interface nodes are those that appear in both time slices, connecting them.
type DynamicBayesianNetwork struct {
	initial    *BayesianNetwork
	transition *BayesianNetwork
}

// NewDynamicBayesianNetwork creates a new empty DynamicBayesianNetwork.
func NewDynamicBayesianNetwork() *DynamicBayesianNetwork {
	return &DynamicBayesianNetwork{
		initial:    NewBayesianNetwork(),
		transition: NewBayesianNetwork(),
	}
}

// Initial returns the initial (t=0) BayesianNetwork.
func (dbn *DynamicBayesianNetwork) Initial() *BayesianNetwork {
	return dbn.initial
}

// Transition returns the transition (t→t+1) BayesianNetwork.
func (dbn *DynamicBayesianNetwork) Transition() *BayesianNetwork {
	return dbn.transition
}

// AddInitialCPD adds a CPD to the initial (t=0) time slice.
// The CPD's variable must be a node in the initial network.
func (dbn *DynamicBayesianNetwork) AddInitialCPD(cpd *factors.TabularCPD) error {
	if cpd == nil {
		return fmt.Errorf("models: cpd must not be nil")
	}
	return dbn.initial.AddCPD(cpd)
}

// AddTransitionCPD adds a CPD to the transition (t→t+1) network.
// The CPD's variable must be a node in the transition network.
func (dbn *DynamicBayesianNetwork) AddTransitionCPD(cpd *factors.TabularCPD) error {
	if cpd == nil {
		return fmt.Errorf("models: cpd must not be nil")
	}
	return dbn.transition.AddCPD(cpd)
}

// GetInterfaceNodes returns the sorted list of nodes that appear in both the
// initial and transition networks, i.e., the nodes connecting time slices.
func (dbn *DynamicBayesianNetwork) GetInterfaceNodes() []string {
	initialNodes := make(map[string]bool)
	for _, n := range dbn.initial.Nodes() {
		initialNodes[n] = true
	}

	var iface []string
	for _, n := range dbn.transition.Nodes() {
		if initialNodes[n] {
			iface = append(iface, n)
		}
	}
	sort.Strings(iface)
	return iface
}

// CheckModel validates both the initial and transition networks.
func (dbn *DynamicBayesianNetwork) CheckModel() error {
	if err := dbn.initial.CheckModel(); err != nil {
		return fmt.Errorf("models: initial network: %w", err)
	}
	if err := dbn.transition.CheckModel(); err != nil {
		return fmt.Errorf("models: transition network: %w", err)
	}
	return nil
}

// Copy returns a deep copy of the DynamicBayesianNetwork.
func (dbn *DynamicBayesianNetwork) Copy() *DynamicBayesianNetwork {
	return &DynamicBayesianNetwork{
		initial:    dbn.initial.Copy(),
		transition: dbn.transition.Copy(),
	}
}
