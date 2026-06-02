package base

import "fmt"

// SimpleCausalModel represents a basic structural causal model (SCM) with a
// DAG encoding the causal structure and functional equations that determine
// each endogenous variable as a function of its parent values.
type SimpleCausalModel struct {
	dag       *DAG
	equations map[string]func(parentValues map[string]float64) float64
}

// NewSimpleCausalModel creates a new SCM backed by the given DAG.
func NewSimpleCausalModel(dag *DAG) *SimpleCausalModel {
	return &SimpleCausalModel{
		dag:       dag,
		equations: make(map[string]func(parentValues map[string]float64) float64),
	}
}

// SetEquation assigns a structural equation to a variable. The function takes
// a map of parent variable names to their values and returns the computed
// value for this variable.
func (m *SimpleCausalModel) SetEquation(variable string, fn func(parentValues map[string]float64) float64) {
	m.equations[variable] = fn
}

// Intervene creates a mutilated model by performing do(variable = value).
// The returned model has the same DAG structure except all incoming edges to
// the intervened variable are removed, and its equation is replaced with a
// constant function returning value.
func (m *SimpleCausalModel) Intervene(variable string, value float64) *SimpleCausalModel {
	if !m.dag.HasNode(variable) {
		return m.Copy()
	}

	newDAG := m.dag.Copy()

	// Remove all incoming edges to the intervened variable.
	parents := newDAG.Parents(variable)
	for _, p := range parents {
		_ = newDAG.RemoveEdge(p, variable)
	}

	newModel := &SimpleCausalModel{
		dag:       newDAG,
		equations: make(map[string]func(parentValues map[string]float64) float64),
	}

	// Copy all equations except for the intervened variable.
	for v, fn := range m.equations {
		if v == variable {
			continue
		}
		newModel.equations[v] = fn
	}

	// Set the intervened variable to a constant.
	newModel.equations[variable] = func(_ map[string]float64) float64 {
		return value
	}

	return newModel
}

// Sample computes the values of all endogenous variables given exogenous
// (root) values. Variables are evaluated in topological order. If a variable
// has no equation set, it uses its exogenous value (defaulting to 0).
func (m *SimpleCausalModel) Sample(exogenous map[string]float64) (map[string]float64, error) {
	order, err := m.dag.TopologicalOrder()
	if err != nil {
		return nil, fmt.Errorf("base: cannot sample: %w", err)
	}

	values := make(map[string]float64, len(order))

	// Seed with exogenous values.
	for k, v := range exogenous {
		values[k] = v
	}

	for _, node := range order {
		fn, hasFn := m.equations[node]
		if !hasFn {
			// No equation: use exogenous value (already in values, or 0).
			continue
		}

		// Gather parent values.
		parents := m.dag.Parents(node)
		parentVals := make(map[string]float64, len(parents))
		for _, p := range parents {
			parentVals[p] = values[p]
		}

		values[node] = fn(parentVals)
	}

	return values, nil
}

// DAG returns the underlying DAG of the causal model.
func (m *SimpleCausalModel) DAG() *DAG {
	return m.dag
}

// Copy returns a deep copy of the SimpleCausalModel. Note that equation
// functions are shared (not deep-copied) since Go functions are reference types.
func (m *SimpleCausalModel) Copy() *SimpleCausalModel {
	newModel := &SimpleCausalModel{
		dag:       m.dag.Copy(),
		equations: make(map[string]func(parentValues map[string]float64) float64, len(m.equations)),
	}
	for v, fn := range m.equations {
		newModel.equations[v] = fn
	}
	return newModel
}
