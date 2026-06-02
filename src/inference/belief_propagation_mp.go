package inference

import (
	"fmt"

	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// MessagePass specifies an explicit message from clique From to clique To.
type MessagePass struct {
	From, To int
}

// BeliefPropagationWithMessagePassing is a variant of BeliefPropagation that
// uses a caller-supplied message schedule rather than the automatic
// collect/distribute traversal of a rooted junction tree.
type BeliefPropagationWithMessagePassing struct {
	bp       *BeliefPropagation
	schedule []MessagePass
}

// NewBeliefPropagationWithMessagePassing creates a new engine with an explicit
// message schedule. Parameters are the same as NewBeliefPropagation with the
// addition of a schedule that lists the order in which messages are sent.
func NewBeliefPropagationWithMessagePassing(
	cliques [][]string,
	separators map[string][]string,
	cliqueFactors map[int][]*factors.DiscreteFactor,
	schedule []MessagePass,
) *BeliefPropagationWithMessagePassing {
	bp := NewBeliefPropagation(cliques, separators, cliqueFactors)
	sched := make([]MessagePass, len(schedule))
	copy(sched, schedule)
	return &BeliefPropagationWithMessagePassing{
		bp:       bp,
		schedule: sched,
	}
}

// Calibrate sends messages in the specified schedule order instead of the
// automatic collect/distribute traversal. After all scheduled messages have
// been sent, clique potentials are updated by absorbing incoming messages.
func (bpm *BeliefPropagationWithMessagePassing) Calibrate() error {
	if err := bpm.bp.initializePotentials(); err != nil {
		return err
	}

	if len(bpm.bp.cliques) <= 1 {
		bpm.bp.calibrated = true
		return nil
	}

	// Validate schedule entries.
	n := len(bpm.bp.cliques)
	for i, mp := range bpm.schedule {
		if mp.From < 0 || mp.From >= n {
			return fmt.Errorf("belief_propagation_mp: schedule[%d].From=%d out of range [0,%d)", i, mp.From, n)
		}
		if mp.To < 0 || mp.To >= n {
			return fmt.Errorf("belief_propagation_mp: schedule[%d].To=%d out of range [0,%d)", i, mp.To, n)
		}
		// Check that From and To are neighbors (share a separator).
		sepKey := edgeKey(mp.From, mp.To)
		if _, ok := bpm.bp.separators[sepKey]; !ok {
			return fmt.Errorf("belief_propagation_mp: schedule[%d] (%d->%d) has no separator", i, mp.From, mp.To)
		}
	}

	// Send messages in schedule order.
	for _, mp := range bpm.schedule {
		msg, err := bpm.bp.computeMessage(mp.From, mp.To)
		if err != nil {
			return fmt.Errorf("belief_propagation_mp: message %d->%d failed: %w", mp.From, mp.To, err)
		}
		bpm.bp.messages[msgKey(mp.From, mp.To)] = msg
	}

	// Update potentials: each clique's belief = initial potential * all incoming messages.
	for i := range bpm.bp.cliques {
		belief := bpm.bp.potentials[i]
		for _, nb := range bpm.bp.neighbors[i] {
			key := msgKey(nb, i)
			if msg, ok := bpm.bp.messages[key]; ok {
				prod, err := factors.FactorProduct(belief, msg)
				if err != nil {
					return fmt.Errorf("belief_propagation_mp: failed to absorb message into clique %d: %w", i, err)
				}
				belief = prod
			}
		}
		bpm.bp.potentials[i] = belief
	}

	bpm.bp.calibrated = true
	return nil
}

// Query computes P(queryVars | evidence) after calibration.
func (bpm *BeliefPropagationWithMessagePassing) Query(queryVars []string, evidence map[string]int) (*factors.DiscreteFactor, error) {
	return bpm.bp.Query(queryVars, evidence)
}

// GetCliqueBelief returns the calibrated potential for the given clique index.
func (bpm *BeliefPropagationWithMessagePassing) GetCliqueBelief(cliqueIndex int) *factors.DiscreteFactor {
	return bpm.bp.GetCliqueBelief(cliqueIndex)
}

// IsCalibrated returns true if Calibrate has been called successfully.
func (bpm *BeliefPropagationWithMessagePassing) IsCalibrated() bool {
	return bpm.bp.IsCalibrated()
}
