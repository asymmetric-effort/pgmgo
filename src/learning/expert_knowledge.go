package learning

// ExpertKnowledge encodes domain expert constraints for structure learning.
// It supports required edges (whitelist), forbidden edges (blacklist), and
// tier ordering (temporal or causal ordering constraints).
type ExpertKnowledge struct {
	requiredEdges  map[[2]string]bool
	forbiddenEdges map[[2]string]bool
	tiers          [][]string     // tiers[i] contains variables in tier i
	tierMap        map[string]int // variable -> tier index
}

// NewExpertKnowledge creates a new empty ExpertKnowledge instance.
func NewExpertKnowledge() *ExpertKnowledge {
	return &ExpertKnowledge{
		requiredEdges:  make(map[[2]string]bool),
		forbiddenEdges: make(map[[2]string]bool),
		tierMap:        make(map[string]int),
	}
}

// AddRequiredEdge marks the edge from -> to as required. Required edges must
// appear in the final learned structure.
func (ek *ExpertKnowledge) AddRequiredEdge(from, to string) {
	ek.requiredEdges[[2]string{from, to}] = true
}

// AddForbiddenEdge marks the edge from -> to as forbidden. Forbidden edges
// must not appear in the final learned structure.
func (ek *ExpertKnowledge) AddForbiddenEdge(from, to string) {
	ek.forbiddenEdges[[2]string{from, to}] = true
}

// AddTierOrdering sets a temporal/causal tier ordering. Variables in earlier
// tiers (lower index) cannot have parents in later tiers (higher index).
// Edges from a later tier to an earlier tier are forbidden.
// Calling this replaces any previously set tier ordering.
func (ek *ExpertKnowledge) AddTierOrdering(tiers [][]string) {
	ek.tiers = make([][]string, len(tiers))
	ek.tierMap = make(map[string]int)
	for i, tier := range tiers {
		ek.tiers[i] = make([]string, len(tier))
		copy(ek.tiers[i], tier)
		for _, v := range tier {
			ek.tierMap[v] = i
		}
	}
}

// IsAllowed returns true if the edge from -> to is allowed by the expert
// knowledge constraints. An edge is allowed if it is not forbidden and does
// not violate tier ordering.
func (ek *ExpertKnowledge) IsAllowed(from, to string) bool {
	// Check explicit forbidden edges.
	if ek.forbiddenEdges[[2]string{from, to}] {
		return false
	}

	// Check tier ordering: from must be in same or earlier tier than to.
	if len(ek.tierMap) > 0 {
		fromTier, fromOK := ek.tierMap[from]
		toTier, toOK := ek.tierMap[to]
		if fromOK && toOK {
			if fromTier > toTier {
				return false
			}
		}
	}

	return true
}

// IsRequired returns true if the edge from -> to is required by the expert
// knowledge constraints.
func (ek *ExpertKnowledge) IsRequired(from, to string) bool {
	return ek.requiredEdges[[2]string{from, to}]
}

// ApplyToHillClimb applies the expert knowledge constraints to a HillClimbSearch
// instance by setting its whitelist and blacklist.
func (ek *ExpertKnowledge) ApplyToHillClimb(hc *HillClimbSearch) {
	// Add required edges to whitelist.
	for edge := range ek.requiredEdges {
		hc.whiteList[edge] = true
	}

	// Add forbidden edges to blacklist.
	for edge := range ek.forbiddenEdges {
		hc.blackList[edge] = true
	}

	// Add tier-violating edges to blacklist.
	if len(ek.tierMap) > 0 {
		// Collect all variables that have a tier assignment.
		vars := make([]string, 0, len(ek.tierMap))
		for v := range ek.tierMap {
			vars = append(vars, v)
		}

		for _, from := range vars {
			for _, to := range vars {
				if from == to {
					continue
				}
				if ek.tierMap[from] > ek.tierMap[to] {
					hc.blackList[[2]string{from, to}] = true
				}
			}
		}
	}
}
