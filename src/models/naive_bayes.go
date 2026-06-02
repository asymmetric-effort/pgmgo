package models

import (
	"fmt"
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
	"github.com/asymmetric-effort/pgmgo/src/factors"
)

// NaiveBayes represents a Naive Bayes classifier — a Bayesian network with a
// star topology where the class variable is the root and all features are
// conditionally independent given the class.
type NaiveBayes struct {
	*BayesianNetwork
	classVariable string
	features      []string
}

// NewNaiveBayes creates a new NaiveBayes model with the given class variable
// and feature variables. It constructs a star topology DAG where classVariable
// is a parent of each feature.
func NewNaiveBayes(classVariable string, features []string) (*NaiveBayes, error) {
	if classVariable == "" {
		return nil, fmt.Errorf("models: classVariable must not be empty")
	}
	if len(features) == 0 {
		return nil, fmt.Errorf("models: features must not be empty")
	}

	// Check for duplicates.
	seen := map[string]bool{classVariable: true}
	for _, f := range features {
		if f == classVariable {
			return nil, fmt.Errorf("models: feature %q is the same as the class variable", f)
		}
		if seen[f] {
			return nil, fmt.Errorf("models: duplicate feature %q", f)
		}
		seen[f] = true
	}

	bn := NewBayesianNetwork()
	if err := bn.AddNode(classVariable); err != nil {
		return nil, err
	}
	for _, f := range features {
		if err := bn.AddNode(f); err != nil {
			return nil, err
		}
		if err := bn.AddEdge(classVariable, f); err != nil {
			return nil, err
		}
	}

	feats := make([]string, len(features))
	copy(feats, features)

	return &NaiveBayes{
		BayesianNetwork: bn,
		classVariable:   classVariable,
		features:        feats,
	}, nil
}

// ClassVariable returns the name of the class variable.
func (nb *NaiveBayes) ClassVariable() string {
	return nb.classVariable
}

// Features returns a copy of the feature variable names.
func (nb *NaiveBayes) Features() []string {
	f := make([]string, len(nb.features))
	copy(f, nb.features)
	return f
}

// Fit estimates parameters from data using maximum likelihood estimation (MLE).
// The DataFrame must contain columns for the class variable and all features.
// All values must be non-negative integers representing discrete state indices.
func (nb *NaiveBayes) Fit(data *tabgo.DataFrame) error {
	if data == nil {
		return fmt.Errorf("models: data must not be nil")
	}
	if data.Len() == 0 {
		return fmt.Errorf("models: data must not be empty")
	}

	nRows := data.Len()
	classVals := data.Column(nb.classVariable).Int()

	// Determine class cardinality.
	classCard := 0
	for _, v := range classVals {
		if v < 0 {
			return fmt.Errorf("models: negative class value %d", v)
		}
		if v+1 > classCard {
			classCard = v + 1
		}
	}

	// Count class occurrences.
	classCounts := make([]float64, classCard)
	for _, v := range classVals {
		classCounts[v]++
	}

	// Build class CPD (prior).
	classProbs := make([][]float64, classCard)
	for i := 0; i < classCard; i++ {
		classProbs[i] = []float64{classCounts[i] / float64(nRows)}
	}
	classCPD, err := factors.NewTabularCPD(nb.classVariable, classCard, classProbs, nil, nil)
	if err != nil {
		return fmt.Errorf("models: failed to create class CPD: %w", err)
	}
	if err := nb.BayesianNetwork.AddCPD(classCPD); err != nil {
		return fmt.Errorf("models: failed to add class CPD: %w", err)
	}

	// For each feature, build a CPD conditioned on the class.
	for _, feat := range nb.features {
		featVals := data.Column(feat).Int()

		// Determine feature cardinality.
		featCard := 0
		for _, v := range featVals {
			if v < 0 {
				return fmt.Errorf("models: negative feature value %d for %q", v, feat)
			}
			if v+1 > featCard {
				featCard = v + 1
			}
		}

		// Count joint occurrences: counts[featState][classState].
		counts := make([][]float64, featCard)
		for i := range counts {
			counts[i] = make([]float64, classCard)
		}
		for row := 0; row < nRows; row++ {
			counts[featVals[row]][classVals[row]]++
		}

		// Normalize each column (class state) to sum to 1.
		for c := 0; c < classCard; c++ {
			colSum := classCounts[c]
			if colSum == 0 {
				// Uniform distribution if no samples for this class.
				for f := 0; f < featCard; f++ {
					counts[f][c] = 1.0 / float64(featCard)
				}
			} else {
				for f := 0; f < featCard; f++ {
					counts[f][c] /= colSum
				}
			}
		}

		cpd, err := factors.NewTabularCPD(feat, featCard, counts,
			[]string{nb.classVariable}, []int{classCard})
		if err != nil {
			return fmt.Errorf("models: failed to create CPD for %q: %w", feat, err)
		}
		if err := nb.BayesianNetwork.AddCPD(cpd); err != nil {
			return fmt.Errorf("models: failed to add CPD for %q: %w", feat, err)
		}
	}

	return nil
}

// PredictProbability returns the posterior probability of each class for each
// row in data. The result is a slice of length data.Len(), where each element
// is a slice of class probabilities.
func (nb *NaiveBayes) PredictProbability(data *tabgo.DataFrame) ([][]float64, error) {
	if data == nil {
		return nil, fmt.Errorf("models: data must not be nil")
	}
	if err := nb.BayesianNetwork.CheckModel(); err != nil {
		return nil, fmt.Errorf("models: model is not valid: %w", err)
	}

	classCPD := nb.GetCPD(nb.classVariable)
	if classCPD == nil {
		return nil, fmt.Errorf("models: no CPD for class variable %q", nb.classVariable)
	}
	classCard := classCPD.VariableCard()
	classPrior := classCPD.ToFactor().Values().Data()

	nRows := data.Len()
	result := make([][]float64, nRows)

	// Pre-fetch feature values.
	featVals := make([][]int, len(nb.features))
	featCPDs := make([]*factors.TabularCPD, len(nb.features))
	for i, feat := range nb.features {
		featVals[i] = data.Column(feat).Int()
		featCPDs[i] = nb.GetCPD(feat)
		if featCPDs[i] == nil {
			return nil, fmt.Errorf("models: no CPD for feature %q", feat)
		}
	}

	for row := 0; row < nRows; row++ {
		posterior := make([]float64, classCard)

		for c := 0; c < classCard; c++ {
			logProb := math.Log(classPrior[c])

			for i := range nb.features {
				cpd := featCPDs[i]
				featCard := cpd.VariableCard()
				fv := featVals[i][row]
				if fv < 0 || fv >= featCard {
					return nil, fmt.Errorf("models: feature value %d out of range for %q (card %d)",
						fv, nb.features[i], featCard)
				}
				// CPD data layout: data[featState * numParentConfigs + parentConfig]
				// Parent config for single parent (class) with state c is just c.
				val := cpd.ToFactor().Values().Data()[fv*classCard+c]
				if val <= 0 {
					logProb = math.Inf(-1)
					break
				}
				logProb += math.Log(val)
			}
			posterior[c] = logProb
		}

		// Convert from log-space, using log-sum-exp for numerical stability.
		maxLog := math.Inf(-1)
		for _, lp := range posterior {
			if lp > maxLog {
				maxLog = lp
			}
		}

		sum := 0.0
		for i := range posterior {
			if math.IsInf(maxLog, -1) {
				posterior[i] = 0
			} else {
				posterior[i] = math.Exp(posterior[i] - maxLog)
			}
			sum += posterior[i]
		}
		if sum > 0 {
			for i := range posterior {
				posterior[i] /= sum
			}
		}

		result[row] = posterior
	}

	return result, nil
}

// Predict returns the predicted class (index of highest posterior probability)
// for each row in data.
func (nb *NaiveBayes) Predict(data *tabgo.DataFrame) ([]int, error) {
	probs, err := nb.PredictProbability(data)
	if err != nil {
		return nil, err
	}

	predictions := make([]int, len(probs))
	for i, p := range probs {
		bestClass := 0
		bestProb := p[0]
		for c := 1; c < len(p); c++ {
			if p[c] > bestProb {
				bestProb = p[c]
				bestClass = c
			}
		}
		predictions[i] = bestClass
	}

	return predictions, nil
}
