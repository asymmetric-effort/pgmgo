package ci_tests

import (
	"math"

	"github.com/asymmetric-effort/pgmgo/lib/scigo"
	"github.com/asymmetric-effort/pgmgo/lib/tabgo"
)

// decisionTreeNode represents a single node in a CART regression tree.
type decisionTreeNode struct {
	isLeaf    bool
	value     float64 // prediction value (mean of targets) for leaf nodes
	feature   int     // split feature index
	threshold float64 // split threshold
	left      *decisionTreeNode
	right     *decisionTreeNode
	impReduct float64 // impurity reduction from this split
}

// decisionTree is a minimal CART regression tree using MSE as the split criterion.
type decisionTree struct {
	root       *decisionTreeNode
	maxDepth   int
	minSamples int
	nFeatures  int
}

// newDecisionTree creates a new decision tree regressor.
func newDecisionTree(maxDepth, minSamples int) *decisionTree {
	if maxDepth <= 0 {
		maxDepth = 10
	}
	if minSamples < 2 {
		minSamples = 2
	}
	return &decisionTree{
		maxDepth:   maxDepth,
		minSamples: minSamples,
	}
}

// fit builds the tree from training data X (n samples x p features) and target y (n samples).
func (dt *decisionTree) fit(X [][]float64, y []float64) {
	n := len(X)
	if n == 0 {
		dt.root = &decisionTreeNode{isLeaf: true, value: 0}
		return
	}
	dt.nFeatures = len(X[0])
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	dt.root = dt.buildNode(X, y, indices, 0)
}

// buildNode recursively builds tree nodes.
func (dt *decisionTree) buildNode(X [][]float64, y []float64, indices []int, depth int) *decisionTreeNode {
	n := len(indices)

	// Compute mean of y for this node.
	mean := 0.0
	for _, idx := range indices {
		mean += y[idx]
	}
	mean /= float64(n)

	// Stop conditions: max depth, too few samples, or zero variance.
	if depth >= dt.maxDepth || n < dt.minSamples {
		return &decisionTreeNode{isLeaf: true, value: mean}
	}

	// Compute current MSE.
	currentMSE := 0.0
	for _, idx := range indices {
		d := y[idx] - mean
		currentMSE += d * d
	}

	// All targets identical: no split needed.
	if currentMSE < 1e-15 {
		return &decisionTreeNode{isLeaf: true, value: mean}
	}

	bestFeature := -1
	bestThreshold := 0.0
	bestReduction := 0.0
	var bestLeft, bestRight []int

	for f := 0; f < dt.nFeatures; f++ {
		// Collect unique sorted values for this feature.
		thresholds := dt.candidateThresholds(X, indices, f)
		for _, thr := range thresholds {
			var leftIdx, rightIdx []int
			var leftSum, rightSum float64
			for _, idx := range indices {
				if X[idx][f] <= thr {
					leftIdx = append(leftIdx, idx)
					leftSum += y[idx]
				} else {
					rightIdx = append(rightIdx, idx)
					rightSum += y[idx]
				}
			}

			nL := len(leftIdx)
			nR := len(rightIdx)
			if nL == 0 || nR == 0 {
				continue
			}

			leftMean := leftSum / float64(nL)
			rightMean := rightSum / float64(nR)

			leftMSE := 0.0
			for _, idx := range leftIdx {
				d := y[idx] - leftMean
				leftMSE += d * d
			}
			rightMSE := 0.0
			for _, idx := range rightIdx {
				d := y[idx] - rightMean
				rightMSE += d * d
			}

			reduction := currentMSE - leftMSE - rightMSE
			if reduction > bestReduction {
				bestReduction = reduction
				bestFeature = f
				bestThreshold = thr
				bestLeft = leftIdx
				bestRight = rightIdx
			}
		}
	}

	// No worthwhile split found.
	if bestFeature < 0 || bestReduction < 1e-15 {
		return &decisionTreeNode{isLeaf: true, value: mean}
	}

	return &decisionTreeNode{
		isLeaf:    false,
		feature:   bestFeature,
		threshold: bestThreshold,
		impReduct: bestReduction,
		left:      dt.buildNode(X, y, bestLeft, depth+1),
		right:     dt.buildNode(X, y, bestRight, depth+1),
	}
}

// candidateThresholds returns midpoints between sorted unique feature values.
func (dt *decisionTree) candidateThresholds(X [][]float64, indices []int, feature int) []float64 {
	// Collect unique values (sorted via insertion into sorted slice).
	seen := make(map[float64]struct{})
	var vals []float64
	for _, idx := range indices {
		v := X[idx][feature]
		if _, ok := seen[v]; !ok {
			seen[v] = struct{}{}
			vals = append(vals, v)
		}
	}
	// Sort vals.
	sortFloat64s(vals)

	if len(vals) <= 1 {
		return nil
	}

	thresholds := make([]float64, len(vals)-1)
	for i := 0; i < len(vals)-1; i++ {
		thresholds[i] = (vals[i] + vals[i+1]) / 2
	}
	return thresholds
}

// sortFloat64s sorts a slice of float64 in ascending order (insertion sort, fine for small slices).
func sortFloat64s(a []float64) {
	for i := 1; i < len(a); i++ {
		key := a[i]
		j := i - 1
		for j >= 0 && a[j] > key {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = key
	}
}

// predict returns predictions for each row in X.
func (dt *decisionTree) predict(X [][]float64) []float64 {
	preds := make([]float64, len(X))
	for i, row := range X {
		preds[i] = dt.predictOne(dt.root, row)
	}
	return preds
}

// predictOne traverses the tree for a single sample.
func (dt *decisionTree) predictOne(node *decisionTreeNode, x []float64) float64 {
	if node.isLeaf {
		return node.value
	}
	if x[node.feature] <= node.threshold {
		return dt.predictOne(node.left, x)
	}
	return dt.predictOne(node.right, x)
}

// featureImportance returns the total impurity reduction for each feature,
// normalized to sum to 1.
func (dt *decisionTree) featureImportance() []float64 {
	imp := make([]float64, dt.nFeatures)
	dt.accumulateImportance(dt.root, imp)

	total := 0.0
	for _, v := range imp {
		total += v
	}
	if total > 0 {
		for i := range imp {
			imp[i] /= total
		}
	}
	return imp
}

// accumulateImportance sums impurity reductions by feature across the tree.
func (dt *decisionTree) accumulateImportance(node *decisionTreeNode, imp []float64) {
	if node == nil || node.isLeaf {
		return
	}
	imp[node.feature] += node.impReduct
	dt.accumulateImportance(node.left, imp)
	dt.accumulateImportance(node.right, imp)
}

// TreeBasedCI is a CITest that uses decision tree regression to test conditional
// independence. It is the pgmgo equivalent of pgmpy's XGBoost-based CI test,
// using simple CART trees instead of gradient boosting.
//
// Approach (residualization via trees, similar to GCM):
//  1. Fit a tree to predict X from Z; compute residuals resid_x = X - tree(Z).
//  2. Fit a tree to predict Y from Z; compute residuals resid_y = Y - tree(Z).
//  3. Test correlation of residuals using Pearson correlation t-test.
//
// When Z is empty, it reduces to a direct Pearson correlation test on X and Y.
var TreeBasedCI CITest = func(x, y string, z []string, data *tabgo.DataFrame, significance float64) (float64, float64, bool) {
	n := data.Len()
	k := len(z)

	if n < 4 {
		return 0, 1, true
	}

	xVals := data.Column(x).Float64()
	yVals := data.Column(y).Float64()

	var resX, resY []float64

	if k == 0 {
		// No conditioning variables: test correlation directly.
		resX = xVals
		resY = yVals
	} else {
		// Build the Z matrix (n x k).
		zMatrix := make([][]float64, n)
		zCols := make([][]float64, k)
		for i, name := range z {
			zCols[i] = data.Column(name).Float64()
		}
		for row := 0; row < n; row++ {
			zMatrix[row] = make([]float64, k)
			for col := 0; col < k; col++ {
				zMatrix[row][col] = zCols[col][row]
			}
		}

		// Choose a reasonable max depth based on sample size.
		maxDepth := 5
		if n < 50 {
			maxDepth = 3
		} else if n > 500 {
			maxDepth = 8
		}

		// Fit tree: X ~ Z, compute residuals.
		treeX := newDecisionTree(maxDepth, 5)
		treeX.fit(zMatrix, xVals)
		predX := treeX.predict(zMatrix)
		resX = make([]float64, n)
		for i := 0; i < n; i++ {
			resX[i] = xVals[i] - predX[i]
		}

		// Fit tree: Y ~ Z, compute residuals.
		treeY := newDecisionTree(maxDepth, 5)
		treeY.fit(zMatrix, yVals)
		predY := treeY.predict(zMatrix)
		resY = make([]float64, n)
		for i := 0; i < n; i++ {
			resY[i] = yVals[i] - predY[i]
		}
	}

	// Test correlation of residuals.
	r, _ := scigo.PearsonCorrelation(resX, resY)

	df := float64(n - 2 - k)
	if df < 1 {
		return 0, 1, true
	}

	denom := 1 - r*r
	if denom < 1e-15 {
		if math.Abs(r) >= 1 {
			return math.Abs(r) * math.Sqrt(df), 0, false
		}
		return 0, 1, true
	}

	tstat := math.Abs(r) * math.Sqrt(df/denom)
	if math.IsNaN(tstat) || math.IsInf(tstat, 0) {
		return 0, 1, true
	}

	tdist := scigo.NewTDistribution(df)
	pvalue := 2 * tdist.SurvivalFunction(tstat)

	return tstat, pvalue, pvalue > significance
}

// Compile-time check that TreeBasedCI satisfies CITest.
var _ CITest = TreeBasedCI
