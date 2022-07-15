package neuralnet

type Gradient struct {
	layers  []*layer
	dLayers []*layer
}

func newGradient(neurons []int, layers []*layer, y []float64) *Gradient {
	g := &Gradient{
		layers:  layers,
		dLayers: newLayers(neurons),
	}

	g.compute(y)

	return g
}

func (g *Gradient) compute(y []float64) {
	// last layer activations
	lastNl := g.layers[len(g.layers)-1]
	if len(lastNl.a) != len(y) {
		panic("insufficient amount of y values")
	}

	lastDl := g.dLayers[len(g.dLayers)-1]
	for j := range lastNl.a {
		lastDl.a[j] = 2 * (lastNl.a[j] - y[j])
	}

	// all but first layer
	for l := len(g.layers) - 1; l >= 1; l-- {
		nl := g.layers[l]
		prevNl := g.layers[l-1]
		dl := g.dLayers[l]
		prevDl := g.dLayers[l-1]

		// weights & biases
		for j := range dl.a {
			dsz := DSigmoid(nl.weightedSum(j))
			dl.b[j] = dsz * dl.a[j]
			for k := range dl.w[j] {
				dl.w[j][k] = prevNl.a[k] * dsz * dl.a[j]
			}
		}

		// previous layer's activations
		for k := range prevDl.a {
			prevDl.a[k] = 0
			for j := range dl.a {
				prevDl.a[k] += nl.w[j][k] * DSigmoid(nl.weightedSum(j)) * dl.a[j]
			}
		}
	}
}
