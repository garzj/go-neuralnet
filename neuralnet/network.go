package neuralnet

type Network struct {
	neurons []int
	layers  []*layer
}

func NewNetwork(neurons []int) *Network {
	if len(neurons) < 1 {
		panic("at least one layer required")
	}

	n := &Network{
		neurons: neurons,
		layers:  newLayers(neurons),
	}

	return n
}

func (n *Network) Compute(x []float64) (y []float64) {
	if len(x) != len(n.layers[0].a) {
		panic("insufficient amount of x values")
	}

	n.layers[0].a = x

	for _, l := range n.layers {
		l.compute()
	}

	return n.layers[len(n.layers)-1].a
}
