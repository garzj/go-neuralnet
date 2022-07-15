package neuralnet

type layer struct {
	neurons int
	in      *layer
	a       []float64
	b       []float64
	w       [][]float64
}

func newLayer(neurons int) *layer {
	return &layer{
		neurons: neurons,
		a:       make([]float64, neurons),
		b:       make([]float64, neurons),
		w:       make([][]float64, neurons),
	}
}

func newLayers(neurons []int) []*layer {
	layers := make([]*layer, len(neurons))

	for l := range layers {
		layers[l] = newLayer(neurons[l])
		if l > 0 {
			layers[l].linkIn(layers[l-1])
		}
	}

	return layers
}

func (this *layer) linkIn(other *layer) {
	this.in = other
	for j := range this.w {
		this.w[j] = make([]float64, other.neurons)
	}
}

func (l *layer) weightedSum(j int) float64 {
	z := l.b[j]
	for k := range l.w[j] {
		z += l.in.a[k] * l.w[j][k]
	}
	return z
}

func (l *layer) compute() {
	if l.in == nil {
		return
	}

	for j := range l.a {
		z := l.weightedSum(j)
		l.a[j] = Sigmoid(z)
	}
}