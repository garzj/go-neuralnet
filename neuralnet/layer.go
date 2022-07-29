package neuralnet

import (
	"math/rand"
)

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

func copyLayers(neurons []int, src []*layer) []*layer {
	dest := make([]*layer, len(src))
	for l := range dest {
		dest[l] = newLayer(neurons[l])

		destL := dest[l]
		srcL := src[l]

		if l > 0 {
			destL.in = dest[l-1]
		}

		copy(destL.a, srcL.a)
		copy(destL.b, srcL.b)

		for j := range destL.w {
			destL.w[j] = make([]float64, len(srcL.w[j]))
			copy(destL.w[j], srcL.w[j])
		}
	}
	return dest
}

func (this *layer) linkIn(other *layer) {
	this.in = other
	for j := range this.w {
		this.w[j] = make([]float64, other.neurons)
	}
}

func (l *layer) randomize() {
	for j := 0; j < l.neurons; j++ {
		l.b[j] = 0
		for k := range l.w[j] {
			l.w[j][k] = (rand.Float64() - .5) * .3
		}
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
