package neuralnet

import (
	"sync"
)

type Batch struct {
	network   *Network
	mu        sync.Mutex
	gradients []*Gradient
}

func NewBatch(network *Network) *Batch {
	return &Batch{
		network: network,
	}
}

func (b *Batch) Add(set *TrainingSet) {
	net := b.network

	layers := copyLayers(net.neurons, net.layers)
	layers[0].a = set.X
	for _, l := range layers {
		l.compute()
	}

	g := newGradient(net.neurons, layers, set.Y)

	b.mu.Lock()
	b.gradients = append(b.gradients, g)
	b.mu.Unlock()
}

func (b *Batch) Apply() {
	// compute average gradient (storing everything in one vector would have made things easier hmm..)
	dls := newLayers(b.network.neurons)

	for _, g := range b.gradients {
		for l, dl := range dls {
			gl := g.dLayers[l]

			for j := range dl.b {
				dl.b[j] += gl.b[j]
				for k := range dl.w[j] {
					dl.w[j][k] += gl.w[j][k]
				}
			}
		}
	}

	batchSize := float64(len(b.gradients))
	for _, dl := range dls {
		for j := range dl.b {
			dl.b[j] /= batchSize
			for k := range dl.w[j] {
				dl.w[j][k] /= batchSize
			}
		}
	}

	// apply it
	for l, nl := range b.network.layers {
		dl := dls[l]

		factor := 10.0

		for j := range nl.a {
			nl.b[j] -= dl.b[j] * factor
			for k := range nl.w[j] {
				nl.w[j][k] -= dl.w[j][k] * factor
			}
		}
	}
}

func (b *Batch) Train(data chan *TrainingSet) {
	var wg sync.WaitGroup
	for {
		set, more := <-data
		if !more {
			break
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			b.Add(set)
		}()
	}
	wg.Wait()

	b.Apply()
}
