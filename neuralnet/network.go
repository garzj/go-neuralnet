package neuralnet

import (
	"fmt"
	"math"
	"sync"
)

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

	n.Randomize()

	return n
}

func (n *Network) Randomize() {
	for _, l := range n.layers {
		l.randomize()
	}
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

func (n *Network) ComputeCost(x []float64, y []float64) float64 {
	resY := n.Compute(x)
	cost := .0
	for j := range y {
		cost += math.Pow(resY[j]-y[j], 2)
	}
	return cost
}

func (n *Network) Train(data chan *TrainingSet, batchSize int) {
	var wg sync.WaitGroup
	var bData chan *TrainingSet

	for i := 0; ; i++ {
		i %= batchSize

		set, more := <-data
		if !more {
			if i != 0 {
				err := fmt.Sprintf("data channel closed on incomplete batch (%d / %d)", i, batchSize)
				panic(err)
			}
			break
		}

		if i == 0 {
			if bData != nil {
				close(bData)
			}
			wg.Wait()

			bData = make(chan *TrainingSet)
			b := NewBatch(n)
			wg.Add(1)
			go func() {
				defer wg.Done()
				b.Train(bData)
			}()
		}

		bData <- set
	}

	if bData != nil {
		close(bData)
	}
	wg.Wait()
}
