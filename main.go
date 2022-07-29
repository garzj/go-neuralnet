package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/garzj/go-neuralnet/mnist"
	"github.com/garzj/go-neuralnet/neuralnet"
)

func pixelsToX(pixels []uint8) (x []float64) {
	x = make([]float64, len(pixels))
	for i := range x {
		x[i] = float64(pixels[i]) / 255
	}
	return x
}

func labelToY(label int) (y []float64) {
	y = make([]float64, 10)
	y[label] = 1
	return y
}

func yToLabel(y []float64) (label int) {
	label = 0
	for i := 1; i < len(y); i++ {
		if y[i] > y[label] {
			label = i
		}
	}
	return label
}

func printImage(image *mnist.Image) {
	for y := 0; y < 28; y++ {
		for x := 0; x < 28; x++ {
			s := fmt.Sprint(image.Pixels[y*28+x])
			fmt.Printf("%3s", s)
		}
		fmt.Println(image.Label)
	}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// setup network
	n := neuralnet.NewNetwork([]int{784, 16, 16, 10})

	// Run a few times
	epochs := 50
	for i := 0; i < epochs; i++ {
		fmt.Printf("Epoch %d/%d\n", i+1, epochs)
		train(n)
		classify(n, 1, false)
		fmt.Println("Done!")
	}

	// Classify some test images
	classify(n, 10, true)
}

func classify(n *neuralnet.Network, exampleCount int, printImages bool) {
	// classify examples
	imageReader, err := mnist.NewImageReader("files/t10k-images-idx3-ubyte", "files/t10k-labels-idx1-ubyte")
	if err != nil {
		panic(err)
	}
	defer imageReader.Close()

	skip := rand.Intn(100)
	for i := 0; i < skip; i++ {
		imageReader.Next()
	}

	for i := 0; i < exampleCount; i++ {
		image, err := imageReader.Next()
		if err != nil {
			panic(err)
		}

		label := yToLabel(n.Compute(pixelsToX(image.Pixels)))
		cost := n.ComputeCost(pixelsToX(image.Pixels), labelToY(image.Label))

		fmt.Println("Actual label:", image.Label)
		fmt.Println("Classified:", label)
		fmt.Println("Cost:", cost, math.Round(cost))

		if printImages {
			printImage(image)
		}
	}

}

func train(n *neuralnet.Network) {
	// setup image reader
	imageReader, err := mnist.NewImageReader("files/train-images-idx3-ubyte", "files/train-labels-idx1-ubyte")
	if err != nil {
		panic(err)
	}
	defer imageReader.Close()

	// train network
	var wg sync.WaitGroup
	data := make(chan *neuralnet.TrainingSet)

	wg.Add(1)
	go func() {
		defer wg.Done()
		n.Train(data, 30)
	}()

	for i := 0; i < 960; i++ {
		image, err := imageReader.Next()
		if err != nil {
			panic(err)
		}

		data <- &neuralnet.TrainingSet{
			X: pixelsToX(image.Pixels),
			Y: labelToY(image.Label),
		}
	}

	close(data)
	wg.Wait()
}
