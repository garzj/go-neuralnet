package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/garzj/go-neuralnet/mnist"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	imageReader, err := mnist.NewImageReader("files/t10k-images-idx3-ubyte", "files/t10k-labels-idx1-ubyte")
	if err != nil {
		panic(err)
	}
	defer imageReader.Close()

	image, err := imageReader.Next()
	if err != nil {
		panic(err)
	}
	fmt.Println("Image")
	fmt.Println("Label:", image.Label)
	fmt.Println("Pixels:", image.Pixels)
}
