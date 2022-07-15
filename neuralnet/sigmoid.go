package neuralnet

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func DSigmoid(x float64) float64 {
	return 1 / (4 * math.Pow(math.Cosh(x/2), 2))
}
