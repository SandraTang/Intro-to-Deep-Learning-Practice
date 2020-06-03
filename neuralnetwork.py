from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# seed random number generator (generates same numbers every time)
		random.seed(1)

		# model a single neuron with 3 input, 1 output connections
		# assign random weights to 3 by 1 matrix (values -1 to 1), mean 0
		# [0, 1] => [0, 2] => [-1, 1]
		self.synaptic_weights = 2 * random.random((3, 1)) - 1

	# activation function
	# sigmoid function (s-shaped curved)
	# pass weighted sum of inputs through function
	# converts to probability between 0 and 1
	def __sigmoid(self, x):
		return 1/(1+exp(-x))

	# calculate derivative of sigmoid, yield gradient (slope)
	# measure confidence of current weight value
	# facilitate backpropogation (propogate error value back into neural network to adjust weights)
	def __sigmoid_derivative(self, x):
		return x*(1-x)

	def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
		for iteration in range(number_of_training_iterations):
			# pass in training set
			output = self.predict(training_set_inputs)
			# calculate error
			error = training_set_outputs - output
			# calculate adjustment: multiply error by input and gradient of sigmoid curve
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
			# adjust_weights
			self.synaptic_weights = self.synaptic_weights + adjustment

	def predict(self, inputs):
		# pass inputs through neural network (single neuron)
		return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':
	# initialize single neuron neural network
	neural_network = NeuralNetwork()

	print('Random starting synaptic weights:')
	print(neural_network.synaptic_weights)

	# training set
	training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
	# T function transposes array from horizontal to vertical
	training_set_outputs = array([0, 1, 1, 0]).T

	# train neural network with training set 10k times
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print('New synaptic weights after training:')
	print(neural_network.synaptic_weights)

	# test neural network
	print('Predicting:')
	print(neural_network.predict(array([1, 0, 0])))