# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

train_info = False

# <-- 0. Process Files and Datasets -->
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# <-- 1. Initialize Network -->
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for neuron_id in range(n_inputs + 1)]} for neuron_id in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for neuron_id in range(n_hidden + 1)]} for neuron_id in range(n_outputs)]
	network.append(output_layer)
	return network
 
# seed(1)
# network = initialize_network(2, 1, 2)
# for layer in network:
# 	print(layer)


# <-- 2. Forward Propagate -->
# Calculate neuron activation for an input
def activate(weights, inputs):
	# activation = sum(weight_i * input_i) + bias
	activation = weights[-1]
	for neuron_id in range(len(weights)-1):
		activation += weights[neuron_id] * inputs[neuron_id]
	return activation

# Transfer neuron activation
def transfer(activation):
	# sigmoid
	# output = 1 / (1 + e^(-activation))
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neurons in layer:
			activation = activate(neurons['weights'], inputs)
			neurons['output'] = transfer(activation)
			new_inputs.append(neurons['output'])
		inputs = new_inputs
	return inputs

# # test forward propagation
# # network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# # 		[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
# row = [1, 0, None]
# output = forward_propagate(network, row)
# print(output)


# <-- 3. Back Propagate Error -->
# Calculate the derivative of an neuron output
def transfer_derivative(output):
	# derivative of sigmoid
	# derivative = output * (1.0 - output)
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for layer_id in reversed(range(len(network))):
		layer = network[layer_id]
		errors = list()
		# other layers
		if layer_id != len(network) - 1:
			for neuron_id in range(len(layer)):
				error = 0.0
				for neuron in network[layer_id + 1]:
					error += (neuron['weights'][neuron_id] * neuron['delta'])
				errors.append(error)
		# last layer
		else:
			for neuron_id in range(len(layer)):
				neuron = layer[neuron_id]
				errors.append(neuron['output'] - expected[neuron_id])
		# update delta
		for neuron_id in range(len(layer)):
			neuron = layer[neuron_id]
			neuron['delta'] = errors[neuron_id] * transfer_derivative(neuron['output'])

# # test backpropagation of error
# network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
# 		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
# expected = [0, 1]
# backward_propagate_error(network, expected)
# for layer in network:
# 	print(layer)


# <-- 4. Train Network -->
# Update network weights with error
def update_weights(network, row, l_rate):
	for layer_id in range(len(network)):
		inputs = row[:-1]
		if layer_id != 0:
			inputs = [neuron['output'] for neuron in network[layer_id - 1]]
		# weight = weight - learning_rate * error * input
		for neuron in network[layer_id]:
			for neuron_id in range(len(inputs)):
				neuron['weights'][neuron_id] -= l_rate * neuron['delta'] * inputs[neuron_id]
			neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for output_id in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[expected_id] - outputs[expected_id]) ** 2 for expected_id in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		if train_info == True:
			print('>epoch=%d, l_rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# # Test training backprop algorithm
# seed(1)
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
# n_inputs = len(dataset[0]) - 1
# n_outputs = len(set([row[-1] for row in dataset]))
# network = initialize_network(n_inputs, 2, n_outputs)
# train_network(network, dataset, 0.5, 20, n_outputs)
# for layer in network:
# 	print(layer)


# <-- 5. Predict -->
# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# # Test making predictions with the network
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
# network = [[{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]}, {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
# 	[{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]}, {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]]
# for row in dataset:
# 	prediction = predict(network, row)
# 	print('Expected=%d, Got=%d' % (row[-1], prediction))


# <-- 6. Others AI Relative Technique -->
# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


# <-- 7. Test -->
def test(filename, seed_id=1):
	# Test Backprop on Seeds dataset
	seed(seed_id)
	# load and prepare data
	# filename = 'seeds_dataset.csv'
	dataset = load_csv(filename)
	for i in range(len(dataset[0]) - 1):
		str_column_to_float(dataset, i)
	# convert class column to integers
	str_column_to_int(dataset, len(dataset[0]) - 1)
	# normalize input variables
	minmax = dataset_minmax(dataset)
	normalize_dataset(dataset, minmax)
	# evaluate algorithm
	n_folds = 5
	l_rate = 0.3
	n_epoch = 500
	n_hidden = 5
	scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

if __name__ == "__main__":
	filename = 'seeds_dataset.csv'
	train_info = False
	test(filename)