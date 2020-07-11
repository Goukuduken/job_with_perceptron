import numpy as np
import math

#first step: create array as matrix (3x3) +
#second step: make perceptron + 
#third step: create situations and hope that neurolink will working :) ! +

"""сделать распознование всех чисел, желательно вынести в отдельный модуль"""

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


input_data_array = np.array([[0, 1, 0],
						[1, 1, 0],
						[0, 1, 0]])

output_data_array = np.array([[0, 1, 0],
							[1, 1, 0],
							[0, 1, 0]])
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 3)) - 1

for elem in range(0,10000):
	output_data = sigmoid(np.dot(input_data_array,synaptic_weights))
	errors = output_data_array - output_data
	weight_adjustments = np.dot(input_data_array.T, errors * (output_data * (1 - output_data)))
	synaptic_weights += weight_adjustments

def discover_num(x):
	global output_data
	if (np.round(x) == np.round(output_data)).all():
		print(x, '\n', "I think it's number 1")
	else:
		print(x, '\n', "Oops! I don't know this number!")

new1_input_data_array = np.array([	[1, 1, 0],
									[1, 1, 0],
									[0, 1, 0]])

new1_output_data = sigmoid(np.dot(new1_input_data_array,synaptic_weights))
discover_num(new1_output_data)

new2_input_data_array = np.array([	[0, 1, 0],
									[0, 1, 0],
									[0, 1, 0]])

new2_output_data = sigmoid(np.dot(new2_input_data_array,synaptic_weights))
discover_num(new2_output_data)
