import numpy as np 

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# create array with input data and array.T with output data
input_data_array = np.array([[1, 0, 0,],
				[0, 1, 1],
				[1, 1 ,0],
				[0, 0 ,1]])
output_data_array = np.array([[1,0,1,0]]).T
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

#do 10k iteration with input data
for elem in range(0,10000):
	output_data = sigmoid(np.dot(input_data_array,synaptic_weights))
	errors = output_data_array - output_data
	weight_adjustments = np.dot(input_data_array.T, errors * (output_data * (1 - output_data)))
	synaptic_weights += weight_adjustments

print(output_data)

#new situation
input_data_array = np.array([[1, 0, 0,],
				[0, 1, 1],
				[1, 1 ,0],
				[1, 0 ,1],
				[0, 0 ,1]])

output_data = sigmoid(np.dot(input_data_array,synaptic_weights))
print(output_data)

#next stage: find neuro libs and try use thems in another projects! glhf for me :)
