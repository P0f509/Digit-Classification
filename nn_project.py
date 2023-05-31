import numpy as np
import matplotlib.pyplot as plt
import cv2, copy
from scipy.special import softmax, expit
from sklearn.utils import shuffle
from keras.datasets import mnist

'''
    Activation & Error Functions
'''

def identity(x):
    return x

def identity_prime(x):
    return 1


def sigmoid(x):
	return expit(x)

def sigmoid_prime(x):
    z = expit(x)
    return z * (1 - z)


def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(x.dtype)


def sum_of_square(output, target):
    return 0.5 * np.sum(np.square(output - target))

def sum_of_square_prime(output, target):
    return output - target


def cross_entropy(output, target):
    return -np.sum(target * np.log(output))

def cross_entropy_prime(output, target):
    return -(target / output)


def cross_entropy_softmax(output, target):
    return cross_entropy(softmax(output), target)

def cross_entropy_softmax_prime(output, target):
    return softmax(output) - target



'''
    This class represents a linear layer of a neural network
    and computes the input of an Activation Layer
'''
class ConnectionLayer:
    '''
        Constructor __init__
        - input_dim: number of neurons of previous layer
        - output_dim: number of neurons of next layer
        - weights: matrix of weights W^l [output_dim * input_dim] 
        - bias: matrix of biases B^l [output_dim * 1]
    '''
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.normal(0, 0.1, (output_dim, input_dim)) 
        self.bias = np.random.normal(0, 0.1, (output_dim, 1)) 


    def forward_step(self, input_data):
        if input_data.shape[0] != self.input_dim:
            raise ValueError('Forward Propagation Error: Invalid Input Dimension')
        
        self.input = input_data
        self.output = np.matmul(self.weights, input_data) + self.bias

        return self.output
    

    def backward_step(self, delta):
        return np.matmul(self.weights.transpose(), delta)
    

    def compute_derivate(self, delta):
        return np.matmul(delta, self.input.transpose())
    

    def update_weights(self, derivates, deltas, lr):
        self.weights = self.weights - (lr * derivates)
        self.bias = self.bias - (lr * deltas)


    def update_weights_rprop(self, derivates, deltas, weights_increment, biases_increment):
        self.weights = self.weights - (np.multiply(np.sign(derivates), weights_increment))
        self.bias = self.bias - (np.multiply(np.sign(deltas), biases_increment))

    
'''
    This class represents a non-linear layer of a neural network
    and computes the activation of the neurons belonging to the layer
'''
class ActivationLayer:
    '''
        Constructor __init__
        - neurons_number: layer size
        - activation_fun: activation function of the layer
        - activation_fun_prime: derivate of activation_fun
    '''
    def __init__(self, neurons_number, activation_fun, activation_fun_prime):
        self.neurons_number = neurons_number
        self.activation_fun = activation_fun
        self.activation_fun_prime = activation_fun_prime
        

    def forward_step(self, input_data):
        if input_data.shape[0] != self.neurons_number:
            raise ValueError('Forward Propagation Error: Invalid Input Dimension')
        
        self.input = input_data
        self.output = self.activation_fun(input_data)
        return self.output
    

    def backward_step(self, delta):
        return np.multiply(self.activation_fun_prime(self.input), delta)
    
     
'''
    This class represents a feed forward neural network with multiple layers
'''
class NeuralNetwork:
    '''
        Constructor __init__
        - loss: error function for learning process
        - loss_prime: derivate of loss
    '''
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime


    def add_layer(self, layer):
        if not (isinstance(layer, ConnectionLayer) or isinstance(layer, ActivationLayer)):
            raise TypeError('Neural Network instantiation Error: Invalid Input')
        
        if ((len(self.layers) == 0) and (not isinstance(layer, ConnectionLayer)) or \
            (len(self.layers) != 0) and isinstance(self.layers[-1], type(layer))):
            raise ValueError('Neural Network instantiation Error: Invalid Layers Sequence')
        
        if layer in self.layers:
            raise ValueError('Neural Network instantiation Error: Layer already exists')
        
        self.layers.append(layer)


    def forward_propagation(self, train_data):
        output = train_data
        for layer in self.layers:
            output = layer.forward_step(output)

        return output
    

    def back_propagation(self, output, target):
        back_layers = self.layers[::-1]
        deltas = []

        delta = self.loss_prime(output, target)
        delta = back_layers[0].backward_step(delta)
        deltas.append(delta)
        
        for i in range(1, len(back_layers) - 1, 2):
            delta = back_layers[i].backward_step(delta)
            delta = back_layers[i+1].backward_step(delta)
            deltas.append(delta)

        deltas.reverse()
        return deltas
    
    
    def compute_derivates(self, deltas):
        derivates = []
        for i in range(0, len(self.layers), 2):
            j = int(i/2)
            derivates.append(self.layers[i].compute_derivate(deltas[j]))
        return derivates
    

    def update_weights(self, derivates, deltas, lr):
        for i in range(0, len(self.layers), 2):
            j = int(i/2)
            self.layers[i].update_weights(derivates[j], deltas[j], lr)


    def update_weights_rprop(self, derivates, deltas, weights_increment, biases_increment):
        for i in range(0, len(self.layers), 2):
            j = int(i/2)
            self.layers[i].update_weights_rprop(derivates[j], deltas[j], weights_increment[j], biases_increment[j])

 
    def learn(self, train_X, train_Y, val_X, val_Y, epoches, lr, mod='B'):

        if len(self.layers) < 2:
            raise RuntimeError('Forward Propagation Error: Invalid Neural Network (too few layers)')
        
        if isinstance(self.layers[-1], ConnectionLayer):
            raise RuntimeError('Forward Propagation Error: Invalid Neural Network (last layer is not a processing layer)')
        
        if epoches < 1:
            raise ValueError('Learning Error: parameter "epoches" must be a positive integer')
        

        train_errors_epoches = []
        val_errors_epoches = []
        best_fitting_network = self

        for _ in range(epoches):

            # resetting variables for each epoque
            train_error = 0
            val_error = 0

            outputs = []
            sum_derivates = []
            sum_deltas = []
            for i in range(0, len(self.layers), 2):
                sum_derivates.append(np.zeros((self.layers[i].output_dim, self.layers[i].input_dim)))
                sum_deltas.append(np.zeros((self.layers[i].output_dim, 1)))


            # running over training set
            for i in range(len(train_X)):

                outputs.append(self.forward_propagation(train_X[i]))
                deltas = self.back_propagation(outputs[i], train_Y[i])
                derivates = self.compute_derivates(deltas)

                if mod == 'O':   
                    self.update_weights(derivates, deltas, lr)
                else:
                    for i in range(len(self.layers)//2):
                        sum_derivates[i] = np.add(sum_derivates[i], derivates[i])
                        sum_deltas[i] = np.add(sum_deltas[i], deltas[i])

            if mod == 'B':
                self.update_weights(sum_derivates, sum_deltas, lr)  

            # computing error on training and validation set
            for i in range(len(train_X)):
                train_error += self.loss(outputs[i], train_Y[i])
            train_errors_epoches.append(train_error)

            for i in range(len(val_X)):
                val_error += self.loss(self.forward_propagation(val_X[i]), val_Y[i])
            if len(val_errors_epoches) == 0:
                best_fitting_network = self
            elif val_error < val_errors_epoches[-1]:
                best_fitting_network = copy.deepcopy(self)
            val_errors_epoches.append(val_error)

        train_errors_epoches = np.array(train_errors_epoches)
        val_errors_epoches = np.array(val_errors_epoches)

        return best_fitting_network, train_errors_epoches, val_errors_epoches
    


    def learn_rprop(self, train_X, train_Y, val_X, val_Y, epoches, eta_plus, eta_minus, delta_zero, delta_max, delta_min):

        if len(self.layers) < 2:
            raise RuntimeError('Forward Propagation Error: Invalid Neural Network (too few layers)')
        
        if isinstance(self.layers[-1], ConnectionLayer):
            raise RuntimeError('Forward Propagation Error: Invalid Neural Network (last layer is not a processing layer)')
        
        if epoches < 1:
            raise ValueError('Learning Error: parameter "epoches" must be a positive integer')
        
        train_errors_epoches = []
        val_errors_epoches = []
        best_fitting_network = self

        prev_derivates = []
        prev_deltas = []
        prev_delta_increments_weights = []
        prev_delta_increments_biases = []
        delta_min_weights = []
        delta_max_weights = []
        delta_min_biases = []
        delta_max_biases = []

        for i in range(0, len(self.layers), 2):
            prev_derivates.append(np.zeros((self.layers[i].output_dim, self.layers[i].input_dim)))
            prev_deltas.append(np.zeros((self.layers[i].output_dim, 1)))
            prev_delta_increments_weights.append(np.full((self.layers[i].output_dim, self.layers[i].input_dim), delta_zero))
            prev_delta_increments_biases.append(np.full((self.layers[i].output_dim, 1), delta_zero))
            delta_min_weights.append(np.full((self.layers[i].output_dim, self.layers[i].input_dim), delta_min))
            delta_min_biases.append(np.full((self.layers[i].output_dim, 1), delta_min))
            delta_max_weights.append(np.full((self.layers[i].output_dim, self.layers[i].input_dim), delta_max))
            delta_max_biases.append(np.full((self.layers[i].output_dim, 1), delta_max))


        for _ in range(epoches):

            # resetting variables for each epoque
            train_error = 0
            val_error = 0

            outputs = []
            curr_derivates = []
            curr_deltas = []
            curr_delta_increments_weights = []
            curr_delta_increments_biases = []
            
            for i in range(0, len(self.layers), 2):
                curr_derivates.append(np.zeros((self.layers[i].output_dim, self.layers[i].input_dim)))
                curr_deltas.append(np.zeros((self.layers[i].output_dim, 1)))              

            # running over training set
            for i in range(len(train_X)):

                outputs.append(softmax(self.forward_propagation(train_X[i])))
                deltas = self.back_propagation(outputs[i], train_Y[i])
                derivates = self.compute_derivates(deltas)
               
                for i in range(len(self.layers)//2):
                    curr_derivates[i] = np.add(curr_derivates[i], derivates[i])
                    curr_deltas[i] = np.add(curr_deltas[i], deltas[i])

            #update weights
            for i in range(len(self.layers)//2):
                curr_delta_increments_weights.append(np.where(curr_derivates[i] * prev_derivates[i] == 0, \
                    prev_delta_increments_weights[i], \
                    np.where(curr_derivates[i] * prev_derivates[i] > 0, \
                    np.minimum(eta_plus * prev_delta_increments_weights[i], delta_max_weights[i]), \
                    np.maximum(eta_minus * prev_delta_increments_weights[i], delta_min_weights[i]))))
                
                curr_delta_increments_biases.append(np.where(curr_deltas[i] * prev_deltas[i] == 0, \
                    prev_delta_increments_biases[i], \
                    np.where(curr_deltas[i] * prev_deltas[i] > 0, \
                    np.minimum(eta_plus * prev_delta_increments_biases[i], delta_max_biases[i]), \
                    np.maximum(eta_minus * prev_delta_increments_biases[i], delta_min_biases[i]))))
                 
            self.update_weights_rprop(curr_derivates, curr_deltas, curr_delta_increments_weights, curr_delta_increments_biases)

            #update previous derivates and increments
            prev_delta_increments_weights = curr_delta_increments_weights
            prev_delta_increments_biases = curr_delta_increments_biases
            prev_derivates = curr_derivates
            prev_deltas = curr_deltas

            # computing error on training and validation set
            for i in range(len(train_X)):
                train_error += self.loss(outputs[i], train_Y[i])
            train_errors_epoches.append(train_error)

            for i in range(len(val_X)):
                val_error += self.loss(softmax(self.forward_propagation(val_X[i])), val_Y[i])
            if len(val_errors_epoches) == 0:
                best_fitting_network = self
            elif val_error < val_errors_epoches[-1]:
                best_fitting_network = copy.deepcopy(self)
            val_errors_epoches.append(val_error)

        train_errors_epoches = np.array(train_errors_epoches)
        val_errors_epoches = np.array(val_errors_epoches)

        return best_fitting_network, train_errors_epoches, val_errors_epoches
    

    def accuracy(self, test_X, test_Y):
        correct_answers = 0
        for i in range(len(test_X)):
            output = self.forward_propagation(test_X[i])
            output = softmax(output)
            if np.argmax(output) == np.argmax(test_Y[i]):
                correct_answers += 1
        return correct_answers / len(test_X)
    
    
'''
    Utility Functions 
'''

def clean_data(x, img_scale):
    data = np.empty((x.shape[0], img_scale * img_scale, 1))
    for i in range(len(x)):
        img = x[i] / 255.0
        img = cv2.resize(img, (img_scale, img_scale))
        data[i] = (img.reshape((img_scale * img_scale, 1)))
    return data


def encode_labels(y):
    labels = np.empty((y.shape[0], 10, 1))
    for i in range(len(y)):
        label = np.zeros((10, 1))
        label[y[i]] = 1
        labels[i] = label
    return labels


def create_network(input_dim, output_dim, hidden_layers_num, neurons_num, act_fun, act_fun_prime, loss, loss_prime):

    NN = NeuralNetwork(loss, loss_prime)

    layers = []

    layers.append(ConnectionLayer(input_dim, neurons_num[0]))
    layers.append(ActivationLayer(neurons_num[0], act_fun[0], act_fun_prime[0]))

    for i in range(1, hidden_layers_num):
        layers.append(ConnectionLayer(neurons_num[i-1], neurons_num[i]))
        layers.append(ActivationLayer(neurons_num[i], act_fun[i], act_fun_prime[i]))

    layers.append(ConnectionLayer(neurons_num[-1], output_dim))
    layers.append(ActivationLayer(output_dim, act_fun[-1], act_fun_prime[-1]))

    for layer in layers:
        NN.add_layer(layer)

    return NN
    

'''
    ################
    ##### MAIN #####
    ################
'''
def main():

    #load dataset
    (train_X, train_y), (test_X, test_y) = mnist.load_data()


    #prepare training & test data
    img_scale = 14
    train_data = clean_data(train_X, img_scale)
    test_data = clean_data(test_X, img_scale)

    
    #encode labels in one-hot encoding
    train_labels = encode_labels(train_y)
    test_labels = encode_labels(test_y)


    #shuffle data 
    train_data, train_labels = shuffle(train_data, train_labels)

    #divide data into training & validation set
    ratio = 0.8
    train_len = int(ratio * len(train_data))

    training_set = train_data[0:train_len]
    training_labels = train_labels[0:train_len]

    validation_set = train_data[train_len:]
    validation_labels = train_labels[train_len:]


    # NN = create_network(img_scale * img_scale, 10, 2, [40, 20], [relu, relu, sigmoid], [relu_prime, relu_prime, sigmoid_prime], cross_entropy_softmax, cross_entropy_softmax_prime)
    
    
    #inizialize Neural Network
    NN = NeuralNetwork(cross_entropy_softmax, cross_entropy_softmax_prime)
    layer1 = ConnectionLayer(img_scale * img_scale, 20)
    layer2 = ActivationLayer(20, sigmoid, sigmoid_prime)
    layer3 = ConnectionLayer(20, 10)
    layer4 = ActivationLayer(10, sigmoid, sigmoid_prime)

    NN.add_layer(layer1)
    NN.add_layer(layer2)
    NN.add_layer(layer3)
    NN.add_layer(layer4)
    

    #start learning (rprop)
    epoches = 500
    eta_plus = 1.2
    eta_minus = 0.5
    delta_zero = 0.5
    delta_min = 0
    delta_max = 50
    best_network, train_error, val_error = NN.learn_rprop(training_set[0:10000], training_labels[0:10000], validation_set[0:2000], validation_labels[0:2000], \
                                                          epoches, eta_plus, eta_minus, delta_zero, delta_max, delta_min)

    '''
    #start learning (gradient descent)
    epoches = 500
    lr = 0.01
    best_network, train_error, val_error = NN.learn(training_set[0:30000], training_labels[0:30000], validation_set, validation_labels, epoches, lr, mod='B')
    '''
    
    #test with random samples on the best performing network
    np.set_printoptions(suppress=True, precision=2)
    print("PREDICTED:", best_network.forward_propagation(validation_set[154]).T)
    print("LABEL:", validation_labels[154].T)
    print("PREDICTED:", best_network.forward_propagation(validation_set[1234]).T)
    print("LABEL:", validation_labels[1234].T)
    print("PREDICTED:", best_network.forward_propagation(validation_set[3254]).T)
    print("LABEL:", validation_labels[3254].T)
    print("PREDICTED:", best_network.forward_propagation(validation_set[1]).T)
    print("LABEL:", validation_labels[1].T)
    

    x_plot = np.arange(epoches)
    plt.xlabel('Epoches')
    plt.ylabel('Error')
    plt.plot(x_plot, train_error, color='g', label='Training Error')
    plt.plot(x_plot, val_error, color='y', label='Validation Error')

    plt.legend()
    plt.show()

    print("Accuracy:", best_network.accuracy(test_data, test_labels))

    

if __name__=="__main__":
    main()
