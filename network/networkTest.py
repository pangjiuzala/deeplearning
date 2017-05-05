'''
@author: liuxing
'''
# # Test network
# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# import network
# net = network.Network([784,  10])
# net.SGD(list(training_data), 5, 10, 5.0, test_data=list(test_data))

# Test network2
from com.tensorflowTest.network import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
from com.tensorflowTest.network import network2
net = network2.Network([784, 30, 10])
net = network2.Network([784, 30, 30, 10])
net.SGD(list(training_data), 30, 10, 0.1, lmbda=5.0,
        evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)
# Test network3
# import network3
# from network3 import Network
# from network3 import FullyConnectedLayer, SoftmaxLayer
# training_data, validation_data, test_data = network3.load_data_shared()
# mini_batch_size = 10
# net = Network([FullyConnectedLayer(n_in=784, n_out=100),
# SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
# net.SGD(list(training_data), 60, mini_batch_size, 0.1,
# list(validation_data), list(test_data))