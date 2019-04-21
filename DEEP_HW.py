import network3
from network3 import Network, ReLU
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()

mini_batch_size = 10
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),filter_shape=(20, 1, 5, 5),poolsize=(2, 2), activation_fn=ReLU),FullyConnectedLayer(
n_in=20*12*12, n_out=100, activation_fn=ReLU, p_dropout=0.0),SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)], mini_batch_size)

n_epochs, eta = 30, 0.1
net.SGD(training_data, n_epochs, mini_batch_size, eta,validation_data, test_data)