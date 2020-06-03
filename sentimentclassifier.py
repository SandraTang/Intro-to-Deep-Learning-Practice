import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imbd

# IMDB Dataset loading
# pkl = pickle = byte stream
# want 10000 words, test 10% of them
train, test, _ = imdb.load_data(path = 'imdb.pkl', n_words = 10000, valid_portion = 0.1)

# split reviews and labels into X and Y values
# train helps fit weights, validation prevents overfitting
trainX, trainY = train
textX, textY = test

# data preprocessing
# sequence padding (put zeroes around array)
trainX = pad_sequences(trainX, maxlen = 100, value = 0.)
testX = pad_sequences(testX, maxlen = 100, value = 0.)

# convert labels to binary vectors
trainY = to_categorical(trainY, nb_classes = 2)
testY = to_categorial(testY, nb_classes = 2)

# network building (layers)
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim = 10000, output_dim = 128)
# dropout randomly turn on/off random pathways to prevent overfitting
net = tflearn.lstm(net, 128, dropout = 0.8)
# fully connected layer
# computationally cheap way to learn non-linear combinations
# soft max takes in vector of values
# squash into vector of output probabilities (0 ot 1) that sum to 1
net = tflearn.fully_connected(net, 2, activation = 'softmax')
# last layer, regression layer
# adam performs gradient descent
# categorical crossentropy defines difference btw expected and actual output
net = tflearn.regression(net, optimizer = 'adam', learning_rate = 0.0001, loss = 'categorical_crossentropy')

# training
model = tflearn.DNN(net, tensorboard_verbose = 0)
# model.fit(trainX, trainY, validation_set(textX, trainY), show_metric = True, batch_size = 32)
model.fit(trainX, trainY, valid_portion(testX, testY), show_metric=True, batch_size=32)