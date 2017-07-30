import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L

WIDTH = 1230
HEIGHT = 1230

class BaseModel(object):

    def __init__(self, lr=0.001, momentum=0.9, epochs=10, batch_size=32):
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size

        # Theano variables
        self.X = T.tensor3('X', 'float32')
        self.Y = T.tensor3('Y', 'float32')

    def fit(self, dataset):
        self.model = self.build_cnn()

        model_params = L.get_all_params(self.model)

        # Training function
        output = L.get_output(self.model)
        loss = lasagne.objectives.squared_error(output, self.Y)
        loss = loss.mean()
        updates = lasagne.updates.adam(loss, model_params, learning_rate=self.lr)

        train_fun = theano.function([self.X, self.Y], loss, updates=updates)
        


        for e in xrange(self.epochs):
            for inp, tar in dataset.iter_batch(self.batch_size):
                batch_loss = train_fun(inp, tar)


    def forward(self, dataset, batch_size):
        inp, tar = dataset.iter_batch(batch_size).next()
        self.model = self.build_cnn()

        output = L.get_output(self.model)
        process = theano.function([self.X], output)

        return process(inp)




    def build_cnn(self):

        # Nonlinearity
        activation = lasagne.nonlinearities.elu

        network = L.InputLayer(input_var=self.X, shape=(None, 3, WIDTH, HEIGHT))

        # First block
        network = L.Conv2DLayer(network, num_filters=32, filter_size=(9, 9),
                                pad='same', nonlinearity=activation)
        network = L.Conv2DLayer(network, num_filters=32, filter_size=(9, 9),
                                pad='same', nonlinearity=activation)
        network = L.Conv2DLayer(network, num_filters=32, filter_size=(9, 9),
                                pad='same', nonlinearity=activation)

        # Second block
        network = L.Conv2DLayer(network, num_filters=32, filter_size=(9, 9),
                                pad='same', nonlinearity=activation)
        network = L.Conv2DLayer(network, num_filters=32, filter_size=(9, 9),
                                pad='same', nonlinearity=activation)
        network = L.Conv2DLayer(network, num_filters=32, filter_size=(9, 9),
                                pad='same', nonlinearity=activation)

        return network