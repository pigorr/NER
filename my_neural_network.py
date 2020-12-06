def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


class Neuron:
    def __init__(self, trainrate=0.1):
        self.weights = np.array([np.random.normal(), np.random.normal()])
        self.bias = np.random.normal()
        self.trainrate = trainrate

    def think(self, input):
        return sigmoid(self.weights[0] * input[0] + self.weights[1] * input[1] + self.bias)

    def train(self, input, error):
        pred_out = self.think(input)

        self.weights[0] -= self.trainrate * error * input[0] * sigmoid_derivative(pred_out)
        self.weights[1] -= self.trainrate * error * input[1] * sigmoid_derivative(pred_out)
        self.bias -= error * sigmoid_derivative(pred_out)


class NeuroNetwork:
    def __init__(self):
        self.h1 = Neuron()
        self.h2 = Neuron()
        self.o = Neuron()

    def think(self, input):
        h1_think_result = self.h1.think(input)
        h2_think_result = self.h2.think(input)
        return self.o.think([h1_think_result,
                             h2_think_result])

    def train(self, input, out):
        out_pred = self.think(input)
        h1_out_pred = self.h1.think(input)
        h2_out_pred = self.h2.think(input)

        d_error = -2 * (out - out_pred)

        self.h1.train(input, d_error * self.o.weights[0])
        self.h2.train(input, d_error * self.o.weights[1])
        self.o.train([h1_out_pred, h2_out_pred], d_error)


def generate_data():
    x = np.array([[4, 6],
                  [-2, -4],
                  [-10, -8],
                  [2, 4]])
    y = np.array([0,
                  1,
                  1,
                  0])
    return x, y


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def get_trained_model():
    x_train, y_train = generate_data()
    model = NeuroNetwork()
    epochs = 1000
    errors = []
    for epoch in range(epochs):
        for input, output in zip(x_train, y_train):
            model.train(input, output)
        if epoch % 5 == 0:
            output_preds = np.apply_along_axis(model.think, 1, data)
            erors.append(mse(output_preds, y_train))
            print("Epoch %d, error: %.3f" % (epoch, error))

    return model, errors


def test(model: NeuroNetwork):
    x_test = np.array([6, 10])
    y_pred = model.think(x_test)
    print("Model evaluation result: %.3f" % y_pred)


def evaluate():
    model, train_errors = get_trained_model()


evaluate()