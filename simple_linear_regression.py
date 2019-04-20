import numpy as np
from sklearn.linear_model import LinearRegression
from feeders import BatchFeeder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import HuberRegressor

class SimpleLinearRegression:
    def __init__(self, max_value):
        self.batchFeeder = BatchFeeder()
        xtrain, ytrain, xtest, ytest = self.make_data(max_value)
        self._x_train = self.set_x(xtrain)
        self._x_test = self.set_x(xtest)
        self._y_train = ytrain
        self._y_test = ytest

        self._reg = None

    def make_data(self, max_value):
        xtrain = np.empty([2, 2])
        ytrain = np.empty([2, 2])
        xtest = np.empty([2, 2])
        ytest = np.array([2, 2])
        input_session = np.empty([2, 2])

        for x in range(0, max_value):
            input_gap_batch, input_session, _, \
            output_gap_batch, _ = \
                self.batchFeeder.create_batch()
            if x == 0:
                xtrain = self.set_x(input_gap_batch)
                ytrain = self.regulate_y(output_gap_batch)
                session_train = self.set_x(input_session)
            elif x == max_value - 1:
                xtest = self.set_x(input_gap_batch)
                stest = self.set_x(input_session)
                x_test_final = np.concatenate((xtest, np.squeeze(stest)), axis=1)
                ytest = self.regulate_y(output_gap_batch)
                x_train_final = np.concatenate((xtrain, np.squeeze(session_train)), axis=1)
            else:
                xtrain = np.concatenate((xtrain, input_gap_batch), axis=0)
                session_train = np.concatenate((session_train, input_session), axis=0)
                ytrain = np.concatenate((ytrain, self.regulate_y(output_gap_batch))
                                        , axis=0)

        return x_train_final, ytrain, x_test_final, ytest

    def set_x(self, data):
        return data

    def regulate_y(self, data):
        shape = np.shape(data)
        index = int(shape[1])
        return data[:, index - 1]

    def learn(self):
        self._reg = LinearRegression().fit(self._x_train, self._y_train)

    def predict(self):
        self._predict = self._reg.predict(self._x_test)
        return self._predict

    def calculate_mae(self):
        mae = mean_absolute_error(self._predict, self._y_test)
        print(mae / 60)
        return mae

    def run(self):
        self.learn()
        self.predict()
        self.calculate_mae()


class SimplerFunction(SimpleLinearRegression):




    def make_data(self, max_value):
        xtrain = np.empty([2, 2])
        ytrain = np.empty([2, 2])
        xtest = np.empty([2, 2])
        ytest = np.array([2, 2])
        input_session = np.empty([2, 2])

        for x in range(0, max_value):
            input_gap_batch, input_session, _, \
            output_gap_batch, _ = \
                self.batchFeeder.create_batch()
            if x == 0:
                xtrain = self.set_x(input_gap_batch)
                ytrain = self.regulate_y(output_gap_batch)
                session_train = self.set_x(input_session)
            elif x == max_value - 1:
                xtest = self.set_x(input_gap_batch)
                stest = self.set_x(input_session)
                x_test_final = np.concatenate((xtest, np.squeeze(stest)), axis=1)
                ytest = self.regulate_y(output_gap_batch)
                x_train_final = np.concatenate((xtrain, np.squeeze(session_train)), axis=1)
            else:
                xtrain = np.concatenate((xtrain, input_gap_batch), axis=0)
                session_train = np.concatenate((session_train, input_session), axis=0)
                ytrain = np.concatenate((ytrain, self.regulate_y(output_gap_batch))
                                        , axis=0)

        return xtrain, ytrain, xtest, ytest


class SimpleHubber(SimpleLinearRegression):
    def learn(self):
        self._reg = HuberRegressor().fit(self._x_train, self._y_train)

    def predict(self):
        self._predict = self._reg.predict(self._x_test)
        return self._predict

class SimplerHubber(SimplerFunction):
    def learn(self):
        self._reg = HuberRegressor().fit(self._x_train, self._y_train)

    def predict(self):
        self._predict = self._reg.predict(self._x_test)
        return self._predict




if __name__ == "__main__":
    simpleLinearRegression = SimpleHubber(1000)
    simpleLinearRegression.run()
    simpleLinearRegression = SimplerHubber(1000)
    simpleLinearRegression.run()
