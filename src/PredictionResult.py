from sklearn.metrics import mean_absolute_error


class PredictionResult:
    def __init__(self):
        self.predictions = []
        self.true_values = []

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def add_predictions(self, predictions):
        self.predictions = predictions

    def add_true_values(self, true_values):
        self.true_values = true_values

    def mean_absolute_error(self):
        assert len(self.predictions) == len(self.true_values)
        return mean_absolute_error(self.true_values, self.predictions)
