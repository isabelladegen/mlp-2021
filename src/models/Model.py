import math

from sklearn.preprocessing import StandardScaler

from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration


class Model:
    def __init__(self, config: Configuration, training_data: Data, features, model, feature_scaling=False):
        self.raw_training_data = training_data
        self.features = features
        self.data_type = config.features_data_type
        self.scaler = None
        if feature_scaling:
            self.scaler = StandardScaler()
        self.X = self.raw_training_data.get_feature_matrix_x_for(self.features, self.data_type)
        self.y = self.raw_training_data.get_y()  # labelled output
        self.model = model
        self.round = config.intermediate_rounding

    def fit(self) -> []:
        x = self.X
        if self.scaler:
            self.scaler.fit(self.X)
            x = self.scaler.transform(x)
        self.model.fit(x, self.y)
        return self.model.get_params()

    def predict(self, data: Data) -> PredictionResult:
        feature_matrix_x = data.get_feature_matrix_x_for(self.features, self.data_type)
        x = feature_matrix_x
        if self.scaler:
            x = self.scaler.transform(feature_matrix_x)
        predictions = self.model.predict(x)

        if self.round:
            predictions = self.round_predictions(predictions)

        y = data.get_y()
        result = PredictionResult(data.get_ids())
        result.add_predictions(predictions)
        result.add_true_values(y)
        result.add_stations(data.get_stations())
        return result

    def round_predictions(self, predictions):
        rounded_predictions = []
        for prediction in predictions:
            rounded_predictions.append(self.round_half_up(prediction))
        return rounded_predictions

    @staticmethod
    def round_half_up(n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier + 0.5) / multiplier
