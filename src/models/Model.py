import math

from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration


class Model:
    def __init__(self, config: Configuration, training_data: Data, features, model):
        self.configuration = config
        self.raw_training_data = training_data
        self.features = features
        self.X = self.raw_training_data.get_feature_matrix_x_for(self.features,
                                                                 config.features_data_type)
        self.y = self.raw_training_data.get_y()  # labelled output
        self.model = model

    def fit(self) -> []:
        self.model.fit(self.X, self.y)
        return self.model.get_params()

    def predict(self, data: Data) -> PredictionResult:
        feature_matrix_x = data.get_feature_matrix_x_for(self.configuration.poisson_features,
                                                         self.configuration.features_data_type)
        predictions = self.model.predict(feature_matrix_x)

        rounded_predictions = []
        for prediction in predictions:
            rounded_predictions.append(self.__round_half_up(prediction))

        y = data.get_y()
        result = PredictionResult(data.get_ids())
        result.add_predictions(rounded_predictions)
        result.add_true_values(y)
        result.add_stations(data.get_stations())
        return result

    @staticmethod
    def __round_half_up(n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier + 0.5) / multiplier
