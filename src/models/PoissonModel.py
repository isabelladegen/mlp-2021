import math
from sklearn import linear_model
import numpy as np

from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier


class PoissonModel:
    def __init__(self, config: Configuration, training_data: Data):
        self.configuration = config
        self.raw_training_data = training_data
        self.X = self.raw_training_data.get_feature_matrix_x_for(config.poisson_features,
                                                                 config.features_data_type)
        self.y = self.raw_training_data.get_y()  # labelled output

        self.model = linear_model.PoissonRegressor(
            alpha=config.poisson_alpha,
            fit_intercept=config.poisson_fit_intercept,
            max_iter=config.poisson_max_iter,
            tol=config.poisson_tol,
            verbose=config.poisson_verbose,
            warm_start=config.poisson_warm_start
        )

    def fit(self) -> []:
        self.model.fit(self.X, self.y)
        return self.model.get_params()

    def predict(self, val_data: Data) -> PredictionResult:
        feature_matrix_x = val_data.get_feature_matrix_x_for(self.configuration.poisson_features,
                                                             self.configuration.features_data_type)
        predictions = self.model.predict(feature_matrix_x)

        rounded_predictions = []
        for prediction in predictions:
            rounded_predictions.append(round_half_up(prediction))

        result = PredictionResult()
        result.add_predictions(rounded_predictions)
        y = val_data.get_y()
        if len(y[~np.isnan(y)]) == len(y):
            result.add_true_values(y)
        else:
            print("Didn't  add true values to results as some or all bikes values were nan")
        return result

    def features(self):
        return self.configuration.poisson_features
