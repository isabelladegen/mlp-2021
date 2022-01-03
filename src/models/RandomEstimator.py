import random

from src.PredictionResult import PredictionResult
from src.configurations import Configuration
from src.Data import Data


class RandomEstimator:
    def __init__(self, config: Configuration):
        self.configuration = config

    def predict_random_numbers_for(self, data: Data):
        result = PredictionResult()
        number_of_docks = data.get_num_of_docks_for_each_station()
        for example in number_of_docks:
            result.add_prediction(random.randint(0, example))
        true_values = data.get_true_values()
        if len(result.predictions) == len(
                true_values):  # only add true values if there is as many as predictions in the given data
            result.add_true_values(true_values)
        return result
