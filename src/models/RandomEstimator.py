import random

from src.PredictionResult import PredictionResult
from src.configurations import Configuration
from src.Data import Data


class RandomEstimator:
    def __init__(self, config: Configuration):
        self.configuration = config

    def predict_random_numbers_for(self, data: Data):
        number_of_docks = data.get_num_of_docks_for_each_station()
        predictions = []
        for example in number_of_docks:
            predictions.append(random.randint(0, example))

        result = PredictionResult(list(range(0, len(predictions))))
        result.add_predictions(predictions)
        result.add_true_values(data.get_true_values())
        return result
