import numpy as np
from sklearn.metrics import mean_absolute_error

from src.Data import Data, PretrainedLinearModels
from src.PredictionResult import PredictionResult
from src.configurations import Configuration
from src.models.Model import Model


class BestPreTrainedModelForAStation:
    def __init__(self, config: Configuration, training_data: Data, trained_models: PretrainedLinearModels):
        self.training_data = training_data
        self.data_type = config.features_data_type
        self.y = self.training_data.get_y()  # labelled output
        self.trained_models = trained_models
        self.stations = self.training_data.get_stations()
        self.best_mae = 100000
        self.best_model_index = None
        self.features = trained_models.features
        self.config = config

    def fit(self) -> str:
        feature_matrix_x = self.training_data.get_feature_matrix_x_for(self.features, self.config.features_data_type)
        # inserts a column of 1 as the first column for intercept of trained model
        predictions_for_each_model = self.__calculate_predictions(feature_matrix_x, self.trained_models.weights_matrix)

        # calculate all maes
        maes = []
        for prediction in predictions_for_each_model.T:
            maes.append(mean_absolute_error(self.y, prediction))

        maes_np = np.array(maes)  # to np array

        # find min mae
        self.best_mae = np.amin(maes_np)
        self.best_model_index = np.where(maes_np == self.best_mae)[0][0]

    def predict(self, data: Data) -> PredictionResult:
        feature_matrix_x = data.get_feature_matrix_x_for(self.features, self.data_type)
        weights_for_best_model = self.trained_models.weights_matrix[:, self.best_model_index]
        predictions = self.__calculate_predictions(feature_matrix_x, weights_for_best_model)

        # return as PredictionResult
        result = PredictionResult(data.get_ids())
        result.add_predictions(list(predictions))
        result.add_true_values(list(data.get_y()))
        result.add_stations(data.get_stations())
        return result

    def __calculate_predictions(self, feature_matrix_x, weights_vector_or_matrix):
        x_with_intercept = np.insert(feature_matrix_x, 0, 1, axis=1)
        # calculate dot product, which is of shape (samples x models)
        # -> each column is the predicted y for that model
        predictions = x_with_intercept.dot(weights_vector_or_matrix)

        if self.config.intermediate_rounding:  # optional rounding
            with np.nditer(predictions, op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = Model.round_half_up(x)
        # replace negative predictions with 0 (station cannot have negative bikes)
        predictions = np.where(predictions < 0, 0, predictions)
        return predictions
