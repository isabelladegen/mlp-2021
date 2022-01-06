from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration


class BestTrainedModelForEachStation:
    def __init__(self, config: Configuration, training_data: Data, trained_models):
        self.raw_training_data = training_data
        self.data_type = config.features_data_type
        self.y = self.raw_training_data.get_y()  # labelled output
        self.trained_models = trained_models
        self.ids = self.raw_training_data.get_ids()
        self.stations = self.raw_training_data.get_stations()
        self.best_mae = 100000
        self.best_model = None

    def fit(self) -> str:
        best_mae = 100000
        best_model = None
        for trained_model in self.trained_models:
            features = trained_model.features()
            feature_matrix_x = self.raw_training_data.get_feature_matrix_x_for(features, self.data_type)
            weights_vector = trained_model.weights()
            result = self.__get_prediction_result(feature_matrix_x, weights_vector)

            current_mae = result.mean_absolute_error()
            # update model if a better one has been found
            if current_mae < best_mae:
                best_mae = current_mae
                best_model = trained_model

        self.best_mae = best_mae
        self.best_model = best_model
        return self.best_mae  # aka MAE for training data

    def predict(self, data: Data) -> PredictionResult:
        feature_matrix_x = data.get_feature_matrix_x_for(self.best_model.features(), self.data_type)
        result = self.__get_prediction_result(feature_matrix_x, self.best_model.weights())

        result.add_stations(data.get_stations())
        return result

    def __get_prediction_result(self, feature_matrix_x, weights_vector):
        predictions = feature_matrix_x * weights_vector  # maybe round?
        result = PredictionResult(self.ids)
        result.add_predictions(predictions)
        result.add_true_values(self.y)
        return result
