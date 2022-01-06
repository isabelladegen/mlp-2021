from sklearn.linear_model import LinearRegression

from src.Data import Data
from src.configurations import Configuration
from src.models.Model import Model


class LinearRegressionModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.random_forest_features
        model = LinearRegression(
        )
        Model.__init__(self, config, training_data, features, model)
