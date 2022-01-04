import math
from sklearn import linear_model

from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration
from src.models.Model import Model


class PoissonModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.poisson_features
        model = linear_model.PoissonRegressor(
            alpha=config.poisson_alpha,
            fit_intercept=config.poisson_fit_intercept,
            max_iter=config.poisson_max_iter,
            tol=config.poisson_tol,
            verbose=config.poisson_verbose,
            warm_start=config.poisson_warm_start
        )
        Model.__init__(self, config, training_data, features, model)
