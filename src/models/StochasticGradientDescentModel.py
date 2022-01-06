from sklearn.linear_model import SGDRegressor
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON

from src.Data import Data
from src.configurations import Configuration
from src.models.Model import Model


class StochasticGradientDescentModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.random_forest_features
        model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.0001,
            l1_ratio=0.15,
            fit_intercept=True,
            max_iter=1000,
            tol=1e-3,
            shuffle=True,
            verbose=0,
            epsilon=DEFAULT_EPSILON,
            random_state=None,
            learning_rate="invscaling",
            eta0=0.01,
            power_t=0.25,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            warm_start=False,
            average=False,
        )
        Model.__init__(self, config, training_data, features, model)
