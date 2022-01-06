from sklearn.neural_network import MLPRegressor

from src.Data import Data
from src.configurations import Configuration
from src.models.Model import Model


class MultiLayerPerceptronRegressorModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.random_forest_features
        model = MLPRegressor(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=300, # upped max iterations
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10,
            max_fun=15000,
        )
        Model.__init__(self, config, training_data, features, model)
