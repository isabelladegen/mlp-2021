from sklearn.neural_network import MLPRegressor

from src.Data import Data
from src.configurations import Configuration
from src.models.Model import Model


class MultiLayerPerceptronRegressorModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.random_forest_features
        model = MLPRegressor(
            hidden_layer_sizes=config.mlp_hidden_layer_sizes,
            activation=config.mlp_activation,
            solver=config.mlp_solver,
            alpha=config.mlp_alpha,
            batch_size=config.mlp_batch_size,
            learning_rate=config.mlp_learning_rate,
            learning_rate_init=config.mlp_learning_rate_init,
            power_t=config.mlp_power_t,
            max_iter=config.mlp_max_iter,
            shuffle=config.mlp_shuffle,
            random_state=config.mlp_random_state,
            tol=config.mlp_tol,
            verbose=False,
            warm_start=False,
            momentum=config.mlp_momentum,
            nesterovs_momentum=config.mlp_nesterovs_momentum,
            early_stopping=config.mlp_early_stopping,
            validation_fraction=config.mlp_validation_fraction,
            beta_1=config.mlp_beta_1,
            beta_2=config.mlp_beta_2,
            epsilon=config.mlp_epsilon,
            n_iter_no_change=config.mlp_n_iter_no_change,
            max_fun=config.mlp_max_fun)
        Model.__init__(self, config, training_data, features, model)
