from sklearn.ensemble import ExtraTreesRegressor

from src.Data import Data
from src.configurations import Configuration
from src.models.Model import Model


class ExtraTreesRegressorModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.random_forest_features
        model = ExtraTreesRegressor(
            n_estimators=config.random_forest_n_estimators,
            criterion=config.random_forest_criterion,
            max_depth=config.random_forest_max_depth,
            min_samples_split=config.random_forest_min_samples_split,
            min_samples_leaf=config.random_forest_min_samples_leaf,
            min_weight_fraction_leaf=config.random_forest_min_weight_fraction_leaf,
            max_features=config.random_forest_max_features,
            max_leaf_nodes=config.random_forest_max_leaf_nodes,
            min_impurity_decrease=config.random_forest_min_impurity_decrease,
            bootstrap=config.random_forest_bootstrap,
            oob_score=config.random_forest_oob_score,
            n_jobs=config.random_forest_n_jobs,
            random_state=config.random_forest_random_state,
            verbose=config.random_forest_verbose,
            warm_start=config.random_forest_warm_start,
            ccp_alpha=config.random_forest_ccp_alpha,
            max_samples=config.random_forest_max_samples
        )
        Model.__init__(self, config, training_data, features, model)
