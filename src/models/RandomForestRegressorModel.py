from sklearn.ensemble import RandomForestRegressor
from src.Data import Data
from src.configurations import Configuration
from src.models.Model import Model


class RandomForestRegressorModel(Model):
    def __init__(self, config: Configuration, training_data: Data):
        features = config.random_forest_features
        model = RandomForestRegressor(
            n_estimators=100,
            criterion="absolute_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )
        Model.__init__(self, config, training_data, features, model)
