from hamcrest import *

from src.PredictionResult import ResultsColumns
from src.configurations import TestConfiguration
from src.models.RandomForestRegressorModel import RandomForestRegressorModel
from utils.station_data_builder import StationDataBuilder
from utils.test_data_builder import TestDataBuilder


def test_uses_configuration_to_configure_the_model():
    configuration = TestConfiguration()
    configuration.random_forest_n_estimators: int = 1
    configuration.random_forest_criterion: str = "squared_error"
    configuration.random_forest_max_depth: int = 1
    configuration.random_forest_min_samples_split: int = 4
    configuration.random_forest_min_samples_leaf: int = 3
    configuration.random_forest_min_weight_fraction_leaf: float = 0.5
    configuration.random_forest_max_features: str = "sqrt"
    configuration.random_forest_max_leaf_nodes: int = 50
    configuration.random_forest_min_impurity_decrease: float = 0.4
    configuration.random_forest_bootstrap: bool = True
    configuration.random_forest_oob_score: bool = True
    configuration.random_forest_n_jobs: int = 2
    configuration.random_forest_random_state: int = 1
    configuration.random_forest_verbose: int = 1
    configuration.random_forest_warm_start: bool = True
    configuration.random_forest_ccp_alpha: float = 0.8  # 0 no pruning

    station1 = StationDataBuilder()
    station1.bikes_3h_ago = 10
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.bikes_3h_ago = 5
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.bikes_3h_ago = 0
    station3.bikes = 5
    data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()

    model = RandomForestRegressorModel(configuration, data)
    params = model.fit()

    assert_that(params['n_estimators'], equal_to(configuration.random_forest_n_estimators))
    assert_that(params['criterion'], equal_to(configuration.random_forest_criterion))
    assert_that(params['max_depth'], equal_to(configuration.random_forest_max_depth))
    assert_that(params['min_samples_split'], equal_to(configuration.random_forest_min_samples_split))
    assert_that(params['min_samples_leaf'], equal_to(configuration.random_forest_min_samples_leaf))
    assert_that(params['min_weight_fraction_leaf'], equal_to(configuration.random_forest_min_weight_fraction_leaf))
    assert_that(params['max_features'], equal_to(configuration.random_forest_max_features))
    assert_that(params['max_leaf_nodes'], equal_to(configuration.random_forest_max_leaf_nodes))
    assert_that(params['min_impurity_decrease'], equal_to(configuration.random_forest_min_impurity_decrease))
    assert_that(params['bootstrap'], equal_to(configuration.random_forest_bootstrap))
    assert_that(params['oob_score'], equal_to(configuration.random_forest_oob_score))
    assert_that(params['n_jobs'], equal_to(configuration.random_forest_n_jobs))
    assert_that(params['random_state'], equal_to(configuration.random_forest_random_state))
    assert_that(params['verbose'], equal_to(configuration.random_forest_verbose))
    assert_that(params['warm_start'], equal_to(configuration.random_forest_warm_start))
    assert_that(params['ccp_alpha'], equal_to(configuration.random_forest_ccp_alpha))
    assert_that(params['max_samples'], equal_to(configuration.random_forest_max_samples))


def test_fits_the_model_and_predicts_bikes():
    configuration = TestConfiguration()

    # training data
    station1 = StationDataBuilder()
    station1.bikes_3h_ago = 10
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.bikes_3h_ago = 5
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.bikes_3h_ago = 0
    station3.bikes = 5

    # testing data
    station_test1 = StationDataBuilder()
    station_test1.bikes_3h_ago = 10
    station_test1.bikes = 0
    station_test2 = StationDataBuilder()
    station_test2.bikes_3h_ago = 5
    station_test2.bikes = 5

    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()
    testing_data = TestDataBuilder(no_nan_in_bikes=False).with_stations([station_test1, station_test2]).build()

    model = RandomForestRegressorModel(configuration, training_data)
    model.fit()
    prediction_result = model.predict(testing_data)

    result_df = prediction_result.results_df
    true_values = list(result_df[ResultsColumns.true_values.value])
    predictions = list(result_df[ResultsColumns.predictions.value])

    assert_that(result_df.shape[0], equal_to(2))
    assert_that(true_values[0], equal_to(station_test1.bikes))
    assert_that(true_values[1], equal_to(station_test2.bikes))
    assert_that(predictions[0], not_none())
    assert_that(predictions[1], not_none())
