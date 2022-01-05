import numpy
from hamcrest import *

from src.Data import Columns
from src.configurations import TestConfiguration
from src.models.PoissonModel import PoissonModel
from utils.station_data_builder import StationDataBuilder
from utils.test_data_builder import TestDataBuilder
from src.PredictionResult import ResultsColumns


def test_returns_parameters_used_for_training_and_set_in_config():
    configuration = TestConfiguration()
    configuration.poisson_warm_start = True
    configuration.poisson_max_iter = 2000
    configuration.poisson_alpha = 1.345
    configuration.poisson_tol = 0.1
    configuration.poisson_fit_intercept = False

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

    model = PoissonModel(configuration, data)
    params = model.fit()

    assert_that(params['alpha'], equal_to(configuration.poisson_alpha))
    assert_that(params['fit_intercept'], equal_to(configuration.poisson_fit_intercept))
    assert_that(params['max_iter'], equal_to(configuration.poisson_max_iter))
    assert_that(params['tol'], equal_to(configuration.poisson_tol))
    assert_that(params['verbose'], equal_to(configuration.poisson_verbose))
    assert_that(params['warm_start'], equal_to(configuration.poisson_warm_start))


def test_predicts_y_for_given_data_and_rounds_outcome():
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

    model = PoissonModel(configuration, training_data)
    model.fit()
    prediction_result = model.predict(testing_data)

    result_df = prediction_result.results_df
    true_values = list(result_df[ResultsColumns.true_values.value])
    predictions = list(result_df[ResultsColumns.predictions.value])

    assert_that(result_df.shape[0], equal_to(2))
    assert_that(true_values[0], equal_to(station_test1.bikes))
    assert_that(true_values[1], equal_to(station_test2.bikes))
    assert_that(predictions[0], equal_to(3.0))
    assert_that(predictions[1], equal_to(5.0))


def test_adds_empty_true_values_to_result_if_theres_no_prediction():
    configuration = TestConfiguration()

    # training data
    station1 = StationDataBuilder()
    station1.bikes_3h_ago = 10
    station1.bikes = 0

    # testing data
    station_test1 = StationDataBuilder()
    station_test1.bikes_3h_ago = 10
    station_test1.bikes = None

    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1]).build()
    testing_data = TestDataBuilder(no_nan_in_bikes=False).with_stations([station_test1]).build()

    model = PoissonModel(configuration, training_data)
    model.fit()
    prediction_result = model.predict(testing_data)

    result_df = prediction_result.results_df
    true_values = list(result_df[ResultsColumns.true_values.value])

    assert_that(result_df.shape[0], equal_to(1))
    assert_that(numpy.isnan(true_values[0]))


def test_uses_provided_features_for_training():
    configuration = TestConfiguration()
    configuration.poisson_features = [Columns.data_3h_ago.value, Columns.num_docks.value]

    # training data
    station1 = StationDataBuilder()
    station1.numDocks = 10
    station1.bikes_3h_ago = 10
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.bikes_3h_ago = 5
    station2.numDocks = 16
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.bikes_3h_ago = 0
    station3.numDocks = 20
    station3.bikes = 5

    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()

    model = PoissonModel(configuration, training_data)
    model.fit()
    features = model.features

    assert_that(len(features), equal_to(2))


def test_can_predict_y_for_more_than_one_feature():
    configuration = TestConfiguration()
    configuration.poisson_features = [Columns.data_3h_ago.value, Columns.num_docks.value]

    # training data
    station1 = StationDataBuilder()
    station1.bikes_3h_ago = 10
    station1.numDocks = 20
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.numDocks = 10
    station2.bikes_3h_ago = 5
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.bikes_3h_ago = 0
    station3.numDocks = 15
    station3.bikes = 5

    # testing data
    station_test1 = StationDataBuilder()
    station_test1.bikes_3h_ago = 10
    station_test1.numDocks = 20
    station_test1.bikes = 0
    station_test2 = StationDataBuilder()
    station_test2.bikes_3h_ago = 5
    station_test2.numDocks = 15
    station_test2.bikes = 5

    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()
    testing_data = TestDataBuilder(no_nan_in_bikes=False).with_stations([station_test1, station_test2]).build()

    model = PoissonModel(configuration, training_data)
    model.fit()
    prediction_result = model.predict(testing_data)
    print(prediction_result.mean_absolute_error())

    result_df = prediction_result.results_df
    true_values = list(result_df[ResultsColumns.true_values.value])
    predictions = list(result_df[ResultsColumns.predictions.value])

    assert_that(result_df.shape[0], equal_to(2))
    assert_that(true_values[0], equal_to(station_test1.bikes))
    assert_that(true_values[1], equal_to(station_test2.bikes))
    assert_that(predictions[0], equal_to(0.0))
    assert_that(predictions[1], equal_to(2.0))


def test_uses_same_data_preprocessing_for_training_and_prediction():
    configuration = TestConfiguration()
    configuration.poisson_features = [Columns.data_3h_ago.value]

    # training data
    station1 = StationDataBuilder()
    station1.bikes_3h_ago = 10
    station1.numDocks = 20
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.numDocks = 10
    station2.bikes_3h_ago = 5
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.bikes_3h_ago = 0
    station3.numDocks = 15
    station3.bikes = 5

    # testing data
    station_test1 = StationDataBuilder()
    station_test1.bikes_3h_ago = 10
    station_test1.numDocks = 20
    station_test1.bikes = 0
    station_test2 = StationDataBuilder()
    station_test2.bikes_3h_ago = 5
    station_test2.numDocks = 15
    station_test2.bikes = 5

    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()
    testing_data = TestDataBuilder(no_nan_in_bikes=False).with_stations([station_test1, station_test2]).build()

    model = PoissonModel(configuration, training_data)
    model.fit()
    # KEEP! modifying config in runtime should not change data preprocessing
    configuration.poisson_features = [Columns.data_3h_ago.value, Columns.num_docks.value]
    # this will throw an error if Model is not implemented properly
    prediction_result = model.predict(testing_data)

    result_df = prediction_result.results_df
    true_values = list(result_df[ResultsColumns.true_values.value])
    predictions = list(result_df[ResultsColumns.predictions.value])

    assert_that(result_df.shape[0], equal_to(2))
    assert_that(true_values[0], equal_to(station_test1.bikes))
    assert_that(true_values[1], equal_to(station_test2.bikes))
    assert_that(predictions[0], equal_to(3.0))
    assert_that(predictions[1], equal_to(5.0))
