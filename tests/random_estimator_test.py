from hamcrest import *

from src.PredictionResult import ResultsColumns
from src.models.RandomEstimator import *
from src.configurations import TestConfiguration
from utils.station_data_builder import StationDataBuilder
from utils.test_data_builder import *


def test_returns_a_random_prediction_for_each_test_data():
    configuration = TestConfiguration()

    station1 = StationDataBuilder()
    station1.station = 345
    station1.numDocks = 12
    station2 = StationDataBuilder()
    station2.station = 1
    station2.numDocks = 45
    station3 = StationDataBuilder()
    station3.station = 400
    station3.numDocks = 25

    test_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()

    model = RandomEstimator(configuration)
    result = model.predict_random_numbers_for(test_data)
    result_df = result.results_df
    predictions = result_df[ResultsColumns.predictions.value]

    assert_that(result_df.shape[0], equal_to(3))
    assert_that(0 <= predictions[0] <= station1.numDocks)
    assert_that(0 <= predictions[1] <= station2.numDocks)
    assert_that(0 <= predictions[2] <= station3.numDocks)


def test_adds_true_values_for_each_row():
    configuration = TestConfiguration()

    station1 = StationDataBuilder()
    station1.bikes = 400
    station2 = StationDataBuilder()
    station1.bikes = 5
    station3 = StationDataBuilder()
    station3.bikes = 20

    test_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()

    model = RandomEstimator(configuration)
    result = model.predict_random_numbers_for(test_data)
    result_df = result.results_df
    true_values = result_df[ResultsColumns.true_values.value]

    assert_that(result_df.shape[0], equal_to(3))
    assert_that(true_values[0], equal_to(station1.bikes))
    assert_that(true_values[1], equal_to(station2.bikes))
    assert_that(true_values[2], equal_to(station3.bikes))
