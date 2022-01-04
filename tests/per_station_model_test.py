from hamcrest import *

from src.PredictionResult import ResultsColumns
from src.configurations import TestConfiguration
from src.models.PerStationModel import PerStationModel
from src.models.PoissonModel import PoissonModel
from utils.station_data_builder import StationDataBuilder
from utils.test_data_builder import TestDataBuilder


def test_fits_a_model_per_station():
    configuration = TestConfiguration()

    # training data
    station1 = StationDataBuilder()
    station1.station = 1
    station1.bikes_3h_ago = 10
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.station = 2
    station2.bikes_3h_ago = 5
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.station = 3
    station3.bikes_3h_ago = 0
    station3.bikes = 5
    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()

    models = PerStationModel(configuration, training_data, PoissonModel)
    models.fit()

    station1_model = models.get_model_for_station(station1.station)
    station2_model = models.get_model_for_station(station2.station)
    station3_model = models.get_model_for_station(station3.station)
    assert_that(len(models.model_per_station), equal_to(3))
    assert_that(station1_model, instance_of(PoissonModel))
    # check that each model got trained by accessing coef which is only set if training happened
    assert_that(station1_model.model.coef_[0], equal_to(0.0))
    assert_that(station2_model.model.coef_[0], equal_to(0.0))
    assert_that(station3_model.model.coef_[0], equal_to(0.0))


def test_predicts_bikes_for_each_station_model_excluding_stations_not_in_training_data():
    configuration = TestConfiguration()

    # training data
    station1 = StationDataBuilder()
    station1.station = 1
    station1.bikes_3h_ago = 10
    station1.bikes = 0
    station2 = StationDataBuilder()
    station2.station = 2
    station2.bikes_3h_ago = 5
    station2.bikes = 10
    station3 = StationDataBuilder()
    station3.station = 3
    station3.bikes_3h_ago = 0
    station3.bikes = 5
    training_data = TestDataBuilder(no_nan_in_bikes=True).with_stations([station1, station2, station3]).build()

    # testing data
    station_test1 = StationDataBuilder()
    station_test1.station = 1
    station_test1.bikes_3h_ago = 10
    station_test1.bikes = 0
    station_test2 = StationDataBuilder()
    station_test2.station = 2
    station_test2.bikes_3h_ago = 5
    station_test2.bikes = 5
    station_doesnt_exist_in_training_data = StationDataBuilder()
    station_doesnt_exist_in_training_data.station = 999
    station_doesnt_exist_in_training_data.bikes_3h_ago = 5
    station_doesnt_exist_in_training_data.bikes = 666
    testing_data = TestDataBuilder(no_nan_in_bikes=True).with_stations(
        [station_test1, station_test2, station_doesnt_exist_in_training_data]).build()

    models = PerStationModel(configuration, training_data, PoissonModel)
    models.fit()

    result = models.predict(testing_data)

    df = result.results_df
    predictions = df[ResultsColumns.predictions.value]
    true_values = df['true values']

    assert_that(df.shape, equal_to((2, 3)))
    assert_that(predictions[0], equal_to(0.0))
    assert_that(predictions[1], equal_to(10.0))
    assert_that(true_values[0], equal_to(0))
    assert_that(true_values[1], equal_to(5))
