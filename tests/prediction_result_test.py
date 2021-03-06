import pandas as pd
from hamcrest import *
import os
import glob

from src.PredictionResult import PredictionResult, ResultsColumns
from src.configurations import TestConfiguration


def test_calculates_mean_average_error_for_predictions():
    predictions = [1, 2, 3, 4, 5, 6]
    true_values = [1, 2, 3, 4, 5, 6]
    result = PredictionResult(list(range(0, len(predictions))))
    result.add_predictions(predictions)
    result.add_true_values(true_values)

    mae = result.mean_absolute_error()

    assert_that(mae, equal_to(0))


def test_returns_mean_absolut_error_for_actual_errors():
    predictions = [1, 2]
    true_values = [0, 2]
    result = PredictionResult(list(range(0, len(predictions))))
    result.add_predictions(predictions)
    result.add_true_values(true_values)

    mae = result.mean_absolute_error()

    assert_that(mae, equal_to(0.5))


def test_creates_a_csv_file_for_the_random_predictions():
    predictions = [10, 2, 3, 4, 9, 7, 4, 9, 10]
    ids = list(range(0, len(predictions)))
    result = PredictionResult(ids)
    result.add_predictions(predictions)

    filename = result.write_to_csv('', TestConfiguration())

    df = pd.read_csv(filename)
    assert_that(df.shape, equal_to((len(predictions), 2)))
    assert_that(list(df.Id.values), equal_to(ids))
    assert_that(list(df.bikes.values), equal_to(predictions))

    cleanup_files()


def test_return_df_of_id_true_value_and_predictions():
    predictions = [1, 2, 3, 10]
    true_values = [0, 2, 3, 5]
    ids = list(range(2, len(predictions) + 2))
    result = PredictionResult(ids)
    result.add_predictions(predictions)
    result.add_true_values(true_values)

    df = result.results_df
    assert_that(df.shape, equal_to((len(predictions), 3)))
    assert_that(list(df.Id), equal_to(ids))
    assert_that(list(df.predictions), equal_to(predictions))
    assert_that(list(df['true values']), equal_to(true_values))


def test_can_create_prediction_results_from_multiple_prediction_results():
    predictions1 = [10, 11, 12]
    ids = [100, 110, 112]
    result1 = PredictionResult(ids)
    result1.add_predictions(predictions1)

    predictions2 = [1, 2, 3]
    ids = [1, 20, 30]
    result2 = PredictionResult(ids)
    result2.add_predictions(predictions2)

    predictions3 = [30, 31, 32]
    ids = [1, 40, 50]
    result3 = PredictionResult(ids)
    result3.add_predictions(predictions3)

    one_result = PredictionResult([result1, result2, result3])

    df = one_result.results_df
    assert_that(df.shape[0], equal_to(9))
    assert_that(list(df[ResultsColumns.id.value]), equal_to([1, 1, 20, 30, 40, 50, 100, 110, 112]))
    assert_that(list(df[ResultsColumns.predictions.value]), equal_to([1, 30, 2, 3, 31, 32, 10, 11, 12]))


def test_can_calculate_mean_absolute_error_per_station():
    predictions = [33, 4, 10]
    true_values = [33, 4, 20]
    stations = [1, 1, 2]

    result = PredictionResult(range(0, len(predictions)))
    result.add_predictions(predictions)
    result.add_true_values(true_values)
    result.add_stations(stations)

    mae_per_station = result.mean_absolute_error_per_station()

    assert_that(mae_per_station[1], equal_to(0.0))
    assert_that(mae_per_station[2], equal_to(10.0))


def test_calculates_mean_prediction_result_of_two_predictions():
    predictions1 = [10, 4, 12]
    ids1 = [1, 2, 3]
    stations1 = [201, 202, 203]
    true_values1 = [10, 10, 10]
    result1 = PredictionResult(ids1)
    result1.add_predictions(predictions1)
    result1.add_stations(stations1)
    result1.add_true_values(true_values1)

    predictions2 = [8, 2, 3]
    ids2 = [1, 2, 4]
    stations2 = [201, 202, 203]
    true_values2 = [10, 10, 10]
    result2 = PredictionResult(ids2)
    result2.add_predictions(predictions2)
    result2.add_stations(stations2)
    result2.add_true_values(true_values2)

    combined_result = PredictionResult([result1, result2], calculate_mean=True)

    df = combined_result.results_df
    assert_that(df.shape, equal_to((4, 6)))
    assert_that(list(df[ResultsColumns.predictions.value]), equal_to([9.0, 3.0, 12.0, 3.0]))
    assert_that(combined_result.mean_absolute_error(), equal_to(17 / 4))


def cleanup_files():
    config = TestConfiguration()
    for filename in glob.glob(os.path.join(config.write_predictions_to_path + config.write_results_start_name + "*")):
        os.remove(filename)
