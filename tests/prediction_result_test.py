import pandas as pd
from hamcrest import *
import os
import glob

from src.PredictionResult import PredictionResult
from src.configurations import TestConfiguration


def test_calculates_mean_average_error_for_predictions():
    predictions = [1, 2, 3, 4, 5, 6]
    true_values = [1, 2, 3, 4, 5, 6]
    result = PredictionResult()
    result.add_predictions(predictions)
    result.add_true_values(true_values)

    mae = result.mean_absolute_error()

    assert_that(mae, equal_to(0))


def test_returns_mean_absolut_error_for_actual_errors():
    predictions = [1, 2]
    true_values = [0, 2]
    result = PredictionResult()
    result.add_predictions(predictions)
    result.add_true_values(true_values)

    mae = result.mean_absolute_error()

    assert_that(mae, equal_to(0.5))


def test_creates_a_csv_file_for_the_random_predictions():
    predictions = [10, 2, 3, 4, 9, 7, 4, 9, 10]
    result = PredictionResult()
    result.add_predictions(predictions)

    filename = result.write_to_csv(TestConfiguration())

    df = pd.read_csv(filename)
    assert_that(df.shape, equal_to((len(predictions), 2)))
    assert_that(list(df.Id.values), equal_to([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert_that(list(df.bikes.values), equal_to(predictions))

    cleanup_files()


def cleanup_files():
    config = TestConfiguration()
    for filename in glob.glob(os.path.join(config.write_predictions_to_path + config.write_results_start_name + "*")):
        os.remove(filename)
