from hamcrest import *

from src.PredictionResult import PredictionResult


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
