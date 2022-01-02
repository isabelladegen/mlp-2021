from hamcrest import *

from src.experiments import *
from src.configurations import TestConfiguration, WandbMode


def test_uses_test_configuration_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()

    wandb_result, result = run(config)
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.DISABLED.value)


def test_creates_random_predictions_if_config_true():
    config = TestConfiguration()
    config.predict_random_numbers = True
    number_of_test_samples = 12
    max_number_of_bikes = 50

    wandb_result, result = run(config)

    predictions = result.random_predictions
    assert_that(len(predictions), equal_to(number_of_test_samples))
    assert_that(all(0 <= pred < max_number_of_bikes for pred in predictions), 'within range')


def test_writes_a_csv_file_with_the_predicted_numbers_of_bikes():
    config = TestConfiguration()
