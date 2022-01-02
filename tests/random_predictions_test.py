from hamcrest import *

from src.random_predictions import *
from src.configurations import TestConfiguration, WandbMode


def test_uses_test_configuration_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()

    result = run(config)
    wandb_result = result[RunResults.wandb]
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.disabled.value)


def test_creates_random_predictions():
    config = TestConfiguration()
    number_of_test_samples = 55800

    result = run(config)
    random_predictions = result[RunResults.random_predictions]
    predictions = random_predictions.predictions
    assert_that(len(predictions), equal_to(number_of_test_samples))
    assert_that(random_predictions.mean_absolute_error(), greater_than(2))

