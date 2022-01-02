from hamcrest import *

from src.experiments import *
from src.configurations import TestConfiguration, WandbMode


def test_uses_test_configuration_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()

    result = run(config)
    wandb_result = result[RunResults.wandb]
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.disabled.value)


def test_writes_a_csv_file_with_the_predicted_numbers_of_bikes():
    config = TestConfiguration()
