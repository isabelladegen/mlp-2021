from hamcrest import *

from src.simple_regression import *
from src.configurations import TestConfiguration, WandbMode

TESTING_DATA_PATH = '../data/Dev/Testing/'


def test_uses_test_configuration_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()
    config.development_data_path = TESTING_DATA_PATH

    result = run(PoissonModel, config)
    wandb_result = result[RunResults.wandb]
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.disabled.value)


def test_uses_test_configuration_for_random_forest_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()
    config.development_data_path = TESTING_DATA_PATH

    result = run(RandomForestRegressorModel, config)
    wandb_result = result[RunResults.wandb]
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.disabled.value)


def test_uses_test_configuration_for_mlp_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()
    config.development_data_path = TESTING_DATA_PATH

    result = run(MultiLayerPerceptronRegressorModel, config)
    wandb_result = result[RunResults.wandb]
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.disabled.value)
