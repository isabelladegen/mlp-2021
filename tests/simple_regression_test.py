from hamcrest import *

from src.simple_regression import *
from src.configurations import TestConfiguration, WandbMode


def test_uses_test_configuration_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()
    config.development_data_path = Configuration().development_data_path # bit of a hack but for now ...

    result = run(config)
    wandb_result = result[RunResults.wandb]
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.disabled.value)

