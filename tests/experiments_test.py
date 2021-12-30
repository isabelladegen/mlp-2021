from hamcrest import *

from src.experiments import *
from src.configurations import TestConfiguration, WandbMode


def test_uses_test_configuration_and_does_not_start_a_wandb_experiment():
    config = TestConfiguration()

    wandb_result = run(config)
    assert_that(wandb_result, not_none())
    assert_that(wandb_result.Settings.mode, WandbMode.DISABLED.value)
