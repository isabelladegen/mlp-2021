from hamcrest import *
from src.configurations import *
from dataclasses import fields


def test_enables_wandb_mode_by_default():
    config = Configuration()
    assert_that(config.wandb_mode, equal_to(WandbMode.online.value))


def test_returns_test_configuration_with_wandb_disabled_and_all_other_fields_the_same():
    config = Configuration()
    test_config = TestConfiguration()

    assert_that(test_config.wandb_mode, equal_to(WandbMode.disabled.value))
    assert_that(len(fields(test_config)), equal_to(len(fields(config))))


def test_returns_dictionary_for_configuration():
    config = Configuration()
    test_config = TestConfiguration()

    config_dict = config.as_dict()
    test_config_dict = test_config.as_dict()

    assert_that(len(config_dict), equal_to(len(fields(config))))
    assert_that(len(test_config_dict), equal_to(len(fields(test_config))))
    assert_that(config_dict['wandb_mode'], equal_to(WandbMode.online.value))
    assert_that(test_config_dict['wandb_mode'], equal_to(WandbMode.disabled.value))
