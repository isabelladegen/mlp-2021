from hamcrest import *
from src.configurations import TestConfiguration
from src.test_data import *


def get_standard_test_config():
    configuration = TestConfiguration()
    configuration.test_data_path = "../data/test.csv"
    return configuration


def test_loads_data_from_specified_csv_file():
    configuration = get_standard_test_config()

    data = TestData(configuration)

    assert_that(data, not_none())


def test_return_pandas_data_frame_for_test_data():
    configuration = get_standard_test_config()

    data = TestData(configuration)
    df = data.get_raw_pd_df()

    assert_that(df.shape, equal_to((2250, 25)))
