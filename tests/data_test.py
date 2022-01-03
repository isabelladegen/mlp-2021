from hamcrest import *
from src.configurations import TestConfiguration
from src.Data import *


def get_standard_test_config():
    configuration = TestConfiguration()
    configuration.test_data_path = "../data/test.csv"
    return configuration


def test_loads_data_from_specified_csv_file():
    configuration = get_standard_test_config()

    data = Data(configuration, configuration.test_data_path)

    assert_that(data, not_none())


def test_return_pandas_data_frame_for_test_data():
    configuration = get_standard_test_config()

    data = Data(configuration, configuration.test_data_path)
    df = data.get_raw_pd_df()

    assert_that(df.shape, equal_to((2250, 25)))


def test_returns_number_of_docs_for_each_row():
    configuration = get_standard_test_config()

    data = Data(configuration, configuration.test_data_path)
    num_docks = data.get_num_of_docks_for_each_station()

    assert_that(len(num_docks), equal_to(2250))
    assert_that(num_docks[0], equal_to(27))
    assert_that(num_docks[30], equal_to(15))
    assert_that(num_docks[2172], equal_to(16))


def test_returns_empty_list_if_rows_have_no_labelled_data():
    configuration = get_standard_test_config()

    data = Data(configuration, configuration.test_data_path)
    true_values = data.get_true_values()

    assert_that(len(true_values), equal_to(0))


def test_return_true_value_for_each_row_when_theres_bike_data():
    configuration = TestConfiguration()

    data = Data(configuration, "../data/Train/station_201_deploy.csv")
    true_values = data.get_true_values()

    rows_without_nan_in_bikes = 744
    assert_that(len(true_values), equal_to(rows_without_nan_in_bikes))
    assert_that(true_values[0], equal_to(1))
    assert_that(true_values[rows_without_nan_in_bikes - 1], equal_to(14))


def test_loads_all_files_into_dataframe_if_path_is_a_dir():
    configuration = TestConfiguration()
    configuration.no_nan_in_bikes = False  # leaves rows with nan values in the dataset

    data = Data(configuration, configuration.training_data_path)
    assert_that(data.raw_pd_df.shape, equal_to((55875, 25)))


def test_loads_removes_nan_in_bikes_column_if_config_set_to_true():
    configuration = TestConfiguration()
    configuration.no_nan_in_bikes = True

    data = Data(configuration, configuration.training_data_path)
    assert_that(data.raw_pd_df.shape, equal_to((55800, 25)))


def test_creates_unique_index_for_each_row_when_reading_multiple_files():
    configuration = TestConfiguration()

    df = Data(configuration, configuration.training_data_path).raw_pd_df
    unique_indices = set(df.index)
    assert_that(len(unique_indices), equal_to(55800))
