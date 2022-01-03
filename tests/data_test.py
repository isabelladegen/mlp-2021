from hamcrest import *
import numpy as np
from src.configurations import TestConfiguration
from src.Data import *


def test_loads_data_from_specified_csv_file():
    configuration = TestConfiguration()

    data = Data(configuration.no_nan_in_bikes, configuration.test_data_path)

    assert_that(data, not_none())


def test_return_pandas_data_frame_for_test_data():
    configuration = TestConfiguration()

    data = Data(configuration.no_nan_in_bikes, configuration.test_data_path)
    df = data.get_raw_pd_df()

    assert_that(df.shape, equal_to((2250, 25)))


def test_returns_number_of_docs_for_each_row():
    configuration = TestConfiguration()

    data = Data(configuration.no_nan_in_bikes, configuration.test_data_path)
    num_docks = data.get_num_of_docks_for_each_station()

    assert_that(len(num_docks), equal_to(2250))
    assert_that(num_docks[0], equal_to(27))
    assert_that(num_docks[30], equal_to(15))
    assert_that(num_docks[2172], equal_to(16))


def test_returns_empty_list_if_rows_have_no_labelled_data():
    configuration = TestConfiguration()

    data = Data(configuration.no_nan_in_bikes, configuration.test_data_path)
    true_values = data.get_true_values()

    assert_that(len(true_values), equal_to(0))


def test_return_true_value_for_each_row_when_theres_bike_data():
    data = Data(True, "../data/Train/station_201_deploy.csv")
    true_values = data.get_true_values()

    rows_without_nan_in_bikes = 744
    assert_that(len(true_values), equal_to(rows_without_nan_in_bikes))
    assert_that(true_values[0], equal_to(1))
    assert_that(true_values[rows_without_nan_in_bikes - 1], equal_to(14))


def test_returns_number_of_bikes_as_y():
    configuration = TestConfiguration()

    data = Data(configuration.no_nan_in_bikes, "../data/Train/station_201_deploy.csv")
    y = data.get_y()

    assert_that(len(y), equal_to(data.raw_pd_df.shape[0]))
    assert_that(y[0], equal_to(1))
    assert_that(y[743], equal_to(14))


def test_loads_all_files_into_dataframe_if_path_is_a_dir():
    configuration = TestConfiguration()
    configuration.no_nan_in_bikes = False  # leaves rows with nan values in the dataset

    data = Data(configuration.no_nan_in_bikes, configuration.training_data_path)
    assert_that(data.raw_pd_df.shape, equal_to((55875, 25)))


def test_removes_nan_in_bikes_column_if_config_set_to_true():
    configuration = TestConfiguration()
    configuration.no_nan_in_bikes = True

    data = Data(configuration.no_nan_in_bikes, configuration.training_data_path)
    assert_that(data.raw_pd_df.shape, equal_to((55800, 25)))


def test_creates_unique_index_for_each_row_when_reading_multiple_files():
    configuration = TestConfiguration()

    df = Data(configuration.no_nan_in_bikes, configuration.training_data_path).raw_pd_df
    unique_indices = set(df.index)
    assert_that(len(unique_indices), equal_to(55800))


def test_creates_x_feature_matrix_for_given_columns():
    configuration = TestConfiguration()
    configuration.poisson_features = [Columns.data_3h_ago.value, Columns.num_docks.value]

    data = Data(configuration.no_nan_in_bikes, configuration.training_data_path)

    features = configuration.poisson_features
    feature_matrix_x = data.get_feature_matrix_x_for(features, configuration.features_data_type)

    assert_that(feature_matrix_x.shape, equal_to((data.raw_pd_df.shape[0], len(features))))


def test_preprocess_feature_matrix_by_fill_missing_values_with_most_frequent_ones():
    configuration = TestConfiguration()
    configuration.poisson_features = [Columns.data_3h_ago.value, Columns.num_docks.value]

    data = Data(configuration.no_nan_in_bikes, configuration.training_data_path)
    feature_matrix_x = data.get_feature_matrix_x_for(configuration.poisson_features, configuration.features_data_type)

    assert_that(np.count_nonzero(np.isnan(feature_matrix_x)), equal_to(0))
