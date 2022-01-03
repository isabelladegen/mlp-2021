import os.path

from hamcrest import *
import pandas as pd

from src.Data import Data
from src.configurations import TestConfiguration
from src.create_dev_and_validation_csv import CsvWriter, DevValidationDataWriter


def test_creates_a_csv_file_from_a_data_frame():
    filename = 'testing.csv'
    column1 = [1, 2, 3, 4, 5, 6]
    column2 = ['a', 'b', 'c', 'd', 'e', 'f']
    column3 = [33.5, 22.2, 44.4, 55.7, 30, 60]
    df = pd.DataFrame(list(zip(column1, column2, column3)), columns=['column 1', 'column 2', 'column 3'])
    CsvWriter.write_csv(df, filename)

    from_disk = pd.read_csv(filename)
    assert_that(list(from_disk['column 1'].values), equal_to(column1))
    assert_that(list(from_disk['column 2'].values), equal_to(column2))
    assert_that(list(from_disk['column 3'].values), equal_to(column3))

    CsvWriter.delete_file(filename)


def test_writes_dev_and_validation_csv_file():
    config = TestConfiguration()
    output_path = config.development_data_path
    dev_filename = config.dev_data_filename
    val_filename = config.val_data_filename

    DevValidationDataWriter.create_new_dev_and_validation_csv(config)

    dev_df = pd.read_csv(os.path.join(output_path, dev_filename))
    val_df = pd.read_csv(os.path.join(output_path, val_filename))

    training_data = Data(config.no_nan_in_bikes, config.training_data_path).raw_pd_df
    assert_that(dev_df.shape, equal_to((50220, 25)))
    assert_that(val_df.shape, equal_to((5580, 25)))
    assert_that(dev_df.shape[0] + val_df.shape[0], equal_to(training_data.shape[0]))
    assert_that(list(dev_df.columns), equal_to(list(training_data.columns)))

    CsvWriter.delete_file(os.path.join(output_path, dev_filename))
    CsvWriter.delete_file(os.path.join(output_path, val_filename))
