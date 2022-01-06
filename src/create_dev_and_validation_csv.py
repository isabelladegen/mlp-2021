# script to split the Training dataset into a dev and a validation csv
# rune once or only when intended, the selected rows are randomly chosen from the full labelled training dataset

import pandas as pd
import os

from src.Data import Data
from src.configurations import Configuration


class CsvWriter:
    @classmethod
    def write_csv(cls, df: pd.DataFrame, filename: str):
        df.to_csv(filename, index=False)  # don't write index to csv

    @classmethod
    def delete_file(cls, filename: str):
        os.remove(filename)


class DevValidationDataWriter:
    @classmethod
    def create_new_dev_and_validation_csv(cls, config: Configuration):
        training_data_path = config.training_data_path
        development_data_output_path = config.development_data_path
        dev_filename = config.dev_data_filename
        val_filename = config.val_data_filename
        split = config.dev_validation_data_split

        assert config.no_nan_in_bikes, 'Wrong configuration: You don\'t want nan values in dev/validation data'
        training_df = Data(config.no_nan_in_bikes, training_data_path).raw_pd_df

        number_of_training_examples = training_df.shape[0]
        validation_data_size: int = round(number_of_training_examples / split)

        validation_df = training_df.sample(n=validation_data_size)
        val_indexes = validation_df.index
        training_df.drop(index=val_indexes, inplace=True)  # change training_df to become dev
        CsvWriter.write_csv(training_df, os.path.join(development_data_output_path, dev_filename))
        CsvWriter.write_csv(validation_df, os.path.join(development_data_output_path, val_filename))

    @classmethod
    def create_new_sweep_dev_and_validation_csv(cls, config: Configuration):
        training_data_path = config.training_data_path
        sweep_dev_file = config.sweep_training_path
        sweep_val_file = config.sweep_validation_path
        split = config.sweep_data_percentage

        assert config.no_nan_in_bikes, 'Wrong configuration: You don\'t want nan values in dev/validation data'
        training_df = Data(config.no_nan_in_bikes, training_data_path).raw_pd_df

        number_of_training_examples = training_df.shape[0]
        sweep_data_size: int = round(number_of_training_examples / split)

        val_df = training_df.sample(n=sweep_data_size)
        dev_df = training_df.sample(n=sweep_data_size)

        df_merge = pd.merge(val_df, dev_df, on=list(val_df.columns), how='inner')
        val_df = val_df.append(df_merge)

        val_df['Duplicated'] = val_df.duplicated(keep=False)  # keep=False marks the duplicated row with a True
        unique_val = val_df[~val_df['Duplicated']]  # selects only rows which are not duplicated.
        del unique_val['Duplicated']  # delete the indicator column

        CsvWriter.write_csv(dev_df, sweep_dev_file)
        CsvWriter.write_csv(unique_val, sweep_val_file)


def create_development_data_set():
    DevValidationDataWriter.create_new_dev_and_validation_csv(Configuration())


def create_sweep_data_set():
    DevValidationDataWriter.create_new_sweep_dev_and_validation_csv(Configuration())


def main():
    create_sweep_data_set()


# be careful before you run this as it will overwrite existing files, you might want to back them up first
if __name__ == "__main__":
    main()
