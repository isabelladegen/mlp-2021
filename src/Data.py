import pandas as pd
import os
import glob
import enum

from src.configurations import Configuration


class Columns(enum.Enum):
    docks = 'numDocks'  # choose for tests
    bikes = 'bikes'


class Data:
    def __init__(self, config: Configuration, data_path):
        self.configuration = config
        if not os.path.exists(data_path):
            raise "ERROR: Path: " + data_path + " does not exist!"
        if os.path.isdir(
                data_path):  # read all csv files into one panda, obviously assume they all have the same format
            self.raw_pd_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(data_path, "*.csv"))))
        else:  # read just the one file
            self.raw_pd_df = pd.read_csv(data_path)
        
        if config.no_nan_in_bikes:
            self.__remove_nan_rows_in_bikes()

    def get_raw_pd_df(self):
        return self.raw_pd_df

    def get_num_of_docks_for_each_station(self):
        return self.raw_pd_df.numDocks

    def get_true_values(self):
        if Columns.bikes.value in self.raw_pd_df:
            return list(self.raw_pd_df.bikes.values)
        return []

    def __remove_nan_rows_in_bikes(self):
        if Columns.bikes.value in self.raw_pd_df: # only drop nan if the data has a bike column
            self.raw_pd_df = self.raw_pd_df.dropna(subset=[Columns.bikes.value])

