import pandas as pd
import os
import glob
import enum

from sklearn.impute import SimpleImputer


class Columns(enum.Enum):
    id = 'Id'
    num_docks = 'numDocks'
    bikes = 'bikes'
    data_3h_ago = 'bikes_3h_ago'


class Data:
    """
        Loads data and does the preprocessing based on configuration
    """

    def __init__(self, no_nan_in_bikes: bool, data_path):
        if not os.path.exists(data_path):
            raise "ERROR: Path: " + data_path + " does not exist!"
        if os.path.isdir(
                data_path):  # read all csv files into one panda, obviously assume they all have the same format
            # ignore_index=True required to ensure that the new dataframe has unique row indices!!!
            self.raw_pd_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(data_path, "*.csv"))), ignore_index=True)
        else:  # read just the one file
            self.raw_pd_df = pd.read_csv(data_path)

        if no_nan_in_bikes:
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
        if Columns.bikes.value in self.raw_pd_df:  # only drop nan if the data has a bike column
            self.raw_pd_df = self.raw_pd_df.dropna(subset=[Columns.bikes.value])

    def get_y(self) -> []:
        return self.raw_pd_df.bikes.values

    def get_feature_matrix_x_for(self, columns_to_include: [str], feature_data_types: {str: str}):
        sub_df = self.raw_pd_df[columns_to_include]
        required_feature_types = {key: feature_data_types[key] for key in columns_to_include}
        typed_df = sub_df.astype(required_feature_types)
        imp = SimpleImputer(strategy="most_frequent")
        non_nan_nd = imp.fit_transform(typed_df)
        return non_nan_nd
