import pandas as pd
import os
import glob
import enum

from sklearn.impute import SimpleImputer


class Columns(enum.Enum):
    id = 'Id'
    station = 'station'
    num_docks = 'numDocks'
    bikes = 'bikes'
    data_3h_ago = 'bikes_3h_ago'
    weekday = "weekday"
    week_hour = 'weekhour'
    is_holiday = 'isHoliday'
    temperature = 'temperature.C'
    wind_mean_speed = 'windMeanSpeed.m.s'
    wind_direction = 'windDirection.grades'
    rel_humidity = 'relHumidity.HR'
    air_pressure = 'airPressure.mb'
    full_profile_3h_diff_bikes = 'full_profile_3h_diff_bikes'
    full_profile_bikes = 'full_profile_bikes'
    short_profile_bikes = 'short_profile_bikes'
    short_profile_3h_diff_bikes = 'short_profile_3h_diff_bikes'


class PretrainedLinearModels:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise "ERROR: Path: " + data_path + " does not exist!"
        result = pd.DataFrame()
        # this is a very slow way to do it, would be better to concat lists but no time right now
        for file_name in glob.glob(os.path.join(data_path, "*.csv")):
            df = pd.read_csv(file_name, header=0).set_index('feature').transpose().reset_index()
            df = df.drop('index', axis=1)
            df.insert(0, 'model', os.path.basename(file_name))
            result = result.append(df, ignore_index=True)
        self.models_df = result.set_index('model')
        self.weights_matrix = self.models_df.to_numpy(na_value=0).transpose()  # weights x models matrix
        # same feature for each model as weight will just be set to 0 if feature not used in model
        self.features = list(self.models_df.columns.drop('(Intercept)'))


class Data:
    """
        Loads data and does the preprocessing based on configuration
    """

    def __init__(self, no_nan_in_bikes: bool, data_path_or_frame):
        self.no_nan_in_bikes = no_nan_in_bikes
        if isinstance(data_path_or_frame, str):
            self.__instantiate_from_path(data_path_or_frame)
        elif isinstance(data_path_or_frame, pd.DataFrame):
            self.raw_pd_df = data_path_or_frame
        else:
            raise "Given data type is not supported: " + type(data_path_or_frame)

        if self.no_nan_in_bikes:
            self.__remove_nan_rows_in_bikes()

        self.raw_pd_df.replace(to_replace={'Monday': '0',
                                           'Tuesday': '1',
                                           'Wednesday': '2',
                                           'Thursday': '3',
                                           'Friday': '4',
                                           'Saturday': '5',
                                           'Sunday': '6'}, inplace=True)

    def __instantiate_from_path(self, data_path):
        if not os.path.exists(data_path):
            raise "ERROR: Path: " + data_path + " does not exist!"
        if os.path.isdir(
                data_path):  # read all csv files into one panda, obviously assume they all have the same format
            # ignore_index=True required to ensure that the new dataframe has unique row indices!!!
            self.raw_pd_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(data_path, "*.csv"))), ignore_index=True)
        else:  # read just the one file
            self.raw_pd_df = pd.read_csv(data_path)

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
        if Columns.bikes.value in self.raw_pd_df:
            return self.raw_pd_df.bikes.values
        else:
            number_of_rows = self.raw_pd_df.shape[0]
            return [None] * number_of_rows

    def get_feature_matrix_x_for(self, columns_to_include: [str], feature_data_types: {str: str}):
        sub_df = self.raw_pd_df[columns_to_include]
        required_feature_types = {key: feature_data_types[key] for key in columns_to_include}
        typed_df = sub_df.astype(required_feature_types)
        imp = SimpleImputer(strategy="most_frequent")
        non_nan_nd = imp.fit_transform(typed_df)
        return non_nan_nd

    def get_data_per_station(self) -> {}:  # { station_id: data}
        result = {}
        stations = set(self.raw_pd_df[Columns.station.value])
        for station in stations:
            station_df = self.raw_pd_df.loc[self.raw_pd_df[Columns.station.value] == station]
            result[station] = Data(self.no_nan_in_bikes, station_df)
        return result

    def get_ids(self):
        if Columns.id.value in self.raw_pd_df:
            return list(self.raw_pd_df[Columns.id.value])
        else:
            return list(range(0, self.raw_pd_df.shape[0]))

    def get_stations(self):
        return list(self.raw_pd_df[Columns.station.value])
