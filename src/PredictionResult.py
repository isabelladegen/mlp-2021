import math

from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os
import pandas as pd
import enum

from src.Data import Columns
from src.configurations import Configuration
from src.create_dev_and_validation_csv import CsvWriter


class ResultsColumns(enum.Enum):
    id = 'Id'
    station = 'station'
    predictions = 'predictions'
    true_values = 'true values'
    mae = 'mae'


class PredictionResult:
    def __init__(self, ids_or_prediction_results: [], calculate_mean=False):
        if calculate_mean and all(isinstance(item, PredictionResult) for item in
                                  ids_or_prediction_results):  # only works for two Prediction Results
            merged = pd.merge(ids_or_prediction_results[0].results_df,
                              ids_or_prediction_results[1].results_df,
                              on=[ResultsColumns.id.value, ResultsColumns.station.value,
                                  ResultsColumns.true_values.value],
                              how='outer')
            merged[ResultsColumns.predictions.value] = list(merged[['predictions_x', 'predictions_y']].mean(axis=1))
            merged[ResultsColumns.predictions.value].apply(self.round_half_up)
            self.results_df = merged.sort_values(by=[ResultsColumns.id.value], ignore_index=True)
        elif all(isinstance(item, PredictionResult) for item in ids_or_prediction_results):
            df = pd.concat([result.results_df for result in ids_or_prediction_results], ignore_index=True)
            self.results_df = df.sort_values(by=[ResultsColumns.id.value], ignore_index=True)
        else:
            self.results_df = pd.DataFrame(ids_or_prediction_results, columns=[ResultsColumns.id.value])

    def add_predictions(self, predictions):
        self.results_df[ResultsColumns.predictions.value] = predictions

    def add_true_values(self, true_values):
        self.results_df[ResultsColumns.true_values.value] = true_values

    def add_stations(self, stations):
        self.results_df[ResultsColumns.station.value] = stations

    def mean_absolute_error(self):
        return mean_absolute_error(list(self.results_df[ResultsColumns.true_values.value]),
                                   list(self.results_df[ResultsColumns.predictions.value]))

    def mean_absolute_error_per_station(self):  # { station: mae}
        list_of_stations = set(self.results_df[ResultsColumns.station.value])
        result = {}
        for station in list_of_stations:
            df = self.results_df.loc[self.results_df[ResultsColumns.station.value] == station]
            result[station] = mean_absolute_error(list(df[ResultsColumns.true_values.value]),
                                                  list(df[ResultsColumns.predictions.value]))
        return result

    def write_to_csv(self, additional_prefix: str = '', config: Configuration = Configuration()) -> str:
        df = self.results_df[[ResultsColumns.id.value, ResultsColumns.predictions.value]]
        submission_df = df.rename(
            columns={ResultsColumns.predictions.value: Columns.bikes.value})  # format for submission

        curr_dt = datetime.now()
        filename = os.path.join(config.write_predictions_to_path,
                                additional_prefix + config.write_results_start_name + str(
                                    int(round(curr_dt.timestamp()))) + ".csv")

        CsvWriter.write_csv(submission_df, filename)
        return filename

    def round_half_up(self, n, decimals=0):
        multiplier = 10 ** decimals
        return math.floor(n * multiplier + 0.5) / multiplier
