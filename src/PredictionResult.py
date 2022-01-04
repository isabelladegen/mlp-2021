from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os
import csv
import pandas as pd
import enum

from src.Data import Columns
from src.configurations import Configuration
from src.create_dev_and_validation_csv import CsvWriter


class ResultsColumns(enum.Enum):
    id = 'Id'
    predictions = 'predictions'
    true_values = 'true values'
    mae = 'mae'


class PredictionResult:
    def __init__(self, ids_or_prediction_results: []):
        if all(isinstance(item, PredictionResult) for item in ids_or_prediction_results):
            df = pd.concat([result.results_df for result in ids_or_prediction_results], ignore_index=True)
            self.results_df = df.sort_values(by=[ResultsColumns.id.value], ignore_index=True)
        else:
            self.results_df = pd.DataFrame(ids_or_prediction_results, columns=[ResultsColumns.id.value])

    def add_predictions(self, predictions):
        self.results_df[ResultsColumns.predictions.value] = predictions

    def add_true_values(self, true_values):
        self.results_df[ResultsColumns.true_values.value] = true_values

    def mean_absolute_error(self):
        return mean_absolute_error(list(self.results_df[ResultsColumns.true_values.value]),
                                   list(self.results_df[ResultsColumns.predictions.value]))

    def write_to_csv(self, config: Configuration = Configuration()) -> str:
        df = self.results_df[[ResultsColumns.id.value, ResultsColumns.predictions.value]]
        df.rename(columns={ResultsColumns.predictions.value: Columns.bikes.value},
                  inplace=True)  # format for submission

        curr_dt = datetime.now()
        filename = os.path.join(config.write_predictions_to_path,
                                config.write_results_start_name + str(int(round(curr_dt.timestamp()))) + ".csv")

        CsvWriter.write_csv(df, filename)
        return filename
