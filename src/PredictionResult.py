from sklearn.metrics import mean_absolute_error
from datetime import datetime
import os
import csv
import pandas as pd

from src.Data import Columns
from src.configurations import Configuration


class PredictionResult:
    def __init__(self):
        self.predictions = []
        self.true_values = []

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def add_predictions(self, predictions):
        self.predictions = predictions

    def add_true_values(self, true_values):
        self.true_values = true_values

    def mean_absolute_error(self):
        assert len(self.predictions) == len(self.true_values)
        return mean_absolute_error(self.true_values, self.predictions)

    def write_to_csv(self, config: Configuration = Configuration()):
        curr_dt = datetime.now()
        filename = os.path.join(config.write_predictions_to_path,
                                config.write_results_start_name + str(int(round(curr_dt.timestamp()))) + ".csv")
        file = open(filename, 'x')  # x so we don't overwrite a file if it exists
        writer = csv.writer(file)
        writer.writerow([Columns.id.value, Columns.bikes.value])
        for index, prediction in enumerate(self.predictions):
            writer.writerow([index + 1, prediction])
        file.close()
        return filename

    def predictions_as_df(self):
        ids = list(range(1, len(self.predictions) + 1))
        if len(self.true_values) == len(self.predictions):  # log true values too
            return pd.DataFrame(list(zip(ids, self.predictions, self.true_values)),
                                columns=[Columns.id.value, Columns.bikes.value, 'true_bikes_values'])
        return pd.DataFrame(list(zip(ids, self.predictions)),
                            columns=[Columns.id.value, Columns.bikes.value])
