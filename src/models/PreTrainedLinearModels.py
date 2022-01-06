from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration
from src.models.BestTrainedModelForEachStation import BestTrainedModelForEachStation

class PretrainedModels:
    def __init__(self, ):

class PretrainedLinearModels:
    def __init__(self, config: Configuration, all_stations_data):
        self.config = config
        self.per_station_data = all_stations_data.get_data_per_station()
        self.pre_trained_models = []  # create a list of pretrained models
        self.best_model_for_station = {}  # {station:BestTrainedModelForEachStation}

    def fit(self):
        for station, station_data in self.per_station_data.items():
            model = BestTrainedModelForEachStation(self.config, station_data, self.pre_trained_models)
            mae = model.fit()
            print(f"Station {station} best model mae training: {mae}")
            self.best_model_for_station[station] = model

    def predict(self, data: Data):
        per_station_data = data.get_data_per_station()  # { station_id: data}

        results_per_station = []
        for station, station_data in per_station_data.items():
            if station in self.best_model_for_station.keys():
                model = self.best_model_for_station[station]
                station_data_result = model.predict(station_data)
                results_per_station.append(station_data_result)

        return PredictionResult(results_per_station)
