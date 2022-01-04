from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration


class PerStationModel:
    def __init__(self, config: Configuration, all_stations_training_data: Data, model_class):
        self.configuration = config
        self.training_data_all_stations = all_stations_training_data

        # get data per station
        self.training_data_per_station = self.training_data_all_stations.get_data_per_station()
        result = self.__instantiate_model_per_station(config, model_class)
        self.model_per_station = result

    def __instantiate_model_per_station(self, config, model_class):
        result = {}
        for station, station_data in self.training_data_per_station.items():
            model = model_class(config, station_data)
            result[station] = model
        return result

    def fit(self):
        for model in self.model_per_station.values():
            model.fit()

    def get_model_for_station(self, station_id):
        return self.model_per_station[station_id]

    def predict(self, data: Data):
        per_station_data = data.get_data_per_station()  # { station_id: data}

        results_per_station = []
        for station, station_data in per_station_data.items():
            if station in self.model_per_station.keys():
                model = self.model_per_station[station]
                station_data_result = model.predict(station_data)
                results_per_station.append(station_data_result)

        return PredictionResult(results_per_station)
