import csv
from src.Data import Data
from src.configurations import Configuration
from utils.station_data_builder import *

header = ["Id", "station", "latitude", "longitude", "numDocks", "timestamp", "year", "month", "day", "hour", "weekday",
          "weekhour", "isHoliday", "windMaxSpeed.m.s", "windMeanSpeed.m.s", "windDirection.grades", "temperature.C",
          "relHumidity.HR", "airPressure.mb", "precipitation.l.m2", "bikes_3h_ago", "full_profile_3h_diff_bikes",
          "full_profile_bikes", "short_profile_3h_diff_bikes", "short_profile_bikes", "bikes"]


class TestDataBuilder:
    def __init__(self, config: Configuration):
        self.config = config
        self.rows = [[]]
        self.testing_data_csv = "testingdata/TestDataBuilder.csv"

    def build(self):
        file = open(self.testing_data_csv, 'w')
        writer = csv.writer(file)
        writer.writerow(header)
        for row in self.rows:
            writer.writerow(row)
        file.close()
        return Data(self.config, self.testing_data_csv)

    def with_stations(self, stations: [StationDataBuilder]):
        for index, station in enumerate(stations):
            row = [index, station.station, station.latitude, station.longitude, station.numDocks, station.timestamp,
                   station.year, station.month, station.day, station.hour, station.weekday, station.weekhour,
                   station.isHoliday, station.windMaxSpeed, station.windMeanSpeed, station.windDirection,
                   station.temperature, station.relHumidity, station.airPressure, station.precipitation,
                   station.bikes_3h_ago, station.full_profile_3h_diff_bikes, station.full_profile_bikes,
                   station.short_profile_3h_diff_bikes, station.short_profile_bikes, station.bikes
                   ]
            self.rows.append(row)
        return self
