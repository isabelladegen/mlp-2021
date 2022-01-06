import wandb
import enum

from src.Data import Data, Columns
from src.configurations import Configuration, WandbLogs
from src.models.PerStationModel import PerStationModel
from src.models.RandomForestRegressorModel import RandomForestRegressorModel


class LogKeys(enum.Enum):
    mae_dev = 'mae dev'
    mae_val = 'mae val'
    mae_per_station_dev = 'mae per station dev'
    mae_per_station_val = 'mae per station val'


def sweep():
    non_sweep_config = Configuration()
    wandb.init(project=non_sweep_config.wandb_project_name,
               entity=non_sweep_config.wandb_entity,
               mode=non_sweep_config.wandb_mode,
               notes="feature testing, temperature and is holiday, avoiding over-fitting",
               tags=['RandomForrest', 'one model', 'model per station'],
               config=non_sweep_config.as_dict())

    sweeped_config = Configuration(**wandb.config)

    # Load sweep data
    sweep_dev_data = Data(sweeped_config.no_nan_in_bikes, sweeped_config.sweep_training_path)
    sweep_validation_data = Data(sweeped_config.no_nan_in_bikes, sweeped_config.sweep_validation_path)

    # Run one model for all stations
    if sweeped_config.sweep_one_model:
        one_model_for_all_station = RandomForestRegressorModel(sweeped_config, sweep_dev_data)
        log_keys = {LogKeys.mae_dev.value: WandbLogs.one_model_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.one_model_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.one_model_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.one_model_mae_per_station_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(one_model_for_all_station, sweep_dev_data, sweep_validation_data,
                                                      log_keys)
    if sweeped_config.sweep_model_per_station:
        per_station_model = PerStationModel(sweeped_config, sweep_dev_data, RandomForestRegressorModel)
        log_keys = {LogKeys.mae_dev.value: WandbLogs.per_station_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.per_station_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.per_station_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.per_station_mae_per_station_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(per_station_model, sweep_dev_data, sweep_validation_data,
                                                      log_keys)


def train_predict_evaluate_log_for_model_and_data(model, training_data, validation_data, keys):
    # train
    model.fit()

    # predict
    training_result = model.predict(training_data)
    validation_result = model.predict(validation_data)

    # evaluate
    mae_dev = training_result.mean_absolute_error()
    one_model_mae_val = validation_result.mean_absolute_error()
    mae_per_station_dev = training_result.mean_absolute_error_per_station()
    mae_per_station_val = validation_result.mean_absolute_error_per_station()
    # log results
    wandb.log({
        keys[LogKeys.mae_dev.value]: mae_dev,
        keys[LogKeys.mae_val.value]: one_model_mae_val,
    })
    log_per_station_mae_to_wand(keys[LogKeys.mae_per_station_dev.value], mae_per_station_dev)
    log_per_station_mae_to_wand(keys[LogKeys.mae_per_station_val.value], mae_per_station_val)


# takes a  dictionary { station : mae }
def log_per_station_mae_to_wand(key: str, per_station_values: {}):  # {station:mae}
    for station, station_mae in per_station_values.items():
        wandb.log({key: station_mae, 'station': station})


# Not sweeped
# random_forest_min_samples_split
# random_forest_max_depth
# random_forest_min_samples_leaf,
# random_forest_min_weight_fraction_leaf,
# random_forest_max_features,
# random_forest_max_leaf_nodes,
# random_forest_min_impurity_decrease,
# random_forest_bootstrap,
# random_forest_oob_score,
# random_forest_n_jobs,
# random_forest_random_state,
# random_forest_verbose,
# random_forest_warm_start,
# random_forest_max_samples

if __name__ == '__main__':
    parameters_to_try = {
        'random_forest_n_estimators': {
            'values': [50, 100, 120]  # made no difference
        },
        'random_forest_criterion': {
            'values': ['squared_error', 'absolute_error']
            # squared_error better than poisson, compare to absolute error
        },
        'random_forest_ccp_alpha': {
            'values': [0.0, 0.001, 0.01, 0.02, 0.005]  # unclear but 0.05 did well
        },
        'random_forest_features': {
            'values': [
                [Columns.station.value,  # best features for default setting
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 ],
                [Columns.station.value,  # with full profiles
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value
                 ],
                [Columns.station.value,  # with profiles and weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.air_pressure,
                 Columns.rel_humidity,
                 Columns.wind_mean_speed,
                 ],
                [Columns.station.value,  # with weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.air_pressure,
                 Columns.rel_humidity,
                 Columns.wind_mean_speed,
                 ],
            ]
        }
    }

    sweep_config_grid = {
        'name': 'Random forest sweep 2',
        'method': 'grid',
        'parameters': parameters_to_try
    }

    sweep_id = wandb.sweep(sweep_config_grid, project=Configuration().wandb_project_name)
    wandb.agent(sweep_id, function=sweep)
