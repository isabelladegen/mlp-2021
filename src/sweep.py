import wandb

from src.Data import Data, Columns
from src.configurations import Configuration, WandbLogs
from src.models.MultiLayerPerceptronRegressorModel import MultiLayerPerceptronRegressorModel
from src.models.PerStationModel import PerStationModel
from src.models.RandomForestRegressorModel import RandomForestRegressorModel
from src.run_utils import LogKeys, train_predict_evaluate_log_for_model_and_data


def sweep(model_klass):
    non_sweep_config = Configuration()
    wandb_run = wandb.init(project=non_sweep_config.wandb_project_name,
                           entity=non_sweep_config.wandb_entity,
                           mode=non_sweep_config.wandb_mode,
                           notes="feature testing, avoiding over-fitting",
                           tags=['RandomForrest', 'model per station'],
                           config=non_sweep_config.as_dict())

    sweeped_config = Configuration(**wandb.config)
    sweeped_config.run_test_predictions = False
    sweeped_config.log_predictions_to_wandb = False

    # Load sweep data
    sweep_dev_data = Data(sweeped_config.no_nan_in_bikes,
                          sweeped_config.development_data_path + sweeped_config.dev_data_filename)
    sweep_validation_data = Data(sweeped_config.no_nan_in_bikes,
                                 sweeped_config.development_data_path + sweeped_config.val_data_filename)

    # Run one model for all stations
    if sweeped_config.run_one_model:
        one_model_for_all_station = model_klass(sweeped_config, sweep_dev_data)
        log_keys = {LogKeys.mae_dev.value: WandbLogs.one_model_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.one_model_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.one_model_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.one_model_mae_per_station_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(one_model_for_all_station, sweep_dev_data, sweep_validation_data,
                                                      log_keys, wandb_run, sweeped_config.log_predictions_to_wandb)
    if sweeped_config.run_model_per_station:
        per_station_model = PerStationModel(sweeped_config, sweep_dev_data, model_klass)
        log_keys = {LogKeys.mae_dev.value: WandbLogs.per_station_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.per_station_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.per_station_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.per_station_mae_per_station_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(per_station_model, sweep_dev_data, sweep_validation_data,
                                                      log_keys, wandb_run, sweeped_config.log_predictions_to_wandb)


# Not sweeped
# random_forest_criterion -> absolute seems best but also takes longest to run
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

def random_forest_sweep_params():
    parameters_to_try = {
        'random_forest_min_impurity_decrease': {
            'values': [0.0, 0.0001, 0.0004, 0.0009, 0.002, 0.0008]
        },
        'random_forest_features': {
            'values': [
                # [Columns.station.value,  # best features for default setting
                #  Columns.data_3h_ago.value,
                #  Columns.num_docks.value,
                #  Columns.week_hour.value,
                #  Columns.is_holiday.value,
                #  ],
                # [Columns.station.value,
                #  Columns.data_3h_ago.value,
                #  Columns.num_docks.value,
                #  Columns.week_hour.value,
                #  ],
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
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                [Columns.station.value,  # with profiles and weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value,
                 Columns.short_profile_bikes.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                [Columns.station.value,  # with profiles and weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                # [Columns.station.value,  # All features
                #  Columns.data_3h_ago.value,
                #  Columns.num_docks.value,
                #  Columns.week_hour.value,
                #  Columns.is_holiday.value,
                #  Columns.full_profile_bikes.value,
                #  Columns.full_profile_3h_diff_bikes.value,
                #  Columns.short_profile_3h_diff_bikes.value,
                #  Columns.short_profile_bikes.value,
                #  Columns.air_pressure.value,
                #  Columns.rel_humidity.value,
                #  Columns.wind_mean_speed.value,
                #  Columns.wind_direction.value,
                #  Columns.temperature.value
                #  ],
                [Columns.data_3h_ago.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.full_profile_bikes.value,
                 Columns.short_profile_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value
                 ],
                # [Columns.data_3h_ago.value,  # some features from the given models
                #  Columns.full_profile_bikes.value,
                #  Columns.full_profile_3h_diff_bikes.value,
                #  Columns.air_pressure.value
                #  ],
                # [Columns.data_3h_ago.value,
                #  Columns.short_profile_bikes.value,
                #  Columns.short_profile_3h_diff_bikes.value,
                #  Columns.air_pressure.value
                #  ],
                # [Columns.data_3h_ago.value,
                #  Columns.full_profile_bikes.value,
                #  Columns.full_profile_3h_diff_bikes.value,
                #  Columns.short_profile_bikes.value,
                #  Columns.short_profile_3h_diff_bikes.value
                #  ],
            ]
        }
    }
    return parameters_to_try


def mlp_sweep_params():
    sweep_params = {
        'mlp_max_iter': {
            'values': [100, 300, 500]
        },
        'mlp_features': {
            'values': [
                [Columns.station.value,  # best features for default setting
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 ],
                [Columns.station.value,
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
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
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                [Columns.station.value,  # with profiles and weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value,
                 Columns.short_profile_bikes.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                [Columns.station.value,  # with profiles and weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                [Columns.station.value,  # All features
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value,
                 Columns.short_profile_bikes.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value,
                 Columns.wind_direction.value,
                 Columns.temperature.value
                 ],
                [Columns.data_3h_ago.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.full_profile_bikes.value,
                 Columns.short_profile_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value
                 ],
                [Columns.data_3h_ago.value,  # some features from the given models
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.air_pressure.value
                 ],
                [Columns.data_3h_ago.value,
                 Columns.short_profile_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value,
                 Columns.air_pressure.value
                 ],
                [Columns.data_3h_ago.value,
                 Columns.full_profile_bikes.value,
                 Columns.full_profile_3h_diff_bikes.value,
                 Columns.short_profile_bikes.value,
                 Columns.short_profile_3h_diff_bikes.value
                 ],
            ]
        }
    }
    return sweep_params


if __name__ == '__main__':
    sweep_params = mlp_sweep_params()

    sweep_config_grid = {
        'name': 'MLP Sweep 1 (7)',
        'method': 'grid',
        'parameters': sweep_params
    }
    sweep_id = wandb.sweep(sweep_config_grid, project=Configuration().wandb_project_name)
    wandb.agent(sweep_id, function=lambda: sweep(MultiLayerPerceptronRegressorModel))
