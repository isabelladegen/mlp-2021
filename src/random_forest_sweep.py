import wandb

from src.Data import Data, Columns
from src.configurations import Configuration, WandbLogs
from src.models.PerStationModel import PerStationModel
from src.models.RandomForestRegressorModel import RandomForestRegressorModel
from src.run_utils import LogKeys, train_predict_evaluate_log_for_model_and_data


def sweep():
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
    sweep_dev_data = Data(sweeped_config.no_nan_in_bikes, sweeped_config.sweep_training_path)
    sweep_validation_data = Data(sweeped_config.no_nan_in_bikes, sweeped_config.sweep_validation_path)

    # Run one model for all stations
    if sweeped_config.run_one_model:
        one_model_for_all_station = RandomForestRegressorModel(sweeped_config, sweep_dev_data)
        log_keys = {LogKeys.mae_dev.value: WandbLogs.one_model_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.one_model_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.one_model_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.one_model_mae_per_station_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(one_model_for_all_station, sweep_dev_data, sweep_validation_data,
                                                      log_keys, wandb_run, sweeped_config.log_predictions_to_wandb)
    if sweeped_config.run_model_per_station:
        per_station_model = PerStationModel(sweeped_config, sweep_dev_data, RandomForestRegressorModel)
        log_keys = {LogKeys.mae_dev.value: WandbLogs.per_station_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.per_station_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.per_station_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.per_station_mae_per_station_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(per_station_model, sweep_dev_data, sweep_validation_data,
                                                      log_keys, wandb_run, sweeped_config.log_predictions_to_wandb)


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
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ],
                [Columns.station.value,  # with weather
                 Columns.data_3h_ago.value,
                 Columns.num_docks.value,
                 Columns.week_hour.value,
                 Columns.is_holiday.value,
                 Columns.air_pressure.value,
                 Columns.rel_humidity.value,
                 Columns.wind_mean_speed.value
                 ]
            ]
        }
    }

    sweep_config_grid = {
        'name': 'Random forest sweep test',
        'method': 'grid',
        'parameters': parameters_to_try
    }

    sweep_id = wandb.sweep(sweep_config_grid, project=Configuration().wandb_project_name)
    wandb.agent(sweep_id, function=sweep)
