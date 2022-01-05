import wandb

from src.Data import Data
from src.configurations import Configuration, WandbLogs
from src.models.RandomForestRegressorModel import RandomForestRegressorModel


def sweep():
    non_sweep_config = Configuration()
    wandb.init(project=non_sweep_config.wandb_project_name,
               entity=non_sweep_config.wandb_entity,
               mode=non_sweep_config.wandb_mode,
               notes="feature testing, temperature and is holiday, avoiding over-fitting",
               tags=['RandomForrest', 'one model', 'model per station'],
               config=non_sweep_config.as_dict())

    sweeped_config = Configuration(**wandb.config)

    # Load training data dev (Create a smaller sweep set)
    sweep_training_data = Data(sweeped_config.no_nan_in_bikes, sweeped_config.sweep_training_path)

    # Single model for all stations
    one_model_for_all_station = RandomForestRegressorModel(sweeped_config, sweep_training_data)  # configure model
    one_model_for_all_station.fit()  # train

    # Load validation data
    sweep_validation_data = Data(sweeped_config.no_nan_in_bikes, sweeped_config.sweep_validation_path)

    # Predict
    one_model_dev_result = one_model_for_all_station.predict(sweep_training_data)
    one_model_val_result = one_model_for_all_station.predict(sweep_validation_data)

    # Evaluate and Log
    # overall mae
    one_model_mae_dev = one_model_dev_result.mean_absolute_error()
    one_model_mae_val = one_model_val_result.mean_absolute_error()

    wandb.log({
        WandbLogs.one_model_mae_dev.value: one_model_mae_dev,
        WandbLogs.one_model_mae_val.value: one_model_mae_val,
    })

    # Per station mae
    one_model_mae_per_station_dev = one_model_dev_result.mean_absolute_error_per_station()
    one_model_mae_per_station_val = one_model_val_result.mean_absolute_error_per_station()

    log_per_station_mae_to_wand(WandbLogs.one_model_mae_per_station_dev.value, one_model_mae_per_station_dev)
    log_per_station_mae_to_wand(WandbLogs.one_model_mae_per_station_val.value, one_model_mae_per_station_val)


# takes a  dictionary { station : mae }
def log_per_station_mae_to_wand(key: str, per_station_values: {}):  # {station:mae}
    for station, station_mae in per_station_values.items():
        wandb.log({key: station_mae, 'station': station})


# Not sweeped
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
            'values': [50, 100, 120]
        },
        'random_forest_criterion': {
            'values': ['squared_error', 'poisson']
        },
        'random_forest_max_depth': {
            'values': [None, 10, 50]
        },
        'random_forest_min_samples_split': {
            'values': [1, 3]
        },
        'random_forest_bootstrap': {
            'values': [False]
        },
        'random_forest_ccp_alpha': {
            'values': [0.0, 0.05, 0.1, 0.3]
        },
    }

    sweep_config_grid = {
        'name': 'Random forest sweep 1',
        'method': 'grid',
        'parameters': parameters_to_try
    }

    sweep_id = wandb.sweep(sweep_config_grid, project=Configuration().wandb_project_name)
    wandb.agent(sweep_id, function=sweep)
