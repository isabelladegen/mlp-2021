import wandb

from src.configurations import Configuration, WandbLogs, RunResults
from src.Data import Data
from src.models.ExtraTreesRegressorModel import ExtraTreesRegressorModel
from src.models.PerStationModel import PerStationModel
from src.models.PoissonModel import PoissonModel
from src.models.RandomForestRegressorModel import RandomForestRegressorModel
from src.run_utils import LogKeys, train_predict_evaluate_log_for_model_and_data


def run(model_class, config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name,
                           entity=config.wandb_entity,
                           mode=config.wandb_mode,
                           notes="Random forest with almost all features",
                           tags=[str(model_class).split('.')[-1].replace('>\'', ''), 'model per station'],
                           config=config.as_dict())

    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training and validation data
    training_dev_data = Data(config.no_nan_in_bikes, config.development_data_path + config.dev_data_filename)
    val_data = Data(config.no_nan_in_bikes, config.development_data_path + config.val_data_filename)

    one_model_for_all_station = None
    per_station_model = None
    one_model_result = None  # relict from testing, leave for now

    # Single model for all stations
    if configuration.run_one_model:
        one_model_for_all_station = model_class(configuration, training_dev_data)  # configure model

        log_keys = {LogKeys.mae_dev.value: WandbLogs.one_model_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.one_model_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.one_model_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.one_model_mae_per_station_val.value,
                    LogKeys.predictions_dev.value: WandbLogs.one_model_predictions_dev.value,
                    LogKeys.predictions_val.value: WandbLogs.one_model_predictions_val.value,
                    }
        one_model_result = train_predict_evaluate_log_for_model_and_data(one_model_for_all_station, training_dev_data,
                                                                         val_data,
                                                                         log_keys, wandb_run,
                                                                         configuration.log_predictions_to_wandb)

    if configuration.run_model_per_station:
        per_station_model = PerStationModel(configuration, training_dev_data, model_class)  # configure model
        log_keys = {LogKeys.mae_dev.value: WandbLogs.per_station_mae_dev.value,
                    LogKeys.mae_val.value: WandbLogs.per_station_mae_val.value,
                    LogKeys.mae_per_station_dev.value: WandbLogs.per_station_mae_per_station_dev.value,
                    LogKeys.mae_per_station_val.value: WandbLogs.per_station_mae_per_station_val.value,
                    LogKeys.predictions_dev.value: WandbLogs.per_station_predictions_dev.value,
                    LogKeys.predictions_val.value: WandbLogs.per_station_predictions_val.value,
                    }
        train_predict_evaluate_log_for_model_and_data(per_station_model, training_dev_data, val_data,
                                                      log_keys, wandb_run, configuration.log_predictions_to_wandb)

    # Write predictions to csv
    if configuration.run_test_predictions:
        test_data = Data(config.no_nan_in_bikes, config.test_data_path)
        if configuration.run_one_model:
            one_model_result_test = one_model_for_all_station.predict(test_data)
            one_model_result_test.write_to_csv('one_model_' + wandb_run.name, configuration)

        if configuration.run_test_predictions:
            per_station_result_test = per_station_model.predict(test_data)
            per_station_result_test.write_to_csv('per_station_model_' + wandb_run.name, configuration)

    # TODO decide what to return
    return {RunResults.predictions: one_model_result, RunResults.wandb: wandb}


# takes a  dictionary { station : mae }
def log_per_station_mae_to_wand(key: str, per_station_values: {}):  # {station:mae}
    for station, station_mae in per_station_values.items():
        wandb.log({key: station_mae, 'station': station})


def main():
    run(RandomForestRegressorModel, Configuration())


if __name__ == "__main__":
    main()
