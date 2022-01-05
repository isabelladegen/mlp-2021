import wandb

from src.configurations import Configuration, WandbLogs, RunResults
from src.Data import Data
from src.models.PerStationModel import PerStationModel
from src.models.PoissonModel import PoissonModel
from src.models.RandomForestRegressorModel import RandomForestRegressorModel


def run(model_class, config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name,
                           entity=config.wandb_entity,
                           mode=config.wandb_mode,
                           notes="feature testing, add weather features",
                           tags=[str(model_class).split('.')[-1], 'one model', 'model per station'],
                           config=config.as_dict())
    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training data dev
    all_station_training_dev_data = Data(config.no_nan_in_bikes,
                                         config.development_data_path + config.dev_data_filename)

    # Single model for all stations
    one_model_for_all_station = model_class(configuration, all_station_training_dev_data)  # configure model
    one_model_for_all_station.fit()  # train

    # Model per station
    per_station_model = PerStationModel(configuration, all_station_training_dev_data,
                                        model_class)  # configure model
    per_station_model.fit()  # train all models

    # Load validation data
    val_data = Data(config.no_nan_in_bikes, config.development_data_path + config.val_data_filename)

    # Predict
    one_model_dev_result = one_model_for_all_station.predict(all_station_training_dev_data)
    one_model_val_result = one_model_for_all_station.predict(val_data)
    per_station_dev_result = per_station_model.predict(all_station_training_dev_data)
    per_station_val_result = per_station_model.predict(val_data)

    # Evaluate and Log
    # overall mae
    one_model_mae_dev = one_model_dev_result.mean_absolute_error()
    one_model_mae_val = one_model_val_result.mean_absolute_error()
    per_station_mae_dev = per_station_dev_result.mean_absolute_error()
    per_station_mae_val = per_station_val_result.mean_absolute_error()

    wandb.log({
        WandbLogs.one_model_mae_dev.value: one_model_mae_dev,
        WandbLogs.one_model_mae_val.value: one_model_mae_val,
        WandbLogs.per_station_mae_dev.value: per_station_mae_dev,
        WandbLogs.per_station_mae_val.value: per_station_mae_val
    })

    # Per station mae
    one_model_mae_per_station_dev = one_model_dev_result.mean_absolute_error_per_station()
    one_model_mae_per_station_val = one_model_val_result.mean_absolute_error_per_station()
    per_station_mae_per_station_dev = per_station_dev_result.mean_absolute_error_per_station()
    per_station_mae_per_station_val = per_station_val_result.mean_absolute_error_per_station()

    log_per_station_mae_to_wand(WandbLogs.one_model_mae_per_station_dev.value, one_model_mae_per_station_dev)
    log_per_station_mae_to_wand(WandbLogs.one_model_mae_per_station_val.value, one_model_mae_per_station_val)
    log_per_station_mae_to_wand(WandbLogs.per_station_mae_per_station_dev.value, per_station_mae_per_station_dev)
    log_per_station_mae_to_wand(WandbLogs.per_station_mae_per_station_val.value, per_station_mae_per_station_val)

    # Log predictions to wandb
    one_model_prediction_table_dev = wandb.Table(dataframe=one_model_dev_result.results_df)
    one_model_prediction_table_val = wandb.Table(dataframe=one_model_val_result.results_df)
    per_station_prediction_table_dev = wandb.Table(dataframe=per_station_dev_result.results_df)
    per_station_prediction_table_val = wandb.Table(dataframe=per_station_val_result.results_df)

    wandb_run.log({WandbLogs.one_model_predictions_dev.value: one_model_prediction_table_dev})
    wandb_run.log({WandbLogs.one_model_predictions_val.value: one_model_prediction_table_val})
    wandb_run.log({WandbLogs.per_station_predictions_dev.value: per_station_prediction_table_dev})
    wandb_run.log({WandbLogs.per_station_predictions_val.value: per_station_prediction_table_val})

    # Write predictions to csv
    if configuration.log_predictions:
        test_data = Data(config.no_nan_in_bikes, config.test_data_path)
        one_model_result_test = one_model_for_all_station.predict(test_data)
        per_station_result_test = per_station_model.predict(test_data)

        one_model_result_test.write_to_csv('one_model_' + wandb_run.name, configuration)
        per_station_result_test.write_to_csv('per_station_model_' + wandb_run.name, configuration)

    # TODO decide what to return
    return {RunResults.predictions: one_model_val_result, RunResults.wandb: wandb}


# takes a  dictionary { station : mae }
def log_per_station_mae_to_wand(key: str, per_station_values: {}):  # {station:mae}
    for station, station_mae in per_station_values.items():
        wandb.log({key: station_mae, 'station': station})


def main():
    run(RandomForestRegressorModel, Configuration())


if __name__ == "__main__":
    main()
