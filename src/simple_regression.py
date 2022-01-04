import wandb

from src.configurations import Configuration, WandbLogs, RunResults
from src.Data import Data
from src.models.PerStationModel import PerStationModel
from src.models.PoissonModel import PoissonModel


def run(config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name,
                           entity=config.wandb_entity,
                           mode=config.wandb_mode,
                           notes="testing",
                           tags=["simple regression", "number of docks", "bikes 3h ago"],
                           config=config.as_dict())
    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training data dev
    all_station_training_dev_data = Data(config.no_nan_in_bikes,
                                         config.development_data_path + config.dev_data_filename)

    # Single model for all stations
    single_model_for_all_station = PoissonModel(configuration, all_station_training_dev_data)  # configure model
    single_model_for_all_station.fit()  # train

    # Model per station
    per_station_model = PerStationModel(configuration, all_station_training_dev_data,
                                        PoissonModel)  # configure model
    per_station_model.fit()  # train all models

    # Load validation data
    val_data = Data(config.no_nan_in_bikes, config.development_data_path + config.val_data_filename)

    # Predict
    one_model_dev_result = single_model_for_all_station.predict(all_station_training_dev_data)
    one_model_val_result = single_model_for_all_station.predict(val_data)
    per_station_dev_result = per_station_model.predict(all_station_training_dev_data)
    per_station_val_result = per_station_model.predict(val_data)

    # Evaluate
    one_model_mae_dev = one_model_dev_result.mean_absolute_error()
    one_model_mae_val = one_model_val_result.mean_absolute_error()
    # TODO
    # one_model_mae_per_station_dev = one_model_dev_result.mean_absolute_error_per_station(
    #     WandbLogs.one_model_mae_per_station_dev.value)
    # one_model_mae_per_station_val = one_model_val_result.mean_absolute_error_per_station(
    #     WandbLogs.one_model_mae_per_station_val.value)
    per_station_mae_dev = per_station_dev_result.mean_absolute_error()
    per_station_mae_val = per_station_val_result.mean_absolute_error()
    # TODO
    # per_station_mae_per_station_dev = per_station_dev_result.mean_absolute_error_per_station(
    #     WandbLogs.per_station_mae_per_station_dev.value)
    # per_station_mae_per_station_val = per_station_val_result.mean_absolute_error_per_station(
    #     WandbLogs.per_station_mae_per_station_val.value)

    # Log mae to wandb
    # Summary metrics
    wandb.log({
        WandbLogs.one_model_mae_dev.value: one_model_mae_dev,
        WandbLogs.one_model_mae_val.value: one_model_mae_val,
        WandbLogs.per_station_mae_dev.value: per_station_mae_dev,
        WandbLogs.per_station_mae_val.value: per_station_mae_val
    })

    # Per station metrics
    # TODO
    # log_per_station_mae_to_wand(one_model_mae_per_station_dev)
    # log_per_station_mae_to_wand(one_model_mae_per_station_val)
    # log_per_station_mae_to_wand(per_station_mae_per_station_dev)
    # log_per_station_mae_to_wand(per_station_mae_per_station_val)

    # Log predictions to wandb
    one_model_prediction_table_dev = wandb.Table(dataframe=one_model_dev_result.results_df)
    one_model_prediction_table_val = wandb.Table(dataframe=one_model_val_result.results_df)
    per_station_prediction_table_dev = wandb.Table(dataframe=per_station_dev_result.results_df)
    per_station_prediction_table_val = wandb.Table(dataframe=per_station_val_result.results_df)

    wandb_run.log({WandbLogs.one_model_predictions_dev.value: one_model_prediction_table_dev})
    wandb_run.log({WandbLogs.one_model_predictions_val.value: one_model_prediction_table_val})
    wandb_run.log({WandbLogs.per_station_predictions_dev.value: per_station_prediction_table_dev})
    wandb_run.log({WandbLogs.per_station_predictions_val.value: per_station_prediction_table_val})

    # TODO decide what to return
    return {RunResults.predictions: one_model_val_result, RunResults.wandb: wandb}


# takes a list of dictionary { 'key': value, 'station': id }
def log_per_station_mae_to_wand(per_station_values):
    for station_mae in per_station_values:
        wandb.log(station_mae)


def main():
    run(Configuration())


if __name__ == "__main__":
    main()
