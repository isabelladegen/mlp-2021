import wandb

from src.Data import Data
from src.PredictionResult import PredictionResult
from src.configurations import Configuration, WandbLogs
from src.models.BestPreTrainedModelForAStation import BestPreTrainedModelForAStation
from src.models.MultiLayerPerceptronRegressorModel import MultiLayerPerceptronRegressorModel
from src.models.PerStationModel import PerStationModel
from src.run_utils import LogKeys, log_per_station_mae_to_wand


def run(config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name,
                           entity=config.wandb_entity,
                           mode=config.wandb_mode,
                           notes="Combination of my model and the best trained",
                           tags=['Best trained model', 'MLP', 'model per station', 'only rounding at the end'],
                           config=config.as_dict())

    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training and validation data
    training_data = Data(config.no_nan_in_bikes, config.development_data_path + config.dev_data_filename)
    val_data = Data(config.no_nan_in_bikes, config.development_data_path + config.val_data_filename)

    # Pre trained model
    pre_trained_models = PerStationModel(configuration, training_data, BestPreTrainedModelForAStation, True)
    pre_trained_models.fit()
    pre_trained_dev_result = pre_trained_models.predict(training_data)
    pre_trained_val_result = pre_trained_models.predict(val_data)

    # MLP model
    mlp_models = PerStationModel(configuration, training_data, MultiLayerPerceptronRegressorModel)
    mlp_models.fit()
    mlp_dev_result = mlp_models.predict(training_data)
    mlp_val_result = mlp_models.predict(val_data)

    combined_result_dev = PredictionResult([pre_trained_dev_result, mlp_dev_result], calculate_mean=True)
    combined_result_val = PredictionResult([pre_trained_val_result, mlp_val_result], calculate_mean=True)

    # evaluate
    mae_dev = combined_result_dev.mean_absolute_error()
    mae_val = combined_result_val.mean_absolute_error()
    mae_per_station_dev = combined_result_dev.mean_absolute_error_per_station()
    mae_per_station_val = combined_result_val.mean_absolute_error_per_station()

    keys = {LogKeys.mae_dev.value: WandbLogs.per_station_mae_dev.value,
            LogKeys.mae_val.value: WandbLogs.per_station_mae_val.value,
            LogKeys.mae_per_station_dev.value: WandbLogs.per_station_mae_per_station_dev.value,
            LogKeys.mae_per_station_val.value: WandbLogs.per_station_mae_per_station_val.value,
            LogKeys.predictions_dev.value: WandbLogs.per_station_predictions_dev.value,
            LogKeys.predictions_val.value: WandbLogs.per_station_predictions_val.value,
            }

    # log
    wandb.log({
        keys[LogKeys.mae_dev.value]: mae_dev,
        keys[LogKeys.mae_val.value]: mae_val,
    })
    log_per_station_mae_to_wand(keys[LogKeys.mae_per_station_dev.value], mae_per_station_dev)
    log_per_station_mae_to_wand(keys[LogKeys.mae_per_station_val.value], mae_per_station_val)

    # Write predictions to csv
    if configuration.run_test_predictions:
        test_data = Data(config.no_nan_in_bikes, config.test_data_path)

        pre_trained_test_result = pre_trained_models.predict(test_data)
        mlp_test_result = mlp_models.predict(test_data)

        combined_test_result = PredictionResult([pre_trained_test_result, mlp_test_result], calculate_mean=True)
        combined_test_result.write_to_csv('combined_pretrained_and_mlp_model_' + wandb_run.name, configuration)


def main():
    run(Configuration())


if __name__ == "__main__":
    main()
