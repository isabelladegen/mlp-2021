import wandb

from src.Data import Data
from src.configurations import Configuration, WandbLogs
from src.models.BestPreTrainedModelForAStation import BestPreTrainedModelForAStation
from src.models.MultiLayerPerceptronRegressorModel import MultiLayerPerceptronRegressorModel
from src.models.PerStationModel import PerStationModel
from src.run_utils import LogKeys, train_predict_evaluate_log_for_model_and_data


def run(config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name,
                           entity=config.wandb_entity,
                           mode=config.wandb_mode,
                           notes="Combination of my model and the best trained",
                           tags=['Best trained model', 'model per station'],
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



    # evaluate
    mae_dev = training_result.mean_absolute_error()
    mae_val = validation_result.mean_absolute_error()
    mae_per_station_dev = training_result.mean_absolute_error_per_station()
    mae_per_station_val = validation_result.mean_absolute_error_per_station()

    # Write predictions to csv
    if configuration.run_test_predictions:
        test_data = Data(config.no_nan_in_bikes, config.test_data_path)

        if configuration.run_test_predictions:
            per_station_result_test = pre_trained_models.predict(test_data)
            per_station_result_test.write_to_csv('per_station_model_' + wandb_run.name, configuration)


def main():
    run(Configuration())


if __name__ == "__main__":
    main()
