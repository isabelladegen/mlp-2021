import wandb

from src.Data import Data
from src.configurations import Configuration, WandbLogs
from src.models.PerStationModel import PerStationModel
from src.models.UseTrainedModels import BestTrainedModelForEachStation

from src.models.PreTrainedLinearModels import PretrainedLinearModels
from src.run_utils import LogKeys, train_predict_evaluate_log_for_model_and_data


def run(config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name,
                           entity=config.wandb_entity,
                           mode=config.wandb_mode,
                           notes="Best trained model",
                           tags=['Best trained model', 'model per station'],
                           config=config.as_dict())

    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training and validation data
    training_dev_data = Data(config.no_nan_in_bikes, config.development_data_path + config.dev_data_filename)
    val_data = Data(config.no_nan_in_bikes, config.development_data_path + config.val_data_filename)

    #Model
    per_station_model = PretrainedLinearModels(configuration, training_dev_data)

    log_keys = {LogKeys.mae_dev.value: WandbLogs.per_station_mae_dev.value,
                LogKeys.mae_val.value: WandbLogs.per_station_mae_val.value,
                LogKeys.mae_per_station_dev.value: WandbLogs.per_station_mae_per_station_dev.value,
                LogKeys.mae_per_station_val.value: WandbLogs.per_station_mae_per_station_val.value,
                LogKeys.predictions_dev.value: WandbLogs.per_station_predictions_dev.value,
                LogKeys.predictions_val.value: WandbLogs.per_station_predictions_val.value,
                }

    train_predict_evaluate_log_for_model_and_data(per_station_model, training_dev_data, val_data,
                                                  log_keys, wandb_run, configuration.log_predictions_to_wandb)
