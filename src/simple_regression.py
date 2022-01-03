import wandb

from src.configurations import Configuration, WandbLogs, RunResults
from src.Data import Data
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
    labelled_dev_data = Data(config.no_nan_in_bikes, config.development_data_path + config.dev_data_filename)

    # Configure Model
    model = PoissonModel(configuration, labelled_dev_data)

    # Train model
    model.fit()

    # Load validation data
    val_data = Data(config.no_nan_in_bikes, config.development_data_path + config.val_data_filename)

    # Use model to predict number of bikes
    val_prediction_result = model.predict(val_data)
    dev_prediction_result = model.predict(labelled_dev_data)

    # Evaluate
    mae_val = val_prediction_result.mean_absolute_error()
    mae_dev = dev_prediction_result.mean_absolute_error()

    wandb.log({
        WandbLogs.mean_absolute_error_validation.value: mae_val,
        WandbLogs.mean_absolute_error_dev.value: mae_dev
    })

    # Log wanddb
    prediction_table = wandb.Table(dataframe=val_prediction_result.predictions_as_df())
    wandb_run.log({WandbLogs.predictions.value: prediction_table})

    return {RunResults.predictions: val_prediction_result, RunResults.wandb: wandb}


def main():
    run(Configuration())


if __name__ == "__main__":
    main()
