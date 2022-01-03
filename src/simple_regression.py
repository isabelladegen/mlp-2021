import wandb

from src.configurations import Configuration, WandbLogs, RunResults
from src.Data import Data
from src.models.PoissonModel import PoissonModel


def run(config: Configuration = Configuration()):
    wandb_run = wandb.init(project=config.wandb_project_name, entity=config.wandb_entity, mode=config.wandb_mode,
                           notes="testing", tags=["simple regression"], config=config.as_dict())
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
    prediction_result = model.predict(val_data)

    # Evaluate
    mae = prediction_result.mean_absolute_error()

    wandb.log({
        WandbLogs.mean_absolute_error.value: mae
    })

    # Log wanddb
    prediction_table = wandb.Table(dataframe=prediction_result.predictions_as_df())
    wandb_run.log({WandbLogs.predictions.value: prediction_table})

    return {RunResults.predictions: prediction_result, RunResults.wandb: wandb}


def main():
    run(Configuration())


if __name__ == "__main__":
    main()
