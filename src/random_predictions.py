import wandb
import enum
import pandas as pd

from src.OneModel import OneModel
from src.configurations import Configuration
from src.Data import Data


class RunResults(enum.Enum):
    random_predictions = 1
    wandb = 2


class WandbLogs(enum.Enum):
    mean_absolute_error = 'Mean average error'
    predictions = 'Predictions'


def run(config: Configuration = Configuration()):
    results = {}  # keep experiment-results of run
    run = wandb.init(project=config.wandb_project_name, entity=config.wandb_entity, mode=config.wandb_mode,
                     name="random prediction", notes="random predictions", tags=["random prediction"],
                     config=config.as_dict())
    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # No training for random predictions
    one_model = OneModel(configuration)

    # Load test data
    labelled_data = Data(configuration, config.training_data_path)  # do all required preprocessing in here

    # Use models to predict random number of bikes
    random_predictions = one_model.predict_random_numbers_for(labelled_data)
    results[RunResults.random_predictions] = random_predictions

    # Calculate and log mean average error
    mae = random_predictions.mean_absolute_error()
    wandb.log({
        WandbLogs.mean_absolute_error.value: mae
    })

    results[RunResults.wandb] = wandb

    # Write predictions to csv
    if configuration.log_predictions:
        csv_filename = random_predictions.write_to_csv(configuration)

        # Log predictions to wandb
        prediction_table = wandb.Table(dataframe=pd.read_csv(csv_filename))
        run.log({WandbLogs.predictions.value: prediction_table})

    return results
