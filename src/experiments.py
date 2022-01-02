import wandb
import enum

from src.OneModel import OneModel
from src.configurations import Configuration
from src.Data import Data


class RunResults(enum.Enum):
    random_predictions = 1
    wandb = 2


class WandbLogs(enum.Enum):
    mean_absolute_error = 'Mean absolute error'


def run(config: Configuration = Configuration()):
    wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        mode=config.wandb_mode,
        # name="getting started", # this names the run
        notes="just testing",
        tags=["testing"],
        config=config.as_dict()
    )
    # Reload the Configuration (to allow for sweeps)
    configuration = Configuration(**wandb.config)

    # Load training data
    # training_data = TrainingData()

    # Train models
    # model_per_station = ModelPerStation()
    one_model = OneModel(configuration)

    # Load test data
    test_data = Data(configuration, config.test_data_path)  # do all required preprocessing in here

    # Use models to predict number of bikes
    results = {}
    # if configuration.predict_random_numbers:
    #     random_predictions = one_model.predict_random_numbers_for(test_data)
    #     experiment-results[RunResults.random_predictions] = random_predictions
    #
    #     # Calculate and log mean average error
    #     mae = random_predictions.mean_absolute_error()
    #     wandb.log({
    #         WandbLogs.mean_absolute_error.value: mae
    #     })

        # Write csv for submission
        # random_predictions.write_results_to_csv()

    results[RunResults.wandb] = wandb
    return results


